"""Prospective interval calibration, reliability scores, and alerts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .defaults import CALIBRATION_DEFAULTS, to_raw_scale
from .red import ROUTED_TAUS
from .metrics import wis

DEFAULT_MINIMUM_HISTORY = CALIBRATION_DEFAULTS.get("minimum_history", 10)
DEFAULT_ADJACENT_LAG_RADIUS = CALIBRATION_DEFAULTS.get("adjacent_lag_radius", 1)
DEFAULT_ETA = CALIBRATION_DEFAULTS.get("eta", .9)
DEFAULT_CAUTION_CUT = CALIBRATION_DEFAULTS.get("caution_cut", .7)
DEFAULT_HIGH_RISK_CUT = CALIBRATION_DEFAULTS.get("high_risk_cut", .9)


# ---------------------------------------------------------------------------
# Matured prospective history selection
# ---------------------------------------------------------------------------

#: Minimum positive-regret support before M_harm_eta is issued, and minimum
#: pre-evaluation pairs before g_harm is fitted. Frozen before evaluation.
DEFAULT_MIN_HARM_SUPPORT = 10


def qhigher(values, q: float) -> float:
    values = np.asarray(values, dtype=float)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:  # NumPy < 1.22, still used by the legacy local runner.
        return float(np.quantile(values, q, interpolation="higher"))


def calibration_stratum(row: pd.Series, history: pd.DataFrame, minimum: int | None = None,
                        adjacent_lag_radius: int | None = None) -> tuple[pd.DataFrame, str]:
    """Select the matured historical forecast set used for calibration
    and reliability.

    Fallback hierarchy, exact-lag evidence always preferred:
    same location + exact lag -> same location + adjacent lags ->
    cross-location + exact lag -> cross-location + adjacent lags.
    Returns ``(selected, source)``; ``source == "insufficient history"`` when
    even the final fallback lacks adequate support.
    """
    minimum = DEFAULT_MINIMUM_HISTORY if minimum is None else minimum
    adjacent_lag_radius = DEFAULT_ADJACENT_LAG_RADIUS if adjacent_lag_radius is None else adjacent_lag_radius
    exact = history[(history.geo_value.eq(row.geo_value)) & history.lag.eq(row.lag)]
    if len(exact) >= minimum:
        return exact, "location/exact-lag"
    local = history[(history.geo_value.eq(row.geo_value)) & (history.lag.sub(row.lag).abs() <= adjacent_lag_radius)]
    if len(local) >= minimum:
        return local, "location/adjacent-lags"
    global_exact = history[history.lag.eq(row.lag)]
    if len(global_exact) >= minimum:
        return global_exact, "global/exact-lag"
    adjacent = history[history.lag.sub(row.lag).abs() <= adjacent_lag_radius]
    return (adjacent, "global/adjacent-lags") if len(adjacent) >= minimum else (adjacent, "insufficient history")


# ---------------------------------------------------------------------------
# Origin x reliability-stratum resolution (model-free spec Sec. 2.2, 2.7)
# ---------------------------------------------------------------------------

def upper_order_statistic(values: np.ndarray, eta: float) -> float:
    """``E_(k_eta(n))`` with ``k_eta(n) = min{n, ceil((n+1) eta)}``.

    The final model-free spec replaces the interpolated quantile with a
    finite-sample upper order statistic: a comparable set is often small, and
    interpolating an extreme quantile from a handful of points invents a value
    no historical case attained. This returns a realized historical value, and
    the ``min{n, .}`` makes it defined for every admissible support size.

    No exact finite-sample coverage is claimed; the construction removes an
    interpolation artefact, not distribution shift.
    """
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = values.size
    if not n:
        return float("nan")
    index = min(n, int(np.ceil((n + 1) * float(eta))))
    return float(np.sort(values)[index - 1])


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a historical proportion.

    Used for ``f_harm = k_harm / n_harmfreq``. Wilson rather than Wald because
    it stays inside [0, 1] and behaves sensibly at k = 0 or k = n, both of
    which occur routinely at small comparable support.

    This quantifies sampling uncertainty about the frequency IN THE COMPARABLE
    HISTORICAL SET. It is not a prospective probability interval for the
    current forecast and must never be presented as one.
    """
    n = int(total)
    if n <= 0:
        return float("nan"), float("nan")
    phat = float(successes) / n
    denominator = 1. + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    spread = (z / denominator) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return float(max(0., centre - spread)), float(min(1., centre + spread))


def stratum_keys(test: pd.DataFrame) -> pd.DataFrame:
    """The unique reliability keys ``(geo_value, lag)`` needed at one origin."""
    return test[["geo_value", "lag"]].drop_duplicates().reset_index(drop=True)


def resolve_strata(test: pd.DataFrame, history: pd.DataFrame, minimum: int,
                   adjacent_lag_radius: int, counter=None) -> dict:
    """Resolve ``C(q)`` ONCE per unique reliability key at one origin.

    ``calibration_stratum`` depends on the current row only through
    ``geo_value`` and ``lag``, so every forecast sharing that key selects the
    identical matured support. The model-free spec (Sec. 2.2/2.7) requires the
    fallback to be resolved once per key and joined onto forecast rows, and
    explicitly prohibits "any implementation that reconstructs C(q) ...
    independently for hundreds of thousands of duplicated forecast rows".

    The archive is indexed once per origin by ``(geo_value, lag)`` and by
    ``lag``, so each key assembles its support from a handful of cached
    position arrays instead of running four boolean scans over the whole
    matured history. Row order within a stratum is irrelevant: every quantity
    taken from it (quantiles and counts) is order-independent.

    ``counter`` decides what "enough history" means. The default counts rows,
    which selects ``C_err(q)`` on adequate TOTAL comparable history. Passing a
    counter over harmful rows selects ``C_harm(q)`` on adequate POSITIVE-HARM
    history instead, which is a different requirement: a set holding
    ``n_min^err`` comparable cases may contain very few harmful ones. The
    hierarchy is identical; only the stopping rule differs, and expansion stops
    as soon as the rule is met.

    Returns ``{(geo_value, lag): (selected, source)}``.
    """
    count = (lambda positions: len(positions)) if counter is None else counter
    keys = stratum_keys(test)
    if history.empty:
        empty = history
        return {(k.geo_value, k.lag): (empty, "insufficient history")
                for k in keys.itertuples(index=False)}
    lags = history.lag.to_numpy()
    geo_lag, by_lag = {}, {}
    for (geo, lag), positions in history.groupby(["geo_value", "lag"], observed=True).indices.items():
        geo_lag[(geo, int(lag))] = positions
    for lag, positions in history.groupby("lag", observed=True).indices.items():
        by_lag[int(lag)] = positions
    radius = range(-int(adjacent_lag_radius), int(adjacent_lag_radius) + 1)

    def gather(parts):
        parts = [x for x in parts if x is not None and len(x)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=int)

    out = {}
    for key in keys.itertuples(index=False):
        geo, lag = key.geo_value, int(key.lag)
        window = [lag + d for d in radius]
        exact = geo_lag.get((geo, lag))
        if exact is not None and count(exact) >= minimum:
            out[(key.geo_value, key.lag)] = (history.take(exact), "location/exact-lag")
            continue
        local = gather([geo_lag.get((geo, w)) for w in window])
        if count(local) >= minimum:
            out[(key.geo_value, key.lag)] = (history.take(local), "location/adjacent-lags")
            continue
        global_exact = by_lag.get(lag)
        if global_exact is not None and count(global_exact) >= minimum:
            out[(key.geo_value, key.lag)] = (history.take(global_exact), "global/exact-lag")
            continue
        adjacent = gather([by_lag.get(w) for w in window])
        out[(key.geo_value, key.lag)] = (
            history.take(adjacent),
            "global/adjacent-lags" if count(adjacent) >= minimum else "insufficient history")
    return out


def stratum_offsets(selected: pd.DataFrame, taus: np.ndarray) -> np.ndarray:
    """Asymmetric endpoint corrections for one stratum, per tau.

    Signed so that the calibrated endpoint is ``base + offset``:
    ``-max(Q_{1-tau}(A^-), 0)`` below the median and ``+max(Q_tau(A^+), 0)``
    above it. The corrections depend only on the stratum, never on the current
    row, which is what makes the per-key cache exact rather than approximate.
    """
    offsets = np.zeros(len(taus))
    truth = selected.truth.to_numpy(float)
    for index, tau in enumerate(taus):
        column = selected[f"q{tau:g}"].to_numpy(float)
        if tau < .5:
            offsets[index] = -max(qhigher(column - truth, 1 - tau), 0.)
        elif tau > .5:
            offsets[index] = max(qhigher(truth - column, tau), 0.)
    return offsets


def enforce_non_crossing(quantiles: np.ndarray, median_index: int) -> np.ndarray:
    """Clamp each side toward the calibrated median (numerical safeguard)."""
    out = np.asarray(quantiles, float).copy()
    if out.ndim == 1:
        out = out[None, :]
    lower = out[:, :median_index]
    if lower.shape[1]:
        out[:, :median_index] = np.minimum(np.maximum.accumulate(lower, axis=1),
                                           out[:, [median_index]])
    upper = out[:, median_index + 1:]
    if upper.shape[1]:
        out[:, median_index + 1:] = np.maximum(np.maximum.accumulate(upper, axis=1),
                                               out[:, [median_index]])
    return out


def empirical_cdf_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``F_ref(x)`` by sorted-array binary search, vectorized over ``values``.

    Spec Sec. 2.7: "evaluate the frozen reference CDFs by sorted-array lookup
    or equivalent vectorized empirical-CDF evaluation; do not compare each
    outer score against every reference observation in a row-wise loop."
    ``side="right"`` reproduces ``mean(reference <= x)`` exactly.
    """
    reference = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
    values = np.asarray(values, float)
    if not len(reference):
        return np.full(values.shape, np.nan)
    return np.searchsorted(reference, values, side="right") / len(reference)


# ---------------------------------------------------------------------------
# Matured prospective interval calibration
# ---------------------------------------------------------------------------

def calibrate(base: np.ndarray, row: pd.Series, history: pd.DataFrame, taus: np.ndarray = ROUTED_TAUS,
             minimum: int | None = None, adjacent_lag_radius: int | None = None) -> tuple[np.ndarray, str, int]:
    """Asymmetric endpoint calibration at every routed quantile.

    For tau < 0.5:  A^- = q_tau - truth;  subtract max(Q_{1-tau}(A^-), 0).
    For tau > 0.5:  A^+ = truth - q_tau;  add      max(Q_tau(A^+), 0).

    The adjustment is applied at every quantile in the configured grid. It
    falls back to the uncalibrated ``base`` quantiles when matured history is
    insufficient. After calibrating each quantile independently, both sides
    are clamped toward the median to prevent crossing quantiles.
    """
    selected, source = calibration_stratum(row, history, minimum, adjacent_lag_radius)
    output = np.asarray(base, dtype=float).copy()
    if source == "insufficient history":
        return output, source, len(selected)
    taus = np.asarray(taus, dtype=float)
    median_index = int(np.argmin(np.abs(taus - 0.5)))
    for index, tau in enumerate(taus):
        column = f"q{tau:g}"
        if tau < .5:
            output[index] -= max(qhigher(selected[column] - selected.truth, 1 - tau), 0.)
        elif tau > .5:
            output[index] += max(qhigher(selected.truth - selected[column], tau), 0.)
    output[:median_index] = np.minimum(np.maximum.accumulate(output[:median_index]), output[median_index])
    output[median_index + 1:] = np.maximum(np.maximum.accumulate(output[median_index + 1:]), output[median_index])
    return output, source, len(selected)


def prospective_calibration(long: pd.DataFrame, methods: tuple[str, ...] | list[str], taus: np.ndarray = ROUTED_TAUS,
                            minimum: int | None = None, adjacent_lag_radius: int | None = None,
                            value_transform: str | None = None) -> pd.DataFrame:
    """Calibrate per fold/method from history matured by that fold's own cutoff.

    ``long`` is a standardized long prediction table (see
    ``autodelphirf.pipeline.standardize_predictions``) with columns
    ``method``, ``fold``, ``cutoff``, ``target_date``, raw-scale ``truth`` and
    ``q{tau:g}``, and their working-scale companions. Endpoint corrections are
    *always* estimated on the working scale and only then inverse-transformed.
    Legacy identity-scale frames remain accepted.

    Implemented at the origin x reliability-stratum level required by the
    model-free spec (Sec. 2.2/2.7): the exact-lag-first fallback and its
    endpoint corrections are resolved ONCE per unique ``(geo_value, lag)`` key
    and then applied to every forecast row sharing that key. Because
    ``calibration_stratum`` depends on a row only through those two fields and
    the corrections depend only on the selected stratum, this is exactly
    equivalent to the per-row definition -- it is not an approximation. The
    previous per-row implementation rescanned the matured archive for every
    row, which is quadratic in the number of forecasts and prohibited by the
    spec.
    """
    minimum = DEFAULT_MINIMUM_HISTORY if minimum is None else minimum
    adjacent_lag_radius = (DEFAULT_ADJACENT_LAG_RADIUS if adjacent_lag_radius is None
                           else adjacent_lag_radius)
    taus = np.asarray(taus, float)
    median_index = int(np.argmin(np.abs(taus - .5)))
    raw_columns = [f"q{tau:g}" for tau in taus]
    work_columns = [f"q{tau:g}_working" for tau in taus]
    frames = []
    subset = long[long.method.isin(methods)]
    if "model_available" in subset:
        subset = subset[subset.model_available]
    if "has_distribution" in subset:
        subset = subset[subset.has_distribution]
    for method, data in subset.groupby("method", observed=True):
        has_working = ("truth_working" in data and all(c in data for c in work_columns))
        transform = value_transform
        if transform is None and "value_transform" in data:
            values = data.value_transform.dropna().unique()
            transform = values[0] if len(values) == 1 else None
        if has_working and transform is None:
            raise ValueError("working-scale calibration requires one resolved value_transform")
        transform = transform or "identity"
        # One working-scale view of the whole method, built once rather than
        # copied per row.
        scale_columns = work_columns if has_working else raw_columns
        truth_column = "truth_working" if has_working else "truth"
        for fold, test in data.groupby("fold", sort=True, observed=True):
            cutoff = test.cutoff.iloc[0]
            history = data[(data.cutoff < cutoff) & (data.target_date <= cutoff)]
            strata = resolve_strata(test, history, minimum, adjacent_lag_radius)
            offsets, sources, counts = {}, {}, {}
            for key, (selected, source) in strata.items():
                counts[key], sources[key] = len(selected), source
                if source == "insufficient history":
                    offsets[key] = np.zeros(len(taus))
                    continue
                view = selected[scale_columns + [truth_column]].rename(
                    columns={**dict(zip(scale_columns, raw_columns)),
                             truth_column: "truth"})
                offsets[key] = stratum_offsets(view, taus)
            keys = list(zip(test.geo_value, test.lag))
            offset_matrix = np.vstack([offsets[k] for k in keys])
            base = test[scale_columns].to_numpy(float)
            calibrated_working = enforce_non_crossing(base + offset_matrix, median_index)
            calibrated = (to_raw_scale(calibrated_working, transform) if has_working
                          else calibrated_working)
            entry = test.copy()
            entry["calibration_source"] = [sources[k] for k in keys]
            entry["calibration_n"] = [counts[k] for k in keys]
            for index, tau in enumerate(taus):
                entry[f"cal_q{tau:g}"] = calibrated[:, index]
                entry[f"cal_q{tau:g}_working"] = calibrated_working[:, index]
            entry["cal_wis_raw"] = wis(calibrated, entry.truth.to_numpy(float))
            entry["cal_wis_working"] = wis(
                calibrated_working, entry[truth_column].to_numpy(float))
            entry["cal_wis"] = entry["cal_wis_raw"]
            frames.append(entry)
    return (pd.concat(frames, ignore_index=True) if frames
            else long.iloc[:0].copy())


# Relative tolerance for interval-coverage boundary comparisons. Set well
# above floating-point round-trip error (~1e-16 relative) and far below any
# scientifically meaningful interval width, so it resolves only exact-boundary
# ties on degenerate predictive intervals.
COVERAGE_BOUNDARY_TOLERANCE = 1e-9


def add_interval_columns(data: pd.DataFrame, levels: dict[int, tuple[int, int]], prefix: str = "",
                         taus: np.ndarray = ROUTED_TAUS) -> pd.DataFrame:
    """Coverage/width/miss-rate columns at nominal central levels.

    ``levels`` maps a nominal percent (e.g. 50, 80, 95) to the (lower,
    upper) tau VALUES bounding that central interval, e.g.
    ``{50: (.25, .75), 80: (.1, .9), 95: (.025, .975)}`` -- the same
    convention already used by ``autodelphirf.report``.
    """
    data = data.copy()
    truth = data.truth
    for level, (lo, hi) in levels.items():
        lower, upper = data[f"{prefix}q{lo:g}"], data[f"{prefix}q{hi:g}"]
        # Boundary equality must be decided at the scale of the data, not at
        # the last bit. A large share of forecasts are degenerate point masses
        # (an already-final value has no remaining revision, so every routed
        # quantile coincides) and for those the truth sits exactly ON the
        # endpoint. Base endpoints are stored raw, whereas calibrated ones are
        # back-transformed, so expm1(log1p(x)) differs from x by about one ulp
        # -- enough to flip a strict `<=` and to charge calibration with a
        # coverage loss it did not cause. On nssp_covid28 this alone moved
        # measured calibrated 95% coverage from 0.844 to 0.920.
        scale = np.maximum(np.abs(truth), np.maximum(np.abs(lower), np.abs(upper)))
        tol = COVERAGE_BOUNDARY_TOLERANCE * scale
        data[f"{prefix}coverage_{level}"] = (lower - tol <= truth) & (truth <= upper + tol)
        data[f"{prefix}width_{level}"] = upper - lower
        data[f"{prefix}lower_miss_{level}"] = truth < lower - tol
        data[f"{prefix}upper_miss_{level}"] = truth > upper + tol
    return data


# ---------------------------------------------------------------------------
# Forecast-quality risk score and alert
# ---------------------------------------------------------------------------

def forecast_quality_summary(history: pd.DataFrame, *, eta: float = DEFAULT_ETA, min_history: int = 20,
                             min_harm_support: int = DEFAULT_MIN_HARM_SUPPORT,
                             error_reference: np.ndarray | None = None, regret_reference: np.ndarray | None = None,
                             caution_cut: float = DEFAULT_CAUTION_CUT, high_risk_cut: float = DEFAULT_HIGH_RISK_CUT) -> dict:
    """Difficulty and historical harm frequency on ``C_err(q)``.

    Implements the first two components of the final model-free spec:

        q_eta^err(q) = E_(k_eta(n_err))          (finite-sample order statistic)
        f_harm(q)    = k_harm(q) / n_harmfreq(q) (descriptive frequency)

    ``f_harm`` is reported with its counts and a 95% Wilson interval, and is
    NOT a prospective probability. There is no probability mapping anywhere in
    this layer: the spec removed ``p_harm``, the pre-evaluation archive and the
    isotonic map after the mapped probability failed its own accuracy test
    (negative Brier skill on 4 of 5 datasets, gaps to 0.45). What survives is
    an interpretable statement of the form "8 of 42 comparable matured cases
    (19%)".

    Conditional harm severity is NOT computed here: it needs its own comparable
    set selected on positive-harm support. See ``harm_severity_summary``.

    ``history`` must contain only forecasts whose target matured before the
    current origin, with columns ``rr_ae`` and ``null_ae``. The reference
    arguments are accepted and ignored so archived callers keep working.
    """
    if not 0 < eta < 1:
        raise ValueError("eta must lie strictly between zero and one")
    if min_history < 1:
        raise ValueError("min_history must be positive")
    unavailable = {"reliability_level": "insufficient reliability history",
                   "profile_status": "insufficient reliability history",
                   "q_error": np.nan, "n_err": 0,
                   "f_harm": np.nan, "k_harm": 0, "n_harmfreq": 0,
                   "f_harm_wilson_low": np.nan, "f_harm_wilson_high": np.nan,
                   "risk_score": np.nan}
    if len(history) < min_history:
        return {**unavailable, "reliability_n": len(history)}
    routed, null = history.rr_ae.to_numpy(float), history.null_ae.to_numpy(float)
    valid = np.isfinite(routed) & np.isfinite(null)
    error, regret = routed[valid], routed[valid] - null[valid]
    if len(error) < min_history:
        return {**unavailable, "reliability_n": len(error)}
    harmful_count = int(np.sum(regret > 0))
    total = int(error.size)
    low, high = wilson_interval(harmful_count, total)
    return {
        # Component 1: prediction difficulty, a realized historical error.
        "q_error": upper_order_statistic(error, eta),
        "n_err": total,
        # Component 2: descriptive comparable-history harm frequency, with the
        # counts that make it interpretable and its sampling uncertainty.
        "f_harm": harmful_count / total,
        "k_harm": harmful_count,
        "n_harmfreq": total,
        "f_harm_wilson_low": low,
        "f_harm_wilson_high": high,
        "reliability_n": total,
        "profile_status": "profile available",
        # Retained so existing report code keeps working; a SUPPORT status.
        "reliability_level": "profile available",
        "risk_score": np.nan,
    }


def harm_severity_summary(history: pd.DataFrame, *, eta: float = DEFAULT_ETA,
                          min_harm_support: int = DEFAULT_MIN_HARM_SUPPORT) -> dict:
    """Conditional harm severity on ``C_harm(q)``: ``Delta^+_(k_eta(m))``.

    Answers "if correction hurts, how bad has the harm historically been?" over
    a set selected until it holds at least ``min_harm_support`` POSITIVE-regret
    cases. Reporting it from the total-support set instead left it unissuable
    for 35-60% of cases on five datasets, which is what motivated giving
    severity its own selection.

    Not a guaranteed prospective bound; its conditional upper-tail
    interpretation must earn support in V2c.
    """
    unavailable = {"m_harm": np.nan, "harm_positive_n": 0,
                   "harm_severity_status": "insufficient harm-severity history"}
    if history is None or history.empty:
        return unavailable
    routed, null = history.rr_ae.to_numpy(float), history.null_ae.to_numpy(float)
    valid = np.isfinite(routed) & np.isfinite(null)
    regret = routed[valid] - null[valid]
    harmful = regret[regret > 0]
    if harmful.size < int(min_harm_support):
        return {**unavailable, "harm_positive_n": int(harmful.size)}
    return {"m_harm": upper_order_statistic(harmful, eta),
            "harm_positive_n": int(harmful.size),
            "harm_severity_status": "severity available"}


def build_reliability_reference(training_predictions: pd.DataFrame, *, eta: float = DEFAULT_ETA,
                                min_history: int = 10, adjacent_lag_radius: int = 1,
                                value_transform: str = "log1p", method: str | None = None,
                                prediction_column: str | None = None,
                                null_column: str | None = None) -> pd.DataFrame:
    """Build frozen q-error/q-regret references from inner rolling predictions.

    The input must be a separate pre-evaluation replay on the dataset's own
    working scale; pass that dataset's ``value_transform`` so the errors below
    are formed on the original reporting scale. It defaults to "log1p", which
    is right for count and directly-supplied-rate datasets but WRONG for a
    num/denom fraction triangle (those are "log"). Each inner case is summarized only from forecasts issued at
    earlier cutoffs whose targets had matured by its cutoff. The result must
    be frozen before outer evaluation: build this from data
    that ends strictly before the outer evaluation window begins.
    """
    required = {"cutoff", "target_date", "geo_value", "lag", "truth"}
    missing = required.difference(training_predictions.columns)
    if missing:
        raise ValueError(f"reliability training predictions missing columns: {sorted(missing)}")
    if not 0 < eta < 1 or min_history < 1 or adjacent_lag_radius < 0:
        raise ValueError("invalid reliability reference settings")
    data = training_predictions.copy()
    if method is not None:
        if "method" not in data:
            raise ValueError("method was supplied but training predictions have no method column")
        data = data[data.method.eq(method)].copy()
    if data.empty:
        return pd.DataFrame(columns=["method", "cutoff", "geo_value", "lag", "case_n", "source",
                                     "history_n", "q_error", "q_regret"])
    # Standardized long prediction archives are already on the raw reporting
    # scale.  Legacy replay-wide archives retain working-scale columns such as
    # ``red_prediction`` and ``Null`` and are inverse-transformed below.
    if prediction_column is None:
        prediction_column = "prediction" if "prediction" in data else "red_prediction"
    if null_column is None:
        null_column = "provisional_value" if "provisional_value" in data else "Null"
    required = {prediction_column, null_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"reliability training predictions missing columns: {sorted(missing)}")
    # Replay frames may carry a large residual-training table in ``attrs``;
    # pandas propagates it through every filter unless explicitly cleared.
    data.attrs = {}
    data["cutoff"] = pd.to_datetime(data.cutoff)
    data["target_date"] = pd.to_datetime(data.target_date)
    standardized_raw = prediction_column == "prediction" and null_column == "provisional_value"
    truth_raw = (data.truth.astype(float) if standardized_raw
                 else to_raw_scale(data.truth.astype(float), value_transform))
    prediction_raw = (data[prediction_column].astype(float) if standardized_raw
                      else to_raw_scale(data[prediction_column].astype(float), value_transform))
    null_raw = (data[null_column].astype(float) if standardized_raw
                else to_raw_scale(data[null_column].astype(float), value_transform))
    data["rr_ae"] = np.abs(truth_raw - prediction_raw)
    data["null_ae"] = np.abs(truth_raw - null_raw)
    rows = []
    for cutoff, current in data.groupby("cutoff", sort=True, observed=True):
        matured = data[(data.cutoff < cutoff) & (data.target_date <= cutoff)]
        cases = current.groupby(["geo_value", "lag"], observed=True).size().rename("case_n").reset_index()
        for _, case in cases.iterrows():
            candidates = [
                (matured[(matured.geo_value.eq(case.geo_value)) & matured.lag.eq(case.lag)],
                 "location/exact-lag"),
                (matured[(matured.geo_value.eq(case.geo_value)) &
                         (matured.lag.sub(case.lag).abs() <= adjacent_lag_radius)],
                 "location/adjacent-lags"),
                (matured[matured.lag.eq(case.lag)], "global/exact-lag"),
                (matured[matured.lag.sub(case.lag).abs() <= adjacent_lag_radius],
                 "global/adjacent-lags"),
            ]
            selected = source = None
            for candidate, label in candidates:
                valid = candidate[np.isfinite(candidate.rr_ae) & np.isfinite(candidate.null_ae)]
                if len(valid) >= min_history:
                    selected, source = valid, label
                    break
            if selected is None:
                continue
            regret = selected.rr_ae.to_numpy(float) - selected.null_ae.to_numpy(float)
            rows.append({"method": method if method is not None else (data.method.iloc[0] if "method" in data else None),
                         "cutoff": cutoff, "geo_value": case.geo_value,
                         "lag": case.lag, "case_n": int(case.case_n),
                         "source": source, "history_n": len(selected),
                         "q_error": float(np.quantile(selected.rr_ae, eta)),
                         "q_regret": float(np.quantile(regret, eta))})
    return pd.DataFrame(rows, columns=["method", "cutoff", "geo_value", "lag", "case_n", "source",
                                      "history_n", "q_error", "q_regret"])


# ---------------------------------------------------------------------------
# Per-method reliability wiring
# ---------------------------------------------------------------------------

def reliability_history(long: pd.DataFrame, method: str, *, eta: float | None = None,
                        min_history: int | None = None,
                        min_harm_support: int = DEFAULT_MIN_HARM_SUPPORT,
                        caution_cut: float | None = None,
                        high_risk_cut: float | None = None, adjacent_lag_radius: int | None = None,
                        error_reference: np.ndarray | None = None, regret_reference: np.ndarray | None = None,
                        prediction_column: str = "prediction", null_column: str = "provisional_value") -> pd.DataFrame:
    """Per-case post-prediction assessment quantities for one evaluated method.

    Resolves TWO comparable sets per origin and reliability key, because the
    final model-free spec requires different support rules:

      * ``C_err(q)``  -- adequate TOTAL history; supports q_eta^err, n_err,
        f_harm, its counts and Wilson interval;
      * ``C_harm(q)`` -- adequate POSITIVE-HARM history; supports
        M_harm_eta and m(q), and is allowed to expand one or more fallback
        levels further only when harmful cases are too few.

    Both use the identical exact-lag-first hierarchy; only the stopping rule
    differs, and ``C_harm`` is not expanded once its minimum is met. Caching is
    per key, so the eligible sets are exactly those the fallback rules define.
    """
    eta = DEFAULT_ETA if eta is None else eta
    min_history = DEFAULT_MINIMUM_HISTORY if min_history is None else min_history
    caution_cut = DEFAULT_CAUTION_CUT if caution_cut is None else caution_cut
    high_risk_cut = DEFAULT_HIGH_RISK_CUT if high_risk_cut is None else high_risk_cut
    adjacent_lag_radius = DEFAULT_ADJACENT_LAG_RADIUS if adjacent_lag_radius is None else adjacent_lag_radius
    base = long[long.method.eq(method)].copy()
    if "model_available" in base:
        base = base[base.model_available].copy()
    if base.empty:
        return base
    # Raw-scale errors for the whole archive once, not per selected stratum.
    base["rr_ae"] = (base.truth - base[prediction_column]).abs()
    base["null_ae"] = (base.truth - base[null_column]).abs()
    carried = [c for c in ("cutoff", "geo_value", "reference_date", "report_date",
                           "target_date", "lag") if c in base.columns]
    err_fields = ("reliability_level", "reliability_n", "q_error", "n_err",
                  "f_harm", "k_harm", "n_harmfreq", "f_harm_wilson_low",
                  "f_harm_wilson_high", "risk_score", "profile_status")
    harm_fields = ("m_harm", "harm_positive_n", "harm_severity_status")
    frames = []
    for fold, test in base.groupby("fold", sort=True, observed=True):
        cutoff = test.cutoff.iloc[0]
        history = base[(base.cutoff < cutoff) & (base.target_date <= cutoff)]
        # C_err: stop on total support.
        err_strata = resolve_strata(test, history, min_history, adjacent_lag_radius)
        # C_harm: same hierarchy, stop on POSITIVE-HARM support. The counter
        # reads a precomputed harmful mask so expansion costs no extra scans.
        if history.empty:
            harm_strata = {k: (history, "insufficient history") for k in err_strata}
        else:
            harmful_mask = (history.rr_ae.to_numpy(float) - history.null_ae.to_numpy(float)) > 0
            harm_strata = resolve_strata(
                test, history, int(min_harm_support), adjacent_lag_radius,
                counter=lambda positions: int(harmful_mask[positions].sum()))
        cache = {}
        for key, (selected, source) in err_strata.items():
            summary = forecast_quality_summary(
                selected, eta=eta, min_history=min_history, caution_cut=caution_cut,
                high_risk_cut=high_risk_cut, error_reference=error_reference,
                regret_reference=regret_reference, min_harm_support=min_harm_support)
            harm_selected, harm_source = harm_strata.get(key, (history.iloc[:0], "insufficient history"))
            severity = harm_severity_summary(harm_selected, eta=eta,
                                             min_harm_support=min_harm_support)
            cache[key] = (source, harm_source, summary, severity)
        keys = list(zip(test.geo_value, test.lag))
        entry = pd.DataFrame({"method": method, "fold": fold,
                              "calibration_source": [cache[k][0] for k in keys],
                              "harm_source": [cache[k][1] for k in keys],
                              "truth": test.truth.to_numpy(),
                              prediction_column: test[prediction_column].to_numpy(),
                              null_column: test[null_column].to_numpy()})
        for column in carried:
            entry[column] = test[column].to_numpy()
        for field in err_fields:
            entry[field] = [cache[k][2].get(field, np.nan) for k in keys]
        for field in harm_fields:
            entry[field] = [cache[k][3].get(field, np.nan) for k in keys]
        frames.append(entry)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = rows
    if not result.empty:
        result["realized_ae"] = (result.truth - result[prediction_column]).abs()
        result["realized_null_ae"] = (result.truth - result[null_column]).abs()
        result["realized_regret"] = result.realized_ae - result.realized_null_ae
        result["realized_harm"] = result.realized_regret > 0
        # Upper-tail interpretation checks (V2a and V2c). The severity check is
        # CONDITIONAL on realized harm, which is what makes it a statement
        # about severity rather than about frequency.
        result["qerr_covered"] = result.realized_ae <= result.q_error
        result["mharm_covered"] = np.where(
            result.realized_harm & np.isfinite(result.m_harm),
            result.realized_regret <= result.m_harm, np.nan)
    return result
