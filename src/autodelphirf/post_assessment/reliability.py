"""Prospective interval calibration, PIT diagnostics, and interval coverage columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.defaults import CALIBRATION_DEFAULTS, to_reporting_scale
from ..comparators.red import ROUTED_TAUS
from .scoring import wis

DEFAULT_MINIMUM_HISTORY = CALIBRATION_DEFAULTS.get("minimum_history", 10)
DEFAULT_ADJACENT_LAG_RADIUS = CALIBRATION_DEFAULTS.get("adjacent_lag_radius", 1)
LEVELS = {50: (.25, .75), 80: (.1, .9), 95: (.025, .975)}


# ---------------------------------------------------------------------------
# Completed-revision history selection
# ---------------------------------------------------------------------------

def eligible_history(frame: pd.DataFrame, as_of) -> pd.DataFrame:
    """Historical forecasts whose lag-L target exists by ``as_of`` (Sec. 3.2.5).

    ``as_of`` is the report date of the forecast being assessed, and
    ``target_date`` is ``t + L``: a forecast becomes history only once its
    target lag has been reached, never on an earlier provisional value. Such a
    forecast was necessarily issued before ``as_of``. Rows still without truth
    carry no information and are excluded before the comparable-set search
    counts support.
    """
    return frame[(frame.target_date <= as_of) & frame.truth.notna()]


class _HistoryByDate:
    """``eligible_history`` for many report dates: completed forecasts sorted
    by target date once, so each report date selects a prefix."""

    def __init__(self, frame: pd.DataFrame):
        completed = frame[frame.truth.notna()]
        self.ordered = completed.sort_values("target_date", kind="stable")
        self.dates = self.ordered.target_date.to_numpy()

    def at(self, as_of) -> pd.DataFrame:
        end = int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(as_of)), side="right"))
        return self.ordered.iloc[:end]


def qhigher(values, q: float) -> float:
    values = np.asarray(values, dtype=float)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:  # NumPy < 1.22, still used by the legacy local runner.
        return float(np.quantile(values, q, interpolation="higher"))


def calibration_stratum(row: pd.Series, history: pd.DataFrame, minimum: int | None = None,
                        adjacent_lag_radius: int | None = None) -> tuple[pd.DataFrame, str]:
    """Select the historical forecast set with available target values used for calibration.

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
# Origin x calibration-stratum resolution (model-free spec Sec. 2.2, 2.7)
# ---------------------------------------------------------------------------

def stratum_keys(test: pd.DataFrame) -> pd.DataFrame:
    """The unique reliability keys ``(geo_value, lag)`` needed at one origin."""
    return test[["geo_value", "lag"]].drop_duplicates().reset_index(drop=True)


def resolve_strata(test: pd.DataFrame, history: pd.DataFrame, minimum: int,
                   adjacent_lag_radius: int, counter=None) -> dict:
    """Resolve ``C(q)`` ONCE per unique reliability key at one origin.

    ``calibration_stratum`` depends on the current row only through
    ``geo_value`` and ``lag``, so every forecast sharing that key selects the
    identical completed-revision support. The model-free spec (Sec. 2.2/2.7) requires the
    fallback to be resolved once per key and joined onto forecast rows, and
    explicitly prohibits "any implementation that reconstructs C(q) ...
    independently for hundreds of thousands of duplicated forecast rows".

    The archive is indexed once per origin by ``(geo_value, lag)`` and by
    ``lag``, so each key assembles its support from a handful of cached
    position arrays instead of running four boolean scans over the whole
    completed revision history. Row order within a stratum is irrelevant: every quantity
    taken from it (quantiles and counts) is order-independent.

    ``counter`` decides what "enough history" means; the default counts rows.
    Expansion stops as soon as the rule is met.

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
    """Resolve quantile crossings outward from the median on both sides.

    Endpoints are calibrated separately (Eq. 3.26), so neighbouring levels can
    cross. Walking outward from the median, each lower quantile is at most its
    inner neighbour and each upper quantile at least its inner neighbour. A
    crossing therefore always moves the OUTER endpoint further out, widening
    the wider interval and never narrowing a narrower one.
    """
    out = np.asarray(quantiles, float).copy()
    if out.ndim == 1:
        out = out[None, :]
    lower = out[:, :median_index]
    if lower.shape[1]:
        inward_first = np.minimum(lower[:, ::-1], out[:, [median_index]])
        out[:, :median_index] = np.minimum.accumulate(inward_first, axis=1)[:, ::-1]
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
# Prospective interval calibration from completed revisions
# ---------------------------------------------------------------------------

def calibrate(base: np.ndarray, row: pd.Series, history: pd.DataFrame, taus: np.ndarray = ROUTED_TAUS,
             minimum: int | None = None, adjacent_lag_radius: int | None = None) -> tuple[np.ndarray, str, int]:
    """Asymmetric endpoint calibration at every routed quantile.

    For tau < 0.5:  A^- = q_tau - truth;  subtract max(Q_{1-tau}(A^-), 0).
    For tau > 0.5:  A^+ = truth - q_tau;  add      max(Q_tau(A^+), 0).

    The adjustment is applied at every quantile in the configured grid. It
    falls back to the uncalibrated ``base`` quantiles when completed revision history is
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
                            working_scale: str | None = None) -> pd.DataFrame:
    """Calibrate each forecast from history whose targets exist by its own report date.

    ``long`` is a standardized long prediction table (see
    ``autodelphirf.workflow.run.standardize_predictions``) with columns
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
    previous per-row implementation rescanned the completed revision history for every
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
        transform = working_scale
        if transform is None and "working_scale" in data:
            values = data.working_scale.dropna().unique()
            transform = values[0] if len(values) == 1 else None
        if has_working and transform is None:
            raise ValueError("working-scale calibration requires one resolved working_scale")
        transform = transform or "identity"
        # One working-scale view of the whole method, built once rather than
        # copied per row.
        scale_columns = work_columns if has_working else raw_columns
        truth_column = "truth_working" if has_working else "truth"
        history_columns = list(dict.fromkeys(
            ["target_date", "geo_value", "lag", "truth", truth_column, *scale_columns]))
        completed = _HistoryByDate(data[history_columns])
        for (fold, issued), test in data.groupby(["fold", "report_date"], sort=True,
                                                  observed=True):
            history = completed.at(issued)
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
            calibrated = (to_reporting_scale(calibrated_working, transform) if has_working
                          else calibrated_working)
            entry = test.copy()
            entry["calibration_source"] = [sources[k] for k in keys]
            entry["calibration_n"] = [counts[k] for k in keys]
            for index, tau in enumerate(taus):
                entry[f"cal_q{tau:g}"] = calibrated[:, index]
                entry[f"cal_q{tau:g}_working"] = calibrated_working[:, index]
            entry["cal_wis_reporting"] = wis(calibrated, entry.truth.to_numpy(float))
            entry["cal_wis_working"] = wis(
                calibrated_working, entry[truth_column].to_numpy(float))
            frames.append(entry)
    return (pd.concat(frames, ignore_index=True) if frames
            else long.iloc[:0].copy())


def _quantile_level_map(selected: pd.DataFrame, columns: list[str], truth_column: str,
                        taus: np.ndarray) -> np.ndarray:
    """Map requested probabilities to base-forecast probabilities.

    ``coverage[j]`` is the historical frequency with which the observation was
    below the forecast at ``taus[j]``.  Inverting that monotone reliability
    curve gives the base quantile level whose historical coverage corresponds
    to each requested level.  End anchors make the mapping defined throughout
    [0, 1]; repeated empirical coverages are collapsed conservatively to their
    mean nominal level before interpolation.
    """
    truth = selected[truth_column].to_numpy(float)
    quantiles = selected[columns].to_numpy(float)
    valid = np.isfinite(truth) & np.isfinite(quantiles).all(axis=1)
    if not valid.any():
        return taus.copy()
    valid_truth = truth[valid, None]
    valid_quantiles = quantiles[valid]
    scale = np.maximum(np.abs(valid_truth), np.abs(valid_quantiles))
    tolerance = COVERAGE_BOUNDARY_TOLERANCE * scale
    coverage = np.mean(valid_truth <= valid_quantiles + tolerance, axis=0)
    coverage = np.maximum.accumulate(coverage)
    frame = pd.DataFrame({"coverage": np.r_[0., coverage, 1.],
                          "nominal": np.r_[0., taus, 1.]})
    knots = frame.groupby("coverage", sort=True, as_index=False).nominal.mean()
    return np.interp(taus, knots.coverage.to_numpy(float),
                     knots.nominal.to_numpy(float))


def _remap_quantiles(base: np.ndarray, source_levels: np.ndarray,
                     taus: np.ndarray) -> np.ndarray:
    """Evaluate each row's base quantile curve at remapped levels."""
    anchors = np.r_[0., taus, 1.]
    base = np.asarray(base, float)
    # Constant extrapolation is deliberate: the fitted model supplies no
    # information beyond its outer routed quantiles. Interpolation is
    # vectorized across rows; only the nine target levels form the small axis.
    values = np.column_stack([base[:, 0], base, base[:, -1]])
    right = np.searchsorted(anchors, source_levels, side="right").clip(1, len(anchors) - 1)
    left = right - 1
    span = anchors[right] - anchors[left]
    weight = np.divide(source_levels - anchors[left], span,
                       out=np.zeros_like(source_levels), where=span > 0)
    return values[:, left] + (values[:, right] - values[:, left]) * weight


def _pool_strata(test: pd.DataFrame, history: pd.DataFrame, assignments: pd.DataFrame,
                 minimum: int) -> dict:
    """Comparable history from the current origin's RevRoute task pools.

    Pool numbers are origin-local.  We therefore use the current assignment to
    recover its member ``(location, lag)`` tasks and select historical rows by
    task membership; numeric pool labels are never compared across origins.
    """
    keys = stratum_keys(test)
    required = {"geo_value", "lag", "pool"}
    if assignments is None or assignments.empty:
        raise ValueError("pool calibration requires current-origin pool assignments")
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"pool assignments missing columns: {sorted(missing)}")
    members = {}
    for pool, group in assignments.groupby("pool", observed=True):
        members[pool] = set(zip(group.geo_value.astype(str), group.lag.astype(int)))
    task_pool = {(str(row.geo_value), int(row.lag)): row.pool
                 for row in assignments.itertuples(index=False)}
    history_positions = {
        (str(geo), int(lag)): positions
        for (geo, lag), positions in
        history.groupby(["geo_value", "lag"], observed=True).indices.items()
    }
    selected_by_pool = {}
    for pool, wanted in members.items():
        parts = [history_positions[task] for task in wanted if task in history_positions]
        positions = np.concatenate(parts) if parts else np.empty(0, dtype=int)
        selected_by_pool[pool] = history.take(positions)
    out = {}
    for key in keys.itertuples(index=False):
        identity = (str(key.geo_value), int(key.lag))
        pool = task_pool.get(identity)
        if pool is None:
            out[(key.geo_value, key.lag)] = (history.iloc[:0], "pool task unavailable")
            continue
        selected = selected_by_pool[pool]
        source = "current-origin pool" if len(selected) >= minimum else "insufficient pool history"
        out[(key.geo_value, key.lag)] = (selected, source)
    return out


def prospective_quantile_mapping(
        long: pd.DataFrame, methods: tuple[str, ...] | list[str], *,
        strategy: str, pool_assignments: pd.DataFrame | None = None,
        taus: np.ndarray = ROUTED_TAUS, minimum: int | None = None,
        adjacent_lag_radius: int | None = None,
        working_scale: str | None = None) -> pd.DataFrame:
    """Prospective monotone quantile mapping by location/lag or RevRoute pool.

    Only forecasts whose targets were observable at issuance enter a mapping.
    Insufficient strata retain their base quantiles and are explicitly marked.
    """
    if strategy not in {"location_lag", "pool"}:
        raise ValueError("quantile-mapping strategy must be 'location_lag' or 'pool'")
    if strategy == "pool" and pool_assignments is not None:
        if not {"fold", "cutoff"}.intersection(pool_assignments.columns):
            raise ValueError("pool assignments require an origin column: 'fold' or 'cutoff'")
    minimum = DEFAULT_MINIMUM_HISTORY if minimum is None else int(minimum)
    adjacent_lag_radius = (DEFAULT_ADJACENT_LAG_RADIUS if adjacent_lag_radius is None
                           else int(adjacent_lag_radius))
    taus = np.asarray(taus, float)
    median_index = int(np.argmin(np.abs(taus - .5)))
    raw_columns = [f"q{tau:g}" for tau in taus]
    work_columns = [f"q{tau:g}_working" for tau in taus]
    subset = long[long.method.isin(methods)]
    if "model_available" in subset:
        subset = subset[subset.model_available]
    if "has_distribution" in subset:
        subset = subset[subset.has_distribution]
    frames = []
    for method, data in subset.groupby("method", observed=True):
        has_working = "truth_working" in data and all(c in data for c in work_columns)
        transform = working_scale
        if transform is None and "working_scale" in data:
            values = data.working_scale.dropna().unique()
            transform = values[0] if len(values) == 1 else None
        if has_working and transform is None:
            raise ValueError("working-scale calibration requires one resolved working_scale")
        transform = transform or "identity"
        columns = work_columns if has_working else raw_columns
        truth_column = "truth_working" if has_working else "truth"
        history_columns = list(dict.fromkeys(
            ["target_date", "geo_value", "lag", "truth", truth_column, *columns]))
        completed = _HistoryByDate(data[history_columns])
        for (fold, issued), test in data.groupby(["fold", "report_date"], sort=True,
                                                  observed=True):
            history = completed.at(issued)
            if strategy == "location_lag":
                strata = resolve_strata(test, history, minimum, adjacent_lag_radius)
            else:
                current = pool_assignments
                if current is not None and not current.empty:
                    if "fold" in current:
                        numeric = pd.to_numeric(current.fold, errors="coerce")
                        fold_number = pd.to_numeric(pd.Series([fold]), errors="coerce").iloc[0]
                        if pd.notna(fold_number) and numeric.notna().any():
                            current = current[numeric.eq(fold_number)]
                        else:
                            current = current[current.fold.astype(str).eq(str(fold))]
                    elif "cutoff" in current:
                        current = current[pd.to_datetime(current.cutoff).eq(pd.Timestamp(test.cutoff.iloc[0]))]
                if current is None or current.empty:
                    raise ValueError(
                        f"no pool assignments match fold={fold!r}, "
                        f"cutoff={test.cutoff.iloc[0]!r}")
                strata = _pool_strata(test, history, current, minimum)
            mapped, sources, counts = {}, {}, {}
            for key, (selected, source) in strata.items():
                sources[key], counts[key] = source, len(selected)
                if source.startswith("insufficient") or source.endswith("unavailable"):
                    mapped[key] = taus.copy()
                else:
                    mapped[key] = _quantile_level_map(selected, columns, truth_column, taus)
            keys = list(zip(test.geo_value, test.lag))
            base = test[columns].to_numpy(float)
            calibrated_working = np.empty_like(base)
            positions = {}
            for index, key in enumerate(keys):
                positions.setdefault(key, []).append(index)
            for key, index in positions.items():
                calibrated_working[index] = _remap_quantiles(base[index], mapped[key], taus)
            calibrated_working = enforce_non_crossing(calibrated_working, median_index)
            calibrated = (to_reporting_scale(calibrated_working, transform) if has_working
                          else calibrated_working)
            entry = test.copy()
            entry["calibration_strategy"] = f"quantile_map_{strategy}"
            entry["calibration_source"] = [sources[k] for k in keys]
            entry["calibration_n"] = [counts[k] for k in keys]
            for index, tau in enumerate(taus):
                entry[f"cal_q{tau:g}"] = calibrated[:, index]
                entry[f"cal_q{tau:g}_working"] = calibrated_working[:, index]
            entry["cal_wis_reporting"] = wis(calibrated, entry.truth.to_numpy(float))
            entry["cal_wis_working"] = wis(
                calibrated_working, entry[truth_column].to_numpy(float))
            frames.append(entry)
    return pd.concat(frames, ignore_index=True) if frames else long.iloc[:0].copy()


def pit_diagnostics(calibrations: dict[str, pd.DataFrame],
                    taus: np.ndarray = ROUTED_TAUS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantile reliability and interval-censored PIT diagnostics.

    Nine routed quantiles do not identify a continuous CDF.  Each PIT is
    therefore reported as the probability bin containing the observation plus
    its midpoint, rather than pretending an exact PIT value is available.
    """
    taus = np.asarray(taus, float)
    reliability_frames, pit_frames = [], []
    identity = ["method", "fold", "cutoff", "geo_value", "reference_date",
                "report_date", "target_date", "lag"]
    for strategy, frame in calibrations.items():
        if frame is None or frame.empty:
            continue
        columns = ([f"q{tau:g}" for tau in taus] if strategy == "base"
                   else [f"cal_q{tau:g}" for tau in taus])
        available = frame.dropna(subset=["truth", *columns]).copy()
        values = available[columns].to_numpy(float)
        truth = available.truth.to_numpy(float)
        scale = np.maximum(np.abs(values), np.abs(truth[:, None]))
        tolerance = COVERAGE_BOUNDARY_TOLERANCE * scale
        covered = truth[:, None] <= values + tolerance
        # Match pandas groupby's historical deterministic method ordering so
        # vectorization changes performance, not the emitted CSV row order.
        method_codes, methods = pd.factorize(available.method, sort=True)
        valid_method = method_codes >= 0
        if valid_method.any():
            counts = np.bincount(method_codes[valid_method], minlength=len(methods))
            successes = np.column_stack([
                np.bincount(method_codes[valid_method], weights=covered[valid_method, index],
                            minlength=len(methods))
                for index in range(len(taus))])
            rates = successes / counts[:, None]
            reliability_frames.append(pd.DataFrame({
                "strategy": strategy,
                "method": np.repeat(methods.to_numpy(), len(taus)),
                "nominal_quantile": np.tile(taus, len(methods)),
                "empirical_coverage": rates.ravel(),
                "coverage_error": (rates - taus).ravel(),
                "n": np.repeat(counts, len(taus)),
            }))
        tied = np.abs(values - truth[:, None]) <= tolerance
        below = np.sum(values < truth[:, None] - tolerance, axis=1)
        at_or_below = np.sum(values <= truth[:, None] + tolerance, axis=1)
        has_tie = tied.any(axis=1)
        lower_levels = np.r_[0., taus]
        upper_levels = np.r_[taus, 1.]
        records = available[[c for c in identity if c in available]].copy()
        records["strategy"] = strategy
        records["pit_lower"] = lower_levels[below]
        # A tie from routed quantiles k through m leaves every probability in
        # the adjacent bins [tau_(k-1), tau_(m+1)] compatible with the
        # observation.  Using tau_m as the upper endpoint would widen only the
        # lower side and bias tied PIT midpoints downward.
        records["pit_upper"] = upper_levels[at_or_below]
        records["pit_midpoint"] = (records.pit_lower + records.pit_upper) / 2
        records["pit_bin"] = below
        records["pit_tied"] = has_tie
        records["pit_tied_quantiles"] = tied.sum(axis=1)
        pit_frames.append(records)
    pit = (pd.concat(pit_frames, ignore_index=True) if pit_frames else pd.DataFrame())
    reliability = (pd.concat(reliability_frames, ignore_index=True)
                   if reliability_frames else pd.DataFrame())
    return reliability, pit


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
    convention used by the assessment tables.
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
