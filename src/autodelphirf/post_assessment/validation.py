"""Prospective validation protocol V1-V2 (RRDelphiRF_post_prediction_validation_expanded.tex).

Pure post-processing. Every function here consumes forecasts that were already
issued and scored -- the standardized long prediction table and the per-case
reliability record -- and uses finalized values only to evaluate them. Nothing is
re-predicted, no model is fitted, and no threshold is tuned on these outputs.

  V1  Prediction validity (Sec. "V1"): point quality against the flat-line
      baseline, pairwise win/tie/loss and loss saved, and interval calibration
      resolved by lag and over the early-lag set.
  V2  Forecast-alert validity (Sec. "V2"): whether the risk quantities have
      their stated upper-tail meaning, whether realized performance worsens
      across the categorical alert, whether the continuous risk score ranks
      forecasts, and the operational binary summaries at the frozen thresholds.

The alert assigned at issuance is never recomputed here; V2 only reads it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.defaults import CALIBRATION_DEFAULTS, EVALUATION_DEFAULTS

#: Nominal central levels and the routed taus bounding them.
LEVELS = {50: (.25, .75), 80: (.1, .9), 95: (.025, .975)}
EARLY_LAG_MAX = EVALUATION_DEFAULTS.get("early_lag_max", 14)
DEFAULT_ETA = CALIBRATION_DEFAULTS.get("eta", .9)


def _subsets(long: pd.DataFrame, early_lag_max: int = EARLY_LAG_MAX):
    """(name, frame) for the evaluation subsets the protocol requires."""
    yield "overall", long
    if "lag" in long:
        yield "early", long[long.lag <= early_lag_max]


# ---------------------------------------------------------------------------
# V1: prediction validity
# ---------------------------------------------------------------------------

def flat_line_comparison(long: pd.DataFrame, baseline: str = "baseline_null",
                         loss: str = "absolute_error",
                         early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Better / worse / tied against the flat-line baseline, and loss saved.

    The spec asks for counts rather than only averages, because a method can
    lower mean error while losing on most individual cases.
    """
    key = [c for c in ("fold", "geo_value", "reference_date", "report_date", "lag")
           if c in long]
    rows = []
    for subset, frame in _subsets(long, early_lag_max):
        pivot = frame.pivot_table(index=key, columns="method", values=loss, aggfunc="first")
        if baseline not in pivot:
            continue
        for method in pivot.columns:
            if method == baseline:
                continue
            paired = pivot[[method, baseline]].dropna()
            if paired.empty:
                continue
            difference = paired[method].to_numpy(float) - paired[baseline].to_numpy(float)
            rows.append({
                "subset": subset, "method": method, "baseline": baseline, "loss": loss,
                "n": len(difference),
                "better_n": int((difference < 0).sum()),
                "worse_n": int((difference > 0).sum()),
                "tied_n": int((difference == 0).sum()),
                "better_fraction": float((difference < 0).mean()),
                "total_loss_saved": float(-difference.sum()),
                "mean_loss_saved": float(-difference.mean()),
                "median_loss_saved": float(-np.median(difference)),
            })
    return pd.DataFrame(rows)
def pairwise_loss_comparison(long: pd.DataFrame, pairs, loss: str = "absolute_error",
                             early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Mean/median Delta, win/tie/loss probabilities and loss saved per pair.

    ``pairs`` are ordered ``(a, b)`` and ``Delta = L_a - L_b``, so a negative
    mean favours ``a``. Complements the episode-clustered bootstrap in
    :mod:`autodelphirf.post_assessment.inference`, which supplies the interval estimate;
    this supplies the counts the protocol also asks for.
    """
    key = [c for c in ("fold", "geo_value", "reference_date", "report_date", "lag")
           if c in long]
    rows = []
    for subset, frame in _subsets(long, early_lag_max):
        pivot = frame.pivot_table(index=key, columns="method", values=loss, aggfunc="first")
        for left, right in pairs:
            if left not in pivot or right not in pivot:
                continue
            paired = pivot[[left, right]].dropna()
            if paired.empty:
                continue
            delta = paired[left].to_numpy(float) - paired[right].to_numpy(float)
            rows.append({
                "subset": subset, "comparison": f"{left} - {right}", "loss": loss,
                "n": len(delta), "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "win_probability": float((delta < 0).mean()),
                "tie_probability": float((delta == 0).mean()),
                "loss_probability": float((delta > 0).mean()),
                "total_loss_saved": float(-delta.sum()),
                "mean_loss_saved": float(-delta.mean()),
            })
    return pd.DataFrame(rows)
def interval_calibration_by_lag(calibration: pd.DataFrame, taus, levels=LEVELS,
                                early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Base vs calibrated coverage, width and miss rates, resolved by lag.

    The overall table already exists; the protocol additionally requires the
    same checks by exact lag and over the early-lag set, because pooling across
    lags can hide a badly calibrated early region behind later cases with finalized values.
    """
    from .reliability import add_interval_columns
    if calibration is None or calibration.empty:
        return pd.DataFrame()
    base = add_interval_columns(calibration, levels, prefix="")
    calibrated_source = calibration.drop(columns=[f"q{tau:g}" for tau in taus]).rename(
        columns={f"cal_q{tau:g}": f"q{tau:g}" for tau in taus})
    calibrated = add_interval_columns(calibrated_source, levels, prefix="")
    rows = []
    for status, frame in (("base", base), ("calibrated", calibrated)):
        wis_column = "cal_wis_reporting" if status == "calibrated" else "wis_reporting"
        for scope, subset in (("by_lag", frame), ("early", frame[frame.lag <= early_lag_max]),
                              ("overall", frame)):
            grouping = ["method", "lag"] if scope == "by_lag" else ["method"]
            if subset.empty:
                continue
            for keys, group in subset.groupby(grouping, observed=True):
                keys = keys if isinstance(keys, tuple) else (keys,)
                for level in levels:
                    rows.append({
                        "scope": scope, "status": status, "method": keys[0],
                        "lag": keys[1] if len(keys) > 1 else np.nan,
                        "nominal": level / 100, "n": len(group),
                        "coverage": float(group[f"coverage_{level}"].mean()),
                        "coverage_error": float(group[f"coverage_{level}"].mean() - level / 100),
                        "mean_width": float(group[f"width_{level}"].mean()),
                        "median_width": float(group[f"width_{level}"].median()),
                        "lower_miss": float(group[f"lower_miss_{level}"].mean()),
                        "upper_miss": float(group[f"upper_miss_{level}"].mean()),
                        "mean_wis": float(group[wis_column].mean())
                        if wis_column in group else np.nan,
                        "median_wis": float(group[wis_column].median())
                        if wis_column in group else np.nan})
    return pd.DataFrame(rows)


def regime_stratified_differences(long: pd.DataFrame, wide: pd.DataFrame, pairs,
                                  loss: str = "absolute_error", bins: int = 4) -> pd.DataFrame:
    """Pairwise loss differences stratified by prospectively available quantities.

    Explanatory only. Every stratifier is known at issuance -- routed regime,
    distance to the routed medoid, number of regimes, exact lag -- so the bins
    never use a finalized value. These do not become an
    estimator selector.
    """
    if wide is None or wide.empty:
        return pd.DataFrame()
    key = [c for c in ("fold", "geo_value", "reference_date", "report_date", "lag")
           if c in long and c in wide]
    # Predictor-agnostic only: the three-axis protocol treats the upstream
    # model as a black box, and its V1 is "not a comparison of predictor
    # internals". RevRoute-specific stratifiers (routed regime, medoid
    # distance, regime count) are therefore excluded; lag and location are
    # available whatever produced the forecast.
    stratifiers = [c for c in ("lag", "geo_value") if c in wide]
    if not stratifiers:
        return pd.DataFrame()
    # ``lag`` is both a merge key and a stratifier, so select it once.
    context = wide[key + [c for c in stratifiers if c not in key]].drop_duplicates(key)
    pivot = long.pivot_table(index=key, columns="method", values=loss, aggfunc="first").reset_index()
    merged = pivot.merge(context, on=key, how="inner")
    rows = []
    for left, right in pairs:
        if left not in merged or right not in merged:
            continue
        frame = merged.dropna(subset=[left, right]).copy()
        if frame.empty:
            continue
        frame["delta"] = frame[left].to_numpy(float) - frame[right].to_numpy(float)
        for column in stratifiers:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() < 10:
                continue
            if values.nunique() <= bins:
                labels = values.astype("object")
            else:
                try:
                    labels = pd.qcut(values, bins, duplicates="drop")
                except ValueError:
                    continue
            for level, group in frame.groupby(labels, observed=True):
                rows.append({"comparison": f"{left} - {right}", "stratifier": column,
                             "bin": str(level), "n": len(group),
                             "mean_delta": float(group.delta.mean()),
                             "median_delta": float(group.delta.median()),
                             "win_probability": float((group.delta < 0).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# V2: forecast-alert validity
# ---------------------------------------------------------------------------

def reliability_coverage(reliability: pd.DataFrame, eta: float = DEFAULT_ETA,
                         early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Do the upper-tail risk quantities mean what they claim?

    ``Cov_eta^err`` is the share of cases whose realized error falls at or below
    the issued ``q_error`` (spec V2.1); for eta = 0.9, a value near 0.90
    supports the stated historical upper-tail interpretation. The companion
    check for ``q_regret`` is gone with the quantity itself; conditional harm
    severity is checked by ``harm_severity_validation`` instead, because its
    upper-tail claim holds only among realized-harm cases.
    """
    if reliability is None or reliability.empty:
        return pd.DataFrame()
    rows = []
    subsets = list(_subsets(reliability, early_lag_max))
    if "lag" in reliability:
        subsets.extend((f"lag={int(lag)}", frame) for lag, frame in reliability.groupby("lag", observed=True))
    for subset, frame in subsets:
        for method, group in frame.groupby("method", observed=True):
            usable = group[np.isfinite(group.get("q_error", np.nan))]
            if usable.empty:
                continue
            rows.append({
                "subset": subset, "method": method, "eta": eta, "n": len(usable),
                "coverage_error_quantity": float(usable.qerr_covered.mean())
                if "qerr_covered" in usable else np.nan,
                "coverage_error_gap": (float(usable.qerr_covered.mean()) - eta)
                if "qerr_covered" in usable else np.nan,
                "mean_q_error": float(usable.q_error.mean()),
                "mean_realized_ae": float(usable.realized_ae.mean()),
                "realized_harm_frequency": float((usable.realized_regret > 0).mean()),
                "mean_n_err": float(usable.n_err.mean()) if "n_err" in usable else np.nan,
                "mharm_issuable_share": (float(np.isfinite(usable.m_harm).mean())
                                         if "m_harm" in usable else np.nan),
            })
    return pd.DataFrame(rows)


def alert_stratification(reliability: pd.DataFrame, long: pd.DataFrame | None = None,
                         early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Realized performance by the categorical forecast alert.

    The intended ordering is supported < caution < high risk in realized error,
    in distribution. WIS is merged in from the long table when available; the
    reliability record itself carries only point quantities.
    """
    if reliability is None or reliability.empty or "reliability_level" not in reliability:
        return pd.DataFrame()
    frame = reliability
    if long is not None and "wis_reporting" in long:
        key = [c for c in ("method", "fold", "geo_value", "reference_date", "report_date", "lag")
               if c in frame and c in long]
        if key:
            frame = frame.merge(long[key + ["wis_reporting"]].drop_duplicates(key), on=key, how="left")
    rows = []
    for subset, subframe in _subsets(frame, early_lag_max):
        for (method, level), group in subframe.groupby(["method", "reliability_level"],
                                                       observed=True):
            rows.append({
                "subset": subset, "method": method, "reliability_level": level, "n": len(group),
                "mean_ae": float(group.realized_ae.mean()),
                "median_ae": float(group.realized_ae.median()),
                "mean_wis": float(group.wis_reporting.mean()) if "wis_reporting" in group else np.nan,
                "median_wis": float(group.wis_reporting.median()) if "wis_reporting" in group else np.nan,
                "harm_probability": float((group.realized_regret > 0).mean()),
                "mean_regret": float(group.realized_regret.mean()),
                "median_regret": float(group.realized_regret.median()),
                "p90_regret": float(group.realized_regret.quantile(.9)),
            })
    return pd.DataFrame(rows)


def risk_discrimination(reliability: pd.DataFrame, bins: int = 5,
                        early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Does the continuous risk signal rank forecasts by realized badness?

    Bins the issued risk quantities and reports realized loss across ordered
    bins, plus the rank association the spec names (Spearman). Categorical
    thresholds are secondary to this.
    """
    if reliability is None or reliability.empty:
        return pd.DataFrame()
    rows = []
    subsets = list(_subsets(reliability, early_lag_max))
    if "lag" in reliability:
        subsets.extend((f"lag={int(lag)}", frame) for lag, frame in reliability.groupby("lag", observed=True))
    for subset, frame in subsets:
        for method, group in frame.groupby("method", observed=True):
            # Risk-score bin edges are frozen a priori on its [0, 1] scale.
            # q-error/regret bins require method-specific inner-reference
            # cutpoints and are intentionally not estimated from outer outcomes.
            # Revised-harm spec V2.1: discrimination is evaluated on the
            # difficulty component q_err only. ``q_regret`` was REMOVED by the
            # spec ("did not discriminate realized regret reliably"), and the
            # measurement here agreed: non-monotone in realized AE on 6 of 6
            # datasets. ``f_harm`` is judged by discrimination in
            # ``harm_frequency_discrimination`` -- never by probability
            # accuracy, since it is not offered as a probability; ``M_harm`` by
            # ``harm_severity_validation``.
            for signal in ("q_error",):
                if signal not in group:
                    continue
                usable = group[np.isfinite(group[signal]) & np.isfinite(group.realized_ae)]
                if len(usable) < 10:
                    continue
                spearman = float(pd.Series(usable[signal]).corr(
                    pd.Series(usable.realized_ae), method="spearman"))
                spearman_regret = float(pd.Series(usable[signal]).corr(
                    pd.Series(usable.realized_regret), method="spearman"))
                # QUANTILE bins, per spec V2.1 ("quantile bins of q_err").
                # Fixed [0,1] edges were right for the retired percentile
                # risk_score but meaningless for q_err, which carries target
                # units -- on nssp (values ~0-0.08) they put 99% of cases in
                # one bin. Quantile edges use only the issued signal, never
                # the outer outcome, so they remain admissible.
                try:
                    labels = pd.qcut(usable[signal], bins, duplicates="drop", labels=False)
                except ValueError:
                    continue
                if labels.isna().all():
                    continue
                for index, bucket in usable.groupby(labels, observed=True):
                    rows.append({
                        "subset": subset, "method": method, "signal": signal,
                        "bin": int(index), "n": len(bucket),
                        "signal_min": float(bucket[signal].min()),
                        "signal_max": float(bucket[signal].max()),
                        "mean_ae": float(bucket.realized_ae.mean()),
                        "median_ae": float(bucket.realized_ae.median()),
                        "mean_regret": float(bucket.realized_regret.mean()),
                        "harm_probability": float((bucket.realized_regret > 0).mean()),
                        "spearman_signal_vs_ae": spearman,
                        "spearman_signal_vs_regret": spearman_regret})
    return pd.DataFrame(rows)


def operational_alert_summary(reliability: pd.DataFrame,   # RETIRED: see note below
                              alert_levels=("caution", "high risk"),
                              early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Sensitivity, specificity, PPV, NPV and false-alert rate at frozen thresholds.

    The "bad forecast" event is ``realized_regret > 0`` -- the forecast did
    worse than the flat-line baseline it was meant to improve on. This is a
    prespecified, outcome-independent definition, and these binary summaries
    describe the frozen alert operating point; they do not replace the
    continuous calibration above.
    """
    if reliability is None or reliability.empty or "reliability_level" not in reliability:
        return pd.DataFrame()
    rows = []
    for subset, frame in _subsets(reliability, early_lag_max):
        for method, group in frame.groupby("method", observed=True):
            usable = group[np.isfinite(group.realized_regret)]
            if usable.empty:
                continue
            bad = (usable.realized_regret > 0).to_numpy()
            alert = usable.reliability_level.isin(alert_levels).to_numpy()
            true_positive = int((alert & bad).sum())
            false_positive = int((alert & ~bad).sum())
            false_negative = int((~alert & bad).sum())
            true_negative = int((~alert & ~bad).sum())
            def ratio(numerator, denominator):
                return float(numerator / denominator) if denominator else np.nan
            rows.append({
                "subset": subset, "method": method,
                "alert_levels": "|".join(alert_levels), "n": len(usable),
                "bad_forecast_rate": float(bad.mean()), "alert_rate": float(alert.mean()),
                "true_positive": true_positive, "false_positive": false_positive,
                "false_negative": false_negative, "true_negative": true_negative,
                "sensitivity": ratio(true_positive, true_positive + false_negative),
                "specificity": ratio(true_negative, true_negative + false_positive),
                "positive_predictive_value": ratio(true_positive, true_positive + false_positive),
                "negative_predictive_value": ratio(true_negative, true_negative + false_negative),
                "false_alert_rate": ratio(false_positive, false_positive + true_negative)})
    return pd.DataFrame(rows)


def run_validation(long: pd.DataFrame, *, wide: pd.DataFrame | None = None,
                   calibration: pd.DataFrame | None = None,
                   reliability: pd.DataFrame | None = None,
                   point_pairs=(), distribution_pairs=(), taus=()) -> dict[str, pd.DataFrame]:
    """Every V1 and V2 table, from already-issued forecasts only."""
    tables = {
        "v1_flat_line_comparison": flat_line_comparison(long),
        "v1_pairwise_point": pairwise_loss_comparison(long, point_pairs, "absolute_error"),
        "v1_pairwise_distribution": (pairwise_loss_comparison(long, distribution_pairs, "wis_reporting")
                                     if distribution_pairs else pd.DataFrame()),
        "v1_interval_calibration_by_lag": interval_calibration_by_lag(calibration, taus),
        "v1_regime_stratified_differences": regime_stratified_differences(
            long, wide, list(point_pairs)),
        # V2, revised-harm spec: the three components of R_rel are validated
        # against their OWN outcomes and never combined -- q_err by upper-tail
        # coverage and discrimination, f_harm by discrimination and support,
        # M_harm by conditional severity on its own comparable set. Neither
        # q_regret nor any harm probability exists.
        "v2_reliability_coverage": reliability_coverage(reliability),
        "v2_qerr_discrimination": risk_discrimination(reliability),
        "v2_fharm_discrimination": harm_frequency_discrimination(reliability),
        "v2_mharm_severity": harm_severity_validation(reliability),
        "v2_qerr_vs_interval_width": interval_width_value_added(reliability, long),
        "v2_opposed_objective_diagnostic": opposed_objective_diagnostic(reliability),
        # Retained for support accounting, no longer a risk category.
        "v2_profile_availability": alert_stratification(reliability, long),
    }
    return {name: frame for name, frame in tables.items() if frame is not None}


# RETIRED: operational_alert_summary summarises the categorical alert system
# that the spec removed. It is kept only so archived runs remain readable and
# is deliberately NOT wired into run_validation.


# ---------------------------------------------------------------------------
# Post-prediction validation of the historical assessments (Sec. 3.2.5)
# ---------------------------------------------------------------------------

def harm_frequency_discrimination(reliability: pd.DataFrame, bins: int = 5,
                                  early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Descriptive/discriminative value of ``f_harm`` (final spec V2b).

    NO probability-accuracy analysis is performed, and that is deliberate:
    ``f_harm`` is a comparable-history frequency, not a prospective
    probability, so a reliability diagram against the identity line, a Brier
    score, and a comparison with a constant-probability predictor are all out
    of scope. The previous revision did score it that way and it failed on 4 of
    5 datasets -- which is the reason the probability claim was dropped rather
    than the metric.

    What is assessed instead: whether realized harm frequency rises across
    bins of issued ``f_harm`` (discrimination), with the support and Wilson
    width that make each bin interpretable, plus the temporal drift of the
    harm base rate -- a drifting base rate is exactly what makes any fixed
    historical frequency a poor guide to the present.
    """
    if reliability.empty or "f_harm" not in reliability:
        return pd.DataFrame()
    rows = []
    for subset, frame in _subsets(reliability, early_lag_max):
        frame = frame.dropna(subset=["f_harm", "realized_regret"])
        if frame.empty:
            continue
        for method, group in frame.groupby("method", observed=True):
            issued = group.f_harm.astype(float)
            realized = (group.realized_regret > 0).astype(float)
            spearman = (float(pd.Series(issued).corr(realized, method="spearman"))
                        if issued.nunique() > 1 else np.nan)
            # Temporal stability of the base rate, per spec V2b item (4).
            drift = np.nan
            if "cutoff" in group:
                by_origin = realized.groupby(pd.to_datetime(group.cutoff)).mean()
                if len(by_origin) > 2:
                    drift = float(by_origin.max() - by_origin.min())
            try:
                edges = pd.qcut(issued, bins, duplicates="drop", labels=False)
            except ValueError:
                edges = pd.Series(np.zeros(len(issued), int), index=issued.index)
            for value, cell in group.groupby(edges, observed=True):
                cell_realized = (cell.realized_regret > 0).astype(float)
                rows.append({
                    "subset": subset, "method": method, "bin": int(value), "n": len(cell),
                    "mean_issued_f_harm": float(cell.f_harm.astype(float).mean()),
                    "realized_harm_frequency": float(cell_realized.mean()),
                    "mean_k_harm": float(cell.k_harm.mean()) if "k_harm" in cell else np.nan,
                    "mean_n_harmfreq": float(cell.n_harmfreq.mean()) if "n_harmfreq" in cell else np.nan,
                    "mean_wilson_width": (float((cell.f_harm_wilson_high - cell.f_harm_wilson_low).mean())
                                          if "f_harm_wilson_high" in cell else np.nan),
                    "spearman_fharm_vs_harm": spearman,
                    "base_rate": float(realized.mean()),
                    "base_rate_drift_across_origins": drift})
    return pd.DataFrame(rows)


def interval_width_value_added(reliability: pd.DataFrame, long: pd.DataFrame,
                               level: str = "0.95", early_lag_max: int = EARLY_LAG_MAX
                               ) -> pd.DataFrame:
    """Does ``q_eta^err`` add information beyond calibrated interval width? (V2a')

    The upstream system already issues calibrated predictive intervals, so a
    difficulty quantity earns its place only if it carries information those
    intervals do not. The decisive test is the third one: whether
    ``q_eta^err`` still stratifies realized AE WITHIN broad bins of the width
    ``W_q``. Marginal association alone cannot distinguish added information
    from a relabelling of width.
    """
    if reliability is None or reliability.empty or long is None or long.empty:
        return pd.DataFrame()
    lower, upper = f"q{(1 - float(level)) / 2:g}", f"q{1 - (1 - float(level)) / 2:g}"
    if lower not in long or upper not in long:
        return pd.DataFrame()
    keys = [c for c in ("method", "fold", "geo_value", "reference_date", "lag") if c in long.columns
            and c in reliability.columns]
    widths = long[keys + [lower, upper]].copy()
    widths["interval_width"] = widths[upper].astype(float) - widths[lower].astype(float)
    merged = reliability.merge(widths[keys + ["interval_width"]], on=keys, how="inner")
    merged = merged[np.isfinite(merged.interval_width) & np.isfinite(merged.q_error)
                    & np.isfinite(merged.realized_ae)]
    if len(merged) < 40:
        return pd.DataFrame()
    rows = []
    for method, group in merged.groupby("method", observed=True):
        marginal = float(pd.Series(group.q_error).corr(pd.Series(group.interval_width),
                                                       method="spearman"))
        try:
            width_bins = pd.qcut(group.interval_width, 3, labels=False, duplicates="drop")
        except ValueError:
            width_bins = pd.Series(np.zeros(len(group), int), index=group.index)
        for band, cell in group.groupby(width_bins, observed=True):
            if len(cell) < 20:
                continue
            try:
                inner = pd.qcut(cell.q_error, 3, labels=False, duplicates="drop")
            except ValueError:
                continue
            trend = cell.groupby(inner, observed=True).realized_ae.mean()
            if len(trend) < 2:
                continue
            rows.append({
                "method": method, "level": level, "width_band": int(band), "n": len(cell),
                "spearman_qerr_vs_width": marginal,
                "mean_interval_width": float(cell.interval_width.mean()),
                "ae_lowest_qerr_tercile": float(trend.iloc[0]),
                "ae_highest_qerr_tercile": float(trend.iloc[-1]),
                "within_band_ae_ratio": float(trend.iloc[-1] / trend.iloc[0]) if trend.iloc[0] > 0 else np.nan,
                "stratifies_within_band": bool(trend.iloc[-1] > trend.iloc[0]),
                "spearman_qerr_vs_ae_within_band": float(
                    pd.Series(cell.q_error).corr(pd.Series(cell.realized_ae), method="spearman"))})
    return pd.DataFrame(rows)


def harm_severity_validation(reliability: pd.DataFrame, eta: float = DEFAULT_ETA,
                             early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Does ``M_harm_eta`` describe harm severity CONDITIONAL on harm? (V2.3)

    Two claims are tested, both restricted to cases where harm was realized,
    which is what makes them statements about severity rather than frequency:

      * the conditional upper-tail interpretation
        ``Cov_eta^M = (1/n_+) sum_{q: H_q=1} 1{Delta_q <= M_harm_eta(q)}``,
        which should sit near ``eta``; and
      * discrimination -- whether a larger issued ``M_harm`` goes with a larger
        realized positive regret, by rank association and within quantile bins
        of the issued value.

    "If M_harm_eta does not discriminate or achieve a defensible conditional
    upper-tail interpretation across datasets, it is abandoned rather than
    replaced by the failed unconditional regret quantile." This table is the
    evidence for that decision.
    """
    if reliability is None or reliability.empty or "m_harm" not in reliability:
        return pd.DataFrame()
    rows = []
    for subset, frame in _subsets(reliability, early_lag_max):
        for method, group in frame.groupby("method", observed=True):
            harmed = group[group.get("realized_harm", False).astype(bool)
                           & np.isfinite(group.m_harm)
                           & np.isfinite(group.realized_regret)]
            if len(harmed) < 10:
                continue
            severity = harmed.realized_regret.to_numpy(float)
            issued = harmed.m_harm.to_numpy(float)
            spearman = float(pd.Series(issued).corr(pd.Series(severity), method="spearman"))
            bins = min(5, max(2, len(harmed) // 20))
            try:
                labels = pd.qcut(pd.Series(issued), bins, labels=False, duplicates="drop")
            except ValueError:
                labels = pd.Series(np.zeros(len(harmed), int))
            grouped = pd.DataFrame({"bin": labels.to_numpy(), "severity": severity}).groupby("bin")
            trend = grouped.severity.mean()
            rows.append({
                "subset": subset, "method": method,
                "n_harm": int(len(harmed)),
                "conditional_coverage": float(np.mean(severity <= issued)),
                "eta": float(eta),
                "coverage_error": float(np.mean(severity <= issued) - eta),
                "spearman_mharm_vs_severity": spearman,
                "mean_severity_lowest_bin": float(trend.iloc[0]) if len(trend) else np.nan,
                "mean_severity_highest_bin": float(trend.iloc[-1]) if len(trend) else np.nan,
                "monotone_in_bins": bool(np.all(np.diff(trend.to_numpy()) >= 0)) if len(trend) > 1 else False,
                "insufficient_severity_share": float(
                    (group.get("harm_severity_status", pd.Series(dtype=object))
                     == "insufficient harm-severity history").mean())
                if "harm_severity_status" in group else np.nan})
    return pd.DataFrame(rows)


def opposed_objective_diagnostic(reliability: pd.DataFrame,
                                 early_lag_max: int = EARLY_LAG_MAX) -> pd.DataFrame:
    """Association of ``q_err`` with regret and harm (spec V2.4).

    A NEGATIVE association here is not a failure of ``q_err``: difficult,
    high-revision cases give correction more room to help, while already
    settled cases are easy to predict yet vulnerable to needless correction.
    This table is the evidence for keeping the two axes separate rather than
    collapsing them, so it reports the sign explicitly.
    """
    if reliability.empty or "q_error" not in reliability:
        return pd.DataFrame()
    rows = []
    for subset, frame in _subsets(reliability, early_lag_max):
        frame = frame.dropna(subset=["q_error", "realized_ae", "realized_regret"])
        if len(frame) < 3:
            continue
        for method, group in frame.groupby("method", observed=True):
            if len(group) < 3:
                continue
            harm = (group.realized_regret > 0).astype(float)
            def rank(a, b):
                return (float(pd.Series(a).corr(pd.Series(b).reset_index(drop=True),
                                                method="spearman"))
                        if pd.Series(a).nunique() > 1 else np.nan)
            error = group.q_error.reset_index(drop=True)
            rows.append({
                "subset": subset, "method": method, "n": len(group),
                "spearman_qerr_vs_ae": rank(error, group.realized_ae),
                "spearman_qerr_vs_regret": rank(error, group.realized_regret),
                "spearman_qerr_vs_harm": rank(error, harm),
                "spearman_fharm_vs_harm": (rank(group.f_harm.reset_index(drop=True), harm)
                                           if "f_harm" in group else np.nan),
                "interpretation": "negative qerr-vs-regret/harm supports keeping the axes separate"})
    return pd.DataFrame(rows)
# End of validation helpers.
