"""Prospective prediction-validity tables (V1).

Pure post-processing. Every function here consumes forecasts that were already
issued and scored -- the standardized long prediction table -- and uses target
values only to evaluate them. Nothing is re-predicted, no model is fitted, and no
threshold is tuned on these outputs.

  V1  Prediction validity: point quality against the flat-line baseline,
      pairwise win/tie/loss and loss saved, interval calibration resolved by lag
      and over the early-lag set, and regime-stratified differences.

The forecast-level reliability and alert layer (former V2) was removed; the
post-forecast re-assessment of issued forecasts lives in ``reassessment.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.defaults import EVALUATION_DEFAULTS

#: Nominal central levels and the routed taus bounding them.
LEVELS = {50: (.25, .75), 80: (.1, .9), 95: (.025, .975)}
EARLY_LAG_MAX = EVALUATION_DEFAULTS.get("early_lag_max", 14)


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
# All V1 tables
# ---------------------------------------------------------------------------

def run_validation(long: pd.DataFrame, *, wide: pd.DataFrame | None = None,
                   calibration: pd.DataFrame | None = None,
                   point_pairs=(), distribution_pairs=(), taus=()) -> dict[str, pd.DataFrame]:
    """Every V1 table, from already-issued forecasts only."""
    tables = {
        "v1_flat_line_comparison": flat_line_comparison(long),
        "v1_pairwise_point": pairwise_loss_comparison(long, point_pairs, "absolute_error"),
        "v1_pairwise_distribution": (pairwise_loss_comparison(long, distribution_pairs, "wis_reporting")
                                     if distribution_pairs else pd.DataFrame()),
        "v1_interval_calibration_by_lag": interval_calibration_by_lag(calibration, taus),
        "v1_regime_stratified_differences": regime_stratified_differences(
            long, wide, list(point_pairs)),
    }
    return {name: frame for name, frame in tables.items() if frame is not None}
