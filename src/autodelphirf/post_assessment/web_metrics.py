"""CSV score summaries for the web results view."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _location_quantile_summary(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Summarize raw-scale scores across locations after averaging within location.

    This keeps a large or densely observed location from contributing more weight
    than another location to the web evaluation bands.  Missing WIS values are
    expected for methods that do not issue a predictive distribution.
    """
    metrics = {"ae": "absolute_error", "wis": "wis_reporting",
               "ae_pct": "relative_error_pct", "wis_pct": "relative_wis_pct"}
    present = {name: source for name, source in metrics.items() if source in data}
    if not present or "geo_value" not in data:
        return pd.DataFrame(columns=groups + ["metric", "n_locations", "median", "q10", "q90"])
    within_location = (data.groupby(groups + ["geo_value"], observed=True)
                       .agg(**{name: (source, "mean") for name, source in present.items()})
                       .reset_index())
    summaries = []
    for metric in present:
        usable = within_location.dropna(subset=[metric])
        if usable.empty:
            continue
        summary = (usable.groupby(groups, observed=True)[metric]
                   .agg(n_locations="size", median="median",
                        q10=lambda values: values.quantile(.1),
                        q90=lambda values: values.quantile(.9))
                   .reset_index())
        summary["metric"] = metric
        summaries.append(summary)
    columns = groups + ["metric", "n_locations", "median", "q10", "q90"]
    return (pd.concat(summaries, ignore_index=True)[columns]
            if summaries else pd.DataFrame(columns=columns))


def _early_lag_location_summary(data: pd.DataFrame, maximum_lag: int = 14) -> pd.DataFrame:
    """Robust score and variability by location for the early-reporting comparison.

    One row represents one location, method, and score. The web uses the
    median and 10th--90th percentiles across matched cases at observed
    reporting lags no greater than ``maximum_lag``. Mean and standard
    deviation remain in the audit table but are not used for the chart because
    forecast errors are nonnegative and can be extremely heavy-tailed.
    """
    columns = ["geo_value", "method", "metric", "maximum_lag", "n", "mean", "std",
               "median", "q10", "q90"]
    if "geo_value" not in data or "lag" not in data:
        return pd.DataFrame(columns=columns)
    early = data[pd.to_numeric(data.lag, errors="coerce").le(maximum_lag)]
    summaries = []
    for metric, source in (("ae", "absolute_error"), ("wis", "wis_reporting"),
                           ("ae_pct", "relative_error_pct"),
                           ("wis_pct", "relative_wis_pct")):
        if source not in early:
            continue
        usable = early.dropna(subset=[source])
        if usable.empty:
            continue
        summary = (usable.groupby(["geo_value", "method"], observed=True)[source]
                   .agg(n="size", mean="mean", std="std", median="median",
                        q10=lambda values: values.quantile(.1),
                        q90=lambda values: values.quantile(.9)).reset_index())
        summary["metric"] = metric
        summary["maximum_lag"] = int(maximum_lag)
        summaries.append(summary)
    return (pd.concat(summaries, ignore_index=True)[columns]
            if summaries else pd.DataFrame(columns=columns))


def write_web_metrics(data: pd.DataFrame, output: Path) -> None:
    """Save three CSV score summaries for the web results view."""
    tables = Path(output) / "web_tables"
    tables.mkdir(parents=True, exist_ok=True)
    keys = ["fold", "geo_value", "reference_date", "report_date", "lag"]
    if data.empty:
        matched = data.copy()
    else:
        groups = data.groupby(keys, observed=True)
        complete = (groups.method.nunique().eq(data.method.nunique()) &
                    groups.model_available.all())
        matched = data[pd.MultiIndex.from_frame(data[keys]).isin(complete[complete].index)].copy()
    matched["cutoff"] = pd.to_datetime(matched["cutoff"])
    outputs = {
        "web_metrics_by_lag": _location_quantile_summary(matched, ["lag", "method"]),
        "web_metrics_by_origin_lag": _location_quantile_summary(
            matched, ["cutoff", "lag", "method"]),
        "web_metrics_by_location": _early_lag_location_summary(matched, maximum_lag=14),
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
