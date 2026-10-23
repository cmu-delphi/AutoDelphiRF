"""Diagnose prepared revision triangles and resolve target lag."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.io import read_csv

REQUIRED = {"geo_value", "reference_date", "report_date", "target_date", "lag",
            "target_lag", "log_value_7dav"}

WEEKDAYS = ("Mon", "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun")
# Share of genuine revisions a weekday needs before it is treated as a
# scheduled release day rather than noise.
ACTIVE_WEEKDAY_SHARE = 0.08
# Minimum support the dropped {others} baseline must retain. At zero the
# retained indicators are identically one and collinear with the intercept.
MIN_BASELINE_SHARE = 1e-9
# Revisions arriving essentially every day: per-day indicators would be six
# noisy columns, so a fixed Mon/Weekends contrast is imposed instead. This is
# a domain prior, not a derivation -- recorded as such in the diagnosis.
UNIFORM_WEEKDAY_GROUPS = {"Mon": ["Mon"], "Weekends": ["Sat", "Sun"]}


def weekday_groups_from_share(weekday_share: dict, temporal_resolution: str) -> dict:
    """Choose the weekday contrasts used by model fitting.

    DelphiRF preprocessing retains the seven canonical weekday indicators.
    AutoDelphiRF chooses a smaller, identifiable set of contrasts from the
    observed reporting pattern. Days outside the returned groups form the
    reference category.
    """
    if temporal_resolution == "weekly":
        return {}
    if temporal_resolution != "daily":
        raise ValueError("temporal_resolution must be 'daily' or 'weekly'")
    normalized = {("Thurs" if day == "Thu" else day): float(value)
                  for day, value in (weekday_share or {}).items()}
    share = {day: normalized.get(day, 0.0) for day in WEEKDAYS}
    active = [day for day in WEEKDAYS if share[day] >= ACTIVE_WEEKDAY_SHARE]
    if len(active) >= 6:
        groups = dict(UNIFORM_WEEKDAY_GROUPS)
    else:
        groups = {day: [day] for day in active}
    covered = {day for members in groups.values() for day in members}
    baseline_share = sum(share[day] for day in WEEKDAYS if day not in covered)
    if groups and baseline_share <= MIN_BASELINE_SHARE:
        # Every observed report day is represented, so the indicators sum to
        # the intercept. Drop the least frequent group as the reference level;
        # a single observed weekday has no estimable weekday contrast.
        reference = min(groups, key=lambda name: sum(share[day] for day in groups[name]))
        groups = {name: members for name, members in groups.items() if name != reference}
    return groups


def derive_revision_profile(prepared: pd.DataFrame) -> dict:
    """Derive reporting-calendar settings from the observed revision process.

    Diagnoses the *revision* process, not the publication schedule: these
    archives are re-issued daily whether or not anything changed, so raw
    report dates say "every weekday" for every dataset. Only genuine revision
    events carry the reporting calendar.

    Returns the weekday one-hot groups and revision-lag terms that the
    preprocessing recipe should use, plus the evidence behind them.
    """
    if "report_date" not in prepared or "genuine_event" not in prepared:
        return {"status": "unavailable: needs report_date and genuine_event"}
    genuine = prepared[prepared.genuine_event.fillna(False).astype(bool)]
    if genuine.empty:
        return {"status": "unavailable: no genuine revision events"}

    names = pd.to_datetime(genuine.report_date).dt.day_name().map({
        "Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed",
        "Thursday": "Thurs", "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun"})
    share = names.value_counts(normalize=True).reindex(WEEKDAYS).fillna(0.)
    active = [day for day in WEEKDAYS if share[day] >= ACTIVE_WEEKDAY_SHARE]

    # Median gap between successive revision report dates for one series.
    gaps = (genuine.sort_values(["geo_value", "reference_date", "report_date"])
            .groupby(["geo_value", "reference_date"], observed=True).report_date
            .apply(lambda s: pd.Series(s.drop_duplicates().diff().dt.days.dropna())))
    median_gap = float(gaps.median()) if len(gaps) else float("nan")

    prior_applied = False
    if len(active) >= 6:
        groups, prior_applied = weekday_groups_from_share(share.to_dict(), "daily"), True
        cadence = "uniform-daily"
    elif not active:
        groups, cadence = {}, "no-dominant-weekday"
    else:
        groups, cadence = {day: [day] for day in active}, "concentrated-weekdays"

    covered = {day for members in groups.values() for day in members}
    baseline_share = float(share[[d for d in WEEKDAYS if d not in covered]].sum())
    # {others} is the dropped reference level. With no support behind it the
    # retained indicators are constant, so no identifiable contrast exists.
    if groups and baseline_share <= MIN_BASELINE_SHARE:
        groups, cadence = {}, "single-weekday (baseline empty, unidentifiable)"

    lag_terms = [1, 7] if np.isfinite(median_gap) and median_gap <= 1.5 else [7, 14]
    return {"status": "derived",
            "weekday_share_pct": {d: round(100 * float(share[d]), 2) for d in WEEKDAYS},
            "active_weekdays": active,
            "weekday_groups": groups,
            "weekday_groups_from_domain_prior": prior_applied,
            "baseline_others_share_pct": round(100 * baseline_share, 2),
            "median_revision_gap_days": median_gap,
            "revision_cadence": cadence,
            "lag_terms": lag_terms,
            "genuine_event_rate_pct": round(100 * float(
                prepared.genuine_event.fillna(False).astype(bool).mean()), 2)}


def observed_weekday_columns(prepared: pd.DataFrame) -> list[str]:
    """Canonical weekday indicators present in the prepared triangle."""
    return [day for day in WEEKDAYS
            if f"{day}_ref" in prepared.columns and f"{day}_issue" in prepared.columns]


def diagnose_prepared_data(prepared: pd.DataFrame, schedule: pd.DataFrame,
                           modeling_profile: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Summarize a prepared triangle.

    ``modeling_profile`` is the single effective profile recorded at
    preparation (``weekday_groups``, ``weekday_groups_source``, ``lag_terms``):
    exactly what training uses. The pattern read off the triangle itself is
    reported for inspection only and never used for modeling.
    """
    target_columns = set(schedule.target_column.dropna().astype(str))
    if len(target_columns) != 1:
        raise ValueError("schedule must define exactly one target_column")
    missing = sorted((REQUIRED | target_columns).difference(prepared.columns))
    if missing:
        raise ValueError(f"prepared triangle missing required columns: {missing}")
    dates = ["reference_date", "report_date", "target_date"]
    summary = {"rows": len(prepared), "locations": int(prepared.geo_value.nunique()),
               "lag_min": int(prepared.lag.min()), "lag_max": int(prepared.lag.max()),
               "target_lags": sorted(map(int, prepared.target_lag.dropna().unique())),
               "schedule_origins": len(schedule), "target_column": next(iter(target_columns)),
               "duplicate_rows": int(prepared.duplicated(["geo_value", "reference_date", "report_date"]).sum())}
    for column in dates:
        summary[f"{column}_min"] = str(pd.to_datetime(prepared[column]).min())
        summary[f"{column}_max"] = str(pd.to_datetime(prepared[column]).max())

    summary["triangle_release_pattern"] = {
        **derive_revision_profile(prepared),
        "note": "read off the prepared triangle for inspection; not used for modeling"}
    effective = dict(modeling_profile or {})

    observed = observed_weekday_columns(prepared)
    expected = list(WEEKDAYS) if observed else []
    summary["weekday_columns_in_triangle"] = observed
    summary["weekday_columns_match_profile"] = (observed == expected)
    summary["weekday_groups_used_for_modeling"] = effective.get("weekday_groups") or {}
    summary["weekday_groups_source"] = effective.get("weekday_groups_source")
    summary["lag_terms_used_for_modeling"] = effective.get("lag_terms")

    quality = pd.DataFrame({"column": prepared.columns,
                            "missing_n": [int(prepared[c].isna().sum()) for c in prepared],
                            "missing_fraction": [float(prepared[c].isna().mean()) for c in prepared]})
    return summary, quality

diagnose = diagnose_prepared_data


# ---------------------------------------------------------------------------
# Target lag
# ---------------------------------------------------------------------------
# The target lag is the user's choice. Its evidence comes only from the raw
# archive (pre_diagnosis.archive.diagnose_raw_archive): a prepared triangle is
# built with an already chosen L and holds no lag at or beyond it, so it cannot
# say where revisions settle.


def resolve_target_lag(diagnosed: dict, user_target_lag: int | None,
                       confirmed: bool, dataset: str) -> tuple[int, str]:
    """Use the user's declared target lag exactly; never infer one."""
    if user_target_lag is None:
        bounds = (diagnosed or {}).get("suggested_target_lag_range") or {}
        evidence = (f" The raw archive suggests {bounds.get('lower')} to {bounds.get('upper')} "
                    "days." if bounds.get("lower") is not None else "")
        raise ValueError(
            f"dataset '{dataset}' has no target lag, and none is inferred from a prepared "
            f"triangle.{evidence} Set target_lag explicitly, or prepare the dataset from its raw "
            "archive (`autodelphirf diagnose --archive ...` shows the evidence, then "
            "`autodelphirf prepare --target-lag L`).")
    return int(user_target_lag), "explicit user choice"
