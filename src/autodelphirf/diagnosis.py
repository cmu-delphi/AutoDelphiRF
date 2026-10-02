from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"geo_value", "reference_date", "report_date", "target_date", "lag",
            "target_lag", "log_value_7dav"}

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
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

    names = pd.to_datetime(genuine.report_date).dt.day_name().str[:3]
    share = names.value_counts(normalize=True).reindex(WEEKDAYS).fillna(0.)
    active = [day for day in WEEKDAYS if share[day] >= ACTIVE_WEEKDAY_SHARE]

    # Median gap between successive revision report dates for one series.
    gaps = (genuine.sort_values(["geo_value", "reference_date", "report_date"])
            .groupby(["geo_value", "reference_date"], observed=True).report_date
            .apply(lambda s: pd.Series(s.drop_duplicates().diff().dt.days.dropna())))
    median_gap = float(gaps.median()) if len(gaps) else float("nan")

    prior_applied = False
    if len(active) >= 6:
        groups, prior_applied = dict(UNIFORM_WEEKDAY_GROUPS), True
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
    """Weekday one-hot groups actually present in the prepared triangle."""
    return sorted({c[:-4] for c in prepared.columns if c.endswith("_ref")}
                  - {"refd"})


def diagnose_prepared_data(prepared: pd.DataFrame, schedule: pd.DataFrame,
                           specified_profile: dict | None = None) -> tuple[dict, pd.DataFrame]:
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

    derived = derive_revision_profile(prepared)
    summary["revision_profile_derived"] = derived
    # A user-supplied profile always wins; the derivation is still reported so
    # a disagreement is visible rather than silent.
    summary["revision_profile_specified"] = specified_profile or None
    summary["revision_profile_effective_source"] = "user-specified" if specified_profile else "derived"
    effective = dict(specified_profile) if specified_profile else {
        k: derived.get(k) for k in ("weekday_groups", "lag_terms")}
    summary["revision_profile_effective"] = effective
    if specified_profile and derived.get("status") == "derived":
        disagreements = {k: {"specified": specified_profile.get(k), "derived": derived.get(k)}
                         for k in ("weekday_groups", "lag_terms")
                         if k in specified_profile and specified_profile[k] != derived.get(k)}
        summary["revision_profile_disagreements"] = disagreements or None

    observed = observed_weekday_columns(prepared)
    expected = sorted((effective.get("weekday_groups") or {}).keys())
    summary["weekday_columns_in_triangle"] = observed
    summary["weekday_columns_match_profile"] = (observed == expected)

    quality = pd.DataFrame({"column": prepared.columns,
                            "missing_n": [int(prepared[c].isna().sum()) for c in prepared],
                            "missing_fraction": [float(prepared[c].isna().mean()) for c in prepared]})
    return summary, quality

diagnose = diagnose_prepared_data


# ---------------------------------------------------------------------------
# Target-lag diagnosis
# ---------------------------------------------------------------------------
# Defaults of the frozen selection rule. The maturity horizon L is the
# EARLIEST cadence-aligned lag at which at least ``completion_rate`` of
# location--reference pairs sit within ``relative_error`` of that episode's
# latest available value. Episodes with less than ``min_followup_days`` of
# follow-up are excluded so a not-yet-mature series cannot be treated as
# finalized. The rule is evidence about the data; it never overrides a user's
# declared choice (see ``resolve_target_lag``).
# The horizon is diagnosed from the data provided: retrospectively that is the
# whole archive, and in deployment whatever was available at the first run
# (pass ``selection_cutoff``). No location--reference-date pair is filtered out
# by default -- a pair simply contributes at the candidate lags where it has an
# as-of value. ``min_followup_days`` and ``max_first_lag`` remain available for
# a caller who wants a population held fixed across lags, but they DISCARD
# provided data, so they are off unless asked for.
TARGET_LAG_RULE = {"completion_rate": 0.90, "relative_error": 0.10,
                   "min_followup_days": 0, "cadence_days": 7, "max_lag": None,
                   "max_first_lag": None}


def diagnose_target_lag(prepared: pd.DataFrame, *, value_column: str = "value_7dav",
                        completion_rate: float = TARGET_LAG_RULE["completion_rate"],
                        relative_error: float = TARGET_LAG_RULE["relative_error"],
                        min_followup_days: int = TARGET_LAG_RULE["min_followup_days"],
                        cadence_days: int = TARGET_LAG_RULE["cadence_days"],
                        max_lag: int | None = None,
                        max_first_lag: int | None = TARGET_LAG_RULE["max_first_lag"],
                        extra_lags=(), selection_cutoff=None) -> dict:
    """Diagnose the maturity horizon L from the observed revision process.

    Returns the selected lag, the per-lag completion evidence behind it, and a
    status string. Selects the earliest cadence-aligned lag where at least
    ``completion_rate`` of tasks have relative discrepancy from their latest
    historical value no greater than ``relative_error``.

    The comparator is each episode's latest available value in the supplied
    archive -- not an assumed final truth -- so a triangle truncated near its
    own configured horizon can only ever recommend a lag inside the range it
    contains. That limit is reported as ``max_lag_examined``.
    """
    if not 0 < completion_rate <= 1 or not 0 <= relative_error or cadence_days < 1:
        raise ValueError("invalid target-lag rule settings")
    if value_column not in prepared.columns:
        return {"selected_target_lag": None, "status": f"no {value_column} column",
                "rule": {}, "completion": []}
    keys = ["geo_value", "reference_date"]
    frame = prepared[keys + ["report_date", "lag", value_column]].dropna(
        subset=[value_column]).copy()
    frame["report_date"] = pd.to_datetime(frame.report_date)
    if selection_cutoff is not None:
        frame = frame[frame.report_date < pd.Timestamp(selection_cutoff)]
    if frame.empty:
        return {"selected_target_lag": None, "status": "no rows before the selection cutoff",
                "rule": {}, "completion": []}
    frame = frame.sort_values(keys + ["report_date"])
    # Only episodes observed far enough past issuance can testify to maturity.
    span = frame.groupby(keys, observed=True).lag.agg(["min", "max"])
    eligible_index = span if not min_followup_days else span[span["max"] >= min_followup_days]
    # ...and only episodes ALSO observed from early on can be compared across
    # candidate lags. A reference date predating the archive's own start first
    # appears at a large lag (p90 = 145 on nssp), so without this filter each
    # candidate lag is scored on a different population -- a third of nssp
    # episodes have no as-of value at lag 14 but do at lag 84 -- and "the
    # earliest lag to cross the rate" compares incomparable subsets.
    if max_first_lag is not None:
        eligible_index = eligible_index[eligible_index["min"] <= max_first_lag]
    mature = eligible_index.index
    frame = frame.set_index(keys)
    frame = frame.loc[frame.index.isin(mature)].reset_index()
    if frame.empty:
        return {"selected_target_lag": None,
                "status": (f"no location-reference pair has both {min_followup_days} days of follow-up "
                           f"and a first release by lag {max_first_lag}"),
                "rule": {}, "completion": []}
    latest = (frame.groupby(keys, observed=True).tail(1)[keys + [value_column]]
              .rename(columns={value_column: "latest_value"}))
    ceiling = int(frame.lag.max() if max_lag is None else min(max_lag, frame.lag.max()))
    # Candidate horizons sit on the retraining cadence (weekly by default).
    candidates = [lag for lag in range(cadence_days, ceiling + 1) if lag % cadence_days == 0]
    # ``extra_lags`` are REPORTED but never selectable: a user's declared
    # horizon is often off the retraining cadence (chng uses 60, not a
    # multiple of 7), and its own within-tolerance rate is exactly what is
    # needed to judge that choice against the rule's criterion.
    reportable = sorted({int(x) for x in extra_lags if 0 < int(x) <= ceiling}
                        .difference(candidates))
    selectable = set(candidates)
    candidates = sorted(set(candidates).union(reportable))
    if not candidates:
        # The archive does not reach the first cadence-aligned horizon, so
        # there is nothing to score. Reported as a status, not raised: it is a
        # property of the data, and the caller (pipeline, or `revroute
        # diagnose`) already handles an unselectable horizon by falling back
        # to the user's declared L. Returning early also keeps the empty
        # candidate list out of ``merge_asof`` below, where an empty
        # ``DataFrame({"lag": []})`` is float64 and fails the exact join-key
        # dtype check against the triangle's integer lag.
        return {"selected_target_lag": None,
                "status": (f"no candidate horizon: the triangle reaches lag {ceiling}, short of "
                           f"the first cadence-aligned horizon at {cadence_days} days. Lower "
                           "cadence_days, or supply a triangle with longer follow-up."),
                "rule": {"cadence_days": cadence_days, "max_lag_examined": ceiling},
                "completion": []}
    # AS-OF comparison. What a forecaster could see at lag l is the latest
    # release at or BEFORE l, carried forward -- not a release dated exactly l.
    # Matching l exactly evaluated only the episodes that happened to publish
    # on that very day, which on a Wed/Fri reporter is ~1.4% of episodes and
    # produced a horizon set by a 96-row sliver. Forward filling gives every
    # mature episode an as-of value at every candidate lag.
    # Explicit key works on the older pandas release still used by the
    # packaged test environment, where ``how="cross"`` incorrectly attempts
    # to infer common columns.
    grid = (frame[keys].drop_duplicates().assign(_cross=1)
            .merge(pd.DataFrame({"lag": candidates, "_cross": 1}), on="_cross")
            .drop(columns="_cross").sort_values("lag", kind="mergesort"))
    asof = pd.merge_asof(grid, frame[keys + ["lag", value_column]].sort_values(
                             "lag", kind="mergesort"),
                         on="lag", by=keys, direction="backward")
    asof = asof.merge(latest, on=keys, how="left").dropna(subset=[value_column, "latest_value"])
    current = asof[value_column].to_numpy(dtype="float64")
    latest_values = asof.latest_value.to_numpy(dtype="float64")
    asof["rel_error"] = (np.abs(current - latest_values)
                         / np.maximum(np.abs(latest_values), 1e-8))
    completion = []
    for lag, group in asof.groupby("lag", observed=True, sort=True):
        values = group.rel_error.replace([np.inf, -np.inf], np.nan).dropna()
        within = ((values <= relative_error) |
                  np.isclose(values, relative_error, rtol=1e-6, atol=1e-8))
        completion.append({
            "lag": int(lag), "pairs": int(len(values)),
            "within_tolerance_rate": float(np.mean(within)) if len(values) else None,
            "median_relative_error": float(values.median()) if len(values) else None})
    eligible = [c for c in completion
                if c["lag"] in selectable
                and c["within_tolerance_rate"] is not None
                and c["within_tolerance_rate"] >= completion_rate]
    rule = {"completion_rate": completion_rate, "relative_error": relative_error,
            "min_followup_days": min_followup_days, "cadence_days": cadence_days,
            "max_first_lag": max_first_lag,
            "reported_only_lags": reportable,
            "max_lag_examined": ceiling, "pairs_used": int(len(mature)),
            "selection_cutoff": None if selection_cutoff is None else str(pd.Timestamp(selection_cutoff).date())}
    if not eligible:
        return {"selected_target_lag": None,
                "status": "no cadence-aligned lag reached the completion-rate criterion",
                "rule": rule, "completion": completion}
    first = eligible[0]
    return {"selected_target_lag": int(first["lag"]),
            "status": (f"earliest lag on a {cadence_days}-day cadence where at least "
                       f"{completion_rate:.0%} of tasks have relative discrepancy "
                       f"<= {relative_error:.0%}"),
            "within_tolerance_rate": first["within_tolerance_rate"],
            "median_relative_error": first["median_relative_error"],
            "rule": rule, "completion": completion}


def resolve_target_lag(diagnosed: dict, user_target_lag: int | None,
                       confirmed: bool, dataset: str) -> tuple[int, str]:
    """Decide which maturity horizon a run uses, and record why.

    The user's declared choice always wins, but never silently: when it
    disagrees with the diagnosis the run stops until the choice is explicitly
    confirmed, so nobody adopts a horizon without having seen the recommended
    one. With no declared choice the diagnosed value is used.
    """
    suggested = diagnosed.get("selected_target_lag")
    if user_target_lag is None:
        if suggested is None:
            raise ValueError(
                f"dataset '{dataset}' has no target_lag in its params file and the "
                f"diagnosis could not select one ({diagnosed.get('status')}). State a "
                "target_lag explicitly.")
        return int(suggested), f"diagnosed ({diagnosed.get('status')})"
    user_target_lag = int(user_target_lag)
    if suggested is None or user_target_lag == suggested:
        return user_target_lag, "user choice (diagnosis agrees or is unavailable)"
    if not confirmed:
        raise ValueError(
            f"dataset '{dataset}' declares target_lag={user_target_lag} but the "
            f"diagnosed horizon is {suggested} ({diagnosed.get('status')}). Set "
            f"\"confirmed\": true for this dataset in the target-lag params file to "
            "proceed with your own choice, or remove target_lag to use the diagnosed one.")
    return user_target_lag, f"user choice, confirmed over diagnosed {suggested}"


def raw_archive_for_diagnosis(dataset: str, manifest_path, max_lag: int = 240):
    """Load a dataset's RAW archive as ``(geo_value, reference_date, lag, value)``.

    The target-lag rule needs follow-up past any candidate horizon, and a
    prepared triangle is truncated at the horizon it was built with -- so
    diagnosing L from it is circular and fails the follow-up requirement on
    every dataset whose triangle stops before 60 days. The raw archive carries
    the full revision history, which is what the rule was defined on.

    Returns ``(frame, note)``; ``frame`` is ``None`` when the archive or its
    manifest entry is unavailable, and ``note`` says why.
    """
    if manifest_path is None:
        return None, ("no raw-archive manifest configured (set AUTODELPHIRF_RAW_ARCHIVE_MANIFEST, or "
                      "place config/raw_archive_manifest.json in the working directory)")
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None, f"no raw-archive manifest at {manifest_path}"
    manifest = json.loads(manifest_path.read_text())
    entry = (manifest.get("datasets", {}) or {}).get(dataset)
    if entry is None:
        return None, f"no raw-archive manifest entry for '{dataset}'"
    path = (manifest_path.parent / entry["raw_csv"]).resolve()
    if not path.exists():
        return None, f"raw archive not found: {path}"
    raw = pd.read_csv(path)
    reference, report = entry["reference_col"], entry["report_col"]
    value, denominator = entry["value_col"], entry.get("denom_col")
    missing = [c for c in (reference, report, value) if c not in raw.columns]
    if missing:
        return None, f"raw archive {path.name} missing {missing}"
    frame = pd.DataFrame({
        "reference_date": pd.to_datetime(raw[reference]),
        "report_date": pd.to_datetime(raw[report]),
        "geo_value": (raw["geo_value"].astype(str) if "geo_value" in raw.columns
                      else "single_location")})
    if denominator and denominator in raw.columns:
        # A two-count fraction signal is diagnosed on the ratio it models.
        frame["value"] = (pd.to_numeric(raw[value], errors="coerce")
                          / pd.to_numeric(raw[denominator], errors="coerce").replace(0, np.nan))
    else:
        frame["value"] = pd.to_numeric(raw[value], errors="coerce")
    frame["lag"] = (frame.report_date - frame.reference_date).dt.days
    frame = frame[frame.lag.between(0, max_lag)].dropna(subset=["value"])
    return frame, f"raw archive {path.name} (lags 0-{max_lag})"
