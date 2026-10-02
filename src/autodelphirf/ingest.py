"""Inspect raw revision archives and recommend preprocessing settings."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

EPS = 1e-8
DEFAULT_CANDIDATE_LAGS = (0, 7, 14, 21, 28, 35, 42, 56, 70, 84, 98, 120, 150, 180, 240, 300, 365)
DEFAULT_COMPLETION_THRESHOLD = 0.10
DEFAULT_COMPLETION_RATE = 0.90
DEFAULT_MATURITY_FLOOR_DAYS = 60
# Fixed design constant, not suggested from data (see module docstring).
DEFAULT_TRAINING_WINDOW_DAYS = 180


@dataclass(frozen=True)
class DiagnosisReport:
    """Recommendation and audit record for one input archive."""
    n_rows: int
    n_locations: int
    reference_date_min: str
    reference_date_max: str
    report_date_min: str
    report_date_max: str
    reference_axis_weekdays: dict
    report_axis_weekdays: dict
    reference_axis_resolution: str
    report_axis_resolution: str
    temporal_resolution: str
    genuine_event_rate: float
    genuine_event_rate_denominator: str
    genuine_event_weekday_share: dict
    reference_axis_cadence_days: float
    reference_axis_feature_lags: list
    revision_magnitude_gap_days: dict
    recommended_target_lag: int
    target_lag_completion_curve: dict
    target_lag_completion_band: dict
    user_target_lag: int | None
    resolved_target_lag: int
    target_lag_resolution: str
    target_lag_confirmation_prompt: str | None
    training_window_days: int
    training_window_is_fixed_default: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def resolve_target_lag(suggested_lag: int, user_target_lag: int | None = None) -> dict:
    """Resolve the final target lag L from the automatic suggestion
    ``suggested_lag`` and an optional user-specified value:

      * No user-specified lag: use the suggestion outright. No confirmation
        needed.
      * User-specified lag larger than the suggestion: a longer lag can only
        make the mature target more stable, so the default is to keep the
        user's larger value. The user is told what the suggestion was and
        asked whether they want to stick with their larger value; if they
        do not respond, the default keeps their larger value.
      * User-specified lag smaller than the suggestion: overriding downward
        risks an immature target, so the default is to use the suggested
        (larger) value instead. The user is told what the suggestion was and
        asked whether they want to stick with their smaller value; if they
        do not respond, the default uses the suggested value, not their
        smaller one.
      * User-specified lag equal to the suggestion: no conflict, no
        confirmation needed.

    Returns a dict with ``resolved_target_lag`` (the value actually used
    absent a reply), ``target_lag_resolution`` (a fixed tag describing which
    branch fired), and ``target_lag_confirmation_prompt`` (the message to
    show the user, or ``None`` when no confirmation is needed).
    """
    if user_target_lag is None:
        return {"resolved_target_lag": int(suggested_lag), "target_lag_resolution": "suggested_no_user_value",
                "target_lag_confirmation_prompt": None}
    user_target_lag = int(user_target_lag)
    suggested_lag = int(suggested_lag)
    if user_target_lag == suggested_lag:
        return {"resolved_target_lag": user_target_lag, "target_lag_resolution": "user_matches_suggested",
                "target_lag_confirmation_prompt": None}
    if user_target_lag > suggested_lag:
        prompt = (f"The automatic diagnosis suggests a target lag of {suggested_lag} days; "
                  f"you specified {user_target_lag} days, which is larger. A longer lag can only "
                  f"make the mature target more stable, so the default is to keep your specified "
                  f"{user_target_lag}-day lag. Reply if you would rather use the suggested "
                  f"{suggested_lag}-day lag instead.")
        return {"resolved_target_lag": user_target_lag, "target_lag_resolution": "user_larger_kept_by_default",
                "target_lag_confirmation_prompt": prompt}
    prompt = (f"The automatic diagnosis suggests a target lag of {suggested_lag} days; you specified "
              f"{user_target_lag} days, which is smaller. Overriding the suggestion downward risks an "
              f"immature target, so the default is to use the suggested {suggested_lag}-day lag "
              f"instead of your {user_target_lag}-day value. Reply if you would rather stick with "
              f"your smaller {user_target_lag}-day lag.")
    return {"resolved_target_lag": suggested_lag, "target_lag_resolution": "suggested_kept_over_smaller_user_value",
            "target_lag_confirmation_prompt": prompt}


def _axis_resolution(dates: pd.Series) -> tuple[str, dict]:
    """finest-observed-cadence rule: daily if the axis visits more than one
    weekday, weekly if it is confined to a single weekday."""
    unique_dates = pd.to_datetime(pd.Series(dates.unique()))
    weekdays = unique_dates.dt.dayofweek
    share = weekdays.value_counts(normalize=True).sort_index()
    share.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in share.index]
    resolution = "daily" if weekdays.nunique() > 1 else "weekly"
    return resolution, share.round(4).to_dict()


def _genuine_events(archive: pd.DataFrame) -> pd.DataFrame:
    """genuine(i,t,s) = 1(Y_its != Y_it,s-): value changed from the
    immediately preceding released vintage for the same (location,
    reference-date) pair. Fixed rule, no tuned threshold."""
    # ``archive`` arrives pre-sorted by (geo_value, reference_date, lag) from
    # ``diagnose_raw_archive``, which is equivalent to sorting by report_date
    # within each group; no further sort is needed here.
    previous = archive.groupby(["geo_value", "reference_date"], observed=True)["value"].shift()
    archive = archive.assign(previous_value=previous)
    archive["genuine_event"] = archive.previous_value.notna() & (archive.value != archive.previous_value)
    return archive


def _revision_magnitude_gaps(archive: pd.DataFrame) -> dict:
    """Report-axis revision-magnitude gaps: day-spacing between successive
    genuine reports for the same (location, reference-date) pair. Kept
    distinct from reference-axis feature lags."""
    genuine = archive[archive.genuine_event]  # already sorted; see note above
    gap_days = genuine.groupby(["geo_value", "reference_date"], observed=True)["report_date"].diff().dt.days
    gap_days = gap_days.dropna()
    if not len(gap_days):
        return {"n": 0}
    quantiles = gap_days.quantile([0.25, 0.5, 0.75]).round(2).to_dict()
    common = gap_days.value_counts().head(3).to_dict()
    return {"n": int(len(gap_days)), "q25": quantiles[0.25], "median": quantiles[0.5],
            "q75": quantiles[0.75], "most_common_gap_days": {int(k): int(v) for k, v in common.items()}}


def _target_lag_suggestion(archive: pd.DataFrame, candidate_lags, maturity_floor_days: int,
                           completion_threshold: float,
                           completion_rate: float) -> tuple[int, dict, dict]:
    """Smallest candidate lag where the requested share of mature episodes
    has relative completion error within ``completion_threshold``.
    """
    cutoff = archive.report_date.max()
    mature_pairs = archive[["geo_value", "reference_date"]].drop_duplicates()
    mature_pairs = mature_pairs[(cutoff - mature_pairs.reference_date).dt.days >= maturity_floor_days]
    keys = ["geo_value", "reference_date"]
    # ``archive`` is already sorted by (geo_value, reference_date, lag).
    last_value = archive.groupby(keys, observed=True).last()[["value"]].rename(columns={"value": "y_last"})
    curve = {}
    band = {}
    chosen = None
    for lag in candidate_lags:
        at_lag = archive[archive.lag <= lag].groupby(keys, observed=True).last()[["value"]]
        merged = mature_pairs.merge(at_lag, on=keys, how="inner").merge(last_value, on=keys, how="inner")
        if merged.empty:
            continue
        current = merged.value.to_numpy(dtype="float64")
        latest_values = merged.y_last.to_numpy(dtype="float64")
        relative_error = pd.Series(
            np.abs(current - latest_values) / np.maximum(np.abs(latest_values), EPS),
            index=merged.index,
        )
        within = ((relative_error <= completion_threshold) |
                  np.isclose(relative_error, completion_threshold,
                             rtol=1e-6, atol=1e-8))
        within_tolerance = float(np.mean(within))
        # Preserve the established diagnostic curve consumed by the website.
        # Selection uses the task-level completion rate below.
        curve[int(lag)] = round(float(relative_error.median()), 4)
        band[int(lag)] = {
            "q10": round(float(relative_error.quantile(0.10)), 4),
            "q90": round(float(relative_error.quantile(0.90)), 4),
        }
        if chosen is None and within_tolerance >= completion_rate:
            chosen = int(lag)
    if chosen is None:
        chosen = max(curve) if curve else int(max(candidate_lags))
    return chosen, curve, band


def _reference_axis_feature_lags(archive: pd.DataFrame) -> tuple[float, list]:
    """Reference-axis cadence and the recommended revision-feature lags.

    Reference-axis lagged terms compare neighboring reference dates at a
    fixed maturity and must respect the reference-date cadence. The cadence
    Delta_ref is the
    typical spacing, in days, between consecutive reference dates for a
    location; the recommended revision-feature lags are its first two
    multiples, {Delta_ref, 2*Delta_ref} -- e.g. a weekly-cadence stream
    (Delta_ref=7) recommends the 7- and 14-day revision-difference features
    already used by the residual specification. This is a fixed,
    dataset-agnostic derivation, not a tuned choice.
    """
    per_location_gaps = []
    for _, dates in archive.groupby("geo_value", observed=True)["reference_date"]:
        unique_sorted = pd.Series(dates.unique()).sort_values()
        gaps = unique_sorted.diff().dt.days.dropna()
        if len(gaps):
            per_location_gaps.append(float(gaps.median()))
    if not per_location_gaps:
        return float("nan"), []
    cadence = float(np.median(per_location_gaps))
    cadence_days = int(round(cadence)) if cadence >= 1 else 1
    return cadence, [cadence_days, 2 * cadence_days]


def diagnose_raw_archive(raw: pd.DataFrame, *, geo_value: str = "geo_value",
                         reference_date: str = "reference_date", report_date: str = "report_date",
                         value: str = "value", candidate_lags=DEFAULT_CANDIDATE_LAGS,
                         completion_threshold: float = DEFAULT_COMPLETION_THRESHOLD,
                         completion_rate: float = DEFAULT_COMPLETION_RATE,
                         maturity_floor_days: int = DEFAULT_MATURITY_FLOOR_DAYS,
                         training_window_days: int | None = None,
                         user_target_lag: int | None = None) -> DiagnosisReport:
    """Diagnose a raw revision archive and recommend RevRoute's preprocessing
    configuration. ``raw`` needs only the four semantic columns named by the
    keyword arguments.

    ``training_window_days`` is a fixed default (180 days,
    ``DEFAULT_TRAINING_WINDOW_DAYS``) unless the caller overrides it; it is
    never suggested from the archive itself (see module docstring).

    ``user_target_lag``, if given, is resolved against the automatic
    suggestion via ``resolve_target_lag`` (see that function's docstring for
    the confirmation policy): the larger of the two is kept by default when
    they disagree, with the smaller one flagged for confirmation either way.
    """
    if not 0 < completion_rate <= 1 or completion_threshold < 0:
        raise ValueError("completion_rate must be in (0, 1] and completion_threshold non-negative")
    required = {geo_value, reference_date, report_date, value}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"raw archive missing required columns: {sorted(missing)}")
    archive = raw[[geo_value, reference_date, report_date, value]].rename(columns={
        geo_value: "geo_value", reference_date: "reference_date",
        report_date: "report_date", value: "value"})
    archive["geo_value"] = archive.geo_value.astype("category")
    archive["value"] = pd.to_numeric(archive.value, errors="coerce", downcast="float")
    archive["reference_date"] = pd.to_datetime(archive.reference_date)
    archive["report_date"] = pd.to_datetime(archive.report_date)
    archive = archive.dropna(subset=["geo_value", "reference_date", "report_date", "value"])
    archive = archive[archive.report_date >= archive.reference_date]
    archive["lag"] = (archive.report_date - archive.reference_date).dt.days.astype("int32")
    archive = archive.drop_duplicates(["geo_value", "reference_date", "report_date"], keep="last")
    archive = archive.sort_values(["geo_value", "reference_date", "lag"]).reset_index(drop=True)

    reference_resolution, reference_weekdays = _axis_resolution(archive.reference_date)
    report_resolution, report_weekdays = _axis_resolution(archive.report_date)
    temporal_resolution = ("daily" if "daily" in (reference_resolution, report_resolution) else "weekly")

    archive = _genuine_events(archive)
    # Denominator is every archived row (one row per (geo_value,
    # reference_date, report_date) triple, after de-duplication above) --
    # NOT the number of distinct (geo_value, reference_date) episodes. A row
    # counts as a genuine event when its value differs from the immediately
    # preceding row for the same episode; each episode's first row can never
    # be genuine (nothing precedes it) and is included in the denominator.
    genuine_rate = float(archive.genuine_event.mean())
    genuine_weekday_share = (archive[archive.genuine_event].report_date.dt.dayofweek
                             .value_counts(normalize=True).sort_index())
    genuine_weekday_share.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]
                                   for i in genuine_weekday_share.index]
    gaps = _revision_magnitude_gaps(archive)
    cadence, feature_lags = _reference_axis_feature_lags(archive)

    target_lag, completion_curve, completion_band = _target_lag_suggestion(
        archive, candidate_lags, maturity_floor_days, completion_threshold,
        completion_rate)
    lag_resolution = resolve_target_lag(target_lag, user_target_lag)
    window = DEFAULT_TRAINING_WINDOW_DAYS if training_window_days is None else int(training_window_days)

    notes = []
    if archive.geo_value.nunique() == 1:
        notes.append("single-location archive; location-level routing coordinates are degenerate "
                     "by construction, not a diagnosis failure.")
    if genuine_rate < 0.02:
        notes.append("genuine-event rate is very low; confirm this stream is not already reduced "
                     "to one release per reference date before AutoDelphiRF ingestion.")

    return DiagnosisReport(
        n_rows=int(len(archive)), n_locations=int(archive.geo_value.nunique()),
        reference_date_min=str(archive.reference_date.min().date()),
        reference_date_max=str(archive.reference_date.max().date()),
        # Both ends of the report axis: the retraining calendar is laid out
        # along it, starting a fixed offset after the first report date.
        report_date_min=str(archive.report_date.min().date()),
        report_date_max=str(archive.report_date.max().date()),
        reference_axis_weekdays=reference_weekdays, report_axis_weekdays=report_weekdays,
        reference_axis_resolution=reference_resolution, report_axis_resolution=report_resolution,
        temporal_resolution=temporal_resolution, genuine_event_rate=round(genuine_rate, 4),
        genuine_event_rate_denominator="rows (one per geo_value x reference_date x report_date, "
                                       "after de-duplication) -- not distinct (geo_value, "
                                       "reference_date) episodes",
        genuine_event_weekday_share=genuine_weekday_share.round(4).to_dict(),
        reference_axis_cadence_days=round(cadence, 2) if cadence == cadence else None,
        reference_axis_feature_lags=feature_lags,
        revision_magnitude_gap_days=gaps, recommended_target_lag=target_lag,
        target_lag_completion_curve=completion_curve,
        target_lag_completion_band=completion_band,
        user_target_lag=(int(user_target_lag) if user_target_lag is not None else None),
        resolved_target_lag=lag_resolution["resolved_target_lag"],
        target_lag_resolution=lag_resolution["target_lag_resolution"],
        target_lag_confirmation_prompt=lag_resolution["target_lag_confirmation_prompt"],
        training_window_days=window,
        training_window_is_fixed_default=(training_window_days is None), notes=tuple(notes))
