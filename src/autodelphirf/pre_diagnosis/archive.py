"""Inspect raw revision archives and recommend preprocessing settings."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

EPS = 1e-8
DEFAULT_COMPLETION_THRESHOLD = 0.01
DEFAULT_GUIDANCE_THRESHOLD = 0.10
DEFAULT_COMPLETION_RATE = 0.90
DEFAULT_LATE_OBSERVATION_FLOOR_DAYS = 60
MAX_OBSERVED_LAG_DAYS = 2 * 365
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
    reference_axis_cadence_days: float
    reference_axis_feature_lags: list
    observed_report_gap_days: dict
    max_observed_lag_days: int
    target_lag_suggested_range: dict
    available_target_lags: list
    target_lag_completion_curve: dict
    target_lag_completion_band: dict
    user_target_lag: int | None
    resolved_target_lag: int | None
    target_lag_resolution: str
    target_lag_confirmation_prompt: str | None
    training_window_days: int
    training_window_is_fixed_default: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def resolve_target_lag(user_target_lag: int | None = None) -> dict:
    """Use only an explicit user choice; diagnosis never selects the target.

    A missing value remains unresolved so callers can show the diagnostic
    evidence and pause before preprocessing or training.
    """
    if user_target_lag is None:
        return {"resolved_target_lag": None, "target_lag_resolution": "awaiting_user_choice",
                "target_lag_confirmation_prompt": "Choose a target lag before continuing."}
    if not 0 <= int(user_target_lag) <= MAX_OBSERVED_LAG_DAYS:
        raise ValueError(
            f"target lag must be between 0 and {MAX_OBSERVED_LAG_DAYS} days")
    return {"resolved_target_lag": int(user_target_lag),
            "target_lag_resolution": "user_choice",
            "target_lag_confirmation_prompt": None}


def _axis_resolution(dates: pd.Series) -> tuple[str, dict]:
    """finest-observed-cadence rule: daily if the axis visits more than one
    weekday, weekly if it is confined to a single weekday."""
    unique_dates = pd.to_datetime(pd.Series(dates.unique()))
    weekdays = unique_dates.dt.dayofweek
    share = weekdays.value_counts(normalize=True).sort_index()
    share.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in share.index]
    resolution = "daily" if weekdays.nunique() > 1 else "weekly"
    return resolution, share.round(4).to_dict()


def _observed_report_gaps(archive: pd.DataFrame) -> dict:
    """Day-spacing between successive observed raw reports for an episode."""
    if archive.empty:
        return {"n": 0}
    # ``archive`` is sorted by location, reference date and reporting lag.
    # Detect episode boundaries directly instead of constructing thousands of
    # small pandas groups and datetime Series objects.
    geo = archive.geo_value.cat.codes.to_numpy(copy=False)
    reference = archive.reference_date.to_numpy(dtype="datetime64[ns]", copy=False)
    report = archive.report_date.to_numpy(dtype="datetime64[ns]", copy=False)
    same_episode = ((geo[1:] == geo[:-1]) & (reference[1:] == reference[:-1]))
    gaps = (report[1:] - report[:-1]).astype("timedelta64[D]").astype("int64")
    gap_days = pd.Series(gaps[same_episode], copy=False)
    if gap_days.empty:
        return {"n": 0}
    quantiles = gap_days.quantile([0.25, 0.5, 0.75]).round(2).to_dict()
    common = gap_days.value_counts().head(3).to_dict()
    return {"n": int(len(gap_days)), "q25": quantiles[0.25], "median": quantiles[0.5],
            "q75": quantiles[0.75], "most_common_gap_days": {int(k): int(v) for k, v in common.items()}}


def _target_lag_evidence(archive: pd.DataFrame, candidate_lags, late_observation_floor_days: int,
                         completion_threshold: float,
                         completion_rate: float) -> tuple[dict, dict, dict]:
    """Return a decision range and per-lag error evidence, never a chosen lag.

    The archive is swept once in lag order. ``current`` holds the most recent
    value observed for every eligible episode, so moving to the next candidate
    lag only applies newly available rows. This avoids the former full-frame
    filter, groupby and two merges for every candidate lag.
    """
    cutoff = archive.report_date.max()

    # The input is already sorted by episode then lag, so adjacent changes in
    # location/reference date define episode ids. This avoids constructing a
    # MultiIndex, which boxes every datetime and dominates large archives.
    geo = archive.geo_value.cat.codes.to_numpy(copy=False)
    reference = archive.reference_date.to_numpy(dtype="datetime64[ns]", copy=False)
    episode_start = np.r_[True, (geo[1:] != geo[:-1]) | (reference[1:] != reference[:-1])]
    episode_codes = np.cumsum(episode_start, dtype="int64") - 1
    start_positions = np.flatnonzero(episode_start)
    end_positions = np.r_[start_positions[1:] - 1, len(archive) - 1]
    n_episodes = len(start_positions)
    values = archive.value.to_numpy(dtype="float64", copy=False)
    lags = archive.lag.to_numpy(dtype="int64", copy=False)

    latest_by_episode = values[end_positions]
    reference_dates = reference[start_positions]
    cutoff_value = np.datetime64(cutoff.to_datetime64(), "ns")
    episode_age_days = ((cutoff_value - reference_dates)
                        .astype("timedelta64[D]").astype("int64"))
    eligible_episode = episode_age_days >= late_observation_floor_days
    eligible_ids = np.flatnonzero(eligible_episode)

    # Compact ids keep the state vector limited to eligible episodes. Rows
    # from recent, ineligible episodes never enter the sweep.
    compact_id = np.full(n_episodes, -1, dtype="int64")
    compact_id[eligible_ids] = np.arange(len(eligible_ids), dtype="int64")
    row_is_eligible = eligible_episode[episode_codes]
    observed_episode = compact_id[episode_codes[row_is_eligible]]
    observed_lag = lags[row_is_eligible]
    observed_value = values[row_is_eligible]
    order = np.argsort(observed_lag, kind="stable")
    observed_episode = observed_episode[order]
    observed_lag = observed_lag[order]
    observed_value = observed_value[order]

    current = np.full(len(eligible_ids), np.nan, dtype="float64")
    latest = latest_by_episode[eligible_ids]
    candidate_lags = sorted({int(lag) for lag in candidate_lags})
    curve = {}
    band = {}
    lower = None
    upper = None
    cursor = 0
    for lag in candidate_lags:
        end = int(np.searchsorted(observed_lag, lag, side="right"))
        if end > cursor:
            ids = observed_episode[cursor:end]
            chunk_values = observed_value[cursor:end]
            # A sparse caller may skip several observed lags. Retain the last
            # newly available row for each episode without a Python row loop.
            _, reverse_positions = np.unique(ids[::-1], return_index=True)
            last_positions = len(ids) - 1 - reverse_positions
            current[ids[last_positions]] = chunk_values[last_positions]
            cursor = end
        available = np.isfinite(current)
        if not np.any(available):
            continue
        relative_error = (np.abs(current[available] - latest[available]) /
                          np.maximum(np.abs(latest[available]), EPS))
        within_upper = ((relative_error <= DEFAULT_GUIDANCE_THRESHOLD) |
                        np.isclose(relative_error, DEFAULT_GUIDANCE_THRESHOLD,
                                   rtol=1e-6, atol=1e-8))
        within_lower = ((relative_error <= completion_threshold) |
                        np.isclose(relative_error, completion_threshold,
                                   rtol=1e-6, atol=1e-8))
        # Preserve the established diagnostic curve consumed by the website.
        # Selection uses the task-level completion rate below.
        quantiles = np.quantile(relative_error, [0.10, 0.50, 0.60, 0.75, 0.90, 0.95])
        curve[int(lag)] = round(float(quantiles[1]), 4)
        band[int(lag)] = {
            "q10": round(float(quantiles[0]), 4),
            "q60": round(float(quantiles[2]), 4),
            "q75": round(float(quantiles[3]), 4),
            "q90": round(float(quantiles[4]), 4),
            "q95": round(float(quantiles[5]), 4),
            "within_1pct": round(float(np.mean(within_lower)), 4),
            "within_10pct": round(float(np.mean(within_upper)), 4),
        }
        if lower is None and float(np.mean(within_upper)) >= completion_rate:
            lower = int(lag)
        if upper is None and float(np.mean(within_lower)) >= completion_rate:
            upper = int(lag)
    suggested_range = {
        "lower": lower, "upper": upper, "completion_rate": completion_rate,
        "lower_relative_error": DEFAULT_GUIDANCE_THRESHOLD,
        "upper_relative_error": completion_threshold,
    }
    return suggested_range, curve, band


def _reference_axis_feature_lags(archive: pd.DataFrame) -> tuple[float, list]:
    """Reference-axis cadence and the recommended revision-feature lags.

    Reference-axis lagged terms compare neighboring reference dates at a
    fixed target lag and must respect the reference-date cadence. The cadence
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
                         value: str = "value", candidate_lags=None,
                         completion_threshold: float = DEFAULT_COMPLETION_THRESHOLD,
                         completion_rate: float = DEFAULT_COMPLETION_RATE,
                         late_observation_floor_days: int = DEFAULT_LATE_OBSERVATION_FLOOR_DAYS,
                         training_window_days: int | None = None,
                         user_target_lag: int | None = None) -> DiagnosisReport:
    """Diagnose a raw revision archive and recommend RevRoute's preprocessing
    configuration. ``raw`` needs only the four semantic columns named by the
    keyword arguments.

    ``training_window_days`` is a fixed default (180 days,
    ``DEFAULT_TRAINING_WINDOW_DAYS``) unless the caller overrides it; it is
    never suggested from the archive itself (see module docstring).

    Target-lag diagnosis provides a range for human review. It never chooses a
    target lag. When ``user_target_lag`` is absent, the result remains unresolved.
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
    # Very late updates are outside the modelled revision horizon. Keeping a
    # single two-year bound here and at the R preparation boundary makes the
    # diagnostic evidence and training population agree while leaving the
    # uploaded source archive untouched.
    archive = archive[archive.lag <= MAX_OBSERVED_LAG_DAYS]
    if archive.empty:
        raise ValueError(
            f"raw archive has no usable observations with reporting lag between 0 and "
            f"{MAX_OBSERVED_LAG_DAYS} days")
    archive = archive.drop_duplicates(["geo_value", "reference_date", "report_date"], keep="last")
    archive = archive.sort_values(["geo_value", "reference_date", "lag"]).reset_index(drop=True)
    # Diagnose every lag that is present in this archive. The web interface
    # uses these same lags as its valid target-lag choices, so no sparse
    # checkpoint grid or interpolated evidence is involved.
    if candidate_lags is None:
        candidate_lags = sorted(int(lag) for lag in archive.lag.unique() if int(lag) > 0)
    else:
        candidate_lags = [int(lag) for lag in candidate_lags
                          if 0 < int(lag) <= MAX_OBSERVED_LAG_DAYS]

    reference_resolution, reference_weekdays = _axis_resolution(archive.reference_date)
    report_resolution, report_weekdays = _axis_resolution(archive.report_date)
    temporal_resolution = ("daily" if "daily" in (reference_resolution, report_resolution) else "weekly")

    report_gaps = _observed_report_gaps(archive)
    cadence, feature_lags = _reference_axis_feature_lags(archive)

    target_lag_range, completion_curve, completion_band = _target_lag_evidence(
        archive, candidate_lags, late_observation_floor_days, completion_threshold,
        completion_rate)
    lag_resolution = resolve_target_lag(user_target_lag)
    window = DEFAULT_TRAINING_WINDOW_DAYS if training_window_days is None else int(training_window_days)

    notes = []
    if archive.geo_value.nunique() == 1:
        notes.append("single-location archive; location-level routing coordinates are degenerate "
                     "by construction, not a diagnosis failure.")
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
        temporal_resolution=temporal_resolution,
        reference_axis_cadence_days=round(cadence, 2) if cadence == cadence else None,
        reference_axis_feature_lags=feature_lags,
        observed_report_gap_days=report_gaps,
        max_observed_lag_days=MAX_OBSERVED_LAG_DAYS,
        target_lag_suggested_range=target_lag_range,
        available_target_lags=sorted(int(lag) for lag in archive.lag.unique() if int(lag) > 0),
        target_lag_completion_curve=completion_curve,
        target_lag_completion_band=completion_band,
        user_target_lag=(int(user_target_lag) if user_target_lag is not None else None),
        resolved_target_lag=lag_resolution["resolved_target_lag"],
        target_lag_resolution=lag_resolution["target_lag_resolution"],
        target_lag_confirmation_prompt=lag_resolution["target_lag_confirmation_prompt"],
        training_window_days=window,
        training_window_is_fixed_default=(training_window_days is None), notes=tuple(notes))
