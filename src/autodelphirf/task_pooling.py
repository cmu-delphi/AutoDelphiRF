"""Shared utilities for location and reporting-lag pooling."""
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _mad_scale(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 0., 1.
    centre = float(np.median(x))
    mad = float(np.median(np.abs(x - centre)))
    return centre, mad if np.isfinite(mad) and mad > 0 else 1.


def _positive_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.median(x)) if len(x) else 1.


def _exact_median(values: np.ndarray) -> float:
    """``float(np.median(values))`` for a 1-D array, without its dispatch cost.

    ``np.median`` spends ~0.44 s per 20k calls regardless of array length --
    essentially all overhead (asanyarray, axis/keepdims handling, the
    ``_median_nancheck`` path) -- and ``_profile`` calls it four times per
    episode-lag suffix, 2.42M times and ~88 s on a chng fold. A direct sort
    with exact middle indexing measured 7-14x faster at lengths 5-60 and
    agrees to 0.0e+00 on finite input.

    NaN is NOT equivalent: ``np.sort`` orders NaN last, so the middle elements
    shift and ``np.median([1, nan, 3])`` would return 3.0 instead of nan. Any
    NaN therefore falls back to ``np.median`` so propagation is unchanged.
    ``+-inf`` needs no guard: it sorts into place and both paths agree.
    """
    n = values.size
    if n == 0 or np.isnan(values).any():
        return float(np.median(values))
    ordered = np.sort(values)
    half = n >> 1
    if n & 1:
        return float(ordered[half])
    return float(ordered[half - 1] + ordered[half]) / 2.


def _profile_statistics(delta, gaps, zero_count=None):
    """The five process-profile coordinates from precomputed increments.

    Split out of ``_profile`` so ``_summarize_episode`` can pass suffix SLICES
    of one episode-wide ``diff`` instead of re-differencing per suffix:
    ``diff(x[k:])`` is exactly ``diff(x)[k:]``. ``zero_count`` likewise lets
    the caller supply the number of zero increments in this suffix from a
    reverse cumulative sum; omitted, it is computed here as before.

    Four medians on four DIFFERENT arrays: gaps, delta, |delta| and
    |delta - med|. ``abs`` is not a permutation of ``delta`` and the MAD
    depends on ``med``, so no sort is shareable between them -- only ``med``
    itself is reused (it is also element 3).
    """
    if not len(delta):
        raise ValueError("a completed task realization needs the task-lag release and target-lag release")
    med = _exact_median(delta)
    share = (float(zero_count) / delta.size if zero_count is not None
             else float(np.mean(delta == 0)))
    return np.array([_exact_median(gaps), share, med,
                     _exact_median(np.abs(delta)), _exact_median(np.abs(delta - med))], float)


def _profile(lags, values, report_days):
    order = np.argsort(lags)
    values, report_days = np.asarray(values)[order], np.asarray(report_days)[order]
    return _profile_statistics(np.diff(values), np.diff(report_days))


def _increments(lags, values, start, target_lag):
    """Spline or linear future increments from the current lag through L."""
    from scipy.interpolate import CubicSpline
    order = np.argsort(lags)
    lags, values = np.asarray(lags, float)[order], np.asarray(values, float)[order]
    grid = np.arange(int(start), int(target_lag) + 1, dtype=float)
    if len(lags) == 2:
        fitted = np.interp(grid, lags, values)
    else:
        fitted = CubicSpline(lags, values, bc_type="natural")(grid)
    return np.diff(fitted)


@dataclass(frozen=True)
class EpisodeSummary:
    """Reusable completed-revision quantities indexed by genuinely observed lag."""
    profiles: dict[int, np.ndarray]
    trajectories: dict[int, np.ndarray]


def _summarize_episode(request):
    """Fit one spline through the target lag and derive every task-lag suffix."""
    key, lags, values, days, target_value, target_lag = request
    from scipy.interpolate import CubicSpline
    # Built with numpy rather than a per-episode DataFrame. Each step is the
    # exact equivalent of what the pandas version did:
    #   frame[isfinite(lag) & isfinite(value)]      -> boolean mask
    #   sort_values("lag", kind="mergesort")        -> argsort(kind="stable")
    #   drop_duplicates("lag", keep="last")         -> last index of each equal-lag run
    #   frame.loc[len(frame)] = [...] then re-sort  -> concatenate then stable re-sort
    # The pandas path cost a DataFrame construction, two sorts, a dedupe and a
    # row append per episode (the append alone was ~10 s on a chng fold).
    lag_array = np.asarray(lags, float)
    value_array = np.asarray(values, float)
    day_array = np.asarray(days, np.int64)
    keep = np.isfinite(lag_array) & np.isfinite(value_array)
    lag_array, value_array, day_array = lag_array[keep], value_array[keep], day_array[keep]
    order = np.argsort(lag_array, kind="stable")
    lag_array, value_array, day_array = lag_array[order], value_array[order], day_array[order]
    if len(lag_array):
        # keep="last" after a stable sort: retain the final row of each run of
        # equal lags, which is the latest in the original order.
        last_of_run = np.empty(len(lag_array), bool)
        last_of_run[-1] = True
        if len(lag_array) > 1:
            last_of_run[:-1] = lag_array[1:] != lag_array[:-1]
        lag_array = lag_array[last_of_run]
        value_array = value_array[last_of_run]
        day_array = day_array[last_of_run]
    endpoint_lag = int(target_lag)
    observed_lags = lag_array.astype(int)
    if len(lag_array) and endpoint_lag not in set(observed_lags.tolist()) and np.isfinite(target_value):
        # The target endpoint is the lag-L release carried by the target column.
        reference_day = int(day_array[0] - lag_array[0])
        lag_array = np.concatenate([lag_array, [float(endpoint_lag)]])
        value_array = np.concatenate([value_array, [float(target_value)]])
        day_array = np.concatenate([day_array, np.array([reference_day + endpoint_lag], np.int64)])
        order = np.argsort(lag_array, kind="stable")
        lag_array, value_array, day_array = lag_array[order], value_array[order], day_array[order]
    if len(lag_array) < 2 or endpoint_lag not in set(lag_array.astype(int).tolist()):
        return key, None
    lags = lag_array
    values = value_array
    days = day_array
    # Evaluate once through the complete prescribed target lag. Tasks still use only
    # genuinely observed start lags; values before the first release are never
    # routed, but retaining the full 0..L grid makes every cached episode share
    # the same indexing convention.
    grid = np.arange(0, int(target_lag) + 1, dtype=float)
    # NOTE: a dense-grid fast path (returning ``values`` directly when every
    # evaluation lag is already an observed knot) was implemented and REVERTED.
    # It is mathematically exact -- a spline interpolates at its own knots --
    # but scipy's evaluation there carries ~1.8e-15 of its own error, and
    # ``_positive_median``'s ``> 0`` filter turns that into a 5.9e-4 relative
    # shift in the curve scale: pairs whose curve distance was EXACTLY zero
    # (identical trajectories) become ~1e-16 and start entering the median.
    # That moved T_NN (1.757553126 -> 1.757471431), the queried-pair set and
    # the final pooling. Worth only 5% of prototype runtime, so the reference
    # spline call is kept.
    fitted = (np.interp(grid, lags, values) if len(lags) == 2 else
              CubicSpline(lags, values, bc_type="natural")(grid))
    increments = np.diff(fitted)
    # One diff per episode; every suffix is a slice of it. Zero-increment
    # counts per suffix come from a reverse cumulative sum, so the
    # ``mean(delta == 0)`` coordinate needs no per-suffix pass either.
    episode_delta = np.diff(values)
    episode_gaps = np.diff(days)
    zeros_at_or_after = np.zeros(episode_delta.size + 1, np.int64)
    if episode_delta.size:
        zeros_at_or_after[:-1] = np.cumsum((episode_delta == 0)[::-1])[::-1]
    profiles, trajectories = {}, {}
    for position, lag_value in enumerate(lags):
        lag = int(lag_value)
        if lag >= int(target_lag):
            continue
        suffix_values, suffix_days = values[position:], days[position:]
        if len(suffix_values) < 2:
            continue
        profiles[lag] = _profile_statistics(
            episode_delta[position:], episode_gaps[position:],
            zero_count=int(zeros_at_or_after[position]))
        offset = lag - int(grid[0])
        trajectories[lag] = increments[offset:int(target_lag) - int(grid[0])].copy()
    return key, EpisodeSummary(profiles, trajectories)


class EpisodeSummaryCache:
    """Replay-scoped, leakage-safe cache of immutable completed-revision summaries.

    ``n_jobs=1`` is deliberately the default. Values above one use threads to
    prepare previously unseen episodes; SciPy's spline evaluation releases the
    GIL, while a thread pool avoids copying large episode arrays to processes.
    """
    def __init__(self):
        self._summaries = {}
        self.hits = 0
        self.misses = 0

    def populate(self, requests, n_jobs=1):
        missing = []
        for request in requests:
            if request[0] in self._summaries:
                self.hits += 1
            else:
                self.misses += 1
                missing.append(request)
        if int(n_jobs) < 1:
            raise ValueError("geometry_n_jobs must be at least 1")
        if int(n_jobs) == 1 or len(missing) < 2:
            completed = map(_summarize_episode, missing)
        else:
            with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
                completed = list(executor.map(_summarize_episode, missing))
        for key, summary in completed:
            self._summaries[key] = summary

    def get(self, key):
        return self._summaries.get(key)


@dataclass
class TaskGeometry:
    tasks: list[tuple[str, int]]
    rows: list[np.ndarray]
    profiles: np.ndarray
    trajectories: list[np.ndarray]
    stages: np.ndarray
    component_scales: dict[str, float]
    distance: np.ndarray
    tree: np.ndarray | None
    gaps: np.ndarray
    candidates: list[int]
    selected_k: int
    labels: np.ndarray
    cache_stats: dict[str, int] = field(default_factory=dict)



def _merge_heights(tree: np.ndarray) -> np.ndarray:
    return np.asarray(tree, dtype=float)[:, 2][::-1]


def _relative_merge_gaps(heights: np.ndarray, k_gap: int, epsilon: float = 1e-12) -> np.ndarray:
    gaps = np.full(max(int(k_gap) - 1, 0), np.nan)
    for index in range(len(gaps)):
        k = index + 2
        if k - 1 >= len(heights):
            break
        upper, lower = heights[k - 2], heights[k - 1]
        gaps[index] = (upper - lower) / (lower + epsilon)
    return gaps


def _candidate_cuts(gaps: np.ndarray) -> tuple[list[int], float]:
    finite = np.asarray(gaps, dtype=float)
    valid = finite[np.isfinite(finite)]
    if len(valid) < 4:
        return [], np.nan
    q1, q3 = np.quantile(valid, .25), np.quantile(valid, .75)
    fence = float(q3 + 1.5 * (q3 - q1))
    return [i + 2 for i, value in enumerate(finite) if np.isfinite(value) and value > fence], fence

def build_task_geometry(training: pd.DataFrame, target_lag: int, *,
                        target_column="log_value_target_7dav",
                        value_column="log_value_7dav", baseline_column=None,
                        k_max_per_location=18, train_genuine_events_only=True,
                        episode_cache: EpisodeSummaryCache | None = None,
                        geometry_n_jobs: int = 1,
                        distance_backend: str = "scalar"):
    """Build all three finite task components from completed revision history only.

    ``distance_backend="prototypes"`` stops after constructing the observable
    task summaries used by RevRoute's sparse search. ``"scalar"`` remains as
    a compact diagnostic implementation of the full distance matrix.
    """
    target_lag = int(target_lag)
    required = {"geo_value", "reference_date", "report_date", "lag", value_column, target_column}
    if required.difference(training.columns):
        return None
    cases = training[(training.lag < target_lag)].copy()
    if train_genuine_events_only and "genuine_event" in cases:
        cases = cases[cases.genuine_event.fillna(False).astype(bool)]
    cases = cases.dropna(subset=[value_column, target_column]).drop_duplicates(
        ["geo_value", "reference_date", "lag"], keep="last").reset_index(drop=True)
    if cases.empty:
        return None
    baseline_column = baseline_column or ("Null" if "Null" in cases else value_column)
    source = training[training.lag <= target_lag].dropna(subset=[value_column]).copy()
    source["_day"] = pd.to_datetime(source.report_date).to_numpy("datetime64[D]").astype("int64")
    cache = episode_cache if episode_cache is not None else EpisodeSummaryCache()
    target_values = (cases.groupby(["geo_value", "reference_date"], observed=True,
                                   sort=False)[target_column].first())
    requests = []
    for key, frame in source.groupby(["geo_value", "reference_date"], observed=True, sort=False):
        episode_key = (str(key[0]), pd.Timestamp(key[1]), target_lag, value_column)
        target_value = target_values.get(key, np.nan)
        requests.append((episode_key, frame.lag.to_numpy(float),
                         frame[value_column].to_numpy(float), frame._day.to_numpy(np.int64),
                         float(target_value), target_lag))
    before_hits, before_misses = cache.hits, cache.misses
    cache.populate(requests, n_jobs=geometry_n_jobs)
    tasks, rowsets, raw_profiles, trajectories, stages = [], [], [], [], []
    for (geo, lag), frame in cases.groupby(["geo_value", "lag"], observed=True, sort=True):
        profiles, curves = [], []
        valid_rows = []
        # ``iterrows`` built a pandas Series per row -- 620k Series objects and
        # ~88 s on a chng fold -- purely to read reference_date and the index.
        # Pre-extracting both as arrays is the same computation with none of
        # that object churn. Index VALUES are preserved because they feed
        # ``cases.loc[valid_rows]`` below.
        reference_dates = frame.reference_date.to_numpy()
        row_index = frame.index.to_numpy()
        geo_text, lag_int = str(geo), int(lag)
        for position in range(len(row_index)):
            summary = cache.get((geo_text, pd.Timestamp(reference_dates[position]),
                                 target_lag, value_column))
            if summary is None or lag_int not in summary.profiles:
                continue
            profiles.append(summary.profiles[lag_int])
            curves.append(summary.trajectories[lag_int])
            valid_rows.append(row_index[position])
        if not valid_rows:
            continue
        tasks.append((str(geo), int(lag)))
        rowsets.append(np.asarray(valid_rows, int))
        raw_profiles.append(np.median(np.vstack(profiles), axis=0))
        trajectories.append(np.median(np.vstack(curves), axis=0))
        f = cases.loc[valid_rows]
        stages.append(float(np.median(np.abs(f[target_column] - f[baseline_column]))))
    if not tasks:
        return None
    raw_profiles = np.vstack(raw_profiles)
    centres, scales = zip(*[_mad_scale(raw_profiles[:, j]) for j in range(5)])
    profiles = (raw_profiles - np.asarray(centres)) / np.asarray(scales)
    n = len(tasks)
    if distance_backend == "prototypes":
        # RR-DelphiRF3 needs the three prototypes but must NOT build a dense
        # all-pairs matrix: its scales, threshold and pools are all defined on
        # a sparse queried subset (spec Sec. "Combined task dissimilarity and
        # sparse calibration"). Returning prototypes only is what makes the
        # sparse search worth doing; materialising n^2 first would defeat it.
        return TaskGeometry(tasks, rowsets, profiles, trajectories,
                            np.asarray(stages), {}, np.empty((0, 0)), None,
                            np.array([]), [], 0, np.zeros(n, int))
    stage = np.abs(np.asarray(stages)[:, None] - np.asarray(stages)[None, :])
    if distance_backend == "scalar":
        proc = np.linalg.norm(profiles[:, None, :] - profiles[None, :, :], axis=2)
        curve = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                h = min(len(trajectories[i]), len(trajectories[j]))
                curve[i, j] = curve[j, i] = np.sqrt(np.mean(
                    (trajectories[i][:h] - trajectories[j][:h]) ** 2))
    else:
        raise ValueError(f"unknown distance_backend: {distance_backend!r}")
    component_scales = {"proc": _positive_median(proc), "curve": _positive_median(curve),
                        "stage": _positive_median(stage)}
    distance = np.sqrt((proc/component_scales["proc"])**2 +
                       (curve/component_scales["curve"])**2 +
                       (stage/component_scales["stage"])**2)
    np.fill_diagonal(distance, 0.)
    if n == 1:
        tree, gaps, candidates, selected_k, labels = None, np.array([]), [], 1, np.ones(1, int)
    else:
        tree = linkage(squareform(distance, checks=False), method="average")
        kmax = min(int(k_max_per_location) * cases.geo_value.nunique(), n)
        gaps = _relative_merge_gaps(_merge_heights(tree), min(kmax, n - 1))
        candidates, _ = _candidate_cuts(gaps)
        selected_k = max(candidates) if candidates else 1
        labels = fcluster(tree, t=selected_k, criterion="maxclust")
    cache_stats = {"hits": cache.hits - before_hits, "misses": cache.misses - before_misses,
                   "entries": len(cache._summaries), "n_jobs": int(geometry_n_jobs)}
    return TaskGeometry(tasks, rowsets, profiles, trajectories, np.asarray(stages),
                        component_scales, distance, tree, gaps, candidates,
                        int(selected_k), np.asarray(labels, int) - 1, cache_stats)



__all__ = ["EpisodeSummaryCache", "TaskGeometry", "build_task_geometry"]
