"""Prospective routing weights for completed historical observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, PchipInterpolator

EPS = 1e-8

# Prospective revision-progress feature V.
# Delphi-RF's released seven-day log revision difference at lag 7.
FEATURE = "log_delta_value_7dav_lag7"


def robust_mad_scale(values: np.ndarray) -> float:
    """Return 1.4826 times MAD(X), floored at EPS."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    return float(max(1.4826 * np.median(np.abs(values - np.median(values))), EPS))


def _robust_median_positive(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values) + EPS) if len(values) else 1.0


def _remaining(rows: pd.DataFrame) -> pd.Series:
    return rows["log_value_target_7dav"].astype(float) - rows["log_value_7dav"].astype(float)


def _penalized_bspline(x: np.ndarray, y: np.ndarray, grid: np.ndarray, *, n_basis: int,
                       penalty: float) -> np.ndarray:
    """Fixed cubic B-spline basis with a second-difference penalty.

    This is a representation-only smoother: a
    continuous latent representation of a *completed* revision trajectory.
    Interpolated values are never used as donor observations or responses.
    """
    if n_basis < 2 or penalty < 0:
        raise ValueError("spline basis size must be at least two and penalty non-negative")
    degree = min(3, n_basis - 1)
    internal_count = n_basis - degree - 1
    internal = (np.linspace(float(grid[0]), float(grid[-1]), internal_count + 2)[1:-1]
                if internal_count else np.array([]))
    knots = np.r_[np.repeat(grid[0], degree + 1), internal, np.repeat(grid[-1], degree + 1)]
    eye = np.eye(n_basis)
    basis = np.column_stack([BSpline(knots, eye[index], degree, extrapolate=True)(x)
                             for index in range(n_basis)])
    grid_basis = np.column_stack([BSpline(knots, eye[index], degree, extrapolate=True)(grid)
                                  for index in range(n_basis)])
    difference = np.diff(eye, n=2, axis=0) if n_basis >= 3 else np.diff(eye, axis=0)
    system = basis.T @ basis + penalty * (difference.T @ difference) + EPS * eye
    coefficients = np.linalg.solve(system, basis.T @ y)
    return grid_basis @ coefficients


def _episode_curve(rows: pd.DataFrame, initial_lag: int, target_lag: int, *,
                   spline_method: str = "pchip", spline_basis: int = 6,
                   spline_penalty: float = 1e-3) -> np.ndarray | None:
    """Return a spline-derived remaining-revision curve for one real episode.

    Returned values are a routing representation only; never appended to the
    donor data frame or treated as observations.
    """
    episode = rows.sort_values("lag").drop_duplicates("lag", keep="last")
    episode = episode[np.isfinite(episode.log_value_7dav) & np.isfinite(episode.log_value_target_7dav)]
    if episode.empty:
        return None
    if initial_lag >= target_lag or not (episode.lag == initial_lag).any():
        return None
    genuine = episode[episode.genuine_event.astype(bool)] if "genuine_event" in episode else episode
    anchors = pd.concat([genuine, episode.iloc[[0]]], ignore_index=True).drop_duplicates("lag", keep="last")
    target_row = anchors.iloc[[-1]].copy()
    target_row.loc[:, "lag"] = target_lag
    target_row.loc[:, "log_value_7dav"] = float(episode.log_value_target_7dav.iloc[0])
    anchors = pd.concat([anchors, target_row], ignore_index=True).sort_values("lag").drop_duplicates("lag", keep="last")
    x = anchors.lag.to_numpy(float)
    y = anchors.log_value_7dav.to_numpy(float)
    if len(x) < 2 or np.any(~np.isfinite(y)):
        return None
    grid = np.arange(initial_lag, target_lag + 1, dtype=float)
    try:
        if spline_method == "pchip":
            fitted = PchipInterpolator(x, y)(grid)
        elif spline_method == "penalized_bspline":
            fitted = _penalized_bspline(x, y, grid, n_basis=spline_basis, penalty=spline_penalty)
        else:
            raise ValueError(f"unknown spline method: {spline_method}")
    except np.linalg.LinAlgError:
        fitted = np.interp(grid, x, y)
    return float(episode.log_value_target_7dav.iloc[0]) - np.asarray(fitted, dtype=float)


@dataclass(frozen=True)
class CurveRouting:
    """Cutoff-valid real donors and curve representations for one outer fold."""
    donors: pd.DataFrame
    curves: dict[str, np.ndarray]
    positions: dict[str, np.ndarray]
    curve_episode_counts: dict[str, int]
    initial_lag: int
    target_lag: int
    curve_scale: float
    stage_scale: float
    gamma: float
    location_labels: tuple[str, ...]
    donor_location_codes: np.ndarray
    donor_template_positions: np.ndarray
    response_sort_order: np.ndarray
    curve_scale_supported: bool
    # Fold-constant values hoisted out of the per-case hot path. Recomputing
    # these per case (a 601k-row pandas->numpy conversion and four 601k-element
    # medians on chng) dominated RED's runtime while depending only on the
    # fold. Values are identical to the previous per-case computation.
    donor_state: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    state_scale: float = 1.0
    responses: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Memoised per-donor coordinates: curve depends only on the target
    # location, stage only on (location, lag). Mutating these dicts is allowed
    # on a frozen dataclass because the binding itself never changes.
    curve_cache: dict = field(default_factory=dict)
    stage_cache: dict = field(default_factory=dict)
    geometry_cache: dict = field(default_factory=dict)
    sorted_responses: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def build_curve_routing(rows: pd.DataFrame, target_lag: int, min_curve_episodes: int = 3,
                        gamma: float = 1.0, *, spline_method: str = "pchip",
                        spline_basis: int = 6, spline_penalty: float = 1e-3,
                        initial_lag: int | None = None) -> CurveRouting:
    """Build cutoff-valid curve templates and retain only real donor rows.

    ``rows`` must already be restricted to the completed rolling history
    (see ``rolling_completed_training``). Donor responses are the observed
    archived state revisions, never values evaluated from the spline grid.
    """
    if min_curve_episodes < 1:
        raise ValueError("min_curve_episodes must be positive")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if spline_method not in {"pchip", "penalized_bspline"}:
        raise ValueError(f"unknown spline method: {spline_method}")
    if spline_basis < 2 or spline_penalty < 0:
        raise ValueError("spline basis must be at least two and penalty non-negative")
    source = rows.copy()
    required = ["geo_value", "reference_date", "lag", "log_value_7dav", "log_value_target_7dav"]
    source = source.dropna(subset=required)
    source = source[(source.lag <= target_lag) & (source.lag >= 0)].copy()
    if initial_lag is None:
        initial_lag = int(source.lag.min()) if not source.empty else 0
    initial_lag = int(initial_lag)
    if initial_lag < 0 or initial_lag >= target_lag:
        raise ValueError("initial_lag must satisfy 0 <= initial_lag < target_lag")
    source = source[source.lag >= initial_lag].copy()
    source["real_remaining_revision"] = _remaining(source)
    source = source[np.isfinite(source.real_remaining_revision)]
    if source.empty:
        empty = np.array([], dtype=float)
        return CurveRouting(source, {}, {}, {}, initial_lag, target_lag, 1., 1., gamma,
                            tuple(), np.array([], dtype=int), np.array([]), np.array([], dtype=int), False,
                            empty, 1.0, empty, {}, {}, {}, empty)
    episode_curves: dict[str, list[np.ndarray]] = {}
    for (location, _), episode in source.groupby(["geo_value", "reference_date"], observed=True):
        curve = _episode_curve(episode, initial_lag, target_lag, spline_method=spline_method,
                               spline_basis=spline_basis, spline_penalty=spline_penalty)
        if curve is not None:
            episode_curves.setdefault(str(location), []).append(curve)
    curves: dict[str, np.ndarray] = {}
    positions: dict[str, np.ndarray] = {}
    counts = {location: len(parts) for location, parts in episode_curves.items()}
    for location, parts in episode_curves.items():
        if len(parts) < min_curve_episodes:
            continue
        median_curve = np.median(np.asarray(parts), axis=0)
        initials = source[source.geo_value.astype(str).eq(location) & source.lag.eq(initial_lag)].real_remaining_revision.abs().to_numpy(float)
        scale = _robust_median_positive(np.asarray(initials))
        normal_curve = median_curve / scale
        curves[location] = normal_curve
        positions[location] = np.abs(normal_curve)
    distances = []
    for left, right in combinations(curves, 2):
        valid = np.isfinite(curves[left]) & np.isfinite(curves[right])
        if valid.any():
            distances.append(float(np.sqrt(np.mean((curves[left][valid] - curves[right][valid]) ** 2))))
    curve_scale = _robust_median_positive(np.asarray(distances))
    curve_scale_supported = bool(distances)
    stage_values = np.concatenate([x[np.isfinite(x)] for x in positions.values()]) if positions else np.array([])
    if len(stage_values) > 4000:
        stage_values = np.sort(stage_values)[np.linspace(0, len(stage_values)-1, 4000, dtype=int)]
    stage_pairs = np.fromiter((abs(float(a)-float(b)) for a, b in combinations(stage_values, 2)), dtype=float)
    stage_scale = _robust_median_positive(stage_pairs)
    donor_columns = [x for x in source.columns if x != "real_remaining_revision"]
    donors = source.drop_duplicates(["geo_value", "reference_date", "lag"], keep="last")[donor_columns + ["real_remaining_revision"]].reset_index(drop=True)
    location_labels = tuple(sorted(donors.geo_value.astype(str).unique()))
    location_code = {location: code for code, location in enumerate(location_labels)}
    donor_location_codes = donors.geo_value.astype(str).map(location_code).to_numpy(int)
    donor_template_positions = np.full(len(donors), np.nan)
    donor_lags = donors.lag.to_numpy(int)
    for location, template in positions.items():
        mask = donors.geo_value.astype(str).to_numpy() == location
        indices = np.clip(donor_lags[mask] - initial_lag, 0, len(template)-1)
        donor_template_positions[mask] = template[indices]
    response_sort_order = np.argsort(donors.real_remaining_revision.to_numpy(float), kind="stable")
    donor_state = (donors[FEATURE].to_numpy(float) if FEATURE in donors.columns
                   else np.full(len(donors), np.nan))
    state_scale = robust_mad_scale(donor_state)
    responses = donors.real_remaining_revision.to_numpy(float)
    return CurveRouting(donors, curves, positions, counts, initial_lag, target_lag, curve_scale, stage_scale, gamma,
                        location_labels, donor_location_codes, donor_template_positions, response_sort_order,
                        curve_scale_supported, donor_state, state_scale, responses, {}, {}, {},
                        responses[response_sort_order] if len(response_sort_order) else responses)


def _position_at(position: np.ndarray, lag: int, initial_lag: int) -> float:
    index = int(np.clip(lag - initial_lag, 0, len(position) - 1))
    value = position[index]
    return float(value) if np.isfinite(value) else np.nan


def _curve_by_donor(route: CurveRouting, location: str) -> np.ndarray:
    cached = route.curve_cache.get(str(location))
    if cached is not None:
        return cached
    values = np.full(len(route.location_labels), np.nan)
    target = route.curves.get(str(location))
    if target is None:
        result = values[route.donor_location_codes]
        route.curve_cache[str(location)] = result
        return result
    for code, donor_location in enumerate(route.location_labels):
        donor_curve = route.curves.get(donor_location)
        if donor_curve is None:
            continue
        valid = np.isfinite(target) & np.isfinite(donor_curve)
        if valid.any():
            values[code] = np.sqrt(np.mean((target[valid] - donor_curve[valid]) ** 2))
    result = values[route.donor_location_codes]
    route.curve_cache[str(location)] = result
    return result


def _geometry_by_donor(route: CurveRouting, location: str, lag: int):
    """Fold-constant part of D^2 for one (location, lag), plus its audit means.

    ``curve`` depends only on the location and ``stage`` only on (location,
    lag), so the normalised squared terms, their finite mask, and the two
    reported mean distances are identical for every case sharing that key.
    Computing them per case meant ~8 extra passes over the full donor vector
    (601k rows on chng) for values that never changed.
    """
    key = (str(location), int(lag))
    cached = route.geometry_cache.get(key)
    if cached is not None:
        return cached
    curve = _curve_by_donor(route, location)
    stage = _stage_by_donor(route, location, lag)
    curve_term = ((curve / route.curve_scale) ** 2 if route.curve_scale_supported
                  else np.zeros(len(route.donors), dtype=float))
    stage_term = (stage / route.stage_scale) ** 2
    base_term = curve_term + stage_term
    finite_geometry = np.isfinite(curve) & np.isfinite(stage)
    entry = (base_term, finite_geometry,
             float(np.nanmean(curve)) if curve.size else np.nan,
             float(np.nanmean(stage)) if stage.size else np.nan,
             curve, stage)
    route.geometry_cache[key] = entry
    return entry


def _stage_by_donor(route: CurveRouting, location: str, lag: int) -> np.ndarray:
    key = (str(location), int(lag))
    cached = route.stage_cache.get(key)
    if cached is not None:
        return cached
    position = route.positions.get(str(location))
    if position is None:
        result = np.full(len(route.donors), np.nan)
    else:
        index = int(np.clip(lag - route.initial_lag, 0, len(position) - 1))
        result = np.abs(float(position[index]) - route.donor_template_positions)
    route.stage_cache[key] = result
    return result


def rolling_completed_training(prepared: pd.DataFrame, cutoff: pd.Timestamp, training_days: int) -> pd.DataFrame:
    """Select completed historical reference dates available at an origin.

    ``training_days`` (W) is a trailing window on each row's target
    availability date, not its reference date. Report-state features must
    also have been released before the cutoff.
    """
    window_start = cutoff - pd.Timedelta(days=training_days)
    eligible = (prepared.target_date > window_start) & (prepared.target_date <= cutoff)
    if "report_date" in prepared:
        eligible &= prepared.report_date < cutoff
    return prepared[eligible].copy()


def attach_revision_progress(cases: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    """Attach V (the prospective revision-progress feature) by report state.

    Missing values are valid cold/fallback cases, not an excuse to use a
    later report; the row-level routing audit reports their availability.
    """
    keys = ["geo_value", "reference_date", "report_date", "lag"]
    state = prepared[keys + [FEATURE]].drop_duplicates(keys, keep="last")
    out = cases.drop(columns=[FEATURE], errors="ignore").merge(state, on=keys, how="left", validate="one_to_one")
    return out


def final_v7_weights(route: CurveRouting, location: str, lag: int, revision_state: float) -> tuple[np.ndarray, dict]:
    """The primary v8 routing distance: D^2 = curve^2 + revision^2 + stage^2.

    This is the exact function the v8 pipeline calls (frozen v7 routing
    geometry). It implements case-wide disabling of the
    revision-state coordinate when V_q is missing; per-donor ineligibility
    (never zero distance) when V_j is missing; the cold-start fallback
    D^2_cold = revision^2 + (|l-l_j|/(L-l_min))^2; the degenerate-support
    fallback when a non-cold target still has no comparable donor template;
    and log-weight stabilization when direct exponentiation would give zero
    total mass.
    """
    donors = route.donors
    donor_state = route.donor_state
    state_active = bool(np.isfinite(revision_state))
    donor_state_observed = np.isfinite(donor_state)
    if state_active:
        revision_distance = np.abs(float(revision_state) - donor_state) / route.state_scale
        revision_distance[~donor_state_observed] = np.inf
    else:
        revision_distance = np.zeros(len(donors), dtype=float)
    cold = str(location) not in route.curves
    fallback = "none"
    if cold:
        lag_distance = np.abs(donors.lag.to_numpy(float) - float(lag)) / max(route.target_lag - route.initial_lag, 1)
        distance_sq = revision_distance ** 2 + lag_distance ** 2
        valid, curve, stage = np.isfinite(distance_sq), None, None
        curve_mean = stage_mean = np.nan
        fallback = ("cold-start revision-state-and-lag" if len(donors)
                    else "insufficient donor history")
        if len(donors) and not valid.any():
            distance_sq = lag_distance ** 2
            valid = np.isfinite(distance_sq)
            fallback = "cold-start lag-only; donor revision state unavailable"
        if len(donors) and not valid.any():
            distance_sq, valid = lag_distance ** 2, np.isfinite(lag_distance)
            fallback = "cold-start lag-only; donor revision state unavailable"
    else:
        base_term, finite_geometry, curve_mean, stage_mean, curve, stage = _geometry_by_donor(
            route, location, lag)
        distance_sq = base_term + revision_distance ** 2
        valid = finite_geometry & np.isfinite(revision_distance)
        if not valid.any() and len(donors):
            lag_distance = (np.abs(donors.lag.to_numpy(float) - float(lag)) /
                            max(route.target_lag - route.initial_lag, 1))
            if state_active and donor_state_observed.any():
                distance_sq = revision_distance ** 2 + lag_distance ** 2
                fallback = "degenerate-support revision-state-and-lag"
            else:
                distance_sq = lag_distance ** 2
                fallback = "degenerate-support lag-only"
            valid = np.isfinite(distance_sq)
    weights = np.zeros(len(donors), dtype=float)
    finite_distance = valid & np.isfinite(distance_sq)
    if finite_distance.any():
        exponent = -route.gamma * distance_sq[finite_distance]
        weights[finite_distance] = np.exp(exponent)
        if not (weights > 0).any():
            weights[finite_distance] = np.exp(exponent - np.max(exponent))
            fallback = "log-weight stabilization"
    response = route.responses
    keep = np.isfinite(response) & (weights > 0)
    normalised = np.zeros(len(donors), dtype=float)
    if keep.any():
        normalised[keep] = weights[keep] / weights[keep].sum()
    return weights, {"cold_start": cold, "routing_fallback": fallback,
                     "revision_state_active": state_active,
                     "donor_revision_state_missing_n": int((~donor_state_observed).sum()),
                     "effective_donor_n": int(keep.sum()),
                     "raw_donor_n": int(len(donors)),
                     "curve_coordinate_active": route.curve_scale_supported,
                     "revision_state_scale": route.state_scale,
                     "point_ess": float(1 / np.square(normalised).sum()) if keep.any() else 0.0,
                     "max_normalized_donor_weight": float(normalised.max()) if keep.any() else np.nan,
                     "mean_curve_distance": np.nan if curve is None else curve_mean,
                     "mean_stage_distance": np.nan if stage is None else stage_mean}


# Preferred, concern-neutral alias. ``final_v7_weights`` is kept as the
# primary name because it is still referenced by history/tests; both names
# resolve to the identical function object.
compute_weights = final_v7_weights
