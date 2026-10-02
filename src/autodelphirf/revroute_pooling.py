"""Adaptive pooling for location and reporting-lag combinations."""
from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform

from .task_pooling import _positive_median, build_task_geometry

# Fixed structural tolerance: a direction closes after this many CONSECUTIVE
# unsupported observed release steps. "The value two is a fixed structural
# tolerance, not a tuned lag window."
CLOSE_AFTER_UNSUPPORTED = 2


def adjacent_observed_lags(lags) -> dict[int, tuple[int | None, int | None]]:
    """``(lag-, lag+)`` in OBSERVED release-lag order, not day adjacency."""
    ordered = sorted({int(x) for x in lags})
    out = {}
    for position, lag in enumerate(ordered):
        out[lag] = (ordered[position - 1] if position > 0 else None,
                    ordered[position + 1] if position + 1 < len(ordered) else None)
    return out


class SparseTaskGeometry:
    """Task prototypes with distances evaluated and cached on demand.

    Every queried unordered pair is cached, so a reverse anchor never
    recomputes a distance. Lag blocks are evaluated with vectorized kernels;
    within a block the values are exactly the scalar definitions.
    """

    def __init__(self, tasks, profiles, trajectories, stages):
        self.tasks = list(tasks)
        self.profiles = np.asarray(profiles, float)
        self.trajectories = [np.asarray(x, float) for x in trajectories]
        self.stages = np.asarray(stages, float)
        self.index = {task: i for i, task in enumerate(self.tasks)}
        self.lags = np.array([lag for _, lag in self.tasks], int)
        self.by_lag: dict[int, np.ndarray] = {
            int(lag): np.flatnonzero(self.lags == lag) for lag in np.unique(self.lags)}
        # Position of each task WITHIN its lag group. Without this, every
        # distance evaluation did a linear scan of the group's index array.
        self.within_lag: dict[int, int] = {}
        for lag, index in self.by_lag.items():
            for offset, position in enumerate(index):
                self.within_lag[int(position)] = offset
        self.by_location: dict[str, dict[int, int]] = {}
        for position, (geo, lag) in enumerate(self.tasks):
            self.by_location.setdefault(geo, {})[int(lag)] = position
        self.adjacent = adjacent_observed_lags(self.lags)
        # Sorted observed stages per location, precomputed once. The searches
        # previously re-sorted these per (anchor, location): 3,356 anchors x 56
        # locations = ~188k redundant sorts of a ~60-element list on chng.
        self.stages_of: dict[str, list[int]] = {
            geo: sorted(stage_map) for geo, stage_map in self.by_location.items()}
        self.scales = {"proc": 1., "curve": 1., "stage": 1.}
        self._cache: dict[tuple[int, int], float] = {}
        self._blocks: dict[tuple[int, int], np.ndarray] = {}
        self.queried: set[tuple[int, int]] = set()

    # -- raw component blocks, vectorized per lag pair ---------------------
    def _raw_blocks(self, left_lag: int, right_lag: int):
        key = (min(left_lag, right_lag), max(left_lag, right_lag))
        if key not in self._blocks:
            a_index, b_index = self.by_lag[key[0]], self.by_lag[key[1]]
            proc = cdist(self.profiles[a_index], self.profiles[b_index], metric="euclidean")
            left = np.vstack([self.trajectories[i] for i in a_index])
            right = np.vstack([self.trajectories[i] for i in b_index])
            common_length = min(left.shape[1], right.shape[1])
            curve = (cdist(left[:, :common_length], right[:, :common_length], metric="euclidean")
                     / np.sqrt(common_length)) if common_length else np.zeros((len(a_index), len(b_index)))
            stage = np.abs(self.stages[a_index][:, None] - self.stages[b_index][None, :])
            self._blocks[key] = np.stack([proc, curve, stage])
        return key, self._blocks[key]

    def raw_components(self, i: int, j: int) -> np.ndarray:
        key, block = self._raw_blocks(int(self.lags[i]), int(self.lags[j]))
        if int(self.lags[i]) == key[0]:
            p, q = self.within_lag[i], self.within_lag[j]
        else:
            p, q = self.within_lag[j], self.within_lag[i]
        return block[:, p, q]

    def distance(self, i: int, j: int) -> float:
        """``D_task(u, v)``, cached, counting the pair as queried."""
        if i == j:
            return 0.
        key = (min(i, j), max(i, j))
        self.queried.add(key)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        raw = self.raw_components(i, j)
        scaled = raw / np.array([self.scales["proc"], self.scales["curve"],
                                 self.scales["stage"]], float)
        value = float(np.sqrt(np.sum(scaled ** 2)))
        self._cache[key] = value
        return value

    def cached(self, i: int, j: int) -> float | None:
        if i == j:
            return 0.
        return self._cache.get((min(i, j), max(i, j)))


def structural_candidates(geometry: SparseTaskGeometry, position: int) -> list[int]:
    """``N_0(u)``: same lag in other locations, plus adjacent lags same location."""
    geo, lag = geometry.tasks[position]
    same_lag = [int(x) for x in geometry.by_lag.get(int(lag), ()) if int(x) != position]
    out = list(same_lag)
    previous, following = geometry.adjacent.get(int(lag), (None, None))
    for neighbour in (previous, following):
        if neighbour is None:
            continue
        candidate = geometry.by_location.get(geo, {}).get(int(neighbour))
        if candidate is not None:
            out.append(int(candidate))
    return out


def calibrate_sparse_scales(geometry: SparseTaskGeometry) -> tuple[dict, np.ndarray, list]:
    """Local scales on ``E_0`` and the nearest-neighbour values ``m_u``.

    Scales are the median POSITIVE raw component distance over ``E_0`` only --
    "it puts all three components on the scale of plausible local analogues,
    which is the same regime used to define T_NN, while avoiding an otherwise
    quadratic all-pairs preprocessing step".
    """
    edges, candidates = set(), {}
    for position in range(len(geometry.tasks)):
        neighbours = structural_candidates(geometry, position)
        if neighbours:
            candidates[position] = neighbours
            for other in neighbours:
                edges.add((min(position, other), max(position, other)))
    if not edges:
        return {"proc": 1., "curve": 1., "stage": 1.}, np.array([]), []
    raw = np.vstack([geometry.raw_components(i, j) for i, j in sorted(edges)])
    scales = {name: _positive_median(raw[:, column])
              for column, name in enumerate(("proc", "curve", "stage"))}
    geometry.scales = scales
    # m_u = min over N_0(u); the pairs are already blocked, so this is cheap.
    nearest, members = [], []
    for position, neighbours in candidates.items():
        nearest.append(min(geometry.distance(position, other) for other in neighbours))
        members.append(position)
    return scales, np.asarray(nearest, float), members


def nearest_neighbor_threshold(nearest: np.ndarray) -> float:
    """``T_NN = Q3 + 1.5 IQR`` of the sparse nearest-neighbour distribution."""
    nearest = np.asarray(nearest, float)
    nearest = nearest[np.isfinite(nearest)]
    if not len(nearest):
        return 0.
    q1, q3 = np.quantile(nearest, [.25, .75])
    value = float(q3 + 1.5 * (q3 - q1))
    return value if np.isfinite(value) else 0.


def same_location_search(geometry: SparseTaskGeometry, position: int, threshold: float) -> None:
    """Search each same-location direction until two CONSECUTIVE misses."""
    geo, lag = geometry.tasks[position]
    stages = geometry.stages_of.get(geo, [])
    if int(lag) not in geometry.by_location.get(geo, {}):
        return
    origin = stages.index(int(lag))
    for step in (1, -1):
        misses = 0
        cursor = origin + step
        while 0 <= cursor < len(stages):
            other = geometry.by_location[geo][stages[cursor]]
            misses = 0 if geometry.distance(position, other) <= threshold else misses + 1
            if misses >= CLOSE_AFTER_UNSUPPORTED:
                break
            cursor += step


def cross_location_valley_search(geometry: SparseTaskGeometry, position: int,
                                 threshold: float) -> None:
    """Directional single-valley search toward each other location.

    A direction whose first adjacent comparison does not improve on the centre
    is closed immediately. A descending direction is followed while it may
    still reach a lower valley, and closes only once two consecutive queried
    distances exceed ``T_NN`` and neither improved the running minimum.
    """
    geo, lag = geometry.tasks[position]
    for other_geo, stage_map in geometry.by_location.items():
        if other_geo == geo:
            continue
        stages = geometry.stages_of[other_geo]
        if not stages:
            continue
        centre = stage_map.get(int(lag))
        if centre is not None:
            centre_distance = geometry.distance(position, centre)
            anchor = stages.index(int(lag))
            for step in (1, -1):
                cursor = anchor + step
                if not (0 <= cursor < len(stages)):
                    continue
                first = geometry.distance(position, stage_map[stages[cursor]])
                # Non-descending: closed immediately by the single-valley rule.
                if not first < centre_distance:
                    continue
                running = min(centre_distance, first)
                previous_unsupported = first > threshold
                cursor += step
                while 0 <= cursor < len(stages):
                    value = geometry.distance(position, stage_map[stages[cursor]])
                    improved = value < running
                    running = min(running, value)
                    unsupported = value > threshold
                    if unsupported and previous_unsupported and not improved:
                        break
                    previous_unsupported = unsupported
                    cursor += step
        else:
            # No centre: start at the nearest available comparison each side and
            # apply the same two-consecutive-unsupported rule once a running
            # minimum exists.
            below = [x for x in stages if x < int(lag)]
            above = [x for x in stages if x > int(lag)]
            for sequence in ([below[-1::-1]] if below else []) + ([above] if above else []):
                running, previous_unsupported = np.inf, False
                for stage in sequence:
                    value = geometry.distance(position, stage_map[stage])
                    improved = value < running
                    running = min(running, value)
                    unsupported = value > threshold
                    if unsupported and previous_unsupported and not improved:
                        break
                    previous_unsupported = unsupported


def coherence_preserving_pools(geometry: SparseTaskGeometry, threshold: float) -> np.ndarray:
    """Thresholded COMPLETE-linkage agglomeration on the queried geometry.

    The spec prescribes GREEDY agglomeration: "Repeatedly merge the pair (A,B)
    having the smallest finite D_CL^Q(A,B) whenever D_CL^Q(A,B) <= T_NN. Stop
    when no admissible merge remains. Ties are broken deterministically by the
    lexicographic order of the sorted task identifiers." That is implemented
    directly here.

    It is NOT interchangeable with building a complete-linkage tree and
    cutting it at T_NN. Merge heights are monotone, so the cut does select a
    prefix of the same merge sequence -- but complete linkage is not
    tie-order independent, and on this geometry ties dominate: on nhsn28
    fold 19, 30,263 of 38,919 admissible pairs are exactly zero (identical
    prototypes) and only 1,553 of the values are distinct. Merging a
    different zero-distance pair reshapes every later pool diameter, so the
    same rule reaches a different partition: 399 admissible merges (K=112)
    under the spec's ordering against 379 (K=132) under scipy's internal one,
    with the first 276 merges identical and the divergence starting at a
    0 vs 1e-9 tie. The scipy answer also moved with the finite sentinel used
    for unqueried pairs (K=132/134/136 at 2x/10x/1000x T_NN), so it was not a
    well-defined target. Both partitions honour max D_task <= T_NN, so the
    diameter guarantee does NOT certify equivalence.

    The tie rule is therefore load-bearing. ``np.argmin`` over the flattened
    matrix takes the lowest (a,b) index pair, and ``geometry.tasks`` is sorted
    by task identifier, so index order is the spec's lexicographic order.

    Speed comes from two things that do not touch the merge rule:
      * the pool-pair matrix is populated with one vectorized scatter rather
        than a Python loop over the queried pairs (1.2M of them on chng);
      * merges use the Lance-Williams update for complete linkage,
        D_CL(A u B, C) = max(D_CL(A,C), D_CL(B,C)), so each step is one
        vectorized row maximum instead of a re-scan over pool members.
    ``+inf`` marks unqueried or unsupported pairs and propagates correctly
    through the maximum, which is exactly the spec's admissibility rule.
    """
    n = len(geometry.tasks)
    if n <= 1:
        return np.zeros(max(n, 0), int)
    cache = geometry._cache
    if not cache:
        return np.arange(n)
    pairs = np.fromiter((x for key in cache for x in key), dtype=np.int64,
                        count=2 * len(cache)).reshape(-1, 2)
    values = np.fromiter(cache.values(), dtype=float, count=len(cache))
    admissible = np.isfinite(values) & (values <= threshold)
    linkage_matrix = np.full((n, n), np.inf)
    rows, columns = pairs[admissible, 0], pairs[admissible, 1]
    linkage_matrix[rows, columns] = values[admissible]
    linkage_matrix[columns, rows] = values[admissible]
    np.fill_diagonal(linkage_matrix, np.inf)
    members = {i: [i] for i in range(n)}
    while True:
        flat = int(np.argmin(linkage_matrix))
        a, b = divmod(flat, n)
        best = linkage_matrix[a, b]
        if not np.isfinite(best) or best > threshold:
            break
        if a > b:
            a, b = b, a
        np.maximum(linkage_matrix[a], linkage_matrix[b], out=linkage_matrix[a])
        linkage_matrix[:, a] = linkage_matrix[a]
        linkage_matrix[b, :] = np.inf
        linkage_matrix[:, b] = np.inf
        linkage_matrix[a, a] = np.inf
        members[a] = members[a] + members.pop(b)
    labels = np.zeros(n, int)
    for new_label, key in enumerate(sorted(members, key=lambda k: sorted(
            geometry.tasks[i] for i in members[k]))):
        for member in members[key]:
            labels[member] = new_label
    return labels


def sparse_task_pooling(training: pd.DataFrame, target_lag: int, **kwargs):
    """Build prototypes, calibrate locally, search sparsely, pool coherently."""
    prototypes = build_task_geometry(training, target_lag, distance_backend="prototypes",
                                     **kwargs)
    if prototypes is None:
        return None
    geometry = SparseTaskGeometry(prototypes.tasks, prototypes.profiles,
                                  prototypes.trajectories, prototypes.stages)
    scales, nearest, members = calibrate_sparse_scales(geometry)
    if not len(nearest):
        # "RevRoute cannot empirically calibrate a similarity boundary and uses
        # the deterministic no-pooling fallback."
        labels = np.arange(len(geometry.tasks))
        return geometry, labels, 0., nearest, prototypes
    threshold = nearest_neighbor_threshold(nearest)
    for position in range(len(geometry.tasks)):
        same_location_search(geometry, position, threshold)
        cross_location_valley_search(geometry, position, threshold)
    labels = coherence_preserving_pools(geometry, threshold)
    return geometry, labels, threshold, nearest, prototypes
