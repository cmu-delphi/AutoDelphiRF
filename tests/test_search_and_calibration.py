"""The thesis stopping rule for cross-location search and symmetric non-crossing."""
from __future__ import annotations

import numpy as np

from autodelphirf.post_assessment.reliability import enforce_non_crossing
from autodelphirf.revroute.pooling.clustering import cross_location_valley_search


class Distances:
    """Location A at lag 5 searching location B's lags with prescribed distances."""

    def __init__(self, distances: dict[int, float]):
        self.tasks = [("A", 5)] + [("B", lag) for lag in sorted(distances)]
        self.by_location = {"A": {5: 0}}
        self.by_location["B"] = {lag: i + 1 for i, lag in enumerate(sorted(distances))}
        self.stages_of = {geo: sorted(lags) for geo, lags in self.by_location.items()}
        self.distances = distances
        self.queried: set[int] = set()

    def distance(self, i: int, j: int) -> float:
        lag = self.tasks[j][1]
        self.queried.add(lag)
        return self.distances[lag]


def test_an_improving_step_above_the_threshold_is_not_a_miss():
    # Lags 6 and 7 improve the minimum; lag 8 is the only miss before lag 9.
    geometry = Distances({5: 3.0, 6: 2.5, 7: 2.0, 8: 2.1, 9: 0.5, 10: 0.4})
    cross_location_valley_search(geometry, 0, threshold=1.0)
    assert {9, 10} <= geometry.queried


def test_two_consecutive_misses_close_the_direction():
    geometry = Distances({5: 3.0, 6: 2.5, 7: 2.6, 8: 2.7, 9: 0.5})
    cross_location_valley_search(geometry, 0, threshold=1.0)
    assert 8 in geometry.queried and 9 not in geometry.queried


def test_without_a_centre_lag_the_same_rule_applies():
    geometry = Distances({6: 2.5, 7: 2.0, 8: 2.1, 9: 0.5})
    cross_location_valley_search(geometry, 0, threshold=1.0)
    assert 9 in geometry.queried


TAUS = [.01, .025, .1, .25, .5, .75, .9, .975, .99]


def test_a_lower_crossing_widens_the_outer_interval():
    calibrated = np.array([3.4, 3.3, 3.5, 3.6, 4.0, 4.4, 4.6, 4.8, 4.9])
    fixed = enforce_non_crossing(calibrated, TAUS.index(.5))[0]
    # 0.01 is pushed down to 0.025's value; the 95% endpoint is untouched.
    assert fixed[1] == 3.3 and fixed[0] == 3.3


def test_an_upper_crossing_widens_the_outer_interval():
    calibrated = np.array([3.0, 3.1, 3.5, 3.6, 4.0, 4.4, 4.6, 5.0, 4.9])
    fixed = enforce_non_crossing(calibrated, TAUS.index(.5))[0]
    assert fixed[7] == 5.0 and fixed[8] == 5.0


def test_quantiles_are_left_alone_when_they_do_not_cross():
    calibrated = np.array([3.0, 3.1, 3.5, 3.6, 4.0, 4.4, 4.6, 4.8, 4.9])
    assert np.array_equal(enforce_non_crossing(calibrated, TAUS.index(.5))[0], calibrated)


def test_non_crossing_output_is_ordered():
    rng = np.random.default_rng(0)
    calibrated = np.sort(rng.normal(size=(200, 9)), axis=1) + rng.normal(scale=.3, size=(200, 9))
    fixed = enforce_non_crossing(calibrated, TAUS.index(.5))
    assert (np.diff(fixed, axis=1) >= 0).all()
