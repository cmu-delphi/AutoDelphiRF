"""The thesis stopping rule for cross-location search and symmetric non-crossing."""
from __future__ import annotations

import numpy as np

import pandas as pd

from autodelphirf.post_assessment.reliability import (
    _pool_strata, _quantile_level_map, _remap_quantiles, enforce_non_crossing, pit_diagnostics,
    prospective_quantile_mapping)
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


def test_quantile_mapping_can_narrow_an_overwide_forecast():
    taus = np.asarray(TAUS)
    # Historical truths are uniform, while the forecast quantile curve is too
    # dispersed around .5. The inverse reliability map moves both central
    # endpoints inward in probability space.
    truth = np.linspace(0, 1, 201)
    history = pd.DataFrame({"truth": truth})
    columns = []
    for tau in taus:
        column = f"q{tau:g}"
        columns.append(column)
        history[column] = .5 + 2 * (tau - .5)
    mapped = _quantile_level_map(history, columns, "truth", taus)
    assert np.all(np.diff(mapped) >= 0)
    assert mapped[TAUS.index(.25)] > .25
    assert mapped[TAUS.index(.75)] < .75


def test_remap_quantiles_uses_constant_extrapolation_beyond_outer_quantiles():
    taus = np.asarray(TAUS)
    base = np.asarray([[10., 20., 30., 40., 50., 60., 70., 80., 90.]])
    source_levels = np.asarray([0., .005, .1, .25, .5, .75, .9, .995, 1.])

    remapped = _remap_quantiles(base, source_levels, taus)[0]

    assert remapped[0] == 10.
    assert remapped[1] == 10.
    assert remapped[-2] == 90.
    assert remapped[-1] == 90.


def _mapping_frame(include_future_target=True):
    report_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-10"])
    target_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-01-20"])
    frame = pd.DataFrame({
        "method": "delphirf", "fold": [1, 2, 3, 4],
        "cutoff": report_dates, "geo_value": "a",
        "reference_date": report_dates - pd.Timedelta(days=1),
        "report_date": report_dates, "target_date": target_dates,
        "lag": 1, "truth": [0.2, 0.8, 1000., 0.5],
        "model_available": True, "has_distribution": True,
    })
    for tau in TAUS:
        frame[f"q{tau:g}"] = tau
    return frame if include_future_target else frame.drop(index=2).reset_index(drop=True)


def test_prospective_quantile_mapping_is_end_to_end_and_does_not_use_future_targets():
    with_future = prospective_quantile_mapping(
        _mapping_frame(), ["delphirf"], strategy="location_lag",
        minimum=2, working_scale="identity")
    without_future = prospective_quantile_mapping(
        _mapping_frame(False), ["delphirf"], strategy="location_lag",
        minimum=2, working_scale="identity")
    current_date = pd.Timestamp("2024-01-10")
    left = with_future[with_future.report_date.eq(current_date)].iloc[0]
    right = without_future[without_future.report_date.eq(current_date)].iloc[0]

    assert left.calibration_source == "location/exact-lag"
    assert left.calibration_n == 2
    assert np.allclose([left[f"cal_q{tau:g}"] for tau in TAUS],
                       [right[f"cal_q{tau:g}"] for tau in TAUS])


def test_prospective_quantile_mapping_falls_back_to_base_with_too_little_history():
    mapped = prospective_quantile_mapping(
        _mapping_frame(False), ["delphirf"], strategy="location_lag",
        minimum=20, working_scale="identity")
    row = mapped[mapped.report_date.eq(pd.Timestamp("2024-01-10"))].iloc[0]

    assert row.calibration_source == "insufficient history"
    assert row.calibration_n == 2
    assert np.allclose([row[f"cal_q{tau:g}"] for tau in TAUS],
                       [row[f"q{tau:g}"] for tau in TAUS])


def test_pool_history_uses_current_origin_members_not_historical_pool_numbers():
    history = pd.DataFrame({
        "geo_value": ["a", "b", "c"], "lag": [1, 2, 3], "truth": [1., 2., 3.]})
    test = pd.DataFrame({"geo_value": ["a"], "lag": [1]})
    current = pd.DataFrame({
        "geo_value": ["a", "b", "c"], "lag": [1, 2, 3], "pool": [7, 7, 9]})
    selected, source = _pool_strata(test, history, current, minimum=2)[("a", 1)]
    assert source == "current-origin pool"
    assert set(zip(selected.geo_value, selected.lag)) == {("a", 1), ("b", 2)}


def test_pool_mapping_requires_an_explicit_origin_column():
    assignments = pd.DataFrame({"geo_value": ["a"], "lag": [1], "pool": [1]})

    try:
        prospective_quantile_mapping(
            pd.DataFrame(), [], strategy="pool", pool_assignments=assignments)
    except ValueError as error:
        assert "origin column" in str(error)
    else:
        raise AssertionError("pool mapping accepted assignments with no fold or cutoff")


def test_pit_is_reported_as_a_bin_for_a_nine_quantile_forecast():
    taus = np.asarray(TAUS)
    frame = pd.DataFrame({
        "method": ["delphirf"], "fold": [1], "cutoff": pd.to_datetime(["2024-01-01"]),
        "geo_value": ["a"], "reference_date": pd.to_datetime(["2023-12-31"]),
        "report_date": pd.to_datetime(["2024-01-01"]),
        "target_date": pd.to_datetime(["2024-01-14"]), "lag": [1], "truth": [5.]})
    for tau, value in zip(taus, np.arange(1., 10.)):
        frame[f"q{tau:g}"] = value
        frame[f"cal_q{tau:g}"] = value
    reliability, pit = pit_diagnostics({"base": frame, "endpoint": frame})
    assert set(reliability.strategy) == {"base", "endpoint"}
    row = pit[pit.strategy.eq("base")].iloc[0]
    assert row.pit_lower == .25 and row.pit_upper == .5
    assert row.pit_midpoint == .375


def test_pit_ties_span_every_equal_quantile_with_boundary_tolerance():
    taus = np.asarray(TAUS)
    frame = pd.DataFrame({"method": ["delphirf"], "truth": [5.]})
    values = [1., 2., 5. + 1e-12, 5., 5. - 1e-12, 6., 7., 8., 9.]
    for tau, value in zip(taus, values):
        frame[f"q{tau:g}"] = value

    reliability, pit = pit_diagnostics({"base": frame})

    row = pit.iloc[0]
    assert row.pit_tied
    assert row.pit_tied_quantiles == 3
    assert row.pit_lower == .025
    assert row.pit_upper == .5
    coverage = reliability.set_index("nominal_quantile").empirical_coverage
    assert coverage.loc[.1] == 1.0
