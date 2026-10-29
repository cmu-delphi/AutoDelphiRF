"""Pipeline input handling: the guards a user's own triangle runs into.

Two of these are regressions from the first end-to-end run of this package on
a freshly built triangle, and both produced an opaque pandas error rather
than a usable message:

  * DelphiRF writes ``lag``/``target_lag`` as doubles (R date arithmetic
    yields numeric), and ``merge_asof`` compares join-key dtypes exactly;
  * a triangle whose follow-up stops short of the first cadence-aligned
    target lag produced an empty candidate list, and an empty
    ``DataFrame({"lag": []})`` is float64.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autodelphirf.pre_diagnosis.diagnosis import (diagnose_prepared_data,
                                                  resolve_target_lag)
from autodelphirf.pre_diagnosis.archive import diagnose_raw_archive
from autodelphirf.workflow.run import (INTEGER_COLUMNS, build_cases,
                                      normalize_lag_dtypes)



def prepared_frame(rows: int = 6, **overrides) -> pd.DataFrame:
    frame = pd.DataFrame({
        "geo_value": ["aa"] * rows,
        "reference_date": pd.date_range("2024-01-01", periods=rows),
        "report_date": pd.date_range("2024-01-02", periods=rows),
        "target_date": pd.date_range("2024-02-01", periods=rows),
        "lag": np.arange(1, rows + 1),
        "target_lag": np.full(rows, 28),
        "log_value_7dav": np.linspace(1.0, 2.0, rows),
        "genuine_event": [True] * rows,
    })
    return frame.assign(**overrides)


# --- lag dtype normalization (regression) ---------------------------------

def test_double_typed_lags_from_delphirf_become_integers():
    """R's date arithmetic yields numeric; merge_asof will not coerce it."""
    frame = prepared_frame().astype({"lag": "float64", "target_lag": "float64"})
    normalized = normalize_lag_dtypes(frame)
    assert normalized.lag.dtype == np.dtype("int64")
    assert normalized.target_lag.dtype == np.dtype("int64")
    assert normalized.lag.tolist() == frame.lag.astype(int).tolist()


def test_already_integer_lags_are_left_untouched():
    frame = prepared_frame()
    normalized = normalize_lag_dtypes(frame)
    assert normalized.lag.dtype == frame.lag.dtype
    pd.testing.assert_series_equal(normalized.lag, frame.lag)


def test_a_missing_target_lag_survives_as_a_nullable_integer():
    """An episode without an available target value has no target_lag; that is not an error."""
    frame = prepared_frame().astype({"target_lag": "float64"})
    frame.loc[frame.index[0], "target_lag"] = np.nan
    normalized = normalize_lag_dtypes(frame)
    assert str(normalized.target_lag.dtype) == "Int64"
    assert pd.isna(normalized.target_lag.iloc[0])
    assert normalized.target_lag.iloc[1] == 28


def test_a_column_with_no_missing_values_avoids_the_extension_dtype():
    """merge_asof rejects Int64 against int64 as firmly as it rejects float64."""
    normalized = normalize_lag_dtypes(prepared_frame().astype({"lag": "float64"}))
    assert normalized.lag.dtype == np.dtype("int64")
    assert not isinstance(normalized.lag.dtype, pd.api.types.pandas_dtype("Int64").__class__)


def test_a_fractional_lag_is_refused_rather_than_truncated():
    """3.5 days means the triangle was built on a different notion of a day."""
    frame = prepared_frame().astype({"lag": "float64"})
    frame.loc[frame.index[2], "lag"] = 3.5
    with pytest.raises(ValueError, match="non-integral"):
        normalize_lag_dtypes(frame)




def test_a_triangle_missing_a_lag_column_entirely_is_not_broken_by_normalization():
    """Normalization is defensive, not a second requiredness check."""
    frame = prepared_frame().drop(columns=["target_lag"])
    assert "target_lag" not in normalize_lag_dtypes(frame).columns


def test_the_normalized_columns_are_the_join_keys_downstream_uses():
    assert set(INTEGER_COLUMNS) == {"lag", "target_lag"}


# --- raw-archive target-lag diagnosis --------------------------------------

def test_raw_archive_target_lag_uses_the_same_ninety_percent_rule():
    rows = []
    for episode in range(10):
        reference_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=episode)
        values = {0: 40.0, 7: 90.0 if episode < 6 else 50.0,
                  14: 90.0, 21: 100.0}
        for lag, value in values.items():
            rows.append({"geo_value": "aa", "reference_date": reference_date,
                         "report_date": reference_date + pd.Timedelta(days=lag),
                         "value": value})
    report = diagnose_raw_archive(
        pd.DataFrame(rows), candidate_lags=(7, 14, 21),
        late_observation_floor_days=0
    )

    # Preserve the website's existing median-error curve even though selection
    # now uses the 90%-of-tasks completion rule.
    assert report.target_lag_completion_curve[7] == pytest.approx(0.1)
    assert report.target_lag_completion_curve[14] == pytest.approx(0.1)
    assert report.target_lag_completion_band[7]["q10"] == pytest.approx(0.1)
    assert report.target_lag_completion_band[7]["q60"] == pytest.approx(0.26)
    assert report.target_lag_completion_band[7]["q75"] == pytest.approx(0.5)
    assert report.target_lag_completion_band[7]["q90"] == pytest.approx(0.5)
    assert report.target_lag_completion_band[7]["q95"] == pytest.approx(0.5)
    assert report.target_lag_completion_band[14]["q10"] == pytest.approx(0.1)
    assert report.target_lag_completion_band[14]["q90"] == pytest.approx(0.1)
    assert report.target_lag_suggested_range == {
        "lower": 14, "upper": 21, "completion_rate": 0.9,
        "lower_relative_error": 0.10, "upper_relative_error": 0.01}
    assert report.available_target_lags == [7, 14, 21]
    assert report.resolved_target_lag is None


def test_raw_archive_default_diagnoses_every_observed_positive_lag():
    reference = pd.Timestamp("2024-01-01")
    raw = pd.DataFrame([
        {"geo_value": "aa", "reference_date": reference,
         "report_date": reference + pd.Timedelta(days=lag), "value": value}
        for lag, value in ((0, 5.0), (7, 8.0), (8, 9.0), (14, 10.0))
    ])
    report = diagnose_raw_archive(raw, late_observation_floor_days=0)

    assert report.available_target_lags == [7, 8, 14]
    assert sorted(report.target_lag_completion_curve) == [7, 8, 14]


def test_raw_diagnosis_ignores_observations_after_two_years():
    reference = pd.Timestamp("2022-01-01")
    raw = pd.DataFrame([
        {"geo_value": "aa", "reference_date": reference,
         "report_date": reference + pd.Timedelta(days=lag), "value": float(lag)}
        for lag in (0, 7, 730, 731, 900)
    ])

    report = diagnose_raw_archive(raw, late_observation_floor_days=0)

    assert report.n_rows == 3
    assert report.max_observed_lag_days == 730
    assert report.available_target_lags == [7, 730]
    assert sorted(report.target_lag_completion_curve) == [7, 730]


def test_raw_diagnosis_uses_observed_reports_directly_without_a_genuine_event_concept():
    reference = pd.Timestamp("2024-01-01")
    raw = pd.DataFrame([
        {"geo_value": "aa", "reference_date": reference,
         "report_date": reference + pd.Timedelta(days=lag), "value": value}
        for lag, value in ((0, 10.0), (1, 10.0), (2, 12.0))
    ])

    report = diagnose_raw_archive(raw, late_observation_floor_days=0)

    # The first release and the unchanged repeat are both raw observations.
    # Their spacing is diagnosed directly; genuine_event only becomes useful
    # after DelphiRF adds completed-grid rows.
    assert report.observed_report_gap_days == {
        "n": 2, "q25": 1.0, "median": 1.0, "q75": 1.0,
        "most_common_gap_days": {1: 2}}
    payload = report.to_dict()
    assert "genuine_event_rate" not in payload
    assert "genuine_event_weekday_share" not in payload


@pytest.mark.parametrize("chosen", [1, 14, 21, 400])
def test_user_target_lag_always_wins_without_range_reconciliation(chosen):
    diagnosed = {"selected_target_lag": None,
                 "suggested_target_lag_range": {"lower": 14, "upper": 21}}
    resolved, reason = resolve_target_lag(diagnosed, chosen, False, "example")
    assert resolved == chosen
    assert reason == "explicit user choice"


def test_pipeline_target_lag_resolution_stops_without_a_user_choice():
    diagnosed = {"suggested_target_lag_range": {"lower": 14, "upper": 21}}
    with pytest.raises(ValueError, match="Set target_lag explicitly"):
        resolve_target_lag(diagnosed, None, False, "example")


# --- input schema ---------------------------------------------------------









# --- case construction ----------------------------------------------------

def schedule_frame(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame({"test_date": [pd.Timestamp("2024-01-03")],
                          "experiment_end_date": [pd.Timestamp("2024-01-06")],
                          "training_days": [180],
                          "target_column": ["log_value_target_7dav"]})
    return frame.assign(**overrides)


def test_a_schedule_window_containing_no_reports_produces_no_cases():
    """The pipeline turns this into a clear error; build_cases just returns empty."""
    prepared = prepared_frame().assign(log_value_target_7dav=1.0)
    far_future = schedule_frame(test_date=[pd.Timestamp("2030-01-01")],
                                experiment_end_date=[pd.Timestamp("2030-02-01")])
    assert build_cases(prepared, far_future, True).empty


def test_genuine_event_filtering_keeps_exactly_the_genuine_rows():
    prepared = prepared_frame(rows=6).assign(log_value_target_7dav=1.0)
    prepared.loc[prepared.index[:3], "genuine_event"] = False
    schedule = schedule_frame(test_date=[pd.Timestamp("2024-01-01")],
                              experiment_end_date=[pd.Timestamp("2024-02-01")])
    filtered = build_cases(prepared, schedule, True)
    unfiltered = build_cases(prepared, schedule, False)
    assert len(unfiltered) == 6 and len(filtered) == 3
    assert set(filtered.report_date) == set(prepared[prepared.genuine_event].report_date)
    # Every case records whether it came from a real release, so an unfiltered
    # run can still be evaluated on the genuine rows alone.
    assert filtered.genuine_event.all()
    assert unfiltered.genuine_event.tolist() == prepared.genuine_event.tolist()


def test_a_missing_genuine_event_flag_is_treated_as_not_genuine():
    """A null flag must not silently pass the filter as a truthy object."""
    prepared = prepared_frame(rows=4).assign(log_value_target_7dav=1.0)
    prepared["genuine_event"] = [True, None, np.nan, True]
    schedule = schedule_frame(test_date=[pd.Timestamp("2024-01-01")],
                              experiment_end_date=[pd.Timestamp("2024-02-01")])
    assert len(build_cases(prepared, schedule, True)) == 2


def test_a_triangle_without_a_genuine_event_column_is_rejected():
    prepared = prepared_frame(rows=4).drop(columns=["genuine_event"]).assign(
        log_value_target_7dav=1.0)
    schedule = schedule_frame(test_date=[pd.Timestamp("2024-01-01")],
                              experiment_end_date=[pd.Timestamp("2024-02-01")])
    with pytest.raises(ValueError, match="requires a genuine_event column"):
        build_cases(prepared, schedule, True)


def test_rows_with_no_available_target_value_remain_forecast_cases():
    """Operational forecasts survive until their truth becomes available."""
    prepared = prepared_frame(rows=4).assign(log_value_target_7dav=[1.0, np.nan, 3.0, np.nan])
    schedule = schedule_frame(test_date=[pd.Timestamp("2024-01-01")],
                              experiment_end_date=[pd.Timestamp("2024-02-01")])
    cases = build_cases(prepared, schedule, True)
    assert len(cases) == 4
    assert cases.truth.notna().sum() == 2


# --- prepared-data diagnosis ---------------------------------------------

def test_diagnosis_reports_the_scheduled_target_column():
    prepared = prepared_frame().assign(dataset_target=2.0)
    summary, _ = diagnose_prepared_data(prepared, schedule_frame(target_column=["dataset_target"]))
    assert summary["target_column"] == "dataset_target"
