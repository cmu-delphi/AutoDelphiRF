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

from autodelphirf.diagnosis import diagnose_prepared_data, diagnose_target_lag
from autodelphirf.ingest import diagnose_raw_archive
from autodelphirf.pipeline import (INTEGER_COLUMNS, apply_input_schema, build_cases,
                                      normalize_lag_dtypes)
from autodelphirf.resources import load_resource

CANONICAL = load_resource("prepared_triangle_schema.json")["columns"]


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


def test_normalization_happens_inside_apply_input_schema():
    """The single choke point: no downstream module should have to care."""
    frame = prepared_frame().astype({"lag": "float64"})
    normalized = apply_input_schema(frame, CANONICAL)
    assert normalized.lag.dtype == np.dtype("int64")


def test_a_triangle_missing_a_lag_column_entirely_is_not_broken_by_normalization():
    """Normalization is defensive, not a second requiredness check."""
    frame = prepared_frame().drop(columns=["target_lag"])
    assert "target_lag" not in normalize_lag_dtypes(frame).columns


def test_the_normalized_columns_are_the_join_keys_downstream_uses():
    assert set(INTEGER_COLUMNS) == {"lag", "target_lag"}


# --- target-lag diagnosis edge cases (regression) --------------------------

def diagnosis_frame(max_lag: int, episodes: int = 40) -> pd.DataFrame:
    rows = []
    for episode in range(episodes):
        reference_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=episode)
        for lag in range(max_lag + 1):
            rows.append({"geo_value": "aa", "reference_date": reference_date,
                         "report_date": reference_date + pd.Timedelta(days=lag),
                         "lag": lag, "value_7dav": 100.0 * (1 - 0.5 * np.exp(-lag / 3))})
    return pd.DataFrame(rows)


def test_a_triangle_shorter_than_the_cadence_reports_instead_of_crashing():
    """No cadence-aligned target lag exists below lag 7; that is a status, not a stack trace."""
    result = diagnose_target_lag(diagnosis_frame(max_lag=6), value_column="value_7dav")
    assert result["selected_target_lag"] is None
    assert "no candidate target lag" in result["status"]
    # The message must say what to do about it.
    assert "cadence_days" in result["status"]
    assert result["completion"] == []


def test_the_reported_ceiling_is_the_triangles_own_reach():
    result = diagnose_target_lag(diagnosis_frame(max_lag=5), value_column="value_7dav")
    assert result["rule"]["max_lag_examined"] == 5


def test_a_shorter_cadence_finds_a_target_lag_in_the_same_short_triangle():
    """Confirms the early return is about the cadence, not about the data being unusable."""
    result = diagnose_target_lag(diagnosis_frame(max_lag=6), value_column="value_7dav",
                                 cadence_days=2)
    assert result["selected_target_lag"] is not None


def test_a_missing_value_column_is_reported_by_name():
    result = diagnose_target_lag(diagnosis_frame(max_lag=30), value_column="absent_column")
    assert result["selected_target_lag"] is None
    assert "absent_column" in result["status"]


def test_an_all_future_selection_cutoff_empties_the_frame_cleanly():
    result = diagnose_target_lag(diagnosis_frame(max_lag=30), value_column="value_7dav",
                                 selection_cutoff="2000-01-01")
    assert result["selected_target_lag"] is None
    assert "selection cutoff" in result["status"]


@pytest.mark.parametrize("bad", [{"completion_rate": 0}, {"completion_rate": 1.5},
                                 {"relative_error": -0.1}, {"cadence_days": 0}])
def test_invalid_rule_settings_are_rejected(bad):
    with pytest.raises(ValueError, match="invalid target-lag rule"):
        diagnose_target_lag(diagnosis_frame(max_lag=30), value_column="value_7dav", **bad)


def test_follow_up_requirement_can_exclude_every_episode_without_crashing():
    result = diagnose_target_lag(diagnosis_frame(max_lag=30), value_column="value_7dav",
                                 min_followup_days=999)
    assert result["selected_target_lag"] is None
    assert result["completion"] == []


def test_target_lag_requires_ninety_percent_of_tasks_within_tolerance():
    rows = []
    for episode in range(10):
        reference_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=episode)
        values = {0: 40.0, 7: 90.0 if episode < 6 else 50.0,
                  14: 90.0, 21: 100.0}
        for lag, value in values.items():
            rows.append({"geo_value": "aa", "reference_date": reference_date,
                         "report_date": reference_date + pd.Timedelta(days=lag),
                         "lag": lag, "value_7dav": value})
    result = diagnose_target_lag(pd.DataFrame(rows), value_column="value_7dav")
    by_lag = {item["lag"]: item for item in result["completion"]}

    # At lag 7 the median error is 10%, but just 60% of tasks are within
    # 10%. The former median rule selected lag 7; the 90%-of-tasks rule must not.
    assert by_lag[7]["median_relative_error"] == pytest.approx(0.1)
    assert by_lag[7]["within_tolerance_rate"] == pytest.approx(0.6)
    assert result["selected_target_lag"] == 14
    assert result["within_tolerance_rate"] == pytest.approx(1.0)


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
    assert report.target_lag_completion_band[7] == pytest.approx({"q10": 0.1, "q90": 0.5})
    assert report.target_lag_completion_band[14] == pytest.approx({"q10": 0.1, "q90": 0.1})
    assert report.recommended_target_lag == 14


# --- input schema ---------------------------------------------------------

def test_a_mapping_missing_a_semantic_field_is_refused():
    incomplete = {k: v for k, v in CANONICAL.items() if k != "target_date"}
    with pytest.raises(ValueError, match="target_date"):
        apply_input_schema(prepared_frame(), incomplete)


def test_a_source_column_that_is_absent_from_the_triangle_is_named():
    mapping = {**CANONICAL, "geo_value": "location_id"}
    with pytest.raises(ValueError, match="location_id"):
        apply_input_schema(prepared_frame(), mapping)


def test_renaming_onto_an_existing_column_is_refused_rather_than_silently_dropping_one():
    frame = prepared_frame().assign(location_id=["aa"] * 6)
    mapping = {**CANONICAL, "geo_value": "location_id"}
    with pytest.raises(ValueError, match="existing column"):
        apply_input_schema(frame, mapping)


def test_noncanonical_source_names_are_renamed_to_the_internal_schema():
    frame = prepared_frame().rename(columns={"geo_value": "location_id",
                                             "log_value_7dav": "log_current"})
    mapping = {**CANONICAL, "geo_value": "location_id", "as_of_value": "log_current"}
    renamed = apply_input_schema(frame, mapping)
    assert "geo_value" in renamed and "log_value_7dav" in renamed
    assert "location_id" not in renamed


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
    # build_cases does not carry genuine_event into its output, so identity is
    # checked on the report dates that survived.
    assert set(filtered.report_date) == set(prepared[prepared.genuine_event].report_date)


def test_a_missing_genuine_event_flag_is_treated_as_not_genuine():
    """A null flag must not silently pass the filter as a truthy object."""
    prepared = prepared_frame(rows=4).assign(log_value_target_7dav=1.0)
    prepared["genuine_event"] = [True, None, np.nan, True]
    schedule = schedule_frame(test_date=[pd.Timestamp("2024-01-01")],
                              experiment_end_date=[pd.Timestamp("2024-02-01")])
    assert len(build_cases(prepared, schedule, True)) == 2


def test_a_triangle_without_a_genuine_event_column_is_not_filtered_away():
    """The column is optional; its absence must not empty the case set."""
    prepared = prepared_frame(rows=4).drop(columns=["genuine_event"]).assign(
        log_value_target_7dav=1.0)
    schedule = schedule_frame(test_date=[pd.Timestamp("2024-01-01")],
                              experiment_end_date=[pd.Timestamp("2024-02-01")])
    assert len(build_cases(prepared, schedule, True)) == 4


def test_rows_with_no_available_target_value_are_excluded_from_evaluation():
    """A case with no truth cannot be scored, so it is not a case."""
    prepared = prepared_frame(rows=4).assign(log_value_target_7dav=[1.0, np.nan, 3.0, np.nan])
    schedule = schedule_frame(test_date=[pd.Timestamp("2024-01-01")],
                              experiment_end_date=[pd.Timestamp("2024-02-01")])
    assert len(build_cases(prepared, schedule, True)) == 2


# --- prepared-data diagnosis ---------------------------------------------

def test_diagnosis_reports_the_scheduled_target_column():
    prepared = prepared_frame().assign(dataset_target=2.0)
    summary, _ = diagnose_prepared_data(prepared, schedule_frame(target_column=["dataset_target"]))
    assert summary["target_column"] == "dataset_target"
