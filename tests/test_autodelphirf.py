from pathlib import Path
import json
import numpy as np
import pandas as pd

from autodelphirf.config import load_dataset_config
from autodelphirf.diagnosis import diagnose_prepared_data
from autodelphirf.pipeline import apply_input_schema, build_cases, standardize_predictions
# ROUTED_TAUS lives in the package's red.py module.
from autodelphirf.red import ROUTED_TAUS


def test_config_paths_are_relative_to_config(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"name": "x", "prepared_dir": "input", "output_dir": "out",
                                "prediction_layers": ["red"]}))
    config = load_dataset_config(path)
    assert config.prepared_dir == tmp_path / "input"
    assert config.output_dir == tmp_path / "out"


def test_diagnosis_uses_scheduled_target_column():
    prepared = pd.DataFrame({"geo_value": ["a"], "reference_date": ["2020-01-01"],
        "report_date": ["2020-01-02"], "target_date": ["2020-01-03"], "lag": [1],
        "target_lag": [2], "log_value_7dav": [1.0], "dataset_target": [2.0]})
    schedule = pd.DataFrame({"target_column": ["dataset_target"]})
    summary, _ = diagnose_prepared_data(prepared, schedule)
    assert summary["target_column"] == "dataset_target"


def test_input_schema_renames_noncanonical_source_columns():
    source = pd.DataFrame({"event": ["2020-01-01"], "issue": ["2020-01-02"],
                           "target_ready": ["2020-01-03"], "place": ["a"], "age": [1],
                           "target_lag_source": [2], "asof": [1.2]})
    mapped = apply_input_schema(source, {"geo_value": "place", "reference_date": "event",
        "report_date": "issue", "target_date": "target_ready", "lag": "age",
        "target_lag": "target_lag_source", "as_of_value": "asof"})
    assert {"geo_value", "reference_date", "report_date", "target_date", "lag",
            "target_lag", "log_value_7dav"}.issubset(mapped.columns)


def test_point_comparator_has_no_degenerate_distribution(tmp_path):
    config_path = tmp_path / "dataset.json"
    config_path.write_text(json.dumps({"name": "x", "prepared_dir": ".", "output_dir": ".",
        "prediction_layers": ["benchmark"], "comparator_columns": {"benchmark": "other"},
        # Synthetic config with no prepared triangle to derive from, so the
        # working scale is stated explicitly here; real configs derive it.
        "value_transform": "log1p"}))
    config = load_dataset_config(config_path)
    row = {"truth": np.log1p(10), "other": np.log1p(8), "lag": 1}
    out = standardize_predictions(pd.DataFrame([row]), config)
    assert np.isclose(out.prediction.iloc[0], 8)
    assert not out.has_distribution.iloc[0]
    assert np.isnan(out.wis.iloc[0])
    assert all(np.isnan(out[f"q{tau:g}"].iloc[0]) for tau in ROUTED_TAUS)


def test_build_cases_respects_genuine_event_filter():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    prepared = pd.DataFrame({"geo_value": ["a", "a"], "reference_date": dates,
        "report_date": dates, "target_date": dates, "lag": [0, 0], "truth_col": [1., 2.],
        "log_value_7dav": [1., 2.], "genuine_event": [True, False]})
    schedule = pd.DataFrame({"test_date": [dates[0]], "experiment_end_date": [dates[-1] + pd.Timedelta(days=1)],
                             "target_column": ["truth_col"]})
    assert len(build_cases(prepared, schedule, True)) == 1
    assert len(build_cases(prepared, schedule, False)) == 2


def test_comparator_validation_catches_missing_method(tmp_path):
    from autodelphirf.comparator_schema import check_comparator_file
    comparator_path = tmp_path / "comparator.csv"
    pd.DataFrame({"geo_value": ["a"], "reference_date": ["2020-01-01"], "report_date": ["2020-01-08"],
                 "lag": [7], "method": ["Naive DelphiRF"], "prediction": [1.0]}).to_csv(comparator_path, index=False)
    issues = check_comparator_file(comparator_path, "method", "prediction",
                                   {"naive_delphirf": "Some Other Method"})
    assert any("does not appear in column" in issue for issue in issues)


def test_comparator_validation_catches_duplicate_rows(tmp_path):
    from autodelphirf.comparator_schema import check_comparator_file
    comparator_path = tmp_path / "comparator.csv"
    pd.DataFrame({"geo_value": ["a", "a"], "reference_date": ["2020-01-01"] * 2,
                 "report_date": ["2020-01-08"] * 2, "lag": [7, 7],
                 "method": ["Naive DelphiRF"] * 2, "prediction": [1.0, 2.0]}).to_csv(comparator_path, index=False)
    issues = check_comparator_file(comparator_path, "method", "prediction", {"naive_delphirf": "Naive DelphiRF"})
    assert any("sharing the same" in issue for issue in issues)


def test_comparator_validation_passes_on_valid_file(tmp_path):
    from autodelphirf.comparator_schema import check_comparator_file
    comparator_path = tmp_path / "comparator.csv"
    pd.DataFrame({"geo_value": ["a", "b"], "reference_date": ["2020-01-01"] * 2,
                 "report_date": ["2020-01-08"] * 2, "lag": [7, 7],
                 "method": ["Naive DelphiRF"] * 2, "prediction": [1.0, 2.0]}).to_csv(comparator_path, index=False)
    issues = check_comparator_file(comparator_path, "method", "prediction", {"naive_delphirf": "Naive DelphiRF"})
    assert issues == []
