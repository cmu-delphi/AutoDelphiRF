"""Target-lag evidence comes only from the raw archive; L is always the user's."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import autodelphirf.pre_diagnosis.diagnosis as diagnosis
from autodelphirf.core.config import load_dataset_config
from autodelphirf.pre_diagnosis.archive import diagnose_raw_archive
from autodelphirf.workflow.run import run_pipeline


def episode(reference, releases):
    return [{"geo_value": "aa", "reference_date": pd.Timestamp(reference),
             "report_date": pd.Timestamp(reference) + pd.Timedelta(days=lag), "value": value}
            for lag, value in releases]


def test_an_episode_counts_as_zero_before_its_first_release():
    # Eight episodes reported from lag 0; two first reported at lag 5. All are
    # followed to lag 60 and settle at 100.
    rows = []
    for k in range(8):
        rows += episode(pd.Timestamp("2024-01-01") + pd.Timedelta(days=k),
                        [(0, 100.), (60, 100.)])
    for k in range(2):
        rows += episode(pd.Timestamp("2024-02-01") + pd.Timedelta(days=k),
                        [(5, 100.), (60, 100.)])
    report = diagnose_raw_archive(pd.DataFrame(rows), candidate_lags=(3, 5))
    # At lag 3 the late episodes are 0 against 100 (error 1) and stay in the
    # denominator: 8 of 10 are within 10%, below the 90% needed.
    assert report.target_lag_completion_band[3]["within_10pct"] == pytest.approx(0.8)
    assert report.target_lag_completion_band[5]["within_10pct"] == pytest.approx(1.0)
    assert report.target_lag_suggested_range["lower"] == 5


def test_values_are_forward_filled_between_releases():
    rows = episode("2024-01-01", [(0, 50.), (10, 100.), (60, 100.)])
    report = diagnose_raw_archive(pd.DataFrame(rows), candidate_lags=(9, 10))
    assert report.target_lag_completion_curve[9] == pytest.approx(0.5)
    assert report.target_lag_completion_curve[10] == pytest.approx(0.0)


def test_the_prepared_triangle_target_lag_diagnosis_is_gone():
    assert not hasattr(diagnosis, "diagnose_target_lag")
    assert hasattr(diagnosis, "diagnose_prepared_data")
    assert hasattr(diagnosis, "weekday_groups_from_share")


def prepared_only_dataset(tmp_path, *, target_lag):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = []
    for k, reference in enumerate(pd.date_range("2024-01-01", periods=40)):
        for lag in range(0, 7):
            rows.append({"geo_value": "aa", "reference_date": reference,
                         "report_date": reference + pd.Timedelta(days=lag),
                         "target_date": reference + pd.Timedelta(days=7), "lag": lag,
                         "target_lag": 7, "log_value_7dav": np.log1p(50. + lag),
                         "log_value_target_7dav": np.log1p(60.),
                         "log_delta_value_7dav_lag7": 0., "genuine_event": True})
    pd.DataFrame(rows).to_csv(prepared / "aa.csv", index=False)
    schedule = {"test_date": ["2024-02-01"], "experiment_end_date": ["2024-02-08"],
                "target_as_of_date": ["2024-02-16"],
                "testing_days": [7], "training_days": [180],
                "target_column": ["log_value_target_7dav"]}
    if target_lag is not None:
        schedule["target_lag"] = [target_lag]
    pd.DataFrame(schedule).to_csv(prepared / "test_dates.csv", index=False)
    # The run records which raw archive it came from (operational.raw_archive_state).
    raw = tmp_path / "raw.csv"
    raw.write_text("geo_value,reference_date,report_date,value\n")
    (prepared / "preparation.json").write_text(json.dumps(
        {"raw_archive": str(raw), "resolved_preprocessing": {}}))
    config = tmp_path / "prepared.json"
    config.write_text(json.dumps({
        "name": "prepared", "prepared_dir": str(prepared), "output_dir": str(tmp_path / "out"),
        "model_methods": ["baseline_null"], "prediction_layers": ["baseline_null"]}))
    return load_dataset_config(config)


def test_a_prepared_dataset_without_a_target_lag_asks_for_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)        # no raw-archive manifest in the working directory
    config = prepared_only_dataset(tmp_path, target_lag=None)
    with pytest.raises(ValueError, match="Set target_lag explicitly"):
        run_pipeline(config)
    # It stops before any case is built or model trained.
    assert not (tmp_path / "out" / "predictions.csv.gz").exists()


def test_a_prepared_dataset_with_a_target_lag_records_its_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = prepared_only_dataset(tmp_path, target_lag=7)
    run_pipeline(config)
    record = json.loads((tmp_path / "out" / "target_lag_resolution.json").read_text())
    assert record["resolved"] == 7
    assert record["diagnosed"]["status"] == "skipped: the target lag was specified"
    assert (tmp_path / "out" / "diagnostics" / "summary.json").is_file()
