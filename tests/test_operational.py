import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autodelphirf.workflow.operational import (
    append_forecast_ledger, ensure_analysis_contract, exclude_issued_cases,
    raw_archive_state, read_deployment_origin, read_forecast_ledger,
    refresh_forecast_ledger_truth, select_update_schedule, write_deployment_origin,
)
from autodelphirf.revroute.pooling import plan as pooling_plan
from autodelphirf.core.history import rolling_completed_training


def preparation(tmp_path, raw, **overrides):
    resolved = {
        "reference_col": "reference_date", "report_col": "report_date",
        "geo_col": "geo_value", "constant_geo": "single_location",
        "value_cols": ["value"], "value_type": "count", "smoothed": True,
        "temporal_resol": "daily", "ref_lag": 28, "lag_terms": [7, 14],
        "weekday_groups": {"Wed": ["Wed"]}, "training_days": 180,
        "testing_days": 7,
        "target_column": "log_value_target_7dav", "triangle_format": "parquet",
        "min_location_rows": 100,
    }
    resolved.update(overrides)
    prepared = tmp_path / "prepared"
    prepared.mkdir(exist_ok=True)
    (prepared / "preparation.json").write_text(json.dumps({
        "raw_archive": str(raw), "resolved_preprocessing": resolved}))
    return prepared


def test_analysis_contract_freezes_first_run_settings(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("geo_value,reference_date,report_date,value\n")
    prepared = preparation(tmp_path, raw)
    first = ensure_analysis_contract(prepared, tmp_path / "results")
    assert first["settings"]["ref_lag"] == 28
    preparation(tmp_path, raw, ref_lag=35)
    with pytest.raises(ValueError, match="ref_lag"):
        ensure_analysis_contract(prepared, tmp_path / "results")


def test_a_contract_frozen_with_target_lag_tolerances_names_them(tmp_path):
    # Contracts frozen before the tolerances were removed still carry them; an
    # update must say which settings differ rather than report an empty list.
    raw = tmp_path / "raw.csv"
    raw.write_text("geo_value,reference_date,report_date,value\n")
    prepared = preparation(tmp_path, raw)
    results = tmp_path / "results"
    contract = ensure_analysis_contract(prepared, results)
    contract["settings"].update(lower=0, upper=0)
    (results / "analysis_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match=r"\['lower', 'upper'\]"):
        ensure_analysis_contract(prepared, results)


def test_raw_archive_state_detects_an_unchanged_or_updated_file(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = pd.DataFrame({
        "geo_value": ["a"], "reference_date": ["2026-01-01"],
        "report_date": ["2026-01-02"], "value": [10.]})
    rows.to_csv(raw, index=False)
    prepared = preparation(tmp_path, raw)
    first = raw_archive_state(prepared, tmp_path / "results")
    assert first["changed"] is True
    assert raw_archive_state(prepared, tmp_path / "results", commit=False)["changed"] is False
    pd.concat([rows, pd.DataFrame({
        "geo_value": ["a"], "reference_date": ["2026-01-01"],
        "report_date": ["2026-01-03"], "value": [12.]})]).to_csv(raw, index=False)
    second = raw_archive_state(prepared, tmp_path / "results", commit=False)
    assert second["changed"] is True
    changed = pd.read_csv(raw)
    changed.loc[0, "value"] = 11
    changed.to_csv(raw, index=False)
    assert raw_archive_state(prepared, tmp_path / "results", commit=False)["changed"] is True


def forecast(value=11., truth=None):
    return pd.DataFrame({
        "dataset": ["d"], "method": ["delphirf"], "cutoff": ["2026-01-02"],
        "geo_value": ["a"], "reference_date": ["2026-01-01"],
        "report_date": ["2026-01-02"], "target_date": ["2026-01-29"],
        "lag": [1], "prediction": [value], "prediction_working": [2.4],
        "provisional_value": [10.], "predicted_remaining_revision": [1.],
        "truth": [truth], "absolute_error": [None if truth is None else abs(truth-value)],
        "q0.5": [value],
    })


def test_forecast_ledger_is_idempotent_and_allows_truth_to_arrive(tmp_path):
    first = append_forecast_ledger(forecast(), tmp_path)
    assert first["new_rows"] == 1
    second = append_forecast_ledger(forecast(truth=12.), tmp_path)
    assert second["new_rows"] == 0
    saved = pd.read_csv(tmp_path / "forecast_ledger.csv.gz")
    assert saved.loc[0, "prediction"] == 11
    assert saved.loc[0, "truth"] == 12
    with pytest.raises(ValueError, match="changed in prediction"):
        append_forecast_ledger(forecast(value=13.), tmp_path)


def test_operational_pooling_preserves_issued_origins(tmp_path, monkeypatch):
    output = tmp_path / "revroute_pools.csv"
    pd.DataFrame({
        "fold": [1], "cutoff": ["2026-01-02"], "geo_value": ["a"],
        "lag": [1], "pool": [17], "threshold": [.2],
    }).to_csv(output, index=False)
    pd.DataFrame({
        "fold": [1], "cutoff": ["2026-01-02"], "represented_tasks": [1],
    }).to_csv(tmp_path / "revroute_pool_summary.csv", index=False)
    schedule = pd.DataFrame({
        "test_date": pd.to_datetime(["2026-01-02", "2026-01-09"]),
        "training_days": [30, 30], "target_lag": [28, 28],
        "target_column": ["target", "target"],
    })

    class Geometry:
        tasks = [("b", 2)]

    calls = []
    monkeypatch.setattr(pooling_plan, "rolling_completed_training",
                        lambda prepared, cutoff, days:
                        calls.append(cutoff) or prepared)
    monkeypatch.setattr(pooling_plan, "sparse_task_pooling",
                        lambda *args, **kwargs: (Geometry(), [3], .4, None, None))
    pooling_plan.learn_task_pools(pd.DataFrame(), schedule, output,
                                  preserve_existing=True)
    saved = pd.read_csv(output)
    assert calls == [pd.Timestamp("2026-01-09")]
    assert saved[["cutoff", "pool"]].to_dict("records") == [
        {"cutoff": "2026-01-02", "pool": 17},
        {"cutoff": "2026-01-09", "pool": 3},
    ]


def test_update_selects_only_latest_applicable_origin(tmp_path):
    ledger = forecast()
    schedule = pd.DataFrame({
        "test_date": pd.to_datetime(["2026-01-02", "2026-01-09"]),
        "experiment_end_date": pd.to_datetime(["2026-01-09", "2026-01-16"]),
    })
    selected, retrain = select_update_schedule(schedule, pd.Timestamp("2026-01-02"))
    assert retrain is True
    assert selected.test_date.tolist() == [pd.Timestamp("2026-01-09")]
    assert selected.fold_id.tolist() == [2]
    selected, retrain = select_update_schedule(schedule, pd.Timestamp("2026-01-09"))
    assert retrain is False
    assert selected.fold_id.tolist() == [2]


def test_training_waits_for_requested_lag_when_target_uses_earlier_fallback():
    prepared = pd.DataFrame({
        "reference_date": pd.to_datetime(["2025-11-01", "2025-10-31"]),
        "report_date": pd.to_datetime(["2025-11-26", "2025-11-25"]),
        # Both fallback values have already been reported, but their stored
        # target date is t + 28; only the second has reached it.
        "target_date": pd.to_datetime(["2025-11-29", "2025-11-28"]),
        "value": [10., 20.],
    })
    selected = rolling_completed_training(
        prepared, pd.Timestamp("2025-11-28"), training_days=180)
    assert selected.reference_date.tolist() == [pd.Timestamp("2025-10-31")]


def test_empty_ledger_initializes_latest_available_origin():
    schedule = pd.DataFrame({
        "test_date": pd.to_datetime(["2026-01-02", "2026-01-09"]),
        "experiment_end_date": pd.to_datetime(["2026-01-09", "2026-01-16"]),
    })
    selected, retrain = select_update_schedule(schedule, None)
    assert retrain is True
    assert selected.test_date.tolist() == [pd.Timestamp("2026-01-09")]
    assert selected.fold_id.tolist() == [2]


def test_deployment_origin_is_persisted_separately_from_forecasts(tmp_path):
    assert read_deployment_origin(tmp_path) is None
    path = write_deployment_origin(tmp_path, pd.Timestamp("2026-01-09"))
    assert path.name == "deployment_state.json"
    assert read_deployment_origin(tmp_path) == pd.Timestamp("2026-01-09")
    assert not (tmp_path / "forecast_ledger.csv.gz").exists()


def test_incremental_cases_are_unresolved_and_not_previously_issued():
    cases = pd.DataFrame({
        "fold": [1, 1, 1], "cutoff": pd.to_datetime(["2026-01-02"] * 3),
        "geo_value": ["a"] * 3,
        "reference_date": pd.to_datetime(["2026-01-01"] * 3),
        "report_date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"]),
        "target_date": pd.to_datetime(["2026-01-29"] * 3),
        "lag": [1, 2, 3], "truth": [None, None, 14.], "Null": [10., 11., 12.],
    })
    ledger = forecast()
    selected = exclude_issued_cases(cases, ledger)
    assert selected.report_date.tolist() == [pd.Timestamp("2026-01-03")]


def test_truth_refresh_scores_ledger_without_regenerating_predictions(tmp_path):
    append_forecast_ledger(forecast(), tmp_path)
    prepared = pd.DataFrame({
        "geo_value": ["a"], "reference_date": pd.to_datetime(["2026-01-01"]),
        "report_date": pd.to_datetime(["2026-01-29"]),
        "target_date": pd.to_datetime(["2026-01-29"]),
        "target": [float(np.log1p(12.))],
    })
    prepared["target_lag"] = 28
    result = refresh_forecast_ledger_truth(
        prepared, "target", "log_value", tmp_path, as_of_date=pd.Timestamp("2026-01-29"))
    saved = read_forecast_ledger(tmp_path)
    assert result["resolved_rows"] == 1
    assert saved.loc[0, "prediction"] == 11
    assert saved.loc[0, "truth"] == pytest.approx(12)
    assert saved.loc[0, "absolute_error"] == pytest.approx(1)
