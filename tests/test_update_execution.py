"""An update runs one origin and records every forecast it issues, failed or not."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autodelphirf.models.delphirf_backend import train_delphirf
from autodelphirf.workflow.operational import (append_forecast_ledger, exclude_issued_cases,
                                               read_forecast_ledger)


def backend(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    raw = tmp_path / "raw.csv"
    raw.write_text("reference_date,report_date,value,geo_value\n")
    (prepared_dir / "preparation.json").write_text(json.dumps({
        "raw_archive": str(raw), "resolved_preprocessing": {
            "reference_col": "reference_date", "report_col": "report_date",
            "value_cols": ["value"], "value_type": "count", "lag_terms": [1, 7]}}))
    captured = {}

    def fake_run(command, check=False):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("autodelphirf.models.delphirf_backend.rscript_executable", lambda: "Rscript")
    monkeypatch.setattr("autodelphirf.models.delphirf_backend.subprocess.run", fake_run)
    config = SimpleNamespace(prepared_dir=prepared_dir, schedule_file="test_dates.csv")

    def run(schedule):
        train_delphirf(config, pd.DataFrame(), schedule, {"delphirf": "RevRoute DelphiRF"},
                       tmp_path / "predictions.csv.gz")
        return captured["command"]
    return run


def test_an_update_executes_only_its_selected_origin(tmp_path, monkeypatch):
    run = backend(tmp_path, monkeypatch)
    selected = pd.DataFrame({"test_date": ["2022-08-30"], "fold_id": [4]})
    command = run(selected)
    assert command[command.index("--fold") + 1] == "4"


def test_a_retrospective_run_executes_every_origin(tmp_path, monkeypatch):
    run = backend(tmp_path, monkeypatch)
    command = run(pd.DataFrame({"test_date": ["2022-06-01", "2022-07-01"]}))
    assert "--fold" not in command


def test_an_update_schedule_must_name_one_origin(tmp_path, monkeypatch):
    run = backend(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exactly one origin"):
        run(pd.DataFrame({"test_date": ["2022-07-31", "2022-08-30"], "fold_id": [3, 4]}))


def issued(method, prediction, available):
    return {"dataset": "d", "method": method, "cutoff": "2022-08-30", "geo_value": "aa",
            "reference_date": "2022-09-10", "report_date": "2022-09-12",
            "target_date": "2022-09-24", "lag": 2, "prediction": prediction,
            "model_available": available, "truth": np.nan}


def test_a_failed_forecast_is_recorded_and_never_issued_again(tmp_path):
    forecasts = pd.DataFrame([issued("baseline_null", 10., True),
                              issued("delphirf", np.nan, False)])
    append_forecast_ledger(forecasts, tmp_path)
    ledger = read_forecast_ledger(tmp_path)
    failed = ledger[ledger.method.eq("delphirf")]
    assert len(failed) == 1
    assert not failed.model_available.iloc[0] and np.isnan(failed.prediction.iloc[0])
    case = pd.DataFrame({"cutoff": ["2022-08-30"], "geo_value": ["aa"],
                         "reference_date": ["2022-09-10"], "report_date": ["2022-09-12"],
                         "lag": [2], "truth": [np.nan]})
    assert exclude_issued_cases(case, ledger).empty
