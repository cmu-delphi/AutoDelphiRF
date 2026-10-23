"""End-to-end: a raw archive in, a report out.

This is the only suite that needs R and DelphiRF, and it is skipped -- not
failed -- when they are absent, because the rest of the package is usable
without them. It is also the slowest: one full rolling-origin replay.

Run it explicitly with:

    pytest tests/test_end_to_end.py

DelphiRF must be installed in R's library as the README describes; a source
checkout is never loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from autodelphirf.app import cli

from conftest import archive_frame

pytestmark = pytest.mark.slow


def delphirf_available() -> bool:
    """True when Rscript exists and can load a DelphiRF with the needed API."""
    executable = os.environ.get("AUTODELPHIRF_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        return False
    required = ('c("ref_lag", "lagged_term_list", "value_type", "temporal_resol", '
                '"smoothed", "revision_start_lag")')
    script = ('suppressMessages(library(DelphiRF)); '
              f'cat(all({required} %in% '
              'names(formals(DelphiRF::data_preprocessing))) && '
              'all(c("model_backend", "working_scale") %in% '
              'names(formals(DelphiRF::revision_forecast))) && '
              'exists("create_weight_anchors", envir = asNamespace("DelphiRF")) && '
              '!"onehot_weekdays" %in% '
              'names(formals(DelphiRF:::create_params_list)))')
    try:
        result = subprocess.run([executable, "-e", script], capture_output=True,
                                text=True, timeout=180)
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and "TRUE" in result.stdout


requires_delphirf = pytest.mark.skipif(
    not delphirf_available(),
    reason="needs R with a current installed DelphiRF")


@requires_delphirf
def test_one_command_turns_a_raw_archive_into_a_report(tmp_path):
    """The documented one-liner, run exactly as a new user would run it."""
    archive = tmp_path / "archive.csv"
    # Slow enough revision that a multi-week target lag can be recommended, and long
    # enough that the training window plus several test origins fit.
    archive_frame(days=300, max_lag=60, halflife=9.0).to_csv(archive, index=False)
    work = tmp_path / "work"

    assert cli.main(["run", "--archive", str(archive), "--name", "example",
                     "--out", str(work), "--target-lag", "21"]) == 0

    prepared = work / "example"
    results = work / "results"

    # The preparation stage produced one triangle per location plus its schedule.
    assert sorted(p.stem for p in prepared.glob("*.parquet")) == ["aa", "bb"]
    assert (prepared / "test_dates.csv").is_file()
    assert (prepared / "inventory.csv").is_file()

    # Provenance: what was diagnosed, and what the user changed.
    provenance = json.loads((prepared / "preparation.json").read_text())
    assert provenance["user_overrides"] == ["target_lag"]
    assert provenance["resolved_preprocessing"]["ref_lag"] > 0
    assert provenance["diagnosis"]["n_locations"] == 2

    # The generated config round-trips.
    from autodelphirf.core.config import load_dataset_config
    config = load_dataset_config(work / "example.json")
    assert config.prepared_dir == prepared

    # The run produced its predictions, its evaluation, and its report.
    assert (results / "predictions.csv.gz").is_file()
    assert (results / "diagnostics" / "summary.json").is_file()
    assert (results / "run_manifest.json").is_file()
    assert (results / "resolved_config.json").is_file()
    assert (results / "assessment_inputs.json").is_file()
    deployment = json.loads((results / "deployment_state.json").read_text())
    schedule = pd.read_csv(prepared / "test_dates.csv")
    assert deployment["latest_trained_origin"] == schedule.test_date.max()
    report = results / "report" / "report.html"
    assert report.is_file() and report.stat().st_size > 1000
    assert list((results / "report" / "tables").glob("*.csv"))

    # Both requested methods were actually scored.
    predictions = pd.read_csv(results / "predictions.csv.gz")
    assert set(predictions.method) >= {"baseline_null", "delphirf"}
    assert predictions.absolute_error.notna().any()
    delphirf = predictions[
        predictions.method.eq("delphirf") & predictions.prediction_working.notna()]
    assert len(delphirf) > 0
    assert np.allclose(delphirf.prediction, np.expm1(delphirf.prediction_working))

    # A fresh process can restart at the saved prediction boundary without R/model fitting.
    reassessment = work / "independent-assessment"
    assert cli.main(["assess", "--results", str(results),
                     "--out", str(reassessment)]) == 0
    assessment_manifest = json.loads((reassessment / "assessment_manifest.json").read_text())
    assert assessment_manifest["model_fitting_performed"] is False
    assert assessment_manifest["rows"]["wide"] == len(
        pd.read_csv(results / "predictions_wide.csv.gz"))
    assert (reassessment / "report" / "report.html").is_file()


@requires_delphirf
def test_returning_user_update_preserves_forecasts_and_advances_deployment(tmp_path):
    """Every release is ingested; intervening ones reuse the deployed model."""
    archive = tmp_path / "archive.csv"
    work = tmp_path / "work"
    raw_config = tmp_path / "dataset.json"
    complete = archive_frame(days=180, max_lag=30, locations=("aa",), halflife=7.0)
    initial_as_of = pd.Timestamp("2022-05-15")
    intervening_release = pd.Timestamp("2022-05-20")
    retraining_release = pd.Timestamp("2022-06-14")
    # After initialization this synthetic stream has exactly two releases. The
    # test invokes update for both, just as an operational ingestion job must.
    operational = complete[
        complete.report_date.le(initial_as_of)
        | complete.report_date.isin([intervening_release, retraining_release])].copy()

    def publish(as_of):
        operational[operational.report_date.le(as_of)].to_csv(archive, index=False)

    def model_hashes(results):
        return {str(path.relative_to(results / "models")):
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (results / "models").rglob("*.rds")}

    publish(initial_as_of)
    raw_config.write_text(json.dumps({
        "name": "returning",
        "archive": str(archive),
        "work_dir": str(work),
        "columns": {
            "reference_date": "reference_date", "report_date": "report_date",
            "geo_value": "geo_value", "values": ["value"],
        },
        "value_type": "count",
        "preprocessing": {
            "target_lag": 14, "training_days": 60, "triangle_format": "csv",
        },
        "retraining": {
            "first_origin_date": str(initial_as_of.date()), "retrain_days": 30,
        },
        "model_methods": ["baseline_null", "delphirf"],
    }, indent=2))

    assert cli.main(["update", "--raw-config", str(raw_config),
                     "--as-of-date", str(initial_as_of.date())]) == 0
    results = work / "results"
    ledger_path = results / "forecast_ledger.csv.gz"
    before = pd.read_csv(ledger_path).sort_values(
        ["dataset", "method", "cutoff", "geo_value", "reference_date", "report_date", "lag"]
    ).reset_index(drop=True)
    first_origin = json.loads((results / "deployment_state.json").read_text())[
        "latest_trained_origin"]
    first_models = model_hashes(results)
    assert first_origin == str(initial_as_of.date())
    assert first_models

    # The next raw release arrives before retraining is due. It must append new
    # forecasts while reusing exactly the deployed model files.
    publish(intervening_release)
    assert cli.main(["update", "--raw-config", str(raw_config),
                     "--as-of-date", str(intervening_release.date())]) == 0

    middle = pd.read_csv(ledger_path)
    middle_origin = json.loads((results / "deployment_state.json").read_text())[
        "latest_trained_origin"]
    middle_manifest = json.loads((results / "run_manifest.json").read_text())
    assert middle_origin == first_origin
    assert middle_manifest["retraining_required"] is False
    assert middle_manifest["new_forecasts"] > 0
    assert model_hashes(results) == first_models

    # The following release is also the next scheduled retraining origin. New
    # models are persisted, without changing forecasts issued by either prior
    # release.
    publish(retraining_release)
    assert cli.main(["update", "--raw-config", str(raw_config),
                     "--as-of-date", str(retraining_release.date())]) == 0

    after = pd.read_csv(ledger_path)
    keys = ["dataset", "method", "cutoff", "geo_value", "reference_date", "report_date", "lag"]
    overlap = before.merge(after, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    middle_overlap = middle.merge(
        after, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    assert len(overlap) == len(before)
    assert len(middle_overlap) == len(middle)
    assert np.allclose(overlap.prediction_before, overlap.prediction_after, equal_nan=True)
    assert np.allclose(overlap.prediction_working_before,
                       overlap.prediction_working_after, equal_nan=True)
    assert np.allclose(middle_overlap.prediction_before,
                       middle_overlap.prediction_after, equal_nan=True)
    assert len(after) > len(middle) > len(before)
    assert after[keys].duplicated().sum() == 0
    second_origin = json.loads((results / "deployment_state.json").read_text())[
        "latest_trained_origin"]
    assert second_origin == str(retraining_release.date())
    assert pd.Timestamp(second_origin) > pd.Timestamp(first_origin)
    manifest = json.loads((results / "run_manifest.json").read_text())
    assert manifest["incremental_update"] is True
    assert manifest["raw_archive_update"]["changed"] is True
    assert manifest["new_forecasts"] > 0
    final_models = model_hashes(results)
    assert set(first_models).issubset(final_models)
    assert len(final_models) > len(first_models)

    # Repeating the same update is an idempotent no-op.
    unchanged = ledger_path.read_bytes()
    assert cli.main(["update", "--raw-config", str(raw_config)]) == 0
    assert ledger_path.read_bytes() == unchanged


@requires_delphirf
def test_diagnosis_only_validates_without_fitting(tmp_path):
    """The fast first step a user is told to run before a full replay."""
    archive = tmp_path / "archive.csv"
    archive_frame(days=200, max_lag=45).to_csv(archive, index=False)
    work = tmp_path / "work"

    assert cli.main(["prepare", "--archive", str(archive), "--name", "d",
                     "--out", str(work), "--target-lag", "19"]) == 0
    assert cli.main(["run", "--config", str(work / "d.json"), "--diagnosis-only"]) == 0

    results = work / "results"
    assert (results / "diagnostics" / "summary.json").is_file()
    # Nothing was fitted, so there are no predictions and no report.
    assert not (results / "predictions.csv.gz").exists()
    assert not (results / "report").exists()


@requires_delphirf
def test_a_csv_triangle_runs_the_same_as_a_parquet_one(tmp_path):
    """--triangle-format csv is the escape hatch for hosts without Arrow."""
    archive = tmp_path / "archive.csv"
    archive_frame(days=200, max_lag=45).to_csv(archive, index=False)
    work = tmp_path / "work"

    assert cli.main(["prepare", "--archive", str(archive), "--name", "d", "--out", str(work),
                     "--triangle-format", "csv", "--target-lag", "19"]) == 0
    prepared = work / "d"
    assert list(prepared.glob("*.csv.gz"))
    assert not list(prepared.glob("*.parquet"))

    from autodelphirf.core.config import load_dataset_config
    from autodelphirf.core.io import load_prepared_triangles
    config = load_dataset_config(work / "d.json")
    frame = load_prepared_triangles(config.prepared_dir)
    assert len(frame) > 0
    assert {"geo_value", "reference_date", "report_date", "lag"} <= set(frame.columns)
