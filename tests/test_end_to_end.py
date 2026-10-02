"""End-to-end: a raw archive in, a report out.

This is the only suite that needs R and DelphiRF, and it is skipped -- not
failed -- when they are absent, because the rest of the package is usable
without them. It is also the slowest: one full rolling-origin replay.

Run it explicitly with:

    pytest tests/test_end_to_end.py

and set AUTODELPHIRF_DELPHIRF_DIR to a DelphiRF source checkout if the installed
DelphiRF predates the ``target_lag_*_tolerance`` / ``onehot_weekdays``
arguments.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from autodelphirf import cli

from conftest import archive_frame

pytestmark = pytest.mark.slow


def delphirf_available() -> bool:
    """True when Rscript exists and can load a DelphiRF with the needed API."""
    executable = os.environ.get("AUTODELPHIRF_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        return False
    source_dir = os.environ.get("AUTODELPHIRF_DELPHIRF_DIR")
    if source_dir:
        script = (f'suppressMessages(pkgload::load_all("{source_dir}", quiet=TRUE)); '
                  'cat("onehot_weekdays" %in% names(formals(data_preprocessing)))')
    else:
        script = ('suppressMessages(library(DelphiRF)); '
                  'cat("onehot_weekdays" %in% names(formals(DelphiRF::data_preprocessing)))')
    try:
        result = subprocess.run([executable, "-e", script], capture_output=True,
                                text=True, timeout=180)
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and "TRUE" in result.stdout


requires_delphirf = pytest.mark.skipif(
    not delphirf_available(),
    reason="needs R with a current DelphiRF (set AUTODELPHIRF_DELPHIRF_DIR for a source checkout)")


@requires_delphirf
def test_one_command_turns_a_raw_archive_into_a_report(tmp_path):
    """The documented one-liner, run exactly as a new user would run it."""
    archive = tmp_path / "archive.csv"
    # Slow enough revision that a multi-week horizon is diagnosable, and long
    # enough that the training window plus several test origins fit.
    archive_frame(days=300, max_lag=60, halflife=9.0).to_csv(archive, index=False)
    work = tmp_path / "work"

    assert cli.main(["run", "--archive", str(archive), "--name", "example",
                     "--out", str(work)]) == 0

    prepared = work / "example"
    results = work / "results"

    # The preparation stage produced one triangle per location plus its schedule.
    assert sorted(p.stem for p in prepared.glob("*.parquet")) == ["aa", "bb"]
    assert (prepared / "test_dates.csv").is_file()
    assert (prepared / "inventory.csv").is_file()

    # Provenance: what was diagnosed, and what the user changed.
    provenance = json.loads((prepared / "preparation.json").read_text())
    assert provenance["user_overrides"] == []
    assert provenance["resolved_preprocessing"]["ref_lag"] > 0
    assert provenance["diagnosis"]["n_locations"] == 2

    # The generated config round-trips.
    from autodelphirf.config import load_dataset_config
    config = load_dataset_config(work / "example.json")
    assert config.prepared_dir == prepared

    # The run produced its predictions, its evaluation, and its report.
    assert (results / "predictions.csv.gz").is_file()
    assert (results / "diagnostics" / "summary.json").is_file()
    assert (results / "run_manifest.json").is_file()
    report = results / "report" / "report.html"
    assert report.is_file() and report.stat().st_size > 1000
    assert list((results / "report" / "tables").glob("*.csv"))

    # Both requested methods were actually scored.
    import pandas as pd
    predictions = pd.read_csv(results / "predictions.csv.gz")
    assert set(predictions.method) >= {"baseline_null", "red"}
    assert predictions.absolute_error.notna().any()


@requires_delphirf
def test_diagnosis_only_validates_without_fitting(tmp_path):
    """The fast first step a user is told to run before a full replay."""
    archive = tmp_path / "archive.csv"
    archive_frame(days=200, max_lag=45).to_csv(archive, index=False)
    work = tmp_path / "work"

    assert cli.main(["prepare", "--archive", str(archive), "--name", "d",
                     "--out", str(work)]) == 0
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
                     "--triangle-format", "csv"]) == 0
    prepared = work / "d"
    assert list(prepared.glob("*.csv.gz"))
    assert not list(prepared.glob("*.parquet"))

    from autodelphirf.config import load_dataset_config
    from autodelphirf.io import load_prepared_triangles
    config = load_dataset_config(work / "d.json")
    frame = load_prepared_triangles(config.prepared_dir)
    assert len(frame) > 0
    assert {"geo_value", "reference_date", "report_date", "lag"} <= set(frame.columns)
