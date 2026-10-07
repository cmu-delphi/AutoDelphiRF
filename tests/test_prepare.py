"""The preparation stage: a user's raw archive, before DelphiRF is called.

Everything here is the Python half -- validating the user's column mapping,
diagnosing the archive, and resolving what to ask ``data_preprocessing()``
for. The R half is not exercised (it needs R and DelphiRF installed); what is
pinned instead is that the arguments handed to it are correct and that a
missing R install produces installation guidance rather than a traceback.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autodelphirf import prepare
from autodelphirf.prepare import (PreparationError, build_spec, diagnose_archive,
                                     read_archive, write_dataset_config)

from conftest import archive_frame


@pytest.fixture
def archive_csv(tmp_path):
    path = tmp_path / "archive.csv"
    archive_frame().to_csv(path, index=False)
    return path


# --- reading the archive --------------------------------------------------

def test_missing_archive_names_the_path(tmp_path):
    with pytest.raises(PreparationError, match="raw archive not found"):
        read_archive(tmp_path / "absent.csv")


def test_gzipped_csv_archives_read(tmp_path):
    path = tmp_path / "archive.csv.gz"
    archive_frame(days=20, max_lag=5).to_csv(path, index=False)
    assert len(read_archive(path)) > 0


def test_missing_columns_are_named_with_what_is_actually_present(tmp_path):
    path = tmp_path / "archive.csv"
    pd.DataFrame({"loc": ["aa"], "when": ["2024-01-01"], "v": [1.0]}).to_csv(path, index=False)
    with pytest.raises(PreparationError) as error:
        diagnose_archive(path, geo_col="loc", reference_col="reference_date",
                         report_col="report_date", value_cols=("v",), value_type="count")
    message = str(error.value)
    assert "reference_date" in message and "report_date" in message
    assert "when" in message          # what the file does have
    assert "--reference-col" in message   # how to fix it


# --- column-mapping validation -------------------------------------------

def test_fraction_signal_requires_two_value_columns(tmp_path, archive_csv):
    with pytest.raises(PreparationError, match="exactly two value columns"):
        build_spec("d", archive_csv, tmp_path / "out", value_type="fraction",
                   value_cols=("value",))


def test_count_signal_rejects_two_value_columns(tmp_path, archive_csv):
    with pytest.raises(PreparationError, match="exactly one --value-col"):
        build_spec("d", archive_csv, tmp_path / "out", value_type="count",
                   value_cols=("numerator", "denominator"))


def test_unknown_value_type_is_rejected(tmp_path, archive_csv):
    with pytest.raises(PreparationError, match="value_type must be"):
        build_spec("d", archive_csv, tmp_path / "out", value_type="ratio")


def test_calendar_endpoints_must_be_given_together(tmp_path, archive_csv):
    with pytest.raises(PreparationError, match="must be given together"):
        build_spec("d", archive_csv, tmp_path / "out", start_date="2022-06-01")
    with pytest.raises(PreparationError, match="must be given together"):
        build_spec("d", archive_csv, tmp_path / "out", end_date="2022-09-01")


def test_invalid_temporal_resolution_is_rejected(tmp_path, archive_csv):
    with pytest.raises(PreparationError, match="daily.*weekly"):
        build_spec("d", archive_csv, tmp_path / "out", temporal_resol="hourly")


# --- diagnosis ------------------------------------------------------------

def test_diagnosis_recovers_the_archive_shape(archive_csv):
    diagnosis = diagnose_archive(archive_csv, geo_col="geo_value",
                                 reference_col="reference_date", report_col="report_date",
                                 value_cols=("value",), value_type="count")
    assert diagnosis["n_locations"] == 2
    assert diagnosis["temporal_resolution"] == "daily"
    assert diagnosis["resolved_target_lag"] > 0


def test_single_location_archive_is_diagnosed_not_rejected(tmp_path):
    """--no-geo is a supported shape; routing is degenerate, not broken."""
    path = tmp_path / "one.csv"
    archive_frame(days=120, locations=("only",)).drop(columns=["geo_value"]).to_csv(
        path, index=False)
    diagnosis = diagnose_archive(path, geo_col=None, reference_col="reference_date",
                                 report_col="report_date", value_cols=("value",),
                                 value_type="count")
    assert diagnosis["n_locations"] == 1
    assert any("single-location" in note for note in diagnosis["notes"])


def test_fraction_signals_are_diagnosed_on_the_ratio_not_the_numerator(tmp_path):
    """A rate's revision behaviour is the rate's, not its numerator's."""
    frame = archive_frame(days=120, locations=("aa",))
    # A numerator that never settles, over a denominator that tracks it: the
    # RATIO is stable from lag 0, so diagnosing the numerator alone would
    # report a long target lag where the modelled quantity needs a short one.
    frame["numerator"] = frame.value
    frame["denominator"] = frame.value * 2
    path = tmp_path / "fraction.csv"
    frame.to_csv(path, index=False)
    ratio = diagnose_archive(path, geo_col="geo_value", reference_col="reference_date",
                             report_col="report_date", value_cols=("numerator", "denominator"),
                             value_type="fraction")
    numerator_only = diagnose_archive(path, geo_col="geo_value", reference_col="reference_date",
                                      report_col="report_date", value_cols=("numerator",),
                                      value_type="count")
    assert ratio["resolved_target_lag"] < numerator_only["resolved_target_lag"]


# --- what gets handed to DelphiRF -----------------------------------------

def test_diagnosed_settings_are_used_when_nothing_is_overridden(tmp_path, archive_csv):
    spec = build_spec("d", archive_csv, tmp_path / "out")
    assert spec.overrides == ()
    assert spec.ref_lag == spec.diagnosis["resolved_target_lag"]
    assert spec.temporal_resol == spec.diagnosis["temporal_resolution"]
    assert list(spec.lag_terms) == spec.diagnosis["reference_axis_feature_lags"]


def test_each_override_is_recorded_so_provenance_survives(tmp_path, archive_csv):
    spec = build_spec("d", archive_csv, tmp_path / "out", temporal_resol="weekly",
                      lag_terms=(7, 14), training_days=365, smoothed=False)
    assert set(spec.overrides) >= {"temporal_resol", "lag_terms", "training_days", "smoothed"}
    assert spec.temporal_resol == "weekly"
    assert spec.lag_terms == (7, 14)
    assert spec.training_days == 365
    assert spec.smoothed is False


def test_a_user_target_lag_is_reconciled_with_the_diagnosis_rather_than_taken_blindly(
        tmp_path, archive_csv):
    """The longer target lag wins by default; the disagreement is kept on record."""
    spec = build_spec("d", archive_csv, tmp_path / "out", target_lag=120)
    assert "target_lag" in spec.overrides
    assert spec.diagnosis["user_target_lag"] == 120
    assert spec.ref_lag == 120                     # larger than diagnosed -> kept
    assert spec.diagnosis["target_lag_confirmation_prompt"]


def test_the_target_column_follows_the_smoothing_choice(tmp_path, archive_csv):
    """The schedule must name the column data_preprocessing actually writes."""
    smoothed = build_spec("d", archive_csv, tmp_path / "out", smoothed=True)
    unsmoothed = build_spec("d", archive_csv, tmp_path / "out", smoothed=False)
    assert smoothed.target_column == "log_value_target_7dav"
    assert unsmoothed.target_column == "log_value_target"


def test_weekly_streams_get_no_weekday_encoding(tmp_path, archive_csv):
    """There is no within-week reporting structure in a weekly stream to model."""
    weekly = build_spec("d", archive_csv, tmp_path / "out", temporal_resol="weekly")
    daily = build_spec("d", archive_csv, tmp_path / "out", temporal_resol="daily")
    assert weekly.weekday_groups == {}
    assert daily.weekday_groups


def test_weekday_groups_are_model_settings_not_preprocessing_arguments(tmp_path, archive_csv):
    spec = build_spec("d", archive_csv, tmp_path / "out",
                      temporal_resol="daily", weekday_groups={"Weekend": ["Sat", "Sun"]})
    payload = spec.to_r_spec()
    assert payload["weekday_groups"] == {"Weekend": ["Sat", "Sun"]}
    assert "onehot_weekdays" not in payload


def test_pre_diagnosis_chooses_model_groups_from_observed_report_weekdays(
        tmp_path, archive_csv, monkeypatch):
    diagnosis = prepare.diagnose_archive(archive_csv, geo_col="geo_value",
                                         reference_col="reference_date",
                                         report_col="report_date", value_cols=("value",),
                                         value_type="count")
    diagnosis["temporal_resolution"] = "daily"
    diagnosis["report_axis_weekdays"] = {"Wed": 0.5, "Fri": 0.5}
    monkeypatch.setattr(prepare, "diagnose_archive", lambda *args, **kwargs: diagnosis)
    spec = build_spec("d", archive_csv, tmp_path / "out")
    assert spec.weekday_groups == {"Fri": ["Fri"]}


def test_weekday_groups_must_be_disjoint_and_leave_a_reference_day(tmp_path, archive_csv):
    with pytest.raises(PreparationError, match="overlap"):
        build_spec("d", archive_csv, tmp_path / "out", temporal_resol="daily",
                   weekday_groups={"A": ["Mon", "Tue"], "B": ["Tue"]})
    with pytest.raises(PreparationError, match="reference category"):
        build_spec("d", archive_csv, tmp_path / "out", temporal_resol="daily",
                   weekday_groups={day: [day] for day in
                                   ("Mon", "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun")})


def test_the_r_spec_is_json_serializable_with_no_python_objects(tmp_path, archive_csv):
    """It crosses a process boundary as JSON; a Path or numpy scalar breaks it."""
    spec = build_spec("d", archive_csv, tmp_path / "out")
    payload = json.loads(json.dumps(spec.to_r_spec()))
    assert payload["raw_csv"].endswith("archive.csv")
    assert isinstance(payload["ref_lag"], int)
    assert all(isinstance(term, int) for term in payload["lag_terms"])


# --- the generated dataset config ----------------------------------------

def test_the_written_config_loads_and_points_at_the_triangle(tmp_path, archive_csv):
    from autodelphirf.config import load_dataset_config

    spec = build_spec("mine", archive_csv, tmp_path / "work" / "mine")
    config_path = write_dataset_config(spec, tmp_path / "work" / "mine.json",
                                       ("baseline_null", "red"))
    config = load_dataset_config(config_path)
    assert config.name == "mine"
    assert config.prepared_dir == spec.output_dir
    assert config.prediction_layers == ("baseline_null", "red")
    assert config.revision_profile == {
        "weekday_groups": spec.weekday_groups,
        "lag_terms": list(spec.lag_terms),
    }


def test_generated_config_paths_are_relative_so_the_directory_can_be_moved(
        tmp_path, archive_csv):
    import shutil

    from autodelphirf.config import load_dataset_config

    work = tmp_path / "work"
    spec = build_spec("mine", archive_csv, work / "mine")
    spec.output_dir.mkdir(parents=True)
    config_path = write_dataset_config(spec, work / "mine.json", ("red",))
    assert not json.loads(config_path.read_text())["prepared_dir"].startswith("/")

    moved = tmp_path / "moved"
    shutil.move(str(work), str(moved))
    config = load_dataset_config(moved / "mine.json")
    assert config.prepared_dir == moved / "mine"


# --- the R boundary -------------------------------------------------------

def test_a_missing_rscript_gives_installation_guidance_not_a_traceback(monkeypatch):
    monkeypatch.delenv("AUTODELPHIRF_RSCRIPT", raising=False)
    monkeypatch.setattr(prepare.shutil, "which", lambda _: None)
    with pytest.raises(PreparationError) as error:
        prepare.rscript_executable()
    message = str(error.value)
    assert "install R" in message
    assert "DelphiRF" in message
    # ...and the way out for someone who already has a prepared triangle.
    assert "autodelphirf run --config" in message


def test_an_explicit_rscript_is_honoured(monkeypatch):
    monkeypatch.setenv("AUTODELPHIRF_RSCRIPT", "/opt/R/bin/Rscript")
    assert prepare.rscript_executable() == "/opt/R/bin/Rscript"


def test_a_failing_r_bridge_is_reported_as_a_preparation_error(tmp_path, archive_csv, monkeypatch):
    """The R traceback goes to the terminal; Python adds one actionable line."""
    class Failed:
        returncode = 3

    monkeypatch.setattr(prepare, "rscript_executable", lambda: "Rscript")
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **k: Failed())
    spec = build_spec("d", archive_csv, tmp_path / "out")
    with pytest.raises(PreparationError, match="exit 3"):
        prepare.run_r_bridge(spec)


def test_the_spec_file_is_cleaned_up_even_when_r_fails(tmp_path, archive_csv, monkeypatch):
    written = {}

    class Failed:
        returncode = 1

    def capture(command, **kwargs):
        written["spec"] = prepare.Path(command[2])
        assert written["spec"].is_file()      # readable while R runs
        return Failed()

    monkeypatch.setattr(prepare, "rscript_executable", lambda: "Rscript")
    monkeypatch.setattr(prepare.subprocess, "run", capture)
    spec = build_spec("d", archive_csv, tmp_path / "out")
    with pytest.raises(PreparationError):
        prepare.run_r_bridge(spec)
    assert not written["spec"].exists()


def test_the_r_bridge_script_ships_and_calls_delphirf(tmp_path, archive_csv):
    """The bridge must call DelphiRF, not reimplement its feature engineering."""
    from autodelphirf.resources import resource_path

    source = resource_path("prepare_triangle.R").read_text()
    assert "data_preprocessing" in source
    assert "DelphiRF" in source
    # It must fail loudly when DelphiRF is absent rather than substituting
    # its own preprocessing.
    assert "install_github" in source
