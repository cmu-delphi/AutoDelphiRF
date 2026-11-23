"""M7: one effective weekday grouping, detected or user-specified, used by training."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from autodelphirf.app import cli
from autodelphirf.models.delphirf_backend import train_delphirf
from autodelphirf.pre_diagnosis import preparation
from autodelphirf.pre_diagnosis.archive import diagnose_raw_archive
from autodelphirf.pre_diagnosis.preparation import (PreparationError, build_spec,
                                                    parse_weekday_groups)
from autodelphirf.workflow.run import run_pipeline

from conftest import archive_frame
from test_target_lag_diagnosis import prepared_only_dataset


@pytest.fixture
def daily_archive(tmp_path):
    path = tmp_path / "archive.csv"
    archive_frame(max_lag=70).to_csv(path, index=False)
    return path


# --- parsing -----------------------------------------------------------------

def test_the_chapter_2_grouping_is_parsed():
    assert parse_weekday_groups({"monday": [1], "weekend": [6, 7]}) == {
        "monday": ["Mon"], "weekend": ["Sat", "Sun"]}


def test_the_wednesday_friday_grouping_is_parsed():
    assert parse_weekday_groups({"wednesday": [3], "friday": [5]}) == {
        "wednesday": ["Wed"], "friday": ["Fri"]}


def test_unassigned_weekdays_form_the_reference_category():
    groups = parse_weekday_groups({"wednesday": [3], "friday": [5]})
    assigned = {day for members in groups.values() for day in members}
    assert set(preparation.WEEKDAYS) - assigned == {"Mon", "Tue", "Thurs", "Sat", "Sun"}


def test_group_names_are_labels_not_keywords():
    assert parse_weekday_groups({"group_a": [1]}) == {"group_a": ["Mon"]}
    assert parse_weekday_groups({"Mon": [6, 7]}) == {"Mon": ["Sat", "Sun"]}


@pytest.mark.parametrize("groups, message", [
    ({"a": [1, 2], "b": [2]}, "assigned to both"),
    ({"a": [1, 1]}, "assigned to both"),
    ({"a": [0]}, "1 \\(Monday\\) through 7"),
    ({"a": [8]}, "1 \\(Monday\\) through 7"),
    ({"a": [1.0]}, "1 \\(Monday\\) through 7"),
    ({"a": ["1"]}, "1 \\(Monday\\) through 7"),
    ({"a": [True]}, "1 \\(Monday\\) through 7"),
    ({"a": []}, "at least one"),
    ({"a": 1}, "at least one"),
    ({"": [1]}, "invalid weekday group name"),
    ({"1st": [1]}, "invalid weekday group name"),
    ({"has space": [1]}, "invalid weekday group name"),
    ({"W1": [1]}, "reserved"),
    ({f"d{day}": [day] for day in range(1, 8)}, "reference category"),
    ([1, 2], "must be an object"),
])
def test_invalid_groups_are_rejected(groups, message):
    with pytest.raises(PreparationError, match=message):
        parse_weekday_groups(groups)


# --- resolution ----------------------------------------------------------------

def uniform_report_weekdays(monkeypatch, daily_archive):
    """Make detection return the Chapter 2 grouping, so an override is visible."""
    original = preparation.diagnose_archive

    def diagnosed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["report_axis_weekdays"] = {day: 1 / 7 for day in
                                          ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")}
        return result
    monkeypatch.setattr(preparation, "diagnose_archive", diagnosed)


@pytest.mark.parametrize("given", [None])
def test_omitted_or_null_groups_use_automatic_detection(given, daily_archive, tmp_path,
                                                        monkeypatch):
    uniform_report_weekdays(monkeypatch, daily_archive)
    spec = build_spec("d", daily_archive, tmp_path / "out", target_lag=21,
                      weekday_groups=given)
    assert spec.weekday_groups == {"Mon": ["Mon"], "Weekends": ["Sat", "Sun"]}
    assert spec.weekday_groups_source == "automatic"
    assert "weekday_groups" not in spec.overrides


def test_a_raw_config_with_null_groups_uses_automatic_detection(daily_archive, tmp_path):
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({"name": "d", "archive": str(daily_archive),
                                  "preprocessing": {"target_lag": 21, "weekday_groups": None}}))
    arguments = cli._load_raw_config(cli.build_parser().parse_args(
        ["prepare", "--raw-config", str(config)]))
    assert arguments.weekday_groups is None
    assert cli._spec_from_arguments(arguments).weekday_groups_source == "automatic"


def test_manual_groups_override_the_detected_ones(daily_archive, tmp_path, monkeypatch):
    uniform_report_weekdays(monkeypatch, daily_archive)
    spec = build_spec("d", daily_archive, tmp_path / "out", target_lag=21,
                      weekday_groups={"wednesday": [3], "friday": [5]})
    assert spec.weekday_groups == {"wednesday": ["Wed"], "friday": ["Fri"]}
    assert spec.weekday_groups_source == "manual"
    assert "weekday_groups" in spec.overrides


def test_manual_groups_do_not_need_the_detection(daily_archive, tmp_path, monkeypatch):
    def unused(*args, **kwargs):
        raise AssertionError("detection must not choose groups the user supplied")
    monkeypatch.setattr(preparation, "weekday_groups_from_share", unused)
    spec = build_spec("d", daily_archive, tmp_path / "out", target_lag=21,
                      weekday_groups={"monday": [1], "weekend": [6, 7]})
    assert spec.weekday_groups == {"monday": ["Mon"], "weekend": ["Sat", "Sun"]}


def test_invalid_manual_groups_never_fall_back_to_detection(daily_archive, tmp_path):
    with pytest.raises(PreparationError, match="assigned to both"):
        build_spec("d", daily_archive, tmp_path / "out", target_lag=21,
                   weekday_groups={"a": [1], "b": [1]})


def test_a_raw_config_supplies_manual_groups(daily_archive, tmp_path):
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({
        "name": "d", "archive": str(daily_archive),
        "preprocessing": {"target_lag": 21, "weekday_groups": {"monday": [1], "weekend": [6, 7]}}}))
    arguments = cli._load_raw_config(cli.build_parser().parse_args(
        ["prepare", "--raw-config", str(config)]))
    spec = cli._spec_from_arguments(arguments)
    assert spec.weekday_groups == {"monday": ["Mon"], "weekend": ["Sat", "Sun"]}
    assert spec.to_r_spec()["weekday_groups_source"] == "manual"


def test_unchanged_releases_count_in_the_report_calendar():
    # One reference date, re-reported every day for two weeks with no change.
    reference = pd.Timestamp("2024-01-01")
    raw = pd.DataFrame({"geo_value": "aa", "reference_date": reference,
                        "report_date": pd.date_range(reference, periods=14), "value": 5.0})
    report = diagnose_raw_archive(raw)
    assert len(report.report_axis_weekdays) == 7
    assert sum(report.report_axis_weekdays.values()) == pytest.approx(1.0, abs=1e-3)


# --- training uses the one effective grouping -------------------------------------

def test_training_receives_the_recorded_groups(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    raw = tmp_path / "raw.csv"
    raw.write_text("geo_value,reference_date,report_date,value\n")
    groups = {"wednesday": ["Wed"], "friday": ["Fri"]}
    (prepared / "preparation.json").write_text(json.dumps({"raw_archive": str(raw),
        "resolved_preprocessing": {"reference_col": "reference_date", "report_col": "report_date",
                                   "value_cols": ["value"], "value_type": "count",
                                   "lag_terms": [7, 14], "weekday_groups": groups,
                                   "weekday_groups_source": "manual"}}))
    captured = {}

    def fake_run(command, check=False):
        captured["command"] = command
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("autodelphirf.models.delphirf_backend.rscript_executable", lambda: "Rscript")
    monkeypatch.setattr("autodelphirf.models.delphirf_backend.subprocess.run", fake_run)
    config = SimpleNamespace(prepared_dir=prepared, schedule_file="test_dates.csv")
    train_delphirf(config, pd.DataFrame(), pd.DataFrame(), {"delphirf": "RevRoute DelphiRF"},
                   tmp_path / "predictions.csv.gz")
    command = captured["command"]
    assert json.loads(command[command.index("--weekdays-json") + 1]) == groups


def test_a_conflicting_revision_profile_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = prepared_only_dataset(tmp_path, target_lag=7)
    path = tmp_path / "prepared.json"
    payload = json.loads(path.read_text())
    payload["revision_profile"] = {"weekday_groups": {"Mon": ["Mon"]}}
    path.write_text(json.dumps(payload))
    from autodelphirf.core.config import load_dataset_config
    with pytest.raises(ValueError, match="revision_profile in the dataset JSON"):
        run_pipeline(load_dataset_config(path), diagnosis_only=True)


def test_an_agreeing_legacy_revision_profile_is_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prepared_only_dataset(tmp_path, target_lag=7)
    path = tmp_path / "prepared.json"
    payload = json.loads(path.read_text())
    payload["revision_profile"] = {"weekday_groups": {}, "lag_terms": []}
    path.write_text(json.dumps(payload))
    from autodelphirf.core.config import load_dataset_config
    summary = run_pipeline(load_dataset_config(path), diagnosis_only=True)
    assert summary["weekday_groups_used_for_modeling"] == {}


def rscript():
    return os.environ.get("AUTODELPHIRF_RSCRIPT") or shutil.which("Rscript")


@pytest.mark.skipif(rscript() is None, reason="needs R")
def test_reference_and_report_weekday_features_follow_the_groups(tmp_path):
    """A label that is also a weekday name cannot corrupt another group."""
    from autodelphirf.core.resources import resource_path
    source = resource_path("train_delphirf.R").read_text().splitlines()
    start = next(i for i, line in enumerate(source) if line.startswith("add_weekday_groups <- function"))
    end = next(i for i in range(start + 1, len(source)) if source[i] == "}")
    function = "\n".join(source[start:end + 1])
    script = tmp_path / "groups.R"
    script.write_text(function + '''
days <- c("Mon", "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun")
frame <- data.frame(row = 1:7)
for (i in seq_along(days)) {
  frame[[paste0(days[i], "_ref")]] <- as.integer(seq_len(7) == i)
  frame[[paste0(days[i], "_issue")]] <- as.integer(seq_len(7) == ((i %% 7) + 1))
}
out <- add_weekday_groups(frame, list(Mon = c("Sat", "Sun"), monday = c("Mon")))
cat(out$Mon_ref, "|", out$monday_ref, "|", out$monday_issue, "\\n")
''')
    result = subprocess.run([rscript(), str(script)], capture_output=True, text=True, check=True)
    weekend_ref, monday_ref, monday_issue = (part.split() for part in result.stdout.split("|"))
    assert weekend_ref == ["0", "0", "0", "0", "0", "1", "1"]
    assert monday_ref == ["1", "0", "0", "0", "0", "0", "0"]      # read before overwrite
    # The report-date Monday column marks row 2 in this frame: the axes stay apart.
    assert monday_issue == ["0", "1", "0", "0", "0", "0", "0"]
