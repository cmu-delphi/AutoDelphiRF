"""The ``autodelphirf`` command line.

The CLI is the whole user-facing surface for anyone who never imports the
package, so these tests cover argument validation, the exit statuses a script
or CI job would branch on, and the parts of the output a user reads. Anything
needing R is exercised up to the R boundary and no further.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autodelphirf.app import cli
from autodelphirf.core.registry import PREDICTION_LAYERS

from conftest import archive_frame


@pytest.fixture
def archive_csv(tmp_path):
    path = tmp_path / "archive.csv"
    archive_frame(days=120, max_lag=30).to_csv(path, index=False)
    return path


# --- parser structure -----------------------------------------------------

def test_no_subcommand_exits_with_usage(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "diagnose" in capsys.readouterr().err


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    from autodelphirf import __version__
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("command", ["diagnose", "prepare", "run", "update", "assess", "init"])
def test_every_subcommand_has_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip()


def test_readme_raw_example_names_the_repository_archive_and_update_workspace():
    root = Path(__file__).parents[1]
    example = json.loads((root / "examples" / "my_archive.json").read_text())
    assert (root / "my_archive.csv").is_file()
    assert example["archive"] == "../my_archive.csv"
    assert example["work_dir"] == "../autodelphirf_work/readme_example"
    assert example["preprocessing"]["target_lag"] == 28


# --- layer validation -----------------------------------------------------

def test_an_unknown_layer_is_rejected_with_the_available_ones(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["prepare", "--archive", "a.csv", "--name", "d", "--layers", "not_a_layer"])
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "not_a_layer" in error
    assert "red" in error          # the list of real ones


def test_known_layers_parse_into_a_tuple():
    assert cli._layers("baseline_null,red") == ("baseline_null", "red")


def test_whitespace_around_layer_names_is_tolerated():
    assert cli._layers(" baseline_null , red ") == ("baseline_null", "red")


def test_an_empty_layer_list_is_rejected():
    with pytest.raises(Exception, match="at least one layer"):
        cli._layers(",,")


def test_the_default_layers_are_all_registered():
    """A default that names a layer nobody registered would fail at run time."""
    assert set(cli.DEFAULT_LAYERS) <= set(PREDICTION_LAYERS)


# --- required arguments ---------------------------------------------------

def test_diagnose_without_an_archive_reports_the_missing_option(capsys):
    assert cli.main(["diagnose"]) == 2
    assert "--archive" in capsys.readouterr().err


def test_prepare_without_a_name_reports_the_missing_option(tmp_path, capsys, archive_csv):
    assert cli.main(["prepare", "--archive", str(archive_csv)]) == 2
    assert "--name" in capsys.readouterr().err


def test_run_requires_either_a_config_or_an_archive(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run"])
    assert exit_info.value.code == 2
    assert "--config" in capsys.readouterr().err


def test_run_refuses_both_a_config_and_an_archive(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--config", "a.json", "--archive", "b.csv"])
    assert exit_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_run_refuses_prepared_and_raw_configs_together(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--config", "prepared.json", "--raw-config", "raw.json"])
    assert exit_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [
    ["--name", "other"], ["--target-lag", "28"], ["--value-col", "count"],
    ["--methods", "baseline_null"],
])
def test_raw_config_refuses_mixed_raw_cli_settings(tmp_path, capsys, extra):
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({"name": "d", "archive": "archive.csv"}))
    assert cli.main(["run", "--raw-config", str(config), *extra]) == 2
    error = capsys.readouterr().err
    assert "complete input" in error
    assert extra[0] in error


def test_run_with_a_config_does_not_demand_preparation_options(tmp_path, capsys):
    """--config is a complete description; --name would be noise."""
    config = tmp_path / "d.json"
    config.write_text(json.dumps({"name": "d", "prepared_dir": "input", "output_dir": "out",
                                  "prediction_layers": ["red"]}))
    # Fails on the absent prepared_dir, NOT on a missing --name.
    assert cli.main(["run", "--config", str(config), "--diagnosis-only"]) == 2
    assert "--name" not in capsys.readouterr().err


# --- failures a user actually hits ---------------------------------------

def test_a_missing_archive_exits_two_with_the_path(tmp_path, capsys):
    assert cli.main(["diagnose", "--archive", str(tmp_path / "absent.csv")]) == 2
    assert "absent.csv" in capsys.readouterr().err


def test_a_missing_config_exits_two(tmp_path, capsys):
    assert cli.main(["run", "--config", str(tmp_path / "absent.json")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_assess_reports_missing_frozen_inputs(tmp_path, capsys):
    results = tmp_path / "results"
    results.mkdir()
    assert cli.main(["assess", "--results", str(results)]) == 2
    assert "predictions.csv.gz" in capsys.readouterr().err


def test_assess_cli_passes_both_directories_and_prints_the_report(tmp_path, capsys, monkeypatch):
    results = tmp_path / "results"
    output = tmp_path / "fresh-assessment"
    called = {}

    def fake_assess(results_dir, output_dir):
        called["arguments"] = (results_dir, output_dir)
        return output_dir

    monkeypatch.setattr("autodelphirf.post_assessment.run.assess_results", fake_assess)
    assert cli.main(["assess", "--results", str(results), "--out", str(output)]) == 0
    assert called["arguments"] == (results, output)
    assert str(output / "report" / "report.html") in capsys.readouterr().out


def test_a_config_pointing_at_no_prepared_dataset_explains_what_was_expected(tmp_path, capsys):
    config = tmp_path / "d.json"
    config.write_text(json.dumps({"name": "d", "prepared_dir": "input", "output_dir": "out",
                                  "prediction_layers": ["red"]}))
    (tmp_path / "input").mkdir()
    assert cli.main(["run", "--config", str(config), "--diagnosis-only"]) == 2
    assert "was not prepared by AutoDelphiRF" in capsys.readouterr().err


def test_an_external_triangle_is_refused_before_anything_is_written(tmp_path, capsys):
    prepared = tmp_path / "input"
    prepared.mkdir()
    pd.DataFrame({"geo_value": ["aa"], "reference_date": ["2024-01-01"],
                  "report_date": ["2024-01-02"], "lag": [1]}).to_csv(prepared / "aa.csv", index=False)
    config = tmp_path / "d.json"
    config.write_text(json.dumps({"name": "d", "prepared_dir": "input", "output_dir": "out",
                                  "prediction_layers": ["baseline_null"]}))
    assert cli.main(["run", "--config", str(config)]) == 2
    error = capsys.readouterr().err
    assert "Prepared triangles built elsewhere are not supported" in error
    assert "autodelphirf prepare --archive" in error
    assert not any((tmp_path / "out").rglob("*.csv*"))


def test_wrong_column_names_are_reported_before_anything_is_written(tmp_path, capsys):
    path = tmp_path / "archive.csv"
    pd.DataFrame({"loc": ["aa"], "date": ["2024-01-01"], "v": [1.0]}).to_csv(path, index=False)
    out = tmp_path / "work"
    assert cli.main(["prepare", "--archive", str(path), "--name", "d", "--out", str(out)]) == 2
    assert "--reference-col" in capsys.readouterr().err
    assert not out.exists()


# --- diagnose -------------------------------------------------------------

def test_diagnose_prints_a_readable_report_and_writes_nothing(tmp_path, archive_csv, capsys):
    before = set(tmp_path.iterdir())
    assert cli.main(["diagnose", "--archive", str(archive_csv)]) == 0
    printed = capsys.readouterr().out
    for expected in ("locations", "cadence", "selected L",
                     "training window"):
        assert expected in printed
    assert "genuine revisions" not in printed
    assert "Nothing was written" in printed
    assert set(tmp_path.iterdir()) == before


def test_diagnose_json_emits_the_full_machine_readable_report(archive_csv, capsys):
    assert cli.main(["diagnose", "--archive", str(archive_csv), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_locations"] == 2
    assert "target_lag_completion_curve" in report


def test_diagnose_reports_an_explicit_target_lag_as_the_user_choice(archive_csv, capsys):
    assert cli.main(["diagnose", "--archive", str(archive_csv), "--target-lag", "365"]) == 0
    output = capsys.readouterr().out
    assert "selected L          365" in output
    assert "user_choice" in output
    assert "CONFIRM" not in output


def test_diagnose_handles_a_single_location_archive(tmp_path, capsys):
    path = tmp_path / "one.csv"
    archive_frame(days=120, max_lag=30, locations=("only",)).drop(
        columns=["geo_value"]).to_csv(path, index=False)
    assert cli.main(["diagnose", "--archive", str(path), "--no-geo"]) == 0
    output = capsys.readouterr().out
    assert "locations           1" in output
    assert "one location named 'single_location'" in output
    assert "pool across lags" in output
    assert "cross-location pooling is not possible" in output


def test_raw_config_drives_diagnosis_with_relative_paths_and_column_mapping(
        tmp_path, capsys):
    archive = tmp_path / "data" / "archive.csv"
    archive.parent.mkdir()
    archive_frame(days=120, max_lag=30).rename(columns={
        "geo_value": "place", "reference_date": "event_date",
        "report_date": "issue_date", "value": "count"}).to_csv(archive, index=False)
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({
        "name": "mapped",
        "archive": "data/archive.csv",
        "work_dir": "work",
        "columns": {"geo_value": "place", "reference_date": "event_date",
                    "report_date": "issue_date", "values": ["count"]},
        "value_type": "count",
        "model_methods": ["baseline_null", "red"],
    }))
    assert cli.main(["diagnose", "--raw-config", str(config)]) == 0
    assert "locations           2" in capsys.readouterr().out


def test_a_fraction_signal_accepts_one_precomputed_rate_column(
        tmp_path, archive_csv, monkeypatch):
    """A one-column rate archive reaches preparation without a denominator."""
    captured = {}

    def fake_bridge(spec, on_output=None):
        captured["spec"] = spec
        spec.output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("autodelphirf.pre_diagnosis.preparation.run_r_bridge", fake_bridge)
    assert cli.main(["prepare", "--archive", str(archive_csv), "--name", "d",
                     "--value-type", "fraction", "--target-lag", "21",
                     "--out", str(tmp_path / "w")]) == 0
    assert captured["spec"].value_cols == ("value",)


@pytest.mark.parametrize("genuine_testing", [True, False])
def test_raw_config_sets_genuine_testing_for_the_run(tmp_path, archive_csv, monkeypatch,
                                                     genuine_testing):
    """The raw config's case selection reaches the dataset JSON and preparation.json."""
    def fake_bridge(spec, on_output=None):
        spec.output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("autodelphirf.pre_diagnosis.preparation.run_r_bridge", fake_bridge)
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({
        "name": "sel", "archive": str(archive_csv), "work_dir": "work",
        "preprocessing": {"target_lag": 21}, "genuine_testing": genuine_testing}))
    assert cli.main(["prepare", "--raw-config", str(config)]) == 0
    dataset = json.loads((tmp_path / "work" / "sel.json").read_text())
    assert dataset["genuine_testing"] is genuine_testing
    assert dataset["genuine_training"] is True
    recorded = json.loads((tmp_path / "work" / "sel" / "preparation.json").read_text())
    assert recorded["case_selection"] == {"genuine_training": True,
                                          "genuine_testing": genuine_testing,
                                          "revroute_min_pool_rows": 200}


def test_raw_config_rejects_a_non_boolean_genuine_testing(tmp_path, archive_csv, capsys):
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({"name": "sel", "archive": str(archive_csv),
                                  "preprocessing": {"target_lag": 21},
                                  "genuine_testing": "false"}))
    assert cli.main(["prepare", "--raw-config", str(config)]) == 2
    assert "genuine_testing must be true or false" in capsys.readouterr().err


def test_raw_config_sets_revroute_min_pool_rows_for_the_run(tmp_path, archive_csv, monkeypatch):
    def fake_bridge(spec, on_output=None):
        spec.output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("autodelphirf.pre_diagnosis.preparation.run_r_bridge", fake_bridge)
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({
        "name": "sel", "archive": str(archive_csv), "work_dir": "work",
        "preprocessing": {"target_lag": 21}, "revroute_min_pool_rows": 100}))
    assert cli.main(["prepare", "--raw-config", str(config)]) == 0
    dataset = json.loads((tmp_path / "work" / "sel.json").read_text())
    assert dataset["revroute_min_pool_rows"] == 100
    recorded = json.loads((tmp_path / "work" / "sel" / "preparation.json").read_text())
    assert recorded["case_selection"]["revroute_min_pool_rows"] == 100


@pytest.mark.parametrize("value", [-1, 1.5, True, "100"])
def test_raw_config_rejects_an_invalid_revroute_min_pool_rows(tmp_path, archive_csv, capsys, value):
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({"name": "sel", "archive": str(archive_csv),
                                  "preprocessing": {"target_lag": 21},
                                  "revroute_min_pool_rows": value}))
    assert cli.main(["prepare", "--raw-config", str(config)]) == 2
    assert "revroute_min_pool_rows must be a nonnegative integer" in capsys.readouterr().err


def test_malformed_lag_terms_are_reported(tmp_path, archive_csv, capsys):
    assert cli.main(["prepare", "--archive", str(archive_csv), "--name", "d",
                     "--lag-terms", "7,fourteen", "--out", str(tmp_path / "w")]) == 2
    assert "comma-separated integers" in capsys.readouterr().err


# --- init -----------------------------------------------------------------

def test_init_writes_a_raw_archive_configuration_by_default(tmp_path, capsys):
    destination = tmp_path / "mine.json"
    assert cli.main(["init", "--name", "mine", "--output", str(destination)]) == 0
    payload = json.loads(destination.read_text())
    assert payload["name"] == "mine"
    assert "mine" in payload["archive"]
    assert "autodelphirf diagnose --raw-config" in capsys.readouterr().out


def test_init_no_longer_offers_a_prepared_configuration(tmp_path):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["init", "--kind", "prepared", "--name", "mine"])


def test_init_can_write_a_raw_archive_configuration(tmp_path, capsys):
    destination = tmp_path / "mine-raw.json"
    assert cli.main(["init", "--kind", "raw", "--name", "mine",
                     "--output", str(destination)]) == 0
    payload = json.loads(destination.read_text())
    assert payload["name"] == "mine"
    assert "mine" in payload["archive"]
    assert payload["columns"]["reference_date"] == "reference_date"
    assert "autodelphirf diagnose --raw-config" in capsys.readouterr().out


def test_init_refuses_to_clobber_an_existing_file(tmp_path, capsys):
    destination = tmp_path / "mine.json"
    destination.write_text('{"keep": "me"}')
    assert cli.main(["init", "--name", "mine", "--output", str(destination)]) == 1
    assert json.loads(destination.read_text()) == {"keep": "me"}
    assert "--force" in capsys.readouterr().err


def test_init_force_overwrites(tmp_path):
    destination = tmp_path / "mine.json"
    destination.write_text('{"keep": "me"}')
    assert cli.main(["init", "--name", "mine", "--output", str(destination), "--force"]) == 0
    assert json.loads(destination.read_text())["name"] == "mine"


def test_init_creates_missing_parent_directories(tmp_path):
    destination = tmp_path / "a" / "b" / "mine.json"
    assert cli.main(["init", "--name", "mine", "--output", str(destination)]) == 0
    assert destination.is_file()


# --- the R boundary -------------------------------------------------------

def test_prepare_reaches_the_r_bridge_with_the_diagnosed_settings(tmp_path, archive_csv,
                                                                  monkeypatch, capsys):
    """Everything up to the DelphiRF call is verified without needing R."""
    captured = {}

    def fake_bridge(spec, on_output=None):
        captured["spec"] = spec
        # The CLI leaves on_output unset so R writes straight to the terminal;
        # only the web UI, which needs the lines as data, passes a callback.
        captured["on_output"] = on_output
        spec.output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("autodelphirf.pre_diagnosis.preparation.run_r_bridge", fake_bridge)
    out = tmp_path / "work"
    assert cli.main(["prepare", "--archive", str(archive_csv), "--name", "mine",
                     "--out", str(out), "--target-lag", "21",
                     "--layers", "baseline_null,red"]) == 0

    spec = captured["spec"]
    assert captured["on_output"] is None
    assert spec.name == "mine"
    assert spec.output_dir == out / "mine"
    assert spec.ref_lag == spec.diagnosis["resolved_target_lag"]

    # The dataset config is written and points back at the triangle.
    config_path = out / "mine.json"
    assert json.loads(config_path.read_text())["prediction_layers"] == ["baseline_null", "red"]
    # Provenance is recorded next to the triangle.
    provenance = json.loads((spec.output_dir / "preparation.json").read_text())
    assert provenance["raw_archive"].endswith("archive.csv")
    assert provenance["diagnosis"]["n_locations"] == 2
    assert "autodelphirf run --config" in capsys.readouterr().out


def test_raw_config_drives_preparation_and_records_requested_methods(
        tmp_path, archive_csv, monkeypatch):
    captured = {}

    def fake_bridge(spec, on_output=None):
        captured["spec"] = spec
        spec.output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("autodelphirf.pre_diagnosis.preparation.run_r_bridge", fake_bridge)
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({
        "name": "mine", "archive": archive_csv.name, "work_dir": "work",
        "columns": {"reference_date": "reference_date", "report_date": "report_date",
                    "geo_value": "geo_value", "values": ["value"]},
        "preprocessing": {"target_lag": 21, "lag_terms": [1, 7], "training_days": 140,
                          "triangle_format": "csv"},
        "model_methods": ["baseline_null", "red"],
    }))
    assert cli.main(["prepare", "--raw-config", str(config)]) == 0
    spec = captured["spec"]
    assert spec.raw_csv == archive_csv.resolve()
    assert spec.output_dir == (tmp_path / "work" / "mine").resolve()
    assert spec.lag_terms == (1, 7)
    assert spec.training_days == 140
    generated = json.loads((tmp_path / "work" / "mine.json").read_text())
    assert generated["model_methods"] == ["baseline_null", "red"]


def test_raw_config_can_drive_the_complete_run(tmp_path, archive_csv, monkeypatch):
    prepared_config = tmp_path / "work" / "mine.json"
    called = {}

    def fake_prepare(spec, prediction_layers, config_path):
        called["spec"] = spec
        called["methods"] = prediction_layers
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "name": "mine", "prepared_dir": "mine", "output_dir": "results",
            "model_methods": list(prediction_layers)}))
        return config_path

    def fake_run(config, diagnosis_only=False, post_assessment=True):
        called["run_config"] = config
        called["post_assessment"] = post_assessment
        return {}

    monkeypatch.setattr(cli, "prepare_dataset", fake_prepare)
    monkeypatch.setattr("autodelphirf.workflow.run.run_pipeline", fake_run)
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({
        "name": "mine", "archive": archive_csv.name, "work_dir": "work",
        "preprocessing": {"target_lag": 21},
        "model_methods": ["baseline_null", "red"],
    }))
    assert cli.main(["run", "--raw-config", str(config)]) == 0
    assert called["methods"] == ("baseline_null", "red")
    assert called["run_config"].name == "mine"
    assert called["post_assessment"] is True
    assert prepared_config.is_file()
    assert cli.main(["run", "--raw-config", str(config), "--no-post-assessment"]) == 0
    assert called["post_assessment"] is False


def test_a_missing_r_install_is_reported_as_guidance(tmp_path, archive_csv, monkeypatch, capsys):
    monkeypatch.delenv("AUTODELPHIRF_RSCRIPT", raising=False)
    monkeypatch.setattr("autodelphirf.pre_diagnosis.preparation.shutil.which", lambda _: None)
    assert cli.main(["prepare", "--archive", str(archive_csv), "--name", "d",
                     "--target-lag", "21", "--out", str(tmp_path / "w")]) == 2
    error = capsys.readouterr().err
    assert "install R" in error and "DelphiRF" in error


def _frozen_run(tmp_path, case_selection):
    (tmp_path / "sel").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "sel" / "preparation.json").write_text(json.dumps({
        "resolved_preprocessing": {"start_date": "2022-01-01"},
        "case_selection": case_selection}))
    (tmp_path / "results" / "analysis_contract.json").write_text(json.dumps({"settings": {
        "ref_lag": 21, "temporal_resol": "daily", "lag_terms": [7, 14],
        "training_days": 365, "smoothed": False, "testing_days": 7}}))


@pytest.mark.parametrize("case_selection, frozen", [
    ({"revroute_min_pool_rows": 100}, 100),
    # A first run prepared before the minimum existed ran without one.
    ({}, 0)])
def test_update_inherits_the_first_runs_revroute_min_pool_rows(tmp_path, case_selection, frozen):
    import argparse
    _frozen_run(tmp_path, case_selection)
    arguments = cli._apply_frozen_update_settings(argparse.Namespace(out=str(tmp_path), name="sel"))
    assert arguments.revroute_min_pool_rows == frozen


def test_update_refuses_a_different_revroute_min_pool_rows(tmp_path):
    import argparse
    _frozen_run(tmp_path, {"revroute_min_pool_rows": 0})
    with pytest.raises(ValueError, match="revroute_min_pool_rows is frozen at 0"):
        cli._apply_frozen_update_settings(argparse.Namespace(
            out=str(tmp_path), name="sel", revroute_min_pool_rows=200))
