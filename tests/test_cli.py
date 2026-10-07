"""The ``autodelphirf`` command line.

The CLI is the whole user-facing surface for anyone who never imports the
package, so these tests cover argument validation, the exit statuses a script
or CI job would branch on, and the parts of the output a user reads. Anything
needing R is exercised up to the R boundary and no further.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from autodelphirf import cli
from autodelphirf.registry import PREDICTION_LAYERS

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


@pytest.mark.parametrize("command", ["diagnose", "prepare", "run", "assess", "init"])
def test_every_subcommand_has_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip()


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

    monkeypatch.setattr("autodelphirf.assessment.assess_results", fake_assess)
    assert cli.main(["assess", "--results", str(results), "--out", str(output)]) == 0
    assert called["arguments"] == (results, output)
    assert str(output / "report" / "report.html") in capsys.readouterr().out


def test_a_config_pointing_at_no_triangle_explains_what_was_expected(tmp_path, capsys):
    config = tmp_path / "d.json"
    config.write_text(json.dumps({"name": "d", "prepared_dir": "input", "output_dir": "out",
                                  "prediction_layers": ["red"]}))
    (tmp_path / "input").mkdir()
    assert cli.main(["run", "--config", str(config), "--diagnosis-only"]) == 2
    assert "no prepared triangles found" in capsys.readouterr().err


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
    for expected in ("locations", "cadence", "genuine revisions", "resolved L",
                     "training window"):
        assert expected in printed
    assert "Nothing was written" in printed
    assert set(tmp_path.iterdir()) == before


def test_diagnose_json_emits_the_full_machine_readable_report(archive_csv, capsys):
    assert cli.main(["diagnose", "--archive", str(archive_csv), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_locations"] == 2
    assert "target_lag_completion_curve" in report


def test_diagnose_surfaces_a_target_lag_disagreement_for_confirmation(archive_csv, capsys):
    """A user's L is never adopted silently when the data disagrees."""
    assert cli.main(["diagnose", "--archive", str(archive_csv), "--target-lag", "365"]) == 0
    assert "CONFIRM" in capsys.readouterr().out


def test_diagnose_handles_a_single_location_archive(tmp_path, capsys):
    path = tmp_path / "one.csv"
    archive_frame(days=120, max_lag=30, locations=("only",)).drop(
        columns=["geo_value"]).to_csv(path, index=False)
    assert cli.main(["diagnose", "--archive", str(path), "--no-geo"]) == 0
    assert "locations           1" in capsys.readouterr().out


def test_a_fraction_signal_needs_two_value_columns(tmp_path, archive_csv, capsys):
    """The arity rule is enforced where the spec is built, i.e. in prepare."""
    assert cli.main(["prepare", "--archive", str(archive_csv), "--name", "d",
                     "--value-type", "fraction", "--out", str(tmp_path / "w")]) == 2
    assert "exactly two value columns" in capsys.readouterr().err


def test_malformed_lag_terms_are_reported(tmp_path, archive_csv, capsys):
    assert cli.main(["prepare", "--archive", str(archive_csv), "--name", "d",
                     "--lag-terms", "7,fourteen", "--out", str(tmp_path / "w")]) == 2
    assert "comma-separated integers" in capsys.readouterr().err


# --- init -----------------------------------------------------------------

def test_init_writes_a_loadable_template(tmp_path, capsys):
    destination = tmp_path / "mine.json"
    assert cli.main(["init", "--name", "mine", "--output", str(destination)]) == 0
    payload = json.loads(destination.read_text())
    assert payload["name"] == "mine"
    assert "mine" in payload["prepared_dir"]
    assert "autodelphirf run --config" in capsys.readouterr().out


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

    monkeypatch.setattr("autodelphirf.prepare.run_r_bridge", fake_bridge)
    out = tmp_path / "work"
    assert cli.main(["prepare", "--archive", str(archive_csv), "--name", "mine",
                     "--out", str(out), "--layers", "baseline_null,red"]) == 0

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


def test_a_missing_r_install_is_reported_as_guidance(tmp_path, archive_csv, monkeypatch, capsys):
    monkeypatch.delenv("AUTODELPHIRF_RSCRIPT", raising=False)
    monkeypatch.setattr("autodelphirf.prepare.shutil.which", lambda _: None)
    assert cli.main(["prepare", "--archive", str(archive_csv), "--name", "d",
                     "--out", str(tmp_path / "w")]) == 2
    error = capsys.readouterr().err
    assert "install R" in error and "DelphiRF" in error
