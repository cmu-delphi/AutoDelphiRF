"""Packaged resources and optional user-configuration lookup tests."""
from __future__ import annotations

import json

import pytest

from autodelphirf import resources


# --- packaged resources ---------------------------------------------------

@pytest.mark.parametrize("name", ["prepared_triangle_schema.json",
                                  "revroute_v8_defaults.json",
                                  "dataset_template.json",
                                  "prepare_triangle.R"])
def test_every_declared_resource_ships_with_the_package(name):
    """Each file pyproject declares as package data is actually present."""
    path = resources.resource_path(name)
    assert path.is_file()
    assert path.read_text().strip()


def test_resource_path_is_inside_the_package_not_a_sibling_directory():
    """The whole point of the change: no reaching out of the install."""
    path = resources.resource_path("revroute_v8_defaults.json")
    package_root = resources.RESOURCE_DIR.parent
    assert package_root in path.parents
    assert path.parent.name == "resources"


def test_missing_resource_names_the_install_not_the_user():
    with pytest.raises(FileNotFoundError, match="incomplete installation"):
        resources.resource_path("no_such_resource.json")


def test_load_resource_parses_json():
    schema = resources.load_resource("prepared_triangle_schema.json")
    assert set(schema["columns"]) >= {"geo_value", "reference_date", "report_date",
                                      "target_date", "lag", "target_lag", "as_of_value"}


def test_defaults_module_reads_its_constants_from_package_data():
    """defaults.py imports at module scope; a broken lookup breaks every import."""
    from autodelphirf import defaults

    assert defaults.CONFIG_PATH == resources.resource_path("revroute_v8_defaults.json")
    assert defaults.VERSION
    assert defaults.ROUTING_DEFAULTS


# --- optional user configuration ------------------------------------------

def test_absent_user_config_is_none_not_an_error(tmp_path, monkeypatch):
    """A user with no target_lag_params.json must still be able to run."""
    monkeypatch.delenv("AUTODELPHIRF_TARGET_LAG_PARAMS", raising=False)
    monkeypatch.delenv("AUTODELPHIRF_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resources.user_config_path("target_lag_params.json") is None


def test_environment_variable_names_an_explicit_file(tmp_path, monkeypatch):
    explicit = tmp_path / "my_lags.json"
    explicit.write_text("{}")
    monkeypatch.setenv("AUTODELPHIRF_TARGET_LAG_PARAMS", str(explicit))
    assert resources.user_config_path("target_lag_params.json") == explicit


def test_environment_variable_pointing_nowhere_is_an_error_not_a_silent_miss(tmp_path, monkeypatch):
    """A user who named a file expects that file, not a fallback."""
    monkeypatch.setenv("AUTODELPHIRF_TARGET_LAG_PARAMS", str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match="AUTODELPHIRF_TARGET_LAG_PARAMS"):
        resources.user_config_path("target_lag_params.json")


def test_config_dir_is_searched_when_no_explicit_file_is_named(tmp_path, monkeypatch):
    directory = tmp_path / "conf"
    directory.mkdir()
    (directory / "raw_archive_manifest.json").write_text("{}")
    monkeypatch.delenv("AUTODELPHIRF_RAW_ARCHIVE_MANIFEST", raising=False)
    monkeypatch.setenv("AUTODELPHIRF_CONFIG_DIR", str(directory))
    found = resources.user_config_path("raw_archive_manifest.json")
    assert found == directory / "raw_archive_manifest.json"


def test_config_dir_without_the_file_falls_through_to_the_working_directory(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    local = tmp_path / "config"
    local.mkdir()
    (local / "target_lag_params.json").write_text("{}")
    monkeypatch.delenv("AUTODELPHIRF_TARGET_LAG_PARAMS", raising=False)
    monkeypatch.setenv("AUTODELPHIRF_CONFIG_DIR", str(empty))
    monkeypatch.chdir(tmp_path)
    assert resources.user_config_path("target_lag_params.json") == local / "target_lag_params.json"


def test_explicit_file_outranks_config_dir(tmp_path, monkeypatch):
    """Search order is most-specific-first, and it is observable."""
    directory = tmp_path / "conf"
    directory.mkdir()
    (directory / "target_lag_params.json").write_text('{"source": "dir"}')
    explicit = tmp_path / "explicit.json"
    explicit.write_text('{"source": "explicit"}')
    monkeypatch.setenv("AUTODELPHIRF_CONFIG_DIR", str(directory))
    monkeypatch.setenv("AUTODELPHIRF_TARGET_LAG_PARAMS", str(explicit))
    found = resources.user_config_path("target_lag_params.json")
    assert json.loads(found.read_text())["source"] == "explicit"


# --- the pipeline's use of them -------------------------------------------

def test_pipeline_runs_without_any_target_lag_params_file(tmp_path, monkeypatch):
    """No declared target lag means 'no choice', not a crash on Path(None)."""
    from autodelphirf.pipeline import load_target_lag_params

    monkeypatch.delenv("AUTODELPHIRF_TARGET_LAG_PARAMS", raising=False)
    monkeypatch.delenv("AUTODELPHIRF_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_target_lag_params("anything") == (None, False, {})


def test_target_lag_params_are_read_when_present(tmp_path, monkeypatch):
    from autodelphirf.pipeline import load_target_lag_params

    params = tmp_path / "params.json"
    params.write_text(json.dumps({
        "defaults": {"rule": {"cadence_days": 7}},
        "datasets": {"mine": {"target_lag": 28, "confirmed": True,
                              "rule": {"min_followup_days": 30}}}}))
    monkeypatch.setenv("AUTODELPHIRF_TARGET_LAG_PARAMS", str(params))
    lag, confirmed, rule = load_target_lag_params("mine")
    assert (lag, confirmed) == (28, True)
    # Dataset rule overrides merge onto the defaults rather than replacing them.
    assert rule == {"cadence_days": 7, "min_followup_days": 30}


def test_dataset_absent_from_the_params_file_still_inherits_default_rules(tmp_path, monkeypatch):
    from autodelphirf.pipeline import load_target_lag_params

    params = tmp_path / "params.json"
    params.write_text(json.dumps({"defaults": {"rule": {"cadence_days": 14}}, "datasets": {}}))
    monkeypatch.setenv("AUTODELPHIRF_TARGET_LAG_PARAMS", str(params))
    assert load_target_lag_params("unlisted") == (None, False, {"cadence_days": 14})


def test_raw_archive_diagnosis_degrades_cleanly_with_no_manifest():
    """An absent manifest is the normal case for a user's own dataset."""
    from autodelphirf.diagnosis import raw_archive_for_diagnosis

    archive, note = raw_archive_for_diagnosis("mine", None)
    assert archive is None
    assert "no raw-archive manifest configured" in note
