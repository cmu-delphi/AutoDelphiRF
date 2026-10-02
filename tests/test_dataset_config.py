"""Dataset-config loading: the contract between a user's JSON and a run.

``load_dataset_config`` is the first thing a user's own file touches, so its
failure modes are the ones they meet first. These tests pin what is required,
how paths resolve, what defaults apply, and what a malformed file does.
"""
from __future__ import annotations

import json

import pytest

from autodelphirf.config import DatasetConfig, load_dataset_config

MINIMAL = {"name": "d", "prepared_dir": "input", "output_dir": "out",
           "prediction_layers": ["red"]}


def write_config(path, **overrides):
    payload = {**MINIMAL, **overrides}
    path.write_text(json.dumps(payload))
    return path


# --- required fields ------------------------------------------------------

@pytest.mark.parametrize("dropped", ["name", "prepared_dir", "output_dir", "prediction_layers"])
def test_each_required_field_is_reported_by_name_when_missing(tmp_path, dropped):
    payload = {k: v for k, v in MINIMAL.items() if k != dropped}
    path = tmp_path / "d.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=dropped):
        load_dataset_config(path)


def test_several_missing_fields_are_reported_together(tmp_path):
    """One run, one list -- not a field-at-a-time guessing game."""
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"name": "d"}))
    with pytest.raises(ValueError) as error:
        load_dataset_config(path)
    message = str(error.value)
    assert "output_dir" in message and "prepared_dir" in message and "prediction_layers" in message


def test_absent_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset_config(tmp_path / "nope.json")


def test_malformed_json_is_a_json_error_not_a_confusing_key_error(tmp_path):
    path = tmp_path / "d.json"
    path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_dataset_config(path)


# --- path resolution ------------------------------------------------------

def test_relative_paths_resolve_against_the_config_not_the_shell(tmp_path, monkeypatch):
    """A config must work from any working directory; this is why."""
    nested = tmp_path / "workspace" / "configs"
    nested.mkdir(parents=True)
    path = write_config(nested / "d.json")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_dataset_config(path)
    assert config.prepared_dir == nested / "input"
    assert config.output_dir == nested / "out"


def test_parent_relative_paths_resolve(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    path = write_config(nested / "d.json", prepared_dir="../data", output_dir="../../out")
    config = load_dataset_config(path)
    assert config.prepared_dir == tmp_path / "a" / "data"
    assert config.output_dir == tmp_path / "out"


def test_absolute_paths_are_left_alone(tmp_path):
    absolute = tmp_path / "somewhere" / "input"
    path = write_config(tmp_path / "d.json", prepared_dir=str(absolute))
    assert load_dataset_config(path).prepared_dir == absolute


def test_optional_paths_stay_none_when_unset(tmp_path):
    config = load_dataset_config(write_config(tmp_path / "d.json"))
    assert config.comparator_file is None
    assert config.reliability_reference_file is None


def test_optional_paths_resolve_when_set(tmp_path):
    path = write_config(tmp_path / "d.json", comparator_file="cmp.csv.gz",
                        reliability_reference_file="ref/reliability.csv")
    config = load_dataset_config(path)
    assert config.comparator_file == tmp_path / "cmp.csv.gz"
    assert config.reliability_reference_file == tmp_path / "ref" / "reliability.csv"


# --- defaults and overrides ----------------------------------------------

def test_defaults_match_the_documented_contract(tmp_path):
    config = load_dataset_config(write_config(tmp_path / "d.json"))
    assert config.schedule_file == "test_dates.csv"
    assert config.include_genuine_events_only is True
    assert config.uncertainty_layer is True
    assert config.value_transform is None      # derived from the triangle, not declared
    assert config.initial_lag is None
    assert config.comparator_method_column == "method"
    assert config.comparator_prediction_column == "prediction"
    assert config.method_params == {}
    assert config.revision_profile is None


def test_input_columns_inherit_the_packaged_schema(tmp_path):
    """A config naming no columns still gets the full canonical mapping."""
    config = load_dataset_config(write_config(tmp_path / "d.json"))
    assert config.input_columns["geo_value"] == "geo_value"
    assert config.input_columns["as_of_value"] == "log_value_7dav"


def test_input_columns_override_only_the_named_field(tmp_path):
    """Partial mappings are the documented usage; they must not blank the rest."""
    path = write_config(tmp_path / "d.json", input_columns={"geo_value": "location_id"})
    config = load_dataset_config(path)
    assert config.input_columns["geo_value"] == "location_id"
    assert config.input_columns["report_date"] == "report_date"
    assert config.input_columns["as_of_value"] == "log_value_7dav"


def test_unknown_keys_are_ignored_so_comments_are_possible(tmp_path):
    """The shipped template carries _comment keys; they must not break loading."""
    path = write_config(tmp_path / "d.json", _comment="explanatory text", _other=[1, 2])
    assert load_dataset_config(path).name == "d"


def test_prediction_layers_become_an_immutable_tuple(tmp_path):
    config = load_dataset_config(write_config(tmp_path / "d.json",
                                              prediction_layers=["baseline_null", "red"]))
    assert config.prediction_layers == ("baseline_null", "red")
    with pytest.raises((AttributeError, TypeError)):
        config.prediction_layers.append("x")


def test_config_is_frozen_so_a_run_cannot_mutate_its_own_settings(tmp_path):
    config = load_dataset_config(write_config(tmp_path / "d.json"))
    assert isinstance(config, DatasetConfig)
    with pytest.raises(Exception):
        config.name = "changed"


def test_empty_prediction_layers_loads_and_defers_to_the_pipeline(tmp_path):
    """An empty list is structurally valid; the run reports having nothing to do."""
    config = load_dataset_config(write_config(tmp_path / "d.json", prediction_layers=[]))
    assert config.prediction_layers == ()


def test_uncertainty_layer_can_be_switched_off(tmp_path):
    config = load_dataset_config(write_config(tmp_path / "d.json", uncertainty_layer=False))
    assert config.uncertainty_layer is False


def test_the_shipped_template_is_itself_loadable(tmp_path):
    """`autodelphirf init` writes this file; it must load once paths are filled in."""
    from autodelphirf.resources import resource_path

    template = json.loads(resource_path("dataset_template.json").read_text())
    template["prepared_dir"] = "input"
    template["output_dir"] = "out"
    path = tmp_path / "from_template.json"
    path.write_text(json.dumps(template))
    config = load_dataset_config(path)
    assert config.name == template["name"]
    assert config.prediction_layers
