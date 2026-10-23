"""Packaged resources and optional user-configuration lookup tests."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from autodelphirf.core import resources
# --- packaged resources ---------------------------------------------------

@pytest.mark.parametrize("name", ["pipeline_defaults.json",
                                  "raw_dataset_template.json",
                                  "prepare_triangle.R"])
def test_every_declared_resource_ships_with_the_package(name):
    """Each file pyproject declares as package data is actually present."""
    path = resources.resource_path(name)
    assert path.is_file()
    assert path.read_text().strip()


def test_resource_path_is_inside_the_package_not_a_sibling_directory():
    """The whole point of the change: no reaching out of the install."""
    path = resources.resource_path("pipeline_defaults.json")
    package_root = resources.RESOURCE_DIR.parent
    assert package_root in path.parents
    assert path.parent.name == "resources"


def test_missing_resource_names_the_install_not_the_user():
    with pytest.raises(FileNotFoundError, match="incomplete installation"):
        resources.resource_path("no_such_resource.json")


def test_load_resource_parses_json():
    defaults = resources.load_resource("pipeline_defaults.json")
    assert {"red_routing", "calibration_reliability", "evaluation"} <= set(defaults)
    # Blocks nothing reads are gone, and RED's bandwidth is labelled as RED's.
    assert not {"routing", "residual", "rr_delphirf", "graph_fused_delphirf"} & set(defaults)
    assert "training_days" not in defaults["red_routing"]
    assert "not Delphi-RF's decay parameter" in defaults["red_routing"]["note"]


def test_defaults_module_reads_its_constants_from_package_data():
    """defaults.py imports at module scope; a broken lookup breaks every import."""
    from autodelphirf.core import defaults

    assert defaults.CONFIG_PATH == resources.resource_path("pipeline_defaults.json")
    assert defaults.VERSION
    assert defaults.ROUTING_DEFAULTS
# End of packaged-resource tests.
