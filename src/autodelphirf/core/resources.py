"""Resolve package resources."""
from __future__ import annotations

import json
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"


def resource_path(name: str) -> Path:
    """Absolute path to one shipped package-data file.

    Raises ``FileNotFoundError`` rather than returning a path that does not
    exist: a missing resource means a broken install, not a user error, and
    the difference is worth stating at the point of failure.
    """
    path = RESOURCE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged resource '{name}' is missing from {RESOURCE_DIR}. This indicates an "
            "incomplete installation: reinstall autodelphirf, or run from a source checkout.")
    return path


def load_resource(name: str) -> dict:
    """Parse one shipped JSON resource."""
    return json.loads(resource_path(name).read_text())
