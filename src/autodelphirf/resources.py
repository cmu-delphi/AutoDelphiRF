"""Resolve package resources and optional user configuration files."""
from __future__ import annotations

import json
import os
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"

#: Environment variable naming an explicit file, per user-config file name.
ENVIRONMENT_OVERRIDES = {
    "target_lag_params.json": "AUTODELPHIRF_TARGET_LAG_PARAMS",
    "raw_archive_manifest.json": "AUTODELPHIRF_RAW_ARCHIVE_MANIFEST",
}


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


def user_config_path(name: str) -> Path | None:
    """Locate an optional user-configuration file, or ``None`` if absent.

    Search order is documented in the module docstring. A path named by an
    environment variable that does not exist is an error, not a silent miss:
    the user asked for that specific file.
    """
    variable = ENVIRONMENT_OVERRIDES.get(name)
    if variable and os.environ.get(variable):
        path = Path(os.environ[variable]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{variable} points at {path}, which does not exist")
        return path
    directory = os.environ.get("AUTODELPHIRF_CONFIG_DIR")
    if directory:
        path = Path(directory).expanduser() / name
        if path.is_file():
            return path
    path = Path.cwd() / "config" / name
    return path if path.is_file() else None
