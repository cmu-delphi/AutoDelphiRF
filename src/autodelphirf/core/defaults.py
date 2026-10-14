"""Load package-owned JSON configuration defaults."""
from __future__ import annotations

import json
from pathlib import Path

from .resources import load_resource, resource_path

# Shipped as package data (``autodelphirf/resources/``) rather than read
# out of a surrounding working directory, so the constants travel with the
# installed package. See ``resources.py``.
CONFIG_PATH = resource_path("pipeline_defaults.json")
DEFAULTS = load_resource("pipeline_defaults.json")

VERSION = DEFAULTS["version"]
ROUTING_DEFAULTS = DEFAULTS["routing"]
RESIDUAL_DEFAULTS = DEFAULTS["residual"]
RR_DELPHIRF_DEFAULTS = DEFAULTS["rr_delphirf"]
PROCESS_DEFAULTS = DEFAULTS["process_monitor"]
# Shared defaults for calibration, reliability, and evaluation.
CALIBRATION_DEFAULTS = DEFAULTS.get("calibration_reliability", {})
EVALUATION_DEFAULTS = DEFAULTS.get("evaluation", {})


# The one place the working-scale -> original-scale inverse is defined, so
# every consumer (pipeline evaluation/export, reliability reference building)
# agrees. Mirrors how the prepared triangle was built, which follows the
# VALUE-COLUMN ARITY rather than "count vs fraction":
#   count, one column          -> log(count + 1)              -> expm1
#   rate supplied directly     -> log(ratio + 1)              -> expm1
#   fraction, num + denom      -> log(num+1) - log(denom+1)   -> exp
def to_raw_scale(values, transform: str):
    """Invert a dataset's working scale back to its original reporting scale."""
    import numpy as np
    values = np.asarray(values, float)
    if transform == "log1p":
        return np.expm1(values)
    if transform == "log":
        return np.exp(values)
    if transform == "identity":
        return values
    raise ValueError(f"unsupported value_transform: {transform}")
