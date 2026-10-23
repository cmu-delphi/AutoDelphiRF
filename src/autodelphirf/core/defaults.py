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
# The optional RED comparator's routing settings.
ROUTING_DEFAULTS = DEFAULTS["red_routing"]
# Shared defaults for calibration, reliability, and evaluation.
CALIBRATION_DEFAULTS = DEFAULTS.get("calibration_reliability", {})
EVALUATION_DEFAULTS = DEFAULTS.get("evaluation", {})


# The as-of value at a case's own lag on the same smoothing level as its
# target. A raw target (Ch. 2: f(Y_itL)) is carried forward from the raw
# provisional value Y_itl; a smoothed target from the smoothed one.
PROVISIONAL_COLUMNS = {"log_value_target": "log_value_raw",
                       "log_value_target_7dav": "log_value_7dav"}


def provisional_column(target_column: str) -> str:
    """The provisional (as-of) value column matching ``target_column``.

    A prepared dataset with its own target name keeps the as-of value column
    the input schema maps, ``log_value_7dav``.
    """
    return PROVISIONAL_COLUMNS.get(str(target_column), "log_value_7dav")


# The one place the working-scale -> original-scale inverse is defined, so
# every consumer (pipeline evaluation/export, reliability reference building)
# agrees. Mirrors how the prepared triangle was built, which follows the
# VALUE-COLUMN ARITY rather than "count vs fraction":
#   count, one column          -> log(count + 1)              -> expm1
#   rate supplied directly     -> log(ratio + 1)              -> expm1
#   fraction, num + denom      -> log(num+1) - log(denom+1)   -> exp
def to_reporting_scale(values, working_scale: str):
    """Invert a dataset's working scale back to its original reporting scale.

    ``working_scale`` takes DelphiRF's values: ``"log_value"`` (one value
    column, log(value + 1)) or ``"log_ratio"`` (numerator and denominator,
    log(num + 1) - log(denom + 1)). ``"identity"`` is used internally for
    predictions that only exist on the reporting scale.
    """
    import numpy as np
    values = np.asarray(values, float)
    if working_scale == "log_value":
        return np.expm1(values)
    if working_scale == "log_ratio":
        return np.exp(values)
    if working_scale == "identity":
        return values
    raise ValueError(f"unsupported working_scale: {working_scale}")
