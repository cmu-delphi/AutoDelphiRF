from __future__ import annotations
from typing import Callable
import numpy as np

PREDICTION_LAYERS: dict[str, Callable] = {}

# Column templates for the layers whose adapter is a plain column read. They
# let ``pipeline.standardize_predictions`` build one method's long rows as
# whole columns instead of a Python dict per row, which is the difference
# between 31k emitted rows (nhsn28) and 5.3M (chng). A layer absent from this
# map still works: it falls back to its row adapter above.
LAYER_COLUMNS: dict[str, tuple[str, str | None]] = {
    "red": ("red_prediction", "red_tau{tau:g}"),
    "baseline_null": ("Null", None),
    "null": ("Null", None),
}

def register_prediction_layer(name: str):
    def decorator(function):
        if name in PREDICTION_LAYERS:
            raise ValueError(f"prediction layer already registered: {name}")
        PREDICTION_LAYERS[name] = function
        return function
    return decorator

@register_prediction_layer("red")
def red_layer(row, taus):
    return row["red_prediction"], np.array([row[f"red_tau{t:g}"] for t in taus], float)

@register_prediction_layer("baseline_null")
def null_layer(row, taus):
    return float(row["Null"]), np.full(len(taus), np.nan)

register_prediction_layer("null")(null_layer)  # legacy dataset configs

def point_layer(column: str):
    def extract(row, taus):
        return float(row[column]), np.full(len(taus), np.nan)
    return extract


for _method in ("delphirf", "naive_delphirf", "similarity_weighted_delphirf",
                "global_delphirf"):
    register_prediction_layer(_method)(point_layer(_method))


#: What ``--layers`` defaults to, and what the browser UI ticks. ``delphirf``
#: is DelphiRF fitted over RevRoute's sparse task pools, and
#: ``baseline_null`` is not a rival to it but the do-nothing reference every
#: result is read against: take the current value as final. Together they are
#: the smallest run that answers "does RevRoute beat doing nothing on my data".
#: RED and the other real DelphiRF pooling choices are opt-in.
#:
#: It lives here, beside the layer registry, so the command line and the page
#: cannot drift apart on what "the default" is.
# The main released method is RevRoute task pooling followed by the real
# DelphiRF estimator. ``baseline_null`` is retained as its reference.
DEFAULT_LAYERS = ("baseline_null", "delphirf")
