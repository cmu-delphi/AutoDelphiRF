"""Method dispatcher for the end-to-end workflow."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ..models.delphirf_backend import train_delphirf
from ..models.registry import DELPHIRF_METHODS
from ..models.revroute_delphirf import (PRIMARY_METHODS, PRIMARY_METHOD_ID,
                                        prepare_revroute_delphirf)


def train_requested_delphirf(config, prepared: pd.DataFrame, schedule: pd.DataFrame,
                             methods: tuple[str, ...], output: Path, *,
                             model_dir: Path | None = None, model_mode: str = "train",
                             issued_forecasts: Path | None = None,
                             unresolved_only: bool = False):
    """Resolve requested model adapters, then execute one shared backend run."""
    unknown = set(methods).difference(DELPHIRF_METHODS)
    if unknown:
        raise ValueError(f"not DelphiRF methods: {sorted(unknown)}")
    pools = None
    if any(method in PRIMARY_METHODS for method in methods):
        pool_options = ({"preserve_existing": True} if model_mode != "train" else {})
        pools = prepare_revroute_delphirf(prepared, schedule, output,
                                          min_pool_rows=config.revroute_min_pool_rows,
                                          **pool_options)
    persistence = {}
    if model_dir is not None:
        persistence["model_dir"] = model_dir
    if model_mode != "train":
        persistence["model_mode"] = model_mode
    if issued_forecasts is not None:
        persistence["issued_forecasts"] = issued_forecasts
    if unresolved_only:
        persistence["unresolved_only"] = True
    path, seconds, provenance = train_delphirf(
        config, prepared, schedule,
        {method: DELPHIRF_METHODS[method] for method in methods}, output, pools=pools,
        **persistence)
    if any(method in PRIMARY_METHODS for method in methods):
        provenance["primary_method_id"] = PRIMARY_METHOD_ID
    return path, seconds, provenance


__all__ = ["DELPHIRF_METHODS", "train_requested_delphirf"]
