"""Method dispatcher for the end-to-end workflow."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ..models.delphirf_backend import train_delphirf
from ..models.registry import DELPHIRF_METHODS
from ..models.revroute_delphirf import (PRIMARY_METHODS, RR_DELPHIRF_CODE_VERSION,
                                        prepare_revroute_delphirf)


def train_requested_delphirf(config, prepared: pd.DataFrame, schedule: pd.DataFrame,
                             methods: tuple[str, ...], output: Path):
    """Resolve requested model adapters, then execute one shared backend run."""
    unknown = set(methods).difference(DELPHIRF_METHODS)
    if unknown:
        raise ValueError(f"not DelphiRF methods: {sorted(unknown)}")
    pools = (prepare_revroute_delphirf(prepared, schedule, output)
             if any(method in PRIMARY_METHODS for method in methods) else None)
    path, seconds, provenance = train_delphirf(
        config, prepared, schedule,
        {method: DELPHIRF_METHODS[method] for method in methods}, output, pools=pools)
    if any(method in PRIMARY_METHODS for method in methods):
        provenance["rr_delphirf_code_version"] = RR_DELPHIRF_CODE_VERSION
    return path, seconds, provenance


__all__ = ["DELPHIRF_METHODS", "train_requested_delphirf"]
