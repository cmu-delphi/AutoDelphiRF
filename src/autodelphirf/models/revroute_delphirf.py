"""Primary RevRoute-pooling plus DelphiRF model strategy."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ..revroute.pooling.plan import learn_task_pools

PRIMARY_METHODS = {"delphirf": "RevRoute DelphiRF"}

#: Stable identifier for RevRoute pooling followed by DelphiRF training.
PRIMARY_METHOD_ID = "revroute_delphirf"


def prepare_revroute_delphirf(prepared: pd.DataFrame, schedule: pd.DataFrame,
                              output: Path, *, preserve_existing: bool = False) -> Path:
    """Create the primary method's pools without applying DelphiRF lag bands."""
    return learn_task_pools(
        prepared, schedule, Path(output).with_name("revroute_pools.csv"),
        preserve_existing=preserve_existing)


__all__ = ["PRIMARY_METHODS", "PRIMARY_METHOD_ID",
           "prepare_revroute_delphirf"]
