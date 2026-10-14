"""Primary RevRoute-pooling plus DelphiRF model strategy."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ..revroute.pooling.plan import learn_task_pools

PRIMARY_METHODS = {"delphirf": "RevRoute DelphiRF"}

#: Saved implementation label for RevRoute pooling followed by real DelphiRF.
#: Increment this only when a new named method version is deliberately frozen.
RR_DELPHIRF_CODE_VERSION = "rrdelphirf0"


def prepare_revroute_delphirf(prepared: pd.DataFrame, schedule: pd.DataFrame,
                              output: Path) -> Path:
    """Create the primary method's pools without applying DelphiRF lag bands."""
    return learn_task_pools(prepared, schedule, Path(output).with_name("revroute_pools.csv"))


__all__ = ["PRIMARY_METHODS", "RR_DELPHIRF_CODE_VERSION",
           "prepare_revroute_delphirf"]
