"""Issued-state feature discovery shared by RED and post-forecast assessment."""
from __future__ import annotations
import re
import pandas as pd

_DELTA = re.compile(r"^log_delta_value_7dav_lag\d+$")
_ONEHOT = re.compile(r"^\w+_(ref|issue)$")


def full_context_columns(frame) -> tuple[list[str], list[str]]:
    present = list(frame.columns)
    numeric = [c for c in ("log_value_7dav",) if c in present]
    numeric += [c for c in present if _DELTA.match(c)]
    numeric += [c for c in ("inv_log_lag",) if c in present]
    categorical = [c for c in ("refd_col",) if c in present]
    categorical += [c for c in present if _ONEHOT.match(c)]
    return numeric, categorical


def attach_issued_context(cases: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    keys = ["geo_value", "reference_date", "report_date", "lag"]
    numeric, categorical = full_context_columns(prepared)
    allowed = numeric + categorical
    state = prepared[keys + allowed].drop_duplicates(keys, keep="last")
    return cases.drop(columns=allowed, errors="ignore").merge(
        state, on=keys, how="left", validate="one_to_one")


__all__ = ["attach_issued_context", "full_context_columns"]
