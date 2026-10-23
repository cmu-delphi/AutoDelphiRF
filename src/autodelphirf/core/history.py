"""Cutoff-valid historical training selection shared across model methods.

The prepared triangle is the one definition of each target's date:
``autodelphirf prepare`` writes ``target_date = reference_date + L`` and
nothing afterwards replaces it. Everything here reads that stored field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REBUILD = ("Rebuild it with the current version: `autodelphirf prepare` (or "
           "`autodelphirf run --archive ...`).")


def target_as_of_date(schedule: pd.DataFrame) -> pd.Timestamp:
    """The as-of date the prepared targets were built with, as recorded at preparation."""
    recorded = (pd.to_datetime(schedule["target_as_of_date"]).dropna()
                if "target_as_of_date" in schedule else pd.Series(dtype="datetime64[ns]"))
    if recorded.empty:
        raise ValueError("the prepared schedule records no target_as_of_date, so it was built "
                         "by an older version. " + REBUILD)
    return pd.Timestamp(recorded.max()).normalize()


def validate_target_dates(prepared: pd.DataFrame, target_lag: int) -> None:
    """Every stored target date must be its reference date plus the target lag."""
    expected = (pd.to_datetime(prepared["reference_date"]) +
                pd.to_timedelta(int(target_lag), unit="D"))
    stored = pd.to_datetime(prepared["target_date"])
    mismatched = int((stored != expected).sum())
    if mismatched:
        raise ValueError(
            f"{mismatched:,} prepared row(s) have a target_date other than reference_date + "
            f"{int(target_lag)} days; the prepared dataset is incompatible. " + REBUILD)


def enforce_target_maturity(prepared: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Withhold every target whose stored date is after the as-of date.

    Until lag L is reached the forward-filled value is still provisional, so a
    reference date has no target: it can be forecast but not trained on,
    scored, or used to calibrate. ``target_date`` itself is never changed.
    """
    result = prepared.copy()
    immature = (pd.to_datetime(result["target_date"]) > pd.Timestamp(as_of_date)).to_numpy()
    for column in result.columns:
        if column.startswith(("value_target", "log_value_target")):
            result.loc[immature, column] = np.nan
    # The requested target lag remains known. Only the report that will supply
    # its value is unknown before the target matures.
    for column in ("target_value_source", "target_source_lag"):
        if column in result:
            result.loc[immature, column] = np.nan
    return result


def rolling_completed_training(prepared: pd.DataFrame, cutoff: pd.Timestamp,
                               training_days: int) -> pd.DataFrame:
    """Select completed historical reference dates available at an origin.

    ``training_days`` is a trailing window on the stored target date, so a
    fallback value observed earlier stays unavailable as training truth until
    that date. Report-state features must also have been released by the
    cutoff.
    """
    window_start = cutoff - pd.Timedelta(days=training_days)
    target_date = pd.to_datetime(prepared.target_date)
    eligible = target_date.gt(window_start) & target_date.le(cutoff)
    if "report_date" in prepared:
        eligible &= prepared.report_date <= cutoff
    return prepared[eligible].copy()
