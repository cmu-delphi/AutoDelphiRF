"""Cutoff-valid historical training selection shared across model methods."""
from __future__ import annotations

import pandas as pd


def rolling_completed_training(prepared: pd.DataFrame, cutoff: pd.Timestamp,
                               training_days: int) -> pd.DataFrame:
    """Select completed historical reference dates available at an origin.

    ``training_days`` is a trailing window on each row's target availability
    date, not its reference date. Report-state features must also have been
    released before the cutoff.
    """
    window_start = cutoff - pd.Timedelta(days=training_days)
    eligible = (prepared.target_date > window_start) & (prepared.target_date <= cutoff)
    if "report_date" in prepared:
        eligible &= prepared.report_date < cutoff
    return prepared[eligible].copy()
