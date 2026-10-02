"""Shared fixtures.

The archive builder lives here rather than in one test module so the CLI and
preparation suites do not have to import each other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def archive_frame(days: int = 200, max_lag: int = 40, locations=("aa", "bb"),
                  halflife: float = 8.0) -> pd.DataFrame:
    """A small raw archive with a genuine, converging revision process.

    One row per (location, reference date, report date). Each episode starts
    at half its final value and closes the gap exponentially, so successive
    vintages differ -- which is what makes them genuine revision events -- and
    the revisions stabilize, so a target lag can be recommended.
    """
    rows = []
    reference_dates = pd.date_range("2022-01-01", periods=days)
    for index, location in enumerate(locations):
        level = 100.0 * (index + 1)
        for offset, reference_date in enumerate(reference_dates):
            final = level * (1 + 0.1 * np.sin(offset / 30))
            for lag in range(max_lag + 1):
                completeness = 1 - 0.5 * float(np.exp(-lag / halflife))
                rows.append({"geo_value": location, "reference_date": reference_date,
                             "report_date": reference_date + pd.Timedelta(days=lag),
                             "value": round(final * completeness, 3)})
    return pd.DataFrame(rows)


@pytest.fixture
def make_archive(tmp_path):
    """Write an archive to a CSV and return its path."""
    def build(name: str = "archive.csv", **keywords):
        path = tmp_path / name
        archive_frame(**keywords).to_csv(path, index=False)
        return path
    return build
