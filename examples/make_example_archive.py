#!/usr/bin/env python3
"""Generate a small synthetic revision archive in RevRoute's input format.

The point is to have something runnable that depends on no real surveillance
data, so `autodelphirf run` can be exercised end to end by anyone who has just
installed the package:

    python examples/make_example_archive.py --out example_archive.csv
    autodelphirf run --archive example_archive.csv --name example --out work/

The generator is not a model of any real stream and makes no claim to be. It
produces the *structure* RevRoute expects -- a value for each
(location, reference date, report date) that starts low and revises upward
toward a stable final value -- with enough locations, history, and
heterogeneity between locations for routing to have something to route on.

Columns written: geo_value, reference_date, report_date, value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: Per-location revision character: how much of the final value is missing at
#: lag 0, and how fast the gap closes. Distinct profiles are the point -- a
#: single shared profile would make every location's donor pool equivalent and
#: leave routing nothing to discriminate.
LOCATION_PROFILES = {
    "aa": {"level": 220.0, "initial_completeness": 0.50, "halflife": 6.0},
    "bb": {"level": 140.0, "initial_completeness": 0.65, "halflife": 5.0},
    "cc": {"level": 380.0, "initial_completeness": 0.30, "halflife": 11.0},
    "dd": {"level": 90.0, "initial_completeness": 0.60, "halflife": 7.0},
    "ee": {"level": 260.0, "initial_completeness": 0.40, "halflife": 9.0},
}


def build_archive(*, days: int, max_lag: int, seed: int,
                  profiles: dict | None = None) -> pd.DataFrame:
    """Build the archive as one row per (location, reference date, report date)."""
    rng = np.random.default_rng(seed)
    profiles = profiles or LOCATION_PROFILES
    start = pd.Timestamp("2022-01-03")
    reference_dates = pd.date_range(start, periods=days, freq="D")
    # A slow seasonal swing plus a weekly cycle, so the series is neither flat
    # nor pure noise; the revision process, not this shape, is what is modelled.
    season = 1 + 0.35 * np.sin(np.arange(days) * 2 * np.pi / 180)
    weekday = 1 + 0.12 * np.sin(np.arange(days) * 2 * np.pi / 7)

    rows = []
    for location, profile in profiles.items():
        final = profile["level"] * season * weekday
        final = final * rng.normal(1.0, 0.08, days).clip(0.6, 1.4)
        for index, reference_date in enumerate(reference_dates):
            target = float(final[index])
            for lag in range(max_lag + 1):
                # Completeness rises from its lag-0 value toward 1 on an
                # exponential schedule, with a little per-report noise so
                # successive vintages differ (a genuine revision event).
                gap = 1.0 - profile["initial_completeness"]
                completeness = 1.0 - gap * float(np.exp(-lag / profile["halflife"]))
                noise = rng.normal(1.0, 0.015) if lag < max_lag else 1.0
                rows.append({
                    "geo_value": location,
                    "reference_date": reference_date,
                    "report_date": reference_date + pd.Timedelta(days=lag),
                    "value": round(max(target * completeness * noise, 0.0), 3),
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("example_archive.csv"),
                        help="where to write the archive (default: example_archive.csv)")
    parser.add_argument("--days", type=int, default=400,
                        help="reference dates to generate (default: 400)")
    parser.add_argument("--max-lag", type=int, default=70,
                        help="reports per reference date, lag 0..N (default: 70)")
    parser.add_argument("--seed", type=int, default=20260916, help="random seed")
    arguments = parser.parse_args()

    archive = build_archive(days=arguments.days, max_lag=arguments.max_lag, seed=arguments.seed)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    archive.to_csv(arguments.out, index=False)
    print(f"Wrote {len(archive):,} rows for {archive.geo_value.nunique()} locations to "
          f"{arguments.out}")
    print(f"  reference dates {archive.reference_date.min().date()} .. "
          f"{archive.reference_date.max().date()}")
    print("\nNext:\n"
          f"  autodelphirf diagnose --archive {arguments.out}\n"
          f"  autodelphirf run --archive {arguments.out} --name example --out work/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
