"""M8: sparse reference dates with daily releases stay in daily mode, cleanly."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autodelphirf.app import cli
from autodelphirf.core.io import load_prepared_triangles, read_csv
from autodelphirf.pre_diagnosis.preparation import PreparationError, build_spec, run_r_bridge

from conftest import archive_frame
from test_end_to_end import requires_delphirf


def saturday_archive(path, weeks=60, max_lag=80):
    """Reference dates on Saturdays only, a release every day."""
    rows = []
    for k, reference in enumerate(pd.date_range("2022-01-01", periods=weeks, freq="7D")):
        for lag in range(max_lag):
            rows.append({"geo_value": "aa", "reference_date": reference,
                         "report_date": reference + pd.Timedelta(days=lag),
                         "value": 100 + k + 2 * min(lag, 10)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def daily_archive_with_gaps(path):
    """Daily reference dates; after lag 5 a release only every third day."""
    rows = []
    for k, reference in enumerate(pd.date_range("2022-01-01", periods=120)):
        for lag in range(40):
            if lag > 5 and lag % 3:
                continue
            rows.append({"geo_value": "bb", "reference_date": reference,
                         "report_date": reference + pd.Timedelta(days=lag),
                         "value": (50 + k) * (1 + 0.1 * min(lag, 5))})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


# --- settings -------------------------------------------------------------------

def test_sparse_reference_dates_keep_daily_mode_without_extra_smoothing(tmp_path):
    spec = build_spec("sat", saturday_archive(tmp_path / "sat.csv"), tmp_path / "sat",
                      target_lag=21)
    assert spec.temporal_resol == "daily"          # daily releases are kept
    assert spec.smoothed is True                   # no further 7-day averaging
    assert spec.lag_terms == (7, 14)               # actual preceding Saturdays


def test_daily_reference_dates_are_smoothed_by_default_and_can_opt_out(make_archive, tmp_path):
    archive = make_archive(max_lag=70)
    assert build_spec("d", archive, tmp_path / "d", target_lag=14).smoothed is False
    assert build_spec("d", archive, tmp_path / "d", target_lag=14, smoothed=True).smoothed is True


def test_seven_day_smoothing_of_sparse_reference_dates_is_refused(tmp_path):
    with pytest.raises(PreparationError, match="7-day smoothing needs daily reference dates"):
        build_spec("sat", saturday_archive(tmp_path / "sat.csv"), tmp_path / "sat",
                   target_lag=21, smoothed=False)


def test_lagged_features_cannot_point_between_reference_dates(tmp_path):
    with pytest.raises(PreparationError, match="multiples of the reference-date spacing"):
        build_spec("sat", saturday_archive(tmp_path / "sat.csv"), tmp_path / "sat",
                   target_lag=21, lag_terms=(1, 7))


# --- prepared triangle --------------------------------------------------------------

@requires_delphirf
def test_synthetic_reference_dates_are_removed_and_daily_releases_kept(tmp_path):
    archive = saturday_archive(tmp_path / "sat.csv")
    spec = build_spec("sat", archive, tmp_path / "sat", target_lag=21, triangle_format="csv")
    run_r_bridge(spec)
    triangle = load_prepared_triangles(tmp_path / "sat")
    triangle["reference_date"] = pd.to_datetime(triangle.reference_date)
    triangle["report_date"] = pd.to_datetime(triangle.report_date)
    assert set(triangle.reference_date.dt.day_name()) == {"Saturday"}
    # Every release before the target lag is a genuine row, on every weekday.
    raw = pd.read_csv(archive, parse_dates=["reference_date", "report_date"])
    assert triangle.genuine_event.sum() == (raw.report_date - raw.reference_date).dt.days.lt(21).sum()
    assert triangle.report_date.dt.dayofweek.nunique() == 7
    # The value features are the values themselves, not diluted by zero days.
    assert np.allclose(triangle.value_7dav, triangle.value_raw)
    row = triangle[(triangle.reference_date == "2022-03-05") & (triangle.lag == 3)].iloc[0]
    previous = triangle[(triangle.reference_date == "2022-02-26") & (triangle.lag == 10)].iloc[0]
    assert row.value_7dav_lag7 == previous.value_raw      # the previous Saturday, same report date


@requires_delphirf
def test_targets_are_read_at_exactly_t_plus_l_smoothed_or_not(tmp_path):
    raw = daily_archive_with_gaps(tmp_path / "daily.csv")
    spec = build_spec("daily", tmp_path / "daily.csv", tmp_path / "daily", target_lag=14,
                      smoothed_target=True, triangle_format="csv")
    run_r_bridge(spec)
    triangle = load_prepared_triangles(tmp_path / "daily")
    triangle["reference_date"] = pd.to_datetime(triangle.reference_date)
    raw["reference_date"] = pd.to_datetime(raw.reference_date)
    raw["report_date"] = pd.to_datetime(raw.report_date)

    def as_of(reference, report):          # forward filled, zero before the first release
        released = raw[(raw.reference_date == reference) & (raw.report_date <= report)]
        return released.sort_values("report_date").value.iloc[-1] if len(released) else 0.0

    reference = pd.Timestamp("2022-03-01")
    at = reference + pd.Timedelta(days=14)
    row = triangle[triangle.reference_date == reference].iloc[0]
    # No release on day 14 (14 % 3 != 0): the value is carried forward to t + L.
    assert row.target_value_source == "carried_forward"
    assert row.target_source_lag < row.target_lag
    assert row.value_target == pytest.approx(as_of(reference, at))
    expected = np.mean([as_of(reference - pd.Timedelta(days=v), at) for v in range(7)])
    assert row.value_target_7dav == pytest.approx(expected)
    assert row.target_lag == 14
    # Forward-filled rows of real episodes stay; only absent reference dates go.
    assert (~triangle.genuine_event.astype(bool)).any()


@requires_delphirf
def test_a_release_on_t_plus_l_is_labelled_exact_release(tmp_path):
    raw = daily_archive_with_gaps(tmp_path / "daily.csv")
    spec = build_spec("daily", tmp_path / "daily.csv", tmp_path / "daily", target_lag=15,
                      triangle_format="csv")
    run_r_bridge(spec)
    triangle = load_prepared_triangles(tmp_path / "daily")
    # Day 15 is a release day (15 % 3 == 0) even though the value no longer changes.
    assert set(triangle.target_value_source.dropna()) == {"exact_release"}
    assert (triangle.target_source_lag.dropna() == triangle.target_lag.dropna()).all()


# --- calendar indicators ---------------------------------------------------------------

@requires_delphirf
def test_constant_calendar_indicators_are_excluded_and_varying_ones_kept(tmp_path):
    archive = saturday_archive(tmp_path / "sat.csv")
    work = tmp_path / "work"
    assert cli.main(["run", "--archive", str(archive), "--name", "sat", "--out", str(work),
                     "--target-lag", "21", "--methods", "baseline_null,delphirf"]) == 0
    provenance = json.loads(next((work / "results").glob("*.provenance.json")).read_text())
    excluded = set(provenance["calendar_indicators_constant_in_triangle"])
    used = set(provenance["effective_base_params"])
    assert {"Mon_ref", "Weekends_ref"} <= excluded            # every reference date is a Saturday
    assert not any(column.endswith("_ref") for column in used)
    assert {"Mon_issue", "Weekends_issue"} <= used             # daily releases vary
    predictions = read_csv(work / "results" / "predictions.csv.gz")
    assert predictions[predictions.method.eq("delphirf")].model_available.all()
