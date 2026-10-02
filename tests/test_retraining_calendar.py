"""The rolling retraining calendar.

Every test origin is a retraining origin, so the calendar decides both how
current each model is and how long a run takes. These check the arithmetic the
UI previews and the R bridge then builds from, without going near R.
"""
from __future__ import annotations

import pytest

from autodelphirf.prepare import (DEFAULT_FIRST_ORIGIN_OFFSET_DAYS, DEFAULT_RETRAIN_DAYS,
                                     PreparationError, retraining_calendar)


def test_the_first_origin_sits_a_fixed_offset_into_the_archive():
    calendar = retraining_calendar("2020-01-01", "2020-12-31", 30)
    assert calendar["first_origin"] == "2020-03-01"          # 60 days after 2020-01-01
    assert calendar["history_days_at_first_origin"] == DEFAULT_FIRST_ORIGIN_OFFSET_DAYS


def test_origins_step_by_the_retraining_interval_up_to_the_last_report_date():
    calendar = retraining_calendar("2020-01-01", "2020-12-31", 30)
    # 2020-03-01 .. 2020-12-31 inclusive, every 30 days.
    assert calendar["n_origins"] == 11
    assert calendar["last_origin"] == "2020-12-26"
    assert calendar["last_origin"] <= "2020-12-31"


def test_the_last_fold_is_given_a_full_interval_to_be_scored_over():
    """Otherwise it spans whatever remainder the range left -- sometimes a day."""
    calendar = retraining_calendar("2020-01-01", "2020-12-31", 30)
    assert calendar["experiment_end_date"] == "2021-01-25"   # last origin + 30


def test_a_shorter_interval_means_proportionally_more_retrainings():
    fortnightly = retraining_calendar("2020-01-01", "2020-12-31", 14)
    monthly = retraining_calendar("2020-01-01", "2020-12-31", 28)
    assert fortnightly["n_origins"] > monthly["n_origins"]
    assert fortnightly["n_origins"] == pytest.approx(monthly["n_origins"] * 2, abs=1)


def test_a_user_first_origin_replaces_the_offset():
    calendar = retraining_calendar("2020-01-01", "2020-12-31", 30, first_origin="2020-07-01")
    assert calendar["first_origin"] == "2020-07-01"
    assert calendar["history_days_at_first_origin"] == 182


@pytest.mark.parametrize("first_origin,match", [
    ("2021-06-01", "after the archive's last report date"),
    ("2019-06-01", "before the archive's first report date"),
])
def test_a_first_origin_outside_the_archive_says_which_end(first_origin, match):
    with pytest.raises(PreparationError, match=match):
        retraining_calendar("2020-01-01", "2020-12-31", 30, first_origin=first_origin)


@pytest.mark.parametrize("retrain_days", [0, -7])
def test_a_non_positive_interval_is_refused(retrain_days):
    with pytest.raises(PreparationError, match="at least one day"):
        retraining_calendar("2020-01-01", "2020-12-31", retrain_days)


def test_an_archive_shorter_than_the_offset_still_yields_one_origin():
    """A last report date exactly at the offset is the shortest workable case."""
    calendar = retraining_calendar("2020-01-01", "2020-03-01", 14)
    assert calendar["n_origins"] == 1
    assert calendar["first_origin"] == calendar["last_origin"] == "2020-03-01"


def test_the_cadence_defaults_differ_by_reporting_resolution():
    """A weekly stream gains an observation per week; a daily one mostly noise."""
    assert DEFAULT_RETRAIN_DAYS["weekly"] == 14
    assert DEFAULT_RETRAIN_DAYS["daily"] == 30
