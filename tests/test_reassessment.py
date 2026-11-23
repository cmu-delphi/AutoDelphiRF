import numpy as np
import pandas as pd
import pytest

from autodelphirf.post_assessment.reassessment import (
    age_counts, average_precision, evaluate_by_age, reassess)


def _forecasts():
    # One target, daily issues on days 0..3; intervals of width 2 around the median.
    issue = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"])
    med = np.array([0.0, 0.5, 3.0, 3.5])
    return pd.DataFrame({
        "geo_value": "ca", "reference_date": pd.Timestamp("2021-12-31"), "report_date": issue,
        "q0.1_working": med - 1, "q0.5_working": med, "q0.9_working": med + 1,
        "miss": [1.0, 1.0, 0.0, np.nan], "target_date": pd.Timestamp("2022-03-01")})


def test_reassess_pairs_flags_and_scores():
    d = reassess(_forecasts(), horizon=2).sort_values(["issue_date", "age"]).reset_index(drop=True)
    # pairs within 2 days: (0,1),(0,2),(1,2),(1,3),(2,3)
    assert list(d.age) == [1, 2, 1, 2, 1]
    # day0 interval [-1,1]: new medians 0.5 (inside), 3.0 (outside)
    # day1 interval [-0.5,1.5]: 3.0 outside, 3.5 outside; day2 interval [2,4]: 3.5 inside
    assert list(d.flag) == [0, 1, 1, 1, 0]
    assert np.allclose(d.jump, [0.25, 1.5, 1.25, 1.5, 0.25])
    assert list(d.lead_days[:2]) == [58, 57]
    assert d.miss.notna().all()          # the last forecast (miss unknown) is never the earlier one


def test_horizon_excludes_older_pairs_and_same_day():
    d = reassess(_forecasts(), horizon=1)
    assert set(d.age) == {1}
    assert len(d) == 3


def test_zero_width_interval_has_flag_but_no_score():
    f = _forecasts()
    f.loc[0, ["q0.1_working", "q0.9_working"]] = 0.0
    d = reassess(f, horizon=1)
    row = d[d.issue_date == pd.Timestamp("2022-01-01")].iloc[0]
    assert row.flag == 1 and np.isnan(row.jump)


def test_boundary_is_not_flagged():
    f = _forecasts()
    f.loc[1, "q0.5_working"] = 1.0          # exactly on day-0 upper bound
    d = reassess(f, horizon=1)
    assert d[d.issue_date == pd.Timestamp("2022-01-01")].flag.iloc[0] == 0


def test_duplicate_issue_rows_rejected():
    f = pd.concat([_forecasts(), _forecasts().iloc[:1]])
    with pytest.raises(ValueError):
        reassess(f)


def test_average_precision_matches_definition_with_ties():
    y = np.array([1, 0, 1, 1, 0, 0])
    s = np.array([.9, .9, .7, .3, .3, .1])
    # thresholds .9: P=1/2,R=1/3; .7: P=2/3,R=2/3; .3: P=3/5,R=1; .1: P=1/2,R=1
    assert average_precision(y, s) == pytest.approx(1/3 * .5 + 1/3 * 2/3 + 1/3 * .6)
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    yy, ss = rng.integers(0, 2, 500), np.round(rng.random(500), 2)
    assert average_precision(yy, ss) == pytest.approx(sk.average_precision_score(yy, ss))


def test_evaluate_by_age_counts_and_rates():
    d = reassess(_forecasts(), horizon=2)
    c = age_counts(d)
    assert c.n.sum() == len(d)
    e = evaluate_by_age(d, n_boot=20).set_index("age")
    # age 1 decisions: (0,1) miss1 flag0, (1,2) miss1 flag1, (2,3) miss0 flag0
    assert e.loc[1, "n"] == 3 and e.loc[1, "base"] == pytest.approx(2/3)
    assert e.loc[1, "precision"] == 1.0 and e.loc[1, "false_omission"] == pytest.approx(.5)
    assert e.loc[1, "sensitivity"] == pytest.approx(.5) and e.loc[1, "specificity"] == 1.0
    assert e.loc[1, "lead"] == 58 - 1
