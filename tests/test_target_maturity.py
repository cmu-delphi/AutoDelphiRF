"""One target definition everywhere: the lag-L value exists from t + L onward.

Before t + L a reference date has no target, whatever DelphiRF fills in: it is
forecast, but never trained on, scored, used to calibrate, or written to the
ledger as truth. Every target is dated t + L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autodelphirf.app import cli
from autodelphirf.core.history import (enforce_target_maturity, target_as_of_date,
                                       validate_target_dates)
from autodelphirf.post_assessment.reliability import eligible_history
from autodelphirf.pre_diagnosis.preparation import build_spec, run_r_bridge
from autodelphirf.workflow.operational import (append_forecast_ledger, read_forecast_ledger,
                                               refresh_forecast_ledger_truth)
from autodelphirf.workflow.run import build_cases

from conftest import archive_frame
from test_end_to_end import requires_delphirf

L = 7


def triangle(as_of: str) -> pd.DataFrame:
    """Reference dates 03-20 .. 03-31 as reported on ``as_of``, lag 1 each.

    Every target is carried forward: no release fell exactly on lag 7.
    Preparation dates each target t + L.
    """
    reference = pd.date_range("2022-03-20", "2022-03-31")
    return pd.DataFrame({
        "geo_value": "aa", "reference_date": reference,
        "report_date": reference + pd.Timedelta(days=1), "lag": 1,
        "target_date": reference + pd.Timedelta(days=L),
        "target_value_source": "carried_forward",
        "target_source_lag": float(L - 1), "target_lag": float(L),
        "value_target": 189., "log_value_target": np.log1p(189.),
        "log_value_7dav": np.log1p(180.), "log_value_raw": np.log1p(181.),
        "genuine_event": True,
    })


def test_a_target_does_not_exist_before_t_plus_l():
    original = triangle("2022-04-02")
    prepared = enforce_target_maturity(original, pd.Timestamp("2022-04-02"))
    by_date = prepared.set_index("reference_date")
    # 03-26 + 7 = 04-02: matured on the as-of date; 03-28 + 7 = 04-04: not yet.
    assert by_date.loc["2022-03-26", "log_value_target"] == pytest.approx(np.log1p(189.))
    assert by_date.loc["2022-03-26", "target_value_source"] == "carried_forward"
    assert np.isnan(by_date.loc["2022-03-28", "log_value_target"])
    assert np.isnan(by_date.loc["2022-03-28", "value_target"])
    assert pd.isna(by_date.loc["2022-03-28", "target_value_source"])
    assert by_date.loc["2022-03-28", "target_lag"] == L
    # The stored target date is never rewritten.
    assert prepared.target_date.equals(original.target_date)


def test_the_as_of_date_comes_only_from_the_prepared_schedule():
    recorded = pd.DataFrame({"target_as_of_date": ["2022-04-02"]})
    assert target_as_of_date(recorded) == pd.Timestamp("2022-04-02")
    legacy = pd.DataFrame({"test_date": ["2022-03-01"]})
    with pytest.raises(ValueError, match="no target_as_of_date"):
        target_as_of_date(legacy)


def test_a_target_date_other_than_t_plus_l_requires_a_rebuild():
    validate_target_dates(triangle("2022-04-02"), L)
    stale = triangle("2022-04-02").assign(
        target_date=lambda frame: frame.reference_date + pd.Timedelta(days=3))
    with pytest.raises(ValueError, match="incompatible"):
        validate_target_dates(stale, L)


def test_an_immature_case_is_forecast_without_truth():
    prepared = enforce_target_maturity(triangle("2022-04-02"), pd.Timestamp("2022-04-02"))
    schedule = pd.DataFrame({
        "test_date": pd.to_datetime(["2022-03-21"]), "testing_days": [30],
        "target_lag": [L], "target_column": ["log_value_target"],
        "experiment_end_date": pd.to_datetime(["2022-04-20"])})
    cases = build_cases(prepared, schedule).set_index("reference_date")
    assert np.isnan(cases.loc["2022-03-28", "truth"])
    assert cases.loc["2022-03-26", "truth"] == pytest.approx(np.log1p(189.))
    assert cases.loc["2022-03-28", "target_date"] == pd.Timestamp("2022-04-04")
    # Null carries forward the raw provisional value, the raw target's scale.
    assert cases.loc["2022-03-28", "Null"] == pytest.approx(np.log1p(181.))


def test_only_releases_before_the_target_lag_become_cases():
    prepared = enforce_target_maturity(triangle("2022-04-20"), pd.Timestamp("2022-04-20"))
    matured = prepared.iloc[[0]].assign(lag=L, report_date=pd.Timestamp("2022-03-27"))
    schedule = pd.DataFrame({
        "test_date": pd.to_datetime(["2022-03-21"]), "testing_days": [30],
        "target_lag": [L], "target_column": ["log_value_target"],
        "experiment_end_date": pd.to_datetime(["2022-04-20"])})
    cases = build_cases(pd.concat([prepared, matured]), schedule)
    assert cases.lag.max() < L


def test_history_is_eligible_only_after_its_target_date_and_with_truth():
    history = pd.DataFrame({
        "cutoff": pd.to_datetime(["2022-03-01"] * 3),
        "target_date": pd.to_datetime(["2022-03-31", "2022-04-04", "2022-03-30"]),
        "truth": [10., 11., np.nan]})
    eligible = eligible_history(history, pd.Timestamp("2022-04-02"))
    assert eligible.truth.tolist() == [10.]


def ledger_row(truth):
    return pd.DataFrame({
        "dataset": ["d"], "method": ["delphirf"], "cutoff": ["2022-03-21"],
        "geo_value": ["aa"], "reference_date": ["2022-03-28"],
        "report_date": ["2022-03-29"], "target_date": ["2022-03-31"], "lag": [1],
        "prediction": [196.], "prediction_working": [np.log1p(196.)],
        "provisional_value": [180.], "predicted_remaining_revision": [16.],
        "truth": [np.nan if truth is None else truth], "truth_working": [np.nan if truth is None else np.log1p(truth)],
        "absolute_error": [np.nan if truth is None else abs(truth - 196.)],
        "q0.5": [196.]})


def matured_triangle(value):
    return pd.DataFrame({
        "geo_value": ["aa"], "reference_date": pd.to_datetime(["2022-03-28"]),
        "report_date": pd.to_datetime(["2022-04-04"]),
        "target_date": pd.to_datetime(["2022-04-04"]),     # stored at preparation: t + L
        "log_value_target": [np.log1p(value)]})


def test_ledger_never_keeps_a_provisional_truth(tmp_path):
    # An older version stored 189 -- the value on 03-31 -- as the 03-28 truth.
    append_forecast_ledger(ledger_row(189.), tmp_path)
    # Refreshed on 04-02, the target date 04-04 has not come: no truth.
    early = refresh_forecast_ledger_truth(
        matured_triangle(189.), "log_value_target", "log_value", tmp_path,
        as_of_date=pd.Timestamp("2022-04-02"))
    saved = read_forecast_ledger(tmp_path)
    assert early["withheld_rows"] == 1
    assert np.isnan(saved.loc[0, "truth"]) and np.isnan(saved.loc[0, "absolute_error"])
    assert saved.loc[0, "target_date"] == pd.Timestamp("2022-04-04")
    # Refreshed on 04-10, the lag-7 value 195 is the truth.
    late = refresh_forecast_ledger_truth(
        matured_triangle(195.), "log_value_target", "log_value", tmp_path,
        as_of_date=pd.Timestamp("2022-04-10"))
    saved = read_forecast_ledger(tmp_path)
    assert late["resolved_rows"] == 1
    assert saved.loc[0, "truth"] == pytest.approx(195.)
    assert saved.loc[0, "absolute_error"] == pytest.approx(1.)
    assert saved.loc[0, "prediction"] == 196.


def test_ledger_refresh_corrects_a_stored_wrong_truth(tmp_path):
    append_forecast_ledger(ledger_row(189.), tmp_path)
    result = refresh_forecast_ledger_truth(
        matured_triangle(195.), "log_value_target", "log_value", tmp_path,
        as_of_date=pd.Timestamp("2022-04-10"))
    assert result["corrected_rows"] == 1
    assert read_forecast_ledger(tmp_path).loc[0, "truth"] == pytest.approx(195.)


def test_reissued_rows_take_the_current_runs_truth(tmp_path):
    append_forecast_ledger(ledger_row(189.), tmp_path)
    append_forecast_ledger(ledger_row(None), tmp_path)
    assert np.isnan(read_forecast_ledger(tmp_path).loc[0, "truth"])


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_target_lag_tolerances_are_not_options(make_archive, tmp_path, side):
    # The target is always the value as of lag L: no tolerance setting exists.
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--archive", "a.csv", "--name", "x",
                           f"--target-lag-{side}-tolerance", "3"])
    with pytest.raises(TypeError):
        build_spec("x", make_archive(), tmp_path / "x", target_lag=14, **{side: 3})
    spec = build_spec("x", make_archive(), tmp_path / "x", target_lag=14)
    assert not hasattr(spec, side) and side not in spec.to_r_spec()


@pytest.mark.parametrize("field", ["target_lag_lower_tolerance", "target_lag_upper_tolerance"])
def test_a_raw_config_cannot_set_a_target_lag_tolerance(tmp_path, make_archive, field):
    config = tmp_path / "raw.json"
    config.write_text(
        '{"name": "x", "archive": "%s", "preprocessing": '
        '{"target_lag": 14, "%s": 0}}' % (make_archive(), field))
    assert cli.main(["prepare", "--raw-config", str(config)]) == 2


def test_the_as_of_date_reaches_the_r_specification(make_archive, tmp_path):
    spec = build_spec("x", make_archive(), tmp_path / "x", target_lag=14,
                      as_of_date="2022-05-01")
    assert spec.to_r_spec()["as_of_date"] == "2022-05-01"


@requires_delphirf
def test_prepared_triangle_keeps_every_release_and_dates_targets_at_t_plus_l(tmp_path):
    archive = tmp_path / "archive.csv"
    frame = archive_frame(days=200, max_lag=20, locations=("aa",))
    frame = frame[pd.to_datetime(frame.report_date) <= "2022-07-22"]
    frame.to_csv(archive, index=False)
    # Origins 03-02, 05-01 and 06-30: the old cut-off at last origin + L would
    # have dropped every release after 07-07.
    spec = build_spec("x", archive, tmp_path / "x", target_lag=L, testing_days=60,
                      triangle_format="csv")
    run_r_bridge(spec)
    prepared = pd.read_csv(tmp_path / "x" / "aa.csv.gz",
                           parse_dates=["reference_date", "report_date", "target_date"])
    schedule = pd.read_csv(tmp_path / "x" / "test_dates.csv")
    assert pd.Timestamp(schedule.test_date.max()) == pd.Timestamp("2022-06-30")
    assert prepared.report_date.max() == pd.Timestamp("2022-07-22")
    assert pd.Timestamp(schedule.target_as_of_date.iloc[0]) == pd.Timestamp("2022-07-22")
    assert (prepared.target_date == prepared.reference_date + pd.Timedelta(days=L)).all()
    # Reference dates from 07-16 reach t + 7 after the archive ends: no target.
    immature = prepared.reference_date >= "2022-07-16"
    assert immature.any() and prepared.loc[immature, "value_target"].isna().all()
    # A matured target is the value as of lag 7 exactly, never a later release.
    mature = prepared[~immature].drop_duplicates("reference_date")
    releases = frame.assign(reference_date=pd.to_datetime(frame.reference_date),
                            report_date=pd.to_datetime(frame.report_date))
    at_lag = releases[releases.report_date == releases.reference_date + pd.Timedelta(days=L)]
    expected = at_lag.set_index("reference_date").value.reindex(mature.reference_date)
    assert np.allclose(mature.value_target.to_numpy(), expected.to_numpy())


@requires_delphirf
def test_an_as_of_date_replays_the_archive_without_later_releases(tmp_path):
    archive = tmp_path / "archive.csv"
    archive_frame(days=120, max_lag=40, locations=("aa",)).to_csv(archive, index=False)
    spec = build_spec("x", archive, tmp_path / "x", target_lag=14, testing_days=30,
                      triangle_format="csv", start_date="2022-03-02",
                      end_date="2022-04-15", as_of_date="2022-04-15")
    run_r_bridge(spec)
    prepared = pd.read_csv(tmp_path / "x" / "aa.csv.gz",
                           parse_dates=["reference_date", "report_date", "target_date"])
    assert prepared.report_date.max() == pd.Timestamp("2022-04-15")
    assert prepared.loc[prepared.target_date > "2022-04-15", "value_target"].isna().all()


def test_history_is_judged_at_each_cases_own_report_date():
    from autodelphirf.post_assessment.reliability import reliability_history
    rows = []
    # Twelve forecasts issued early in the fold; their targets exist from 03-05.
    for i in range(12):
        rows.append({"report_date": "2022-03-01", "target_date": "2022-03-05",
                     "reference_date": f"2022-02-{10 + i:02d}", "truth": 100. + i})
    # Two later forecasts in the same fold, issued before and after 03-05.
    rows.append({"report_date": "2022-03-03", "target_date": "2022-03-20",
                 "reference_date": "2022-03-02", "truth": np.nan})
    rows.append({"report_date": "2022-03-10", "target_date": "2022-03-27",
                 "reference_date": "2022-03-09", "truth": np.nan})
    frame = pd.DataFrame(rows).assign(
        method="delphirf", fold=1, cutoff=pd.Timestamp("2022-03-01"), geo_value="aa", lag=1,
        prediction=105., provisional_value=100.)
    for column in ("report_date", "target_date", "reference_date"):
        frame[column] = pd.to_datetime(frame[column])
    result = reliability_history(frame, "delphirf", min_history=10).set_index("report_date")
    assert result.loc["2022-03-03", "n_err"] != 12
    assert result.loc["2022-03-10", "n_err"] == 12
