"""Default smoothing, text location identifiers, and target-lag evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autodelphirf.app import cli
from autodelphirf.core.defaults import provisional_column
from autodelphirf.core.io import load_prepared_triangles, read_csv
from autodelphirf.pre_diagnosis.archive import diagnose_raw_archive
from autodelphirf.pre_diagnosis.preparation import build_spec, diagnose_archive, read_archive
from autodelphirf.revroute.pooling import plan as pooling_plan

from conftest import archive_frame
from test_end_to_end import requires_delphirf


# --- smoothing --------------------------------------------------------------

def test_a_daily_stream_defaults_to_the_unsmoothed_target(make_archive, tmp_path):
    spec = build_spec("x", make_archive(max_lag=70), tmp_path / "x", target_lag=14)
    assert spec.temporal_resol == "daily"
    assert spec.smoothed is False and spec.target_column == "log_value_target"
    assert "smoothed" not in spec.overrides


def test_smoothing_remains_a_user_choice(make_archive, tmp_path):
    spec = build_spec("x", make_archive(max_lag=70), tmp_path / "x", target_lag=14,
                      smoothed=True)
    assert spec.smoothed is True and spec.target_column == "log_value_target_7dav"
    assert "smoothed" in spec.overrides
    arguments = cli.build_parser().parse_args(
        ["run", "--archive", "a.csv", "--name", "x", "--smoothed"])
    assert arguments.smoothed is True


def test_the_provisional_value_is_on_the_targets_smoothing_level():
    assert provisional_column("log_value_target") == "log_value_raw"
    assert provisional_column("log_value_target_7dav") == "log_value_7dav"


def test_revroute_trajectories_use_the_targets_smoothing_level(tmp_path, monkeypatch):
    seen = {}

    class Geometry:
        tasks = [("aa", 1)]

    def fake_pooling(training, target_lag, **kwargs):
        seen.update(kwargs)
        return Geometry(), [0], .1, None, None

    monkeypatch.setattr(pooling_plan, "rolling_completed_training",
                        lambda prepared, cutoff, days: prepared)
    monkeypatch.setattr(pooling_plan, "sparse_task_pooling", fake_pooling)
    schedule = pd.DataFrame({"test_date": pd.to_datetime(["2022-03-01"]),
                             "training_days": [180], "target_lag": [14],
                             "target_column": ["log_value_target"]})
    pooling_plan.learn_task_pools(pd.DataFrame(), schedule, tmp_path / "pools.csv")
    assert seen["target_column"] == "log_value_target"
    assert seen["value_column"] == "log_value_raw"


# --- location identity -------------------------------------------------------

def test_locations_are_read_as_text(tmp_path):
    path = tmp_path / "frame.csv"
    pd.DataFrame({"geo_value": ["01", "1", "06"], "value": [1, 2, 3]}).to_csv(path, index=False)
    assert read_csv(path).geo_value.tolist() == ["01", "1", "06"]
    assert read_archive(path, geo_col="geo_value").geo_value.tolist() == ["01", "1", "06"]


def test_01_and_1_are_different_locations_in_diagnosis(tmp_path):
    frame = archive_frame(days=60, max_lag=10, locations=("01", "1"))
    path = tmp_path / "archive.csv"
    frame.to_csv(path, index=False)
    diagnosis = diagnose_archive(path, geo_col="geo_value", reference_col="reference_date",
                                 report_col="report_date", value_cols=("value",),
                                 value_type="count")
    assert diagnosis["n_locations"] == 2


def test_preserved_pools_keep_leading_zeros(tmp_path, monkeypatch):
    output = tmp_path / "revroute_pools.csv"
    pd.DataFrame({"fold": [1], "cutoff": ["2026-01-02"], "geo_value": ["01"],
                  "lag": [1], "pool": [17], "threshold": [.2]}).to_csv(output, index=False)
    pd.DataFrame({"fold": [1], "cutoff": ["2026-01-02"], "represented_tasks": [1]}).to_csv(
        tmp_path / "revroute_pool_summary.csv", index=False)

    class Geometry:
        tasks = [("06", 2)]

    monkeypatch.setattr(pooling_plan, "rolling_completed_training",
                        lambda prepared, cutoff, days: prepared)
    monkeypatch.setattr(pooling_plan, "sparse_task_pooling",
                        lambda *args, **kwargs: (Geometry(), [3], .4, None, None))
    schedule = pd.DataFrame({"test_date": pd.to_datetime(["2026-01-02", "2026-01-09"]),
                             "training_days": [30, 30], "target_lag": [28, 28],
                             "target_column": ["target", "target"]})
    pooling_plan.learn_task_pools(pd.DataFrame(), schedule, output, preserve_existing=True)
    assert read_csv(output).geo_value.tolist() == ["01", "06"]


@requires_delphirf
def test_numeric_looking_locations_run_end_to_end(tmp_path):
    archive = tmp_path / "archive.csv"
    archive_frame(days=300, max_lag=60, halflife=9.0,
                  locations=("01", "1", "06")).to_csv(archive, index=False)
    work = tmp_path / "work"
    assert cli.main(["run", "--archive", str(archive), "--name", "numeric",
                     "--out", str(work), "--target-lag", "21"]) == 0
    assert sorted(p.stem for p in (work / "numeric").glob("*.parquet")) == ["01", "06", "1"]
    predictions = read_csv(work / "results" / "predictions.csv.gz")
    delphirf = predictions[predictions.method.eq("delphirf")]
    assert set(delphirf.geo_value) == {"01", "1", "06"}
    assert delphirf.model_available.all()


# --- target-lag evidence ----------------------------------------------------

def episodes(reference_dates, lags, value_at):
    rows = []
    for reference in pd.to_datetime(reference_dates):
        for lag in lags:
            rows.append({"geo_value": "aa", "reference_date": reference,
                         "report_date": reference + pd.Timedelta(days=lag),
                         "value": value_at(lag)})
    return pd.DataFrame(rows)


def test_only_episodes_followed_to_lag_60_inform_the_target_lag():
    # Followed to lag 70: the value settles at lag 14.
    long = episodes(pd.date_range("2022-01-01", periods=30), range(0, 71),
                    lambda lag: 100. if lag >= 14 else 50.)
    # Last seen at lag 40, so their latest value is not a finalized value; on
    # their own they would suggest lag 1.
    short = episodes(pd.date_range("2022-03-01", periods=60), range(0, 41),
                     lambda lag: 100. if lag >= 1 else 50.)
    report = diagnose_raw_archive(pd.concat([long, short], ignore_index=True))
    assert report.target_lag_suggested_range["lower"] == 14


def test_an_archive_never_followed_to_lag_60_gives_no_evidence():
    short = episodes(pd.date_range("2022-01-01", periods=30), range(0, 41),
                     lambda lag: 100. if lag >= 7 else 50.)
    report = diagnose_raw_archive(short)
    assert report.target_lag_suggested_range["lower"] is None
    assert report.target_lag_completion_curve == {}


def test_a_specified_target_lag_skips_the_target_lag_diagnosis():
    long = episodes(pd.date_range("2022-01-01", periods=30), range(0, 71),
                    lambda lag: 100. if lag >= 14 else 50.)
    report = diagnose_raw_archive(long, user_target_lag=21)
    assert report.target_lag_suggested_range["skipped"] is True
    assert report.target_lag_completion_curve == {}
    assert report.resolved_target_lag == 21


def test_red_learns_remaining_revision_on_the_targets_smoothing_level():
    from autodelphirf.comparators.red_routing import build_curve_routing
    rows = []
    for reference in pd.date_range("2022-01-01", periods=12):
        for lag in range(0, 7):
            rows.append({"geo_value": "aa", "reference_date": reference, "lag": lag,
                         "genuine_event": True,
                         "log_value_raw": np.log1p(50. + 5 * lag),
                         "log_value_7dav": np.log1p(1000.),
                         "log_value_target": np.log1p(90.),
                         "log_value_target_7dav": np.log1p(2000.)})
    route = build_curve_routing(pd.DataFrame(rows), target_lag=7,
                                target_column="log_value_target", initial_lag=0)
    expected = np.log1p(90.) - np.log1p(50. + 5 * route.donors.lag.to_numpy())
    assert np.allclose(route.donors.real_remaining_revision.to_numpy(), expected)


def test_a_smoothed_target_keeps_the_seven_day_averaging(make_archive, tmp_path):
    spec = build_spec("x", make_archive(max_lag=70), tmp_path / "x", target_lag=14,
                      smoothed_target=True)
    assert spec.smoothed is False
    assert spec.target_column == "log_value_target_7dav"
    assert "smoothed_target" in spec.overrides
    assert provisional_column(spec.target_column) == "log_value_7dav"
    arguments = cli.build_parser().parse_args(
        ["run", "--archive", "a.csv", "--name", "x", "--smoothed-target"])
    assert arguments.smoothed_target is True and arguments.smoothed is None


@requires_delphirf
def test_a_fraction_with_a_smoothed_target_runs_end_to_end(tmp_path):
    frame = archive_frame(days=300, max_lag=60, halflife=9.0)
    frame["numerator"] = frame.value
    frame["denominator"] = (frame.value * 4 + 50).round(3)
    archive = tmp_path / "fraction.csv"
    frame.to_csv(archive, index=False)
    work = tmp_path / "work"
    assert cli.main(["run", "--archive", str(archive), "--name", "frac", "--out", str(work),
                     "--value-type", "fraction", "--value-col", "numerator",
                     "--value-col", "denominator", "--target-lag", "21",
                     "--smoothed-target", "--triangle-format", "csv"]) == 0
    triangle = load_prepared_triangles(work / "frac")
    for column in ("reference_date", "report_date"):
        triangle[column] = pd.to_datetime(triangle[column])
    # The 7-day average is applied: the smoothed value is not the raw one.
    assert not np.allclose(triangle.log_value_7dav.dropna(),
                           triangle.loc[triangle.log_value_7dav.notna(), "log_value_raw"])
    keys = ["geo_value", "reference_date", "report_date"]
    predictions = read_csv(work / "results" / "predictions.csv.gz",
                           parse_dates=["reference_date", "report_date"])
    scored = predictions[predictions.method.eq("baseline_null") & predictions.truth.notna()]
    joined = scored.merge(triangle[keys + ["log_value_target_7dav", "log_value_7dav"]],
                          on=keys, how="inner")
    assert len(joined) == len(scored) > 0
    assert np.allclose(joined.truth_working, joined.log_value_target_7dav)
    assert np.allclose(joined.prediction_working, joined.log_value_7dav)
    model = predictions[predictions.method.eq("delphirf")]
    assert model.model_available.all()
    available = model[model.prediction_working.notna()]
    assert len(available) > 0
    # A numerator/denominator target is a log ratio, so DelphiRF—not Python—
    # returns exp(working) on the reporting scale.
    assert np.allclose(available.prediction, np.exp(available.prediction_working))


@requires_delphirf
def test_a_one_column_rate_is_inverse_transformed_once_by_delphirf(tmp_path):
    archive = tmp_path / "rate.csv"
    frame = archive_frame(days=220, max_lag=45, halflife=8.0)
    frame["value"] = frame["value"] / 1000
    frame.to_csv(archive, index=False)
    work = tmp_path / "work"
    assert cli.main(["run", "--archive", str(archive), "--name", "rate",
                     "--out", str(work), "--value-type", "fraction",
                     "--target-lag", "21", "--triangle-format", "csv"]) == 0
    predictions = read_csv(work / "results" / "predictions.csv.gz")
    model = predictions[
        predictions.method.eq("delphirf") & predictions.prediction_working.notna()]
    assert len(model) > 0
    # A precomputed rate is still one value column and uses log(value + 1).
    assert np.allclose(model.prediction, np.expm1(model.prediction_working))


@requires_delphirf
def test_location_specific_comparators_forecast_every_lag_below_the_target(tmp_path):
    archive = tmp_path / "archive.csv"
    archive_frame(days=260, max_lag=75, halflife=9.0, locations=("aa",)).to_csv(
        archive, index=False)
    work = tmp_path / "work"
    assert cli.main(["run", "--archive", str(archive), "--name", "long", "--out", str(work),
                     "--target-lag", "70", "--methods", "baseline_null,naive_delphirf"]) == 0
    predictions = read_csv(work / "results" / "predictions.csv.gz")
    naive = predictions[predictions.method.eq("naive_delphirf")]
    assert naive.lag.max() == 69
    # The first origin has no matured L = 70 target to train on at any lag;
    # wherever the method could fit at all, lags 60-69 are forecast too.
    trained = naive[naive.groupby("fold").model_available.transform("any")]
    assert trained[trained.lag >= 60].model_available.all()


def test_the_target_lag_must_be_positive(make_archive, tmp_path):
    from autodelphirf.pre_diagnosis.preparation import PreparationError
    with pytest.raises((PreparationError, ValueError), match="between 1 and"):
        build_spec("x", make_archive(), tmp_path / "x", target_lag=0)


@requires_delphirf
def test_a_prepared_dataset_reruns_without_its_raw_archive(tmp_path):
    import json
    archive = tmp_path / "archive.csv"
    archive_frame(days=200, max_lag=40, halflife=8.0).to_csv(archive, index=False)
    work = tmp_path / "work"
    assert cli.main(["run", "--archive", str(archive), "--name", "rerun",
                     "--out", str(work), "--target-lag", "21"]) == 0
    recorded = json.loads((work / "rerun" / "preparation.json").read_text())["raw_archive_state"]
    archive.unlink()                     # the prepared artifacts are all a rerun needs
    assert cli.main(["run", "--config", str(work / "rerun.json")]) == 0
    manifest = json.loads((work / "results" / "run_manifest.json").read_text())
    assert manifest["raw_archive_update"]["sha256"] == recorded["sha256"]
    assert manifest["raw_archive_update"]["source"] == "recorded at preparation"
    predictions = read_csv(work / "results" / "predictions.csv.gz")
    assert predictions[predictions.method.eq("delphirf")].model_available.any()


@requires_delphirf
def test_no_post_assessment_writes_evaluated_predictions_and_stops(tmp_path):
    import json
    archive = tmp_path / "archive.csv"
    archive_frame(days=200, max_lag=40, halflife=8.0).to_csv(archive, index=False)
    work = tmp_path / "work"
    assert cli.main(["run", "--archive", str(archive), "--name", "quick", "--out", str(work),
                     "--target-lag", "21", "--methods", "baseline_null,naive_delphirf",
                     "--no-post-assessment"]) == 0
    results = work / "results"
    predictions = read_csv(results / "predictions.csv.gz")
    scored = predictions[predictions.truth.notna()]
    assert set(scored.method) == {"baseline_null", "naive_delphirf"}
    assert scored[["absolute_error", "wis_reporting", "relative_error_pct"]].notna().all().all()
    manifest = json.loads((results / "run_manifest.json").read_text())
    assert manifest["post_assessment"] is False
    # Ready for a later `autodelphirf update`: deployment origin and issued forecasts.
    assert (results / "deployment_state.json").is_file()
    assert (results / "forecast_ledger.csv.gz").is_file()
    # Test rows DelphiRF could not forecast are listed next to the fit failures.
    dropped = read_csv(results / "_delphirf_predictions.csv.gz.dropped_test_rows.csv")
    assert {"method", "cutoff", "geo_value", "reference_date", "report_date", "lag",
            "missing_features", "reason"} <= set(dropped.columns)
    assert manifest["delphirf_provenance"]["dropped_test_row_count"] == len(dropped)
    unavailable = predictions[predictions.method.eq("naive_delphirf")
                              & ~predictions.model_available.astype(bool)]
    assert len(unavailable) >= len(dropped)
    for skipped in ("calibration.csv.gz", "reliability.csv.gz",
                    "bootstrap_pairwise_comparisons.csv", "report"):
        assert not (results / skipped).exists()
