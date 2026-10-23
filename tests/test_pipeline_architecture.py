from pathlib import Path
from types import SimpleNamespace
import json
import warnings

import numpy as np
import pandas as pd

from autodelphirf.comparators.red_prediction import replay
from autodelphirf.workflow.methods import (DELPHIRF_METHODS,
                                           train_requested_delphirf as train_delphirf)
from autodelphirf.models.revroute_delphirf import PRIMARY_METHOD_ID
from autodelphirf.workflow.run import build_cases
from autodelphirf.post_assessment.inference import primary_comparison_pairs
from autodelphirf.core.registry import DEFAULT_LAYERS, PREDICTION_LAYERS
from autodelphirf.core.resources import resource_path


def test_red_implementation_lives_only_with_comparators():
    from autodelphirf.comparators.red_prediction import replay as comparator_replay
    from autodelphirf.comparators.red_routing import compute_weights as comparator_weights

    assert comparator_replay.__module__ == "autodelphirf.comparators.red_prediction"
    assert comparator_weights.__module__ == "autodelphirf.comparators.red_routing"
    revroute = Path(__file__).parents[1] / "src" / "autodelphirf" / "revroute"
    assert not (revroute / "prediction.py").exists()
    assert not (revroute / "routing.py").exists()


def test_revroute_package_contains_only_pooling_strategy():
    revroute = Path(__file__).parents[1] / "src" / "autodelphirf" / "revroute"
    modules = {path.name for path in revroute.glob("*.py")}
    assert modules == {"__init__.py"}
    assert (revroute / "pooling").is_dir()


def test_workflow_dispatches_primary_and_comparator_owned_methods():
    from autodelphirf.models.revroute_delphirf import PRIMARY_METHODS
    from autodelphirf.comparators.delphirf import COMPARATOR_DELPHIRF_METHODS

    assert PRIMARY_METHODS == {"delphirf": "RevRoute DelphiRF"}
    assert set(COMPARATOR_DELPHIRF_METHODS) == {
        "naive_delphirf", "similarity_weighted_delphirf", "global_delphirf"}
    assert DELPHIRF_METHODS == {**PRIMARY_METHODS, **COMPARATOR_DELPHIRF_METHODS}


def test_primary_dispatch_builds_revroute_pools_without_comparator_adapter(tmp_path, monkeypatch):
    import autodelphirf.workflow.methods as dispatcher

    pools = tmp_path / "revroute_pools.csv"
    captured = {}
    monkeypatch.setattr(dispatcher, "prepare_revroute_delphirf",
                        lambda prepared, schedule, output: pools)

    def fake_backend(config, prepared, schedule, labels, output, pools=None):
        captured.update(labels=labels, pools=pools)
        return Path(output), 1.0, {}

    monkeypatch.setattr(dispatcher, "train_delphirf", fake_backend)
    _, _, provenance = dispatcher.train_requested_delphirf(
        SimpleNamespace(), pd.DataFrame(), pd.DataFrame(), ("delphirf",),
        tmp_path / "predictions.csv.gz")
    assert captured == {"labels": {"delphirf": "RevRoute DelphiRF"}, "pools": pools}
    assert provenance["primary_method_id"] == "revroute_delphirf"


def test_revroute_backend_does_not_enter_delphirf_lag_band_loop():
    source = resource_path("train_delphirf.R").read_text()
    primary = source.index('if ("RevRoute DelphiRF" %in% requested_methods)')
    lag_bands = source.index("for (group in lag_groups)")
    assert primary < lag_bands
    primary_block = source[primary:lag_bands]
    assert "fit_one(pool_train, pool_test" in primary_block
    assert "train_group" not in primary_block


def test_revroute_trains_pools_without_immediate_forecast_rows():
    source = resource_path("train_delphirf.R").read_text()
    assert "untested_pools <- setdiff" in source
    assert "train_only = TRUE" in source
    assert "make_predictions = !train_only" in source


def test_similarity_weights_are_anchored_at_each_forecast_origin():
    source = resource_path("train_delphirf.R").read_text()
    # One set of anchors per origin, from all data reported by the cutoff and
    # kept per location, passed explicitly to every weighted fit.
    assert ('anchors <- DelphiRF::create_weight_anchors(as.data.frame(all_data), cutoff,'
            in source)
    assert 'by = "geo_value")' in source
    assert "DelphiRF::add_weights_related(as.data.frame(train), anchors)" in source
    assert source.count("anchors = anchors") == 5
    assert "add_weights_related(fe$train)" not in source


def test_scheduled_retraining_is_not_gated_by_new_forecast_cases():
    source = (Path(__file__).parents[1] / "src" / "autodelphirf" /
              "workflow" / "run.py").read_text()
    assert "retraining_required or not incremental_cases.empty" in source


def test_default_is_revroute_pooling_plus_real_delphirf():
    assert PRIMARY_METHOD_ID == "revroute_delphirf"
    assert DEFAULT_LAYERS == ("baseline_null", "delphirf")
    assert set(DELPHIRF_METHODS) == {
        "delphirf", "naive_delphirf", "similarity_weighted_delphirf", "global_delphirf"}
    assert set(DELPHIRF_METHODS) <= set(PREDICTION_LAYERS)


def test_default_postforecast_comparison_uses_the_carry_forward_reference():
    point, distribution = primary_comparison_pairs(
        {"null": "baseline_null", "candidate": "delphirf"})
    assert point == [("delphirf", "baseline_null")]
    assert distribution == []


def test_r_backend_calls_the_installed_delphirf_package():
    source = resource_path("train_delphirf.R").read_text()
    assert "library(DelphiRF)" in source
    assert "DelphiRF::revision_forecast" in source
    assert "DelphiRF::prospective_pooling_plan" not in source
    assert "prospective_pooling_plan(" in source
    assert "observation_weights_col" not in source
    assert "pkgload::load_all" not in source


def test_r_training_accepts_targets_observable_on_the_origin_date():
    source = resource_path("train_delphirf.R").read_text()
    # The stored target date is used as is; it is never rebuilt per fold.
    assert "report_date <= cutoff & target_date <= cutoff" in source
    assert "target_date > cutoff - schedule$training_days[[fold]]" in source
    assert "requested_maturity" not in source


def test_r_training_filters_genuine_rows_independently():
    source = resource_path("train_delphirf.R").read_text()
    assert 'genuine_training <- get_bool_flag("genuine-training", TRUE)' in source
    assert 'genuine_testing <- get_bool_flag("genuine-testing", TRUE)' in source
    assert "if (genuine_training) training <- training[genuine_event == TRUE]" in source
    assert "if (genuine_testing) testing <- testing[genuine_event == TRUE]" in source


def test_preprocessing_never_loads_a_delphirf_source_checkout():
    source = resource_path("prepare_triangle.R").read_text()
    assert "library(DelphiRF)" in source
    assert "pkgload" not in source
    assert "delphirf_dir" not in source


def test_test_window_can_vary_by_schedule_row():
    dates = pd.date_range("2024-01-01", periods=10)
    prepared = pd.DataFrame({
        "geo_value": "x", "reference_date": dates, "report_date": dates,
        "target_date": dates, "lag": 0, "target_lag": 7,
        "log_value_7dav": 1.0, "log_value_target_7dav": 2.0,
        "genuine_event": True,
    })
    schedule = pd.DataFrame({
        "test_date": pd.to_datetime(["2024-01-01", "2024-01-06"]),
        "testing_days": [2, 4], "training_days": [30, 60],
        "experiment_end_date": pd.to_datetime(["2024-01-10"] * 2),
        "target_column": ["log_value_target_7dav"] * 2,
    })
    cases = build_cases(prepared, schedule)
    assert cases.groupby("fold").size().to_dict() == {1: 2, 2: 4}


def test_replay_uses_each_folds_training_days(monkeypatch):
    seen = []
    monkeypatch.setattr("autodelphirf.comparators.red_prediction.rolling_completed_training",
                        lambda prepared, cutoff, days:
                        seen.append(days) or prepared)
    cases = pd.DataFrame({"fold": [1, 2], "cutoff": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                          "geo_value": ["x", "x"], "lag": [1, 1], "Null": [0., 0.]})
    prepared = pd.DataFrame({"lag": [1]})
    replay(cases, prepared, {1: 30, 2: 90}, include_red=False, target_lag=28)
    assert seen == [30, 90]


def test_delphirf_training_builds_an_installed_package_command(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    raw = tmp_path / "raw.csv"
    raw.write_text("reference_date,report_date,value,geo_value\n")
    metadata = {"raw_archive": str(raw), "resolved_preprocessing": {
        "reference_col": "reference_date", "report_col": "report_date",
        "value_cols": ["value"], "value_type": "count", "lag_terms": [1, 7],
        "weekday_groups": {"Mon": ["Mon"]}}}
    (prepared_dir / "preparation.json").write_text(json.dumps(metadata))
    (prepared_dir / "test_dates.csv").write_text("test_date\n")
    captured = {}
    monkeypatch.setattr("autodelphirf.models.delphirf_backend.rscript_executable", lambda: "Rscript")
    monkeypatch.setattr("autodelphirf.models.delphirf_backend.subprocess.run",
                        lambda command, check=False: captured.setdefault("command", command) or SimpleNamespace(returncode=0))
    # Make the fake result object explicit despite dict truthiness.
    def fake_run(command, check=False):
        captured["command"] = command
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("autodelphirf.models.delphirf_backend.subprocess.run", fake_run)
    config = SimpleNamespace(prepared_dir=prepared_dir, schedule_file="test_dates.csv",
                             genuine_training=False, genuine_testing=True)
    _, _, provenance = train_delphirf(
        config, pd.DataFrame(), pd.DataFrame(), ("naive_delphirf",),
        tmp_path / "predictions.csv.gz")
    command = captured["command"]
    assert "train_delphirf.R" in command[1]
    assert command[command.index("--weekdays-json") + 1] == '{"Mon": ["Mon"]}'
    assert command[command.index("--genuine-training") + 1] == "false"
    assert command[command.index("--genuine-testing") + 1] == "true"
    assert not any("DelphiRF" in part and Path(part).is_dir() for part in command)
    assert "primary_method_id" not in provenance


def test_delphirf_training_warns_when_r_reports_partial_fit_failures(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    raw = tmp_path / "raw.csv"
    raw.write_text("reference_date,report_date,value,geo_value\n")
    metadata = {"raw_archive": str(raw), "resolved_preprocessing": {
        "reference_col": "reference_date", "report_col": "report_date",
        "value_cols": ["value"], "value_type": "count", "lag_terms": [1, 7]}}
    (prepared_dir / "preparation.json").write_text(json.dumps(metadata))
    (prepared_dir / "test_dates.csv").write_text("test_date\n")
    output = tmp_path / "predictions.csv.gz"
    Path(str(output) + ".provenance.json").write_text(json.dumps({
        "fit_status": "partial_failure", "fit_failure_count": 2,
        "fit_failures_file": str(output) + ".fit_failures.csv"}))

    monkeypatch.setattr("autodelphirf.models.delphirf_backend.rscript_executable", lambda: "Rscript")
    monkeypatch.setattr("autodelphirf.models.delphirf_backend.subprocess.run",
                        lambda command, check=False: SimpleNamespace(returncode=0))
    config = SimpleNamespace(prepared_dir=prepared_dir, schedule_file="test_dates.csv",
                             genuine_training=True, genuine_testing=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, provenance = train_delphirf(
            config, pd.DataFrame(), pd.DataFrame(), ("naive_delphirf",), output)

    assert provenance["fit_failure_count"] == 2
    assert len(caught) == 1
    assert "predictions are incomplete" in str(caught[0].message)


def test_single_location_training_passes_a_string_geo_and_constant_label(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    raw = tmp_path / "raw.csv"
    raw.write_text("reference_date,report_date,value\n")
    metadata = {"raw_archive": str(raw), "resolved_preprocessing": {
        "reference_col": "reference_date", "report_col": "report_date",
        "value_cols": ["value"], "value_type": "count", "lag_terms": [1, 7],
        "geo_col": None, "constant_geo": "single_location"}}
    (prepared_dir / "preparation.json").write_text(json.dumps(metadata))
    (prepared_dir / "test_dates.csv").write_text("test_date\n")
    captured = {}

    def fake_run(command, check=False):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("autodelphirf.models.delphirf_backend.rscript_executable", lambda: "Rscript")
    monkeypatch.setattr("autodelphirf.models.delphirf_backend.subprocess.run", fake_run)
    config = SimpleNamespace(prepared_dir=prepared_dir, schedule_file="test_dates.csv",
                             genuine_training=True, genuine_testing=True)
    train_delphirf(config, pd.DataFrame(), pd.DataFrame(), ("naive_delphirf",),
                   tmp_path / "predictions.csv.gz")
    command = captured["command"]
    assert all(isinstance(argument, str) for argument in command)
    assert command[command.index("--geo-col") + 1] == "geo_value"
    assert command[command.index("--constant-geo") + 1] == "single_location"


def test_r_backend_builds_model_groups_from_canonical_weekday_columns():
    source = resource_path("train_delphirf.R").read_text()
    assert 'canonical_weekdays <- c("Mon", "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun")' in source
    assert "add_weekday_groups(all_data, weekday_groups)" in source
    assert "setdiff(base, canonical)" in source
    assert "params_list = alignment_params" in source
