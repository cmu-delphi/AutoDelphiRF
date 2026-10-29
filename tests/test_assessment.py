from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from autodelphirf.post_assessment.run import (assess_results, freeze_reliability_reference,
                                              save_assessment_outputs, sha256)
from autodelphirf.post_assessment import reporting
from autodelphirf.post_assessment.reporting import (_early_lag_location_summary,
                                                     _location_quantile_summary)
from autodelphirf.core.config import load_dataset_config, write_resolved_config
from autodelphirf.workflow.run import (_genuine_event_rows, forecast_time_assessment,
                                       run_reliability_and_calibration)


def test_reporting_uses_a_headless_matplotlib_backend():
    import matplotlib

    assert reporting.plt.get_backend().lower() == "agg"
    assert matplotlib.get_backend().lower() == "agg"


def test_report_with_no_matured_outcomes_explains_the_pending_state(tmp_path):
    """A real-time first origin can issue forecasts before any target matures."""
    output = tmp_path / "report"

    reporting.build_report(pd.DataFrame(), output, dataset_name="new-deployment")

    report = (output / "report.html").read_text()
    assert "new-deployment" in report
    assert "No forecast outcomes are observable yet" in report
    assert (output / "tables" / "forecast_evaluation_status.csv").is_file()


def test_web_score_bands_weight_locations_equally_after_within_location_averaging():
    scored = pd.DataFrame({
        "lag": [1, 1, 1, 1],
        "method": ["delphirf"] * 4,
        "geo_value": ["a", "a", "b", "b"],
        "absolute_error": [1.0, 3.0, 10.0, 14.0],
        "wis_reporting": [2.0, 4.0, 20.0, 24.0],
        "relative_error_pct": [5.0, 7.0, 20.0, 24.0],
        "relative_wis_pct": [6.0, 8.0, 22.0, 26.0],
    })
    summary = _location_quantile_summary(scored, ["lag", "method"])
    ae = summary[summary.metric.eq("ae")].iloc[0]
    wis = summary[summary.metric.eq("wis")].iloc[0]
    assert ae.n_locations == 2
    assert ae["median"] == pytest.approx(7.0)  # median of location means 2 and 12
    assert ae.q10 == pytest.approx(3.0)
    assert ae.q90 == pytest.approx(11.0)
    assert wis["median"] == pytest.approx(12.5)
    assert set(summary.metric) == {"ae", "wis", "ae_pct", "wis_pct"}


def test_early_lag_location_summary_reports_mean_and_sample_sd_through_day_14():
    scored = pd.DataFrame({
        "lag": [1, 14, 15, 1, 14],
        "method": ["delphirf"] * 5,
        "geo_value": ["a", "a", "a", "b", "b"],
        "absolute_error": [1.0, 3.0, 100.0, 4.0, 8.0],
        "wis_reporting": [2.0, 6.0, 200.0, float("nan"), float("nan")],
        "relative_error_pct": [5.0, 7.0, 100.0, 8.0, 12.0],
        "relative_wis_pct": [6.0, 10.0, 200.0, float("nan"), float("nan")],
    })

    summary = _early_lag_location_summary(scored)
    a_ae = summary[(summary.geo_value == "a") & (summary.metric == "ae")].iloc[0]
    b_ae = summary[(summary.geo_value == "b") & (summary.metric == "ae")].iloc[0]

    assert a_ae["mean"] == pytest.approx(2.0)
    assert a_ae["std"] == pytest.approx(2 ** 0.5)
    assert a_ae.n == 2 and a_ae.maximum_lag == 14
    assert b_ae["mean"] == pytest.approx(6.0)
    assert set(summary[summary.metric.eq("wis")].geo_value) == {"a"}
    assert set(summary.metric) == {"ae", "wis", "ae_pct", "wis_pct"}


def config_payload(tmp_path, results):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    return {
        "name": "example",
        "prepared_dir": str(prepared),
        "output_dir": str(results),
        "working_scale": "log_value",
        "model_methods": ["baseline_null", "delphirf"],
        "prediction_layers": ["baseline_null", "delphirf"],
        "comparator_methods": {"delphirf": "RevRoute DelphiRF"},
        "comparator_quantile_template": "q{tau:g}",
        "uncertainty_layer": True,
    }


def write_restart_bundle(tmp_path, *, reliability_reference=None):
    """Create the complete, model-free side of the forecast/assessment boundary."""
    results = tmp_path / "results"
    results.mkdir()
    source = tmp_path / "source.json"
    payload = config_payload(tmp_path, results)
    if reliability_reference is not None:
        payload["reliability_reference_file"] = str(reliability_reference)
    source.write_text(json.dumps(payload))
    config = load_dataset_config(source)
    reliability_record = None
    if reliability_reference is not None:
        config, reliability_record = freeze_reliability_reference(config, results)
    write_resolved_config(config, results / "resolved_config.json")
    (results / "assessment_inputs.json").write_text(json.dumps({
        "reliability_reference": reliability_record}) + "\n")
    frame = pd.DataFrame({
        "fold": [1], "cutoff": ["2024-01-01"], "geo_value": ["aa"],
        "reference_date": ["2023-12-31"], "report_date": ["2024-01-01"],
        "target_date": ["2024-01-28"], "lag": [1], "genuine_event": [True],
    })
    frame.to_csv(results / "predictions.csv.gz", index=False, compression="gzip")
    frame.to_csv(results / "predictions_wide.csv.gz", index=False, compression="gzip")
    return results, config, reliability_record


def test_resolved_config_round_trips_every_assessment_setting(tmp_path):
    source = tmp_path / "source.json"
    results = tmp_path / "results"
    source.write_text(json.dumps(config_payload(tmp_path, results)))
    original = load_dataset_config(source)
    snapshot = write_resolved_config(original, results / "resolved_config.json")
    loaded = load_dataset_config(snapshot)
    assert loaded == original
    raw = json.loads(snapshot.read_text())
    assert raw["working_scale"] == "log_value"
    assert raw["comparator_quantile_template"] == "q{tau:g}"


def test_reliability_reference_is_copied_and_hashed(tmp_path):
    results = tmp_path / "results"
    source = tmp_path / "reference.csv"
    source.write_text("q_error,q_regret\n1,2\n")
    config_file = tmp_path / "source.json"
    payload = config_payload(tmp_path, results)
    payload["reliability_reference_file"] = str(source)
    config_file.write_text(json.dumps(payload))
    frozen, record = freeze_reliability_reference(load_dataset_config(config_file), results)
    assert frozen.reliability_reference_file.is_file()
    assert frozen.reliability_reference_file.parent == results / "inputs"
    assert record["sha256"] and record["bytes"] == source.stat().st_size


def test_compressed_reliability_reference_keeps_both_suffixes(tmp_path):
    results = tmp_path / "results"
    source = tmp_path / "reference.csv.gz"
    pd.DataFrame({"q_error": [1]}).to_csv(source, index=False, compression="gzip")
    config_file = tmp_path / "source.json"
    payload = config_payload(tmp_path, results)
    payload["reliability_reference_file"] = str(source)
    config_file.write_text(json.dumps(payload))

    frozen, record = freeze_reliability_reference(load_dataset_config(config_file), results)

    assert frozen.reliability_reference_file.name == "reliability_reference.csv.gz"
    assert record["sha256"] == sha256(source)


def test_assessment_refuses_a_modified_frozen_reliability_reference(tmp_path):
    source = tmp_path / "reference.csv"
    source.write_text("q_error,q_regret\n1,2\n")
    results, config, _ = write_restart_bundle(
        tmp_path, reliability_reference=source)
    config.reliability_reference_file.write_text("q_error,q_regret\n3,4\n")

    with pytest.raises(ValueError, match="does not match assessment_inputs"):
        assess_results(results)


def test_assessment_restarts_from_predictions_without_model_fitting(tmp_path, monkeypatch):
    results, _, _ = write_restart_bundle(tmp_path)
    pd.DataFrame({"method": ["delphirf"], "standalone_total_seconds": [12.5]}).to_csv(
        results / "runtime_by_method.csv", index=False)
    empty = pd.DataFrame()
    called = {}

    def fake_assessment(long, config, wide=None, mode="full"):
        called["rows"] = (len(long), len(wide))
        called["mode"] = mode
        called["date_types"] = (long.cutoff.dtype, wide.reference_date.dtype)
        return {"calibration": empty, "interval_calibration": empty,
                "reliability": empty, "bootstrap": empty,
                "v1_example": pd.DataFrame({"passed": [True]})}

    def fake_report(data, output, **kwargs):
        called["runtime"] = kwargs["runtime"]
        called["validation"] = kwargs["validation"]
        output.mkdir(parents=True)
        (output / "report.html").write_text("ok")

    monkeypatch.setattr("autodelphirf.post_assessment.run.run_reliability_and_calibration", fake_assessment)
    monkeypatch.setattr("autodelphirf.post_assessment.run.build_report", fake_report)
    output = assess_results(results, tmp_path / "assessment")
    assert called["rows"] == (1, 1)
    assert called["mode"] == "full"
    assert all(str(dtype).startswith("datetime64") for dtype in called["date_types"])
    assert called["runtime"].standalone_total_seconds.tolist() == [12.5]
    assert list(called["validation"]) == ["v1_example"]
    assert (output / "v1_example.csv").is_file()
    manifest = json.loads((output / "assessment_manifest.json").read_text())
    assert manifest["model_fitting_performed"] is False
    assert set(manifest["inputs"]) == {
        "predictions.csv.gz", "predictions_wide.csv.gz", "resolved_config.json"}
    for name, record in manifest["inputs"].items():
        assert record["sha256"] == sha256(results / name)
        assert record["bytes"] == (results / name).stat().st_size
    assert (output / "report" / "report.html").is_file()
    with pytest.raises(FileExistsError, match="not empty"):
        assess_results(results, output)


def test_assessment_manifest_includes_the_verified_reliability_reference(tmp_path, monkeypatch):
    source = tmp_path / "reference.csv"
    source.write_text("q_error,q_regret\n1,2\n")
    results, _, record = write_restart_bundle(tmp_path, reliability_reference=source)
    empty = pd.DataFrame()
    monkeypatch.setattr(
        "autodelphirf.post_assessment.run.run_reliability_and_calibration",
        lambda *args, **kwargs: {"calibration": empty, "interval_calibration": empty,
                                "reliability": empty, "bootstrap": empty})
    monkeypatch.setattr("autodelphirf.post_assessment.run.build_report", lambda *args, **kwargs: None)

    output = assess_results(results, tmp_path / "assessment")
    manifest = json.loads((output / "assessment_manifest.json").read_text())

    frozen = results / record["bundled_path"]
    assert manifest["inputs"]["reliability_reference"]["sha256"] == sha256(frozen)
    assert manifest["model_fitting_performed"] is False


def test_assessment_writes_optional_calibration_diagnostics(tmp_path):
    empty = pd.DataFrame()
    outputs = {
        "calibration": empty, "interval_calibration": empty,
        "reliability": empty, "bootstrap": empty,
        "calibration_variants": pd.DataFrame({"calibration_strategy": ["quantile_map_pool"]}),
        "calibration_comparison": pd.DataFrame({"strategy": ["endpoint"]}),
        "quantile_reliability": pd.DataFrame({"strategy": ["base"]}),
        "pit_diagnostics": pd.DataFrame({"pit_lower": [.1], "pit_upper": [.25]}),
    }
    save_assessment_outputs(outputs, tmp_path)
    assert {"calibration_variants.csv.gz", "calibration_strategy_comparison.csv",
            "quantile_reliability.csv", "pit_diagnostics.csv.gz"} <= {
                path.name for path in tmp_path.iterdir()}


def test_pit_only_writes_diagnostics_without_recalibration_files(tmp_path):
    empty = pd.DataFrame()
    outputs = {
        "calibration": empty, "interval_calibration": empty,
        "reliability": empty, "bootstrap": empty,
        "calibration_variants": empty, "calibration_comparison": empty,
        "quantile_reliability": pd.DataFrame({"strategy": ["base"]}),
        "pit_diagnostics": pd.DataFrame({"strategy": ["base"]}),
    }

    save_assessment_outputs(outputs, tmp_path, mode="pit")

    names = {path.name for path in tmp_path.iterdir()}
    assert names == {"quantile_reliability.csv", "pit_diagnostics.csv.gz"}


def test_recalibration_only_writes_exact_recalibration_file_set(tmp_path):
    empty = pd.DataFrame()
    outputs = {
        "calibration": empty, "interval_calibration": empty,
        "reliability": empty, "bootstrap": empty,
        "calibration_variants": empty, "calibration_comparison": empty,
        "quantile_reliability": empty, "pit_diagnostics": empty,
    }

    save_assessment_outputs(outputs, tmp_path, mode="recalibration")

    assert {path.name for path in tmp_path.iterdir()} == {
        "calibration.csv.gz", "interval_calibration.csv",
        "calibration_variants.csv.gz", "calibration_strategy_comparison.csv",
        "quantile_reliability.csv", "pit_diagnostics.csv.gz",
    }


@pytest.mark.parametrize("mode", ["pit", "recalibration"])
def test_reliability_and_calibration_modes_return_only_their_outputs(monkeypatch, mode):
    row = pd.DataFrame({"method": ["delphirf"], "truth": [1.]})
    for tau in (.01, .025, .1, .25, .5, .75, .9, .975, .99):
        row[f"q{tau:g}"] = tau
        row[f"cal_q{tau:g}"] = tau
    row["cal_wis_reporting"] = 1.
    calibrated = row.assign(calibration_strategy="endpoint")
    variants = row.assign(calibration_strategy="quantile_map_location_lag")
    quantile_reliability = pd.DataFrame({"strategy": ["base"]})
    pit = pd.DataFrame({"strategy": ["base"]})
    monkeypatch.setattr("autodelphirf.workflow.run.prospective_calibration",
                        lambda *args, **kwargs: calibrated.copy())
    monkeypatch.setattr("autodelphirf.workflow.run.prospective_quantile_mapping",
                        lambda *args, **kwargs: variants.copy())
    monkeypatch.setattr("autodelphirf.workflow.run.pit_diagnostics",
                        lambda *args, **kwargs: (quantile_reliability, pit))
    monkeypatch.setattr("autodelphirf.workflow.run.LEVELS", (50,))
    monkeypatch.setattr("autodelphirf.workflow.run.add_interval_columns",
                        lambda frame, *args, **kwargs: frame.assign(
                            coverage_50=True, width_50=1., lower_miss_50=False,
                            upper_miss_50=False, wis_reporting=1., cal_wis_reporting=1.))
    config = SimpleNamespace(
        comparator_quantile_template=None, comparator_methods=[],
        prediction_layers=["delphirf"], uncertainty_layer=True,
        working_scale="identity")

    outputs = run_reliability_and_calibration(row, config, mode=mode)

    assert not outputs["quantile_reliability"].empty
    assert not outputs["pit_diagnostics"].empty
    assert outputs["reliability"].empty
    assert outputs["bootstrap"].empty
    if mode == "pit":
        assert outputs["calibration"].empty
        assert outputs["calibration_variants"].empty
        assert outputs["calibration_comparison"].empty
        assert outputs["interval_calibration"].empty
    else:
        assert not outputs["calibration"].empty
        assert not outputs["calibration_variants"].empty
        assert not outputs["calibration_comparison"].empty
        assert not outputs["interval_calibration"].empty


def test_assessment_defaults_to_genuine_events_and_can_include_all(tmp_path, monkeypatch):
    results, _, _ = write_restart_bundle(tmp_path)
    frame = pd.read_csv(results / "predictions.csv.gz")
    frame = pd.concat([frame.assign(genuine_event=True),
                       frame.assign(genuine_event=False, geo_value="bb")], ignore_index=True)
    frame.to_csv(results / "predictions.csv.gz", index=False, compression="gzip")
    frame.to_csv(results / "predictions_wide.csv.gz", index=False, compression="gzip")
    seen = []
    empty = pd.DataFrame()

    def fake_assessment(long, config, wide=None, **kwargs):
        seen.append((len(long), len(wide), kwargs))
        return {"calibration": empty, "interval_calibration": empty,
                "reliability": empty, "bootstrap": empty,
                "calibration_variants": empty, "calibration_comparison": empty,
                "quantile_reliability": empty, "pit_diagnostics": empty}

    monkeypatch.setattr(
        "autodelphirf.post_assessment.run.run_reliability_and_calibration", fake_assessment)
    assess_results(results, tmp_path / "genuine", assessment_mode="pit")
    assess_results(results, tmp_path / "all", assessment_mode="pit", event_filter="all")
    assess_results(results, tmp_path / "carried", assessment_mode="pit",
                   event_filter="carried")

    assert seen[0][0:2] == (1, 1)
    assert seen[0][2]["mode"] == "pit"
    assert seen[1][0:2] == (2, 2)
    assert seen[2][0:2] == (1, 1)


def test_carried_filter_requires_event_provenance(tmp_path):
    results, _, _ = write_restart_bundle(tmp_path)
    for name in ("predictions.csv.gz", "predictions_wide.csv.gz"):
        path = results / name
        frame = pd.read_csv(path).drop(columns="genuine_event")
        frame.to_csv(path, index=False, compression="gzip")

    with pytest.raises(ValueError, match="requires a genuine_event column"):
        assess_results(results, tmp_path / "assessment", assessment_mode="pit",
                       event_filter="carried")


def test_carried_only_update_intentionally_has_no_calibrated_rows():
    carried = pd.DataFrame({"genuine_event": [False]})

    outputs = forecast_time_assessment(carried, carried, config=None)

    assert outputs["calibration"].empty
    assert outputs["calibration_variants"].empty
    assert outputs["pit_diagnostics"].empty


def test_forecast_time_assessment_filters_carried_rows_from_every_history_path(monkeypatch):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    history = pd.DataFrame({
        "method": "delphirf", "cutoff": dates, "geo_value": ["a", "b", "c"],
        "reference_date": dates, "report_date": dates, "target_date": dates,
        "lag": 1, "genuine_event": [True, False, True],
    })
    newly_issued = history.iloc[1:].copy()
    seen = []

    def capture(frame, *args, **kwargs):
        seen.append(frame.copy())
        result = frame.copy()
        if "strategy" in kwargs:
            result["calibration_strategy"] = f"quantile_map_{kwargs['strategy']}"
        return result

    monkeypatch.setattr("autodelphirf.workflow.run.prospective_calibration", capture)
    monkeypatch.setattr("autodelphirf.workflow.run.prospective_quantile_mapping", capture)
    monkeypatch.setattr("autodelphirf.workflow.run.reliability_history", capture)
    monkeypatch.setattr("autodelphirf.workflow.run.pit_diagnostics",
                        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()))
    config = SimpleNamespace(
        comparator_quantile_template=None, comparator_methods=[],
        prediction_layers=["delphirf"], uncertainty_layer=True,
        working_scale="identity", reliability_reference_file=None)

    outputs = forecast_time_assessment(history, newly_issued, config)

    assert seen
    assert all(frame.genuine_event.all() for frame in seen)
    assert set(outputs["calibration"].geo_value) == {"c"}


def test_run_pipeline_and_forecast_assessment_share_the_genuine_event_filter():
    frame = pd.DataFrame({"genuine_event": [True, False, None], "value": [1, 2, 3]})

    filtered = _genuine_event_rows(frame)

    assert filtered.value.tolist() == [1]


@pytest.mark.parametrize("missing", ["predictions.csv.gz", "resolved_config.json"])
def test_assessment_names_each_missing_required_input(tmp_path, missing):
    results, _, _ = write_restart_bundle(tmp_path)
    (results / missing).unlink()

    with pytest.raises(FileNotFoundError, match=missing):
        assess_results(results)
