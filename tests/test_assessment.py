from __future__ import annotations

import json

import pandas as pd
import pytest

from autodelphirf.assessment import assess_results, freeze_reliability_reference, sha256
from autodelphirf.config import load_dataset_config, write_resolved_config


def config_payload(tmp_path, results):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    return {
        "name": "example",
        "prepared_dir": str(prepared),
        "output_dir": str(results),
        "value_transform": "log1p",
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
        "target_date": ["2024-01-28"], "lag": [1],
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
    assert raw["value_transform"] == "log1p"
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

    def fake_assessment(long, config, wide=None):
        called["rows"] = (len(long), len(wide))
        called["date_types"] = (long.cutoff.dtype, wide.reference_date.dtype)
        return {"calibration": empty, "interval_calibration": empty,
                "reliability": empty, "bootstrap": empty,
                "v1_example": pd.DataFrame({"passed": [True]})}

    def fake_report(data, output, **kwargs):
        called["runtime"] = kwargs["runtime"]
        called["validation"] = kwargs["validation"]
        output.mkdir(parents=True)
        (output / "report.html").write_text("ok")

    monkeypatch.setattr("autodelphirf.assessment.run_reliability_and_calibration", fake_assessment)
    monkeypatch.setattr("autodelphirf.assessment.build_report", fake_report)
    output = assess_results(results, tmp_path / "assessment")
    assert called["rows"] == (1, 1)
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
        "autodelphirf.assessment.run_reliability_and_calibration",
        lambda *args, **kwargs: {"calibration": empty, "interval_calibration": empty,
                                "reliability": empty, "bootstrap": empty})
    monkeypatch.setattr("autodelphirf.assessment.build_report", lambda *args, **kwargs: None)

    output = assess_results(results, tmp_path / "assessment")
    manifest = json.loads((output / "assessment_manifest.json").read_text())

    frozen = results / record["bundled_path"]
    assert manifest["inputs"]["reliability_reference"]["sha256"] == sha256(frozen)
    assert manifest["model_fitting_performed"] is False


@pytest.mark.parametrize("missing", [
    "predictions.csv.gz", "predictions_wide.csv.gz", "resolved_config.json"])
def test_assessment_names_each_missing_required_input(tmp_path, missing):
    results, _, _ = write_restart_bundle(tmp_path)
    (results / missing).unlink()

    with pytest.raises(FileNotFoundError, match=missing):
        assess_results(results)
