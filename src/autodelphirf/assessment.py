"""Restartable post-forecast assessment from frozen prediction artifacts."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from . import __version__
from .config import DatasetConfig, load_dataset_config, write_resolved_config
from .pipeline import run_reliability_and_calibration
from .report import build_report


DATE_COLUMNS = ("cutoff", "reference_date", "report_date", "target_date")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in DATE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def freeze_reliability_reference(config: DatasetConfig, results_dir: Path) -> tuple[DatasetConfig, dict | None]:
    """Copy an external reliability reference into the result bundle."""
    source = config.reliability_reference_file
    if source is None:
        return config, None
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reliability reference not found: {source}")
    inputs = Path(results_dir) / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    suffix = "".join(source.suffixes) or ".csv"
    destination = inputs / f"reliability_reference{suffix}"
    if source != destination.resolve():
        shutil.copy2(source, destination)
    record = {"source": str(source), "bundled_path": str(destination.relative_to(results_dir)),
              "sha256": sha256(destination), "bytes": destination.stat().st_size}
    return replace(config, reliability_reference_file=destination.resolve()), record


def save_assessment_outputs(outputs: dict[str, pd.DataFrame], output: Path) -> None:
    outputs["calibration"].to_csv(output / "calibration.csv.gz", index=False, compression="gzip")
    outputs["interval_calibration"].to_csv(output / "interval_calibration.csv", index=False)
    outputs["reliability"].to_csv(output / "reliability.csv.gz", index=False, compression="gzip")
    outputs["bootstrap"].to_csv(output / "bootstrap_pairwise_comparisons.csv", index=False)
    for name, frame in outputs.items():
        if name.startswith(("v1_", "v2_", "v3_", "v4_")) and frame is not None and not frame.empty:
            frame.to_csv(output / f"{name}.csv", index=False)


def assess_results(results_dir: Path, output_dir: Path | None = None) -> Path:
    """Recompute assessment and report files without fitting any model."""
    results = Path(results_dir).resolve()
    required = [results / "predictions.csv.gz", results / "predictions_wide.csv.gz",
                results / "resolved_config.json"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"result directory {results} is missing assessment input(s): {', '.join(missing)}")
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = results / "assessments" / stamp
    else:
        output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"assessment output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = load_dataset_config(results / "resolved_config.json")
    frozen_inputs_path = results / "assessment_inputs.json"
    frozen_inputs = (json.loads(frozen_inputs_path.read_text())
                     if frozen_inputs_path.is_file() else {})
    reliability_input = frozen_inputs.get("reliability_reference")
    if reliability_input is not None:
        reliability_path = results / reliability_input["bundled_path"]
        if not reliability_path.is_file():
            raise FileNotFoundError(f"frozen reliability reference not found: {reliability_path}")
        if sha256(reliability_path) != reliability_input["sha256"]:
            raise ValueError(
                "frozen reliability reference does not match assessment_inputs.json: "
                f"{reliability_path}")
    long = _read_predictions(results / "predictions.csv.gz")
    wide = _read_predictions(results / "predictions_wide.csv.gz")
    outputs = run_reliability_and_calibration(long, config, wide=wide)
    save_assessment_outputs(outputs, output)
    runtime_path = results / "runtime_by_method.csv"
    runtime = pd.read_csv(runtime_path) if runtime_path.is_file() else None
    build_report(long, output / "report", wide=wide, runtime=runtime,
                 interval_calibration=outputs["interval_calibration"],
                 reliability=outputs["reliability"], bootstrap=outputs["bootstrap"],
                 validation={k: v for k, v in outputs.items()
                             if k.startswith(("v1_", "v2_", "v3_", "v4_"))})
    assessment_config = replace(config, output_dir=output)
    write_resolved_config(assessment_config, output / "resolved_config.json")
    inputs = {}
    for name in ("predictions.csv.gz", "predictions_wide.csv.gz", "resolved_config.json"):
        path = results / name
        inputs[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    if reliability_input is not None:
        reliability_path = results / reliability_input["bundled_path"]
        inputs["reliability_reference"] = {
            "path": str(reliability_path), "sha256": sha256(reliability_path),
            "bytes": reliability_path.stat().st_size}
    manifest = {
        "generated_by": "autodelphirf assess",
        "autodelphirf_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_results": str(results),
        "assessment_output": str(output),
        "model_fitting_performed": False,
        "inputs": inputs,
        "rows": {"long": len(long), "wide": len(wide)},
        "methods": list(config.prediction_layers),
        "value_transform": config.value_transform,
        "uncertainty_layer": config.uncertainty_layer,
    }
    (output / "assessment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
