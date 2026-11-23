"""Restartable post-forecast assessment from frozen prediction artifacts."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .. import __version__
from ..core.config import load_dataset_config, write_resolved_config
from ..core.io import read_csv
from ..workflow.run import run_reliability_and_calibration
from .reliability import ROUTED_TAUS
from .reassessment import write_reassessment


DATE_COLUMNS = ("cutoff", "reference_date", "report_date", "target_date")
PIT_COLUMNS = frozenset({
    "method", "fold", "cutoff", "geo_value", "reference_date", "report_date",
    "target_date", "lag", "truth", "genuine_event", "model_available",
    "has_distribution", *(f"q{tau:g}" for tau in ROUTED_TAUS),
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_predictions(path: Path, columns: set[str] | frozenset[str] | None = None) -> pd.DataFrame:
    if columns is None:
        frame = read_csv(path)
    else:
        available = read_csv(path, nrows=0).columns
        frame = read_csv(path, usecols=[column for column in available if column in columns])
    for column in DATE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def save_assessment_outputs(outputs: dict[str, pd.DataFrame], output: Path,
                            mode: str = "full") -> None:
    if mode != "pit":
        outputs["calibration"].to_csv(output / "calibration.csv.gz", index=False, compression="gzip")
        outputs["interval_calibration"].to_csv(output / "interval_calibration.csv", index=False)
    if mode == "full":
        outputs["bootstrap"].to_csv(output / "bootstrap_pairwise_comparisons.csv", index=False)
    optional = {
        "calibration_variants": ("calibration_variants.csv.gz", "gzip"),
        "calibration_comparison": ("calibration_strategy_comparison.csv", None),
        "quantile_reliability": ("quantile_reliability.csv", None),
        "pit_diagnostics": ("pit_diagnostics.csv.gz", "gzip"),
    }
    for key, (name, compression) in optional.items():
        if mode == "pit" and key not in {"quantile_reliability", "pit_diagnostics"}:
            continue
        frame = outputs.get(key)
        if frame is not None:
            frame.to_csv(output / name, index=False, compression=compression)


def assess_results(results_dir: Path, output_dir: Path | None = None, *,
                   assessment_mode: str = "reassessment",
                   event_filter: str = "genuine") -> Path:
    """Recompute assessment tables without fitting any model."""
    mode = assessment_mode
    if mode not in {"reassessment", "full", "pit", "recalibration"}:
        raise ValueError("assessment_mode must be 'reassessment', 'full', 'pit', or 'recalibration'")
    if event_filter not in {"genuine", "carried", "all"}:
        raise ValueError("event_filter must be 'genuine', 'carried', or 'all'")
    if mode == "reassessment" and event_filter != "genuine":
        raise ValueError("forecast-to-forecast reassessment requires --event-filter genuine")
    results = Path(results_dir).resolve()
    required = [results / "predictions.csv.gz", results / "resolved_config.json"]
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
    if mode == "reassessment":
        ledger = results / "forecast_ledger.csv.gz"
        source = ledger if ledger.is_file() else results / "predictions.csv.gz"
        forecasts = _read_predictions(source)
        summary = write_reassessment(forecasts, output)
        write_resolved_config(replace(config, output_dir=output), output / "resolved_config.json")
        inputs = {path.name: {"path": str(path), "sha256": sha256(path),
                              "bytes": path.stat().st_size}
                  for path in (source, results / "resolved_config.json")}
        manifest = {
            "generated_by": "autodelphirf assess",
            "autodelphirf_version": __version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "forecast_results": str(results), "assessment_output": str(output),
            "model_fitting_performed": False, "assessment_mode": mode,
            "event_filter": event_filter, "inputs": inputs,
            "rows": {"long": len(forecasts), "wide": None},
            "methods": ["delphirf"], "working_scale": config.working_scale,
            "horizon_days": 14, "bootstrap_resamples": 1000,
            "reassessment": summary,
        }
        (output / "assessment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return output
    # Assessment scores only forecasts whose target exists, exactly as the run
    # that produced these predictions did.
    long = _read_predictions(
        results / "predictions.csv.gz", columns=PIT_COLUMNS if mode == "pit" else None)
    wide_path = results / "predictions_wide.csv.gz"
    wide = (_read_predictions(wide_path)
            if mode != "pit" and wide_path.is_file() else None)
    if event_filter != "all" and "genuine_event" not in long:
        raise ValueError(
            f"--event-filter {event_filter} requires a genuine_event column in predictions")
    if event_filter != "all" and wide is not None and "genuine_event" not in wide:
        raise ValueError(
            f"--event-filter {event_filter} requires a genuine_event column in predictions_wide")
    if event_filter == "genuine" and "genuine_event" in long:
        long = long[long.genuine_event.fillna(False).astype(bool)].copy()
    if event_filter == "genuine" and wide is not None and "genuine_event" in wide:
        wide = wide[wide.genuine_event.fillna(False).astype(bool)].copy()
    if event_filter == "carried" and "genuine_event" in long:
        long = long[~long.genuine_event.fillna(False).astype(bool)].copy()
    if event_filter == "carried" and wide is not None and "genuine_event" in wide:
        wide = wide[~wide.genuine_event.fillna(False).astype(bool)].copy()
    if "truth" in long:
        long = long[long.truth.notna()].copy()
    if wide is not None and "truth" in wide:
        wide = wide[wide.truth.notna()].copy()
    pools_path = results / "revroute_pools.csv"
    pools = (read_csv(pools_path, parse_dates=["cutoff"])
             if mode != "pit" and pools_path.is_file() else None)
    outputs = run_reliability_and_calibration(
        long, config, wide=wide, pool_assignments=pools, mode=mode)
    save_assessment_outputs(outputs, output, mode=mode)
    assessment_config = replace(config, output_dir=output)
    write_resolved_config(assessment_config, output / "resolved_config.json")
    inputs = {}
    input_names = ["predictions.csv.gz", "resolved_config.json"]
    if mode != "pit":
        input_names.insert(1, "predictions_wide.csv.gz")
    for name in input_names:
        path = results / name
        if not path.is_file():
            continue
        inputs[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    if mode != "pit" and pools_path.is_file():
        inputs["revroute_pools.csv"] = {
            "path": str(pools_path), "sha256": sha256(pools_path),
            "bytes": pools_path.stat().st_size}
    manifest = {
        "generated_by": "autodelphirf assess",
        "autodelphirf_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_results": str(results),
        "assessment_output": str(output),
        "model_fitting_performed": False,
        "assessment_mode": mode,
        "event_filter": event_filter,
        "inputs": inputs,
        "rows": {"long": len(long), "wide": len(wide) if wide is not None else None},
        "methods": list(config.prediction_layers),
        "working_scale": config.working_scale,
        "uncertainty_layer": config.uncertainty_layer,
    }
    (output / "assessment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
