"""Shared bridge to the installed DelphiRF estimator backend."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from time import perf_counter
import warnings

import pandas as pd

from ..pre_diagnosis.preparation import PreparationError, rscript_executable
from ..core.resources import resource_path


def _preparation_metadata(prepared_dir: Path) -> dict:
    path = Path(prepared_dir) / "preparation.json"
    if not path.is_file():
        raise PreparationError(
            "real DelphiRF training needs preparation.json beside the prepared triangle; "
            "re-run `autodelphirf prepare` so the raw archive and feature settings are recorded")
    return json.loads(path.read_text())


def train_delphirf(config, prepared: pd.DataFrame, schedule: pd.DataFrame,
                    method_labels: dict[str, str], output: Path,
                    pools: Path | None = None, *, model_dir: Path | None = None,
                    model_mode: str = "train", issued_forecasts: Path | None = None,
                    unresolved_only: bool = False) -> tuple[Path, float, dict]:
    """Fit already-resolved method strategies with the installed DelphiRF package."""
    if not method_labels:
        raise ValueError("at least one DelphiRF method is required")
    metadata = _preparation_metadata(config.prepared_dir)
    resolved = metadata["resolved_preprocessing"]
    raw = Path(metadata["raw_archive"]).resolve()
    # Only the similarity-weighted comparator reads raw release profiles; every
    # other method trains from the prepared triangle alone.
    if "Similarity-weighted DelphiRF" in method_labels.values() and not raw.is_file():
        raise PreparationError(
            f"Similarity-weighted DelphiRF reads the raw archive, which no longer exists: {raw}")
    names = ",".join(method_labels.values())
    weekday_groups = json.dumps(
        resolved.get("weekday_groups", resolved.get("onehot_weekdays", {})) or {})
    # Older preparation.json files predate the explicit geo_col field and used
    # geo_value by convention. Only an explicit null means --no-geo.
    source_geo_col = resolved.get("geo_col", "geo_value")
    training_geo_col = source_geo_col or "geo_value"
    constant_geo = resolved.get("constant_geo", "single_location") if source_geo_col is None else ""
    genuine_training = bool(getattr(config, "genuine_training", True))
    genuine_testing = bool(getattr(config, "genuine_testing", True))
    if model_mode not in {"train", "auto"}:
        raise ValueError("model_mode must be 'train' or 'auto'")
    default_model_dir = Path(getattr(config, "output_dir", Path(output).parent)) / "models"
    model_dir = Path(model_dir or default_model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    command = [rscript_executable(), str(resource_path("train_delphirf.R")),
               str(config.prepared_dir), str(raw),
               str(Path(config.prepared_dir) / config.schedule_file), str(output),
               "--ref-col", resolved["reference_col"],
               "--report-col", resolved["report_col"],
               "--geo-col", training_geo_col, "--constant-geo", constant_geo,
               "--value-col", resolved["value_cols"][0],
               "--value-type", resolved["value_type"],
               "--lag-terms", ",".join(map(str, resolved["lag_terms"])),
               "--weekdays-json", weekday_groups, "--methods", names,
               "--genuine-training", str(genuine_training).lower(),
               "--genuine-testing", str(genuine_testing).lower(),
               "--model-save-dir", str(model_dir), "--model-mode", model_mode,
               "--taus", ".01,.025,.1,.25,.5,.75,.9,.975,.99"]
    if len(resolved["value_cols"]) > 1:
        command += ["--denom-col", resolved["value_cols"][1]]
    if pools is not None:
        command += ["--revroute-pools", str(pools)]
    if issued_forecasts is not None:
        command += ["--issued-forecasts", str(issued_forecasts)]
    if unresolved_only:
        command += ["--unresolved-only", "true"]
    # An update executes exactly one origin of the full calendar. R reads the
    # full schedule file, so name that origin; otherwise every historical origin
    # would be refitted or reloaded and predicted again.
    if "fold_id" in schedule:
        folds = pd.Series(schedule["fold_id"]).dropna().astype(int).unique()
        if len(folds) != 1:
            raise ValueError("a selected update schedule must contain exactly one origin")
        command += ["--fold", str(int(folds[0]))]
    began = perf_counter()
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise PreparationError(f"DelphiRF model training failed (Rscript exit {result.returncode})")
    seconds = perf_counter() - began
    provenance_path = Path(str(output) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    provenance["model_save_dir"] = str(model_dir)
    provenance["model_mode"] = model_mode
    failure_count = int(provenance.get("fit_failure_count", 0))
    if failure_count:
        failure_file = provenance.get("fit_failures_file", str(output) + ".fit_failures.csv")
        warnings.warn(
            f"DelphiRF completed with {failure_count} failed fit(s); "
            f"predictions are incomplete. Details: {failure_file}",
            RuntimeWarning,
            stacklevel=2,
        )
    dropped_count = int(provenance.get("dropped_test_row_count", 0))
    if dropped_count:
        dropped_file = provenance.get("dropped_test_rows_file",
                                      str(output) + ".dropped_test_rows.csv")
        warnings.warn(
            f"DelphiRF left {dropped_count} test row(s) unforecast because a predictor is "
            f"missing; they are reported as unavailable. Details: {dropped_file}",
            RuntimeWarning,
            stacklevel=2,
        )
    return Path(output), seconds, provenance


__all__ = ["train_delphirf"]
