"""Stage 4: rolling model training through explicit estimator backends."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from time import perf_counter

import pandas as pd

from .prepare import PreparationError, rscript_executable
from .resources import resource_path
from .revroute import learn_task_pools

DELPHIRF_METHODS = {
    "delphirf": "RevRoute DelphiRF",
    "naive_delphirf": "Naive DelphiRF",
    "similarity_weighted_delphirf": "Similarity-weighted DelphiRF",
    "global_delphirf": "Global DelphiRF",
}
MODEL_METHODS = tuple(DELPHIRF_METHODS) + ("red",)
DEFAULT_MODEL_METHODS = ("delphirf",)
#: Saved implementation label for RevRoute pooling followed by real DelphiRF.
#: Increment this only when a new named method version is deliberately frozen.
RR_DELPHIRF_CODE_VERSION = "rrdelphirf0"


def _preparation_metadata(prepared_dir: Path) -> dict:
    path = Path(prepared_dir) / "preparation.json"
    if not path.is_file():
        raise PreparationError(
            "real DelphiRF training needs preparation.json beside the prepared triangle; "
            "re-run `autodelphirf prepare` so the raw archive and feature settings are recorded")
    return json.loads(path.read_text())


def train_delphirf(config, prepared: pd.DataFrame, schedule: pd.DataFrame,
                    methods: tuple[str, ...], output: Path) -> tuple[Path, float, dict]:
    """Fit requested methods with the installed R DelphiRF package."""
    unknown = set(methods).difference(DELPHIRF_METHODS)
    if unknown:
        raise ValueError(f"not DelphiRF methods: {sorted(unknown)}")
    metadata = _preparation_metadata(config.prepared_dir)
    resolved = metadata["resolved_preprocessing"]
    raw = Path(metadata["raw_archive"]).resolve()
    if not raw.is_file():
        raise PreparationError(f"recorded raw archive no longer exists: {raw}")
    pools = None
    if "delphirf" in methods:
        pools = learn_task_pools(prepared, schedule, Path(output).with_name("revroute_pools.csv"))
    names = ",".join(DELPHIRF_METHODS[name] for name in methods)
    weekdays = json.dumps(resolved.get("onehot_weekdays") or {})
    command = [rscript_executable(), str(resource_path("train_delphirf.R")),
               str(config.prepared_dir), str(raw),
               str(Path(config.prepared_dir) / config.schedule_file), str(output),
               "--ref-col", resolved["reference_col"],
               "--report-col", resolved["report_col"],
               "--geo-col", resolved.get("geo_col", "geo_value"),
               "--value-col", resolved["value_cols"][0],
               "--value-type", resolved["value_type"],
               "--lag-terms", ",".join(map(str, resolved["lag_terms"])),
               "--weekdays-json", weekdays, "--methods", names,
               "--taus", ".01,.025,.1,.25,.5,.75,.9,.975,.99"]
    if len(resolved["value_cols"]) > 1:
        command += ["--denom-col", resolved["value_cols"][1]]
    if pools is not None:
        command += ["--revroute-pools", str(pools)]
    began = perf_counter()
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise PreparationError(f"DelphiRF model training failed (Rscript exit {result.returncode})")
    seconds = perf_counter() - began
    provenance_path = Path(str(output) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    provenance["rr_delphirf_code_version"] = RR_DELPHIRF_CODE_VERSION
    return Path(output), seconds, provenance


__all__ = ["DEFAULT_MODEL_METHODS", "DELPHIRF_METHODS", "MODEL_METHODS",
           "RR_DELPHIRF_CODE_VERSION", "train_delphirf"]
