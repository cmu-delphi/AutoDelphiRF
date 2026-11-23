"""Persistent state shared by retrospective runs and later archive updates.

This module deliberately contains storage and consistency checks only. Model
fitting and prediction remain in DelphiRF; preparation remains in the existing
pre-diagnosis bridge.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ..comparators.red import ROUTED_TAUS
from ..core.defaults import to_reporting_scale
from ..core.io import read_csv
from ..post_assessment.scoring import wis


CONTRACT_FIELDS = (
    "reference_col", "report_col", "geo_col", "constant_geo", "value_cols",
    "value_type", "smoothed", "temporal_resol", "ref_lag", "lag_terms",
    "weekday_groups", "training_days", "testing_days",
    "target_column", "triangle_format", "min_location_rows",
)

FORECAST_KEY = (
    "dataset", "method", "cutoff", "geo_value", "reference_date",
    "report_date", "lag",
)

CASE_KEY = ("cutoff", "geo_value", "reference_date", "report_date", "lag")

# Ledger fields derived from the lag-L target. They are recomputed from the
# current archive whenever it is read, unlike the issued forecast itself.
TRUTH_FIELDS = ("truth", "truth_working", "absolute_error", "absolute_error_working",
                "true_remaining_revision", "wis_reporting", "wis_working")


def _read_preparation(prepared_dir: Path) -> dict:
    path = Path(prepared_dir) / "preparation.json"
    if not path.is_file():
        raise FileNotFoundError(f"prepared dataset is missing {path}")
    return json.loads(path.read_text())


def read_autodelphirf_preparation(prepared_dir: Path) -> dict:
    """The provenance ``autodelphirf prepare`` writes, or a clear refusal.

    Every supported dataset is prepared by AutoDelphiRF from a raw archive. A
    triangle built elsewhere has no record of its raw archive, settings, or
    target-lag evidence, and is not supported.
    """
    path = Path(prepared_dir) / "preparation.json"
    message = (f"{prepared_dir} was not prepared by AutoDelphiRF: it has no valid "
               "preparation.json. Prepared triangles built elsewhere are not supported. "
               "Prepare the dataset from its raw revision archive with "
               "`autodelphirf prepare --archive ...` (or `autodelphirf run --archive ...`), "
               "then rerun it with `autodelphirf run --config <the dataset JSON it writes>`.")
    if not path.is_file():
        raise ValueError(message)
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(message) from error
    if not isinstance(metadata.get("resolved_preprocessing"), dict) or "raw_archive" not in metadata:
        raise ValueError(message)
    return metadata


def ensure_analysis_contract(prepared_dir: Path, output_dir: Path) -> dict:
    """Freeze first-run preparation choices and reject later drift."""
    metadata = _read_preparation(prepared_dir)
    resolved = metadata.get("resolved_preprocessing") or {}
    settings = {field: resolved.get(field) for field in CONTRACT_FIELDS}
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    contract = {
        "version": 1,
        "settings": settings,
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    path = Path(output_dir) / "analysis_contract.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous.get("settings") != settings:
            # Compare over both field sets, so a setting that exists in only one
            # contract (e.g. one frozen by an older release) is named too.
            frozen = previous.get("settings", {})
            changed = sorted(field for field in set(frozen) | set(settings)
                             if frozen.get(field) != settings.get(field))
            raise ValueError(
                "updated preparation does not match the first-run analysis contract; "
                f"changed setting(s): {changed}")
        return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, default=str) + "\n")
    return contract


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".csv.gz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_archive_state(prepared_dir: Path, output_dir: Path, *, commit: bool = True) -> dict:
    """Identify the complete raw archive used by a run.

    The forecast ledger, not a second copy of the raw data, protects issued
    forecasts. A compact file signature is sufficient to skip an unchanged
    archive and to document exactly which input produced each update.
    """
    metadata = _read_preparation(prepared_dir)
    archive = Path(metadata["raw_archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"recorded raw archive no longer exists: {archive}")
    current = {"archive": str(archive.resolve()), "bytes": archive.stat().st_size,
               "sha256": _sha256(archive)}
    snapshot = Path(output_dir) / "inputs" / "raw_archive_state.json"
    previous = json.loads(snapshot.read_text()) if snapshot.is_file() else None
    current["changed"] = previous is None or previous.get("sha256") != current["sha256"]
    if commit:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps({key: value for key, value in current.items()
                                        if key != "changed"}, indent=2) + "\n")
    return {**current, "snapshot": str(snapshot)}


def recorded_archive_state(prepared_dir: Path, output_dir: Path, *, commit: bool = True) -> dict:
    """The raw-archive identity recorded at preparation, without rereading it.

    A rerun of an already-prepared dataset needs only the prepared artifacts.
    The snapshot written here has the format :func:`raw_archive_state` writes,
    so a later ``update`` still detects whether the raw archive has changed.
    """
    metadata = _read_preparation(prepared_dir)
    recorded = metadata.get("raw_archive_state") or {}
    state = {"archive": metadata.get("raw_archive"), "bytes": recorded.get("bytes"),
             "sha256": recorded.get("sha256"), "source": "recorded at preparation"}
    snapshot = Path(output_dir) / "inputs" / "raw_archive_state.json"
    if commit and state["sha256"]:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps({key: state[key] for key in ("archive", "bytes", "sha256")},
                                       indent=2) + "\n")
    return {**state, "snapshot": str(snapshot)}


def append_forecast_ledger(predictions: pd.DataFrame, output_dir: Path) -> dict:
    """Append new forecast identities and fill targets without changing forecasts."""
    path = Path(output_dir) / "forecast_ledger.csv.gz"
    current = predictions.copy()
    missing = [column for column in FORECAST_KEY if column not in current]
    if missing:
        raise ValueError(f"predictions are missing forecast-ledger key column(s): {missing}")
    if current.duplicated(list(FORECAST_KEY)).any():
        raise ValueError("current predictions contain duplicate forecast identities")
    if not path.is_file():
        _atomic_csv(current, path)
        return {"path": str(path), "previous_rows": 0, "new_rows": len(current),
                "total_rows": len(current)}

    previous = read_csv(path)
    for column in ("cutoff", "reference_date", "report_date", "target_date"):
        previous[column] = pd.to_datetime(previous[column]).dt.strftime("%Y-%m-%d")
        current[column] = pd.to_datetime(current[column]).dt.strftime("%Y-%m-%d")
    keys = list(FORECAST_KEY)
    overlap = previous.merge(current, on=keys, how="inner", suffixes=("_old", "_new"),
                             validate="one_to_one")
    prediction_columns = ["prediction", "prediction_working", "provisional_value",
                          "predicted_remaining_revision"]
    prediction_columns += [column for column in current if column.startswith("q")]
    for column in prediction_columns:
        old_name, new_name = f"{column}_old", f"{column}_new"
        if old_name not in overlap or new_name not in overlap:
            continue
        old = pd.to_numeric(overlap[old_name], errors="coerce")
        new = pd.to_numeric(overlap[new_name], errors="coerce")
        conflict = ~(np.isclose(old, new, equal_nan=True, rtol=1e-10, atol=1e-10))
        if conflict.any():
            raise ValueError(
                f"{int(conflict.sum()):,} previously issued forecast(s) changed in {column}")

    indexed_old = previous.set_index(keys)
    indexed_new = current.set_index(keys)
    shared = indexed_old.index.intersection(indexed_new.index)
    # Forecast columns above have been proven immutable. Every other field of a
    # re-issued row, including its target and scores, comes from the current
    # run: a stored truth is never kept just because it is not empty, since it
    # may predate the target's maturity date.
    for column in indexed_new.columns:
        if column in prediction_columns or column not in indexed_old:
            continue
        indexed_old.loc[shared, column] = indexed_new.loc[shared, column]
    added = indexed_new.loc[indexed_new.index.difference(indexed_old.index)]
    combined = pd.concat([indexed_old, added]).reset_index()
    _atomic_csv(combined, path)
    return {"path": str(path), "previous_rows": len(previous), "new_rows": len(added),
            "total_rows": len(combined)}


def read_forecast_ledger(output_dir: Path) -> pd.DataFrame:
    """Read the durable forecast history with normalized calendar columns."""
    path = Path(output_dir) / "forecast_ledger.csv.gz"
    if not path.is_file():
        return pd.DataFrame()
    frame = read_csv(path)
    for column in ("cutoff", "reference_date", "report_date", "target_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def read_deployment_origin(output_dir: Path) -> pd.Timestamp | None:
    """Return the latest successfully persisted training origin."""
    path = Path(output_dir) / "deployment_state.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text()).get("latest_trained_origin")
    return pd.Timestamp(value).normalize() if value else None


def write_deployment_origin(output_dir: Path, origin: pd.Timestamp) -> Path:
    """Atomically record a successfully completed training origin."""
    path = Path(output_dir) / "deployment_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "latest_trained_origin": pd.Timestamp(origin).date().isoformat()}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".json", mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)
    return path


def select_update_schedule(schedule: pd.DataFrame,
                           trained_origin: pd.Timestamp | None) -> tuple[pd.DataFrame, bool]:
    """Select the single model origin applicable to the newest archive release.

    A later scheduled origin triggers one retraining. Otherwise the latest
    successfully trained origin is reused. Historical origins are never returned.
    ``fold_id`` preserves the origin's stable position in the full calendar
    even though the execution schedule contains exactly one row.
    """
    if schedule.empty:
        raise ValueError("cannot update from an empty retraining schedule")
    ordered = schedule.sort_values("test_date").reset_index(drop=True).copy()
    ordered["fold_id"] = np.arange(1, len(ordered) + 1)
    if trained_origin is None:
        # The same update command initializes a new deployment.  The prepared
        # archive contains only information visible at this calendar date, so
        # fitting the latest scheduled origin is the real-time-safe first step.
        return ordered.iloc[[-1]].reset_index(drop=True), True
    last_origin = pd.Timestamp(trained_origin).normalize()
    available = ordered[pd.to_datetime(ordered.test_date) <= pd.to_datetime(
        ordered.test_date).max()]
    latest_origin = pd.Timestamp(available.test_date.iloc[-1]).normalize()
    retrain = latest_origin > last_origin
    origin = latest_origin if retrain else last_origin
    selected = ordered[pd.to_datetime(ordered.test_date).dt.normalize().eq(origin)].copy()
    if selected.empty:
        raise ValueError(
            f"the previous training origin {last_origin.date()} is absent from the rebuilt schedule")
    return selected.reset_index(drop=True), retrain


def exclude_issued_cases(cases: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Keep unresolved cases for which no forecast has previously been issued."""
    unresolved = cases[cases["truth"].isna()].copy()
    if unresolved.empty or ledger.empty:
        return unresolved
    keys = list(CASE_KEY)
    issued = ledger[keys].drop_duplicates().copy()
    for column in ("cutoff", "reference_date", "report_date"):
        unresolved[column] = pd.to_datetime(unresolved[column])
        issued[column] = pd.to_datetime(issued[column])
    marked = unresolved.merge(issued.assign(_issued=True), on=keys, how="left")
    return marked[marked["_issued"].isna()].drop(columns="_issued")


def refresh_forecast_ledger_truth(prepared: pd.DataFrame, target_column: str,
                                  working_scale: str, output_dir: Path, *,
                                  as_of_date) -> dict:
    """Recompute every ledger target from the current prepared triangle.

    A target exists once its prepared ``target_date`` (t + L) is on or before
    the as-of date recorded at preparation. The ledger takes that target_date
    from the triangle rather than reconstructing it. Matured targets are
    (re)written, so a provisional value stored by an earlier version is
    corrected; a row whose target date is still in the future carries no truth
    and no score. Issued forecasts never change.
    """
    path = Path(output_dir) / "forecast_ledger.csv.gz"
    ledger = read_forecast_ledger(output_dir)
    if ledger.empty:
        return {"path": str(path), "resolved_rows": 0, "corrected_rows": 0,
                "withheld_rows": 0, "total_rows": 0}
    as_of_date = pd.Timestamp(as_of_date)
    dates = prepared[["geo_value", "reference_date", "target_date"]].drop_duplicates(
        ["geo_value", "reference_date"]).rename(columns={"target_date": "prepared_target_date"})
    targets = prepared[["geo_value", "reference_date", "target_date", target_column]].copy()
    targets[target_column] = pd.to_numeric(targets[target_column], errors="coerce")
    targets = targets.dropna(subset=[target_column]).drop_duplicates(
        ["geo_value", "reference_date"], keep="last")
    targets = targets[pd.to_datetime(targets["target_date"]).le(as_of_date).to_numpy()]
    targets = targets.drop(columns="target_date")
    for frame in (dates, targets):
        frame["reference_date"] = pd.to_datetime(frame["reference_date"])
    joined = (ledger.merge(dates, on=["geo_value", "reference_date"], how="left",
                           validate="many_to_one")
              .merge(targets, on=["geo_value", "reference_date"], how="left",
                     validate="many_to_one"))
    # The ledger's target date is the prepared one; rows whose reference date
    # has left the prepared window keep the date they were issued with.
    prepared_dates = pd.to_datetime(joined["prepared_target_date"])
    ledger["target_date"] = prepared_dates.where(prepared_dates.notna(),
                                                 pd.to_datetime(ledger["target_date"])).to_numpy()
    # Keep an immutable before-refresh snapshot. ``to_numpy`` may otherwise
    # share storage with the ledger and change when truth is written below.
    previous_truth = pd.to_numeric(
        ledger.get("truth"), errors="coerce").to_numpy(float).copy()
    immature = (ledger["target_date"] > as_of_date).to_numpy()
    available = joined[target_column].notna().to_numpy() & ~immature

    for column in TRUTH_FIELDS:
        if column not in ledger:
            ledger[column] = np.nan
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce")
    ledger.loc[immature, list(TRUTH_FIELDS)] = np.nan
    positions = np.flatnonzero(available)
    if len(positions):
        truth_working = joined.loc[positions, target_column].to_numpy(float)
        truth_raw = to_reporting_scale(truth_working, working_scale)
        ledger.loc[positions, "truth_working"] = truth_working
        ledger.loc[positions, "truth"] = truth_raw
        prediction = pd.to_numeric(ledger.loc[positions, "prediction"], errors="coerce").to_numpy()
        prediction_working = pd.to_numeric(
            ledger.loc[positions, "prediction_working"], errors="coerce").to_numpy()
        provisional = pd.to_numeric(
            ledger.loc[positions, "provisional_value"], errors="coerce").to_numpy()
        ledger.loc[positions, "absolute_error"] = np.abs(truth_raw - prediction)
        ledger.loc[positions, "absolute_error_working"] = np.abs(
            truth_working - prediction_working)
        ledger.loc[positions, "true_remaining_revision"] = truth_raw - provisional
        raw_q = [f"q{tau:g}" for tau in ROUTED_TAUS]
        work_q = [f"q{tau:g}_working" for tau in ROUTED_TAUS]
        complete = (ledger.loc[positions, raw_q].notna().all(axis=1).to_numpy()
                    if all(column in ledger for column in raw_q)
                    else np.zeros(len(positions), bool))
        if complete.any():
            scored = positions[complete]
            ledger.loc[scored, "wis_reporting"] = wis(
                ledger.loc[scored, raw_q].to_numpy(float),
                ledger.loc[scored, "truth"].to_numpy(float))
            if all(column in ledger for column in work_q):
                ledger.loc[scored, "wis_working"] = wis(
                    ledger.loc[scored, work_q].to_numpy(float),
                    ledger.loc[scored, "truth_working"].to_numpy(float))
    current_truth = ledger["truth"].to_numpy(float)
    was_missing = np.isnan(previous_truth)
    resolved = int((was_missing & ~np.isnan(current_truth)).sum())
    corrected = int((~was_missing & ~np.isnan(current_truth) &
                     ~np.isclose(previous_truth, current_truth, rtol=1e-10, atol=1e-10)).sum())
    withheld = int((~was_missing & np.isnan(current_truth)).sum())
    _atomic_csv(ledger, path)
    return {"path": str(path), "resolved_rows": resolved, "corrected_rows": corrected,
            "withheld_rows": withheld, "total_rows": len(ledger)}


def write_model_manifest(model_dir: Path, output_dir: Path, provenance: dict | None) -> Path:
    """Record the persistent model inventory after a successful run."""
    model_dir = Path(model_dir)
    files = []
    for path in sorted(model_dir.rglob("*.rds")):
        files.append({"path": str(path.relative_to(model_dir)), "bytes": path.stat().st_size})
    payload = {"version": 1, "model_dir": str(model_dir), "models": files,
               "backend": provenance or {}}
    destination = Path(output_dir) / "model_manifest.json"
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return destination
