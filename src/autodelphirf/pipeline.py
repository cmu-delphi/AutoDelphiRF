from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

from .red import ROUTED_TAUS
from dataclasses import replace
from .io import load_prepared_triangles
from .metrics import wis
from .defaults import to_raw_scale
from .engine import replay
from .weighting import attach_revision_progress
from .context import attach_issued_context
from .process_monitor import build_monitor_index, process_state_monitor
from .defaults import PROCESS_DEFAULTS
from .defaults import EVALUATION_DEFAULTS
from .calibration import add_interval_columns, prospective_calibration, reliability_history
from .inference import bootstrap, primary_comparison_pairs
from .validation import run_validation

from .config import DatasetConfig
from .comparator_schema import validate_comparator_file
from .diagnosis import (diagnose_prepared_data, diagnose_target_lag,
                        raw_archive_for_diagnosis, resolve_target_lag)
from .registry import LAYER_COLUMNS, PREDICTION_LAYERS, point_layer
from .model_training import DELPHIRF_METHODS, train_delphirf
from .resources import user_config_path
from .report import LEVELS, build_report

KEYS = ["fold", "cutoff", "geo_value", "reference_date", "report_date", "target_date", "lag"]

# Wide-frame column carrying each layer's "was a model actually fitted" flag.
# A layer with no entry here is always available (Null, RED, and any imported
# point comparator), so a missing flag must never be read as unavailable.
AVAILABILITY_FIELD = {}

# Methods that issue predictive quantiles. Point-only comparators never get
# invented distributions or WIS. Only these are
# eligible for interval calibration and the risk-score/alert layer.
DISTRIBUTION_METHODS = {"red", "delphirf", "naive_delphirf",
                        "similarity_weighted_delphirf", "global_delphirf"}

# Fixed registry-name -> comparison-role mapping for clustered bootstrap
# comparisons. A dataset that
# configures fewer of these prediction_layers simply gets fewer pairs;
# nothing here requires every method to be present.
BOOTSTRAP_ROLE_BY_METHOD = {
    "baseline_null": "null", "naive_delphirf": "naive",
    "similarity_weighted_delphirf": "similarity", "red": "red",
    "delphirf": "candidate",
    "global_delphirf": "global",
}


def build_cases(prepared: pd.DataFrame, schedule: pd.DataFrame,
                genuine_only: bool = True) -> pd.DataFrame:
    """Build forecast cases from the states actually visible in each test window."""
    target = str(schedule.target_column.iloc[0])
    rows = []
    ordered = schedule.sort_values("test_date").reset_index(drop=True)
    for index, item in ordered.iterrows():
        cutoff = pd.Timestamp(item.test_date)
        if "testing_days" in ordered and pd.notna(item.get("testing_days")):
            end = cutoff + pd.Timedelta(days=int(item.testing_days))
        else:
            end = (pd.Timestamp(ordered.test_date.iloc[index + 1])
                   if index + 1 < len(ordered) else pd.Timestamp(item.experiment_end_date))
        visible = prepared[prepared.report_date.ge(cutoff) & prepared.report_date.lt(end)].copy()
        if genuine_only and "genuine_event" in visible:
            visible = visible[visible.genuine_event.fillna(False).astype(bool)]
        visible = visible[visible[target].notna() & visible.log_value_7dav.notna()].copy()
        visible["fold"], visible["cutoff"] = index + 1, cutoff
        visible["truth"], visible["Null"] = visible[target].astype(float), visible.log_value_7dav.astype(float)
        rows.append(visible[KEYS + ["truth", "Null"]])
    if not rows:
        return pd.DataFrame(columns=KEYS + ["truth", "Null"])
    return pd.concat(rows, ignore_index=True)


def attach_reporting_process_assessment(cases: pd.DataFrame,
                                        prepared: pd.DataFrame) -> pd.DataFrame:
    """Attach the target-free reporting-process assessment."""
    result = cases.copy()
    rows = []
    settings = {
        "recent_days": PROCESS_DEFAULTS["recent_days"],
        "history_days": PROCESS_DEFAULTS["history_days"],
        "min_recent_dates": PROCESS_DEFAULTS["minimum_recent_report_dates"],
        "min_history_dates": PROCESS_DEFAULTS["minimum_history_report_dates"],
        "watch_threshold": PROCESS_DEFAULTS["watch_threshold"],
        "rediagnose_threshold": PROCESS_DEFAULTS["rediagnose_threshold"],
        "epsilon_scale": PROCESS_DEFAULTS["epsilon_scale"],
    }
    for cutoff, group in result.groupby("cutoff", sort=False, observed=True):
        index = build_monitor_index(prepared, pd.Timestamp(cutoff))
        for geo, lag in group[["geo_value", "lag"]].drop_duplicates().itertuples(index=False):
            state, score, details = process_state_monitor(
                prepared, pd.Timestamp(cutoff), str(geo), int(lag),
                monitor_index=index, return_details=True, **settings)
            rows.append({"cutoff": cutoff, "geo_value": geo, "lag": lag,
                         "process_state": state, "process_shift_score": score,
                         "process_shift_driver": (max(details, key=details.get)
                                                  if details else None),
                         "process_shift_components": json.dumps(details, sort_keys=True)})
    context = pd.DataFrame(rows)
    return (result.merge(context, on=["cutoff", "geo_value", "lag"], how="left",
                         validate="many_to_one") if not context.empty else result)


def apply_input_schema(prepared: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Rename configured source columns to the frozen internal prepared-triangle schema."""
    required = {"geo_value", "reference_date", "report_date", "target_date", "lag", "target_lag", "as_of_value"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"input_columns mapping missing semantic fields: {sorted(missing)}")
    canonical = {"as_of_value": "log_value_7dav"}
    rename = {}
    for semantic, source in columns.items():
        destination = canonical.get(semantic, semantic)
        if source != destination:
            if source not in prepared:
                raise ValueError(f"configured source column {source!r} for {semantic!r} is absent")
            if destination in prepared:
                raise ValueError(f"cannot rename {source!r} to existing column {destination!r}")
            rename[source] = destination
    return normalize_lag_dtypes(prepared.rename(columns=rename))


#: Columns that must be integer-typed for the joins downstream to work.
#: ``merge_asof`` refuses to match an int64 key against a float64 one, and a
#: lag is conceptually a whole number of days either way.
INTEGER_COLUMNS = ("lag", "target_lag")


def normalize_lag_dtypes(prepared: pd.DataFrame) -> pd.DataFrame:
    """Coerce the lag columns to a nullable integer dtype.

    DelphiRF's ``data_preprocessing`` writes ``lag``/``target_lag`` as doubles
    (date arithmetic in R yields a numeric difftime), and a hand-built CSV
    triangle picks up float64 the moment one value is missing. Either one
    reaches ``pandas.merge_asof`` in :func:`diagnosis.diagnose_target_lag`,
    which compares join-key dtypes exactly and raises rather than coercing.
    Normalizing once here means no downstream module has to care which
    producer wrote the triangle.

    A non-integral value is an error, not something to round away: a lag of
    3.5 days means the triangle was built against a different notion of a
    reporting day, and silently truncating it would misalign every target date.

    A column with no missing values becomes plain ``int64`` rather than the
    nullable ``Int64``, because ``merge_asof`` compares join-key dtypes
    exactly and an extension dtype on one side is as much a mismatch as a
    float. Only ``target_lag``, which is legitimately missing for an episode
    without an available target value needs the nullable dtype.
    """
    for column in INTEGER_COLUMNS:
        if column not in prepared or pd.api.types.is_integer_dtype(prepared[column]):
            continue
        values = pd.to_numeric(prepared[column], errors="coerce")
        observed = values.dropna()
        fractional = observed[observed != observed.round()]
        if not fractional.empty:
            raise ValueError(
                f"prepared-triangle column {column!r} holds non-integral values "
                f"(e.g. {fractional.iloc[0]}); a lag must be a whole number of days.")
        dtype = "Int64" if values.isna().any() else "int64"
        prepared = prepared.assign(**{column: values.astype(dtype)})
    return prepared


def _raw(values, transform: str):
    """Working scale -> original reporting scale (see defaults.to_raw_scale)."""
    return to_raw_scale(values, transform)


# Optional USER configuration, not package data: both files record choices
# about particular datasets rather than properties of the method, so they are
# looked up in the user's environment/working directory and may be absent.
# See ``resources.user_config_path`` for the search order, and the fallbacks
# below for what a run does without them (diagnose L from the prepared
# triangle; honour the schedule's own ``target_lag``).
# Resolved per call rather than at import: the search consults the process
# environment and the working directory, both of which a caller may set after
# importing this module (the CLI does exactly that).
def target_lag_params_path() -> Path | None:
    return user_config_path("target_lag_params.json")


def raw_archive_manifest_path() -> Path | None:
    return user_config_path("raw_archive_manifest.json")


def load_target_lag_params(dataset: str, path: Path | None = None) -> tuple[int | None, bool, dict]:
    """The user's declared target lag for one dataset, if the params file has one.

    Returns ``(target_lag, confirmed, rule_overrides)``. A missing file or a
    missing entry means "no declared choice", which routes the run to the
    recommended target lag.
    """
    path = path or target_lag_params_path()
    if path is None or not Path(path).exists():
        return None, False, {}
    params = json.loads(Path(path).read_text())
    defaults = params.get("defaults", {}) or {}
    entry = (params.get("datasets", {}) or {}).get(dataset)
    rule = dict(defaults.get("rule", {}) or {})
    if entry is None:
        return None, False, rule
    rule.update(entry.get("rule", {}) or {})
    return entry.get("target_lag"), bool(entry.get("confirmed", False)), rule


def derive_value_transform(prepared: pd.DataFrame, config: DatasetConfig) -> str:
    """Determine the working scale from the triangle itself, not from a declaration.

    DelphiRF always smooths by adding one in the counting domain and then logs,
    so the working column's form follows the VALUE-COLUMN ARITY:

      one column (a count, or a rate supplied directly)
          -> log(value + 1)                      -> invert with expm1  ("log1p")
      a numerator/denominator pair
          -> log(num + 1) - log(denom + 1)       -> invert with exp    ("log")

    An arity-2 triangle carries the numerator and denominator feature families
    alongside the combined column, so their presence is the decisive evidence.
    This is derived rather than configured because it is a property of the data;
    a dataset with both counts is ALWAYS on the log-ratio scale.
    """
    base = config.input_columns.get("as_of_value", "log_value_7dav")
    if f"{base}_num" in prepared.columns and f"{base}_denom" in prepared.columns:
        return "log"
    return "log1p"


def _check_transform(wide: pd.DataFrame, config: DatasetConfig) -> None:
    """Fail loudly when the declared value_transform cannot be the one that
    built this triangle, instead of silently emitting shifted raw values.

    log1p(x) is >= 0 for every x >= 0, so a negative working value proves the
    column is a log-ratio (log1p(num) - log1p(denom)) that must invert with
    exp. This is the mismatch that would otherwise report every quidel level
    as value - 1; difference-based metrics (AE/WIS/coverage) survive the shift,
    so nothing else would have flagged it.
    """
    if config.value_transform != "log1p" or "truth" not in wide:
        return
    truth = pd.to_numeric(wide["truth"], errors="coerce")
    smallest = truth.min()
    if pd.notna(smallest) and smallest < 0:
        raise ValueError(
            f"dataset '{config.name}' declares value_transform='log1p' but its working "
            f"column reaches {smallest:.4g} (< 0), so it cannot be log1p of a non-negative "
            "value. A num/denom triangle stores log1p(num) - log1p(denom); declare "
            "value_transform='log' for it.")


AUDIT_PASSTHROUGH = ("process_state", "process_shift_score", "process_shift_driver",
                     "process_shift_components", "weight_fallback",
                     "effective_donor_count", "cold_start", "point_ess",
                     "max_normalized_donor_weight", "mean_curve_distance",
                     "mean_stage_distance", "revision_state_active")


def _method_rows(wide: pd.DataFrame, method: str, point: np.ndarray, quantiles: np.ndarray,
                 config: DatasetConfig) -> pd.DataFrame:
    """One method's standardized long rows, built as columns rather than dicts.

    ``point`` is the working-scale point prediction per case and ``quantiles``
    is ``(n_cases, len(ROUTED_TAUS))``, NaN where the method issues no
    distribution. Every requested method-case combination is retained. A
    method that could not fit or predict has ``model_available=False`` and
    missing prediction, quantile, and score fields.
    """
    frame = wide
    transform = config.value_transform
    truth_working = frame["truth"].to_numpy(float)
    truth_raw = _raw(truth_working, transform)
    point_raw = _raw(point, transform)
    # A row keeps its untransformed quantiles unless every one of them is
    # finite, matching the per-case rule: a partial distribution is not
    # back-transformed into a reportable one.
    complete = np.isfinite(quantiles).all(axis=1)
    quantiles_raw = quantiles.copy()
    if complete.any():
        quantiles_raw[complete] = _raw(quantiles[complete], transform)
    has_distribution = np.isfinite(quantiles_raw).all(axis=1)
    null_working = (frame["Null"].to_numpy(float) if "Null" in frame else truth_working)
    provisional = _raw(null_working, transform)
    out = {key: (frame[key].to_numpy() if key in frame else None) for key in KEYS}
    out.update(dataset=config.name, value_transform=transform,
               prediction_scale="raw", method=method, truth=truth_raw, prediction=point_raw,
               absolute_error=np.abs(truth_raw - point_raw), has_distribution=has_distribution)
    # Working-scale companions. Evaluation is reported on BOTH scales: the
    # original reporting scale (primary) and the model's own working scale.
    # They rank methods differently whenever errors scale with level -- raw AE
    # is dominated by high-volume locations, working-scale AE weights locations
    # roughly equally -- so neither alone is the whole picture. Taken from the
    # untransformed inputs directly, so these are exact rather than a
    # back-transformation.
    out["truth_working"] = truth_working
    out["prediction_working"] = point
    out["absolute_error_working"] = np.abs(truth_working - point)
    out["provisional_value"] = provisional
    out["true_remaining_revision"] = truth_raw - provisional
    out["predicted_remaining_revision"] = point_raw - provisional
    availability = AVAILABILITY_FIELD.get(method)
    fitted = np.isfinite(point)
    out["model_available"] = (frame[availability].fillna(False).to_numpy(bool) & fitted
                              if availability and availability in frame else fitted)
    for index, tau in enumerate(ROUTED_TAUS):
        out[f"q{tau:g}"] = quantiles_raw[:, index]
        out[f"q{tau:g}_working"] = quantiles[:, index]
    scored = np.full(len(frame), np.nan)
    scored_working = np.full(len(frame), np.nan)
    if has_distribution.any():
        scored[has_distribution] = wis(quantiles_raw[has_distribution], truth_raw[has_distribution])
        scored_working[has_distribution] = wis(quantiles[has_distribution],
                                               truth_working[has_distribution])
    # Evaluate every strategy on the original reporting scale. The scoring
    # function remains DelphiRF/evalcast's twice-mean-pinball definition.
    out["wis"], out["wis_raw"], out["wis_working"] = scored, scored, scored_working
    for field in AUDIT_PASSTHROUGH:
        if field in frame:
            out[field] = frame[field].to_numpy()
    return pd.DataFrame(out)


def standardize_predictions(wide: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Convert method-specific columns into one stable, long prediction schema."""
    if config.value_transform is None:
        raise ValueError(
            f"dataset '{config.name}' has no resolved value_transform. It is normally derived from "
            "the prepared triangle by run_pipeline() (see derive_value_transform); when calling "
            "standardize_predictions directly, supply a config that already has it set.")
    _check_transform(wide, config)
    adapters = dict(PREDICTION_LAYERS)
    adapters.update({name: point_layer(column) for name, column in config.comparator_columns.items()})
    adapters.update({name: point_layer(name) for name in config.comparator_methods})
    unknown = set(config.prediction_layers).difference(adapters)
    if unknown:
        raise ValueError(f"unregistered prediction layers: {sorted(unknown)}")
    # Point-only imported comparators read one column and have no quantiles.
    columns = dict(LAYER_COLUMNS)
    columns.update({name: (column, None) for name, column in config.comparator_columns.items()})
    quantile_template = ("{name}_tau{{tau:g}}" if config.comparator_quantile_template else None)
    columns.update({name: (name, quantile_template.format(name=name) if quantile_template else None)
                    for name in config.comparator_methods})
    frames = []
    for method in config.prediction_layers:
        template = columns.get(method)
        if template is not None and template[0] in wide:
            point_column, tau_template = template
            point = wide[point_column].to_numpy(float)
            if tau_template is None:
                # Treat the carry-forward baseline as a degenerate distribution for
                # which reporting-scale WIS equals absolute error. Other
                # point-only imported methods remain ineligible for WIS.
                quantiles = (np.repeat(point[:, None], len(ROUTED_TAUS), axis=1)
                             if method in {"baseline_null", "null"}
                             else np.full((len(wide), len(ROUTED_TAUS)), np.nan))
            else:
                quantiles = np.column_stack([wide[tau_template.format(tau=tau)].to_numpy(float)
                                             for tau in ROUTED_TAUS])
        else:
            # A custom layer registered through ``register_prediction_layer``
            # with no column template still works, one row at a time.
            extracted = [adapters[method](row, ROUTED_TAUS) for row in wide.to_dict("records")]
            point = np.array([float(x[0]) for x in extracted])
            quantiles = np.vstack([np.asarray(x[1], float) for x in extracted])
        frames.append(_method_rows(wide, method, point, quantiles, config))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_reliability_reference(config: DatasetConfig, method: str | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a frozen, pre-evaluation q_error/q_regret reference if configured.

    See ``DatasetConfig.reliability_reference_file``: without one, the
    risk-score layer still reports continuous quantities and degrades the
    categorical alert to "reference distribution unavailable", rather than
    fabricating a reference from data
    that overlaps this run's own outer evaluation.
    """
    if not config.reliability_reference_file:
        return None, None
    reference = pd.read_csv(config.reliability_reference_file)
    if "method" in reference:
        if method is None:
            raise ValueError("method-specific reliability references require a method")
        reference = reference[reference.method.eq(method)]
        if reference.empty:
            return None, None
    required = {"q_error", "q_regret"}
    if not required.issubset(reference.columns):
        raise ValueError(f"reliability reference missing columns: {sorted(required - set(reference.columns))}")
    multiplicity = (reference.case_n.to_numpy(int) if "case_n" in reference
                    else np.ones(len(reference), dtype=int))
    error_reference = np.repeat(reference.q_error.to_numpy(float), multiplicity)
    regret_reference = np.repeat(reference.q_regret.to_numpy(float), multiplicity)
    return error_reference, regret_reference


def run_reliability_and_calibration(long: pd.DataFrame, config: DatasetConfig,
                                    wide: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Interval calibration from completed revisions, risk-score/alert, bootstrap CIs.

    Wired into the standard pipeline output (not an optional side
    computation): every dataset run produces a calibration table, a
    per-method reliability table for every method that issues genuine
    predictive quantiles, and episode-clustered bootstrap CIs for whichever
    primary-comparison pairs the configured ``prediction_layers`` support.
    """
    eligible = set(DISTRIBUTION_METHODS)
    if config.comparator_quantile_template:
        eligible |= set(config.comparator_methods)
    distribution_methods = ([m for m in config.prediction_layers if m in eligible]
                            if config.uncertainty_layer else [])
    calibration = (prospective_calibration(long, distribution_methods,
                                            value_transform=config.value_transform) if distribution_methods
                  else long.iloc[:0].copy())
    if not calibration.empty:
        base_intervals = add_interval_columns(calibration, LEVELS, prefix="")
        calibrated_source = calibration.drop(columns=[f"q{tau:g}" for tau in ROUTED_TAUS]).rename(
            columns={f"cal_q{tau:g}": f"q{tau:g}" for tau in ROUTED_TAUS})
        calibrated_intervals = add_interval_columns(calibrated_source, LEVELS, prefix="")
        interval_rows = []
        for level in LEVELS:
            for status, frame in (("base", base_intervals), ("calibrated", calibrated_intervals)):
                for method, group in frame.groupby("method", observed=True):
                    interval_rows.append({
                        "method": method, "status": status, "nominal": level / 100, "n": len(group),
                        "coverage": group[f"coverage_{level}"].mean(),
                        "coverage_error": group[f"coverage_{level}"].mean() - level / 100,
                        "mean_width": group[f"width_{level}"].mean(),
                        "median_width": group[f"width_{level}"].median(),
                        "lower_miss": group[f"lower_miss_{level}"].mean(),
                        "upper_miss": group[f"upper_miss_{level}"].mean(),
                        "mean_wis": group["cal_wis" if status == "calibrated" else "wis"].mean()
                        if ("cal_wis" if status == "calibrated" else "wis") in group else np.nan,
                        "median_wis": group["cal_wis" if status == "calibrated" else "wis"].median()
                        if ("cal_wis" if status == "calibrated" else "wis") in group else np.nan})
        interval_calibration = pd.DataFrame(interval_rows)
    else:
        interval_calibration = pd.DataFrame(columns=["method", "status", "nominal", "n", "coverage",
                                                      "coverage_error", "mean_width", "median_width",
                                                      "lower_miss", "upper_miss", "mean_wis", "median_wis"])
    reliability_frames = []
    for method in distribution_methods:
        error_reference, regret_reference = _load_reliability_reference(config, method)
        reliability_frames.append(reliability_history(long, method, error_reference=error_reference,
                                                      regret_reference=regret_reference))
    reliability = (pd.concat(reliability_frames, ignore_index=True) if reliability_frames
                  else long.iloc[:0].copy())
    methods_present = {role: name for name, role in BOOTSTRAP_ROLE_BY_METHOD.items()
                       if name in config.prediction_layers}
    point_pairs, distribution_pairs = primary_comparison_pairs(methods_present)
    early_max = EVALUATION_DEFAULTS.get("early_lag_max", 14)
    bootstrap_rows = pd.concat([
        bootstrap(long, "overall", point_pairs, distribution_pairs),
        bootstrap(long[long.lag <= early_max], "early", point_pairs, distribution_pairs),
    ], ignore_index=True) if (point_pairs or distribution_pairs) else pd.DataFrame(
        columns=["subset", "comparison", "metric", "estimate", "ci_low", "ci_high",
                 "improvement_probability", "episodes"])
    # V1/V2 prospective validation. Pure post-processing on forecasts that were
    # already issued and scored: nothing is re-predicted and no threshold is
    # tuned here.
    from .validation import alignment_frame
    validation = run_validation(long, wide=wide, calibration=calibration, reliability=reliability,
                                point_pairs=point_pairs, distribution_pairs=distribution_pairs,
                                taus=ROUTED_TAUS)
    validation["_alignment"] = alignment_frame(reliability, long)
    return {"calibration": calibration, "interval_calibration": interval_calibration,
            "reliability": reliability, "bootstrap": bootstrap_rows, **validation}


def run_pipeline(config: DatasetConfig, diagnosis_only: bool = False, should_stop=None) -> dict:
    started = perf_counter()
    pipeline_timings = []
    def record_stage(name, began):
        pipeline_timings.append({"stage": name, "seconds": perf_counter() - began})
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stage = perf_counter()
    prepared = load_prepared_triangles(config.prepared_dir)
    prepared = apply_input_schema(prepared, config.input_columns)
    for column in ("reference_date", "report_date", "target_date"):
        prepared[column] = pd.to_datetime(prepared[column])
    derived_transform = derive_value_transform(prepared, config)
    if config.value_transform is not None and config.value_transform != derived_transform:
        raise ValueError(
            f"dataset '{config.name}' declares value_transform='{config.value_transform}' but its "
            f"prepared triangle is '{derived_transform}'. The working scale follows the value-column "
            "arity and is not a free choice: a numerator/denominator triangle is always "
            "log(num+1)-log(denom+1) ('log'); a single-column triangle is always log(value+1) "
            "('log1p'). Remove the declaration to derive it, or correct it.")
    config = replace(config, value_transform=derived_transform)
    record_stage("load_prepared_data", stage)
    stage = perf_counter()
    schedule = pd.read_csv(config.prepared_dir / config.schedule_file,
                           parse_dates=["test_date", "experiment_end_date"])
    summary, missingness = diagnose_prepared_data(prepared, schedule, config.revision_profile)
    (output / "diagnostics").mkdir(exist_ok=True)
    (output / "diagnostics" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    missingness.to_csv(output / "diagnostics" / "missingness.csv", index=False)
    record_stage("input_diagnosis", stage)
    if diagnosis_only:
        return summary
    # Every DelphiRF-labelled method crosses one auditable boundary into the
    # installed R package. RevRoute itself contributes only the per-origin
    # task pools consumed by the default ``delphirf`` method.
    requested_delphirf = tuple(
        method for method in config.prediction_layers if method in DELPHIRF_METHODS)
    delphirf_seconds, delphirf_provenance = 0., None
    if requested_delphirf:
        # DelphiRF necessarily returns predictions on the response's working
        # scale. This is an explicitly named intermediate; predictions.csv.gz
        # below is the public artifact and always carries raw-scale prediction
        # and q* columns after the inverse transform in _method_rows().
        trained = output / "_working_scale_delphirf_predictions.csv.gz"
        _, delphirf_seconds, delphirf_provenance = train_delphirf(
            config, prepared, schedule, requested_delphirf, trained)
        config = replace(
            config, comparator_file=trained,
            comparator_methods={**config.comparator_methods,
                                **{name: DELPHIRF_METHODS[name]
                                   for name in requested_delphirf}},
            comparator_quantile_template="q{tau:g}")
    stage = perf_counter()
    cases = build_cases(prepared, schedule, config.include_genuine_events_only)
    if cases.empty:
        raise ValueError("the configured test windows produced no forecast cases")
    cases = attach_reporting_process_assessment(
        attach_issued_context(attach_revision_progress(cases, prepared), prepared), prepared)
    record_stage("case_construction_and_features", stage)
    params = dict(config.method_params)
    if config.initial_lag is not None:
        params["initial_lag"] = config.initial_lag
    params["include_red"] = "red" in config.prediction_layers
    params["target_column"] = str(schedule.target_column.iloc[0])
    # Target lag L: recommended from the revision process, then
    # reconciled with the user's declared choice. The user always wins, but
    # never without having been shown the recommendation.
    user_lag, confirmed, rule = load_target_lag_params(config.name)
    if user_lag is None and "target_lag" in schedule:
        # A schedule that already states one acts as the declared choice, so a
        # dataset set up before the params file keeps working unchanged.
        target_lags = set(schedule.target_lag.dropna().astype(int))
        if len(target_lags) == 1:
            user_lag, confirmed = target_lags.pop(), True
    # Diagnose from the RAW archive when one is registered: a prepared
    # triangle is truncated at the target lag it was built with, so diagnosing L
    # from it is circular and fails the follow-up requirement outright on
    # every dataset whose triangle stops short of it.
    # A triangle created by ``autodelphirf prepare`` carries the diagnosis made
    # from the full raw archive. Reuse it: re-diagnosing from the prepared
    # triangle is circular because that triangle is already truncated at L,
    # and can produce a spurious smaller recommendation in the run log.
    preparation_path = config.prepared_dir / "preparation.json"
    recorded = None
    if preparation_path.is_file():
        try:
            recorded = json.loads(preparation_path.read_text()).get("diagnosis")
        except (OSError, json.JSONDecodeError):
            recorded = None
    if recorded and recorded.get("recommended_target_lag") is not None:
        diagnosed = {
            "selected_target_lag": int(recorded["recommended_target_lag"]),
            "status": "raw archive diagnosis recorded during preprocessing",
            "rule": {"completion_rate": 0.90, "relative_error": 0.10},
            "completion": recorded.get("target_lag_completion_curve", {}),
            "source": "preparation.json (full raw archive)",
        }
        source = diagnosed["source"]
    else:
        archive, source = raw_archive_for_diagnosis(config.name, raw_archive_manifest_path())
        if archive is not None:
            diagnosed = diagnose_target_lag(archive.rename(columns={"value": "value_7dav"}),
                                            value_column="value_7dav", **rule)
        else:
            diagnosed = diagnose_target_lag(
                prepared, value_column=config.input_columns.get("raw_value", "value_7dav"), **rule)
            source = f"prepared triangle ({source}; truncated, so the diagnosis is limited)"
        diagnosed["source"] = source
    resolved, reason = resolve_target_lag(diagnosed, user_lag, confirmed, config.name)
    # The triangle encodes its target lag: data_preprocessing() builds the target
    # column and target_date from the ref_lag it was given, and writes the same
    # value into the schedule. So a run cannot adopt a different L than the one
    # its triangle was built with -- the eligibility filter would cut at one
    # target lag while the targets held another. Diagnosis informs preprocessing;
    # by the time a run reads a triangle, the target lag is already encoded and
    # can only be checked.
    if "target_lag" in schedule:
        built = set(schedule.target_lag.dropna().astype(int))
        if len(built) == 1 and resolved != built.copy().pop():
            baked = built.pop()
            raise ValueError(
                f"dataset '{config.name}': this run would use L={resolved} ({reason}) but its "
                f"prepared triangle was built with L={baked}, so its target column and "
                f"target_date encode {baked}. Re-run `autodelphirf prepare` from the raw "
                f"archive with --target-lag {resolved}, or keep target_lag={baked} for this run.")
    print(f"Target lag: {resolved} days; recommendation: "
          f"{diagnosed.get('selected_target_lag')} days ({reason})", flush=True)
    params["target_lag"] = resolved
    target_lag_record = {"resolved": resolved, "reason": reason,
                         "user_target_lag": user_lag, "user_confirmed": confirmed,
                         "diagnosed": diagnosed}
    (output / "target_lag_resolution.json").write_text(
        json.dumps(target_lag_record, indent=2, default=str) + "\n")
    stage = perf_counter()
    # Training windows are schedule data, not a dataset-wide constant. Pass
    # the fold -> days mapping so retrospective studies may vary them across
    # origins without splitting the dataset into separate runs.
    training_windows = dict(zip(schedule.index + 1,
                                schedule.training_days.astype(int)))
    wide, timing = replay(cases, prepared, training_windows,
                          should_stop=should_stop, **params)
    record_stage("revroute_execution", stage)
    rr_audit = rr_clusters = None
    if config.comparator_file:
        stage = perf_counter()
        validate_comparator_file(config.comparator_file, config.comparator_method_column,
                                 config.comparator_prediction_column, config.comparator_methods,
                                 config.comparator_columns)
        comparator = pd.read_csv(config.comparator_file)
        for column in ("cutoff", "reference_date", "report_date", "target_date"):
            if column in comparator:
                comparator[column] = pd.to_datetime(comparator[column])
        merge_keys = [key for key in KEYS if key in comparator]
        template = config.comparator_quantile_template
        for output_name, source_name in config.comparator_methods.items():
            selected = comparator[comparator[config.comparator_method_column].eq(source_name)]
            wanted = {config.comparator_prediction_column: output_name}
            if template:
                # Rename the file's quantile columns into this method's own
                # namespace so several comparators can each carry a full
                # distribution without colliding.
                missing = [template.format(tau=tau) for tau in ROUTED_TAUS
                           if template.format(tau=tau) not in selected.columns]
                if missing:
                    raise ValueError(
                        f"comparator '{source_name}' declares comparator_quantile_template "
                        f"{template!r} but is missing {missing}")
                wanted.update({template.format(tau=tau): f"{output_name}_tau{tau:g}"
                               for tau in ROUTED_TAUS})
            values = selected[merge_keys + list(wanted)].rename(columns=wanted)
            # Each method keeps its own availability pattern. Taking an inner
            # intersection here would let one failed method erase valid
            # predictions from every other method for that case.
            wide = wide.merge(values, on=merge_keys, how="left", validate="one_to_one")
        needed = list(config.comparator_columns.values())
        if needed:
            values = comparator[merge_keys + needed].drop_duplicates()
            wide = wide.merge(values, on=merge_keys, how="left", validate="one_to_one")
        record_stage("external_comparator_ingestion", stage)
    stage = perf_counter()
    long = standardize_predictions(wide, config)
    record_stage("standardize_predictions", stage)
    stage = perf_counter()
    reliability_outputs = run_reliability_and_calibration(long, config, wide=wide)
    record_stage("calibration_and_reliability", stage)
    wide.to_csv(output / "predictions_wide.csv.gz", index=False, compression="gzip")
    long.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    reliability_outputs["calibration"].to_csv(output / "calibration.csv.gz", index=False, compression="gzip")
    reliability_outputs["interval_calibration"].to_csv(output / "interval_calibration.csv", index=False)
    reliability_outputs["reliability"].to_csv(output / "reliability.csv.gz", index=False, compression="gzip")
    reliability_outputs["bootstrap"].to_csv(output / "bootstrap_pairwise_comparisons.csv", index=False)
    for name, frame in reliability_outputs.items():
        if name.startswith(("v1_", "v2_", "v3_", "v4_")) and frame is not None and not frame.empty:
            frame.to_csv(output / f"{name}.csv", index=False)
    timing.to_csv(output / "runtime_by_origin.csv", index=False)
    shared_columns = ["training_selection_seconds", "route_build_seconds", "red_prediction_seconds",
                      "process_monitor_seconds", "residual_history_seconds"]
    shared_seconds = float(timing[[c for c in shared_columns if c in timing]].sum().sum())
    runtime_rows = [{"method": "baseline_null", "timing_status": "fresh", "shared_seconds": 0.,
                     "incremental_seconds": 0., "standalone_total_seconds": 0.}]
    if params["include_red"]:
        runtime_rows.append({"method": "red", "timing_status": "fresh", "shared_seconds": shared_seconds,
                             "incremental_seconds": 0., "standalone_total_seconds": shared_seconds})
    for method in config.comparator_methods:
        fresh = method in requested_delphirf
        runtime_rows.append({"method": method,
            "timing_status": "fresh installed DelphiRF fit" if fresh else "prediction import; model-fit runtime unavailable",
            "shared_seconds": 0. if fresh else np.nan,
            "incremental_seconds": delphirf_seconds if fresh else np.nan,
            "standalone_total_seconds": delphirf_seconds if fresh else np.nan})
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(output / "runtime_by_method.csv", index=False)
    manifest = {"dataset": config.name, "cases": len(wide), "origins": int(wide.cutoff.nunique()),
                "methods": list(config.prediction_layers), "evaluation_scale": "raw",
                "value_transform": config.value_transform, "wall_seconds": perf_counter() - started,
                "method_params": params,
                "delphirf_provenance": delphirf_provenance,
                "uncertainty_layer": config.uncertainty_layer,
                "reliability_reference": (str(config.reliability_reference_file)
                                          if config.reliability_reference_file else None)}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    stage = perf_counter()
    build_report(long, output / "report", wide=wide, runtime=runtime,
                interval_calibration=reliability_outputs["interval_calibration"],
                reliability=reliability_outputs["reliability"],
                bootstrap=reliability_outputs["bootstrap"], rr_regimes=rr_audit,
                rr_clusters=rr_clusters,
                validation={k: v for k, v in reliability_outputs.items()
                            if k.startswith(("v1_", "v2_", "v3_", "v4_"))})
    record_stage("report_generation", stage)
    pd.DataFrame(pipeline_timings).to_csv(output / "runtime_pipeline_stages.csv", index=False)
    return manifest
