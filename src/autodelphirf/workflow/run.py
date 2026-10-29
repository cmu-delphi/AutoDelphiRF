from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

from ..comparators.red import ROUTED_TAUS
from dataclasses import replace
from ..core.io import (describe_skipped_locations, load_prepared_triangles, read_csv,
                       read_skipped_locations)
from ..post_assessment.scoring import wis
from ..core.defaults import provisional_column, to_reporting_scale
from ..comparators.red_prediction import replay
from ..comparators.red_routing import attach_revision_progress
from ..core.context import attach_issued_context
from ..core.history import enforce_target_maturity, target_as_of_date, validate_target_dates
from ..core.defaults import EVALUATION_DEFAULTS
from ..post_assessment.reliability import (add_interval_columns, pit_diagnostics,
                                           prospective_calibration,
                                           prospective_quantile_mapping,
                                           reliability_history)
from ..post_assessment.inference import bootstrap, primary_comparison_pairs
from ..post_assessment.validation import run_validation

from ..core.config import DatasetConfig, write_resolved_config
from ..comparators.schema import validate_comparator_file
from ..pre_diagnosis.diagnosis import diagnose_prepared_data, resolve_target_lag
from ..core.registry import LAYER_COLUMNS, PREDICTION_LAYERS, point_layer
from .methods import DELPHIRF_METHODS, train_requested_delphirf
from ..post_assessment.reporting import LEVELS, build_report

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
    """Build forecast cases from states visible in each test window.

    Only releases before the target lag (l < L) need a forecast. A case whose
    reference date has not reached t + L carries a missing truth (see
    ``enforce_target_maturity``): it is forecast, and scored only once a later
    archive supplies its target.
    """
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
        provisional = provisional_column(target)
        visible = visible[visible.log_value_7dav.notna() & visible[provisional].notna()].copy()
        if pd.notna(item.get("target_lag")):
            visible = visible[visible.lag < int(item.target_lag)].copy()
        visible["fold"] = int(item.get("fold_id", index + 1))
        visible["cutoff"] = cutoff
        visible["truth"] = pd.to_numeric(visible[target], errors="coerce")
        # Null carries forward the provisional value Y_itl on the target's scale.
        visible["Null"] = visible[provisional].astype(float)
        # Whether a release was actually reported at this lag (False for a value
        # carried forward by the grid), so evaluation can always be split by it.
        visible["genuine_event"] = (visible.genuine_event.fillna(False).astype(bool)
                                    if "genuine_event" in visible else pd.NA)
        rows.append(visible[KEYS + ["truth", "Null", "genuine_event"]])
    if not rows:
        return pd.DataFrame(columns=KEYS + ["truth", "Null", "genuine_event"])
    return pd.concat(rows, ignore_index=True)


#: Columns that must be integer-typed for the joins downstream to work.
#: ``merge_asof`` refuses to match an int64 key against a float64 one, and a
#: lag is conceptually a whole number of days either way.
INTEGER_COLUMNS = ("lag", "target_lag")


def normalize_lag_dtypes(prepared: pd.DataFrame) -> pd.DataFrame:
    """Coerce the lag columns to a nullable integer dtype.

    DelphiRF's ``data_preprocessing`` writes ``lag``/``target_lag`` as doubles
    (date arithmetic in R yields a numeric difftime), and a hand-built CSV
    triangle picks up float64 the moment one value is missing. Either one
    reaches joins and lag arithmetic that compare dtypes exactly and raise
    rather than coerce.
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


def _reporting(values, working_scale: str):
    """Working scale -> original reporting scale (see defaults.to_reporting_scale)."""
    return to_reporting_scale(values, working_scale)


def derive_working_scale(prepared: pd.DataFrame, config: DatasetConfig) -> str:
    """Determine the working scale from the triangle itself, not from a declaration.

    DelphiRF always smooths by adding one in the counting domain and then logs,
    so the working column's form follows the VALUE-COLUMN ARITY:

      one column (a count, or a rate supplied directly)
          -> log(value + 1)                      -> invert with expm1  ("log_value")
      a numerator/denominator pair
          -> log(num + 1) - log(denom + 1)       -> invert with exp    ("log_ratio")

    The names are DelphiRF's ``working_scale`` values.

    An arity-2 triangle carries the numerator and denominator feature families
    alongside the combined column, so their presence is the decisive evidence.
    This is derived rather than configured because it is a property of the data;
    a dataset with both counts is ALWAYS on the log-ratio scale.
    """
    base = "log_value_7dav"
    if f"{base}_num" in prepared.columns and f"{base}_denom" in prepared.columns:
        return "log_ratio"
    return "log_value"


def _check_transform(wide: pd.DataFrame, config: DatasetConfig) -> None:
    """Fail loudly when the declared working_scale cannot be the one that
    built this triangle, instead of silently emitting shifted raw values.

    log1p(x) is >= 0 for every x >= 0, so a negative working value proves the
    column is a log-ratio (log1p(num) - log1p(denom)) that must invert with
    exp. This is the mismatch that would otherwise report every quidel level
    as value - 1; difference-based metrics (AE/WIS/coverage) survive the shift,
    so nothing else would have flagged it.
    """
    if config.working_scale != "log_value" or "truth" not in wide:
        return
    truth = pd.to_numeric(wide["truth"], errors="coerce")
    smallest = truth.min()
    if pd.notna(smallest) and smallest < 0:
        raise ValueError(
            f"dataset '{config.name}' declares working_scale='log_value' but its working "
            f"column reaches {smallest:.4g} (< 0), so it cannot be log1p of a non-negative "
            "value. A num/denom triangle stores log1p(num) - log1p(denom); declare "
            "working_scale='log_ratio' for it.")


AUDIT_PASSTHROUGH = ("weight_fallback",
                     "effective_donor_count", "cold_start", "point_ess",
                     "max_normalized_donor_weight", "mean_curve_distance",
                     "mean_stage_distance", "revision_state_active")


def _method_rows(wide: pd.DataFrame, method: str, point: np.ndarray, quantiles: np.ndarray,
                 config: DatasetConfig, *, point_raw: np.ndarray | None = None,
                 quantiles_raw: np.ndarray | None = None) -> pd.DataFrame:
    """One method's standardized long rows, built as columns rather than dicts.

    ``point`` is the working-scale point prediction per case and ``quantiles``
    is ``(n_cases, len(ROUTED_TAUS))``, NaN where the method issues no
    distribution. Every requested method-case combination is retained. A
    method that could not fit or predict has ``model_available=False`` and
    missing prediction, quantile, and score fields.
    """
    frame = wide
    working_scale = config.working_scale
    truth_working = frame["truth"].to_numpy(float)
    truth_raw = _reporting(truth_working, working_scale)
    if point_raw is None:
        point_raw = _reporting(point, working_scale)
    # A row keeps its untransformed quantiles unless every one of them is
    # finite, matching the per-case rule: a partial distribution is not
    # back-transformed into a reportable one.
    complete = np.isfinite(quantiles).all(axis=1)
    if quantiles_raw is None:
        quantiles_raw = quantiles.copy()
        if complete.any():
            quantiles_raw[complete] = _reporting(quantiles[complete], working_scale)
    has_distribution = np.isfinite(quantiles_raw).all(axis=1)
    null_working = (frame["Null"].to_numpy(float) if "Null" in frame else truth_working)
    provisional = _reporting(null_working, working_scale)
    out = {key: (frame[key].to_numpy() if key in frame else None) for key in KEYS}
    out["genuine_event"] = frame["genuine_event"].to_numpy() if "genuine_event" in frame else None
    out.update(dataset=config.name, working_scale=working_scale,
               prediction_scale="reporting", method=method, truth=truth_raw, prediction=point_raw,
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
    # Convert the log-scale prediction/target difference into a percentage.
    # For log1p count models this is the relative difference between value + 1,
    # which remains defined when the target is zero; for log fraction models it
    # is the ordinary prediction/target ratio.
    relative_point = 100 * np.expm1(point - truth_working)
    out["relative_error_pct"] = np.abs(relative_point)
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
    scored_relative = np.full(len(frame), np.nan)
    if has_distribution.any():
        scored[has_distribution] = wis(quantiles_raw[has_distribution], truth_raw[has_distribution])
        scored_working[has_distribution] = wis(quantiles[has_distribution],
                                               truth_working[has_distribution])
        relative_quantiles = 100 * np.expm1(
            quantiles[has_distribution] - truth_working[has_distribution, None])
        scored_relative[has_distribution] = wis(
            relative_quantiles, np.zeros(has_distribution.sum()))
    # Evaluate every strategy on the original reporting scale. The scoring
    # function remains DelphiRF/evalcast's twice-mean-pinball definition.
    # The same names as DelphiRF's revision_forecast() scores.
    out["wis_reporting"], out["wis_working"] = scored, scored_working
    out["relative_wis_pct"] = scored_relative
    for field in AUDIT_PASSTHROUGH:
        if field in frame:
            out[field] = frame[field].to_numpy()
    return pd.DataFrame(out)


def standardize_predictions(wide: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Convert method-specific columns into one stable, long prediction schema."""
    if config.working_scale is None:
        raise ValueError(
            f"dataset '{config.name}' has no resolved working_scale. It is normally derived from "
            "the prepared triangle by run_pipeline() (see derive_working_scale); when calling "
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
            point_raw = quantiles_raw = None
            working_point_column = f"{point_column}_working"
            if method in DELPHIRF_METHODS and working_point_column not in wide:
                raise ValueError(
                    f"DelphiRF method '{method}' supplied no paired working-scale prediction; "
                    "install the current DelphiRF package and regenerate the model output")
            if working_point_column in wide:
                # Paired columns mean the producing model already performed
                # its own inverse transformation. The unsuffixed columns are
                # reporting-scale values; the suffixed columns are the exact
                # model-scale values used for scoring and calibration.
                point_raw = point
                point = wide[working_point_column].to_numpy(float)
                if tau_template is not None:
                    raw_quantiles = quantiles
                    working_columns = [f"{tau_template.format(tau=tau)}_working"
                                       for tau in ROUTED_TAUS]
                    if not all(column in wide for column in working_columns):
                        raise ValueError(
                            f"method '{method}' has raw predictions but incomplete "
                            "working-scale quantiles")
                    quantiles = np.column_stack(
                        [wide[column].to_numpy(float) for column in working_columns])
                    quantiles_raw = raw_quantiles
        else:
            # A custom layer registered through ``register_prediction_layer``
            # with no column template still works, one row at a time.
            extracted = [adapters[method](row, ROUTED_TAUS) for row in wide.to_dict("records")]
            point = np.array([float(x[0]) for x in extracted])
            quantiles = np.vstack([np.asarray(x[1], float) for x in extracted])
            point_raw = quantiles_raw = None
        frames.append(_method_rows(wide, method, point, quantiles, config,
                                   point_raw=point_raw, quantiles_raw=quantiles_raw))
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
    reference = read_csv(config.reliability_reference_file)
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
                                    wide: pd.DataFrame | None = None,
                                    pool_assignments: pd.DataFrame | None = None,
                                    mode: str = "full") -> dict[str, pd.DataFrame]:
    """Interval calibration from completed revisions, risk-score/alert, bootstrap CIs.

    Wired into the standard pipeline output (not an optional side
    computation): every dataset run produces a calibration table, a
    per-method reliability table for every method that issues genuine
    predictive quantiles, and episode-clustered bootstrap CIs for whichever
    primary-comparison pairs the configured ``prediction_layers`` support.
    """
    if mode not in {"full", "pit", "recalibration"}:
        raise ValueError("mode must be 'full', 'pit', or 'recalibration'")
    eligible = set(DISTRIBUTION_METHODS)
    if config.comparator_quantile_template:
        eligible |= set(config.comparator_methods)
    distribution_methods = ([m for m in config.prediction_layers if m in eligible]
                            if config.uncertainty_layer else [])
    if mode == "pit":
        base = long[long.method.isin(distribution_methods)].copy()
        if "model_available" in base:
            base = base[base.model_available]
        if "has_distribution" in base:
            base = base[base.has_distribution]
        quantile_reliability, pit = pit_diagnostics({"base": base})
        empty = pd.DataFrame()
        return {"calibration": empty, "calibration_variants": empty,
                "calibration_comparison": empty,
                "quantile_reliability": quantile_reliability, "pit_diagnostics": pit,
                "interval_calibration": empty, "reliability": empty, "bootstrap": empty}
    calibration = (prospective_calibration(long, distribution_methods,
                                            working_scale=config.working_scale) if distribution_methods
                  else long.iloc[:0].copy())
    if not calibration.empty:
        calibration["calibration_strategy"] = "endpoint"
    mapped_frames = []
    for strategy in ("location_lag", "pool"):
        if strategy == "pool" and (pool_assignments is None or pool_assignments.empty):
            continue
        mapped_frames.append(prospective_quantile_mapping(
            long, distribution_methods, strategy=strategy,
            pool_assignments=pool_assignments, working_scale=config.working_scale))
    calibration_variants = (pd.concat([frame for frame in mapped_frames if not frame.empty],
                                      ignore_index=True)
                            if any(not frame.empty for frame in mapped_frames)
                            else long.iloc[:0].copy())
    if not calibration.empty:
        base_intervals = add_interval_columns(calibration, LEVELS, prefix="")
        calibrated_source = calibration.drop(columns=[f"q{tau:g}" for tau in ROUTED_TAUS]).rename(
            columns={f"cal_q{tau:g}": f"q{tau:g}" for tau in ROUTED_TAUS})
        calibrated_intervals = add_interval_columns(calibrated_source, LEVELS, prefix="")
        interval_rows = []
        for level in LEVELS:
            for status, frame in (("base", base_intervals), ("calibrated", calibrated_intervals)):
                wis_column = "cal_wis_reporting" if status == "calibrated" else "wis_reporting"
                for method, group in frame.groupby("method", observed=True):
                    interval_rows.append({
                        "method": method, "status": status, "nominal": level / 100, "n": len(group),
                        "coverage": group[f"coverage_{level}"].mean(),
                        "coverage_error": group[f"coverage_{level}"].mean() - level / 100,
                        "mean_width": group[f"width_{level}"].mean(),
                        "median_width": group[f"width_{level}"].median(),
                        "lower_miss": group[f"lower_miss_{level}"].mean(),
                        "upper_miss": group[f"upper_miss_{level}"].mean(),
                        "mean_wis": group[wis_column].mean() if wis_column in group else np.nan,
                        "median_wis": group[wis_column].median() if wis_column in group else np.nan})
        interval_calibration = pd.DataFrame(interval_rows)
    else:
        interval_calibration = pd.DataFrame(columns=["method", "status", "nominal", "n", "coverage",
                                                      "coverage_error", "mean_width", "median_width",
                                                      "lower_miss", "upper_miss", "mean_wis", "median_wis"])
    comparison_rows = []
    calibration_sets = {"base": calibration, "endpoint": calibration}
    if not calibration_variants.empty:
        for strategy, frame in calibration_variants.groupby("calibration_strategy", observed=True):
            calibration_sets[strategy] = frame
    for strategy, frame in calibration_sets.items():
        if frame.empty:
            continue
        source = (frame if strategy == "base" else
                  frame.drop(columns=[f"q{tau:g}" for tau in ROUTED_TAUS]).rename(
                      columns={f"cal_q{tau:g}": f"q{tau:g}" for tau in ROUTED_TAUS}))
        measured = add_interval_columns(source, LEVELS)
        wis_column = "wis_reporting" if strategy == "base" else "cal_wis_reporting"
        for method, group in measured.groupby("method", observed=True):
            for level in LEVELS:
                comparison_rows.append({
                    "strategy": strategy, "method": method, "nominal": level / 100,
                    "n": len(group), "coverage": group[f"coverage_{level}"].mean(),
                    "coverage_error": group[f"coverage_{level}"].mean() - level / 100,
                    "mean_width": group[f"width_{level}"].mean(),
                    "median_width": group[f"width_{level}"].median(),
                    "lower_miss": group[f"lower_miss_{level}"].mean(),
                    "upper_miss": group[f"upper_miss_{level}"].mean(),
                    "mean_wis": group[wis_column].mean()})
    calibration_comparison = pd.DataFrame(comparison_rows)
    quantile_reliability, pit = pit_diagnostics(calibration_sets)
    if mode == "recalibration":
        empty = pd.DataFrame()
        return {"calibration": calibration, "calibration_variants": calibration_variants,
                "calibration_comparison": calibration_comparison,
                "quantile_reliability": quantile_reliability, "pit_diagnostics": pit,
                "interval_calibration": interval_calibration,
                "reliability": empty, "bootstrap": empty}
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
    validation = run_validation(long, wide=wide, calibration=calibration, reliability=reliability,
                                point_pairs=point_pairs, distribution_pairs=distribution_pairs,
                                taus=ROUTED_TAUS)
    return {"calibration": calibration, "calibration_variants": calibration_variants,
            "calibration_comparison": calibration_comparison,
            "quantile_reliability": quantile_reliability, "pit_diagnostics": pit,
            "interval_calibration": interval_calibration,
            "reliability": reliability, "bootstrap": bootstrap_rows, **validation}


def forecast_time_assessment(history: pd.DataFrame, newly_issued: pd.DataFrame,
                             config: DatasetConfig,
                             pool_assignments: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Attach prospective calibration and reliability only to new forecasts.

    Accuracy, coverage, and realized-harm fields remain unavailable until a
    later archive exposes the target. Historical ledger rows supply only the
    information that would have been observable at the new forecast cutoff.
    """
    assessment_history = _genuine_event_rows(history)
    assessment_new = _genuine_event_rows(newly_issued)
    if assessment_new.empty:
        empty = newly_issued.iloc[:0].copy()
        return {"calibration": empty, "calibration_variants": empty.copy(),
                "quantile_reliability": empty.copy(), "pit_diagnostics": empty.copy(),
                "reliability": empty.copy()}
    eligible = set(DISTRIBUTION_METHODS)
    if config.comparator_quantile_template:
        eligible |= set(config.comparator_methods)
    methods = ([method for method in config.prediction_layers if method in eligible]
               if config.uncertainty_layer else [])
    calibrated = (prospective_calibration(
        assessment_history, methods, working_scale=config.working_scale) if methods
                  else assessment_new.iloc[:0].copy())
    if not calibrated.empty:
        calibrated["calibration_strategy"] = "endpoint"
    variants = []
    for strategy in ("location_lag", "pool"):
        if strategy == "pool" and (pool_assignments is None or pool_assignments.empty):
            continue
        variants.append(prospective_quantile_mapping(
            assessment_history, methods, strategy=strategy, pool_assignments=pool_assignments,
            working_scale=config.working_scale))
    variants = (pd.concat([frame for frame in variants if not frame.empty], ignore_index=True)
                if any(not frame.empty for frame in variants) else assessment_new.iloc[:0].copy())
    reliability_parts = []
    for method in methods:
        error_reference, regret_reference = _load_reliability_reference(config, method)
        reliability_parts.append(reliability_history(
            assessment_history, method, error_reference=error_reference,
            regret_reference=regret_reference))
    reliability = (pd.concat(reliability_parts, ignore_index=True)
                   if reliability_parts else assessment_new.iloc[:0].copy())
    identity = ["method", "cutoff", "geo_value", "reference_date",
                "report_date", "target_date", "lag"]
    wanted = assessment_new[identity].drop_duplicates()
    def select(frame):
        if frame.empty:
            return frame
        selected = frame.merge(wanted.assign(_new=True), on=identity, how="inner")
        return selected.drop(columns="_new")
    diagnostic_sets = {"base": calibrated, "endpoint": calibrated}
    if not variants.empty:
        for strategy, frame in variants.groupby("calibration_strategy", observed=True):
            diagnostic_sets[strategy] = frame
    quantile_reliability, pit = pit_diagnostics(diagnostic_sets)
    calibrated, variants, reliability = select(calibrated), select(variants), select(reliability)
    for frame in (calibrated, variants, reliability):
        for column in ("realized_ae", "realized_null_ae", "realized_regret",
                       "realized_harm", "qerr_covered", "mharm_covered"):
            if column in frame:
                frame[column] = np.nan
    return {"calibration": calibrated, "calibration_variants": variants,
            "quantile_reliability": quantile_reliability, "pit_diagnostics": pit,
            "reliability": reliability}


def _genuine_event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return genuine forecast events, preserving legacy inputs without provenance."""
    if "genuine_event" not in frame:
        return frame
    return frame[frame.genuine_event.fillna(False).astype(bool)].copy()


def write_runtime_tables(output: Path, timing: pd.DataFrame, params: dict, config: DatasetConfig,
                         requested_delphirf, delphirf_seconds: float) -> pd.DataFrame:
    """Write per-origin and per-method runtime tables; return the per-method table."""
    timing.to_csv(output / "runtime_by_origin.csv", index=False)
    shared_columns = ["training_selection_seconds", "route_build_seconds", "red_prediction_seconds",
                      "residual_history_seconds"]
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
    return runtime


def run_pipeline(config: DatasetConfig, diagnosis_only: bool = False, should_stop=None,
                 *, model_mode: str = "train", incremental_update: bool = False,
                 post_assessment: bool = True) -> dict:
    started = perf_counter()
    pipeline_timings = []
    def record_stage(name, began):
        pipeline_timings.append({"stage": name, "seconds": perf_counter() - began})
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    from .operational import (append_forecast_ledger, ensure_analysis_contract,
                              exclude_issued_cases, raw_archive_state,
                              read_autodelphirf_preparation, recorded_archive_state,
                              read_deployment_origin, read_forecast_ledger,
                              refresh_forecast_ledger_truth, select_update_schedule,
                              write_deployment_origin, write_model_manifest)
    stage = perf_counter()
    preparation = read_autodelphirf_preparation(config.prepared_dir)
    # The diagnosis (Python) and preparation (R) clean the raw archive with one
    # contract; their retained rows legitimately differ by scope (lag bound,
    # as-of date), which each audit states. Only a contract mismatch is an error.
    diagnosis_audit = (preparation.get("diagnosis") or {}).get("raw_cleaning_audit") or {}
    preparation_audit_path = config.prepared_dir / "raw_cleaning_audit.json"
    if diagnosis_audit and preparation_audit_path.is_file():
        preparation_audit = json.loads(preparation_audit_path.read_text())
        if diagnosis_audit.get("contract_version") != preparation_audit.get("contract_version"):
            raise ValueError(
                f"dataset '{config.name}': the diagnosis used raw-cleaning contract "
                f"{diagnosis_audit.get('contract_version')} but preparation used "
                f"{preparation_audit.get('contract_version')}; re-run `autodelphirf prepare`.")
    prepared = load_prepared_triangles(config.prepared_dir)
    # A location preparation could not build is absent from every forecast and
    # score below, so say which and why, and keep the record with the results.
    skipped_locations = read_skipped_locations(config.prepared_dir)
    skipped_locations.to_csv(output / "skipped_locations.csv", index=False)
    if not skipped_locations.empty:
        print(f"{len(skipped_locations)} location(s) were skipped during preparation and are "
              "not forecast or evaluated:", flush=True)
        print("\n".join(describe_skipped_locations(skipped_locations)), flush=True)
    ensure_analysis_contract(config.prepared_dir, output)
    prepared = normalize_lag_dtypes(prepared)
    for column in ("reference_date", "report_date", "target_date"):
        prepared[column] = pd.to_datetime(prepared[column])
    derived_scale = derive_working_scale(prepared, config)
    if config.working_scale is not None and config.working_scale != derived_scale:
        raise ValueError(
            f"dataset '{config.name}' declares working_scale='{config.working_scale}' but its "
            f"prepared triangle is '{derived_scale}'. The working scale follows the value-column "
            "arity and is not a free choice: a numerator/denominator triangle is always "
            "log(num+1)-log(denom+1) ('log_ratio'); a single-column triangle is always "
            "log(value+1) ('log_value'). Remove the declaration to derive it, or correct it.")
    config = replace(config, working_scale=derived_scale)
    record_stage("load_prepared_data", stage)
    stage = perf_counter()
    full_schedule = pd.read_csv(config.prepared_dir / config.schedule_file,
                                parse_dates=["test_date", "experiment_end_date"])
    existing_ledger = read_forecast_ledger(output) if incremental_update else pd.DataFrame()
    retraining_required = None
    if incremental_update:
        schedule, retraining_required = select_update_schedule(
            full_schedule, read_deployment_origin(output))
    else:
        schedule = full_schedule
    resolved_preprocessing = preparation["resolved_preprocessing"]
    modeling_profile = {
        "weekday_groups": resolved_preprocessing.get("weekday_groups") or {},
        "weekday_groups_source": resolved_preprocessing.get("weekday_groups_source"),
        "lag_terms": list(resolved_preprocessing.get("lag_terms") or []),
    }
    # The weekday groups and lag terms are fixed at preparation and recorded in
    # preparation.json, which training reads. A revision_profile left in a
    # dataset JSON by an older version is not a second setting: it must agree.
    stale = config.revision_profile or {}
    conflicts = [key for key in ("weekday_groups", "lag_terms")
                 if key in stale and stale[key] != modeling_profile[key]]
    if conflicts:
        raise ValueError(
            f"dataset '{config.name}': revision_profile in the dataset JSON sets {conflicts} "
            "differently from the prepared dataset, which is what training uses. Weekday groups "
            "and lag terms are chosen at preparation: set preprocessing.weekday_groups (ISO "
            "weekday numbers) in the raw configuration and re-run `autodelphirf prepare`, then "
            "remove revision_profile from the dataset JSON.")
    summary, missingness = diagnose_prepared_data(prepared, full_schedule, modeling_profile)
    (output / "diagnostics").mkdir(exist_ok=True)
    (output / "diagnostics" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    missingness.to_csv(output / "diagnostics" / "missingness.csv", index=False)
    record_stage("input_diagnosis", stage)
    if diagnosis_only:
        return summary
    # The target lag is settled before any case is built or model trained.
    # Target lag L is the user's choice, made at `autodelphirf prepare` and
    # recorded in the prepared schedule: the triangle's target column and
    # target_date encode it. Its evidence comes only from the raw archive and
    # is recorded in preparation.json; a prepared triangle is never diagnosed.
    built = (set(schedule.target_lag.dropna().astype(int)) if "target_lag" in schedule
             else set())
    if len(built) > 1:
        raise ValueError(f"dataset '{config.name}': the schedule mixes target lags "
                         f"{sorted(built)}; a run uses one target lag")
    user_lag = built.pop() if built else None
    recorded = preparation.get("diagnosis") or {}
    recorded_range = recorded.get("target_lag_suggested_range")
    diagnosed = {
        "selected_target_lag": None,
        "suggested_target_lag_range": recorded_range,
        "status": ("skipped: the target lag was specified"
                   if user_lag is not None and (recorded_range is None or recorded_range.get("skipped"))
                   else "raw archive diagnosis recorded during preprocessing"
                   if recorded_range is not None else "not diagnosed"),
        "completion": recorded.get("target_lag_completion_curve", {}),
        "source": "preparation.json (raw archive)",
    }
    resolved, reason = resolve_target_lag(diagnosed, user_lag, user_lag is not None, config.name)
    print(f"Target lag: {resolved} days ({reason}); diagnostic range: "
          f"{diagnosed.get('suggested_target_lag_range')}", flush=True)
    target_lag_record = {"resolved": resolved, "reason": reason,
                         "user_target_lag": user_lag, "user_confirmed": user_lag is not None,
                         "diagnosed": diagnosed}
    (output / "target_lag_resolution.json").write_text(
        json.dumps(target_lag_record, indent=2, default=str) + "\n")
    # One target definition for training, scoring, calibration, reliability and
    # the ledger: the prepared target_date (t + L), used as stored, and the
    # as-of date recorded at preparation.
    target_column = str(schedule.target_column.iloc[0])
    validate_target_dates(prepared, resolved)
    as_of_date = target_as_of_date(full_schedule)
    prepared = enforce_target_maturity(prepared, as_of_date)
    if incremental_update:
        truth_update = refresh_forecast_ledger_truth(
            prepared, target_column, config.working_scale, output, as_of_date=as_of_date)
        existing_ledger = read_forecast_ledger(output)
        issued_forecasts_file = output / "_issued_forecast_cases.csv"
        issued_columns = ["cutoff", "geo_value", "reference_date", "report_date",
                          "target_date", "lag"]
        (existing_ledger[issued_columns].drop_duplicates()
         if not existing_ledger.empty else pd.DataFrame(columns=issued_columns)).to_csv(
             issued_forecasts_file, index=False)
        incremental_cases = exclude_issued_cases(
            build_cases(prepared, schedule, config.genuine_testing),
            existing_ledger)
    else:
        truth_update = None
        issued_forecasts_file = None
        incremental_cases = None
    # Every DelphiRF-labelled method crosses one auditable boundary into the
    # installed R package. RevRoute itself contributes only the per-origin
    # task pools consumed by the default ``delphirf`` method.
    requested_delphirf = tuple(
        method for method in config.prediction_layers if method in DELPHIRF_METHODS)
    delphirf_seconds, delphirf_provenance = 0., None
    # Retraining is scheduled by calendar origin, independently of whether a
    # report happens to arrive on that date.  A quiet retraining date must
    # still persist the new models for the next release; it simply produces no
    # forecasts during this update.
    should_run_delphirf = (
        not incremental_update or retraining_required or not incremental_cases.empty
    )
    if requested_delphirf and should_run_delphirf:
        # DelphiRF returns paired reporting- and working-scale predictions.
        # AutoDelphiRF preserves both without independently inverting the
        # model output.
        trained = output / "_delphirf_predictions.csv.gz"
        model_dir = output / "models"
        _, delphirf_seconds, delphirf_provenance = train_requested_delphirf(
            config, prepared, schedule, requested_delphirf, trained,
            model_dir=model_dir, model_mode=model_mode,
            issued_forecasts=issued_forecasts_file,
            unresolved_only=incremental_update)
        write_model_manifest(model_dir, output, delphirf_provenance)
        if incremental_update and retraining_required:
            write_deployment_origin(output, pd.Timestamp(schedule.test_date.iloc[0]))
        elif not incremental_update:
            # A retrospective run trains every scheduled origin and leaves the
            # final one ready for operational use.  Persist that fact alongside
            # the saved models so the first later update can reuse it directly
            # instead of treating the deployment as uninitialized.
            write_deployment_origin(output, pd.to_datetime(schedule.test_date).max())
        config = replace(
            config, comparator_file=trained,
            comparator_methods={**config.comparator_methods,
                                **{name: DELPHIRF_METHODS[name]
                                   for name in requested_delphirf}},
            comparator_quantile_template="q{tau:g}")
    # Freeze every effective default and bundle any external assessment input.
    # This is the restart boundary used by ``autodelphirf assess``; it is
    # intentionally written before post-forecast work begins.
    from ..post_assessment.run import freeze_reliability_reference
    config, reliability_input = freeze_reliability_reference(config, output)
    write_resolved_config(config, output / "resolved_config.json")
    input_manifest = {"reliability_reference": reliability_input}
    (output / "assessment_inputs.json").write_text(
        json.dumps(input_manifest, indent=2, default=str) + "\n")
    stage = perf_counter()
    cases = (incremental_cases if incremental_update else
             build_cases(prepared, schedule, config.genuine_testing))
    if cases.empty:
        if incremental_update:
            raw_update = raw_archive_state(config.prepared_dir, output)
            manifest = {
                "dataset": config.name, "cases": 0, "origins": 0,
                "methods": list(config.prediction_layers), "incremental_update": True,
                "retraining_required": retraining_required,
                "new_forecasts": 0, "truth_update": truth_update,
                "raw_archive_update": raw_update,
                "forecast_ledger": {"path": str(output / "forecast_ledger.csv.gz"),
                                    "previous_rows": len(existing_ledger), "new_rows": 0,
                                    "total_rows": len(existing_ledger)},
                "wall_seconds": perf_counter() - started,
            }
            (output / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str) + "\n")
            return manifest
        raise ValueError("the configured test windows produced no forecast cases")
    cases = attach_issued_context(attach_revision_progress(cases, prepared), prepared)
    record_stage("case_construction_and_features", stage)
    params = dict(config.method_params)
    if config.initial_lag is not None:
        params["initial_lag"] = config.initial_lag
    params["include_red"] = "red" in config.prediction_layers
    params["target_column"] = str(schedule.target_column.iloc[0])
    params["target_lag"] = resolved
    stage = perf_counter()
    # Training windows are schedule data, not a dataset-wide constant. Pass
    # the fold -> days mapping so retrospective studies may vary them across
    # origins without splitting the dataset into separate runs.
    fold_ids = (schedule["fold_id"].astype(int) if "fold_id" in schedule
                else pd.Series(schedule.index + 1, index=schedule.index))
    training_windows = dict(zip(fold_ids, schedule.training_days.astype(int)))
    wide, timing = replay(cases, prepared, training_windows,
                          should_stop=should_stop, **params)
    record_stage("revroute_execution", stage)
    rr_audit = rr_clusters = None
    if config.comparator_file:
        stage = perf_counter()
        comparator = read_csv(config.comparator_file)
        if not (incremental_update and comparator.empty):
            validate_comparator_file(config.comparator_file, config.comparator_method_column,
                                     config.comparator_prediction_column,
                                     config.comparator_methods, config.comparator_columns)
        for column in ("cutoff", "reference_date", "report_date", "target_date"):
            if column in comparator:
                comparator[column] = pd.to_datetime(comparator[column])
        merge_keys = [key for key in KEYS if key in comparator]
        template = config.comparator_quantile_template
        for output_name, source_name in config.comparator_methods.items():
            selected = comparator[comparator[config.comparator_method_column].eq(source_name)]
            wanted = {config.comparator_prediction_column: output_name}
            # DelphiRF owns the inverse transformation and emits both scales.
            # Keep the paired working-scale columns alongside the public raw
            # predictions; AutoDelphiRF must not reconstruct model outputs.
            if "prediction_working" in selected.columns:
                wanted["prediction_working"] = f"{output_name}_working"
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
                working_quantiles = {
                    f"{template.format(tau=tau)}_working":
                    f"{output_name}_tau{tau:g}_working"
                    for tau in ROUTED_TAUS
                }
                present_working = {source: destination
                                   for source, destination in working_quantiles.items()
                                   if source in selected.columns}
                if present_working and len(present_working) != len(working_quantiles):
                    raise ValueError(
                        f"comparator '{source_name}' has an incomplete set of working-scale "
                        "quantile columns")
                wanted.update(present_working)
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
    # A failed model forecast is recorded as unavailable (model_available False,
    # no prediction), in the ledger as everywhere else. Its case is issued once
    # and never forecast again by a later update.
    record_stage("standardize_predictions", stage)
    stage = perf_counter()
    scored_long = long[long.truth.notna()].copy()
    scored_wide = wide[wide.truth.notna()].copy()
    if not post_assessment:
        if incremental_update:
            raise ValueError("an update always assesses its newly issued forecasts; "
                             "post_assessment=False applies to a full run only")
        # Predictions and their evaluation (truth, AE, WIS, relative error) only.
        # Calibration, reliability, bootstrap and the report are left to a later
        # `autodelphirf assess`, which restarts from the files written here.
        wide.to_csv(output / "predictions_wide.csv.gz", index=False, compression="gzip")
        long.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
        # The run has recorded its deployment origin, so a later update needs
        # the issued forecasts too; the ledger is part of the predictions.
        ledger = append_forecast_ledger(long, output)
        write_runtime_tables(output, timing, params, config, requested_delphirf, delphirf_seconds)
        manifest = {"dataset": config.name, "cases": len(wide),
                    "origins": int(wide.cutoff.nunique()),
                    "methods": list(config.prediction_layers), "evaluation_scale": "reporting",
                    "working_scale": config.working_scale,
                    "wall_seconds": perf_counter() - started, "method_params": params,
                    "delphirf_provenance": delphirf_provenance, "model_mode": model_mode,
                    "post_assessment": False, "forecast_ledger": ledger,
                    "delphirf_seconds": delphirf_seconds,
                    "skipped_locations": skipped_locations.to_dict("records")}
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        pd.DataFrame(pipeline_timings).to_csv(output / "runtime_pipeline_stages.csv", index=False)
        return manifest
    if incremental_update:
        ledger = append_forecast_ledger(long, output)
        combined_ledger = read_forecast_ledger(output)
        pool_path = output / "revroute_pools.csv"
        pool_assignments = (read_csv(pool_path, parse_dates=["cutoff"])
                            if pool_path.is_file() else None)
        forecast_outputs = forecast_time_assessment(
            combined_ledger, long, config, pool_assignments=pool_assignments)
        reliability_outputs = {
            "calibration": forecast_outputs["calibration"],
            "calibration_variants": forecast_outputs["calibration_variants"],
            "quantile_reliability": forecast_outputs["quantile_reliability"],
            "pit_diagnostics": forecast_outputs["pit_diagnostics"],
            "reliability": forecast_outputs["reliability"],
            "calibration_comparison": pd.DataFrame(),
            "interval_calibration": pd.DataFrame(), "bootstrap": pd.DataFrame(),
        }
    else:
        pool_path = output / "revroute_pools.csv"
        pool_assignments = (read_csv(pool_path, parse_dates=["cutoff"])
                            if pool_path.is_file() else None)
        # Calibration and PIT describe model-issued forecasts, not values that
        # were merely carried forward.  The main report still receives the
        # complete scored population below, preserving its historical scope.
        assessment_long = _genuine_event_rows(scored_long)
        assessment_wide = _genuine_event_rows(scored_wide)
        reliability_outputs = run_reliability_and_calibration(
            assessment_long, config, wide=assessment_wide,
            pool_assignments=pool_assignments)
    record_stage("calibration_and_reliability", stage)
    wide.to_csv(output / "predictions_wide.csv.gz", index=False, compression="gzip")
    long.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    if not incremental_update:
        ledger = append_forecast_ledger(long, output)
    # An update has just re-prepared from the current raw archive; a rerun of
    # an already-prepared dataset uses the state recorded at preparation and
    # never rereads the raw archive.
    raw_update = (raw_archive_state(config.prepared_dir, output) if incremental_update
                  else recorded_archive_state(config.prepared_dir, output))
    reliability_outputs["calibration"].to_csv(output / "calibration.csv.gz", index=False, compression="gzip")
    reliability_outputs["interval_calibration"].to_csv(output / "interval_calibration.csv", index=False)
    reliability_outputs["reliability"].to_csv(output / "reliability.csv.gz", index=False, compression="gzip")
    reliability_outputs["bootstrap"].to_csv(output / "bootstrap_pairwise_comparisons.csv", index=False)
    reliability_outputs["calibration_variants"].to_csv(
        output / "calibration_variants.csv.gz", index=False, compression="gzip")
    reliability_outputs["calibration_comparison"].to_csv(
        output / "calibration_strategy_comparison.csv", index=False)
    reliability_outputs["quantile_reliability"].to_csv(
        output / "quantile_reliability.csv", index=False)
    reliability_outputs["pit_diagnostics"].to_csv(
        output / "pit_diagnostics.csv.gz", index=False, compression="gzip")
    if incremental_update:
        (output / "forecast_assessment_status.json").write_text(json.dumps({
            "scope": "newly issued unresolved forecasts only",
            "new_forecast_rows": len(long),
            "calibrated_forecast_rows": len(reliability_outputs["calibration"]),
            "reliability_rows": len(reliability_outputs["reliability"]),
            "realized_scores": "pending until target values become observable",
            "truth_rows_resolved_this_update": truth_update["resolved_rows"],
            "truth_rows_corrected_this_update": truth_update["corrected_rows"],
            "truth_rows_withheld_until_target_date": truth_update["withheld_rows"],
        }, indent=2) + "\n")
    for name, frame in reliability_outputs.items():
        if name.startswith(("v1_", "v2_", "v3_", "v4_")) and frame is not None and not frame.empty:
            frame.to_csv(output / f"{name}.csv", index=False)
    runtime = write_runtime_tables(output, timing, params, config, requested_delphirf,
                                   delphirf_seconds)
    manifest = {"dataset": config.name, "cases": len(wide), "origins": int(wide.cutoff.nunique()),
                "methods": list(config.prediction_layers), "evaluation_scale": "reporting",
                "working_scale": config.working_scale, "wall_seconds": perf_counter() - started,
                "method_params": params,
                "delphirf_provenance": delphirf_provenance,
                "uncertainty_layer": config.uncertainty_layer,
                "model_mode": model_mode, "forecast_ledger": ledger,
                "incremental_update": incremental_update,
                "retraining_required": retraining_required,
                "new_forecasts": len(long) if incremental_update else None,
                "truth_update": truth_update,
                "raw_archive_update": raw_update,
                "skipped_locations": skipped_locations.to_dict("records"),
                "reliability_reference": (str(config.reliability_reference_file)
                                          if config.reliability_reference_file else None)}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    stage = perf_counter()
    if not incremental_update:
        build_report(scored_long, output / "report", wide=scored_wide, runtime=runtime,
                    skipped_locations=skipped_locations,
                    interval_calibration=reliability_outputs["interval_calibration"],
                    calibration_comparison=reliability_outputs["calibration_comparison"],
                    quantile_reliability=reliability_outputs["quantile_reliability"],
                    reliability=reliability_outputs["reliability"],
                    bootstrap=reliability_outputs["bootstrap"], rr_regimes=rr_audit,
                    rr_clusters=rr_clusters,
                    validation={k: v for k, v in reliability_outputs.items()
                                if k.startswith(("v1_", "v2_", "v3_", "v4_"))},
                    dataset_name=config.name)
    record_stage("report_generation", stage)
    pd.DataFrame(pipeline_timings).to_csv(output / "runtime_pipeline_stages.csv", index=False)
    return manifest
