"""Dataset configuration contracts shared across workflow stages."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import os



def _working_scale_field(raw: dict) -> str | None:
    """Read DelphiRF's ``working_scale`` from a dataset config."""
    if "value_transform" in raw:
        raise ValueError(
            "dataset config field 'value_transform' was renamed to 'working_scale', with "
            "DelphiRF's values 'log_value' or 'log_ratio'; rename it, or remove it so the "
            "working scale is derived from the prepared triangle")
    scale = raw.get("working_scale")
    if scale is not None and scale not in ("log_value", "log_ratio"):
        raise ValueError(f"working_scale must be 'log_value' or 'log_ratio', got {scale!r}")
    return scale


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    prepared_dir: Path
    output_dir: Path
    schedule_file: str
    # DelphiRF's working scale, "log_value" or "log_ratio". None means "derive
    # from the prepared triangle" (the normal case); a declared value is
    # validated against the derived one, never trusted over it.
    working_scale: str | None
    initial_lag: int | None
    # Independent controls for DelphiRF model fitting and forecast cases.
    genuine_training: bool
    genuine_testing: bool
    # Smallest number of training rows a RevRoute pool may have; smaller pools
    # are merged into their nearest neighbouring pool. 0 keeps the pools as
    # the similarity search builds them. Preparation always writes it; a
    # dataset file without it predates the option and ran with no minimum.
    revroute_min_pool_rows: int
    prediction_layers: tuple[str, ...]
    comparator_file: Path | None
    comparator_columns: dict[str, str]
    comparator_method_column: str
    comparator_prediction_column: str
    comparator_methods: dict[str, str]
    # Column template for a comparator that supplies a full predictive
    # distribution, e.g. "q{tau:g}" for columns q0.01 ... q0.99 on the routed
    # quantile grid. Without it an imported method stays point-only, which is
    # the historical behaviour and still the right one for a comparator file
    # that genuinely carries only a median.
    comparator_quantile_template: str | None
    method_params: dict
    # Optional user-specified reporting calendar. When present it OVERRIDES
    # the diagnosis-derived profile; the derived values are still reported so
    # any disagreement is visible.
    revision_profile: dict | None
    # Optional frozen reliability reference: a CSV with
    # q_error/q_regret columns (and optionally case_n) built from a separate
    # replay that ends strictly before this dataset's outer evaluation
    # window, e.g. via post_assessment.reliability.build_reliability_reference.
    # When absent, the risk-score layer still reports continuous quantities
    # (q_error, q_regret, harm_frequency) but the categorical alert reports
    # "reference distribution unavailable" as the configured fallback.
    reliability_reference_file: Path | None
    # Interval calibration from completed revisions and per-case risk score are
    # per-row prospective scans over growing completed revision history. That is fine at
    # a few thousand evaluated cases and prohibitive at a few million (chng
    # issues 1.77M), so a dataset may switch them off. Bootstrap CIs are not
    # affected. Default True: every existing dataset keeps the full layer.
    uncertainty_layer: bool


def load_dataset_config(path: Path) -> DatasetConfig:
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    base = path.parent
    def resolve(value):
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else (base / p).resolve()
    required = {"name", "prepared_dir", "output_dir"}
    missing = required.difference(raw)
    if "model_methods" not in raw and "prediction_layers" not in raw:
        missing.add("prediction_layers/model_methods")
    if missing:
        raise ValueError(f"dataset config missing: {sorted(missing)}")
    if ("model_methods" in raw and "prediction_layers" in raw and
            tuple(raw["model_methods"]) != tuple(raw["prediction_layers"])):
        raise ValueError("model_methods and compatibility alias prediction_layers disagree")
    if "include_genuine_events_only" in raw:
        raise ValueError(
            "include_genuine_events_only has been removed; set genuine_training and "
            "genuine_testing explicitly")
    genuine_training = raw.get("genuine_training", True)
    genuine_testing = raw.get("genuine_testing", True)
    for name, value in (("genuine_training", genuine_training),
                        ("genuine_testing", genuine_testing)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
    min_pool_rows = raw.get("revroute_min_pool_rows", 0)
    if isinstance(min_pool_rows, bool) or not isinstance(min_pool_rows, int) or min_pool_rows < 0:
        raise ValueError("revroute_min_pool_rows must be a nonnegative integer")
    return DatasetConfig(
        name=raw["name"], prepared_dir=resolve(raw["prepared_dir"]),
        output_dir=resolve(raw["output_dir"]), schedule_file=raw.get("schedule_file", "test_dates.csv"),
        working_scale=_working_scale_field(raw), initial_lag=raw.get("initial_lag"),
        genuine_training=genuine_training, genuine_testing=genuine_testing,
        revroute_min_pool_rows=min_pool_rows,
        prediction_layers=tuple(raw.get("model_methods", raw.get("prediction_layers", ()))),
        comparator_file=resolve(raw.get("comparator_file")),
        comparator_columns=dict(raw.get("comparator_columns", {})),
        comparator_method_column=raw.get("comparator_method_column", "method"),
        comparator_prediction_column=raw.get("comparator_prediction_column", "prediction"),
        comparator_methods=dict(raw.get("comparator_methods", {})),
        comparator_quantile_template=raw.get("comparator_quantile_template"),
        method_params=dict(raw.get("method_params", {})),
        revision_profile=(dict(raw["revision_profile"]) if raw.get("revision_profile") else None),
        reliability_reference_file=resolve(raw.get("reliability_reference_file")),
        uncertainty_layer=bool(raw.get("uncertainty_layer", True)))


def write_resolved_config(config: DatasetConfig, path: Path) -> Path:
    """Write the effective, post-resolution configuration for later assessment.

    Unlike the user-authored dataset JSON, this snapshot includes defaults and
    the value transform derived from the prepared triangle. Paths are relative
    to the snapshot where possible so a complete run directory can be moved.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def relative(value):
        if value is None:
            return None
        target = Path(value).resolve()
        try:
            return os.path.relpath(target, path.parent)
        except ValueError:
            return str(target)

    payload = {
        "_generated_by": "autodelphirf run",
        "name": config.name,
        "prepared_dir": relative(config.prepared_dir),
        "output_dir": relative(config.output_dir),
        "schedule_file": config.schedule_file,
        "working_scale": config.working_scale,
        "initial_lag": config.initial_lag,
        "genuine_training": config.genuine_training,
        "genuine_testing": config.genuine_testing,
        "revroute_min_pool_rows": config.revroute_min_pool_rows,
        "model_methods": list(config.prediction_layers),
        "prediction_layers": list(config.prediction_layers),
        "comparator_file": relative(config.comparator_file),
        "comparator_columns": config.comparator_columns,
        "comparator_method_column": config.comparator_method_column,
        "comparator_prediction_column": config.comparator_prediction_column,
        "comparator_methods": config.comparator_methods,
        "comparator_quantile_template": config.comparator_quantile_template,
        "method_params": config.method_params,
        "reliability_reference_file": relative(config.reliability_reference_file),
        "uncertainty_layer": config.uncertainty_layer,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
