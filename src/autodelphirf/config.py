from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

from .resources import load_resource


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    prepared_dir: Path
    output_dir: Path
    schedule_file: str
    # None means "derive from the prepared triangle" (the normal case).
    # A declared value is validated against the derived one, never trusted over it.
    value_transform: str | None
    initial_lag: int | None
    include_genuine_events_only: bool
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
    input_columns: dict[str, str]
    method_params: dict
    # Optional user-specified reporting calendar. When present it OVERRIDES
    # the diagnosis-derived profile; the derived values are still reported so
    # any disagreement is visible.
    revision_profile: dict | None
    # Optional frozen reliability reference: a CSV with
    # q_error/q_regret columns (and optionally case_n) built from a separate
    # replay that ends strictly before this dataset's outer evaluation
    # window, e.g. via autodelphirf.calibration.build_reliability_reference.
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
    schema = load_resource("prepared_triangle_schema.json")
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
    return DatasetConfig(
        name=raw["name"], prepared_dir=resolve(raw["prepared_dir"]),
        output_dir=resolve(raw["output_dir"]), schedule_file=raw.get("schedule_file", "test_dates.csv"),
        value_transform=raw.get("value_transform"), initial_lag=raw.get("initial_lag"),
        include_genuine_events_only=raw.get("include_genuine_events_only", True),
        prediction_layers=tuple(raw.get("model_methods", raw.get("prediction_layers", ()))),
        comparator_file=resolve(raw.get("comparator_file")),
        comparator_columns=dict(raw.get("comparator_columns", {})),
        comparator_method_column=raw.get("comparator_method_column", "method"),
        comparator_prediction_column=raw.get("comparator_prediction_column", "prediction"),
        comparator_methods=dict(raw.get("comparator_methods", {})),
        comparator_quantile_template=raw.get("comparator_quantile_template"),
        input_columns={**schema["columns"], **dict(raw.get("input_columns", {}))},
        method_params=dict(raw.get("method_params", {})),
        revision_profile=(dict(raw["revision_profile"]) if raw.get("revision_profile") else None),
        reliability_reference_file=resolve(raw.get("reliability_reference_file")),
        uncertainty_layer=bool(raw.get("uncertainty_layer", True)))
