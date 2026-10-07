"""AutoDelphiRF's command-line, Python, and web forecasting pipeline."""

#: Distribution version. Kept in step with ``pyproject.toml``.
__version__ = "0.1.1.dev0"

from .calibration import (add_interval_columns, build_reliability_reference,
                          calibrate, calibration_stratum, forecast_quality_summary,
                          prospective_calibration, reliability_history)
from .config import DatasetConfig, load_dataset_config, write_resolved_config
from .ingest import diagnose_raw_archive, resolve_target_lag
from .io import load_prepared_triangles, triangle_files
from .prepare import PreparationError, PreparationSpec, build_spec, prepare_dataset
from .resources import load_resource, resource_path, user_config_path
from .defaults import (CALIBRATION_DEFAULTS, EVALUATION_DEFAULTS, PROCESS_DEFAULTS,
                       RESIDUAL_DEFAULTS, ROUTING_DEFAULTS, RR_DELPHIRF_DEFAULTS, VERSION)
from .engine import replay
from .inference import bootstrap, primary_comparison_pairs
from .metrics import wis
from .red import ROUTED_TAUS, weighted_median, weighted_quantiles
from .registry import PREDICTION_LAYERS, register_prediction_layer
from .context import attach_issued_context
from .revroute import learn_task_pools
from .model_training import (DELPHIRF_METHODS, MODEL_METHODS,
                             RR_DELPHIRF_CODE_VERSION, train_delphirf)
from .weighting import (FEATURE, attach_revision_progress, build_curve_routing,
                        compute_weights, final_v7_weights, robust_mad_scale,
                        rolling_completed_training)

__all__ = [
    "__version__",
    "DatasetConfig", "load_dataset_config", "write_resolved_config",
    "diagnose_raw_archive", "resolve_target_lag",
    "load_prepared_triangles", "triangle_files",
    "PreparationError", "PreparationSpec", "build_spec", "prepare_dataset",
    "load_resource", "resource_path", "user_config_path",
    "PREDICTION_LAYERS", "register_prediction_layer",
    "VERSION", "ROUTING_DEFAULTS", "RESIDUAL_DEFAULTS", "PROCESS_DEFAULTS",
    "RR_DELPHIRF_DEFAULTS", "learn_task_pools", "DELPHIRF_METHODS",
    "MODEL_METHODS", "RR_DELPHIRF_CODE_VERSION", "train_delphirf",
    "CALIBRATION_DEFAULTS", "EVALUATION_DEFAULTS",
    "replay", "wis", "ROUTED_TAUS", "weighted_quantiles", "weighted_median",
    "attach_issued_context", "attach_revision_progress",
    "build_curve_routing", "compute_weights", "final_v7_weights",
    "robust_mad_scale", "rolling_completed_training", "FEATURE",
    "calibration_stratum", "calibrate", "prospective_calibration",
    "add_interval_columns", "forecast_quality_summary",
    "build_reliability_reference", "reliability_history",
    "bootstrap", "primary_comparison_pairs",
]
