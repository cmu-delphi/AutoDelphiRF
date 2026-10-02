"""Clustered bootstrap confidence intervals for method comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .defaults import EVALUATION_DEFAULTS

DEFAULT_REPS = EVALUATION_DEFAULTS.get("bootstrap_repetitions", 500)
DEFAULT_KEY = ["fold", "geo_value", "reference_date", "report_date", "lag"]


def primary_comparison_pairs(methods: dict[str, str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build default point and distribution comparison pairs.

    ``methods`` maps a role to the method name actually present in the
    data, e.g. ``{"red": "red", "null": "baseline_null",
    "naive": "naive_delphirf", "similarity": "similarity_weighted_delphirf",
    "hybrid_calendar": "residual_calendar", "hybrid_full": "residual_full"}``.
    Missing roles are skipped. Point comparisons use the carry-forward method
    as the reference when it is available. Distribution comparisons use RED as
    the reference when selected, otherwise the main DelphiRF method.
    """
    ordered_roles = ("candidate", "red", "naive", "global", "similarity")
    available = [methods[role] for role in ordered_roles if role in methods]
    point_reference = methods.get("null") or methods.get("red")
    point_pairs = ([(name, point_reference) for name in available if name != point_reference]
                   if point_reference else [])

    distribution_reference = methods.get("red") or methods.get("candidate")
    distribution_pairs = (
        [(name, distribution_reference) for name in available if name != distribution_reference]
        if distribution_reference else [])
    return point_pairs, distribution_pairs


def bootstrap(data: pd.DataFrame, subset: str, point_pairs: list[tuple[str, str]],
              distribution_pairs: list[tuple[str, str]], *, reps: int | None = None, seed: int = 8,
              key: list[str] = DEFAULT_KEY, episode_column: str = "reference_date") -> pd.DataFrame:
    """Episode-clustered bootstrap CIs for mean-AE / mean-WIS differences.

    The reference date is the resampling unit: all rows sharing one reference
    date are resampled together, never independently.
    ``data`` must be a long prediction table with columns ``method``,
    ``absolute_error``, ``wis``, and every column in ``key``.
    """
    reps = DEFAULT_REPS if reps is None else reps
    rng = np.random.default_rng(seed)
    rows = []
    for metric, pairs in (("absolute_error", point_pairs), ("wis", distribution_pairs)):
        for left, right in pairs:
            pivot = data.pivot_table(index=key, columns="method", values=metric, aggfunc="first")
            if left not in pivot or right not in pivot:
                continue
            pivot = pivot.dropna(subset=[left, right])
            if pivot.empty:
                continue
            diff = (pivot[left] - pivot[right]).reset_index(name="difference")
            cluster = diff.groupby(episode_column)["difference"].agg(["sum", "size"])
            if cluster.empty:
                continue
            index = rng.integers(0, len(cluster), size=(reps, len(cluster)))
            values = cluster["sum"].to_numpy()[index].sum(1) / cluster["size"].to_numpy()[index].sum(1)
            rows.append({"subset": subset, "comparison": f"{left} - {right}",
                        "metric": "mean AE" if metric == "absolute_error" else "mean WIS",
                        "estimate": diff["difference"].mean(),
                        "ci_low": np.quantile(values, .025), "ci_high": np.quantile(values, .975),
                        "improvement_probability": float(np.mean(values < 0)), "episodes": len(cluster)})
    return pd.DataFrame(rows)
