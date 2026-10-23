"""Stage 3: RevRoute task pooling, with no estimator implementation.

RevRoute ends at a prospective ``(location, lag) -> pool`` assignment for
each retraining origin.  The following model-training stage sends those pools
to the installed DelphiRF package.  Keeping this module estimator-free makes
it impossible to accidentally label a Python quantile regression as DelphiRF.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .clustering import sparse_task_pooling
from ...core.history import rolling_completed_training
from ...core.defaults import provisional_column
from ...core.io import read_csv


def learn_task_pools(prepared: pd.DataFrame, schedule: pd.DataFrame, output: Path,
                     *, preserve_existing: bool = False) -> Path:
    """Learn RevRoute pools independently at every scheduled origin.

    Operational updates must retain the routing decision issued at an earlier
    origin.  New raw releases can add test tasks to that origin, but they must
    not silently relabel its pools and thereby attach an old model filename to
    a different collection of tasks.
    """
    output = Path(output)
    old = (read_csv(output, parse_dates=["cutoff"]) if preserve_existing and output.exists()
           else pd.DataFrame())
    summary_path = output.with_name("revroute_pool_summary.csv")
    old_summary = (read_csv(summary_path, parse_dates=["cutoff"])
                   if preserve_existing and summary_path.exists() else pd.DataFrame())
    retained_cutoffs = set(pd.to_datetime(old.get("cutoff", pd.Series(dtype="datetime64[ns]"))))
    rows, summaries = [], []
    for position, item in enumerate(schedule.itertuples(index=False), start=1):
        cutoff = pd.Timestamp(item.test_date)
        fold_id = int(getattr(item, "fold_id", position))
        if cutoff in retained_cutoffs:
            continue
        training = rolling_completed_training(prepared, cutoff, int(item.training_days))
        # Trajectories and the stage term use the transformed releases Z_itl on
        # the same smoothing level as the target Z*_it (Eqs. 3.5-3.6).
        built = sparse_task_pooling(
            training, int(item.target_lag), target_column=str(item.target_column),
            value_column=provisional_column(item.target_column),
            train_genuine_events_only=True)
        if built is None:
            continue
        geometry, labels, threshold, _, _ = built
        task_frame = pd.DataFrame(geometry.tasks, columns=["geo_value", "lag"])
        task_frame["pool"] = labels
        pool_sizes = task_frame.groupby("pool", observed=True).size()
        represented_tasks, pool_count = len(task_frame), int(pool_sizes.size)
        proportions = pool_sizes.to_numpy(float) / represented_tasks
        pool_shape = task_frame.groupby("pool", observed=True).agg(
            locations=("geo_value", "nunique"), lags=("lag", "nunique"), tasks=("lag", "size"))
        summaries.append({
            "fold": fold_id, "cutoff": cutoff, "represented_tasks": represented_tasks,
            "pool_count": pool_count,
            "effective_pool_count": float(1.0 / (proportions ** 2).sum()),
            "pools_multi_location_pct": float(100 * pool_shape.locations.gt(1).mean()),
            "pools_multi_lag_pct": float(100 * pool_shape.lags.gt(1).mean()),
            "pools_singleton_pct": float(100 * pool_shape.tasks.eq(1).mean()),
            "tasks_multi_location_pct": float(100 * pool_shape.loc[pool_shape.locations.gt(1), "tasks"].sum() / represented_tasks),
            "tasks_multi_lag_pct": float(100 * pool_shape.loc[pool_shape.lags.gt(1), "tasks"].sum() / represented_tasks),
            "tasks_singleton_pct": float(100 * pool_shape.loc[pool_shape.tasks.eq(1), "tasks"].sum() / represented_tasks),
            "threshold": float(threshold),
            "unrepresented_task_policy": "nearest lag in same location; nearest lag globally for new location",
            "small_pool_policy": "if <10 rows, expand to matching test lags; if still <10, use all fold rows with available target values",
            "fallback_is_engineering_extension": True})
        for (geo, lag), label in zip(geometry.tasks, labels):
            rows.append({"fold": fold_id, "cutoff": cutoff, "geo_value": str(geo),
                         "lag": int(lag), "pool": int(label),
                         "threshold": float(threshold)})
    frame = pd.DataFrame(rows)
    if not old.empty:
        frame = pd.concat([old, frame], ignore_index=True)
    if frame.empty:
        raise ValueError("RevRoute could not learn any location-lag task pools")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["fold", "geo_value", "lag"]).to_csv(output, index=False)
    summary_frame = pd.DataFrame(summaries)
    if not old_summary.empty:
        summary_frame = pd.concat([old_summary, summary_frame], ignore_index=True)
    summary_frame.sort_values("fold").to_csv(summary_path, index=False)
    return output


__all__ = ["learn_task_pools"]
