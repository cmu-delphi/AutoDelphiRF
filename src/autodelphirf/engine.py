"""Rolling replay for the optional Python RED model.

DelphiRF models are trained in :mod:`autodelphirf.model_training`; RevRoute
pooling is learned in :mod:`autodelphirf.revroute`.  This engine deliberately
contains neither a DelphiRF reimplementation nor residual hybrid estimators.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd

from .defaults import ROUTING_DEFAULTS
from .red import ROUTED_TAUS, _weighted_quantiles_presorted
from .weighting import (FEATURE, build_curve_routing, final_v7_weights,
                        rolling_completed_training)


def replay(cases: pd.DataFrame, prepared: pd.DataFrame, training_days: int | dict, *,
           gamma: float = ROUTING_DEFAULTS["gamma"],
           min_curve_episodes: int = ROUTING_DEFAULTS["min_curve_episodes"],
           spline_method: str = ROUTING_DEFAULTS.get("spline_method", "penalized_bspline"),
           spline_basis: int = ROUTING_DEFAULTS["spline_basis"],
           spline_penalty: float = ROUTING_DEFAULTS["spline_penalty"],
           initial_lag: int | None = None, include_red: bool = False,
           target_lag: int | None = None, **_) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay test windows, fitting RED only when explicitly requested."""
    if target_lag is None:
        raise ValueError("replay() requires the schedule's explicit target_lag")
    initial_lag = (int(prepared.lag.dropna().min()) if initial_lag is None
                   else int(initial_lag))
    should_stop = _.get("should_stop")
    issued, timings = [], []
    folds = cases[["fold", "cutoff"]].drop_duplicates().sort_values("cutoff")
    for position, item in enumerate(folds.itertuples(index=False), start=1):
        if should_stop:
            should_stop()
        began = perf_counter()
        fold, cutoff = item.fold, pd.Timestamp(item.cutoff)
        test = cases[cases.fold.eq(fold)].copy().reset_index(drop=True)
        days = int(training_days[fold]) if isinstance(training_days, dict) else int(training_days)
        training = rolling_completed_training(prepared, cutoff, days)
        fit_started = perf_counter()
        if include_red:
            route = build_curve_routing(
                training, int(target_lag), min_curve_episodes, gamma,
                spline_method=spline_method, spline_basis=spline_basis,
                spline_penalty=spline_penalty, initial_lag=initial_lag)
            rows = []
            for row_number, (_, case) in enumerate(test.iterrows()):
                if should_stop and row_number % 100 == 0:
                    should_stop()
                weights, audit = final_v7_weights(
                    route, str(case.geo_value), int(case.lag), case.get(FEATURE, np.nan))
                remaining = _weighted_quantiles_presorted(
                    route.sorted_responses, weights, route.response_sort_order, ROUTED_TAUS)
                rows.append({**case.to_dict(), **audit,
                             "red_prediction": float(case.Null) + remaining[4],
                             **{f"red_tau{tau:g}": float(case.Null) + remaining[j]
                                for j, tau in enumerate(ROUTED_TAUS)}})
            result = pd.DataFrame(rows)
        else:
            result = test
        fit_seconds = perf_counter() - fit_started
        issued.append(result)
        timings.append({"fold": fold, "cutoff": cutoff, "training_days": days,
                        "red_fit_and_prediction_seconds": fit_seconds,
                        "fold_total_seconds": perf_counter() - began})
        print(f"Evaluated retraining date {position} of {len(folds)}", flush=True)
    return (pd.concat(issued, ignore_index=True) if issued else cases.iloc[:0].copy(),
            pd.DataFrame(timings))
