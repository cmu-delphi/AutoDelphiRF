"""Evaluation metrics using DelphiRF's exact scoring definition."""
from __future__ import annotations

import numpy as np

from .red import ROUTED_TAUS


def wis(q: np.ndarray, truth: np.ndarray, taus: np.ndarray = ROUTED_TAUS) -> np.ndarray:
    """DelphiRF weighted interval score: twice the mean pinball loss.

    This is a direct vectorization of ``DelphiRF::weighted_interval_score``.
    DelphiRF calls the prediction-minus-observation values ``residuals`` and
    evaluates ``mean(2 * max(tau * -residual, (1-tau) * residual))``.
    """
    q = np.asarray(q, dtype=float)
    truth = np.asarray(truth, dtype=float)
    taus = np.asarray(taus, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(taus):
        raise ValueError("q must have one column per quantile level")
    error = q - truth[:, None]
    return np.mean(2 * np.maximum(taus * (-error), (1 - taus) * error), axis=1)
