"""Revision-pattern matching with a routed empirical distribution."""
from __future__ import annotations

import numpy as np

ROUTED_TAUS = np.array([.01, .025, .1, .25, .5, .75, .9, .975, .99])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Lower weighted median, robust to invalid or zero-weight donor rows."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.nan
    order = np.argsort(values[keep], kind="stable")
    y, w = values[keep][order], weights[keep][order]
    return float(y[np.searchsorted(np.cumsum(w), w.sum() / 2, side="left")])


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, taus: np.ndarray = ROUTED_TAUS) -> np.ndarray:
    """Weighted empirical quantiles of observed donor responses.

    The smallest response whose cumulative normalized donor weight reaches
    the requested probability tau. Deliberately does not interpolate donor
    responses.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    taus = np.asarray(taus, dtype=float)
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("weighted quantiles must lie strictly in (0, 1)")
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.full(len(taus), np.nan)
    order = np.argsort(values[keep], kind="stable")
    y, w = values[keep][order], weights[keep][order]
    cumulative = np.cumsum(w)
    return y[np.searchsorted(cumulative, taus * cumulative[-1], side="left")]


def _weighted_quantiles_sorted(values: np.ndarray, weights: np.ndarray, order: np.ndarray,
                               taus: np.ndarray) -> np.ndarray:
    """Weighted quantiles using the cutoff-cached donor-response ordering."""
    values, weights, taus = np.asarray(values, float), np.asarray(weights, float), np.asarray(taus, float)
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("weighted quantiles must lie strictly in (0, 1)")
    if not len(order):
        return np.full(len(taus), np.nan)
    y, w = values[order], weights[order]
    keep = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return np.full(len(taus), np.nan)
    y, w = y[keep], w[keep]
    cumulative = np.cumsum(w)
    return y[np.searchsorted(cumulative, taus * cumulative[-1], side="left")]


def _weighted_quantiles_presorted(sorted_values: np.ndarray, weights: np.ndarray,
                                  order: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """As ``_weighted_quantiles_sorted`` but with the donor responses already
    gathered into sorted order once per fold, so only the per-case weights are
    permuted. Identical output; one fewer full-length gather per case."""
    if not len(order):
        return np.full(len(taus), np.nan)
    w = weights[order]
    keep = np.isfinite(sorted_values) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return np.full(len(taus), np.nan)
    y, w = sorted_values[keep], w[keep]
    cumulative = np.cumsum(w)
    return y[np.searchsorted(cumulative, np.asarray(taus, float) * cumulative[-1], side="left")]


def _weighted_median_sorted(values: np.ndarray, weights: np.ndarray, order: np.ndarray) -> float:
    """Weighted median using the cutoff-cached ordering of real responses."""
    if not len(order):
        return np.nan
    y, w = np.asarray(values, dtype=float)[order], np.asarray(weights, dtype=float)[order]
    keep = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return np.nan
    y, w = y[keep], w[keep]
    return float(y[np.searchsorted(np.cumsum(w), w.sum() / 2, side="left")])
