"""V1 prospective-validation tests.

The validation layer is pure post-processing: it must read already-issued
forecasts and target values and never re-predict, re-fit, or retune anything. These tests pin the arithmetic of each summary and the
direction of each comparison.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from autodelphirf.post_assessment.validation import flat_line_comparison, pairwise_loss_comparison


def long_frame(losses: dict, lags=None):
    """A standardized long prediction table with one row per (case, method)."""
    n = len(next(iter(losses.values())))
    lags = lags if lags is not None else [1] * n
    rows = []
    for method, values in losses.items():
        for index, value in enumerate(values):
            rows.append({"method": method, "fold": 1, "geo_value": "a",
                         "reference_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                         "report_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=index),
                         "lag": lags[index], "absolute_error": value, "wis_reporting": value / 2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# V1
# ---------------------------------------------------------------------------

def test_flat_line_counts_wins_losses_ties_and_total_saved():
    long = long_frame({"baseline_null": [1., 2., 3., 4.],
                       "rr_delphirf_hard": [0.5, 2., 5., 1.]})
    out = flat_line_comparison(long)
    row = out[out.subset.eq("overall") & out.method.eq("rr_delphirf_hard")].iloc[0]
    assert row.n == 4
    assert row.better_n == 2 and row.worse_n == 1 and row.tied_n == 1
    assert row.better_fraction == pytest.approx(.5)
    # saved = sum(baseline - method) = (0.5) + (0) + (-2) + (3) = 1.5
    assert row.total_loss_saved == pytest.approx(1.5)
    assert row.mean_loss_saved == pytest.approx(1.5 / 4)


def test_flat_line_respects_the_early_lag_subset():
    long = long_frame({"baseline_null": [1., 1., 1., 1.],
                       "m": [0., 0., 5., 5.]}, lags=[1, 2, 40, 50])
    out = flat_line_comparison(long, early_lag_max=14)
    early = out[out.subset.eq("early")].iloc[0]
    overall = out[out.subset.eq("overall")].iloc[0]
    assert early.n == 2 and early.better_n == 2
    assert overall.n == 4 and overall.better_n == 2
    assert early.total_loss_saved == pytest.approx(2.)
    assert overall.total_loss_saved == pytest.approx(-6.)


def test_pairwise_delta_sign_and_win_probability():
    """Delta = L_a - L_b, so a negative mean favours the first method."""
    long = long_frame({"a": [1., 1., 1.], "b": [2., 2., 0.]})
    out = pairwise_loss_comparison(long, [("a", "b")])
    row = out[out.subset.eq("overall")].iloc[0]
    assert row.comparison == "a - b"
    assert row.mean_delta == pytest.approx((-1 - 1 + 1) / 3)
    assert row.win_probability == pytest.approx(2 / 3)
    assert row.loss_probability == pytest.approx(1 / 3)
    assert row.tie_probability == pytest.approx(0.)
    assert row.total_loss_saved == pytest.approx(1.)


def test_pairwise_comparison_skips_absent_methods():
    long = long_frame({"a": [1., 2.], "b": [2., 1.]})
    assert pairwise_loss_comparison(long, [("a", "missing")]).empty


# ---------------------------------------------------------------------------
# V2
# ---------------------------------------------------------------------------


def test_validation_never_refits_or_retunes():
    """No estimator, solver or threshold-tuning symbol may appear in the layer."""
    from autodelphirf.post_assessment import validation
    source = pathlib.Path(validation.__file__).read_text()
    for forbidden in ("QuantileRegressor", "fit_rr_delphirf", "predict_rr_delphirf",
                      "linkage(", "revision_forecast", ".fit("):
        assert forbidden not in source, f"validation must not reference {forbidden}"


# ---------------------------------------------------------------------------
# Post-prediction validation of the historical assessments (Sec. 3.2.5)
# ---------------------------------------------------------------------------


def test_c_harm_expands_further_than_c_err_when_harm_is_sparse():
    """The two selections differ exactly when harmful cases are too few."""
    from autodelphirf.post_assessment.reliability import resolve_strata
    # Location "a" lag 5 has plenty of rows but almost no harmful ones; the
    # wider pool does have harmful cases.
    rows = []
    for i in range(40):
        rows.append({"geo_value": "a", "lag": 5, "rr_ae": 1., "null_ae": 5.})   # helpful
    for i in range(40):
        rows.append({"geo_value": "b", "lag": 5, "rr_ae": 9., "null_ae": 5.})   # harmful
    history = pd.DataFrame(rows)
    test = pd.DataFrame({"geo_value": ["a"], "lag": [5]})
    harmful = (history.rr_ae.to_numpy() - history.null_ae.to_numpy()) > 0
    err = resolve_strata(test, history, 20, 1)
    harm = resolve_strata(test, history, 10, 1,
                          counter=lambda pos: int(harmful[pos].sum()))
    assert err[("a", 5)][1] == "location/exact-lag"      # total support is fine
    # C_harm must walk past the local level to find positive-harm support.
    assert harm[("a", 5)][1] in {"global/exact-lag", "global/adjacent-lags"}
    assert int(((harm[("a", 5)][0].rr_ae - harm[("a", 5)][0].null_ae) > 0).sum()) >= 10
