"""V1/V2 prospective-validation tests.

The validation layer is pure post-processing: it must read already-issued
forecasts and matured outcomes and never re-predict, re-fit, or retune an
alert threshold. These tests pin the arithmetic of each summary and the
direction of each comparison.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from autodelphirf.validation import (alert_stratification, flat_line_comparison,
                                        operational_alert_summary, pairwise_loss_comparison,
                                        reliability_coverage, risk_discrimination)


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
                         "lag": lags[index], "absolute_error": value, "wis": value / 2})
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

def reliability_frame(n=100, seed=0):
    rng = np.random.default_rng(seed)
    realized = rng.exponential(1., n)
    return pd.DataFrame({
        "method": "rr_delphirf_hard", "fold": 1, "lag": rng.integers(0, 30, n),
        "q_error": np.full(n, 2.5), "q_regret": np.full(n, .5),
        "risk_score": rng.random(n), "realized_ae": realized,
        "realized_null_ae": realized + rng.normal(0, .1, n),
        "realized_regret": rng.normal(0, .5, n),
        "reliability_level": rng.choice(["supported", "caution", "high risk"], n),
    }).assign(qerr_covered=lambda d: d.realized_ae <= d.q_error,
              qreg_covered=lambda d: d.realized_regret <= d.q_regret)


def test_reliability_coverage_matches_the_indicator_means():
    frame = reliability_frame()
    out = reliability_coverage(frame, eta=.9)
    row = out[out.subset.eq("overall")].iloc[0]
    assert row.coverage_error_quantity == pytest.approx(frame.qerr_covered.mean())
    assert row.coverage_error_gap == pytest.approx(frame.qerr_covered.mean() - .9)
    assert row.realized_harm_frequency == pytest.approx((frame.realized_regret > 0).mean())


def test_alert_stratification_reports_every_issued_level():
    frame = reliability_frame()
    out = alert_stratification(frame)
    overall = out[out.subset.eq("overall")]
    assert set(overall.reliability_level) == {"supported", "caution", "high risk"}
    assert overall.n.sum() == len(frame)
    for _, row in overall.iterrows():
        group = frame[frame.reliability_level.eq(row.reliability_level)]
        assert row.mean_ae == pytest.approx(group.realized_ae.mean())
        assert row.harm_probability == pytest.approx((group.realized_regret > 0).mean())


def test_qerr_discrimination_recovers_a_planted_ranking():
    """Discrimination is measured on q_err, the primary difficulty axis.

    The three-axis spec forms no composite, so ``risk_score`` no longer
    exists; a table keyed on it emitted nothing at all.
    """
    n = 200
    rng = np.random.default_rng(1)
    difficulty = rng.random(n)
    frame = pd.DataFrame({
        "method": "m", "lag": 3,
        "q_error": difficulty, "q_regret": difficulty,
        "realized_ae": difficulty * 10 + rng.normal(0, .2, n),
        "realized_regret": difficulty - .5,
    })
    out = risk_discrimination(frame, bins=4)
    rows = out[out.subset.eq("overall") & out.signal.eq("q_error")].sort_values("bin")
    assert len(rows) == 4
    assert rows.spearman_signal_vs_ae.iloc[0] > .9
    # Realized error must increase across the ordered bins.
    assert list(rows.mean_ae) == sorted(rows.mean_ae)
    # risk_score must not resurface as a signal.
    assert "risk_score" not in set(out.signal)


def test_operational_alert_summary_confusion_arithmetic():
    frame = pd.DataFrame({
        "method": "m", "lag": 1,
        "reliability_level": ["high risk", "caution", "supported", "supported"],
        "realized_regret": [1., -1., 1., -1.],      # bad, good, bad, good
        "realized_ae": [1., 1., 1., 1.], "q_error": [1.] * 4, "q_regret": [1.] * 4,
    })
    row = operational_alert_summary(frame).iloc[0]
    # alert = {high risk, caution} -> rows 0,1 ; bad = regret > 0 -> rows 0,2
    assert (row.true_positive, row.false_positive) == (1, 1)
    assert (row.false_negative, row.true_negative) == (1, 1)
    assert row.sensitivity == pytest.approx(.5)
    assert row.specificity == pytest.approx(.5)
    assert row.positive_predictive_value == pytest.approx(.5)
    assert row.false_alert_rate == pytest.approx(.5)
    assert row.bad_forecast_rate == pytest.approx(.5)


def test_validation_never_refits_or_retunes():
    """No estimator, solver or threshold-tuning symbol may appear in the layer."""
    from autodelphirf import validation
    source = pathlib.Path(validation.__file__).read_text()
    for forbidden in ("QuantileRegressor", "fit_rr_delphirf", "predict_rr_delphirf",
                      "linkage(", "revision_forecast", ".fit("):
        assert forbidden not in source, f"validation must not reference {forbidden}"


# ---------------------------------------------------------------------------
# Three-axis post-prediction (RRDelphiRF_post_prediction_three_axis.tex)
# ---------------------------------------------------------------------------



def test_opposed_objective_diagnostic_records_the_sign():
    """A negative q_err-vs-regret association is the expected structure."""
    from autodelphirf.validation import opposed_objective_diagnostic
    rng = np.random.default_rng(5)
    n = 300
    difficulty = rng.random(n)
    frame = pd.DataFrame({
        "method": "m", "fold": 1, "lag": 3,
        "q_error": difficulty,
        "realized_ae": difficulty * 5 + rng.normal(0, .2, n),   # tracks difficulty
        "realized_regret": -difficulty + rng.normal(0, .1, n),  # OPPOSED
        "f_harm": rng.random(n),
    })
    row = opposed_objective_diagnostic(frame).iloc[0]
    assert row.spearman_qerr_vs_ae > .9          # difficulty ranks error
    assert row.spearman_qerr_vs_regret < -.8     # and anti-ranks regret
    assert "separate" in row.interpretation


def test_v4_keeps_the_two_joint_displays_apart():
    from autodelphirf.validation import three_axis_heatmaps
    rng = np.random.default_rng(7)
    n = 400
    frame = pd.DataFrame({
        "method": "m", "fold": 1, "lag": 3,
        "q_error": rng.random(n), "f_harm": rng.random(n),
        "realized_ae": rng.exponential(1., n),
        "realized_regret": rng.normal(0, 1., n),
        "process_shift_score": rng.random(n) * 5,
    })
    out = three_axis_heatmaps(frame, long=None, bins=3)
    assert set(out) == {"difficulty_vs_process", "harm_vs_process"}
    # Difficulty x process reports ERROR; harm x process reports HARM RATE.
    assert "mean_ae" in out["difficulty_vs_process"].columns
    assert "observed_harm_frequency" in out["harm_vs_process"].columns
    assert "observed_harm_frequency" not in out["difficulty_vs_process"].columns


def test_process_monitor_nondegeneracy_calls_a_flat_score_a_failure():
    from autodelphirf.validation import process_monitor_nondegeneracy
    flat = pd.DataFrame({"method": "m", "lag": 1, "process_shift_score": np.zeros(50),
                         "process_state": "stable"})
    assert bool(process_monitor_nondegeneracy(flat).monitor_degenerate.iloc[0])
    assert "MONITOR FAILURE" in process_monitor_nondegeneracy(flat).verdict.iloc[0]
    live = pd.DataFrame({"method": "m", "lag": 1,
                         "process_shift_score": np.linspace(0, 5, 50),
                         "process_state": "stable"})
    assert not bool(process_monitor_nondegeneracy(live).monitor_degenerate.iloc[0])


# ---------------------------------------------------------------------------
# Final model-free spec: order statistics, descriptive f_harm, separate C_harm
# ---------------------------------------------------------------------------

def test_qerr_is_a_realized_order_statistic_not_an_interpolated_quantile():
    """q_eta^err = E_(k_eta(n)) with k_eta(n) = min{n, ceil((n+1) eta)}."""
    from autodelphirf.calibration import upper_order_statistic
    values = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    # n=10, eta=0.9 -> k = min(10, ceil(11*0.9)=10) = 10 -> the maximum.
    assert upper_order_statistic(values, .9) == 10.
    # Every returned value must be one that actually occurred.
    for eta in (.5, .75, .9, .95):
        assert upper_order_statistic(values, eta) in set(values.tolist())
    # Tiny support stays defined and never extrapolates past the maximum.
    assert upper_order_statistic(np.array([4., 9.]), .9) == 9.
    assert upper_order_statistic(np.array([7.]), .9) == 7.
    assert np.isnan(upper_order_statistic(np.array([]), .9))
    # It differs from the interpolated quantile, which is the point.
    assert upper_order_statistic(values, .9) != float(np.quantile(values, .9))


def test_wilson_interval_stays_in_unit_range_at_the_boundaries():
    """Wilson rather than Wald: k=0 and k=n occur routinely at small support."""
    from autodelphirf.calibration import wilson_interval
    for k, n in ((0, 12), (12, 12), (3, 40), (1, 2)):
        low, high = wilson_interval(k, n)
        assert 0. <= low <= high <= 1., (k, n, low, high)
    # A wider interval for less support, at the same proportion.
    narrow = wilson_interval(20, 200); wide = wilson_interval(2, 20)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
    assert np.isnan(wilson_interval(0, 0)[0])


def test_profile_reports_descriptive_f_harm_with_counts_and_no_probability():
    """R_rel = (q_eta^err, f_harm, M_harm) and f_harm is NOT a probability."""
    from autodelphirf.calibration import forecast_quality_summary
    rng = np.random.default_rng(3)
    history = pd.DataFrame({"rr_ae": rng.gamma(2., 1., 200),
                            "null_ae": rng.gamma(2., 1., 200)})
    out = forecast_quality_summary(history, min_history=20)
    for gone in ("q_regret", "p_harm", "harm_probability_status"):
        assert gone not in out
    assert np.isfinite(out["q_error"]) and out["n_err"] == 200
    # The counts that make the frequency interpretable must be present.
    assert out["k_harm"] + 0 == out["k_harm"] and out["n_harmfreq"] == 200
    assert out["f_harm"] == out["k_harm"] / out["n_harmfreq"]
    assert out["f_harm_wilson_low"] <= out["f_harm"] <= out["f_harm_wilson_high"]
    # Severity is NOT computed from the total-support set any more.
    assert "m_harm" not in out


def test_severity_uses_its_own_positive_harm_support():
    """M_harm = Delta^+_(k_eta(m)) over C_harm, with m reported."""
    from autodelphirf.calibration import harm_severity_summary
    # 30 harmful cases of increasing severity plus 200 helpful ones.
    harmful = np.arange(1., 31.)
    history = pd.DataFrame({"rr_ae": np.concatenate([harmful + 10., np.full(200, 1.)]),
                            "null_ae": np.concatenate([np.full(30, 10.), np.full(200, 5.)])})
    out = harm_severity_summary(history, eta=.9, min_harm_support=10)
    assert out["harm_severity_status"] == "severity available"
    assert out["harm_positive_n"] == 30
    # k_eta(30) = min(30, ceil(31*0.9)=28) = 28 -> the 28th smallest positive regret.
    assert out["m_harm"] == float(np.sort(harmful)[27])
    # Below the minimum positive support it reports, rather than inventing.
    thin = pd.DataFrame({"rr_ae": np.concatenate([[20., 21.], np.full(100, 1.)]),
                         "null_ae": np.concatenate([[10., 10.], np.full(100, 5.)])})
    out = harm_severity_summary(thin, eta=.9, min_harm_support=10)
    assert np.isnan(out["m_harm"])
    assert out["harm_severity_status"] == "insufficient harm-severity history"
    assert out["harm_positive_n"] == 2


def test_c_harm_expands_further_than_c_err_when_harm_is_sparse():
    """The two selections differ exactly when harmful cases are too few."""
    from autodelphirf.calibration import resolve_strata
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
