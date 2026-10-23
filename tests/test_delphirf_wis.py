import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from autodelphirf.post_assessment.scoring import wis
from autodelphirf.workflow.run import _method_rows, standardize_predictions
from autodelphirf.comparators.red import ROUTED_TAUS

from test_end_to_end import requires_delphirf


def test_wis_matches_delphirf_twice_mean_pinball_definition():
    taus = np.array([0.1, 0.5, 0.9])
    predictions = np.array([[8.0, 10.0, 14.0], [4.0, 6.0, 7.0]])
    truth = np.array([12.0, 5.0])
    residuals = predictions - truth[:, None]
    expected = np.mean(
        2 * np.maximum(taus * -residuals, (1 - taus) * residuals), axis=1
    )
    np.testing.assert_allclose(wis(predictions, truth, taus), expected)


def test_delphirf_median_only_wis_is_absolute_error():
    predictions = np.array([[8.0], [12.0], [10.0]])
    truth = np.array([10.0, 10.0, 10.0])
    np.testing.assert_allclose(wis(predictions, truth, np.array([0.5])), [2.0, 2.0, 0.0])


def test_generic_method_row_builder_can_report_both_scales():
    wide = pd.DataFrame({
        "fold": [1], "cutoff": pd.to_datetime(["2024-01-01"]), "geo_value": ["x"],
        "reference_date": pd.to_datetime(["2023-12-31"]),
        "report_date": pd.to_datetime(["2024-01-01"]),
        "target_date": pd.to_datetime(["2024-01-07"]), "lag": [1],
        "truth": [np.log1p(20.0)], "Null": [np.log1p(10.0)]})
    working_quantiles = np.log1p(np.arange(11.0, 20.0))[None, :]
    result = _method_rows(
        wide, "delphirf", np.array([np.log1p(15.0)]), working_quantiles,
        SimpleNamespace(name="example", working_scale="log_value"))
    np.testing.assert_allclose(result.loc[0, "prediction"], 15.0)
    np.testing.assert_allclose(result.loc[0, [f"q{t:g}" for t in ROUTED_TAUS]].astype(float),
                               np.arange(11.0, 20.0))
    assert result.loc[0, "prediction_scale"] == "reporting"
    assert result.loc[0, "working_scale"] == "log_value"
    # DelphiRF's score names: wis_reporting and wis_working, and nothing else.
    assert "wis" not in result and "wis_raw" not in result
    assert {"wis_reporting", "wis_working"} <= set(result.columns)
    np.testing.assert_allclose(result.loc[0, "relative_error_pct"], 100 * abs(16 / 21 - 1))
    relative_quantiles = 100 * ((np.arange(11.0, 20.0) + 1) / 21 - 1)
    np.testing.assert_allclose(
        result.loc[0, "relative_wis_pct"],
        wis(relative_quantiles[None, :], np.array([0.0]))[0])


def test_delphirf_paired_scales_are_consumed_without_a_second_transform():
    wide = pd.DataFrame({
        "fold": [1], "cutoff": pd.to_datetime(["2024-01-01"]), "geo_value": ["x"],
        "reference_date": pd.to_datetime(["2023-12-31"]),
        "report_date": pd.to_datetime(["2024-01-01"]),
        "target_date": pd.to_datetime(["2024-01-07"]), "lag": [1],
        "truth": [np.log1p(20.0)], "Null": [np.log1p(10.0)],
        "delphirf": [15.0], "delphirf_working": [np.log1p(15.0)]})
    for index, tau in enumerate(ROUTED_TAUS, start=11):
        wide[f"delphirf_tau{tau:g}"] = [float(index)]
        wide[f"delphirf_tau{tau:g}_working"] = [np.log1p(float(index))]
    config = SimpleNamespace(
        name="example", working_scale="log_value", prediction_layers=("delphirf",),
        comparator_columns={}, comparator_methods={"delphirf": "RevRoute DelphiRF"},
        comparator_quantile_template="q{tau:g}")

    result = standardize_predictions(wide, config)

    assert result.loc[0, "prediction"] == 15.0
    np.testing.assert_allclose(result.loc[0, "prediction_working"], np.log1p(15.0))
    np.testing.assert_allclose(
        result.loc[0, [f"q{tau:g}" for tau in ROUTED_TAUS]].astype(float),
        np.arange(11.0, 20.0))


def test_delphirf_requires_model_supplied_working_and_reporting_scales():
    wide = pd.DataFrame({
        "fold": [1], "cutoff": pd.to_datetime(["2024-01-01"]),
        "geo_value": ["x"], "reference_date": pd.to_datetime(["2023-12-31"]),
        "report_date": pd.to_datetime(["2024-01-01"]),
        "target_date": pd.to_datetime(["2024-01-07"]), "lag": [1],
        "truth": [np.log1p(20.0)], "Null": [np.log1p(10.0)],
        "delphirf": [15.0]})
    for tau in ROUTED_TAUS:
        wide[f"delphirf_tau{tau:g}"] = [15.0]
    config = SimpleNamespace(
        name="example", working_scale="log_value", prediction_layers=("delphirf",),
        comparator_columns={}, comparator_methods={"delphirf": "RevRoute DelphiRF"},
        comparator_quantile_template="q{tau:g}")

    with pytest.raises(ValueError, match="no paired working-scale prediction"):
        standardize_predictions(wide, config)


def test_null_is_a_degenerate_distribution_with_raw_wis_equal_to_raw_ae():
    wide = pd.DataFrame({
        "fold": [1], "cutoff": pd.to_datetime(["2024-01-01"]), "geo_value": ["x"],
        "reference_date": pd.to_datetime(["2023-12-31"]),
        "report_date": pd.to_datetime(["2024-01-01"]),
        "target_date": pd.to_datetime(["2024-01-07"]), "lag": [1],
        "truth": [np.log1p(20.0)], "Null": [np.log1p(10.0)]})
    config = SimpleNamespace(
        name="example", working_scale="log_value", prediction_layers=("baseline_null",),
        comparator_columns={}, comparator_methods={}, comparator_quantile_template=None)
    result = standardize_predictions(wide, config)
    assert result.loc[0, "has_distribution"]
    np.testing.assert_allclose(result.loc[0, "wis_reporting"], 10.0)
    np.testing.assert_allclose(result.loc[0, "wis_reporting"], result.loc[0, "absolute_error"])


def test_unavailable_method_case_is_retained_without_scores():
    wide = pd.DataFrame({
        "fold": [1, 1], "cutoff": pd.to_datetime(["2024-01-01"] * 2),
        "geo_value": ["x", "y"],
        "reference_date": pd.to_datetime(["2023-12-31"] * 2),
        "report_date": pd.to_datetime(["2024-01-01"] * 2),
        "target_date": pd.to_datetime(["2024-01-07"] * 2), "lag": [1, 1],
        "truth": np.log1p([20.0, 30.0]), "Null": np.log1p([10.0, 10.0]),
        "delphirf": [15.0, np.nan],
        "delphirf_working": [np.log1p(15.0), np.nan]})
    for tau in ROUTED_TAUS:
        wide[f"delphirf_tau{tau:g}"] = [15.0, np.nan]
        wide[f"delphirf_tau{tau:g}_working"] = [np.log1p(15.0), np.nan]
    config = SimpleNamespace(
        name="example", working_scale="log_value", prediction_layers=("delphirf",),
        comparator_columns={}, comparator_methods={"delphirf": "RevRoute DelphiRF"},
        comparator_quantile_template="q{tau:g}")
    result = standardize_predictions(wide, config)
    assert len(result) == 2
    assert result.model_available.tolist() == [True, False]
    assert np.isnan(result.loc[1, "prediction"])
    assert np.isnan(result.loc[1, "wis_reporting"])


@requires_delphirf
def test_scores_and_scales_match_delphirf_exactly(tmp_path):
    """AutoDelphiRF's Python scoring is DelphiRF's, computed by DelphiRF itself."""
    import json
    import shutil
    import subprocess

    from autodelphirf.core.defaults import to_reporting_scale

    rng = np.random.default_rng(3)
    quantiles = np.sort(rng.normal(5, 1, size=(40, len(ROUTED_TAUS))), axis=1)
    truth = rng.normal(5, 1, size=40)
    working = rng.normal(0, 2, size=40)
    payload = tmp_path / "inputs.json"
    payload.write_text(json.dumps({"q": quantiles.tolist(), "truth": truth.tolist(),
                                   "taus": list(map(float, ROUTED_TAUS)),
                                   "working": working.tolist()}))
    script = f"""
      suppressMessages(library(DelphiRF)); x <- jsonlite::fromJSON("{payload}")
      df <- as.data.frame(x$q); names(df) <- paste0("predicted_tau", x$taus); df$y <- x$truth
      scale <- get("to_reporting_scale", asNamespace("DelphiRF"))
      cat(jsonlite::toJSON(list(
        wis = DelphiRF::evaluate(df, x$taus, "y", column = "w")$w,
        log_value = scale(x$working, "log_value"),
        log_ratio = scale(x$working, "log_ratio")), digits = NA))
    """
    executable = shutil.which("Rscript")
    result = subprocess.run([executable, "-e", script], capture_output=True, text=True, check=True)
    delphirf = json.loads(result.stdout)
    np.testing.assert_allclose(wis(quantiles, truth), delphirf["wis"], rtol=1e-12)
    for scale in ("log_value", "log_ratio"):
        np.testing.assert_allclose(to_reporting_scale(working, scale), delphirf[scale], rtol=1e-12)
