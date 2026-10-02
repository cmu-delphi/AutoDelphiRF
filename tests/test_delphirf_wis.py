import numpy as np
import pandas as pd
from types import SimpleNamespace

from autodelphirf.metrics import wis
from autodelphirf.pipeline import _method_rows, standardize_predictions
from autodelphirf.red import ROUTED_TAUS


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


def test_public_predictions_are_inverse_transformed_to_raw_scale():
    wide = pd.DataFrame({
        "fold": [1], "cutoff": pd.to_datetime(["2024-01-01"]), "geo_value": ["x"],
        "reference_date": pd.to_datetime(["2023-12-31"]),
        "report_date": pd.to_datetime(["2024-01-01"]),
        "target_date": pd.to_datetime(["2024-01-07"]), "lag": [1],
        "truth": [np.log1p(20.0)], "Null": [np.log1p(10.0)]})
    working_quantiles = np.log1p(np.arange(11.0, 20.0))[None, :]
    result = _method_rows(
        wide, "delphirf", np.array([np.log1p(15.0)]), working_quantiles,
        SimpleNamespace(name="example", value_transform="log1p"))
    np.testing.assert_allclose(result.loc[0, "prediction"], 15.0)
    np.testing.assert_allclose(result.loc[0, [f"q{t:g}" for t in ROUTED_TAUS]].astype(float),
                               np.arange(11.0, 20.0))
    assert result.loc[0, "prediction_scale"] == "raw"
    np.testing.assert_allclose(result.loc[0, "wis"], result.loc[0, "wis_raw"])


def test_null_is_a_degenerate_distribution_with_raw_wis_equal_to_raw_ae():
    wide = pd.DataFrame({
        "fold": [1], "cutoff": pd.to_datetime(["2024-01-01"]), "geo_value": ["x"],
        "reference_date": pd.to_datetime(["2023-12-31"]),
        "report_date": pd.to_datetime(["2024-01-01"]),
        "target_date": pd.to_datetime(["2024-01-07"]), "lag": [1],
        "truth": [np.log1p(20.0)], "Null": [np.log1p(10.0)]})
    config = SimpleNamespace(
        name="example", value_transform="log1p", prediction_layers=("baseline_null",),
        comparator_columns={}, comparator_methods={}, comparator_quantile_template=None)
    result = standardize_predictions(wide, config)
    assert result.loc[0, "has_distribution"]
    np.testing.assert_allclose(result.loc[0, "wis"], 10.0)
    np.testing.assert_allclose(result.loc[0, "wis"], result.loc[0, "absolute_error"])


def test_unavailable_method_case_is_retained_without_scores():
    wide = pd.DataFrame({
        "fold": [1, 1], "cutoff": pd.to_datetime(["2024-01-01"] * 2),
        "geo_value": ["x", "y"],
        "reference_date": pd.to_datetime(["2023-12-31"] * 2),
        "report_date": pd.to_datetime(["2024-01-01"] * 2),
        "target_date": pd.to_datetime(["2024-01-07"] * 2), "lag": [1, 1],
        "truth": np.log1p([20.0, 30.0]), "Null": np.log1p([10.0, 10.0]),
        "delphirf": [np.log1p(15.0), np.nan]})
    for tau in ROUTED_TAUS:
        wide[f"delphirf_tau{tau:g}"] = [np.log1p(15.0), np.nan]
    config = SimpleNamespace(
        name="example", value_transform="log1p", prediction_layers=("delphirf",),
        comparator_columns={}, comparator_methods={"delphirf": "RevRoute DelphiRF"},
        comparator_quantile_template="q{tau:g}")
    result = standardize_predictions(wide, config)
    assert len(result) == 2
    assert result.model_available.tolist() == [True, False]
    assert np.isnan(result.loc[1, "prediction"])
    assert np.isnan(result.loc[1, "wis"])
