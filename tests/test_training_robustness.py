"""Low-severity training fixes: contained failures, train-only bookkeeping, pools."""
from __future__ import annotations

import pandas as pd
import pytest

from autodelphirf.core.resources import resource_path
from autodelphirf.revroute.pooling.geometry import build_task_geometry

SOURCE = resource_path("train_delphirf.R").read_text()


def fit_one_body() -> str:
    start = SOURCE.index("fit_one <- function(")
    return SOURCE[start:SOURCE.index("\n}\n", start)]


def test_no_step_of_a_fit_can_abort_the_whole_run():
    # Alignment, duplicate and quantile checks run inside a tryCatch that
    # records the failure; fit_one itself never calls stop().
    after_fit = fit_one_body().split("result <- tryCatch(", 1)[1]
    assert "stop(" not in after_fit
    assert "tryCatch(collect_predictions(" in after_fit
    assert 'stop("cannot align predictions' in SOURCE[SOURCE.index("collect_predictions <- function"):]


def test_prediction_alignment_handles_constant_response_fallback_without_guessing():
    """The bridge must mirror both DelphiRF test-row filtering paths.

    Ordinary fits require complete predictors, while DelphiRF's constant-
    response fallback only requires the basic date/lag keys.  Matching must
    also verify those keys in order; equal row counts alone are unsafe for a
    multi-location pool.
    """
    fit_body = fit_one_body()
    collect_body = SOURCE[SOURCE.index("collect_predictions <- function"):]

    assert "feature_complete = aligned_test" in fit_body
    assert "basic_complete = basic_aligned_test" in fit_body
    assert "basic_test <- as.data.frame(test)[, basic_columns, drop = FALSE]" in fit_body
    assert "same_ordered_keys <- function(candidate)" in collect_body
    assert "nrow(result) != nrow(candidate)" in collect_body
    assert "as.data.frame(result)[, key_columns, drop = FALSE]" in collect_body
    assert "as.data.frame(candidate)[, key_columns, drop = FALSE]" in collect_body
    assert "identical(result_keys, candidate_keys)" in collect_body
    assert "matched_test <- valid_candidates[[1L]]" in collect_body


def test_train_only_failures_do_not_count_as_missing_forecasts():
    assert 'kind = if (train_only) "train_only" else "prediction"' in SOURCE
    assert 'fit_status = if (prediction_failures) "partial_failure" else "complete"' in SOURCE
    assert "train_only_failure_count = train_only_failures" in SOURCE
    # A fit skipped for lack of training rows is recorded, not silent.
    assert 'paste0("fewer than 10 training rows (", nrow(train), ")")' in fit_one_body()


def test_unrepresented_tasks_do_not_add_training_rows_to_their_pool():
    assert "learned_pools <- copy(fold_pools)" in SOURCE
    assert "merge(training, learned_pools[, .(geo_value, lag, pool)]" in SOURCE
    assert "merge(testing, fold_pools[, .(geo_value, lag, pool)]" in SOURCE


def test_only_the_thesis_geometry_is_available():
    with pytest.raises(ValueError, match="task prototypes only"):
        build_task_geometry(pd.DataFrame(), 7, distance_backend="scalar")
