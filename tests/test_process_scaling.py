"""M9: process features are scaled by MAD, then IQR, then range; constants drop out."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autodelphirf.revroute.pooling.geometry import _robust_scale, build_task_geometry


def test_mad_is_used_when_positive():
    centre, scale = _robust_scale([1., 2., 3., 4., 10.])
    assert centre == 3. and scale == 1.


def test_iqr_is_used_when_the_mad_is_zero():
    values = [0., 0., 0., 0., 0., 0., 1., 1.]          # MAD 0, IQR 0.25
    centre, scale = _robust_scale(values)
    assert centre == 0. and scale == pytest.approx(np.quantile(values, .75) - np.quantile(values, .25))


def test_range_is_used_when_mad_and_iqr_are_zero():
    centre, scale = _robust_scale([7., 7., 7., 7., 7., 7., 7., 14.])
    assert centre == 7. and scale == 7.


def test_a_constant_feature_has_no_scale():
    assert _robust_scale([1., 1., 1.])[1] is None


def test_a_feature_constant_across_tasks_is_left_out_of_the_process_distance():
    rows = []
    for k, geo in enumerate(("aa", "bb", "cc")):
        for day, reference in enumerate(pd.date_range("2022-01-01", periods=30)):
            for lag in range(11):                         # a release every day
                value = np.log1p((50 + day) * (1 + (0.1 + 0.05 * k) * min(lag, 6)))
                rows.append({"geo_value": geo, "reference_date": reference,
                             "report_date": reference + pd.Timedelta(days=lag), "lag": lag,
                             "log_value_7dav": value, "genuine_event": True})
    training = pd.DataFrame(rows)
    final = training[training.lag == 10].set_index(["geo_value", "reference_date"]).log_value_7dav
    training["log_value_target_7dav"] = final.reindex(
        pd.MultiIndex.from_frame(training[["geo_value", "reference_date"]])).to_numpy()
    geometry = build_task_geometry(training, 10, distance_backend="prototypes")
    # The interval between releases is one day for every task: constant, so
    # dropped rather than entering the distance in raw days.
    assert geometry.profiles.shape[1] < 5
    assert np.isfinite(geometry.profiles).all()
