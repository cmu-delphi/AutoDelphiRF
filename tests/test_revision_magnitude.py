"""M4: revision magnitudes are measured from the initial report (l_min, Ch. 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from autodelphirf.core.io import load_prepared_triangles
from autodelphirf.pre_diagnosis.preparation import build_spec, run_r_bridge

from test_end_to_end import requires_delphirf


@requires_delphirf
def test_a_stream_first_reported_at_lag_1_measures_revisions_from_lag_1(tmp_path):
    rows = []
    for k, reference in enumerate(pd.date_range("2022-01-01", periods=120)):
        for lag in range(1, 40):                       # nothing is released on lag 0
            rows.append({"geo_value": "aa", "reference_date": reference,
                         "report_date": reference + pd.Timedelta(days=lag),
                         "value": 100 + k + 3 * min(lag, 8)})
    pd.DataFrame(rows).to_csv(tmp_path / "late.csv", index=False)
    spec = build_spec("late", tmp_path / "late.csv", tmp_path / "late", target_lag=14,
                      smoothed=True, lag_terms=(1, 7), triangle_format="csv")
    run_r_bridge(spec)
    triangle = load_prepared_triangles(tmp_path / "late")
    triangle["reference_date"] = pd.to_datetime(triangle.reference_date)
    row = triangle[(triangle.reference_date == "2022-02-15") & (triangle.lag == 3)].iloc[0]
    # Reference date 02-08 (k = 38), now at lag 10, against its lag-1 report.
    assert row.value_7dav_start_lag7 == 100 + 38 + 3
    assert np.isclose(row.log_delta_value_7dav_lag7,
                      np.log(100 + 38 + 24 + 1) - np.log(100 + 38 + 3 + 1))
    # The feature is a revision magnitude, no longer a copy of the level feature.
    assert not np.allclose(triangle.log_delta_value_7dav_lag7.dropna(),
                           triangle.log_value_7dav_lag7[triangle.log_delta_value_7dav_lag7.notna()])
