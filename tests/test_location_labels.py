"""A declared geography column must name a location; blanks are dropped, labels stay text."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autodelphirf.core.io import load_prepared_triangles
from autodelphirf.pre_diagnosis.preparation import (PreparationError, build_spec,
                                                    diagnose_archive, run_r_bridge)

from conftest import archive_frame
from test_end_to_end import requires_delphirf


def archive_with_bad_labels(path):
    frame = archive_frame(days=60, max_lag=10, locations=("01", "1"))
    bad = frame.iloc[:3].copy()
    bad["geo_value"] = [np.nan, "", "   "]
    pd.concat([frame, bad], ignore_index=True).to_csv(path, index=False)
    return path


def diagnose(path, geo_col="geo_value"):
    return diagnose_archive(path, geo_col=geo_col, reference_col="reference_date",
                            report_col="report_date", value_cols=("value",), value_type="count")


def test_missing_or_blank_locations_are_dropped_and_counted(tmp_path):
    diagnosis = diagnose(archive_with_bad_labels(tmp_path / "a.csv"))
    assert diagnosis["dropped_missing_location_rows"] == 3
    assert diagnosis["n_locations"] == 2                  # "01" and "1", nothing else
    assert any("missing or blank location" in note for note in diagnosis["notes"])


def test_no_valid_location_is_an_error(tmp_path):
    frame = archive_frame(days=10, max_lag=5, locations=("aa",)).assign(geo_value="  ")
    frame.to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(PreparationError, match="no valid location on any row"):
        diagnose(tmp_path / "a.csv")


def test_single_location_mode_ignores_the_column(tmp_path):
    diagnosis = diagnose(archive_with_bad_labels(tmp_path / "a.csv"), geo_col=None)
    assert diagnosis["n_locations"] == 1
    assert diagnosis["dropped_missing_location_rows"] == 0


@requires_delphirf
def test_preparation_drops_the_same_rows(tmp_path):
    path = archive_with_bad_labels(tmp_path / "a.csv")
    frame = archive_frame(days=120, max_lag=30, locations=("01", "1"))
    bad = frame.iloc[:3].copy()
    bad["geo_value"] = [np.nan, "", "   "]
    pd.concat([frame, bad], ignore_index=True).to_csv(path, index=False)
    spec = build_spec("labels", path, tmp_path / "labels", target_lag=14, smoothed=True,
                      triangle_format="csv")
    run_r_bridge(spec)
    triangle = load_prepared_triangles(tmp_path / "labels")
    assert sorted(triangle.geo_value.unique()) == ["01", "1"]
    assert sorted(p.name for p in (tmp_path / "labels").glob("*.csv.gz")) == ["01.csv.gz", "1.csv.gz"]


def parquet_like_frame():
    """What reading a Parquet archive yields: object labels with None and NaN."""
    frame = archive_frame(days=60, max_lag=10, locations=("01", "1"))
    bad = frame.iloc[:4].copy()
    bad["geo_value"] = [None, np.nan, "", "  "]
    return pd.concat([frame, bad], ignore_index=True).astype({"geo_value": object})


def test_parquet_labels_are_checked_before_they_become_text(tmp_path, monkeypatch):
    from autodelphirf.pre_diagnosis import preparation
    path = tmp_path / "a.parquet"
    path.write_bytes(b"")                       # the reader below stands in for the engine
    monkeypatch.setattr(preparation.pd, "read_parquet",
                        lambda *args, **kwargs: parquet_like_frame())
    diagnosis = diagnose(path)
    assert diagnosis["dropped_missing_location_rows"] == 4
    assert diagnosis["n_locations"] == 2                  # no "None" or "nan" location


def test_a_real_parquet_archive_drops_missing_labels(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "a.parquet"
    parquet_like_frame().to_parquet(path, index=False)
    diagnosis = diagnose(path)
    assert diagnosis["dropped_missing_location_rows"] == 4
    assert diagnosis["n_locations"] == 2
