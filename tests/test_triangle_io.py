"""Loading prepared triangles from a directory the user assembled.

``load_prepared_triangles`` concatenates one file per location. The failure
modes worth pinning are all about *which* files it picks up: a directory that
also holds the schedule, the inventory, and a stale export is the normal
case, not an exotic one, and silently concatenating any of those would
corrupt the run rather than fail it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from autodelphirf.io import (NON_TRIANGLE_STEMS, TRIANGLE_SUFFIXES,
                                load_prepared_triangles, triangle_files)


def triangle(location: str, rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "geo_value": [location] * rows,
        "reference_date": pd.date_range("2024-01-01", periods=rows),
        "report_date": pd.date_range("2024-01-02", periods=rows),
        "lag": range(1, rows + 1),
        "log_value_7dav": [1.0] * rows,
    })


def write_csv_triangles(directory, locations, suffix=".csv"):
    directory.mkdir(parents=True, exist_ok=True)
    for location in locations:
        triangle(location).to_csv(directory / f"{location}{suffix}", index=False)
    return directory


# --- discovery ------------------------------------------------------------

def test_missing_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        load_prepared_triangles(tmp_path / "absent")


def test_a_file_where_a_directory_belongs_says_so(tmp_path):
    path = tmp_path / "triangle.parquet"
    path.write_text("")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        load_prepared_triangles(path)


def test_empty_directory_explains_what_was_expected(tmp_path):
    empty = tmp_path / "prepared"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as error:
        load_prepared_triangles(empty)
    message = str(error.value)
    assert "no prepared triangles found" in message
    # The message must tell the user how to produce one.
    assert "autodelphirf prepare" in message
    assert ".parquet" in message


def test_directory_holding_only_unrelated_files_is_still_empty(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    (directory / "notes.txt").write_text("hello")
    (directory / "archive.json").write_text("{}")
    assert triangle_files(directory) == []


# --- csv triangles --------------------------------------------------------

def test_csv_triangles_load_without_pyarrow(tmp_path):
    """A user with no working Arrow build must not be shut out."""
    directory = write_csv_triangles(tmp_path / "prepared", ["aa", "bb"])
    frame = load_prepared_triangles(directory)
    assert len(frame) == 6
    assert set(frame.geo_value) == {"aa", "bb"}


def test_gzipped_csv_triangles_load(tmp_path):
    directory = write_csv_triangles(tmp_path / "prepared", ["aa"], suffix=".csv.gz")
    assert len(load_prepared_triangles(directory)) == 3


def test_locations_load_in_a_stable_order(tmp_path):
    """Concatenation order affects nothing downstream only if it is fixed."""
    directory = write_csv_triangles(tmp_path / "prepared", ["cc", "aa", "bb"])
    first = load_prepared_triangles(directory).geo_value.tolist()
    second = load_prepared_triangles(directory).geo_value.tolist()
    assert first == second
    assert first[0] == "aa" and first[-1] == "cc"


# --- files that are in the directory but are not triangles -----------------

@pytest.mark.parametrize("stem", sorted(NON_TRIANGLE_STEMS))
def test_sidecar_files_written_next_to_the_triangles_are_not_concatenated(tmp_path, stem):
    """`autodelphirf prepare` writes these into the same directory by design."""
    directory = write_csv_triangles(tmp_path / "prepared", ["aa"])
    # Deliberately give the sidecar a schema that would corrupt the triangle.
    pd.DataFrame({"unrelated": [1, 2, 3, 4]}).to_csv(directory / f"{stem}.csv", index=False)
    frame = load_prepared_triangles(directory)
    assert len(frame) == 3
    assert "unrelated" not in frame.columns


def test_schedule_file_in_the_prepared_directory_is_never_loaded_as_a_location(tmp_path):
    """The real shape of the bug: test_dates.csv always sits beside the triangles."""
    directory = write_csv_triangles(tmp_path / "prepared", ["aa", "bb"])
    pd.DataFrame({"test_date": ["2024-03-01"], "training_days": [180],
                  "target_column": ["log_value_target_7dav"]}).to_csv(
        directory / "test_dates.csv", index=False)
    frame = load_prepared_triangles(directory)
    assert set(frame.geo_value) == {"aa", "bb"}
    assert "test_date" not in frame.columns


# --- format precedence ----------------------------------------------------

def test_one_format_wins_outright_so_a_stale_export_cannot_double_count(tmp_path):
    """A leftover ak.csv beside ak.parquet must not load that location twice."""
    directory = tmp_path / "prepared"
    directory.mkdir()
    # A .parquet that is never read, only detected: presence decides the format.
    (directory / "aa.parquet").write_bytes(b"PAR1")
    triangle("aa").to_csv(directory / "aa.csv", index=False)
    chosen = triangle_files(directory)
    assert [p.name for p in chosen] == ["aa.parquet"]


def test_format_preference_follows_the_declared_order(tmp_path):
    directory = tmp_path / "prepared"
    write_csv_triangles(directory, ["aa"], suffix=".csv")
    write_csv_triangles(directory, ["bb"], suffix=".csv.gz")
    # .csv.gz precedes .csv in TRIANGLE_SUFFIXES, so the gzipped set wins.
    assert TRIANGLE_SUFFIXES.index(".csv.gz") < TRIANGLE_SUFFIXES.index(".csv")
    assert [p.name for p in triangle_files(directory)] == ["bb.csv.gz"]
