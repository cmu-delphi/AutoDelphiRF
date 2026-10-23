"""N7: duplicates keep the last release; skipped locations are always visible."""
from __future__ import annotations

import json

import pandas as pd

from autodelphirf.app import cli
from autodelphirf.app.web import results_summary
from autodelphirf.core.io import load_prepared_triangles
from autodelphirf.core.resources import resource_path
from autodelphirf.pre_diagnosis.preparation import build_spec, run_r_bridge

from conftest import archive_frame
from test_end_to_end import requires_delphirf


@requires_delphirf
def test_conflicting_duplicate_releases_keep_the_last_row(tmp_path):
    frame = archive_frame(days=120, max_lag=30, locations=("aa", "bb"))
    reference, report = frame.reference_date.iloc[5], frame.report_date.iloc[5]
    first = frame.iloc[[5]].assign(value=-1.0)          # an earlier, conflicting copy
    last = frame.iloc[[5]].assign(value=999.0)          # the copy that must win
    pd.concat([first, frame, last], ignore_index=True).to_csv(tmp_path / "dup.csv", index=False)
    spec = build_spec("dup", tmp_path / "dup.csv", tmp_path / "dup", target_lag=14,
                      smoothed=True, triangle_format="csv")
    run_r_bridge(spec)
    assert not (tmp_path / "dup" / "skipped_locations.csv").exists()
    triangle = load_prepared_triangles(tmp_path / "dup")
    row = triangle[(triangle.geo_value == "aa")
                   & (pd.to_datetime(triangle.reference_date) == pd.Timestamp(reference))
                   & (pd.to_datetime(triangle.report_date) == pd.Timestamp(report))]
    assert row.value_raw.tolist() == [999.0]


@requires_delphirf
def test_a_skipped_location_is_reported_everywhere(tmp_path, capsys):
    frame = archive_frame(days=300, max_lag=60, halflife=9.0, locations=("aa", "bb"))
    tiny = archive_frame(days=3, max_lag=5, locations=("cc",))     # too few rows to prepare
    pd.concat([frame, tiny], ignore_index=True).to_csv(tmp_path / "archive.csv", index=False)
    work = tmp_path / "work"
    assert cli.main(["run", "--archive", str(tmp_path / "archive.csv"), "--name", "skip",
                     "--out", str(work), "--target-lag", "21",
                     "--methods", "baseline_null,delphirf"]) == 0
    output = capsys.readouterr().out
    assert "1 location(s) were skipped during preparation" in output
    assert "cc: fewer than 100 archive rows" in output

    results = work / "results"
    recorded = pd.read_csv(results / "skipped_locations.csv")
    assert recorded.geo_value.tolist() == ["cc"]
    manifest = json.loads((results / "run_manifest.json").read_text())
    assert [row["geo_value"] for row in manifest["skipped_locations"]] == ["cc"]
    report = (results / "report" / "report.html").read_text()
    assert "Locations skipped during preparation" in report
    assert "fewer than 100 archive rows" in report
    page = results_summary(results)
    assert [row["geo_value"] for row in page["tables"]["skipped_locations"]] == ["cc"]


def test_the_results_page_renders_skipped_locations():
    script = resource_path("web/app.js").read_text()
    page = resource_path("web/index.html").read_text()
    assert "renderSkippedLocations(tables.skipped_locations" in script
    assert 'id="skipped-locations"' in page



@requires_delphirf
def test_a_duplicate_without_a_value_does_not_replace_one_with_a_value(tmp_path):
    frame = archive_frame(days=120, max_lag=30, locations=("aa", "bb"))
    reference, report = frame.reference_date.iloc[5], frame.report_date.iloc[5]
    kept = frame.iloc[[5]].assign(value=999.0)
    empty = frame.iloc[[5]].assign(value=float("nan"))     # last in file order, but no value
    body = frame.drop(index=5)
    pd.concat([body, kept, empty], ignore_index=True).to_csv(tmp_path / "dup.csv", index=False)
    spec = build_spec("dup", tmp_path / "dup.csv", tmp_path / "dup", target_lag=14,
                      smoothed=True, triangle_format="csv")
    run_r_bridge(spec)
    triangle = load_prepared_triangles(tmp_path / "dup")
    row = triangle[(triangle.geo_value == "aa")
                   & (pd.to_datetime(triangle.reference_date) == pd.Timestamp(reference))
                   & (pd.to_datetime(triangle.report_date) == pd.Timestamp(report))]
    assert row.value_raw.tolist() == [999.0]
    assert row.genuine_event.astype(bool).tolist() == [True]
