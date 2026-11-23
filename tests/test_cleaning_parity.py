"""Python diagnosis and R preparation clean a raw archive under one contract.

One fixture holds rows that each break exactly one rule. Under the same scope
(lag bound and as-of date) both sides must keep the same releases and values,
make the same duplicate choice, and count the same removals per reason. Their
own scopes legitimately differ (two-year diagnosis bound vs. target lag + 100
days and an as-of date for preparation), and each audit records its scope.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autodelphirf.pre_diagnosis.archive import (CLEANING_CONTRACT_VERSION, clean_raw_archive,
                                                raw_cleaning_audit)
from autodelphirf.pre_diagnosis.preparation import (build_spec, diagnose_archive, run_r_bridge,
                                                    standardize_raw_archive)

from conftest import archive_frame
from test_end_to_end import requires_delphirf

TARGET_LAG = 14
HORIZON = TARGET_LAG + 100           # R preparation's feature horizon
AS_OF = pd.Timestamp("2024-01-02")   # reaches the lag-731 row from 2022-01-01
COLUMNS = dict(geo_col="geo_value", reference_col="reference_date",
               report_col="report_date", value_cols=("numerator", "denominator"),
               value_type="fraction")


def row(geo, reference, lag=None, *, report=None, numerator=50.0, denominator=200.0):
    reference = pd.Timestamp(reference) if reference is not None else None
    if report is None and reference is not None and lag is not None:
        report = reference + pd.Timedelta(days=lag)
    return {"geo_value": geo, "reference_date": reference, "report_date": report,
            "numerator": numerator, "denominator": denominator}


def fixture(path):
    base = archive_frame(days=120, max_lag=30, locations=("01", "1"))
    base = base.assign(numerator=base.value, denominator=base.value * 2 + 10).drop(columns="value")
    special = [
        row(np.nan, "2022-02-01", 40), row("", "2022-02-02", 40), row("   ", "2022-02-03", 40),
        row("01", None, report=pd.Timestamp("2022-02-10")),          # missing reference date
        row("01", "2022-02-04", report=None),                        # missing report date
        row("01", "2022-02-05", report=pd.Timestamp("2022-02-03")),  # negative lag
        row("01", "2022-02-06", HORIZON),                            # at the preparation bound
        row("01", "2022-02-07", HORIZON + 1),                        # just beyond it
        row("01", "2022-01-01", 730), row("1", "2022-01-01", 731),
        row("01", "2022-02-08", 40, numerator="abc"),                # non-numeric value
        row("01", "2022-02-09", 40, numerator=np.nan),               # missing value
        row("01", "2022-02-10", 40, denominator=0.0),
        row("01", "2022-02-11", 40, denominator=-5.0),
        row("01", "2022-03-01", report=pd.Timestamp("2024-02-01")),  # after the as-of date
        row("1", "2022-03-02", report=pd.Timestamp("2024-02-02")),
    ]
    first = row("01", "2022-03-03", 40, numerator=10.0)
    last = row("01", "2022-03-03", 40, numerator=20.0)                # duplicate: this one wins
    kept = row("1", "2022-03-04", 40, numerator=30.0)
    empty = row("1", "2022-03-04", 40, numerator=np.nan)             # final duplicate, no value
    rows = pd.concat([base, pd.DataFrame(special)], ignore_index=True)
    rows = rows.sample(frac=1, random_state=7).reset_index(drop=True)  # unsorted input
    middle = len(rows) // 2
    rows = pd.concat([rows.iloc[:middle], pd.DataFrame([first, kept]),
                      rows.iloc[middle:], pd.DataFrame([last, empty])], ignore_index=True)
    rows.to_csv(path, index=False)
    return path


def python_cleaning(path, **scope):
    work, missing_location, input_rows = standardize_raw_archive(path, **COLUMNS)
    retained, counts = clean_raw_archive(work, **scope)
    return retained, missing_location, input_rows, counts


@pytest.fixture
def archive(tmp_path):
    return fixture(tmp_path / "archive.csv")


def test_python_audit_counts_each_rule_once(archive):
    retained, missing_location, _, counts = python_cleaning(archive)
    assert missing_location == 3
    counts.pop("as_of")
    assert counts == {"invalid_date_rows": 2, "future_report_rows": 0,
                      "unusable_value_rows": 5,  # abc, nan, 0, -5, final duplicate
                      "negative_lag_rows": 1, "over_lag_limit_rows": 1,
                      "duplicate_rows_removed": 1}


def test_diagnosis_records_its_scope(archive):
    audit = diagnose_archive(archive, **COLUMNS)["raw_cleaning_audit"]
    assert audit["scope"] == "pre_diagnosis"
    assert audit["as_of_source"] == "inferred"
    assert audit["effective_as_of_date"] == audit["max_visible_report_date"] == "2024-02-02"
    assert audit["contract_version"] == CLEANING_CONTRACT_VERSION
    assert audit["lag_upper_bound"] == 730
    assert audit["missing_location_rows"] == 3 and audit["retained_locations"] == 2


@requires_delphirf
def test_python_and_r_agree_under_the_same_scope(archive, tmp_path):
    spec = build_spec("parity", archive, tmp_path / "parity", target_lag=TARGET_LAG,
                      as_of_date=str(AS_OF.date()), smoothed=True, triangle_format="csv",
                      **{k: v for k, v in COLUMNS.items()})
    run_r_bridge(replace(spec, write_cleaned_releases=True))
    r_audit = json.loads((tmp_path / "parity" / "raw_cleaning_audit.json").read_text())
    r_kept = pd.read_csv(tmp_path / "parity" / "cleaned_releases.csv", dtype={"geo_value": str},
                         parse_dates=["reference_date", "report_date"])

    py_kept, missing_location, input_rows, counts = python_cleaning(
        archive, lag_upper_bound=HORIZON, as_of_date=AS_OF)
    py_audit = raw_cleaning_audit("feature_preparation", input_rows=input_rows,
                                  missing_location_rows=missing_location, counts=counts,
                                  retained=py_kept, lag_upper_bound=HORIZON)

    # Shared rules: the same audit for every shared field (R's audit also
    # records its own experiment window and location-size rule), and the same
    # releases, values and duplicate choice.
    assert {key: r_audit[key] for key in py_audit} == py_audit
    assert py_audit["as_of_source"] == "explicit"
    keys = ["geo_value", "reference_date", "report_date"]
    r_kept = r_kept.assign(value=r_kept.numerator / r_kept.denominator).sort_values(keys)
    py_kept = py_kept.sort_values(keys)
    assert set(r_kept.geo_value) == set(py_kept.geo_value) == {"01", "1"}
    assert r_kept[keys].reset_index(drop=True).equals(py_kept[keys].reset_index(drop=True))
    assert np.allclose(r_kept.value.to_numpy(), py_kept.value.to_numpy())
    duplicate = r_kept[(r_kept.geo_value == "01") & (r_kept.reference_date == "2022-03-03")
                       & (r_kept.report_date == "2022-04-12")]
    assert duplicate.numerator.tolist() == [20.0]
    final_without_value = r_kept[(r_kept.geo_value == "1") & (r_kept.reference_date == "2022-03-04")
                                 & (r_kept.report_date == "2022-04-13")]
    assert final_without_value.numerator.tolist() == [30.0]


@requires_delphirf
def test_each_side_keeps_its_own_scope_and_says_so(archive, tmp_path):
    spec = build_spec("scope", archive, tmp_path / "scope", target_lag=TARGET_LAG,
                      as_of_date=str(AS_OF.date()), smoothed=True, triangle_format="csv",
                      **{k: v for k, v in COLUMNS.items()})
    run_r_bridge(spec)
    r_audit = json.loads((tmp_path / "scope" / "raw_cleaning_audit.json").read_text())
    diagnosis_audit = spec.diagnosis["raw_cleaning_audit"]
    py_kept, *_ = python_cleaning(archive)

    assert r_audit["contract_version"] == diagnosis_audit["contract_version"]
    # The diagnosis keeps lags through two years and every report date.
    assert diagnosis_audit["lag_upper_bound"] == 730
    assert (py_kept.lag == 730).any() and (py_kept.report_date > AS_OF).any()
    # Preparation stops at the target lag + 100 days and at its as-of date,
    # and also applies its experiment window and minimum location size.
    assert r_audit["scope"] == "feature_preparation" and diagnosis_audit["scope"] == "pre_diagnosis"
    assert r_audit["lag_upper_bound"] == HORIZON
    assert r_audit["effective_as_of_date"] == str(AS_OF.date())
    assert {"min_reference_date", "max_report_date"} <= set(r_audit["experiment_window"])
    assert "outside_experiment_window_rows" in r_audit and "min_location_rows" in r_audit
    assert "experiment_window" not in diagnosis_audit
    assert r_audit["over_lag_limit_rows"] == 3            # lags 115, 730, 731
    assert r_audit["future_report_rows"] == 2
    assert r_audit["retained_rows"] < diagnosis_audit["retained_rows"]


def test_a_run_refuses_mismatched_cleaning_contracts(tmp_path, monkeypatch):
    from test_target_lag_diagnosis import prepared_only_dataset
    from autodelphirf.workflow.run import run_pipeline
    monkeypatch.chdir(tmp_path)
    config = prepared_only_dataset(tmp_path, target_lag=7)
    preparation = config.prepared_dir / "preparation.json"
    metadata = json.loads(preparation.read_text())
    metadata["diagnosis"] = {"raw_cleaning_audit": {"contract_version": 1, "retained_rows": 10}}
    preparation.write_text(json.dumps(metadata))
    audit = config.prepared_dir / "raw_cleaning_audit.json"
    # Different retained totals alone are expected and accepted.
    audit.write_text(json.dumps({"contract_version": 1, "retained_rows": 7}))
    run_pipeline(config, diagnosis_only=True)
    audit.write_text(json.dumps({"contract_version": 2, "retained_rows": 7}))
    with pytest.raises(ValueError, match="raw-cleaning contract"):
        run_pipeline(config, diagnosis_only=True)


# --- the date rule ------------------------------------------------------------------

def timestamped_archive(path):
    """Report dates written as timestamps, as the Epidata archive exports them."""
    frame = archive_frame(days=60, max_lag=20, locations=("aa", "bb"))
    report = pd.to_datetime(frame.report_date).dt.strftime("%Y-%m-%d")
    # Alternate the accepted spellings of one calendar date.
    spellings = ["{}T00:00:00Z", "{}T23:30:00-05:00", "{} 00:00:00", "{}"]
    frame["report_date"] = [spellings[i % 4].format(day) for i, day in enumerate(report)]
    invalid = frame.iloc[:3].copy()
    invalid["report_date"] = ["2022/01/20", "2022-02-30", "01/20/2022"]
    frame = pd.concat([frame, invalid], ignore_index=True)
    frame.to_csv(path, index=False)
    return frame


def test_dates_are_the_calendar_date_as_written(tmp_path):
    timestamped_archive(tmp_path / "a.csv")
    work, _, _ = standardize_raw_archive(tmp_path / "a.csv", geo_col="geo_value",
                                         reference_col="reference_date",
                                         report_col="report_date", value_cols=("value",),
                                         value_type="count")
    retained, counts = clean_raw_archive(work)
    assert counts["invalid_date_rows"] == 3
    # The time and offset are ignored, not converted: 23:30 at -05:00 stays that day.
    plain = archive_frame(days=60, max_lag=20, locations=("aa", "bb"))
    assert sorted(retained.report_date.dt.strftime("%Y-%m-%d")) == sorted(
        pd.to_datetime(plain.report_date).dt.strftime("%Y-%m-%d"))
    assert retained.report_date.dt.tz is None


@requires_delphirf
def test_python_and_r_read_timestamped_dates_alike(tmp_path):
    timestamped_archive(tmp_path / "a.csv")
    spec = build_spec("stamped", tmp_path / "a.csv", tmp_path / "stamped", target_lag=7,
                      smoothed=True, triangle_format="csv", value_cols=("value",),
                      value_type="count")
    run_r_bridge(replace(spec, write_cleaned_releases=True))
    r_audit = json.loads((tmp_path / "stamped" / "raw_cleaning_audit.json").read_text())
    r_kept = pd.read_csv(tmp_path / "stamped" / "cleaned_releases.csv", dtype={"geo_value": str})
    py_audit = spec.diagnosis["raw_cleaning_audit"]
    assert r_audit["invalid_date_rows"] == py_audit["invalid_date_rows"] == 3
    assert r_audit["effective_as_of_date"] == py_audit["effective_as_of_date"]
    assert pd.to_datetime(r_kept.report_date).max() == pd.Timestamp(
        py_audit["effective_as_of_date"])


# --- the as-of date (Phase 3) -----------------------------------------------------

def as_of_archive(path):
    frame = archive_frame(days=60, max_lag=20, locations=("aa",))
    frame.to_csv(path, index=False)
    return frame


def test_an_omitted_as_of_date_is_the_latest_report(tmp_path):
    frame = as_of_archive(tmp_path / "a.csv")
    audit = diagnose_archive(tmp_path / "a.csv", geo_col="geo_value", reference_col="reference_date",
                             report_col="report_date", value_cols=("value",),
                             value_type="count")["raw_cleaning_audit"]
    latest = str(pd.to_datetime(frame.report_date).max().date())
    assert audit["as_of_source"] == "inferred"
    assert audit["effective_as_of_date"] == audit["max_visible_report_date"] == latest
    assert audit["future_report_rows"] == 0


def test_an_earlier_as_of_date_hides_later_reports(tmp_path):
    frame = as_of_archive(tmp_path / "a.csv")
    diagnosis = diagnose_archive(tmp_path / "a.csv", geo_col="geo_value",
                                 reference_col="reference_date", report_col="report_date",
                                 value_cols=("value",), value_type="count", as_of_date="2022-02-01")
    audit = diagnosis["raw_cleaning_audit"]
    later = int((pd.to_datetime(frame.report_date) > "2022-02-01").sum())
    assert audit["as_of_source"] == "explicit" and audit["effective_as_of_date"] == "2022-02-01"
    assert audit["future_report_rows"] == later
    assert audit["max_visible_report_date"] == "2022-02-01"
    assert diagnosis["report_date_max"] == "2022-02-01"


def test_a_later_as_of_date_is_the_analysis_clock(tmp_path):
    frame = as_of_archive(tmp_path / "a.csv")
    audit = diagnose_archive(tmp_path / "a.csv", geo_col="geo_value", reference_col="reference_date",
                             report_col="report_date", value_cols=("value",),
                             value_type="count", as_of_date="2030-01-01")["raw_cleaning_audit"]
    # Quiet days after the last release still count: the as-of date is not moved back.
    assert audit["effective_as_of_date"] == "2030-01-01"
    assert audit["max_visible_report_date"] == str(pd.to_datetime(frame.report_date).max().date())
    assert audit["future_report_rows"] == 0


@requires_delphirf
def test_preparation_records_both_audits(archive, tmp_path):
    spec = build_spec("both", archive, tmp_path / "both", target_lag=TARGET_LAG,
                      as_of_date=str(AS_OF.date()), smoothed=True, triangle_format="csv",
                      **{k: v for k, v in COLUMNS.items()})
    from autodelphirf.pre_diagnosis.preparation import prepare_dataset
    prepare_dataset(spec, prediction_layers=("baseline_null",), config_path=tmp_path / "both.json")
    audits = json.loads((tmp_path / "both" / "preparation.json").read_text())["raw_cleaning_audits"]
    assert audits["pre_diagnosis"]["scope"] == "pre_diagnosis"
    assert audits["feature_preparation"]["scope"] == "feature_preparation"
    assert (audits["pre_diagnosis"]["contract_version"]
            == audits["feature_preparation"]["contract_version"] == CLEANING_CONTRACT_VERSION)


@requires_delphirf
def test_r_rejects_an_unsupported_contract_version(archive, tmp_path):
    from autodelphirf.pre_diagnosis.preparation import PreparationError, PreparationSpec
    spec = build_spec("v", archive, tmp_path / "v", target_lag=TARGET_LAG, smoothed=True,
                      triangle_format="csv", **{k: v for k, v in COLUMNS.items()})

    class Future(PreparationSpec):
        def to_r_spec(self):
            return {**super().to_r_spec(), "raw_contract_version": 99}
    with pytest.raises(PreparationError):
        run_r_bridge(Future(**{f: getattr(spec, f) for f in spec.__dataclass_fields__}))
