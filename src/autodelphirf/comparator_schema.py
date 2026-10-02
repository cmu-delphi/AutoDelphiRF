"""Validation for externally supplied prediction files."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# Same identifying-key set pipeline.py merges on (autodelphirf.pipeline.KEYS).
IDENTIFYING_KEYS = ["fold", "cutoff", "geo_value", "reference_date", "report_date", "target_date", "lag"]
# The subset that must be present for rows to be uniquely identifiable even
# when the optional fold/cutoff/target_date columns are omitted (matches the
# README's "the other five keys identify a row uniquely" -- five including
# method_column, four positional plus lag).
MINIMUM_IDENTIFYING_KEYS = ["geo_value", "reference_date", "report_date", "lag"]
DATE_COLUMNS = ["cutoff", "reference_date", "report_date", "target_date"]


def check_comparator_file(comparator_path, method_column: str, prediction_column: str,
                          methods: dict[str, str], comparator_columns: dict[str, str] | None = None,
                          ) -> list[str]:
    """Check a comparator file against the ingestion contract and return a
    list of human-readable issues (empty list means the file is valid).

    Never raises for a missing/malformed file itself -- callers that want a
    hard failure should use ``validate_comparator_file`` instead, which wraps
    this in a single ``ValueError`` listing every issue found.
    """
    comparator_columns = comparator_columns or {}
    issues: list[str] = []
    path = Path(comparator_path)
    if not path.exists():
        return [f"comparator_file does not exist: {path}"]
    try:
        comparator = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a validation issue, not a crash
        return [f"comparator_file could not be read as CSV ({path}): {exc}"]

    if comparator.empty:
        issues.append("comparator_file has no rows.")

    present_identifying = [key for key in IDENTIFYING_KEYS if key in comparator.columns]
    missing_minimum = [key for key in MINIMUM_IDENTIFYING_KEYS if key not in comparator.columns]
    if missing_minimum:
        issues.append("comparator_file is missing minimum identifying columns "
                      f"{missing_minimum} (needs at least {MINIMUM_IDENTIFYING_KEYS}, "
                      f"optionally also {[k for k in IDENTIFYING_KEYS if k not in MINIMUM_IDENTIFYING_KEYS]}).")

    if method_column not in comparator.columns:
        issues.append(f"comparator_method_column '{method_column}' is not a column in comparator_file "
                      f"(columns present: {list(comparator.columns)}).")
    if prediction_column not in comparator.columns:
        issues.append(f"comparator_prediction_column '{prediction_column}' is not a column in comparator_file "
                      f"(columns present: {list(comparator.columns)}).")

    # Every configured source method name must actually occur, or the inner
    # join in pipeline.py silently produces zero merged rows for *every*
    # configured comparator method, not just the missing one.
    if method_column in comparator.columns:
        seen_methods = set(comparator[method_column].dropna().unique())
        for output_name, source_name in methods.items():
            if source_name not in seen_methods:
                issues.append(f"comparator_methods maps '{output_name}' -> '{source_name}', but "
                              f"'{source_name}' does not appear in column '{method_column}' "
                              f"(values present: {sorted(seen_methods)}). This would silently zero out "
                              "every comparator method's merged rows, not just this one.")

    # Duplicate (identifying keys, method) rows break the one-to-one merge in
    # pipeline.py with a bare pandas MergeError; check explicitly and name the
    # offending rows instead.
    if present_identifying and method_column in comparator.columns and prediction_column in comparator.columns:
        for output_name, source_name in methods.items():
            selected = comparator[comparator[method_column].eq(source_name)]
            dup_mask = selected.duplicated(subset=present_identifying, keep=False)
            if dup_mask.any():
                n_dup = int(dup_mask.sum())
                issues.append(f"method '{source_name}' (-> '{output_name}') has {n_dup} rows sharing the same "
                              f"{present_identifying} -- each case must have exactly one row per method.")
        if prediction_column in comparator.columns:
            non_numeric = pd.to_numeric(comparator[prediction_column], errors="coerce").isna() & comparator[prediction_column].notna()
            if non_numeric.any():
                issues.append(f"comparator_prediction_column '{prediction_column}' has "
                              f"{int(non_numeric.sum())} non-numeric value(s).")

    for column in comparator_columns.values():
        if column not in comparator.columns:
            issues.append(f"comparator_columns references column '{column}', which is not present in "
                          f"comparator_file (columns present: {list(comparator.columns)}).")

    for column in DATE_COLUMNS:
        if column in comparator.columns:
            parsed = pd.to_datetime(comparator[column], errors="coerce")
            bad = parsed.isna() & comparator[column].notna()
            if bad.any():
                issues.append(f"column '{column}' has {int(bad.sum())} value(s) that do not parse as dates.")

    return issues


def validate_comparator_file(comparator_path, method_column: str, prediction_column: str,
                             methods: dict[str, str], comparator_columns: dict[str, str] | None = None) -> None:
    """Raise ``ValueError`` with every issue found, or return silently if the
    file is valid. Call this before ingesting a comparator_file (see
    ``pipeline.py``'s ``external_comparator_ingestion`` stage)."""
    issues = check_comparator_file(comparator_path, method_column, prediction_column, methods, comparator_columns)
    if issues:
        listed = "\n".join(f"  - {issue}" for issue in issues)
        raise ValueError(f"comparator_file failed validation ({comparator_path}):\n{listed}")
