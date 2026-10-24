"""Convert a raw revision archive into prepared AutoDelphiRF inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd

from ..core.io import read_csv
from .diagnosis import WEEKDAYS, weekday_groups_from_share
from .archive import (CLEANING_CONTRACT_VERSION, DEFAULT_TRAINING_WINDOW_DAYS,
                      MAX_OBSERVED_LAG_DAYS, diagnose_raw_archive)
from ..core.resources import resource_path

#: Fraction signals are the one case where the archive carries two value
#: columns (numerator and denominator) and DelphiRF models their ratio.
VALUE_TYPES = ("count", "fraction")

#: Emitted by ``data_preprocessing`` for a smoothed build vs. an unsmoothed
#: one. The schedule names the one the run evaluates against.
TARGET_COLUMNS = {True: "log_value_target_7dav", False: "log_value_target"}

#: Days between retraining origins, by diagnosed reporting cadence. Every test
#: origin is a retrain: the models are refitted on the history visible at that
#: date, so this is the retraining interval, not merely an evaluation stride.
#: A weekly stream gains a new observation per reference date each week, so it
#: is retrained fortnightly; a daily stream's week-to-week movement is mostly
#: within-week noise, so it is retrained monthly. Both are defaults the user
#: resets.
DEFAULT_RETRAIN_DAYS = {"weekly": 14, "daily": 30}

#: How long after the archive's first report date the first retraining origin
#: sits. The models need *some* completed revision history to train on before
#: the first origin, and 60 days is the shortest span that reliably contains
#: one for either cadence.
DEFAULT_FIRST_ORIGIN_OFFSET_DAYS = 60

#: Locations with fewer archive rows than this cannot support lagged features
#: and a target; they are reported in ``skipped_locations.csv``, not dropped
#: silently.
DEFAULT_MIN_LOCATION_ROWS = 100

#: Rows read to diagnose the archive. The diagnosis needs the whole revision
#: history, so this is a guard against an accidental multi-gigabyte read
#: rather than a sample: exceeding it is reported, never silently truncated.
DIAGNOSIS_ROW_LIMIT = 20_000_000


class PreparationError(RuntimeError):
    """A prepared triangle could not be built."""


#: ISO weekday numbers (Monday = 1) -> DelphiRF's canonical weekday columns.
ISO_WEEKDAYS = dict(zip(range(1, 8), WEEKDAYS))
#: Group labels whose ``_ref``/``_issue`` columns would overwrite DelphiRF's
#: week-of-month indicators.
RESERVED_GROUP_LABELS = frozenset({"W1", "W2", "W3"})


def parse_weekday_groups(groups) -> dict:
    """Validate user weekday groups given as ISO weekday numbers.

    ``{"monday": [1], "weekend": [6, 7]}`` assigns Monday to one group and
    Saturday and Sunday to another; every weekday left out of the groups forms
    the reference category; ``{}`` means no weekday contrasts at all. Returns
    DelphiRF's canonical form, ``{"monday": ["Mon"], "weekend": ["Sat", "Sun"]}``.
    An invalid specification raises; it never falls back to automatic detection.
    """
    if not isinstance(groups, dict):
        raise PreparationError(
            "weekday_groups must be an object mapping group names to lists of ISO weekday "
            "numbers (1 = Monday ... 7 = Sunday), e.g. {\"monday\": [1], \"weekend\": [6, 7]}; "
            "omit it or use null for automatic detection")
    parsed, used = {}, {}
    for name, members in groups.items():
        if (not isinstance(name, str) or not name or not name[0].isalpha()
                or not name.replace("_", "").isalnum()):
            raise PreparationError(
                f"invalid weekday group name {name!r}: use letters, digits and underscores, "
                "starting with a letter")
        if name in RESERVED_GROUP_LABELS:
            raise PreparationError(
                f"weekday group name {name!r} is reserved for DelphiRF's week-of-month features")
        if not isinstance(members, list) or not members:
            raise PreparationError(
                f"weekday group {name!r} must list at least one ISO weekday number (1-7)")
        days = []
        for day in members:
            if isinstance(day, bool) or not isinstance(day, int) or day not in ISO_WEEKDAYS:
                raise PreparationError(
                    f"weekday group {name!r} contains {day!r}; weekdays are the integers "
                    "1 (Monday) through 7 (Sunday)")
            if day in used:
                raise PreparationError(
                    f"weekday {day} is assigned to both {used[day]!r} and {name!r}")
            used[day] = name
            days.append(ISO_WEEKDAYS[day])
        parsed[name] = days
    if len(used) == len(ISO_WEEKDAYS):
        raise PreparationError(
            "weekday_groups assigns all seven weekdays; at least one weekday must be left "
            "out to form the reference category")
    return parsed


def iso_weekday_groups(groups: dict) -> dict:
    """The canonical form back in ISO weekday numbers (for a frozen setting)."""
    numbers = {day: number for number, day in ISO_WEEKDAYS.items()}
    return {name: [numbers[day] for day in members] for name, members in (groups or {}).items()}


def resolve_weekday_groups(manual, report_weekday_share: dict,
                           temporal_resolution: str) -> tuple[dict, str]:
    """The single effective weekday grouping and where it came from.

    Valid manual groups win; without them (``None``) the groups are detected
    from the weekday share of distinct report dates. Weekly data has no
    weekday contrasts.
    """
    if temporal_resolution == "weekly":
        if manual:
            raise PreparationError("weekday groups cannot be used with weekly data")
        return {}, "not applicable (weekly)"
    if manual is not None:
        return parse_weekday_groups(manual), "manual"
    return weekday_groups_from_share(report_weekday_share or {}, temporal_resolution), "automatic"


@dataclass
class PreparationSpec:
    """Everything ``prepare_triangle.R`` needs, and where each value came from.

    ``overrides`` names the fields the user set explicitly; every other field
    was diagnosed from the archive. Both are written to ``preparation.json``.
    """
    name: str
    raw_csv: Path
    output_dir: Path
    reference_col: str
    report_col: str
    value_cols: tuple[str, ...]
    geo_col: str | None
    constant_geo: str
    value_type: str
    smoothed: bool
    temporal_resol: str
    ref_lag: int
    max_observed_lag_days: int
    lag_terms: tuple[int, ...]
    weekday_groups: dict
    weekday_groups_source: str
    training_days: int
    testing_days: int
    target_column: str
    start_date: str | None
    end_date: str | None
    experiment_end_date: str | None
    triangle_format: str
    min_location_rows: int
    overrides: tuple[str, ...] = ()
    diagnosis: dict = field(default_factory=dict)
    as_of_date: str | None = None
    #: Also write the cleaned releases R passes to DelphiRF (cleaned_releases.csv),
    #: for checking that Python and R apply the same cleaning rules.
    write_cleaned_releases: bool = False
    #: Train on, and forecast and score, only rows with a release actually
    #: reported at that lag (DelphiRF's ``genuine_event``). False keeps rows the
    #: daily grid carries forward; predictions record ``genuine_event`` either way.
    genuine_training: bool = True
    genuine_testing: bool = True

    def to_r_spec(self) -> dict:
        """The JSON handed to the R bridge (paths as strings, no provenance)."""
        return {
            "name": self.name, "raw_csv": str(self.raw_csv), "output_dir": str(self.output_dir),
            "reference_col": self.reference_col, "report_col": self.report_col,
            "value_cols": list(self.value_cols), "geo_col": self.geo_col,
            "constant_geo": self.constant_geo, "value_type": self.value_type,
            "smoothed": self.smoothed, "temporal_resol": self.temporal_resol,
            "ref_lag": int(self.ref_lag),
            "max_observed_lag_days": int(self.max_observed_lag_days),
            "lag_terms": [int(t) for t in self.lag_terms],
            "weekday_groups": self.weekday_groups,
            "weekday_groups_source": self.weekday_groups_source,
            "training_days": int(self.training_days),
            "testing_days": int(self.testing_days), "target_column": self.target_column,
            "start_date": self.start_date, "end_date": self.end_date,
            "experiment_end_date": self.experiment_end_date,
            "triangle_format": self.triangle_format,
            "min_location_rows": int(self.min_location_rows),
            "as_of_date": self.as_of_date,
            "write_cleaned_releases": bool(self.write_cleaned_releases),
            "raw_contract_version": CLEANING_CONTRACT_VERSION,
        }


def read_archive(path: Path, columns: list[str] | None = None,
                 geo_col: str | None = None) -> pd.DataFrame:
    """Read a raw archive from CSV (optionally gzipped) or Parquet.

    The geography column is read as text, so "01" and "1" stay distinct. A
    missing label stays missing (pandas' nullable string type), never the
    literal text "None" or "nan", so the missing-location check can see it.
    """
    path = Path(path)
    if not path.is_file():
        raise PreparationError(f"raw archive not found: {path}")
    if path.suffix == ".parquet":
        try:
            frame = pd.read_parquet(path, columns=columns)
            if geo_col and geo_col in frame:
                frame[geo_col] = frame[geo_col].astype("string")
            return frame
        except ImportError as error:
            raise PreparationError(
                f"reading {path.name} needs pyarrow: pip install 'autodelphirf[parquet]', or "
                "supply the archive as CSV.") from error
    frame = read_csv(path, geo_col=geo_col, usecols=columns, nrows=DIAGNOSIS_ROW_LIMIT + 1)
    if len(frame) > DIAGNOSIS_ROW_LIMIT:
        raise PreparationError(
            f"{path.name} has more than {DIAGNOSIS_ROW_LIMIT:,} rows. Diagnosis needs the whole "
            "revision history, so it is not sampled; preprocess or partition the archive first.")
    return frame


def standardize_raw_archive(raw_csv: Path, *, geo_col: str | None, reference_col: str,
                            report_col: str, value_cols: tuple[str, ...],
                            value_type: str) -> tuple[pd.DataFrame, int, int]:
    """The raw archive as geo_value, reference_date, report_date, value.

    Returns the frame, the rows removed for a missing location, and the
    input row count. The remaining rules are in ``archive.clean_raw_archive``.
    """
    needed = [c for c in (geo_col, reference_col, report_col, *value_cols) if c]
    frame = read_archive(raw_csv, columns=None, geo_col=geo_col)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise PreparationError(
            f"raw archive is missing column(s) {missing}. Present: "
            f"{sorted(frame.columns)[:40]}. Name your columns with --reference-col / "
            "--report-col / --value-col / --geo-col.")
    # A declared geography column must name a location on every row: a missing
    # or blank label is invalid and the row is dropped, before the labels are
    # read as text ("01" and "1" stay distinct locations).
    dropped_locations = 0
    input_rows = len(frame)
    if geo_col:
        labels = frame[geo_col]
        # Missingness is judged before any conversion to plain strings, which
        # would turn a missing label into the text "None" or "nan".
        invalid = labels.isna() | labels.astype("string").str.strip().eq("").fillna(True)
        dropped_locations = int(invalid.sum())
        frame = frame.loc[~invalid].copy()
        frame[geo_col] = frame[geo_col].astype(str)
        if frame.empty:
            raise PreparationError(
                f"the geography column {geo_col!r} has no valid location on any row; check "
                "--geo-col, or use --no-geo for a single-location archive")
    work = pd.DataFrame({
        "reference_date": frame[reference_col],
        "report_date": frame[report_col],
        "geo_value": (frame[geo_col].astype(str) if geo_col else "single_location")})
    if value_type == "fraction" and len(value_cols) == 2:
        numerator = pd.to_numeric(frame[value_cols[0]], errors="coerce")
        denominator = pd.to_numeric(frame[value_cols[1]], errors="coerce")
        work["value"] = numerator / denominator.where(denominator > 0)
    else:
        work["value"] = pd.to_numeric(frame[value_cols[0]], errors="coerce")
    return work, dropped_locations, input_rows


def diagnose_archive(raw_csv: Path, *, geo_col: str | None, reference_col: str,
                     report_col: str, value_cols: tuple[str, ...], value_type: str,
                     user_target_lag: int | None = None, as_of_date=None) -> dict:
    """Diagnose a raw archive, returning :func:`diagnose_raw_archive`'s report.

    A fraction signal is diagnosed on the ratio it models, not on its
    numerator: the revision behaviour of a rate is the thing RevRoute routes
    on, and numerator and denominator revise on their own schedules.
    """
    work, dropped_locations, input_rows = standardize_raw_archive(
        raw_csv, geo_col=geo_col, reference_col=reference_col, report_col=report_col,
        value_cols=value_cols, value_type=value_type)
    # The as-of date is the analysis clock: nothing reported after it is seen.
    diagnosis = diagnose_raw_archive(work, user_target_lag=user_target_lag,
                                     as_of_date=as_of_date).to_dict()
    diagnosis["dropped_missing_location_rows"] = dropped_locations
    cleaning = dict(diagnosis.pop("cleaning"))
    diagnosis["raw_cleaning_audit"] = {
        "scope": "pre_diagnosis", "contract_version": CLEANING_CONTRACT_VERSION,
        **cleaning.pop("as_of"), "lag_upper_bound": MAX_OBSERVED_LAG_DAYS,
        "input_rows": int(input_rows), "missing_location_rows": dropped_locations, **cleaning,
        "retained_rows": int(diagnosis["n_rows"]),
        "retained_locations": int(diagnosis["n_locations"])}
    if dropped_locations:
        diagnosis["notes"] = list(diagnosis.get("notes") or ()) + [
            f"Dropped {dropped_locations:,} row(s) with a missing or blank location."]
    if geo_col is None:
        diagnosis["notes"] = list(diagnosis.get("notes") or ()) + [
            "No geography column was specified. AutoDelphiRF will treat the archive as "
            "one location named 'single_location'; RevRoute can pool across lags, but "
            "cross-location pooling is not possible."
        ]
    return diagnosis


def retraining_calendar(report_date_min, report_date_max, retrain_days: int,
                       first_origin: str | None = None,
                       first_origin_offset_days: int = DEFAULT_FIRST_ORIGIN_OFFSET_DAYS) -> dict:
    """Lay out the rolling retraining origins along the report axis.

    One origin every ``retrain_days`` from the first origin to the archive's
    last report date. The first origin defaults to
    ``first_origin_offset_days`` after the archive's first report date, which
    is what leaves the models some completed revision history to train on.

    The web UI previews a schedule with this function before the run starts,
    so what it shows is what :func:`build_spec` goes on to build.
    """
    first_report = pd.Timestamp(report_date_min).normalize()
    last_report = pd.Timestamp(report_date_max).normalize()
    if retrain_days < 1:
        raise PreparationError(f"the retraining interval must be at least one day, got {retrain_days}")
    try:
        start = (pd.Timestamp(first_origin).normalize() if first_origin is not None
                 else first_report + pd.Timedelta(days=first_origin_offset_days))
    except ValueError as error:
        raise PreparationError(
            f"the first retraining origin must be a date, got {first_origin!r}") from error
    if start > last_report:
        raise PreparationError(
            f"the first retraining origin ({start.date()}) is after the archive's last report "
            f"date ({last_report.date()}), so there is nothing to retrain on. Choose an earlier "
            "first origin.")
    if start < first_report:
        raise PreparationError(
            f"the first retraining origin ({start.date()}) is before the archive's first report "
            f"date ({first_report.date()}).")
    origins = pd.date_range(start, last_report, freq=pd.Timedelta(days=retrain_days))
    # The last origin is evaluated over one further interval, so its window
    # matches every other fold's instead of being whatever remainder the range
    # happened to leave.
    return {"first_origin": str(origins[0].date()), "last_origin": str(origins[-1].date()),
            "experiment_end_date": str((origins[-1] + pd.Timedelta(days=retrain_days)).date()),
            "retrain_days": int(retrain_days), "n_origins": int(len(origins)),
            "history_days_at_first_origin": int((origins[0] - first_report).days)}


def build_spec(name: str, raw_csv: Path, output_dir: Path, *, reference_col: str = "reference_date",
               report_col: str = "report_date", value_cols: tuple[str, ...] = ("value",),
               geo_col: str | None = "geo_value", value_type: str = "count",
               smoothed: bool | None = None, target_lag: int | None = None,
               temporal_resol: str | None = None, lag_terms: tuple[int, ...] | None = None,
               training_days: int | None = None, testing_days: int | None = None,
               first_origin_date: str | None = None,
               start_date: str | None = None,
               end_date: str | None = None, triangle_format: str = "parquet",
               min_location_rows: int = DEFAULT_MIN_LOCATION_ROWS,
               weekday_groups: dict | None = None,
               confirm_target_lag: bool = False,
               as_of_date: str | None = None,
               smoothed_target: bool | None = None,
               genuine_training: bool = True,
               genuine_testing: bool = True) -> PreparationSpec:
    """Diagnose the archive and resolve every preprocessing argument.

    A keyword left at ``None`` is diagnosed from the archive; a keyword given
    explicitly is recorded as an override and used as given. ``target_lag``
    is intentionally not diagnosed: the archive supplies an evidence range,
    then preparation pauses until the user supplies the final value.
    """
    if value_type not in VALUE_TYPES:
        raise PreparationError(f"value_type must be one of {VALUE_TYPES}, got {value_type!r}")
    for flag, value in (("genuine_training", genuine_training), ("genuine_testing", genuine_testing)):
        if not isinstance(value, bool):
            raise PreparationError(f"{flag} must be true or false, got {value!r}")
    if value_type == "fraction" and len(value_cols) not in (1, 2):
        raise PreparationError(
            "a fraction signal needs either one precomputed rate column or two columns, "
            "numerator then denominator: --value-col rate, or "
            "--value-col numerator --value-col denominator")
    if value_type == "count" and len(value_cols) != 1:
        raise PreparationError("a count signal takes exactly one --value-col")
    if (start_date is None) != (end_date is None):
        raise PreparationError("--start-date and --end-date must be given together")
    if target_lag is not None and not 1 <= int(target_lag) <= MAX_OBSERVED_LAG_DAYS:
        raise PreparationError(
            f"target_lag must be between 1 and {MAX_OBSERVED_LAG_DAYS} days; "
            "later raw observations are outside the modelled revision horizon.")

    diagnosis = diagnose_archive(raw_csv, geo_col=geo_col, reference_col=reference_col,
                                 report_col=report_col, value_cols=value_cols,
                                 value_type=value_type, user_target_lag=target_lag,
                                 as_of_date=as_of_date)
    overrides = []
    if target_lag is not None:
        overrides.append("target_lag")
    resolved = diagnosis.get("resolved_target_lag")
    if resolved is None:
        bounds = diagnosis.get("target_lag_suggested_range") or {}
        raise PreparationError(
            "target_lag requires a user decision before preprocessing. The diagnostic "
            f"range is {bounds.get('lower')} to {bounds.get('upper')} days (90% within "
            "10% to 1% relative error). Re-run with --target-lag L.")
    resolved_lag = int(resolved)
    if not 1 <= resolved_lag <= MAX_OBSERVED_LAG_DAYS:
        raise PreparationError(
            f"target_lag must be between 1 and {MAX_OBSERVED_LAG_DAYS} days; "
            "later raw observations are outside the modelled revision horizon.")
    if confirm_target_lag:
        if target_lag is None:
            raise PreparationError(
                "confirm_target_lag was set without a target_lag to confirm; there is nothing "
                "to confirm when the target lag comes from the diagnosis.")
        overrides.append("target_lag_confirmed")

    if temporal_resol is None:
        temporal_resol = diagnosis["temporal_resolution"]
    else:
        overrides.append("temporal_resol")
    if temporal_resol not in ("daily", "weekly"):
        raise PreparationError(f"temporal_resol must be 'daily' or 'weekly', got {temporal_resol!r}")

    # Reference dates and releases have separate frequencies: a stream can have
    # weekly reference dates and daily releases, and stays in DelphiRF's daily
    # mode to keep its daily releases. DelphiRF then lays reference dates out
    # day by day; the days absent from the archive are removed after
    # preprocessing, and nothing may be computed from them.
    cadence = diagnosis.get("reference_axis_cadence_days")
    reference_step = max(1, int(round(cadence))) if cadence else 1
    daily_reference = reference_step == 1

    if lag_terms is None:
        diagnosed_lags = [int(t) for t in diagnosis.get("reference_axis_feature_lags") or []]
        # A stream with no diagnosable reference-axis cadence still needs
        # lagged terms; fall back to the cadence implied by the resolution.
        lag_terms = tuple(diagnosed_lags) or ((1, 7) if temporal_resol == "daily" else (7, 14))
    else:
        overrides.append("lag_terms")
    if temporal_resol == "daily" and any(int(t) % reference_step for t in lag_terms):
        raise PreparationError(
            f"lag terms {list(lag_terms)} must be multiples of the reference-date spacing "
            f"({reference_step} days): a lagged feature would otherwise read a reference date "
            "that is not in the archive")

    if smoothed is None:
        # Daily reference dates: 7-day trailing averages of the values (the
        # Delphi-RF model of Ch. 2). Reference dates spaced further apart: a
        # 7-day window holds a single reference date, so no further smoothing;
        # DelphiRF then uses the values themselves as the value features.
        # (DelphiRF's weekly mode never averages, so its default is unchanged.)
        smoothed = temporal_resol == "daily" and not daily_reference
    else:
        overrides.append("smoothed")
        if temporal_resol == "daily" and not daily_reference and not smoothed:
            raise PreparationError(
                f"7-day smoothing needs daily reference dates; this archive's reference dates "
                f"are {reference_step} days apart, so a 7-day window would average one real "
                "reference date with days that are not in the archive. Leave smoothing off "
                "for this archive (do not pass --no-smoothed).")
    # The target's own smoothing level. A smoothed target is DelphiRF's 7-day
    # value at lag L, (1/7) sum_v Y_{t-v, L+v}; for a fraction, the 7-day mean
    # numerator over the 7-day mean denominator. It is reported by t + L, like
    # the unsmoothed Y_itL. An already-smoothed archive has only that target.
    if smoothed_target is None:
        smoothed_target = False
    else:
        overrides.append("smoothed_target")
    target_is_smoothed = bool(smoothed) or bool(smoothed_target)

    if training_days is None:
        training_days = int(diagnosis.get("training_window_days") or DEFAULT_TRAINING_WINDOW_DAYS)
    else:
        overrides.append("training_days")

    # ``weekday_groups`` is the user's grouping in ISO weekday numbers, or None
    # to use the groups detected from the distinct report dates.
    weekday_groups, weekday_groups_source = resolve_weekday_groups(
        weekday_groups, diagnosis.get("report_axis_weekdays") or {}, temporal_resol)
    if weekday_groups_source == "manual":
        overrides.append("weekday_groups")

    # Every test origin is a retraining origin, so ``testing_days`` IS the
    # retraining interval. Left unset it follows the diagnosed cadence.
    if testing_days is None:
        testing_days = DEFAULT_RETRAIN_DAYS[temporal_resol]
    else:
        overrides.append("retrain_days")

    experiment_end_date = None
    if start_date is None:
        if first_origin_date is not None:
            overrides.append("first_origin_date")
        calendar = retraining_calendar(diagnosis["report_date_min"], diagnosis["report_date_max"],
                                       testing_days, first_origin=first_origin_date)
        start_date, end_date = calendar["first_origin"], calendar["last_origin"]
        experiment_end_date = calendar["experiment_end_date"]
    else:
        if first_origin_date is not None:
            raise PreparationError(
                "give either first_origin_date or an explicit start_date/end_date pair, not "
                "both: they name the same thing.")
        overrides.append("experiment_calendar")

    return PreparationSpec(
        name=name, raw_csv=Path(raw_csv).resolve(), output_dir=Path(output_dir).resolve(),
        reference_col=reference_col, report_col=report_col, value_cols=tuple(value_cols),
        geo_col=geo_col, constant_geo="single_location", value_type=value_type,
        smoothed=bool(smoothed), temporal_resol=temporal_resol, ref_lag=resolved_lag,
        max_observed_lag_days=MAX_OBSERVED_LAG_DAYS,
        lag_terms=tuple(int(t) for t in lag_terms), weekday_groups=dict(weekday_groups),
        weekday_groups_source=weekday_groups_source,
        training_days=int(training_days), testing_days=int(testing_days),
        target_column=TARGET_COLUMNS[target_is_smoothed],
        start_date=start_date, end_date=end_date,
        experiment_end_date=experiment_end_date, triangle_format=triangle_format,
        min_location_rows=int(min_location_rows),
        as_of_date=None if as_of_date is None else str(pd.Timestamp(as_of_date).date()),
        genuine_training=genuine_training, genuine_testing=genuine_testing,
        overrides=tuple(dict.fromkeys(overrides)),
        diagnosis={**diagnosis, "user_confirmed_target_lag": bool(confirm_target_lag),
                   "final_target_lag": resolved_lag})


def rscript_executable() -> str:
    """The Rscript to call, or raise with installation guidance."""
    executable = os.environ.get("AUTODELPHIRF_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        raise PreparationError(
            "Rscript was not found on PATH. AutoDelphiRF calls DelphiRF (an R package) to build a "
            "prepared triangle from a raw archive:\n"
            "  1. install R          https://cran.r-project.org\n"
            "  2. install R packages as shown in the AutoDelphiRF README (jsonlite, "
            "data.table, arrow, and DelphiRF with its dependencies)\n"
            "Set AUTODELPHIRF_RSCRIPT to use a specific Rscript. If you already have a prepared "
            "triangle, skip this stage and run `autodelphirf run --config <dataset>.json` instead.")
    return executable


def run_r_bridge(spec: PreparationSpec, on_output=None, should_stop=None) -> None:
    """Invoke ``prepare_triangle.R`` on ``spec``, streaming its output.

    With no ``on_output`` the R process inherits this process's stdout and
    stderr, so its progress appears live in the terminal -- which is what a
    CLI user wants. A caller that has nowhere for those file descriptors to
    go (the web UI, which needs the lines as data) passes a callback and
    receives them one line at a time instead.
    """
    executable = rscript_executable()
    script = resource_path("prepare_triangle.R")
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec.to_r_spec(), handle, indent=2)
        spec_path = Path(handle.name)
    command = [executable, str(script), str(spec_path)]
    try:
        if on_output is None:
            returncode = subprocess.run(command, check=False).returncode
        else:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    if should_stop:
                        should_stop()
                    on_output(line.rstrip("\n"))
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
    finally:
        spec_path.unlink(missing_ok=True)
    if returncode != 0:
        raise PreparationError(
            f"DelphiRF preprocessing failed (Rscript exit {returncode}). The R error is "
            + ("in the log above." if on_output else "printed above."))


def write_dataset_config(spec: PreparationSpec, config_path: Path,
                         prediction_layers: tuple[str, ...]) -> Path:
    """Write the dataset JSON naming the triangle this spec just built.

    Paths are written relative to the JSON itself, matching how
    :func:`autodelphirf.core.config.load_dataset_config` resolves them, so the
    whole output directory can be moved or shared without editing.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    def relative(target: Path) -> str:
        target = Path(target).resolve()
        try:
            return os.path.relpath(target, config_path.parent.resolve())
        except ValueError:  # different drive on Windows; an absolute path still loads
            return str(target)

    payload = {
        "_generated_by": "autodelphirf prepare",
        "name": spec.name,
        "prepared_dir": relative(spec.output_dir),
        "output_dir": relative(spec.output_dir.parent / "results"),
        "schedule_file": "test_dates.csv",
        "genuine_training": spec.genuine_training,
        "genuine_testing": spec.genuine_testing,
        "model_methods": list(prediction_layers),
        "prediction_layers": list(prediction_layers),
        "method_params": {},
        "comparator_file": None,
        "comparator_methods": {},
        "reliability_reference_file": None,
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n")
    return config_path


def _read_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def _archive_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"archive": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def prepare_dataset(spec: PreparationSpec, *, prediction_layers: tuple[str, ...],
                    config_path: Path | None = None, on_output=None, should_stop=None) -> Path:
    """Build the triangle, record provenance, and write the dataset config.

    Returns the path of the dataset JSON, which is what ``autodelphirf run``
    consumes. ``on_output`` is forwarded to :func:`run_r_bridge`.
    """
    if should_stop is None:
        run_r_bridge(spec, on_output=on_output)
    else:
        run_r_bridge(spec, on_output=on_output, should_stop=should_stop)
    provenance = {
        "name": spec.name,
        "raw_archive": str(spec.raw_csv),
        # Recorded once here, so rerunning the prepared dataset never needs to
        # reread the raw archive to document which input produced it.
        "raw_archive_state": _archive_identity(spec.raw_csv),
        "resolved_preprocessing": spec.to_r_spec(),
        "user_overrides": list(spec.overrides),
        "case_selection": {"genuine_training": spec.genuine_training,
                           "genuine_testing": spec.genuine_testing},
        "diagnosis": spec.diagnosis,
        # Both stages' cleaning audits. Their retained rows differ by scope
        # (lag bound, as-of date, experiment window), which each records.
        "raw_cleaning_audits": {
            "pre_diagnosis": spec.diagnosis.get("raw_cleaning_audit"),
            "feature_preparation": _read_json(spec.output_dir / "raw_cleaning_audit.json")},
    }
    (spec.output_dir / "preparation.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n")
    config_path = config_path or (spec.output_dir.parent / f"{spec.name}.json")
    return write_dataset_config(spec, config_path, prediction_layers)
