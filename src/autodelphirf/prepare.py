"""Convert a raw revision archive into prepared AutoDelphiRF inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd

from .ingest import DEFAULT_TRAINING_WINDOW_DAYS, diagnose_raw_archive
from .resources import resource_path

#: Fraction signals are the one case where the archive carries two value
#: columns (numerator and denominator) and DelphiRF models their ratio.
VALUE_TYPES = ("count", "fraction")

#: Emitted by ``data_preprocessing`` for a smoothed build vs. an unsmoothed
#: one. The schedule names the one the run evaluates against.
TARGET_COLUMNS = {True: "log_value_target_7dav", False: "log_value_target"}

#: Weekday one-hot groups, by diagnosed reporting cadence. A weekly stream has
#: no within-week reporting structure to encode, so it gets none; a daily
#: stream gets suitable weekday groups for its reporting cadence.
DEFAULT_WEEKDAY_GROUPS = {
    "daily": {"Mon": ["Mon"], "Weekends": ["Sat", "Sun"]},
    "weekly": {},
}

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
    lag_terms: tuple[int, ...]
    onehot_weekdays: dict
    training_days: int
    testing_days: int
    lower: int
    upper: int
    target_column: str
    start_date: str | None
    end_date: str | None
    experiment_end_date: str | None
    triangle_format: str
    min_location_rows: int
    delphirf_dir: str | None
    overrides: tuple[str, ...] = ()
    diagnosis: dict = field(default_factory=dict)

    def to_r_spec(self) -> dict:
        """The JSON handed to the R bridge (paths as strings, no provenance)."""
        return {
            "name": self.name, "raw_csv": str(self.raw_csv), "output_dir": str(self.output_dir),
            "reference_col": self.reference_col, "report_col": self.report_col,
            "value_cols": list(self.value_cols), "geo_col": self.geo_col,
            "constant_geo": self.constant_geo, "value_type": self.value_type,
            "smoothed": self.smoothed, "temporal_resol": self.temporal_resol,
            "ref_lag": int(self.ref_lag), "lag_terms": [int(t) for t in self.lag_terms],
            "onehot_weekdays": self.onehot_weekdays, "training_days": int(self.training_days),
            "testing_days": int(self.testing_days), "lower": int(self.lower),
            "upper": int(self.upper), "target_column": self.target_column,
            "start_date": self.start_date, "end_date": self.end_date,
            "experiment_end_date": self.experiment_end_date,
            "triangle_format": self.triangle_format,
            "min_location_rows": int(self.min_location_rows),
            "delphirf_dir": self.delphirf_dir,
        }


def read_archive(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a raw archive from CSV (optionally gzipped) or Parquet."""
    path = Path(path)
    if not path.is_file():
        raise PreparationError(f"raw archive not found: {path}")
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path, columns=columns)
        except ImportError as error:
            raise PreparationError(
                f"reading {path.name} needs pyarrow: pip install 'autodelphirf[parquet]', or "
                "supply the archive as CSV.") from error
    frame = pd.read_csv(path, usecols=columns, nrows=DIAGNOSIS_ROW_LIMIT + 1)
    if len(frame) > DIAGNOSIS_ROW_LIMIT:
        raise PreparationError(
            f"{path.name} has more than {DIAGNOSIS_ROW_LIMIT:,} rows. Diagnosis needs the whole "
            "revision history, so it is not sampled; preprocess or partition the archive first.")
    return frame


def diagnose_archive(raw_csv: Path, *, geo_col: str | None, reference_col: str,
                     report_col: str, value_cols: tuple[str, ...], value_type: str,
                     user_target_lag: int | None = None) -> dict:
    """Diagnose a raw archive, returning :func:`diagnose_raw_archive`'s report.

    A fraction signal is diagnosed on the ratio it models, not on its
    numerator: the revision behaviour of a rate is the thing RevRoute routes
    on, and numerator and denominator revise on their own schedules.
    """
    needed = [c for c in (geo_col, reference_col, report_col, *value_cols) if c]
    frame = read_archive(raw_csv, columns=None)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise PreparationError(
            f"raw archive is missing column(s) {missing}. Present: "
            f"{sorted(frame.columns)[:40]}. Name your columns with --reference-col / "
            "--report-col / --value-col / --geo-col.")
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
    return diagnose_raw_archive(work, user_target_lag=user_target_lag).to_dict()


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
               lower: int = 0, upper: int = 0, start_date: str | None = None,
               end_date: str | None = None, triangle_format: str = "parquet",
               min_location_rows: int = DEFAULT_MIN_LOCATION_ROWS,
               weekday_groups: dict | None = None, confirm_target_lag: bool = False,
               delphirf_dir: str | None = None) -> PreparationSpec:
    """Diagnose the archive and resolve every preprocessing argument.

    A keyword left at ``None`` is diagnosed from the archive; a keyword given
    explicitly is recorded as an override and used as given. ``target_lag``
    is the one exception worth stating: when the user's value disagrees with
    the diagnosed one, :func:`autodelphirf.ingest.resolve_target_lag`
    decides, and its prompt is carried into ``preparation.json`` -- no
    horizon is adopted without the recommendation having been recorded.
    """
    if value_type not in VALUE_TYPES:
        raise PreparationError(f"value_type must be one of {VALUE_TYPES}, got {value_type!r}")
    if value_type == "fraction" and len(value_cols) != 2:
        raise PreparationError(
            "a fraction signal needs exactly two value columns, numerator then denominator: "
            "--value-col numerator --value-col denominator")
    if value_type == "count" and len(value_cols) != 1:
        raise PreparationError("a count signal takes exactly one --value-col")
    if (start_date is None) != (end_date is None):
        raise PreparationError("--start-date and --end-date must be given together")

    diagnosis = diagnose_archive(raw_csv, geo_col=geo_col, reference_col=reference_col,
                                 report_col=report_col, value_cols=value_cols,
                                 value_type=value_type, user_target_lag=target_lag)
    overrides = []
    if target_lag is not None:
        overrides.append("target_lag")
    # resolve_target_lag already reconciled a user value with the suggestion,
    # keeping the safer (longer) horizon by default and recording the
    # disagreement for confirmation. ``confirm_target_lag`` is the user having
    # ANSWERED that confirmation: the prompt says the default applies "if you
    # do not respond", so a response has to be able to win. Without it a user
    # who deliberately wants a shorter horizon than the diagnosis has no way
    # to say so explicitly.
    resolved_lag = int(diagnosis["resolved_target_lag"])
    if confirm_target_lag:
        if target_lag is None:
            raise PreparationError(
                "confirm_target_lag was set without a target_lag to confirm; there is nothing "
                "to confirm when the horizon comes from the diagnosis.")
        resolved_lag = int(target_lag)
        overrides.append("target_lag_confirmed")

    if temporal_resol is None:
        temporal_resol = diagnosis["temporal_resolution"]
    else:
        overrides.append("temporal_resol")
    if temporal_resol not in ("daily", "weekly"):
        raise PreparationError(f"temporal_resol must be 'daily' or 'weekly', got {temporal_resol!r}")

    if lag_terms is None:
        diagnosed_lags = [int(t) for t in diagnosis.get("reference_axis_feature_lags") or []]
        # A stream with no diagnosable reference-axis cadence still needs
        # lagged terms; fall back to the cadence implied by the resolution.
        lag_terms = tuple(diagnosed_lags) or ((1, 7) if temporal_resol == "daily" else (7, 14))
    else:
        overrides.append("lag_terms")

    if smoothed is None:
        # Smoothing is a property of the stream, not a free choice: a daily
        # stream is modelled on its 7-day trailing average, a weekly stream is
        # already an aggregate.
        smoothed = temporal_resol == "daily"
    else:
        overrides.append("smoothed")

    if training_days is None:
        training_days = int(diagnosis.get("training_window_days") or DEFAULT_TRAINING_WINDOW_DAYS)
    else:
        overrides.append("training_days")

    if weekday_groups is None:
        weekday_groups = DEFAULT_WEEKDAY_GROUPS[temporal_resol]
    else:
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
        lag_terms=tuple(int(t) for t in lag_terms), onehot_weekdays=dict(weekday_groups),
        training_days=int(training_days), testing_days=int(testing_days),
        lower=int(lower), upper=int(upper), target_column=TARGET_COLUMNS[bool(smoothed)],
        start_date=start_date, end_date=end_date,
        experiment_end_date=experiment_end_date, triangle_format=triangle_format,
        min_location_rows=int(min_location_rows), delphirf_dir=delphirf_dir,
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
            "  2. install DelphiRF   Rscript -e 'remotes::install_github(\"cmu-delphi/DelphiRF\")'\n"
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
    :func:`autodelphirf.config.load_dataset_config` resolves them, so the
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
        "include_genuine_events_only": True,
        "model_methods": list(prediction_layers),
        "prediction_layers": list(prediction_layers),
        "method_params": {},
        "comparator_file": None,
        "comparator_methods": {},
        "reliability_reference_file": None,
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n")
    return config_path


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
        "resolved_preprocessing": spec.to_r_spec(),
        "user_overrides": list(spec.overrides),
        "diagnosis": spec.diagnosis,
    }
    (spec.output_dir / "preparation.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n")
    config_path = config_path or (spec.output_dir.parent / f"{spec.name}.json")
    return write_dataset_config(spec, config_path, prediction_layers)
