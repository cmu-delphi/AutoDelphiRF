"""A local web UI: drop an archive in, answer the questions, get a report.

``autodelphirf web`` starts a small HTTP server on localhost and opens a page that
walks one dataset through the whole workflow. It exists because the workflow
has genuine *questions* in it -- above all the target lag L, where
``ingest.resolve_target_lag`` composes a prompt that says the default applies
"if you do not respond", and until now there was nowhere to respond. On the
command line that prompt is a line of text scrolling past. Here it is a
question with buttons, shown next to the completion curve the recommendation
was derived from.

The server runs the real pipeline in this process: it calls
``prepare.diagnose_archive``, then ``prepare.prepare_dataset`` (which shells
out to DelphiRF), then ``pipeline.run_pipeline``. Nothing is uploaded
anywhere; the archive is copied into a working directory on this machine and
the generated report is served from there.

Security. This server writes files and runs subprocesses on behalf of
whoever can reach it, so it is deliberately not a general-purpose service:

  * it binds 127.0.0.1 unless a host is passed explicitly, and warns loudly
    if one is;
  * every API call must carry the session token that was minted at startup
    and embedded in the page, so another program on the machine cannot drive
    it just by knowing the port;
  * the ``Host`` header is checked against the address actually bound, which
    is what stops a DNS-rebinding page in the user's browser from using the
    token-less same-origin path to reach it;
  * uploads are capped and filenames are reduced to a safe stem.

It is a single-session tool by design: one archive, one job at a time. That
is what a person at a laptop needs, and it keeps the state model small enough
to reason about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

import pandas as pd

# The report stage draws figures with matplotlib, and this server runs it on a
# background thread. A GUI backend -- MacOSX is the default on macOS, and Tk
# elsewhere -- must do its drawing on the main thread, so from a worker thread
# it does not raise: it blocks forever, leaving a run that has written its
# tables sitting at 0 figures and no report.html. Agg draws to a file and needs
# no main thread, which is all this server ever wants. Set before anything
# imports pyplot, and only defaulted, so a user who deliberately chose a
# backend keeps it.
os.environ.setdefault("MPLBACKEND", "Agg")

from .prepare import (DEFAULT_FIRST_ORIGIN_OFFSET_DAYS, DEFAULT_MIN_LOCATION_ROWS,
                      DEFAULT_RETRAIN_DAYS, PreparationError, build_spec, diagnose_archive,
                      prepare_dataset, read_archive, retraining_calendar)
from .registry import DEFAULT_LAYERS, PREDICTION_LAYERS
from .resources import resource_path

#: Largest archive accepted through the browser. Bigger archives are a
#: command-line job: the browser would have to hold the whole file in memory
#: to send it, and `autodelphirf run --archive` reads it straight off disk.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

#: Archive formats the preparation stage can read.
ALLOWED_SUFFIXES = (".csv", ".csv.gz", ".parquet")

#: Column-name guesses, by semantic role. First match wins, so the exact
#: canonical name is checked before the looser patterns.
COLUMN_HINTS = {
    "geo": ("geo_value", "geo", "location", "location_id", "state", "region", "fips"),
    "reference": ("reference_date", "reference_time", "time_value", "date", "event_date",
                  "refd", "ref_date", "epiweek"),
    "report": ("report_date", "report_time", "issue_date", "issue", "as_of", "asof",
               "vintage", "publication_date"),
    "value": ("value", "value_raw", "count", "cases", "admissions", "n", "num", "total"),
}

#: Layers offered in the UI, with the one-line description shown beside each.
LAYER_DESCRIPTIONS = {
    "baseline_null": "Uses the latest reported value as the forecast. This is the reference for comparing other methods.",
    "red": ("Matches historical observations with similar revision paths and reporting stages, "
            "then uses their remaining changes to forecast the final value and uncertainty."),
    "delphirf": "Groups similar location and reporting-lag combinations, then fits DelphiRF.",
    "naive_delphirf": "Separate local DelphiRF fits without cross-location pooling.",
    "similarity_weighted_delphirf": "DelphiRF with its prospective similarity weights.",
    "global_delphirf": "One global DelphiRF fit with location fixed effects.",
}

LAYER_LABELS = {
    "baseline_null": "Latest reported value",
    "red": "Revision-pattern matching",
    "delphirf": "RevRoute pooling + DelphiRF",
    "naive_delphirf": "Separate DelphiRF by location",
    "similarity_weighted_delphirf": "Similarity-weighted DelphiRF",
    "global_delphirf": "One DelphiRF model for all locations",
}



#: The replay prints one of these per retraining origin it finishes. Parsed
#: rather than plumbed through a callback because ``run_pipeline`` reports on
#: stdout, which this server is already capturing line by line.
FOLD_PROGRESS = re.compile(
    r"(?:Evaluated retraining date|(?:revroute v8 )?replay: fold) (\d+)(?: of |/)(\d+)")

#: Layers that fit forests and cost substantially more than RED alone. Shown
#: as a warning beside the checkbox rather than hidden.
EXPENSIVE_LAYERS = frozenset({"delphirf", "naive_delphirf",
                              "similarity_weighted_delphirf", "global_delphirf"})

DELPHIRF_LAYERS = EXPENSIVE_LAYERS


class RunStopped(RuntimeError):
    """Raised at a safe checkpoint after the user requests Stop."""


def local_delphirf_dir() -> str | None:
    """Find the explicitly configured or adjacent DelphiRF source checkout."""
    configured = os.environ.get("AUTODELPHIRF_DELPHIRF_DIR")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        return str(candidate) if (candidate / "DESCRIPTION").is_file() else None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "DelphiRF"
        if (candidate / "DESCRIPTION").is_file():
            return str(candidate)
    return None


def safe_stem(name: str) -> str:
    """Reduce an uploaded filename to a safe dataset stem.

    The name reaches the filesystem and the R bridge's command line, so it is
    rebuilt from scratch out of known-good characters rather than filtered.
    """
    base = Path(unquote(name or "")).name
    for suffix in (".csv.gz", ".csv", ".parquet"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    return cleaned[:64] or "dataset"


def archive_suffix(name: str) -> str:
    """The accepted suffix of an uploaded filename, or raise."""
    lowered = (name or "").lower()
    for suffix in ALLOWED_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    raise PreparationError(
        f"unsupported file type: {name!r}. Drop a .csv, .csv.gz, or .parquet revision archive.")


def guess_columns(columns: list[str]) -> dict:
    """Guess which column plays which role, for pre-filling the mapping form."""
    lowered = {str(column).lower(): str(column) for column in columns}
    guesses = {}
    for role, hints in COLUMN_HINTS.items():
        for hint in hints:
            if hint in lowered:
                guesses[role] = lowered[hint]
                break
        else:
            # Fall back to a substring match, which catches names like
            # "covid_value_total" that no exact hint lists.
            for hint in hints:
                match = next((original for lower, original in lowered.items() if hint in lower),
                             None)
                if match:
                    guesses[role] = match
                    break
    return guesses


def optional_int(value, field: str) -> int | None:
    """Read a whole number out of a JSON request field, or ``None`` if unset.

    The page is the only intended caller, but a request field is still
    untrusted input: a value that is not a number has to come back as the
    400 that names the field, not as a TypeError from deep inside the
    pipeline.
    """
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PreparationError(f"{field} must be a whole number of days, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise PreparationError(
            f"{field} must be a whole number of days, got {value!r}") from error


def environment_report() -> dict:
    """Whether R and a usable DelphiRF are present, for the page to show up front.

    Checked at startup rather than at the moment preprocessing runs, so a
    user missing a prerequisite learns it before uploading anything.
    """
    executable = os.environ.get("AUTODELPHIRF_RSCRIPT") or shutil.which("Rscript")
    report = {"rscript": executable, "preprocessing": False, "delphirf": False,
              "delphirf_dir": None, "detail": ""}
    if not executable:
        report["detail"] = ("Rscript was not found. Diagnosis works without it, but building a "
                            "prepared triangle needs R and DelphiRF.")
        return report
    source_dir = local_delphirf_dir()
    if source_dir:
        quoted = json.dumps(source_dir)
        script = (f'suppressMessages(pkgload::load_all({quoted}, quiet=TRUE)); '
                  'cat("onehot_weekdays" %in% names(formals(data_preprocessing)))')
    else:
        script = ('suppressMessages(library(DelphiRF)); '
                  'cat("onehot_weekdays" %in% names(formals(DelphiRF::data_preprocessing)))')
    try:
        result = subprocess.run([executable, "-e", script], capture_output=True,
                                text=True, timeout=180)
    except (subprocess.SubprocessError, OSError) as error:
        report["detail"] = f"Rscript found but could not be run: {error}"
        return report
    if result.returncode != 0:
        report["detail"] = ("DelphiRF could not be loaded. Install it with "
                            "the released DelphiRF package.")
        return report
    if "TRUE" not in result.stdout:
        report["detail"] = ("The installed DelphiRF predates the arguments AutoDelphiRF needs. "
                            "Update the installed DelphiRF package.")
        return report
    report["preprocessing"] = True
    report["delphirf_dir"] = source_dir
    # Preprocessing may load an adjacent source checkout through pkgload, while
    # model training deliberately calls the installed package. Check those two
    # capabilities independently so keeping the transferable DelphiRF source
    # beside AutoDelphiRF does not disable a compatible installed copy.
    installed_script = (
        'suppressMessages(library(DelphiRF)); '
        'cat("model_backend" %in% names(formals(DelphiRF::revision_forecast)) && '
        'exists("fit_quantreg_lasso", envir=asNamespace("DelphiRF"), inherits=FALSE))'
    )
    try:
        installed = subprocess.run([executable, "-e", installed_script], capture_output=True,
                                   text=True, timeout=180)
        report["delphirf"] = installed.returncode == 0 and "TRUE" in installed.stdout
    except (subprocess.SubprocessError, OSError):
        report["delphirf"] = False
    if source_dir:
        report["detail"] = f"compatible local DelphiRF preprocessing ({source_dir})"
        if report["delphirf"]:
            report["detail"] += "; compatible installed DelphiRF model backend"
    else:
        report["detail"] = "installed DelphiRF"
    return report


#: Tables the results view reads out of a finished run's output directory.
#: Every one is optional -- which exist depends on which layers ran -- so a
#: missing file means "that layer was not requested", never an error.
RESULT_TABLES = {
    "headline": "v1_pairwise_point.csv",
    "by_origin": "report/tables/comparison_by_origin.csv",
    "availability": "report/tables/model_availability.csv",
    "revroute_pools": "revroute_pools.csv",
    "revroute_pool_profile": "report/tables/rr_delphirf_cluster_profile.csv",
}

#: Rows of any one table sent to the page. A run with hundreds of retraining
#: origins produces a pool profile with thousands of rows, and the browser
#: does not need them all to show the shape of the result; the CSVs on disk
#: remain complete either way.
MAX_RESULT_ROWS = 2000


def results_summary(output_dir) -> dict:
    """Read the finished run's headline numbers and pooling summaries.

    Read back off disk rather than held from the run, so the page shows what
    was actually written -- and so reloading the browser after a run still
    produces the results view.
    """
    root = Path(output_dir)
    tables, truncated = {}, []
    for key, relative in RESULT_TABLES.items():
        path = root / relative
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if key == "headline" and "subset" in frame:
            # The per-subset breakdown is in the report; the page wants the
            # one line per comparison that answers "did it help overall".
            frame = frame[frame.subset.eq("overall")]
        if len(frame) > MAX_RESULT_ROWS:
            truncated.append(key)
            frame = frame.head(MAX_RESULT_ROWS)
        tables[key] = json.loads(frame.to_json(orient="records", date_format="iso"))
    origins = tables.get("by_origin") or []
    return {"output_dir": str(root), "tables": tables, "truncated": truncated,
            "n_origins": len({row["cutoff"] for row in origins}),
            "report_available": (root / "report" / "report.html").is_file()}


@dataclass
class Job:
    """One background run, and everything the page polls for."""
    state: str = "idle"          # idle | running | done | failed
    stage: str = ""
    log: list = field(default_factory=list)
    error: str = ""
    report_available: bool = False
    output_dir: str = ""
    started: float = 0.0
    finished: float = 0.0
    origins_done: int = 0
    origins_total: int = 0
    #: Origins the schedule asked for. The replay evaluates only those with
    #: eligible cases, which near the end of an archive is fewer: the target
    #: needs L days of follow-up that the newest origins do not have yet.
    origins_scheduled: int = 0
    replay_started: float = 0.0
    stop_requested: bool = False

    def append(self, line: str) -> None:
        # Bounded: a long replay prints per-origin progress and the page only
        # ever renders the tail.
        self.log.append(line)
        del self.log[:-500]
        match = FOLD_PROGRESS.search(line)
        if match:
            self.origins_done, self.origins_total = int(match[1]), int(match[2])

    def snapshot(self) -> dict:
        elapsed = ((self.finished or time.time()) - self.started) if self.started else 0.0
        done, total = self.origins_done, self.origins_total
        # Extrapolate from the origins finished so far, timed from the start of
        # the REPLAY: building the triangle happens first and takes as long as
        # it takes, so counting it would inflate every estimate. Only offered
        # once a couple of origins are done, rather than as a wild first guess.
        replay_elapsed = (time.time() - self.replay_started) if self.replay_started else 0.0
        remaining = (round(replay_elapsed / done * (total - done), 1)
                     if self.state == "running" and done >= 2 and total else None)
        return {"state": self.state, "stage": self.stage, "log": list(self.log),
                "error": self.error, "report_available": self.report_available,
                "output_dir": self.output_dir, "elapsed_seconds": round(elapsed, 1),
                "origins_done": done, "origins_total": total,
                "origins_scheduled": self.origins_scheduled,
                "remaining_seconds": remaining}


class Session:
    """The one dataset this server instance is working on."""

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.archive_path: Path | None = None
        self.dataset_name: str = "dataset"
        self.columns: list[str] = []
        self.diagnosis: dict | None = None
        self.job = Job()
        self.report_root: Path | None = None
        self.last_request: dict | None = None

    def snapshot(self) -> dict:
        """Run status plus enough in-memory workflow state to restore a reload."""
        status = self.job.snapshot()
        if self.archive_path is not None and self.archive_path.is_file():
            status["session"] = {
                "archive": {
                    "filename": self.archive_path.name,
                    "size_bytes": self.archive_path.stat().st_size,
                    "columns": self.columns,
                    "guesses": guess_columns(self.columns),
                },
                "diagnosis": self.diagnosis,
                "last_request": self.last_request,
            }
        return status

    # -- upload ------------------------------------------------------------

    def store_archive(self, filename: str, payload: bytes) -> dict:
        suffix = archive_suffix(filename)
        stem = safe_stem(filename)
        destination = self.work_dir / f"{stem}{suffix}"
        destination.write_bytes(payload)
        # Read the header only: a multi-hundred-megabyte archive should not be
        # parsed twice just to populate the column dropdowns.
        try:
            preview = read_archive(destination).head(5)
        except Exception as error:                      # noqa: BLE001 - reported to the page
            destination.unlink(missing_ok=True)
            raise PreparationError(f"could not read {filename}: {error}") from error
        self.archive_path = destination
        self.dataset_name = stem
        self.columns = [str(column) for column in preview.columns]
        self.diagnosis = None
        self.job = Job()
        self.report_root = None
        self.last_request = None
        return {
            "name": stem,
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "columns": self.columns,
            "guesses": guess_columns(self.columns),
            "preview": json.loads(preview.to_json(orient="records", date_format="iso")),
        }

    # -- diagnosis ---------------------------------------------------------

    def diagnose(self, request: dict) -> dict:
        if self.archive_path is None:
            raise PreparationError("no archive has been uploaded yet")
        value_cols = tuple(request.get("value_cols") or ())
        if not value_cols:
            raise PreparationError("choose at least one value column")
        self.diagnosis = diagnose_archive(
            self.archive_path,
            geo_col=request.get("geo_col") or None,
            reference_col=request["reference_col"], report_col=request["report_col"],
            value_cols=value_cols, value_type=request.get("value_type", "count"),
            user_target_lag=optional_int(request.get("target_lag"), "target_lag"))
        # The retraining schedule follows from the same diagnosis, so the page
        # is handed the default it should prefill rather than rebuilding the
        # cadence rule in JavaScript.
        self.diagnosis["retraining"] = self.schedule_preview({})
        return self.diagnosis

    # -- retraining schedule -----------------------------------------------

    def schedule_preview(self, request: dict) -> dict:
        """What calendar a given cadence and first origin would produce.

        The page calls this as the two fields change, so the origin count it
        shows before the run is the one the run will actually use: the answer
        comes from the same :func:`prepare.retraining_calendar` that
        ``build_spec`` lays the schedule out with.
        """
        if self.diagnosis is None:
            raise PreparationError("diagnose the archive before choosing a retraining schedule")
        default_days = DEFAULT_RETRAIN_DAYS[self.diagnosis["temporal_resolution"]]
        retrain_days = optional_int(request.get("retrain_days"), "retrain_days") or default_days
        calendar = retraining_calendar(
            self.diagnosis["report_date_min"], self.diagnosis["report_date_max"], retrain_days,
            first_origin=(request.get("first_origin_date") or None))
        return {**calendar,
                "default_retrain_days": default_days,
                "default_first_origin_offset_days": DEFAULT_FIRST_ORIGIN_OFFSET_DAYS,
                "temporal_resolution": self.diagnosis["temporal_resolution"],
                "report_date_min": self.diagnosis["report_date_min"],
                "report_date_max": self.diagnosis["report_date_max"]}

    # -- run ---------------------------------------------------------------

    def start(self, request: dict) -> None:
        normalized_request = json.loads(json.dumps(request, sort_keys=True))
        with self.lock:
            if self.job.state == "running":
                raise PreparationError("a run is already in progress")
            if (self.job.state == "done" and self.last_request == normalized_request
                    and self.report_root is not None
                    and (self.report_root / "report" / "report.html").is_file()):
                return
            self.job = Job(state="running", stage="preparing", started=time.time())
            self.last_request = normalized_request
        thread = threading.Thread(target=self._run, args=(request,), daemon=True)
        thread.start()

    def stop(self) -> None:
        with self.lock:
            if self.job.state != "running":
                raise PreparationError("there is no running experiment to stop")
            self.job.stop_requested = True
            self.job.stage = "stopping"
            self.job.append("Stop requested. Finishing the current safe work unit...")

    def resume(self) -> None:
        if self.job.state != "stopped" or self.last_request is None:
            raise PreparationError("there is no stopped experiment to resume")
        self.start(self.last_request)

    def _check_stop(self) -> None:
        if self.job.stop_requested:
            raise RunStopped("stopped by the user")

    def _run(self, request: dict) -> None:
        job = self.job
        try:
            layers = tuple(request.get("layers") or DEFAULT_LAYERS)
            unknown = [name for name in layers if name not in PREDICTION_LAYERS]
            if unknown:
                raise PreparationError(f"unknown prediction layer(s): {unknown}")
            spec = build_spec(
                self.dataset_name, self.archive_path, self.work_dir / self.dataset_name,
                reference_col=request["reference_col"], report_col=request["report_col"],
                value_cols=tuple(request["value_cols"]),
                geo_col=request.get("geo_col") or None,
                value_type=request.get("value_type", "count"),
                target_lag=optional_int(request.get("target_lag"), "target_lag"),
                confirm_target_lag=bool(request.get("confirm_target_lag")),
                training_days=optional_int(request.get("training_days"), "training_days"),
                testing_days=optional_int(request.get("retrain_days"), "retrain_days"),
                first_origin_date=(request.get("first_origin_date") or None),
                min_location_rows=(optional_int(request.get("min_location_rows"),
                                                "min_location_rows")
                                   or DEFAULT_MIN_LOCATION_ROWS),
                triangle_format=request.get("triangle_format", "parquet"),
                delphirf_dir=local_delphirf_dir())

            resolution = ("your confirmed choice" if request.get("confirm_target_lag")
                          else "matches the recommendation")
            job.append(f"Target lag: {spec.ref_lag} days ({resolution})")
            job.append(f"Methods: {', '.join(LAYER_LABELS.get(name, name) for name in layers)}")
            job.append(f"Retraining every {spec.testing_days} days from {spec.start_date} "
                       f"to {spec.end_date}")
            job.append("Building the prepared triangle with DelphiRF...")
            config_path = prepare_dataset(spec, prediction_layers=layers,
                                          config_path=self.work_dir / f"{spec.name}.json",
                                          on_output=job.append, should_stop=self._check_stop)

            # The schedule the R bridge actually wrote, which is the honest
            # denominator for the progress bar until the replay prints its own
            # (it counts only the origins with eligible cases).
            schedule = pd.read_csv(spec.output_dir / "test_dates.csv")
            job.origins_total = job.origins_scheduled = len(schedule)
            job.stage = "replaying"
            job.replay_started = time.time()
            job.append("")
            job.append(f"Scheduled {len(schedule)} retraining dates. Dates without enough "
                       "follow-up to score a forecast will be excluded from evaluation.")
            # run_pipeline reports progress on stdout; route it into the log so
            # the page can show which fold is running.
            _force_non_interactive_backend()
            from .config import load_dataset_config
            from .pipeline import run_pipeline
            config = load_dataset_config(config_path)
            with _capture_stdout(job.append):
                run_pipeline(config, should_stop=self._check_stop)

            report = config.output_dir / "report" / "report.html"
            self.report_root = config.output_dir
            job.output_dir = str(config.output_dir)
            job.report_available = report.is_file()
            job.stage = "done"
            job.state = "done"
            job.append("")
            job.append(f"Done. Report written to {report}")
        except RunStopped:
            job.state = "stopped"
            job.stage = "stopped"
            job.append("Stopped. Resume will restart this experiment with the saved settings.")
        except Exception as error:                      # noqa: BLE001 - shown to the user
            job.state = "failed"
            job.stage = "failed"
            job.error = str(error) or error.__class__.__name__
            job.append("")
            job.append(f"FAILED: {job.error}")
            # The traceback goes to the terminal, where a developer can see it;
            # the page gets the message.
            traceback.print_exc()
        finally:
            job.finished = time.time()


def _force_non_interactive_backend() -> None:
    """Switch matplotlib to Agg even if it was imported before MPLBACKEND was set.

    ``os.environ.setdefault`` above covers the normal case. This covers the
    one where something imported matplotlib first -- an embedding process, or
    a test that touched the report module -- because by then the environment
    variable has already been read and ignored.
    """
    try:
        import matplotlib
    except ImportError:
        return
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)


class _capture_stdout:
    """Route the job thread's ``print`` output into the job log, line by line.

    ``sys.stdout`` is process-global, so swapping it wholesale would also
    swallow anything the HTTP handler threads -- or the user's own code, in an
    embedded use -- printed while a run was in flight, and send it to the page
    instead of the terminal. Writes are therefore routed by thread: the thread
    that entered this context gets the callback, every other thread passes
    through to the real stdout untouched.
    """

    def __init__(self, on_line):
        self.on_line = on_line
        self.buffer = ""
        self.previous = None
        self.owner = threading.get_ident()

    def write(self, text: str) -> int:
        if threading.get_ident() != self.owner:
            return self.previous.write(text)
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.on_line(line)
        return len(text)

    def flush(self) -> None:
        if threading.get_ident() != self.owner:
            self.previous.flush()
            return
        if self.buffer:
            self.on_line(self.buffer)
            self.buffer = ""

    # A few libraries interrogate the stream rather than just writing to it.
    def isatty(self) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self.previous, name)

    def __enter__(self):
        import sys
        self.previous = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exception):
        import sys
        self.flush()
        sys.stdout = self.previous
        return False


class Handler(BaseHTTPRequestHandler):
    server_version = "AutoDelphiRF"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, format, *args):   # noqa: A002 - BaseHTTPRequestHandler's name
        """Quiet by default; the terminal belongs to the pipeline's own output."""
        if self.server.verbose:
            super().log_message(format, *args)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page embeds the session token, so it must never be cached by a
        # shared proxy; it is localhost-only anyway.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _host_is_local(self) -> bool:
        """Reject a request whose Host header is not the address we bound.

        This is the DNS-rebinding guard: a page on an attacker's domain can
        make the browser resolve that domain to 127.0.0.1 and send requests
        here, but it cannot change the Host header it sends.
        """
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in self.server.allowed_hosts

    def _authorized(self) -> bool:
        supplied = self.headers.get("X-AutoDelphiRF-Token", "")
        return secrets.compare_digest(supplied, self.server.token)

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:                                    # noqa: N802
        if not self._host_is_local():
            self._json({"error": "unrecognized Host header"}, 403)
            return
        route = urlparse(self.path)
        path = route.path
        if path in ("/", "/index.html"):
            self._serve_index()
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/environment":
            self._guarded(lambda: self._json(self.server.environment))
        elif path == "/api/status":
            self._guarded(lambda: self._json(self.server.session.snapshot()))
        elif path == "/api/results":
            self._guarded(self._serve_results)
        elif path == "/api/layers":
            self._guarded(lambda: self._json({
                "layers": [{"name": name, "label": LAYER_LABELS.get(name, name),
                            "description": LAYER_DESCRIPTIONS.get(name, ""),
                            "default": name in DEFAULT_LAYERS,
                            "expensive": name in EXPENSIVE_LAYERS}
                           for name in LAYER_DESCRIPTIONS if name in PREDICTION_LAYERS]}))
        elif path.startswith("/report"):
            self._serve_report(path[len("/report"):])
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:                                   # noqa: N802
        if not self._host_is_local():
            self._json({"error": "unrecognized Host header"}, 403)
            return
        if not self._authorized():
            self._json({"error": "missing or invalid session token"}, 403)
            return
        route = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            self._json({"error": f"file exceeds the {MAX_UPLOAD_BYTES // (1024**3)} GB browser "
                                 "upload limit; use `autodelphirf run --archive` instead"}, 413)
            return
        body = self.rfile.read(length) if length else b""
        try:
            if route.path == "/api/upload":
                query = parse_qs(route.query)
                filename = (query.get("name") or ["archive.csv"])[0]
                self._json(self.server.session.store_archive(filename, body))
            elif route.path == "/api/diagnose":
                self._json(self.server.session.diagnose(json.loads(body or b"{}")))
            elif route.path == "/api/schedule":
                self._json(self.server.session.schedule_preview(json.loads(body or b"{}")))
            elif route.path == "/api/run":
                self.server.session.start(json.loads(body or b"{}"))
                self._json({"started": True})
            elif route.path == "/api/stop":
                self.server.session.stop()
                self._json({"stopping": True})
            elif route.path == "/api/resume":
                self.server.session.resume()
                self._json({"started": True})
            else:
                self._json({"error": "not found"}, 404)
        except PreparationError as error:
            self._json({"error": str(error)}, 400)
        except (KeyError, ValueError) as error:
            self._json({"error": f"bad request: {error}"}, 400)
        except Exception as error:                     # noqa: BLE001 - reported to the page
            traceback.print_exc()
            self._json({"error": f"{error.__class__.__name__}: {error}"}, 500)

    def _guarded(self, action) -> None:
        if not self._authorized():
            self._json({"error": "missing or invalid session token"}, 403)
            return
        action()

    def _serve_results(self) -> None:
        job = self.server.session.job
        if not job.output_dir:
            self._json({"error": "no finished run yet"}, 404)
            return
        try:
            self._json({**results_summary(job.output_dir),
                        "origins_scheduled": job.origins_scheduled})
        except OSError as error:
            self._json({"error": f"the run's output could not be read: {error}"}, 500)

    # -- static & report ---------------------------------------------------

    def _serve_index(self) -> None:
        html = resource_path("web/index.html").read_text()
        # The token is minted per server start and handed to the page here;
        # everything the page does afterwards carries it.
        html = html.replace("__AUTODELPHIRF_TOKEN__", self.server.token)
        self._bytes(html.encode(), "text/html; charset=utf-8")

    def _serve_static(self, relative: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", relative):
            self._json({"error": "not found"}, 404)
            return
        try:
            path = resource_path(f"web/{relative}")
        except FileNotFoundError:
            self._json({"error": "not found"}, 404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._bytes(path.read_bytes(), content_type)

    def _serve_report(self, relative: str) -> None:
        """Serve the generated report directory, and nothing else.

        Every path is resolved and checked to be inside the report root, so a
        crafted ``..`` cannot walk out of it into the rest of the filesystem.
        """
        root = self.server.session.report_root
        if root is None:
            self._json({"error": "no report yet"}, 404)
            return
        relative = unquote(relative).lstrip("/") or "report/report.html"
        target = (Path(root) / relative).resolve()
        try:
            target.relative_to(Path(root).resolve())
        except ValueError:
            self._json({"error": "not found"}, 404)
            return
        if not target.is_file():
            self._json({"error": "not found"}, 404)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._bytes(target.read_bytes(), content_type)


class AutoDelphiRFServer(ThreadingHTTPServer):
    """The server, plus the single session it is working on."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, work_dir: Path, verbose: bool = False):
        super().__init__(address, Handler)
        self.token = secrets.token_urlsafe(32)
        self.session = Session(work_dir)
        self.verbose = verbose
        self.environment = environment_report()
        host = address[0]
        self.allowed_hosts = {host, "localhost", "127.0.0.1", "::1", ""}


def serve(host: str = "127.0.0.1", port: int = 8765, work_dir: Path | None = None,
          open_browser: bool = True, verbose: bool = False) -> AutoDelphiRFServer:
    """Start the UI and block until interrupted."""
    work_dir = Path(work_dir or Path.cwd() / "autodelphirf_work").resolve()
    server = AutoDelphiRFServer((host, port), work_dir, verbose=verbose)
    bound_port = server.server_address[1]
    url = f"http://{host}:{bound_port}/"

    print(f"AutoDelphiRF is running at {url}")
    print(f"Working directory: {work_dir}")
    environment = server.environment
    if environment["preprocessing"]:
        print(f"Preprocessing: ready ({environment['detail']})")
        if not environment["delphirf"]:
            print("DelphiRF model methods: NOT ready; the latest-value reference and "
                  "revision-pattern matching are available.")
    else:
        print(f"Preprocessing: NOT ready -- {environment['detail']}")
        print("  Diagnosis still works; building a prepared triangle does not.")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n  WARNING: bound to {host}, not localhost. This server runs programs and "
              "writes\n  files on this machine. Do not expose it to a network you do not "
              "control.\n")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return server
