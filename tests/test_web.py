"""The local web UI.

The server writes files and runs subprocesses for whoever can reach it, so
the access controls get as much attention here as the workflow does. The
pipeline itself is not run (that is `test_end_to_end.py`'s job); everything
up to the point where the background job starts is.
"""
from __future__ import annotations

import json
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from autodelphirf import web
from autodelphirf.prepare import PreparationError
from autodelphirf.web import (ALLOWED_SUFFIXES, DEFAULT_LAYERS, AutoDelphiRFServer, Session,
                                 archive_suffix, guess_columns, optional_int,
                                 results_summary, safe_stem)

from conftest import archive_frame


# --- filename handling ----------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("nhsn_archive.csv", "nhsn_archive"),
    ("nhsn archive.csv", "nhsn_archive"),
    ("NHSN-2024.csv.gz", "NHSN-2024"),
    ("data.parquet", "data"),
    ("/etc/passwd.csv", "passwd"),
    ("../../escape.csv", "escape"),
    ("a/b/c/deep.csv", "deep"),
    ("../../../.ssh/id_rsa.csv", "id_rsa"),
])
def test_uploaded_names_are_reduced_to_a_safe_stem(name, expected):
    """The stem reaches the filesystem and an R command line."""
    assert safe_stem(name) == expected


def test_a_name_with_nothing_usable_still_yields_a_stem():
    assert safe_stem("///.csv") == "dataset"
    assert safe_stem("") == "dataset"


def test_a_very_long_name_is_truncated():
    assert len(safe_stem("x" * 500 + ".csv")) == 64


def test_url_encoded_traversal_is_also_neutralised():
    assert safe_stem(quote("../../etc/shadow.csv")) == "shadow"


@pytest.mark.parametrize("suffix", ALLOWED_SUFFIXES)
def test_accepted_suffixes_are_recognised(suffix):
    assert archive_suffix(f"archive{suffix}") == suffix


@pytest.mark.parametrize("name", ["notes.txt", "archive.xlsx", "archive", "script.sh",
                                  "archive.csv.exe"])
def test_other_file_types_are_refused_with_guidance(name):
    with pytest.raises(PreparationError, match="Drop a .csv"):
        archive_suffix(name)


def test_suffix_matching_is_case_insensitive():
    assert archive_suffix("ARCHIVE.CSV") == ".csv"


# --- column guessing ------------------------------------------------------

def test_canonical_names_are_guessed_exactly():
    guesses = guess_columns(["geo_value", "reference_date", "report_date", "value"])
    assert guesses == {"geo": "geo_value", "reference": "reference_date",
                       "report": "report_date", "value": "value"}


def test_common_alternative_names_are_guessed():
    guesses = guess_columns(["state", "time_value", "issue_date", "count"])
    assert guesses["geo"] == "state"
    assert guesses["reference"] == "time_value"
    assert guesses["report"] == "issue_date"
    assert guesses["value"] == "count"


def test_an_exact_hint_outranks_a_substring_match():
    """'value' must win over 'value_total' when both are present."""
    assert guess_columns(["value_total", "value"])["value"] == "value"


def test_unguessable_columns_are_simply_absent_not_wrong():
    guesses = guess_columns(["col_a", "col_b", "col_c"])
    assert "reference" not in guesses and "report" not in guesses


def test_guessing_is_case_insensitive():
    assert guess_columns(["Geo_Value", "Reference_Date"])["geo"] == "Geo_Value"


# --- session workflow -----------------------------------------------------

@pytest.fixture
def session(tmp_path):
    return Session(tmp_path / "work")


@pytest.fixture
def archive_bytes():
    return archive_frame(days=120, max_lag=30).to_csv(index=False).encode()


def test_storing_an_archive_reports_its_columns_and_a_preview(session, archive_bytes):
    stored = session.store_archive("my archive.csv", archive_bytes)
    assert stored["name"] == "my_archive"
    assert stored["columns"] == ["geo_value", "reference_date", "report_date", "value"]
    assert stored["guesses"]["reference"] == "reference_date"
    assert len(stored["preview"]) == 5
    assert session.archive_path.is_file()


def test_an_unreadable_upload_is_rejected_and_leaves_no_file(session, tmp_path):
    with pytest.raises(PreparationError, match="could not read"):
        session.store_archive("broken.csv", b"\x00\x01\x02 not,a,csv\n\"unclosed")
    assert list((tmp_path / "work").glob("*.csv")) == []


def test_diagnosing_before_uploading_says_so(session):
    with pytest.raises(PreparationError, match="no archive has been uploaded"):
        session.diagnose({"reference_col": "a", "report_col": "b", "value_cols": ["c"]})


def test_diagnosis_without_a_value_column_is_refused(session, archive_bytes):
    session.store_archive("a.csv", archive_bytes)
    with pytest.raises(PreparationError, match="at least one value column"):
        session.diagnose({"reference_col": "reference_date", "report_col": "report_date",
                          "value_cols": []})


def test_diagnosis_returns_the_curve_the_page_plots(session, archive_bytes):
    session.store_archive("a.csv", archive_bytes)
    report = session.diagnose({"reference_col": "reference_date", "report_col": "report_date",
                               "geo_col": "geo_value", "value_cols": ["value"],
                               "value_type": "count"})
    assert report["recommended_target_lag"] > 0
    curve = report["target_lag_completion_curve"]
    assert len(curve) >= 2
    # The chart plots lag -> median relative error; both must be numeric.
    assert all(float(lag) >= 0 and 0 <= float(error) for lag, error in curve.items())


def test_a_user_target_lag_produces_the_confirmation_prompt_the_page_shows(session, archive_bytes):
    session.store_archive("a.csv", archive_bytes)
    report = session.diagnose({"reference_col": "reference_date", "report_col": "report_date",
                               "geo_col": "geo_value", "value_cols": ["value"],
                               "value_type": "count", "target_lag": 400})
    assert report["user_target_lag"] == 400
    assert report["target_lag_confirmation_prompt"]


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), (14, 14), ("14", 14),
                                            (14.0, 14)])
def test_a_target_lag_field_reads_the_values_the_page_can_send(value, expected):
    assert optional_int(value, "target_lag") == expected


@pytest.mark.parametrize("value", [{}, {"a": 1}, [], [7], True, "abc"])
def test_a_target_lag_field_that_is_not_a_number_names_itself(value):
    """A 400 naming the field, not a TypeError from inside the pipeline.

    The page once passed its click handler's Event straight through as the
    target lag; an Event serialises to ``{}``, and ``int({})`` surfaced as an
    opaque 500. Whatever the page sends, a bad value has to say which field
    it was.
    """
    with pytest.raises(PreparationError, match="target_lag must be a whole number"):
        optional_int(value, "target_lag")


def test_a_non_numeric_target_lag_is_refused_before_the_archive_is_read(session, archive_bytes):
    session.store_archive("a.csv", archive_bytes)
    with pytest.raises(PreparationError, match="target_lag must be a whole number"):
        session.diagnose({"reference_col": "reference_date", "report_col": "report_date",
                          "geo_col": "geo_value", "value_cols": ["value"],
                          "value_type": "count", "target_lag": {}})


# --- the retraining schedule ---------------------------------------------

@pytest.fixture
def diagnosed(session, archive_bytes):
    session.store_archive("a.csv", archive_bytes)
    session.diagnose({"reference_col": "reference_date", "report_col": "report_date",
                      "geo_col": "geo_value", "value_cols": ["value"], "value_type": "count"})
    return session


def test_the_diagnosis_carries_the_schedule_the_page_prefills(diagnosed):
    """The page must not re-derive the cadence rule in JavaScript."""
    retraining = diagnosed.diagnosis["retraining"]
    assert retraining["n_origins"] >= 1
    assert retraining["retrain_days"] == retraining["default_retrain_days"]
    assert retraining["history_days_at_first_origin"] == \
        retraining["default_first_origin_offset_days"]


def test_previewing_a_schedule_before_diagnosing_says_so(session):
    with pytest.raises(PreparationError, match="diagnose the archive before"):
        session.schedule_preview({"retrain_days": 30})


def test_a_longer_interval_previews_fewer_retrainings(diagnosed):
    tight = diagnosed.schedule_preview({"retrain_days": 7})
    loose = diagnosed.schedule_preview({"retrain_days": 56})
    assert tight["n_origins"] > loose["n_origins"]


def test_an_empty_field_previews_the_default_rather_than_failing(diagnosed):
    """The page sends null for "leave it alone"; the server says what that was."""
    preview = diagnosed.schedule_preview({"retrain_days": None, "first_origin_date": None})
    assert preview["retrain_days"] == preview["default_retrain_days"]


def test_an_impossible_first_origin_is_refused_by_the_preview(diagnosed):
    with pytest.raises(PreparationError, match="after the archive's last report date"):
        diagnosed.schedule_preview({"first_origin_date": "2099-01-01"})


def test_the_preview_is_what_the_run_would_build(diagnosed, monkeypatch):
    """The origin count shown before the run has to be the run's own schedule.

    Both come from ``retraining_calendar``; this pins that they are called with
    the same archive bounds, so the page cannot advertise one number and the
    R bridge lay out another.
    """
    from autodelphirf import prepare
    preview = diagnosed.schedule_preview({"retrain_days": 21, "first_origin_date": None})
    direct = prepare.retraining_calendar(diagnosed.diagnosis["report_date_min"],
                                         diagnosed.diagnosis["report_date_max"], 21)
    assert preview["first_origin"] == direct["first_origin"]
    assert preview["last_origin"] == direct["last_origin"]
    assert preview["n_origins"] == direct["n_origins"]


# --- progress and results -------------------------------------------------

def test_the_replay_log_drives_the_progress_bar():
    job = web.Job(state="running", started=time.time(), replay_started=time.time())
    job.append("revroute v8 replay: fold 3/17 (schedule fold 4)")
    snapshot = job.snapshot()
    assert (snapshot["origins_done"], snapshot["origins_total"]) == (3, 17)


def test_no_time_estimate_is_offered_from_a_single_origin():
    """One origin is not a rate; a first guess from it would be wild."""
    job = web.Job(state="running", started=time.time(), replay_started=time.time())
    job.append("revroute v8 replay: fold 1/40 (schedule fold 1)")
    assert job.snapshot()["remaining_seconds"] is None


def test_the_estimate_ignores_the_time_spent_building_the_triangle():
    """Preprocessing runs before the replay; counting it inflates every estimate."""
    now = time.time()
    job = web.Job(state="running", started=now - 600, replay_started=now - 10)
    job.append("revroute v8 replay: fold 5/10 (schedule fold 5)")
    # 10s for 5 origins, 5 left -> about 10s, not 600s-scaled.
    assert job.snapshot()["remaining_seconds"] == pytest.approx(10, abs=3)


def test_an_ordinary_log_line_does_not_move_the_progress_bar():
    job = web.Job(state="running", started=time.time())
    job.append("Building the prepared triangle with DelphiRF...")
    assert job.snapshot()["origins_total"] == 0


def test_the_default_run_scores_revroutes_predictor_against_the_reference():
    assert "delphirf" in DEFAULT_LAYERS
    assert "baseline_null" in DEFAULT_LAYERS


def test_results_of_a_run_that_wrote_nothing_are_empty_not_an_error(tmp_path):
    summary = results_summary(tmp_path)
    assert summary["tables"] == {} and summary["n_origins"] == 0
    assert summary["report_available"] is False


def test_results_read_back_the_tables_a_run_wrote(tmp_path):
    (tmp_path / "report" / "tables").mkdir(parents=True)
    (tmp_path / "v1_pairwise_point.csv").write_text(
        "subset,comparison,mean_delta,win_probability\n"
        "overall,rr_delphirf3 - baseline_null,-12.5,0.81\n"
        "early,rr_delphirf3 - baseline_null,-9.0,0.77\n")
    (tmp_path / "report" / "tables" / "comparison_by_origin.csv").write_text(
        "cutoff,method,n,mean_ae\n2020-01-01,rr_delphirf3,4,1.5\n2020-02-01,rr_delphirf3,4,1.1\n")
    (tmp_path / "revroute_pools.csv").write_text(
        "fold,cutoff,k_selected,tasks\n1,2020-01-01,3,9\n2,2020-02-01,4,9\n")
    summary = results_summary(tmp_path)
    # Only the pooled line: the per-subset breakdown belongs to the report.
    assert len(summary["tables"]["headline"]) == 1
    assert summary["tables"]["headline"][0]["subset"] == "overall"
    assert summary["n_origins"] == 2
    # RevRoute relearns its pools every retraining, so there is one row per origin.
    assert len(summary["tables"]["revroute_pools"]) == 2


def test_a_huge_pool_profile_is_truncated_rather_than_sent_whole(tmp_path):
    rows = "\n".join(f"1,2020-01-01,{i}" for i in range(web.MAX_RESULT_ROWS + 50))
    profile = tmp_path / "report" / "tables" / "rr_delphirf_cluster_profile.csv"
    profile.parent.mkdir(parents=True)
    profile.write_text(f"fold,cutoff,pool\n{rows}\n")
    summary = results_summary(tmp_path)
    assert len(summary["tables"]["revroute_pool_profile"]) == web.MAX_RESULT_ROWS
    assert "revroute_pool_profile" in summary["truncated"]


def test_two_runs_at_once_are_refused(session, archive_bytes, monkeypatch):
    """One session, one job: the state model depends on it."""
    session.store_archive("a.csv", archive_bytes)
    blocked = threading.Event()
    monkeypatch.setattr(web.Session, "_run", lambda self, request: blocked.wait(5))
    session.start({"reference_col": "reference_date", "report_col": "report_date",
                   "value_cols": ["value"]})
    try:
        with pytest.raises(PreparationError, match="already in progress"):
            session.start({"reference_col": "reference_date", "report_col": "report_date",
                           "value_cols": ["value"]})
    finally:
        blocked.set()


def test_a_failing_job_is_reported_rather_than_raised(session, archive_bytes):
    """The page needs the error as data; the traceback goes to the terminal."""
    session.store_archive("a.csv", archive_bytes)
    session.start({"reference_col": "reference_date", "report_col": "report_date",
                   "value_cols": ["value"], "layers": ["no_such_layer"]})
    for _ in range(100):
        if session.job.state in ("done", "failed"):
            break
        threading.Event().wait(0.05)
    assert session.job.state == "failed"
    assert "no_such_layer" in session.job.error


def test_the_job_log_is_bounded(session):
    for index in range(2000):
        session.job.append(f"line {index}")
    assert len(session.job.log) == 500
    assert session.job.log[-1] == "line 1999"


# --- HTTP surface ---------------------------------------------------------

@pytest.fixture
def server(tmp_path, monkeypatch):
    # environment_report() shells out to Rscript; the HTTP tests do not need
    # a real answer and should not pay 3 seconds for one.
    monkeypatch.setattr(web, "environment_report",
                        lambda: {"rscript": None, "delphirf": False, "detail": "stubbed"})
    instance = AutoDelphiRFServer(("127.0.0.1", 0), tmp_path / "work")
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()


def request(server, path, *, body=None, token="use-real", host=None, method=None):
    """Make one HTTP call, returning (status, parsed-or-raw body)."""
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    data = body if isinstance(body, bytes) else (json.dumps(body).encode()
                                                 if body is not None else None)
    call = Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    if token == "use-real":
        call.add_header("X-AutoDelphiRF-Token", server.token)
    elif token:
        call.add_header("X-AutoDelphiRF-Token", token)
    if host:
        call.add_header("Host", host)
    try:
        with urlopen(call, timeout=30) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def test_the_page_is_served_and_carries_the_session_token(server):
    status, body = request(server, "/", token=None)
    assert status == 200
    text = body.decode()
    assert server.token in text
    assert "__AUTODELPHIRF_TOKEN__" not in text     # the placeholder was substituted


def test_static_assets_are_served(server):
    for asset in ("app.js", "style.css"):
        status, body = request(server, f"/static/{asset}", token=None)
        assert status == 200 and len(body) > 100


@pytest.mark.parametrize("path", ["/static/../web.py", "/static/..%2Fweb.py",
                                  "/static/sub/dir.js", "/static/absent.js"])
def test_static_serving_cannot_escape_its_directory(server, path):
    status, _ = request(server, path, token=None)
    assert status == 404


@pytest.mark.parametrize("endpoint", ["/api/environment", "/api/status", "/api/layers"])
def test_api_reads_require_the_session_token(server, endpoint):
    assert request(server, endpoint, token=None)[0] == 403
    assert request(server, endpoint, token="wrong-token")[0] == 403
    assert request(server, endpoint)[0] == 200


def test_api_writes_require_the_session_token(server):
    status, _ = request(server, "/api/upload?name=a.csv", body=b"x", token=None)
    assert status == 403


def test_a_foreign_host_header_is_refused(server):
    """The DNS-rebinding guard: a remote page cannot change the Host it sends."""
    assert request(server, "/", token=None, host="evil.example.com")[0] == 403
    assert request(server, "/api/environment", host="evil.example.com")[0] == 403


def test_localhost_host_headers_are_accepted(server):
    for host in ("127.0.0.1", "localhost"):
        assert request(server, "/", token=None, host=host)[0] == 200


def test_the_layer_list_only_offers_registered_layers(server):
    from autodelphirf.registry import PREDICTION_LAYERS

    status, body = request(server, "/api/layers")
    assert status == 200
    layers = json.loads(body)["layers"]
    assert layers
    assert all(layer["name"] in PREDICTION_LAYERS for layer in layers)
    assert any(layer["default"] for layer in layers)
    assert all(layer["description"] for layer in layers)


def test_status_before_any_run_is_idle(server):
    snapshot = json.loads(request(server, "/api/status")[1])
    assert snapshot["state"] == "idle"
    assert snapshot["log"] == []


def test_uploading_and_diagnosing_over_http(server, archive_bytes):
    status, body = request(server, "/api/upload?name=web%20test.csv", body=archive_bytes)
    assert status == 200
    assert json.loads(body)["name"] == "web_test"

    status, body = request(server, "/api/diagnose", body={
        "reference_col": "reference_date", "report_col": "report_date",
        "geo_col": "geo_value", "value_cols": ["value"], "value_type": "count"})
    assert status == 200
    assert json.loads(body)["recommended_target_lag"] > 0


def test_an_unsupported_upload_is_a_400_with_a_readable_message(server):
    status, body = request(server, "/api/upload?name=notes.txt", body=b"hello")
    assert status == 400
    assert "Drop a .csv" in json.loads(body)["error"]


def test_a_bad_column_name_is_a_400_not_a_500(server, archive_bytes):
    request(server, "/api/upload?name=a.csv", body=archive_bytes)
    status, body = request(server, "/api/diagnose", body={
        "reference_col": "not_a_column", "report_col": "report_date",
        "value_cols": ["value"], "value_type": "count"})
    assert status == 400
    assert "not_a_column" in json.loads(body)["error"]


def test_malformed_json_is_a_400(server):
    assert request(server, "/api/diagnose", body=b"{not json")[0] == 400


def test_an_unknown_route_is_404(server):
    assert request(server, "/api/nope")[0] == 404
    assert request(server, "/nope", token=None)[0] == 404


def test_the_report_route_refuses_before_a_report_exists(server):
    assert request(server, "/report/report/report.html", token=None)[0] == 404


def test_the_report_route_cannot_walk_out_of_the_report_directory(server, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    root = tmp_path / "work" / "results"
    (root / "report").mkdir(parents=True)
    (root / "report" / "report.html").write_text("<html>report</html>")
    server.session.report_root = root

    assert request(server, "/report/report/report.html", token=None)[0] == 200
    for attempt in ("/report/../secret.txt", "/report/..%2F..%2Fsecret.txt",
                    "/report/report/../../../secret.txt"):
        status, _ = request(server, attempt, token=None)
        assert status == 404, attempt


# --- matplotlib backend ---------------------------------------------------

def test_the_server_defaults_matplotlib_to_a_non_interactive_backend():
    """A GUI backend blocks forever when driven from a worker thread.

    The report stage draws figures, and Session._run is a background thread.
    With the macOS default (MacOSX) or Tk, that deadlocks after the tables are
    written and before report.html exists -- a run that looks finished but
    never completes.
    """
    import os

    assert os.environ.get("MPLBACKEND") == "Agg"


def test_the_backend_is_forced_even_if_matplotlib_was_imported_first():
    """MPLBACKEND is read at import, so it is too late for an embedding process."""
    import matplotlib

    matplotlib.use("Agg", force=True)          # baseline
    web._force_non_interactive_backend()
    assert matplotlib.get_backend().lower() == "agg"


def test_forcing_the_backend_is_safe_when_matplotlib_is_absent(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    web._force_non_interactive_backend()       # must not raise


# --- stdout routing -------------------------------------------------------

def test_only_the_owning_thread_s_output_reaches_the_job_log():
    """sys.stdout is process-global; other threads must not be swallowed."""
    import sys

    captured = []
    other_thread_wrote = []

    class Recorder:
        def write(self, text):
            other_thread_wrote.append(text)
            return len(text)

        def flush(self):
            pass

    original = sys.stdout
    sys.stdout = Recorder()
    try:
        with web._capture_stdout(captured.append):
            print("from the owning thread")
            worker = threading.Thread(target=lambda: print("from another thread"))
            worker.start()
            worker.join()
    finally:
        sys.stdout = original

    assert "from the owning thread" in captured
    assert not any("another thread" in line for line in captured)
    assert any("another thread" in text for text in other_thread_wrote)
