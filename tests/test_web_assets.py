"""Static checks on the page the server ships.

There is no browser in the test environment, so these do what a browser would
have caught: every element the script reaches for exists in the markup, every
endpoint it calls is routed, and the chart's geometry lands inside its own
viewBox. A typo in an element id is otherwise a silent no-op at runtime.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest

from autodelphirf import web
from autodelphirf.resources import resource_path


@pytest.fixture(scope="module")
def html() -> str:
    return resource_path("web/index.html").read_text()


@pytest.fixture(scope="module")
def script() -> str:
    return resource_path("web/app.js").read_text()


@pytest.fixture(scope="module")
def stylesheet() -> str:
    return resource_path("web/style.css").read_text()


def element_ids(html: str) -> set:
    return set(re.findall(r'\bid="([^"]+)"', html))


def referenced_ids(script: str) -> set:
    """Ids the script looks up, via its `$` helper or getElementById."""
    return (set(re.findall(r"\$\('([^']+)'\)", script))
            | set(re.findall(r"getElementById\('([^']+)'\)", script))
            | set(re.findall(r"\bshow\('([^']+)'\)", script))
            | set(re.findall(r"scrollIntoView\('([^']+)'\)", script)))


# --- markup/script consistency -------------------------------------------

def test_every_element_the_script_looks_up_exists_in_the_markup(html, script):
    present = element_ids(html)
    # Ids the script creates at runtime rather than finding in the markup.
    created_at_runtime = {"chart-tooltip", "choose-other", "col-reference", "col-report",
                          "col-value", "col-geo", "col-denominator"}
    missing = referenced_ids(script) - present - created_at_runtime
    assert not missing, f"app.js references ids that do not exist: {sorted(missing)}"


def test_ids_created_at_runtime_are_actually_created(script):
    """The exemptions above must be real, not a way to hide a typo.

    Two mechanisms create them: an inline ``id="..."`` in a template literal,
    or the ``columnSelect`` helper, which stamps the id it is handed. Either
    way the literal has to appear in the file.
    """
    assert 'id="${id}"' in script, "columnSelect no longer stamps the id it is given"
    for identifier in ("chart-tooltip", "choose-other", "col-reference", "col-report",
                       "col-value", "col-geo", "col-denominator"):
        emitted_inline = f'id="{identifier}"' in script
        passed_to_helper = f"'{identifier}'" in script
        assert emitted_inline or passed_to_helper, (
            f"{identifier} is exempted from the markup check but never created")


def test_no_duplicate_element_ids(html):
    found = re.findall(r'\bid="([^"]+)"', html)
    assert len(found) == len(set(found)), "duplicate id in index.html"


def test_the_token_placeholder_is_present_for_the_server_to_substitute(html):
    assert "__AUTODELPHIRF_TOKEN__" in html
    assert 'data-token="__AUTODELPHIRF_TOKEN__"' in html


def test_the_script_reads_the_token_and_sends_it(script):
    assert "document.body.dataset.token" in script
    assert "X-AutoDelphiRF-Token" in script


def test_the_script_parses(script, tmp_path):
    """A syntax error in app.js is a blank page, and nothing else here catches it.

    There is no browser in the test environment, but macOS ships JavaScriptCore
    via osascript. ``new Function(src)`` parses without executing, so the DOM
    calls inside are irrelevant. Skipped where that is unavailable.
    """
    if not shutil.which("osascript"):
        pytest.skip("no JavaScript engine available to parse with")
    source = resource_path("web/app.js")
    checker = tmp_path / "check.js"
    checker.write_text(
        "ObjC.import('Foundation');\n"
        "const src = ObjC.unwrap($.NSString."
        "stringWithContentsOfFileEncodingError(%r, $.NSUTF8StringEncoding, null));\n"
        "try { new Function(src); 'OK'; } catch (e) { 'FAIL: ' + e.message; }\n" % str(source))
    result = subprocess.run(["osascript", "-l", "JavaScript", str(checker)],
                            capture_output=True, text=True, timeout=120)
    assert result.stdout.strip() == "OK", result.stdout.strip() or result.stderr.strip()


# --- handler wiring -------------------------------------------------------

def test_no_handler_that_takes_an_argument_is_registered_bare(script):
    """A listener is called with the event, which is never a sensible argument.

    ``addEventListener('click', runDiagnosis)`` handed the click event to
    ``runDiagnosis``'s ``userTargetLag`` parameter. An Event serialises to
    ``{}``, so the page posted an object where the server expected a number
    of days. Handlers that take parameters must be wrapped.
    """
    taking_arguments = {name for name, arguments
                        in re.findall(r"function\s+(\w+)\s*\(([^)]*)\)", script)
                        if arguments.strip()}
    for handler in re.findall(r"""(?:addEventListener\(\s*['"]\w+['"]\s*,|\.on\w+\s*=)\s*"""
                              r"(\w+)\s*[,)\n;]", script):
        assert handler not in taking_arguments, (
            f"{handler}() takes arguments but is registered as a bare handler, so it will "
            "receive the event instead")


def test_the_diagnose_button_is_wired_idempotently(script):
    """``buildColumnForm`` runs again on every upload; a listener would stack."""
    assert "$('diagnose-button').onclick" in script


def test_custom_target_lag_continues_to_method_settings(script):
    assert "runDiagnosis(value, true)" in script
    accept_lag = script.split("function acceptLag", 1)[1].split("function initSchedule", 1)[0]
    assert "show('card-schedule')" in accept_lag
    assert "show('card-methods')" in accept_lag
    assert "scrollIntoView('card-methods')" in accept_lag


def test_target_lag_wording_matches_the_selection_rule(html, script):
    assert "within 10% for at least 90% of location and reference-date" in script
    assert "at least 90% of location and" in html
    assert "median relative error as supplementary diagnostic" in script


def test_diagnosis_names_both_date_axis_cadences(script):
    assert "fact('Reference-date cadence', report.reference_axis_resolution)" in script
    assert "fact('Report-date cadence', report.report_axis_resolution)" in script
    assert "fact('Cadence', report.temporal_resolution)" not in script


# --- routes ---------------------------------------------------------------

def test_every_endpoint_the_script_calls_is_routed_by_the_server(script):
    called = set(re.findall(r"""['"`](/api/[a-z_]+)""", script))
    assert called, "no API endpoints found in app.js; the pattern is stale"
    server_source = Path(web.__file__).read_text()
    for endpoint in called:
        assert f'"{endpoint}"' in server_source, f"{endpoint} is called but not routed"


def test_the_report_link_matches_the_servers_report_route(script):
    """The link is built by hand in the page; a mismatch 404s after a long run."""
    assert "/report/report/report.html" in script


def test_no_external_resources_are_referenced(html, script, stylesheet):
    """This page is served by a local process that may have no internet access."""
    for name, text in (("index.html", html), ("app.js", script), ("style.css", stylesheet)):
        for pattern in ("http://", "https://"):
            for match in re.findall(rf'{pattern}[^\s"\'<>)]+', text):
                # Links a person clicks are fine; fetched subresources are not.
                assert "github.com" in match, f"{name} loads an external resource: {match}"


# --- the chart ------------------------------------------------------------

def chart_geometry(lags, errors):
    """Re-implement app.js's scale arithmetic to check it stays in bounds."""
    width, height = 640, 240
    pad = {"top": 14, "right": 18, "bottom": 34, "left": 52}
    plot_width = width - pad["left"] - pad["right"]
    plot_height = height - pad["top"] - pad["bottom"]
    max_lag = max(lags)
    max_error = max([0.12, *errors])
    xs = [pad["left"] + (lag / max_lag) * plot_width for lag in lags]
    ys = [pad["top"] + plot_height - (error / max_error) * plot_height for error in errors]
    return width, height, pad, xs, ys


def test_chart_points_land_inside_the_viewbox():
    lags = [0, 7, 14, 21, 28, 35, 42, 56, 70, 84, 98, 120, 150, 180, 240, 300, 365]
    errors = [0.501, 0.16, 0.061, 0.029, 0.018, 0.013, 0.011, 0.009, 0.008,
              0.007, 0.006, 0.005, 0.004, 0.004, 0.003, 0.003, 0.002]
    width, height, pad, xs, ys = chart_geometry(lags, errors)
    assert all(pad["left"] <= x <= width - pad["right"] + 1e-9 for x in xs)
    assert all(pad["top"] - 1e-9 <= y <= height - pad["bottom"] + 1e-9 for y in ys)


def run_in_jsc(tmp_path, call: str, exports="drawOriginChart", node="origin-chart") -> str:
    """Run part of app.js in JavaScriptCore against a stub DOM, return one node's HTML.

    The completion-curve tests above re-implement the scale arithmetic in
    Python and check that. For the per-origin chart the shipped function is
    run directly instead, so the thing under test is the code that ships
    rather than a second copy of its maths.
    """
    if not shutil.which("osascript"):
        pytest.skip("no JavaScript engine available")
    source = resource_path("web/app.js")
    driver = tmp_path / "render.js"
    driver.write_text("""
ObjC.import('Foundation');
const read = (p) => ObjC.unwrap($.NSString.stringWithContentsOfFileEncodingError(
  p, $.NSUTF8StringEncoding, null));
const nodes = {};
const stub = (id) => (nodes[id] = nodes[id] || {id, innerHTML: '', textContent: '',
                                                style: {}, hidden: false});
globalThis.document = {
  getElementById: stub, body: {dataset: {token: 't'}},
  querySelectorAll: (sel) => (sel || '').includes('layer-list') ? [{value: 'baseline_null'}] : [],
  querySelector: () => ({value: 'count', checked: true}),
};
globalThis.window = globalThis;
globalThis.fetch = () => { throw new Error('this harness must not reach the network'); };
let src = read(%r);
src = src.slice(0, src.indexOf('/* --- go ---'));   // definitions only, no bootstrap
const api = new Function(src + `
; return {%s};`)();
%s
(nodes[%r] || {innerHTML: ''}).innerHTML;
""" % (str(source), exports, call, node))
    result = subprocess.run(["osascript", "-l", "JavaScript", str(driver)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_a_run_is_refused_up_front_when_delphirf_is_not_ready(tmp_path):
    """Preprocessing is a run's first step and the first thing to fail without it.

    The environment card says so at step 1, but by step 6 that warning is five
    cards up the page. The refusal has to arrive where the button is, and it
    has to carry the fix.
    """
    html = run_in_jsc(tmp_path, """
api.state.environment = {delphirf: false, detail: 'DelphiRF predates what RevRoute needs.'};
api.startRun();
""", exports="startRun, state", node="run-result")
    assert "AUTODELPHIRF_DELPHIRF_DIR" in html, "the refusal does not say how to fix it"
    assert "DelphiRF predates what RevRoute needs." in html, "the reason is not passed through"


def test_a_ready_environment_does_not_block_the_run(tmp_path):
    """The guard must not be a permanent roadblock: with DelphiRF present the
    run proceeds (and this harness has no network, so it fails at fetch)."""
    html = run_in_jsc(tmp_path, """
api.state.environment = {delphirf: true, detail: 'installed DelphiRF'};
try { api.startRun(); } catch (e) {}
""", exports="startRun, state", node="run-result")
    assert "AUTODELPHIRF_DELPHIRF_DIR" not in html


def test_local_preprocessing_allows_baseline_without_installed_delphirf(tmp_path):
    html = run_in_jsc(tmp_path, """
api.state.environment = {preprocessing: true, delphirf: false,
  detail: 'compatible local DelphiRF preprocessing'};
try { api.startRun(); } catch (e) {}
""", exports="startRun, state", node="run-result")
    assert "The selected DelphiRF model is not ready" not in html
    assert "DelphiRF preprocessing is not ready" not in html


ORIGIN_CHART_CALL = """
const rows = [], methods = ['baseline_null', 'rr_delphirf3'], cutoffs = [];
for (let i = 0; i < %d; i++) {
  const cutoff = `2020-${String(1 + (i %% 12)).padStart(2, '0')}-${String(1 + (i %% 27)).padStart(2, '0')}`;
  cutoffs.push(cutoff);
  methods.forEach((method, m) => rows.push(
    {cutoff, method, n: 5, mean_ae: %s}));
}
api.drawOriginChart(rows, methods, cutoffs);
"""


def plotted_points(svg: str):
    return [(float(x), float(y)) for x, y in re.findall(r"[ML](-?[\d.]+),(-?[\d.]+)", svg)]


@pytest.mark.parametrize("origins", [2, 40, 172])
def test_origin_chart_points_land_inside_the_viewbox(tmp_path, origins):
    svg = run_in_jsc(tmp_path, ORIGIN_CHART_CALL % (origins, "100 + i * (m + 1)"))
    points = plotted_points(svg)
    assert len(points) == origins * 2
    # pad: left 62, right 18 of 720; top 14, bottom 38 of 280.
    assert all(62 <= x <= 702 + 1e-9 for x, _ in points)
    assert all(14 - 1e-9 <= y <= 242 + 1e-9 for _, y in points)


def test_origin_chart_labels_the_last_origin_so_the_axis_does_not_stop_short(tmp_path):
    svg = run_in_jsc(tmp_path, ORIGIN_CHART_CALL % (172, "100 + i * (m + 1)"))
    labels = re.findall(r'<text x="([\d.]+)" y="264"[^>]*>([\d-]+)</text>', svg)
    assert labels, "no dated labels on the x axis"
    assert float(labels[-1][0]) == pytest.approx(702, abs=0.5), "last origin is unlabelled"
    positions = [float(x) for x, _ in labels]
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert min(gaps) > 40, f"x labels collide: gaps {gaps}"


def test_origin_chart_axis_ticks_are_readable_at_count_scale(tmp_path):
    """Counts run to thousands, where toPrecision alone yields "1.10e+3"."""
    svg = run_in_jsc(tmp_path, ORIGIN_CHART_CALL % (12, "1000 + i * 400"))
    ticks = re.findall(r'text-anchor="end"[^>]*>([^<]+)</text>', svg)
    assert ticks, "no y-axis ticks"
    assert not any("e+" in tick for tick in ticks), f"exponent notation on the axis: {ticks}"
    assert any(tick.endswith("k") for tick in ticks), f"thousands not abbreviated: {ticks}"


def test_origin_chart_survives_a_method_that_is_exactly_right(tmp_path):
    """An all-zero error column would otherwise divide by zero and plot NaN."""
    svg = run_in_jsc(tmp_path, ORIGIN_CHART_CALL % (6, "0"))
    assert "NaN" not in svg
    assert plotted_points(svg), "nothing was plotted"


def test_origin_chart_drops_markers_once_they_would_merge(tmp_path):
    few = run_in_jsc(tmp_path, ORIGIN_CHART_CALL % (10, "100 + i"))
    many = run_in_jsc(tmp_path, ORIGIN_CHART_CALL % (172, "100 + i"))
    assert few.count("<circle") == 20
    assert many.count("<circle") == 0


def test_origin_chart_names_every_series_outside_the_colour(tmp_path):
    """Three of the eight slots fail light-mode contrast; the relief is a legend."""
    svg = run_in_jsc(tmp_path, ORIGIN_CHART_CALL % (6, "100 + i"))
    assert svg.count('class="swatch"') == 2
    assert "Latest reported value" in svg and "rr delphirf3" in svg
    assert 'role="img"' in svg and "aria-label" in svg


def test_series_colours_are_assigned_in_fixed_slot_order_never_cycled(tmp_path):
    svg = run_in_jsc(tmp_path, ORIGIN_CHART_CALL % (6, "100 + i"))
    assert "var(--series-1)" in svg and "var(--series-2)" in svg


def test_the_stylesheet_defines_every_categorical_slot_in_both_modes(stylesheet):
    for slot in range(1, 9):
        # light :root, the dark media query, and the explicit dark theme scope.
        assert stylesheet.count(f"--series-{slot}:") == 3, f"--series-{slot} is not defined 3 times"


def test_a_flat_curve_does_not_divide_by_zero():
    """Every candidate horizon equally good: max_error floors at 0.12."""
    _, _, pad, _, ys = chart_geometry([0, 7, 14], [0.0, 0.0, 0.0])
    assert all(y == pad["top"] + (240 - pad["top"] - pad["bottom"]) for y in ys)


def test_an_error_above_one_still_scales_inside_the_plot():
    """A stream that revises by more than 100% must not overflow the axis."""
    width, height, pad, _, ys = chart_geometry([0, 7, 14], [3.5, 1.2, 0.05])
    assert all(pad["top"] - 1e-9 <= y <= height - pad["bottom"] + 1e-9 for y in ys)


def test_the_chart_degrades_when_there_is_nothing_to_plot(script):
    """Fewer than two candidate horizons is a message, not a broken axis."""
    assert "points.length < 2" in script


def test_a_refresh_restores_the_last_run(script):
    assert "async function restoreLastRun()" in script
    assert "restoreLastRun();" in script
    assert "snapshot.state === 'running'" in script


def test_completion_chart_has_quantile_band_and_named_y_axis(script):
    assert "target_lag_completion_band" in script
    assert "10th–90th percentile" in script
    assert ">relative error</text>" in script
    assert "Not enough candidate horizons" in script


def test_the_chart_ships_a_table_view_of_the_same_numbers(script, html):
    """Accessibility: identity and value are never carried by the mark alone."""
    assert "renderCompletionTable" in script
    assert 'id="completion-table"' in html
    assert "Show these numbers as a table" in html


def test_the_chart_marks_the_threshold_the_rule_is_defined_against(script):
    assert "10% tolerance" in script
    assert "const tolerance = 0.10" in script


def test_the_chart_has_a_hover_layer(script):
    assert "attachChartHover" in script
    assert "mouseenter" in script and "focus" in script     # pointer and keyboard


def test_the_chart_uses_one_series_colour_and_ink_tokens_for_text(script):
    """Text wears text tokens; only the mark carries the series colour."""
    assert "var(--series-1)" in script
    for text_element in re.findall(r"<text[^>]*>", script):
        assert "var(--series-1)" not in text_element, text_element


# --- styling --------------------------------------------------------------

def test_dark_mode_is_defined_under_both_scopes(stylesheet):
    """The OS setting and an explicit theme stamp must both work."""
    assert "@media (prefers-color-scheme: dark)" in stylesheet
    assert ':root:not([data-theme="light"])' in stylesheet
    assert ':root[data-theme="dark"]' in stylesheet


def test_every_colour_token_has_a_light_definition(stylesheet):
    """No token may exist only inside a dark block."""
    root_block = stylesheet.split(":root {", 1)[1].split("}", 1)[0]
    light_tokens = set(re.findall(r"(--[a-z0-9-]+):", root_block))
    dark_block = stylesheet.split(':root[data-theme="dark"] {', 1)[1].split("}", 1)[0]
    dark_tokens = set(re.findall(r"(--[a-z0-9-]+):", dark_block))
    assert dark_tokens <= light_tokens, f"dark-only tokens: {sorted(dark_tokens - light_tokens)}"


def test_the_page_is_usable_at_phone_width(stylesheet):
    assert "@media (max-width: 520px)" in stylesheet
    assert "max-width: 860px" in stylesheet      # the content column is bounded


def test_reduced_motion_is_respected(stylesheet):
    assert "prefers-reduced-motion" in stylesheet
