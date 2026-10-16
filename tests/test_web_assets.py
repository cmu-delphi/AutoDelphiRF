"""Static checks on the page the server ships.

There is no browser in the test environment, so these do what a browser would
have caught: every element the script reaches for exists in the markup, every
endpoint it calls is routed, and the interactive result views retain their
required controls. A typo in an element id is otherwise a silent no-op at runtime.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest

from autodelphirf.app import web
from autodelphirf.core.resources import resource_path


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
    created_at_runtime = {"completion-area-status", "completion-lag-input",
                          "completion-use-lag",
                          "col-reference", "col-report", "col-value", "col-geo",
                          "col-denominator"}
    missing = referenced_ids(script) - present - created_at_runtime
    assert not missing, f"app.js references ids that do not exist: {sorted(missing)}"


def test_ids_created_at_runtime_are_actually_created(script):
    """The exemptions above must be real, not a way to hide a typo.

    Two mechanisms create them: an inline ``id="..."`` in a template literal,
    or the ``columnSelect`` helper, which stamps the id it is handed. Either
    way the literal has to appear in the file.
    """
    assert 'id="${id}"' in script, "columnSelect no longer stamps the id it is given"
    for identifier in ("completion-area-status", "completion-lag-input",
                       "completion-use-lag",
                       "col-reference", "col-report",
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
    assert "acceptLag(value, false)" in script
    accept_lag = script.split("function acceptLag", 1)[1].split("function initSchedule", 1)[0]
    assert "show('card-schedule')" in accept_lag
    assert "show('card-methods')" in accept_lag
    assert "unlockStage(2)" in accept_lag
    assert "goToStage(2)" in accept_lag


def test_workflow_navigation_is_gated_and_page_based(html, script):
    assert html.count('data-wizard-stage=') == 4
    assert html.count('data-wizard-page=') == 4
    assert 'if (stage > state.unlockedStage) return' in script
    assert "button.disabled = stage > state.unlockedStage" in script
    assert "page.hidden = Number(page.dataset.wizardPage) !== state.currentStage" in script


def test_intro_explains_automation_independence_and_retrospective_use(html):
    copy = " ".join(html.split())
    assert "without preliminary data exploration" in copy
    assert "independent for downstream use" in copy
    assert "mimics real-time forecasting" in copy
    assert "rolling training and evaluation windows" in copy
    assert "This web interface is in development" in html
    assert "Feedback" in html and "welcome" in html


def test_decorative_step_numbers_and_beta_badge_are_absent(html):
    assert 'class="step"' not in html
    assert 'class="development-mark"' not in html
    assert "β" not in html
    assert not re.search(r'data-wizard-stage="\d"[^>]*>\s*<span>\d', html)


def test_diagnosis_summary_contains_only_the_requested_archive_facts(script):
    block = script.split("$('diagnosis-facts').innerHTML", 1)[1].split("].join('');", 1)[0]
    for label in ("Rows", "Locations", "Reference dates",
                  "Report dates", "Reference-date cadence", "Report-date cadence"):
        assert f"fact('{label}'" in block
    assert "Genuine revisions" not in block
    assert "Revision feature lags" not in block
    assert "Training window" not in block


def test_training_window_is_a_visible_configure_analysis_control(html, script):
    schedule = html.split('id="card-schedule"', 1)[1].split('id="card-methods"', 1)[0]
    advanced = html.split('<details class="advanced">', 1)[1].split('</details>', 1)[0]
    assert 'id="training-days"' in schedule
    assert "Default: 180 days" in schedule
    assert 'id="training-days"' not in advanced
    assert "request.training_days = trainingDays" in script


def test_schedule_preview_is_a_compact_actual_settings_summary(script):
    block = script.split("function renderSchedule(preview) {", 1)[1].split(
        "/* --- the completion curve", 1)[0]
    assert 'class="schedule-summary"' in block
    assert "day training window" in block
    assert "Retrain every" in block
    assert "rolling fits" in block
    assert '<dl class="facts">' not in block


def test_schedule_summary_responds_to_all_three_schedule_inputs(script):
    block = script.split("function renderSchedule(preview) {", 1)[1].split(
        "/* --- the completion curve", 1)[0]
    assert "$('training-days').value" in block
    assert "preview.first_origin" in block
    assert "preview.retrain_days" in block


def test_concept_diagram_explains_windows_before_the_inputs(html):
    schedule = html.split('id="card-schedule"', 1)[1].split('id="card-methods"', 1)[0]
    copy = " ".join(schedule.split())
    assert schedule.index('class="rolling-concept"') < schedule.index('id="training-days"')
    assert 'title="Training window"' in copy
    assert 'title="Held-out evaluation window"' in copy
    assert "Origin 2" in copy and "Last origin" in copy
    assert "future-information leakage" in copy
    assert "recreates real-time revision forecasting" in copy
    assert "retrained less often" in copy


def test_concept_windows_are_locked_to_one_grid_row(stylesheet):
    assert ".concept-row > * { grid-row: 1; }" in stylesheet


def test_first_evaluation_window_ends_at_the_consecutive_next_origin(stylesheet):
    assert ".concept-testing.first { grid-column: 8 / 9; }" in stylesheet
    assert ".concept-origin.next { grid-column: 9; }" in stylesheet


def test_concept_uses_three_named_origins_and_an_ellipsis(html, stylesheet):
    assert '<span class="concept-row-name">Origin 1</span>' in html
    assert '<span class="concept-row-name">Origin 2</span>' in html
    assert '<span class="concept-row-name">Last origin</span>' in html
    assert 'class="concept-gap"' in html
    assert "Calendar time" in html
    assert ".concept-origin i" not in stylesheet
    assert "height: 28px" in stylesheet


def test_concept_evaluation_windows_have_equal_widths(stylesheet):
    assert ".concept-testing.first { grid-column: 8 / 9; }" in stylesheet
    assert ".concept-testing.next { grid-column: 9 / 10; }" in stylesheet
    assert ".concept-testing.last { grid-column: 12 / 13; }" in stylesheet


def test_concept_marks_recent_dates_without_available_target_values(html, stylesheet):
    assert "Target-lag values<br>not yet available" in html
    assert 'class="concept-unavailable"' in html
    assert "unavailable-key" not in html
    assert ".concept-unavailable" in stylesheet
    assert "border-left: 2px solid var(--border-strong)" in stylesheet
    assert "repeating-linear-gradient" in stylesheet
    explanation = html.split('class="rolling-explanation"', 1)[1].split('</div>', 1)[0]
    assert "not yet available" not in explanation


def test_rolling_explanation_is_two_plain_full_width_paragraphs(html, stylesheet):
    explanation = html.split('class="rolling-explanation"', 1)[1].split('</div>', 1)[0]
    assert explanation.count("<p>") == 2
    assert "future-information leakage" in explanation
    assert "retrained less often" in explanation
    assert ".rolling-explanation p + p { margin-top: 10px; }" in stylesheet
    paragraph_rule = stylesheet.split(".rolling-explanation p {", 1)[1].split("}", 1)[0]
    assert "border-left" not in paragraph_rule


def test_visible_date_ranges_and_origins_name_their_date_axis(html, script):
    copy = " ".join((html + script).split())
    assert "Report-date origins:" in copy
    assert "First report-date retraining origin" in copy
    assert "report-date origin" in copy
    assert "days after the reference date" in copy
    assert "Reference dates" in copy and "Report dates" in copy


def test_workflow_stages_unlock_only_after_their_prerequisites(script):
    assert "unlockStage(1)" in script       # diagnosis returned
    assert "unlockStage(2)" in script       # target lag accepted
    assert "unlockStage(3)" in script       # assessment results loaded
    assert "unlockStage(4)" not in script


def test_four_stage_workflow_has_one_explicit_grid_column_per_node_and_arrow(stylesheet):
    assert "minmax(150px, 1fr) 24px minmax(190px, 1fr) 24px" in stylesheet
    assert "minmax(150px, 1fr) 24px minmax(150px, 1fr);" in stylesheet
    assert "white-space: nowrap" in stylesheet


def test_rolling_configuration_and_run_share_one_workflow_page(html, script):
    rolling_page = html.split('data-wizard-page="2"', 1)[1].split(
        'data-wizard-page="3"', 1)[0]
    assert "Recreate historical real-time forecasts" in rolling_page
    assert 'id="card-methods"' in rolling_page
    assert 'id="card-run"' in rolling_page
    assert "Rolling forecast</button>" in " ".join(html.split())
    start = script.split("async function startRun()", 1)[1].split(
        "function pollStatus", 1)[0]
    assert "goToStage(2)" in start
    assert "goToStage(3)" not in start


def test_target_lag_wording_matches_the_selection_rule(html, script):
    copy = " ".join((html + script).split())
    assert "choose the target lag yourself" in copy
    assert "use your value exactly as entered" in copy
    assert "Inside the suggested area" not in copy
    assert "Outside the suggested area" not in copy


def test_target_lag_figure_and_table_support_linked_exploration_and_selection(script):
    assert 'class="completion-band q95"' in script
    assert 'class="completion-median"' in script
    assert 'class="completion-hit-area"' in script
    assert "hitArea.addEventListener('pointermove'" in script
    assert "renderPoint(pointFromPointer(event))" in script
    assert 'id="completion-lag-input"' in script
    assert "lagInput.addEventListener('input'" in script
    assert "updateGuide(point)" in script
    assert "$('custom-lag-input').value = point.lag" in script
    assert "$('completion-use-lag').addEventListener('click'" in script
    assert "acceptLag(requested, false)" in script


def test_diagnosis_names_both_date_axis_cadences(script):
    assert "fact('Reference-date cadence', report.reference_axis_resolution," in script
    assert "fact('Report-date cadence', report.report_axis_resolution," in script
    assert "fact('Cadence', report.temporal_resolution)" not in script


# --- routes ---------------------------------------------------------------

def test_every_endpoint_the_script_calls_is_routed_by_the_server(script):
    called = set(re.findall(r"""['"`](/api/[a-z_]+)""", script))
    assert called, "no API endpoints found in app.js; the pattern is stale"
    server_source = Path(web.__file__).read_text()
    for endpoint in called:
        assert f'"{endpoint}"' in server_source, f"{endpoint} is called but not routed"


def test_results_view_is_the_focused_web_evaluation_not_the_full_report(html, script):
    assert "Open the full report" not in html + script
    assert "Absolute error (AE)" in html
    assert "AE over reporting lags" in html
    assert "AE by location at early reporting lags" in html
    assert "earliest well-supported reporting lag" in html
    assert "first well-supported lag at or" in html
    assert "at least 80%" in html
    assert html.count('data-result-metric="ae"') == 1
    assert html.count('data-result-metric="wis"') == 1
    assert 'data-result-metric="ae">AE</button>' in html
    assert 'id="lag-score-chart"' in html
    assert 'id="lag-wis-chart"' not in html
    assert 'id="lag-ae-chart"' not in html
    assert 'id="location-score-chart"' in html
    assert "reporting lag ≤14 days" in html
    assert "Whiskers span the 10th–90th percentiles" in html
    assert '<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"' in script
    assert "drawLagEvaluation();" in script
    assert "drawLocationEvaluation();" in script
    assert "drawEarlyLagCharts();" in script
    assert "raw scale" not in (html + script).lower()


def test_time_series_charts_share_axes_and_sit_in_one_row(script, stylesheet):
    assert "yMax: sharedYAxis.maximum" in script
    assert "yTicks: sharedYAxis.ticks" in script
    assert "xKeys: sharedXKeys" in script
    assert "function niceAxis(" in script
    assert "Number.isFinite(options.yMax)" in script
    assert "options.xKeys ||" in script
    assert "options.yTicks ||" in script
    assert ".early-lag-grid" in stylesheet
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in stylesheet


def test_method_selector_uses_current_names_and_explanations_even_before_server_restart(script):
    assert "delphirf: 'RevRoute-DelphiRF'" in script
    assert "naive_delphirf: 'Naive DelphiRF'" in script
    assert "global_delphirf: 'Global DelphiRF'" in script
    assert "The primary AutoDelphiRF method" in script
    assert "small reporting lags are modeled separately" in script
    assert "combines all locations and reporting lags" in script
    assert "sparse reporting histories" in script
    assert "WEB_HIDDEN_METHODS.has(layer.name)" in script


def test_archive_terms_and_value_shapes_are_explained_in_plain_language(html):
    copy = " ".join(html.split())
    assert "Reference date:</strong> when the measured event or surveillance period occurred" in copy
    assert "Report date:</strong> when that value was released or updated" in copy
    assert "Multiple report dates for the same location and reference date" in copy
    assert "A single reported value" in copy
    assert "A numerator and denominator" in copy
    assert "A count or fraction" not in copy


def test_diagnosis_help_and_lag_definitions_are_available_without_replacing_labels(html, script):
    copy = " ".join((html + script).split())
    assert 'class="term-help"' in script
    assert "How often new measured events or surveillance periods occur" in script
    assert "How often reports or updates arrive" in script
    assert "How long until reported values are sufficiently stable?" in html
    assert "Target lag</dt>" in script and "Reporting lag</dt>" in script


def test_evaluation_explains_ae_and_wis_before_the_figures(html):
    definitions = html.index('class="score-definitions"')
    figures = html.index('id="lag-score-grid"')
    assert definitions < figures
    copy = " ".join(html.split())
    assert "Absolute error equals the absolute value of the prediction minus the target" in copy
    assert '<math display="block"' in copy
    assert '<mi mathvariant="normal">AE</mi>' in copy
    assert '<mi mathvariant="normal">WIS</mi>' in copy
    assert "<munderover>" in copy and "<mfrac>" in copy
    assert '<mover accent="true"><mi>q</mi><mo>ˆ</mo></mover>' in copy
    assert "estimated predictive quantile" in copy
    assert "Weighted interval score (WIS)" in copy
    assert "Evaluates the full probabilistic nowcast" in copy
    assert "standard quantile-loss function" in copy


def test_no_external_resources_are_referenced(html, script, stylesheet):
    """This page is served by a local process that may have no internet access."""
    for name, text in (("index.html", html), ("app.js", script), ("style.css", stylesheet)):
        for pattern in ("http://", "https://"):
            for match in re.findall(rf'{pattern}[^\s"\'<>)]+', text):
                # Links a person clicks are fine; fetched subresources are not.
                assert "github.com" in match, f"{name} loads an external resource: {match}"


# --- interactive result views --------------------------------------------


def run_in_jsc(tmp_path, call: str, exports="drawOriginChart", node="origin-chart") -> str:
    """Run part of app.js in JavaScriptCore against a stub DOM, return one node's HTML.

    The target-lag view is a semantic table-like grid and is checked statically.
    For the per-origin chart the shipped function is run directly, so the thing
    under test is the code that ships rather than a second copy of its maths.
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
    assert "as the README describes" in html, \
        "the refusal does not say how to fix it"
    assert "DelphiRF predates what RevRoute needs." in html, "the reason is not passed through"


def test_a_ready_environment_does_not_block_the_run(tmp_path):
    """The guard must not be a permanent roadblock: with DelphiRF present the
    run proceeds (and this harness has no network, so it fails at fetch)."""
    html = run_in_jsc(tmp_path, """
api.state.environment = {delphirf: true, detail: 'installed DelphiRF'};
try { api.startRun(); } catch (e) {}
""", exports="startRun, state", node="run-result")
    assert "is not ready" not in html


def test_preprocessing_alone_allows_baseline_without_delphirf_models(tmp_path):
    html = run_in_jsc(tmp_path, """
api.state.environment = {preprocessing: true, delphirf: false,
  detail: 'installed DelphiRF'};
try { api.startRun(); } catch (e) {}
""", exports="startRun, state", node="run-result")
    assert "The selected DelphiRF model is not ready" not in html
    assert "DelphiRF preprocessing is not ready" not in html


BAND_CHART_CALL = """
const rows = [], methods = ['baseline_null', 'rr_delphirf3'];
for (let lag = 1; lag <= %d; lag++) {
  methods.forEach((method, m) => rows.push({
    lag, method, metric: 'ae', n_locations: 5,
    median: %s, q10: %s, q90: %s,
  }));
}
api.drawQuantileBandChart('band-chart', rows, {
  xField: 'lag', xLabel: 'reporting lag (days)', yLabel: 'absolute error', xKind: 'number',
});
"""


def plotted_points(svg: str):
    return [(float(x), float(y)) for x, y in re.findall(r"[ML](-?[\d.]+),(-?[\d.]+)", svg)]


@pytest.mark.parametrize("lags", [2, 40, 172])
def test_quantile_band_chart_points_land_inside_the_viewbox(tmp_path, lags):
    svg = run_in_jsc(
        tmp_path,
        BAND_CHART_CALL % (lags, "100 + lag * (m + 1)", "80 + lag", "120 + lag"),
        exports="drawQuantileBandChart", node="band-chart")
    points = plotted_points(svg)
    assert len(points) == lags * 2
    # pad: left 88, right 20 of 960; top 14, bottom 52 of 350.
    assert all(88 <= x <= 940 + 1e-9 for x, _ in points)
    assert all(14 - 1e-9 <= y <= 298 + 1e-9 for _, y in points)
    assert svg.count("<polygon") == 2, "one 10th–90th percentile band per method"


def test_quantile_band_chart_survives_zero_error(tmp_path):
    svg = run_in_jsc(
        tmp_path, BAND_CHART_CALL % (6, "0", "0", "0"),
        exports="drawQuantileBandChart", node="band-chart")
    assert "NaN" not in svg
    assert plotted_points(svg), "median lines were not plotted"


def test_location_chart_draws_grouped_bars_and_standard_deviation_whiskers(tmp_path):
    html = run_in_jsc(tmp_path, """
const rows = [
  {geo_value: 'aa', method: 'baseline_null', metric: 'ae', n: 20, median: 4, q10: 1, q90: 6},
  {geo_value: 'aa', method: 'delphirf', metric: 'ae', n: 20, median: 3, q10: 0.5, q90: 5},
  {geo_value: 'bb', method: 'baseline_null', metric: 'ae', n: 18, median: 8, q10: 2, q90: 12},
  {geo_value: 'bb', method: 'delphirf', metric: 'ae', n: 18, median: 5, q10: 1.5, q90: 9},
];
api.drawLocationBarChart('location-chart', rows, {yLabel: 'Median absolute error'});
""", exports="drawLocationBarChart", node="location-chart")
    assert html.count("<rect") == 4
    assert "10th–90th percentile 2–12" in html and "n=18" in html
    assert "through 14 days" in html


def test_quantile_band_chart_names_methods_and_uses_fixed_colour_slots(tmp_path):
    svg = run_in_jsc(
        tmp_path, BAND_CHART_CALL % (6, "100 + lag", "80 + lag", "120 + lag"),
        exports="drawQuantileBandChart", node="band-chart")
    assert svg.count('class="swatch"') == 2
    assert "Latest reported value" in svg and "rr delphirf3" in svg
    assert "var(--series-1)" in svg and "var(--series-2)" in svg
    assert 'role="img"' in svg and "aria-label" in svg


def test_the_stylesheet_defines_every_categorical_slot_in_both_modes(stylesheet):
    for slot in range(1, 9):
        # light :root, the dark media query, and the explicit dark theme scope.
        assert stylesheet.count(f"--series-{slot}:") == 3, f"--series-{slot} is not defined 3 times"


def test_target_lag_table_degrades_when_there_is_nothing_to_show(script):
    """No candidate target lags is a message, not an empty control."""
    assert "if (!points.length)" in script
    assert "No candidate target lags could be evaluated" in script


def test_a_refresh_restores_the_last_run(script):
    assert "async function restoreLastRun()" in script
    assert "restoreLastRun();" in script
    assert "snapshot.state === 'running'" in script


def test_target_lag_table_has_all_requested_percentiles(script):
    assert "target_lag_completion_band" in script
    for quantile in ("q60", "q75", "q90", "q95"):
        assert f"point.{quantile}" in script
    for key in ("median", "q60", "q75", "q90", "q95"):
        assert f"data-percentile-cell=\"${{key}}\"" in script or key == "median"
    assert 'class="percentile-explorer"' in script


def test_target_lag_figure_has_a_fixed_zero_to_365_day_display_without_limiting_selection(script):
    assert "const minLag = 0;" in script
    assert "const maxLag = 365;" in script
    assert "allPoints.filter((point) => point.lag >= 0 && point.lag <= 365)" in script
    assert "const xTicks = [0, 60, 120, 180, 240, 300, 365]" in script
    assert "allPoints.find((point) => point.lag === requested)" in script
    assert "outside the displayed 0–365 day range" in script


def test_target_lag_figure_names_both_axes(script):
    assert "Target lag (days after reference date)</text>" in script
    assert "Relative error (%)</text>" in script


def test_target_lag_figure_explains_relative_error_and_percentiles_for_new_users(html, script):
    assert "How to read this figure" in html
    assert "1% error means" in html
    assert "The median is" in html
    assert "90% of records had no more" in html
    for label in ("60th percentile", "75th percentile", "90th percentile", "95th percentile"):
        assert label in script


def test_target_lag_figure_has_nested_percentile_layers_and_both_tolerances(script):
    for quantile in ("q60", "q75", "q90", "q95"):
        assert f"percentilePath('{quantile}')" in script
    assert "const toleranceLines = [0.01, 0.10]" in script
    assert "Percentile layers" in script


def test_target_lag_suggested_range_is_drawn_and_labelled_on_the_figure(html, script, stylesheet):
    assert 'class="completion-suggested-area"' in script
    assert script.count('class="completion-suggested-boundary"') == 2
    assert 'class="completion-suggested-bracket"' in script
    assert "Suggested lag range</text>" in script
    assert script.count('class="completion-suggested-lag-label"') == 2
    assert "${suggestedStart} days" in script and "${suggestedEnd} days" in script
    assert ".completion-suggested-area { fill: var(--development-soft); }" in stylesheet
    assert "stroke: var(--development)" in stylesheet
    assert 'class="range-method"' not in html


def test_target_lag_explanation_states_the_accuracy_timeliness_tradeoff(script):
    copy = " ".join(script.split())
    assert "finalized value is unavailable" in copy
    assert "Choosing a target lag is therefore a compromise" in copy
    assert "Accuracy." in copy
    assert "Timeliness." in copy
    assert "reporting patterns change abruptly" in copy


def test_target_lag_table_puts_the_input_before_the_statistics(script):
    selector = script.index('class="lag-selector"')
    percentiles = script.index('class="percentile-cell"')
    assert selector < percentiles
    assert "Target lag, in days" in script
    assert "Median" in script and "th percentile" in script


def test_target_lag_table_marks_both_tolerance_levels(script):
    assert "value <= 0.01" in script
    assert "value <= 0.10" in script
    for level in ("within-one", "within-ten", "above-ten"):
        assert level in script
    assert "Within 1% error" in script and "Within 10% error" in script


def test_pre_diagnosis_interaction_uses_a_readable_font_floor(stylesheet):
    assert ".completion-legend" in stylesheet and "font-size: 15px" in stylesheet
    assert ".completion-axis-label { fill: var(--text-secondary); font-size: 14px; }" in stylesheet
    assert ".percentile-cell > span { color: var(--text-primary); font-size: 15px" in stylesheet
    assert ".percentile-cell small { margin-top: auto; font-size: 15px" in stylesheet
    assert ".lag-area-status { color: var(--series-1); font-size: 14px" in stylesheet


def test_target_lag_table_uses_direct_entry_limited_to_available_lags(script):
    assert 'type="number" id="completion-lag-input"' in script
    assert 'max=' not in script[script.index('id="completion-lag-input"') - 50:
                                  script.index('id="completion-lag-input"') + 150]
    assert "lagInput.addEventListener('input'" in script
    assert "No data exist at lag" in script
    assert "Nearest available" in script
    assert "$('completion-use-lag').disabled = true" in script
    assert "availableSet.has(requested)" in script
    assert "report.available_target_lags" in script
    assert "nearest diagnostic checkpoint" not in script


def test_available_lag_state_is_declared_inside_the_interactive_view_scope(script):
    view = script.split("function drawCompletionCurve(report) {", 1)[1].split(
        "/* --- 5. methods", 1)[0]
    question = script.split("function renderLagQuestion(report, userTargetLag) {", 1)[1].split(
        "function acceptLag", 1)[0]
    assert "const availableSet = new Set(availableLags);" in view
    assert "availableSet.has(requested)" in view
    assert "availableSet" not in question


def test_target_lag_table_has_distinct_styles_for_each_tolerance_level(stylesheet):
    assert ".percentile-cell.within-one" in stylesheet
    assert ".percentile-cell.within-ten" in stylesheet
    assert ".percentile-cell.above-ten" in stylesheet


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
    assert "max-width: 1180px" in stylesheet      # one bounded grid across the page


def test_reduced_motion_is_respected(stylesheet):
    assert "prefers-reduced-motion" in stylesheet
