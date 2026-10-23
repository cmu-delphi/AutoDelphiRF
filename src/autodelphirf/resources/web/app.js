/* AutoDelphiRF local UI.
 *
 * One dataset, six steps, one question that genuinely needs answering (the
 * target lag L). Vanilla JS on purpose: this page is served by a local
 * process that may have no internet access, so nothing is fetched from a CDN.
 */
'use strict';

const TOKEN = document.body.dataset.token;
const state = {
  columns: [],
  diagnosis: null,
  answeredLag: null,        // the L the user settled on
  confirmLag: false,        // retained in the request schema for compatibility
  environment: null,        // R/DelphiRF readiness, as reported at startup
  schedule: null,           // the retraining calendar last previewed by the server
  results: null,            // the finished run's summary, as /api/results returned it
  polling: null,
  unlockedStage: 0,         // later workflow pages stay gated until prior work succeeds
  currentStage: 0,
};

/* --- plumbing --------------------------------------------------------- */

async function api(path, {method = 'GET', body = null, raw = false} = {}) {
  // The documentation demo supplies an in-page, read-only API fixture so the
  // same interface can be reviewed without starting the local web service.
  // Production pages do not define this hook and continue to use fetch below.
  if (typeof window.AUTODELPHIRF_DEMO_API === 'function') {
    return window.AUTODELPHIRF_DEMO_API(path, {method, body, raw});
  }
  const headers = {'X-AutoDelphiRF-Token': TOKEN};
  if (body && !raw) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {
    method,
    headers,
    body: raw ? body : (body ? JSON.stringify(body) : null),
  });
  const payload = await response.json().catch(() => ({error: `HTTP ${response.status}`}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

const $ = (id) => document.getElementById(id);

function show(id) { $(id).hidden = false; }

function renderWizard() {
  document.querySelectorAll('[data-wizard-page]').forEach((page) => {
    page.hidden = Number(page.dataset.wizardPage) !== state.currentStage;
  });
  document.querySelectorAll('[data-wizard-stage]').forEach((button) => {
    const stage = Number(button.dataset.wizardStage);
    button.disabled = stage > state.unlockedStage;
    button.classList.toggle('completed', stage < state.unlockedStage);
    if (stage === state.currentStage) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
}

function unlockStage(stage) {
  state.unlockedStage = Math.max(state.unlockedStage, stage);
  renderWizard();
}

function goToStage(stage) {
  if (stage > state.unlockedStage) return;
  state.currentStage = stage;
  renderWizard();
  const heading = document.querySelector(`[data-wizard-page="${stage}"] h2`);
  if (heading) {
    heading.tabIndex = -1;
    heading.focus({preventScroll: true});
  }
  window.scrollTo({top: 0, behavior: 'smooth'});
}

function initWizard() {
  document.querySelectorAll('[data-wizard-stage]').forEach((button) => {
    button.addEventListener('click', () => goToStage(Number(button.dataset.wizardStage)));
  });
  renderWizard();
}

function notice(kind, html) {
  return `<div class="notice ${kind}">${html}</div>`;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => (
    {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[index]}`;
}

/* Nothing on this page is worth debugging from a blank screen. An uncaught
 * error used to leave a control that simply did nothing, with the message
 * parked in whichever status line the handler happened to know about --
 * often one scrolled far off-screen. Failures are shown here instead. */
function showFailure(message) {
  const banner = $('js-failure');
  banner.hidden = false;
  banner.innerHTML = `<strong>The page hit an error.</strong>
    <code>${escapeHtml(String(message))}</code>
    <span class="hint">Nothing was lost. Reload to start over, and report this line.</span>`;
}

function clearFailure() {
  $('js-failure').hidden = true;
}

function scrollIntoView(id) {
  $(id).scrollIntoView({behavior: 'smooth', block: 'start'});
}

/* --- 1. environment --------------------------------------------------- */

async function loadEnvironment() {
  const body = $('environment-body');
  try {
    const report = await api('/api/environment');
    state.environment = report;
    if (report.preprocessing) {
      body.innerHTML = `<span class="pill good">Ready</span>
        <span>Preprocessing, the latest-value reference, and revision-pattern matching are ready &mdash;
        ${escapeHtml(report.detail)}.</span>`;
      if (!report.delphirf) {
        body.insertAdjacentHTML('afterend', notice('warn',
          `The installed DelphiRF model package still needs updating. You can run
           the latest-value reference and revision-pattern matching now; DelphiRF methods
           remain unavailable until you reinstall DelphiRF as the README describes and
           reload this page.`));
      }
    } else {
      body.innerHTML = `<span class="pill warn">Diagnosis only</span>
        <span>${escapeHtml(report.detail)}</span>`;
      body.insertAdjacentHTML('afterend', notice('warn',
        `You can still drop an archive and see the full diagnosis. Building a prepared
         triangle needs R and DelphiRF:<br>
         <code>Rscript -e 'install.packages(c("remotes","jsonlite","data.table","arrow")); remotes::install_github("cmu-delphi/DelphiRF@refactor-clean", dependencies=TRUE)'</code><br>
         then reload this page.`));
    }
    if (report.build) {
      const source = [report.build.branch, report.build.commit].filter(Boolean).join(' @ ');
      body.insertAdjacentHTML('beforeend', `<div class="hint">AutoDelphiRF
        ${escapeHtml(report.build.version)}${source ? ` (${escapeHtml(source)})` : ''}</div>`);
    }
  } catch (error) {
    body.innerHTML = `<span class="pill bad">Error</span>
      <span>${escapeHtml(error.message)}</span>`;
  }
}

/* --- 2. upload -------------------------------------------------------- */

function initDropzone() {
  const zone = $('dropzone');
  const input = $('file-input');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => {
    if (input.files.length) upload(input.files[0]);
  });

  ['dragenter', 'dragover'].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault();
    zone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault();
    if (name === 'dragleave' && zone.contains(event.relatedTarget)) return;
    zone.classList.remove('dragging');
  }));
  zone.addEventListener('drop', (event) => {
    const file = event.dataTransfer.files[0];
    if (file) upload(file);
  });

  // Dropping anywhere else on the page should not make the browser navigate
  // to the file, which looks like the app silently losing the upload.
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => event.preventDefault());
}

function upload(file) {
  const result = $('upload-result');
  const progress = $('upload-progress');
  const fill = $('upload-fill');
  result.hidden = true;
  progress.hidden = false;
  fill.style.width = '0%';
  $('upload-progress-text').textContent = `Sending ${escapeHtml(file.name)}…`;

  // XMLHttpRequest rather than fetch: it reports upload progress, which
  // matters for a multi-hundred-megabyte archive.
  const request = new XMLHttpRequest();
  request.open('POST', `/api/upload?name=${encodeURIComponent(file.name)}`);
  request.setRequestHeader('X-AutoDelphiRF-Token', TOKEN);
  request.upload.addEventListener('progress', (event) => {
    if (event.lengthComputable) {
      const percent = (event.loaded / event.total) * 100;
      fill.style.width = `${percent}%`;
      if (percent >= 100) $('upload-progress-text').textContent = 'Reading the archive…';
    }
  });
  request.addEventListener('load', () => {
    progress.hidden = true;
    let payload;
    try { payload = JSON.parse(request.responseText); } catch { payload = {}; }
    if (request.status >= 400) {
      result.hidden = false;
      result.innerHTML = notice('bad', escapeHtml(payload.error || `HTTP ${request.status}`));
      return;
    }
    onUploaded(payload, file);
  });
  request.addEventListener('error', () => {
    progress.hidden = true;
    result.hidden = false;
    result.innerHTML = notice('bad', 'The upload failed. Is <code>autodelphirf web</code> still running?');
  });
  request.send(file);
}

function onUploaded(payload, file) {
  state.columns = payload.columns;
  const zone = $('dropzone');
  zone.classList.add('has-file');
  zone.querySelector('.drop-title').textContent = file.name;
  zone.querySelector('.drop-sub').textContent = 'Click to choose a different file';

  const result = $('upload-result');
  result.hidden = false;
  result.innerHTML = notice('good',
    `Read <strong>${escapeHtml(payload.filename)}</strong> (${formatBytes(payload.size_bytes)})
     with ${payload.columns.length} columns.`);

  buildColumnForm(payload.guesses);
  show('card-columns');
  scrollIntoView('card-columns');
}

/* --- 3. columns ------------------------------------------------------- */

function columnSelect(id, label, selected, {allowNone = false} = {}) {
  const options = state.columns.map((column) =>
    `<option value="${escapeHtml(column)}"${column === selected ? ' selected' : ''}>
       ${escapeHtml(column)}</option>`).join('');
  const none = allowNone
    ? `<option value=""${selected ? '' : ' selected'}>— none (single location) —</option>`
    : '';
  return `<label for="${id}">${label}<select id="${id}">${none}${options}</select></label>`;
}

function buildColumnForm(guesses) {
  $('column-form').innerHTML = [
    columnSelect('col-reference', `<span class="field-term">Reference date${termHelp(
                   'Reference date',
                   'When it happened: the date of the event or surveillance period the value measures.'
                 )}</span>`, guesses.reference),
    columnSelect('col-report', `<span class="field-term">Report date${termHelp(
                   'Report date',
                   'When it was published: the date this version of the value was released or updated.'
                 )}</span>`, guesses.report),
    columnSelect('col-value', 'Value', guesses.value),
    columnSelect('col-geo', 'Location', guesses.geo, {allowNone: true}),
  ].join('');

  document.querySelectorAll('input[name="value-type"]').forEach((radio) => {
    radio.addEventListener('change', onValueTypeChange);
  });
  // Wrapped, not passed directly: as a listener runDiagnosis would receive the
  // click event as its `userTargetLag`, and an Event serialises to `{}`, which
  // the server then tries to read as a number. Assigning onclick rather than
  // adding a listener also keeps a second upload from stacking handlers.
  $('diagnose-button').onclick = () => runDiagnosis();
}

function onValueTypeChange() {
  const fraction = document.querySelector('input[name="value-type"]:checked').value === 'fraction';
  const field = $('denominator-field');
  field.hidden = !fraction;
  if (fraction && !field.innerHTML) {
    field.innerHTML = `<div class="field-grid">${
      columnSelect('col-denominator', 'Denominator column', null)}</div>`;
    const valueLabel = document.querySelector('label[for="col-value"]');
    if (valueLabel) valueLabel.childNodes[0].textContent = 'Numerator column';
  }
  if (!fraction) {
    const valueLabel = document.querySelector('label[for="col-value"]');
    if (valueLabel) valueLabel.childNodes[0].textContent = 'Value';
  }
}

function currentMapping() {
  const valueType = document.querySelector('input[name="value-type"]:checked').value;
  const valueCols = [$('col-value').value];
  if (valueType === 'fraction') {
    const denominator = $('col-denominator');
    if (!denominator || !denominator.value) {
      throw new Error('Choose a denominator column for a rate or share.');
    }
    valueCols.push(denominator.value);
  }
  return {
    reference_col: $('col-reference').value,
    report_col: $('col-report').value,
    geo_col: $('col-geo').value || null,
    value_cols: valueCols,
    value_type: valueType,
  };
}

async function runDiagnosis(userTargetLag = null) {
  const button = $('diagnose-button');
  const status = $('diagnose-status');
  button.disabled = true;
  clearFailure();
  status.innerHTML = '<span class="spinner"></span> Diagnosing the revision process…';
  try {
    const mapping = currentMapping();
    if (userTargetLag != null) mapping.target_lag = userTargetLag;
    state.diagnosis = await api('/api/diagnose', {method: 'POST', body: mapping});
    status.textContent = '';
    renderDiagnosis(state.diagnosis, userTargetLag);
    show('card-diagnosis');
    unlockStage(1);
    goToStage(1);
    // A re-diagnosis is an answer to the target-lag question, so scroll to the
    // Keep the decision control visible after a diagnosis refresh.
    if (userTargetLag != null) scrollIntoView('lag-answers');
  } catch (error) {
    status.innerHTML = `<span class="pill bad">${escapeHtml(error.message)}</span>`;
    // The custom-value box is the other place this is triggered from, and it
    // is nowhere near `diagnose-status`; say it where the click happened too.
    if (userTargetLag != null) {
      $('custom-lag-note').textContent = `Could not re-diagnose: ${error.message}`;
      showFailure(error.message);
    }
  } finally {
    button.disabled = false;
  }
}

/* --- 4. diagnosis and the target-lag question ------------------------- */

// A "?" button that reveals a one-sentence explanation of a term.
function termHelp(term, explanation) {
  return `<details class="term-help">
    <summary aria-label="Explain ${escapeHtml(term)}">?</summary>
    <span>${escapeHtml(explanation)}</span>
  </details>`;
}

function fact(term, value, explanation = '') {
  const help = explanation ? termHelp(term, explanation) : '';
  return `<div><dt>${term}${help}</dt><dd>${value}</dd></div>`;
}

function renderDiagnosis(report, userTargetLag) {
  $('diagnosis-facts').innerHTML = [
    fact('Rows', report.n_rows.toLocaleString(),
         'Raw archive rows retained after invalid and duplicate records are removed.'),
    fact('Locations', report.n_locations.toLocaleString(),
         'Distinct surveillance locations or reporting units in the archive.'),
    fact('Reference dates', `${report.reference_date_min} → ${report.reference_date_max}`,
         'The date range of the events or surveillance periods being measured.'),
    fact('Report dates', `${report.report_date_min} → ${report.report_date_max}`,
         'The date range over which values were released or updated.'),
    fact('Reference-date cadence', report.reference_axis_resolution,
         'How often new measured events or surveillance periods occur.'),
    fact('Report-date cadence', report.report_axis_resolution,
         'How often reports or updates arrive in the archive.'),
  ].join('');
  $('training-days').value = report.training_window_days || 180;

  const notes = (report.notes || []).map((note) => notice('warn', escapeHtml(note))).join('');
  $('diagnosis-notes').innerHTML = notes;

  drawCompletionCurve(report);
  renderLagQuestion(report, userTargetLag);
  if (report.retraining) initSchedule(report.retraining);
}

function renderLagQuestion(report, userTargetLag) {
  const range = report.target_lag_suggested_range || {};
  const body = $('lag-question-body');
  const answers = $('lag-answers');
  const custom = $('custom-lag');

  body.innerHTML = `<dl class="lag-definitions">
      <div><dt>Target lag</dt><dd>The maturity point, measured in days after the reference date,
        used to define training outcomes.</dd></div>
      <div><dt>Reporting lag</dt><dd>How old an observation was when a nowcast was made: the number
        of days between its reference date and report date.</dd></div>
    </dl>
    <p>In real time, the finalized value is unavailable, and we cannot know exactly
    when reporting will be complete. Choosing a target lag is therefore a compromise.</p>
    <ol class="lag-tradeoffs">
      <li><strong>Accuracy.</strong> A lag that is too short can define targets that are still far
        from their eventual values.</li>
      <li><strong>Timeliness.</strong> A lag that is too long removes the newest reports from model
        training. That lost revision history matters most when reporting patterns change abruptly.</li>
    </ol>
    <p>Use the figure and table to judge the trade-off, then choose the target lag yourself.
    AutoDelphiRF will use your value exactly as entered.</p>`;
  answers.innerHTML = '';
  custom.hidden = true;
  if (userTargetLag != null) $('custom-lag-input').value = userTargetLag;
  $('custom-lag-note').textContent = 'A target lag must be selected before the analysis can continue.';
  $('custom-lag-apply').onclick = () => {
    const value = Number($('custom-lag-input').value);
    if (!Number.isFinite(value) || value < 1) {
      $('custom-lag-note').textContent = 'Enter a whole number of days.';
      return;
    }
    acceptLag(value, false);
  };
}

function acceptLag(lag, confirm) {
  state.answeredLag = lag;
  state.confirmLag = confirm;
  show('card-schedule');
  loadLayers().then(() => {
    show('card-methods');
    unlockStage(2);
    goToStage(2);
  }).catch((error) => showFailure(error.message));
}

/* --- 5. the retraining schedule --------------------------------------- */

function initSchedule(retraining) {
  // Prefilled from the diagnosis, then previewed by the server on every
  // change so the origin count shown is the one the run will use.
  $('retrain-days').value = retraining.retrain_days;
  $('first-origin-date').value = retraining.first_origin;
  $('retrain-days-hint').textContent =
    `default ${retraining.default_retrain_days} for a ${retraining.temporal_resolution} stream`;
  $('first-origin-hint').textContent =
    `default ${retraining.default_first_origin_offset_days} days into the archive, `
    + `whose report-date range is ${retraining.report_date_min} to ${retraining.report_date_max}`;
  renderSchedule(retraining);

  $('retrain-days').oninput = previewSchedule;
  $('first-origin-date').onchange = previewSchedule;
  $('training-days').oninput = () => {
    if (state.schedule) renderSchedule(state.schedule);
  };
  $('schedule-reset').onclick = () => {
    $('training-days').value = state.diagnosis.training_window_days || 180;
    $('retrain-days').value = retraining.default_retrain_days;
    $('first-origin-date').value = retraining.first_origin;
    previewSchedule();
  };
  $('schedule-continue').onclick = () => {
    loadLayers().then(() => {
      show('card-methods');
      unlockStage(2);
      goToStage(2);
    });
  };
}

let schedulePending = null;

async function previewSchedule() {
  const body = {
    retrain_days: $('retrain-days').value || null,
    first_origin_date: $('first-origin-date').value || null,
  };
  // Typing "140" passes through "1" and "14"; only the last answer counts.
  const token = {};
  schedulePending = token;
  try {
    const preview = await api('/api/schedule', {method: 'POST', body});
    if (schedulePending !== token) return;
    // The resolved values are reported in the preview below, never written
    // back into the fields: a response landing mid-keystroke would otherwise
    // overwrite the digits still being typed.
    renderSchedule(preview);
  } catch (error) {
    if (schedulePending !== token) return;
    state.schedule = null;
    $('schedule-preview').innerHTML = notice('bad', escapeHtml(error.message));
    $('schedule-continue').disabled = true;
  }
}

function renderSchedule(preview) {
  state.schedule = preview;
  $('schedule-continue').disabled = false;
  const origins = preview.n_origins;
  // Every origin refits every ticked method, so the count is the single
  // number that decides how long the run takes. Said plainly, and loudly
  // once it is large enough to matter.
  const cost = origins > 120
    ? notice('warn', `That is <strong>${origins}</strong> retrainings. Every ticked method is
        refitted at each one, so a run this long is an overnight job rather than a coffee
        break. A longer interval, or a later first report-date origin, shortens it.`)
    : '';
  const trainingDays = Number($('training-days').value) || 180;
  $('schedule-preview').innerHTML = `
    <div class="schedule-summary">
      <strong>${trainingDays}-day training window</strong>
      <span>Retrain every ${preview.retrain_days} days</span>
      <span>${origins} report-date rolling fits</span>
      <span>Report-date origins: ${preview.first_origin} to ${preview.last_origin}</span>
    </div>${cost}`;
}

/* --- the completion curve --------------------------------------------- */

function drawCompletionCurve(report) {
  const curve = report.target_lag_completion_curve || {};
  const band = report.target_lag_completion_band || {};
  const allPoints = Object.keys(curve)
    .map((lag) => ({
      lag: Number(lag),
      error: Number(curve[lag]),
      q10: Number((band[lag] || {}).q10),
      q60: Number((band[lag] || {}).q60),
      q75: Number((band[lag] || {}).q75),
      q90: Number((band[lag] || {}).q90),
      q95: Number((band[lag] || {}).q95),
    }))
    .filter((point) => Number.isFinite(point.lag) && Number.isFinite(point.error))
    .sort((a, b) => a.lag - b.lag);
  const points = allPoints.filter((point) => point.lag >= 0 && point.lag <= 365);

  const container = $('completion-chart');
  if (!points.length) {
    container.innerHTML = `<p class="hint">No candidate target lags could be evaluated.</p>`;
    return;
  }
  const range = report.target_lag_suggested_range || {};
  const availableLags = (report.available_target_lags || allPoints.map((point) => point.lag))
    .map(Number).filter((lag) => Number.isInteger(lag) && lag >= 1).sort((a, b) => a - b);
  const availableSet = new Set(availableLags);
  const initial = points.find((point) => point.lag === Number(range.lower)) || points[0];

  const width = 680;
  const height = 334;
  const pad = {top: 44, right: 22, bottom: 64, left: 76};
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const minLag = 0;
  const maxLag = 365;
  const lagSpan = Math.max(1, maxLag - minLag);
  const finiteErrors = points.flatMap((point) =>
    [point.error, point.q60, point.q75, point.q90, point.q95].filter(Number.isFinite));
  const maxError = Math.max(0.10, ...finiteErrors) * 1.06;
  const x = (lag) => pad.left + ((lag - minLag) / lagSpan) * plotWidth;
  const y = (error) => pad.top + plotHeight - (error / maxError) * plotHeight;
  const bottom = pad.top + plotHeight;
  const hasSuggestedRange = range.lower != null && range.upper != null
    && Number.isFinite(Number(range.lower)) && Number.isFinite(Number(range.upper))
    && Number(range.lower) >= minLag && Number(range.lower) <= maxLag
    && Number(range.upper) >= minLag && Number(range.upper) <= maxLag;
  const suggestedStart = hasSuggestedRange
    ? Math.max(minLag, Math.min(maxLag, Math.min(Number(range.lower), Number(range.upper)))) : null;
  const suggestedEnd = hasSuggestedRange
    ? Math.max(minLag, Math.min(maxLag, Math.max(Number(range.lower), Number(range.upper)))) : null;
  const suggestedLabelX = hasSuggestedRange
    ? Math.max(pad.left + 64, Math.min(width - pad.right - 64,
      (x(suggestedStart) + x(suggestedEnd)) / 2)) : null;
  const suggestedFill = hasSuggestedRange ? `
    <rect x="${x(suggestedStart).toFixed(1)}" y="${pad.top}"
          width="${Math.max(1, x(suggestedEnd) - x(suggestedStart)).toFixed(1)}"
          height="${plotHeight}" class="completion-suggested-area"/>` : '';
  const suggestedMarkers = hasSuggestedRange ? `
    <line x1="${x(suggestedStart).toFixed(1)}" x2="${x(suggestedStart).toFixed(1)}"
          y1="${pad.top}" y2="${bottom}" class="completion-suggested-boundary"/>
    <line x1="${x(suggestedEnd).toFixed(1)}" x2="${x(suggestedEnd).toFixed(1)}"
          y1="${pad.top}" y2="${bottom}" class="completion-suggested-boundary"/>
    <path d="M${x(suggestedStart).toFixed(1)},${pad.top - 4}
             V${pad.top - 9} H${x(suggestedEnd).toFixed(1)} V${pad.top - 4}"
          class="completion-suggested-bracket"/>
    <text x="${suggestedLabelX.toFixed(1)}" y="${pad.top - 15}"
          text-anchor="middle" class="completion-suggested-label">Suggested lag range</text>
    <text x="${(x(suggestedStart) + 7).toFixed(1)}" y="${pad.top + 7}"
          transform="rotate(90 ${(x(suggestedStart) + 7).toFixed(1)} ${pad.top + 7})"
          class="completion-suggested-lag-label">${suggestedStart} days</text>
    <text x="${(x(suggestedEnd) + 7).toFixed(1)}" y="${pad.top + 7}"
          transform="rotate(90 ${(x(suggestedEnd) + 7).toFixed(1)} ${pad.top + 7})"
          class="completion-suggested-lag-label">${suggestedEnd} days</text>` : '';
  const percentilePath = (key) => {
    const valid = points.filter((point) => Number.isFinite(point[key]));
    if (!valid.length) return '';
    return `M${x(valid[0].lag).toFixed(1)},${bottom.toFixed(1)} `
      + valid.map((point) =>
        `L${x(point.lag).toFixed(1)},${y(point[key]).toFixed(1)}`).join(' ')
      + ` L${x(valid[valid.length - 1].lag).toFixed(1)},${bottom.toFixed(1)} Z`;
  };
  const medianLine = points.map((point, index) =>
    `${index ? 'L' : 'M'}${x(point.lag).toFixed(1)},${y(point.error).toFixed(1)}`).join(' ');
  const yTicks = Array.from({length: 5}, (_, index) => (maxError * index) / 4);
  const grid = yTicks.map((tick) => `
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${y(tick).toFixed(1)}"
          y2="${y(tick).toFixed(1)}" class="completion-gridline"/>
    <text x="${pad.left - 8}" y="${(y(tick) + 4).toFixed(1)}" text-anchor="end"
          class="completion-axis-label">${(tick * 100).toFixed(0)}%</text>`).join('');
  const xTicks = [0, 60, 120, 180, 240, 300, 365];
  const xLabels = xTicks.map((lag) => `
    <text x="${x(lag).toFixed(1)}" y="${bottom + 25}" text-anchor="middle"
          class="completion-axis-label">${lag}</text>`).join('');
  const toleranceLines = [0.01, 0.10].filter((value) => value <= maxError).map((value) => `
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${y(value).toFixed(1)}"
          y2="${y(value).toFixed(1)}" class="completion-tolerance"/>
    <text x="${width - pad.right - 4}" y="${(y(value) - 5).toFixed(1)}" text-anchor="end"
          class="completion-tolerance-label">${(value * 100).toFixed(0)}%</text>`).join('');

  container.innerHTML = `
    <div class="completion-legend" aria-label="Percentile layers">
      <span class="median">Median</span><span class="q60">60th percentile</span>
      <span class="q75">75th percentile</span><span class="q90">90th percentile</span>
      <span class="q95">95th percentile</span>
    </div>
    <div class="completion-plot">
      <svg viewBox="0 0 ${width} ${height}" role="img"
           aria-label="Relative error percentiles by available target lag. Move across the figure to inspect a lag in the table below.">
        ${suggestedFill}
        ${grid}
        ${toleranceLines}
        <path d="${percentilePath('q95')}" class="completion-band q95"/>
        <path d="${percentilePath('q90')}" class="completion-band q90"/>
        <path d="${percentilePath('q75')}" class="completion-band q75"/>
        <path d="${percentilePath('q60')}" class="completion-band q60"/>
        <path d="${medianLine}" class="completion-median"/>
        ${suggestedMarkers}
        <line x1="${pad.left}" x2="${width - pad.right}" y1="${bottom}" y2="${bottom}"
              class="completion-axis"/>
        ${xLabels}
        <text x="${pad.left + plotWidth / 2}" y="${bottom + 50}" text-anchor="middle"
              class="completion-axis-title">Target lag (days after reference date)</text>
        <text x="18" y="${pad.top + plotHeight / 2}" text-anchor="middle"
              transform="rotate(-90 18 ${pad.top + plotHeight / 2})"
              class="completion-axis-title">Relative error (%)</text>
        <g class="completion-guide" aria-hidden="true">
          <line y1="${pad.top}" y2="${bottom}"/>
          ${['median', 'q60', 'q75', 'q90', 'q95'].map((key) =>
            `<circle data-guide-point="${key}" r="4"/>`).join('')}
        </g>
        <rect x="${pad.left}" y="${pad.top}" width="${plotWidth}" height="${plotHeight}"
              class="completion-hit-area" tabindex="0"
              aria-label="Use the target-lag selector below for keyboard exploration"/>
      </svg>
    </div>
    <div class="percentile-explorer" role="group" aria-label="Relative error by available target lag">
      <label class="lag-selector">Target lag, in days after the reference date
        <input type="number" id="completion-lag-input" min="1" step="1"
               value="${initial.lag}" inputmode="numeric">
        <span class="lag-area-status" id="completion-area-status"></span>
        <button type="button" id="completion-use-lag" class="primary compact">Use this lag</button>
      </label>
      ${['median', 'q60', 'q75', 'q90', 'q95'].map((key) => `
        <div class="percentile-cell" data-percentile-cell="${key}">
          <span>${key === 'median' ? 'Median' : `${key.slice(1)}th percentile`}</span>
          <strong>—</strong>
          <small></small>
        </div>`).join('')}
    </div>`;

  const describeTolerance = (value) => {
    if (!Number.isFinite(value)) return {label: 'unavailable', level: 'unknown'};
    if (value <= 0.01 || Math.abs(value - 0.01) < 1e-8) {
      return {label: 'Within 1% error', level: 'within-one'};
    }
    if (value <= 0.10 || Math.abs(value - 0.10) < 1e-8) {
      return {label: 'Within 10% error', level: 'within-ten'};
    }
    return {label: 'Above 10% error', level: 'above-ten'};
  };
  const updateGuide = (point) => {
    const guide = container.querySelector('.completion-guide');
    const displayed = point.lag >= minLag && point.lag <= maxLag;
    guide.style.display = displayed ? '' : 'none';
    if (!displayed) return;
    const guideX = x(point.lag);
    guide.querySelector('line').setAttribute('x1', guideX.toFixed(1));
    guide.querySelector('line').setAttribute('x2', guideX.toFixed(1));
    const values = {median: point.error, q60: point.q60, q75: point.q75,
                    q90: point.q90, q95: point.q95};
    Object.entries(values).forEach(([key, value]) => {
      const marker = guide.querySelector(`[data-guide-point="${key}"]`);
      marker.hidden = !Number.isFinite(value);
      if (Number.isFinite(value)) {
        marker.setAttribute('cx', guideX.toFixed(1));
        marker.setAttribute('cy', y(value).toFixed(1));
      }
    });
  };
  const renderPoint = (point, {syncInput = true, context = ''} = {}) => {
    const values = {median: point.error, q60: point.q60, q75: point.q75,
                    q90: point.q90, q95: point.q95};
    Object.entries(values).forEach(([key, value]) => {
      const cell = container.querySelector(`[data-percentile-cell="${key}"]`);
      const tolerance = describeTolerance(value);
      cell.className = `percentile-cell ${tolerance.level}`;
      cell.querySelector('strong').textContent = Number.isFinite(value)
        ? `${(value * 100).toFixed(1)}%` : '—';
      cell.querySelector('small').textContent = tolerance.label;
    });
    $('completion-area-status').textContent = point.lag > maxLag
      ? 'This available lag is outside the displayed 0–365 day range.' : context;
    if (syncInput) $('completion-lag-input').value = point.lag;
    $('custom-lag-input').value = point.lag;
    $('custom-lag-note').textContent =
      `Review the error distribution for ${point.lag} days, then confirm to continue.`;
    updateGuide(point);
  };
  const lagInput = $('completion-lag-input');
  lagInput.addEventListener('input', () => {
    const requested = Number(lagInput.value);
    const valid = Number.isInteger(requested) && requested >= 1;
    $('completion-use-lag').disabled = !valid;
    if (!valid) {
      $('completion-area-status').textContent = 'Enter a positive whole number.';
      return;
    }
    if (availableSet.has(requested)) {
      const exact = allPoints.find((point) => point.lag === requested);
      if (exact) {
        renderPoint(exact, {syncInput: false});
        return;
      }
      $('completion-use-lag').disabled = true;
      $('completion-area-status').textContent =
        `Lag ${requested} exists, but its diagnostic percentages are unavailable. Re-run diagnosis.`;
      return;
    }
    $('completion-use-lag').disabled = true;
    const lowerLags = availableLags.filter((lag) => lag < requested);
    const below = lowerLags[lowerLags.length - 1];
    const above = availableLags.find((lag) => lag > requested);
    const neighbours = [below, above].filter((lag) => lag != null).map((lag) => `${lag} days`);
    $('completion-area-status').textContent = neighbours.length
      ? `No data exist at lag ${requested}. Nearest available: ${neighbours.join(' and ')}.`
      : `No data exist at lag ${requested}.`;
  });
  $('completion-use-lag').addEventListener('click', () => {
    const requested = Number(lagInput.value);
    if (Number.isInteger(requested) && requested >= 1 && availableSet.has(requested)) {
      acceptLag(requested, false);
    }
  });
  const hitArea = container.querySelector('.completion-hit-area');
  const pointFromPointer = (event) => {
    const box = hitArea.getBoundingClientRect();
    const svgX = pad.left + ((event.clientX - box.left) / Math.max(1, box.width)) * plotWidth;
    return points.reduce((closest, point) =>
      Math.abs(x(point.lag) - svgX) < Math.abs(x(closest.lag) - svgX) ? point : closest,
    points[0]);
  };
  hitArea.addEventListener('pointermove', (event) => renderPoint(pointFromPointer(event)));
  hitArea.addEventListener('pointerdown', (event) => renderPoint(pointFromPointer(event)));
  renderPoint(initial);
}

/* --- 5. methods ------------------------------------------------------- */

let layersLoaded = false;

const METHOD_LABELS = {
  baseline_null: 'Latest reported value',
  red: 'Revision-pattern matching',
  delphirf: 'RevRoute-DelphiRF',
  naive_delphirf: 'Naive DelphiRF',
  similarity_weighted_delphirf: 'Similarity-weighted DelphiRF',
  global_delphirf: 'Global DelphiRF',
};

const METHOD_DESCRIPTIONS = {
  baseline_null: 'No adjustment. Carries the currently available value forward as the nowcast.',
  delphirf: 'The primary AutoDelphiRF method. It automatically decides which reporting lags and locations can share information.',
  naive_delphirf: 'Fits locations separately; small reporting lags are modeled separately and larger reporting lags are grouped.',
  global_delphirf: 'Uses one model that combines all locations and reporting lags.',
  red: 'Finds historical observations with similar revision patterns. Primarily intended for sparse reporting histories, such as weekly data.',
};

const WEB_HIDDEN_METHODS = new Set(['similarity_weighted_delphirf']);

function methodLabel(name) {
  return METHOD_LABELS[name] || String(name).replace(/_/g, ' ');
}

function comparisonLabel(comparison) {
  return String(comparison).split(' - ').map(methodLabel).join(' compared with ');
}

async function loadLayers() {
  if (layersLoaded) return;
  const {layers} = await api('/api/layers');
  $('layer-list').innerHTML = layers.filter((layer) => !WEB_HIDDEN_METHODS.has(layer.name)).map((layer) => `
    <label class="layer">
      <input type="checkbox" value="${escapeHtml(layer.name)}"${layer.default ? ' checked' : ''}>
      <span>
        <span class="layer-name">${escapeHtml(methodLabel(layer.name))}</span><br>
        <span class="layer-desc">${escapeHtml(METHOD_DESCRIPTIONS[layer.name] || layer.description)}</span>
        ${layer.expensive ? '<br><span class="layer-cost">Fits a model — slower</span>' : ''}
      </span>
    </label>`).join('');
  $('run-button').addEventListener('click', startRun);
  layersLoaded = true;
}

/* --- 6. run ----------------------------------------------------------- */

async function startRun() {
  const button = $('run-button');
  const status = $('run-status');
  const layers = [...document.querySelectorAll('#layer-list input:checked')]
    .map((input) => input.value);
  if (!layers.length) {
    status.innerHTML = '<span class="pill bad">Choose at least one method.</span>';
    return;
  }
  // Preprocessing is the first thing a run does and the first thing that fails
  // without DelphiRF. The environment card says so at startup, but a warning
  // five steps up the page is not a warning by the time you get here: refuse
  // now, with the fix, rather than after the schedule and methods are set.
  const delphiMethods = new Set([
    'delphirf', 'naive_delphirf', 'similarity_weighted_delphirf', 'global_delphirf'
  ]);
  const needsDelphiModel = layers.some((layer) => delphiMethods.has(layer));
  const preprocessingReady = state.environment &&
    (state.environment.preprocessing ?? state.environment.delphirf);
  if (state.environment && (!preprocessingReady || (needsDelphiModel && !state.environment.delphirf))) {
    status.innerHTML = '';
    $('run-result').hidden = true;
    show('card-run');
    $('run-log').textContent = '';
    $('run-state').textContent = 'blocked';
    $('run-state').className = 'badge failed';
    $('run-result').hidden = false;
    $('run-result').innerHTML = notice('bad',
      `<strong>${needsDelphiModel ? 'The selected DelphiRF model is not ready.'
                                  : 'DelphiRF preprocessing is not ready.'}</strong><br>
       ${escapeHtml(state.environment.detail)}<br>
       ${needsDelphiModel
         ? 'Choose “Latest reported value” and/or “Revision-pattern matching” for now, or reinstall DelphiRF as the README describes.'
         : 'Install DelphiRF in R as the README describes.'}<br>
       <span class="hint">Your archive, target lag and schedule are unaffected &mdash; this
       check runs when the page loads, so reload the page after reinstalling DelphiRF.</span>`);
    goToStage(2);
    return;
  }

  button.disabled = true;
  status.innerHTML = '<span class="spinner"></span> Starting…';

  try {
    const request = {
      ...currentMapping(),
      layers,
      target_lag: state.answeredLag,
      confirm_target_lag: state.confirmLag,
      retrain_days: Number($('retrain-days').value) || null,
      first_origin_date: $('first-origin-date').value || null,
      min_location_rows: Number($('min-location-rows').value) || 100,
      triangle_format: $('triangle-format').value,
    };
    const trainingDays = Number($('training-days').value);
    if (Number.isFinite(trainingDays) && trainingDays > 0) request.training_days = trainingDays;

    await api('/api/run', {method: 'POST', body: request});
    status.textContent = '';
    show('card-run');
    goToStage(2);
    pollStatus();
  } catch (error) {
    button.disabled = false;
    status.innerHTML = `<span class="pill bad">${escapeHtml(error.message)}</span>`;
  }
}

function pollStatus() {
  if (state.polling) clearInterval(state.polling);
  const tick = async () => {
    let snapshot;
    try {
      snapshot = await api('/api/status');
    } catch (error) {
      $('run-state').textContent = 'connection lost';
      $('run-state').className = 'badge failed';
      clearInterval(state.polling);
      return;
    }
    renderRun(snapshot);
    if (['done', 'failed', 'stopped'].includes(snapshot.state)) {
      clearInterval(state.polling);
      state.polling = null;
      $('run-button').disabled = false;
    }
  };
  tick();
  state.polling = setInterval(tick, 1200);
}

async function stopRun() {
  $('stop-run').disabled = true;
  try {
    await api('/api/stop', {method: 'POST', body: {}});
    pollStatus();
  } catch (error) {
    showFailure(error.message);
  }
}

async function resumeRun() {
  $('resume-run').disabled = true;
  try {
    await api('/api/resume', {method: 'POST', body: {}});
    pollStatus();
  } catch (error) {
    $('resume-run').disabled = false;
    showFailure(error.message);
  }
}

async function restoreLastRun() {
  try {
    const snapshot = await api('/api/status');
    const saved = snapshot.session || {};
    const request = saved.last_request;
    if (saved.archive) {
      onUploaded(saved.archive, {name: saved.archive.filename});
    }
    if (request) {
      const valueType = request.value_type || 'count';
      const radio = document.querySelector(`input[name="value-type"][value="${valueType}"]`);
      if (radio) radio.checked = true;
      onValueTypeChange();
      const assignments = {
        'col-reference': request.reference_col,
        'col-report': request.report_col,
        'col-value': (request.value_cols || [])[0],
        'col-denominator': (request.value_cols || [])[1],
        'col-geo': request.geo_col || '',
      };
      Object.entries(assignments).forEach(([id, value]) => {
        const element = $(id);
        if (element && value != null) element.value = value;
      });
    }
    if (saved.diagnosis) {
      state.diagnosis = saved.diagnosis;
      renderDiagnosis(saved.diagnosis, request ? request.target_lag : null);
      show('card-diagnosis');
      unlockStage(1);
    }
    if (request && saved.diagnosis) {
      state.answeredLag = Number(request.target_lag);
      state.confirmLag = Boolean(request.confirm_target_lag);
      show('card-schedule');
      if (request.retrain_days) $('retrain-days').value = request.retrain_days;
      if (request.first_origin_date) $('first-origin-date').value = request.first_origin_date;
      await previewSchedule();
      await loadLayers();
      document.querySelectorAll('#layer-list input').forEach((input) => {
        input.checked = (request.layers || []).includes(input.value);
      });
      if (request.training_days) $('training-days').value = request.training_days;
      if (request.min_location_rows) $('min-location-rows').value = request.min_location_rows;
      if (request.triangle_format) $('triangle-format').value = request.triangle_format;
      show('card-methods');
      unlockStage(2);
    }
    if (snapshot.state === 'idle') return;
    show('card-run');
    unlockStage(2);
    goToStage(2);
    renderRun(snapshot);
    if (snapshot.state === 'running') pollStatus();
  } catch (_) {
    // The environment card reports connection failures. With no prior run,
    // recovery has nothing useful to add.
  }
}

function renderRun(snapshot) {
  const badge = $('run-state');
  badge.textContent = snapshot.state === 'running'
    ? (snapshot.stage || 'running') : snapshot.state;
  badge.className = `badge ${snapshot.state}`;
  $('run-elapsed').textContent = snapshot.elapsed_seconds
    ? `${formatDuration(snapshot.elapsed_seconds)} elapsed` : '';
  renderProgress(snapshot);
  $('stop-run').hidden = snapshot.state !== 'running';
  $('stop-run').disabled = snapshot.stage === 'stopping';
  $('resume-run').hidden = snapshot.state !== 'stopped';
  $('resume-run').disabled = false;

  const log = $('run-log');
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  log.textContent = snapshot.log.join('\n');
  if (atBottom) log.scrollTop = log.scrollHeight;

  const result = $('run-result');
  if (snapshot.state === 'done') {
    result.hidden = false;
    result.innerHTML = notice('good',
      `Finished in ${formatDuration(snapshot.elapsed_seconds)}.`);
    loadResults();
  } else if (snapshot.state === 'failed') {
    result.hidden = false;
    result.innerHTML = notice('bad',
      `<strong>The run failed.</strong><br>${escapeHtml(snapshot.error)}<br>
       <span class="hint">The full traceback is in the terminal running
       <code>autodelphirf web</code>.</span>`);
  } else if (snapshot.state === 'stopped') {
    result.hidden = false;
    result.innerHTML = notice('warn',
      `<strong>The experiment was stopped.</strong><br>
       Resume restarts it with the same saved settings. Partial fitted results are not reused.`);
  } else {
    result.hidden = true;
  }
}

function formatDuration(seconds) {
  const total = Math.round(seconds || 0);
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m ${String(total % 60).padStart(2, '0')}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

function renderProgress(snapshot) {
  const wrap = $('run-progress');
  const {origins_done: done, origins_total: total} = snapshot;
  if (!total || snapshot.state === 'failed') {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  const share = Math.min(1, done / total);
  $('run-fill').style.width = `${(share * 100).toFixed(1)}%`;
  const remaining = snapshot.remaining_seconds != null
    ? ` &middot; about ${escapeHtml(formatDuration(snapshot.remaining_seconds))} left`
    : '';
  $('run-progress-text').innerHTML = snapshot.state === 'done'
    ? `${total} retrainings complete`
    : `Retraining ${done} of ${total}${remaining}`;
}

/* --- 8. results -------------------------------------------------------- */

async function loadResults() {
  let results;
  try {
    results = await api('/api/results');
  } catch (error) {
    $('results-message').innerHTML = notice('warn',
      `The run finished, but its tables could not be read: ${escapeHtml(error.message)}`);
    show('card-results');
    unlockStage(3);
    goToStage(3);
    return;
  }
  state.results = results;
  const tables = results.tables || {};
  renderSkippedLocations(tables.skipped_locations || []);
  renderLagEvaluation(tables.web_metrics_by_lag || []);
  renderLocationEvaluation(tables.web_metrics_by_location || []);
  renderEarlyLagEvaluation(tables.web_metrics_by_origin_lag || []);
  show('card-results');
  unlockStage(3);
  goToStage(3);
}

// Locations preparation could not build are absent from every result, so the
// page names them, with the reason, rather than letting them vanish.
function renderSkippedLocations(rows) {
  const box = $('skipped-locations');
  if (!rows.length) {
    box.hidden = true;
    box.innerHTML = '';
    return;
  }
  const items = rows.map((row) =>
    `<li><strong>${escapeHtml(String(row.geo_value))}</strong>: ${escapeHtml(String(row.reason))}</li>`);
  box.innerHTML = notice('warn',
    `${rows.length} location${rows.length === 1 ? ' was' : 's were'} skipped during preparation `
    + `and ${rows.length === 1 ? 'is' : 'are'} not forecast or evaluated:<ul>${items.join('')}</ul>`);
  box.hidden = false;
}

const SERIES_SLOTS = 8;
let resultTimeMetric = 'ae';
let resultDisplayScale = 'percent';
let resultLagRows = [];
let resultLocationRows = [];
let resultOriginRows = [];

function metricRows(rows, metric) {
  return rows.filter((row) => row.metric === metric
    && Number.isFinite(row.median) && Number.isFinite(row.q10) && Number.isFinite(row.q90));
}

function activeResultMetric() {
  return resultDisplayScale === 'percent' ? `${resultTimeMetric}_pct` : resultTimeMetric;
}

function resultMetricLabel() {
  if (resultDisplayScale === 'percent') {
    return resultTimeMetric === 'wis' ? 'Relative WIS (%)' : 'Relative error (%)';
  }
  return resultTimeMetric === 'wis' ? 'WIS' : 'AE';
}

function scaleMetricNames() {
  return resultDisplayScale === 'percent' ? ['ae_pct', 'wis_pct'] : ['ae', 'wis'];
}

function sharedMetricAxis(rows) {
  const names = new Set(scaleMetricNames());
  const values = rows.filter((row) => names.has(row.metric)).flatMap((row) =>
    [Number(row.q90), Number(row.median)]).filter(Number.isFinite);
  return niceAxis(Math.max(...values, 0) || 1);
}

function renderLagEvaluation(rows) {
  const usable = rows.filter((row) => Number(row.n_locations) > 0);
  resultLagRows = usable;
  if (!usable.length) {
    $('lag-score-grid').hidden = true;
    $('results-message').innerHTML = notice('warn',
      'No location-level evaluation table was written for this run.');
    return;
  }
  $('results-message').innerHTML = '';
  $('lag-score-grid').hidden = false;
  bindResultMetricSwitch();
  drawLagEvaluation();
}

function renderEarlyLagEvaluation(rows) {
  const section = $('early-lag-section');
  resultOriginRows = rows;
  if (!rows.length) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  bindResultMetricSwitch();
  drawEarlyLagCharts();
}

function renderLocationEvaluation(rows) {
  resultLocationRows = rows.filter((row) => Number(row.n) > 0
    && Number.isFinite(Number(row.median))
    && Number.isFinite(Number(row.q10))
    && Number.isFinite(Number(row.q90)));
  drawLocationEvaluation();
}

function bindResultMetricSwitch() {
  document.querySelectorAll('[data-result-metric]').forEach((button) => {
    button.onclick = () => {
      resultTimeMetric = button.dataset.resultMetric;
      document.querySelectorAll('[data-result-metric]').forEach((candidate) =>
        candidate.classList.toggle('active', candidate.dataset.resultMetric === resultTimeMetric));
      drawLagEvaluation();
      drawLocationEvaluation();
      drawEarlyLagCharts();
    };
  });
  document.querySelectorAll('[data-result-scale]').forEach((button) => {
    button.onclick = () => {
      resultDisplayScale = button.dataset.resultScale;
      document.querySelectorAll('[data-result-scale]').forEach((candidate) =>
        candidate.classList.toggle('active', candidate.dataset.resultScale === resultDisplayScale));
      $('evaluation-scale-note').textContent = resultDisplayScale === 'percent'
        ? 'Percentages show each error relative to the size of the later target value, so errors at locations of different sizes can be compared.'
        : "Scores use the indicator's original reported units.";
      drawLagEvaluation();
      drawLocationEvaluation();
      drawEarlyLagCharts();
    };
  });
}

function drawLocationEvaluation() {
  const container = $('location-score-chart');
  if (!container) return;
  const label = resultMetricLabel();
  $('location-score-title').textContent = `${label} by location at early reporting lags`;
  const rows = resultLocationRows.filter((row) => row.metric === activeResultMetric());
  if (!rows.length) {
    container.innerHTML = `<p class="hint empty-chart">No ${label} location summary is available. Re-run post-assessment with the current version to create it.</p>`;
    return;
  }
  const sharedYAxis = sharedMetricAxis(resultLocationRows);
  drawLocationBarChart('location-score-chart', rows, {
    yLabel: `Median ${label}`, yMax: sharedYAxis.maximum, yTicks: sharedYAxis.ticks,
  });
}

function drawLagEvaluation() {
  if (!resultLagRows.length) return;
  const label = resultMetricLabel();
  const sharedYAxis = sharedMetricAxis(resultLagRows);
  $('lag-score-title').textContent = `${label} over reporting lags`;
  drawQuantileBandChart('lag-score-chart', metricRows(resultLagRows, activeResultMetric()), {
    xField: 'lag', xLabel: 'Reporting lag (days)',
    yLabel: label, xKind: 'number', yMax: sharedYAxis.maximum, yTicks: sharedYAxis.ticks,
  });
}

function drawEarlyLagCharts() {
  const container = $('early-lag-grid');
  if (!container || !resultOriginRows.length) return;
  const eligible = metricRows(resultOriginRows, activeResultMetric());
  if (!eligible.length) {
    container.innerHTML = `<p class="hint">${resultMetricLabel()} is unavailable for the selected methods.</p>`;
    return;
  }
  const allLags = [...new Set(eligible.map((row) => Number(row.lag)))]
    .filter(Number.isFinite).sort((a, b) => a - b).slice(0, 2);
  const selectedRows = eligible.filter((row) => allLags.includes(Number(row.lag)));
  const axisRows = resultOriginRows.filter((row) => allLags.includes(Number(row.lag)));
  const sharedYAxis = sharedMetricAxis(axisRows);
  const sharedXKeys = [...new Set(selectedRows.map((row) => row.cutoff))].sort();
  container.innerHTML = allLags.map((lag, index) => `
    <figure class="result-figure early-lag-figure">
      <figcaption><span class="fig-title">Lag ${escapeHtml(String(lag))} days</span>
        <span class="fig-sub">${resultMetricLabel()} · lower is better</span></figcaption>
      <div class="result-chart" id="early-lag-chart-${index}"></div>
    </figure>`).join('');
  allLags.forEach((lag, index) => drawQuantileBandChart(
    `early-lag-chart-${index}`, eligible.filter((row) => Number(row.lag) === lag), {
      xField: 'cutoff', xLabel: 'report-date origin',
      yLabel: resultMetricLabel(), xKind: 'date',
      compact: true, yMax: sharedYAxis.maximum, yTicks: sharedYAxis.ticks,
      xKeys: sharedXKeys,
    }));
}

function niceAxis(observedMaximum, tickCount = 5) {
  const maximum = Number.isFinite(observedMaximum) && observedMaximum > 0
    ? observedMaximum : 1;
  const roughStep = maximum / Math.max(1, tickCount - 1);
  const power = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / power;
  const factor = [1, 2, 2.5, 5, 10].find((candidate) => candidate >= normalized) || 10;
  const step = factor * power;
  const niceMaximum = step * Math.ceil(maximum / step);
  const ticks = Array.from({length: Math.round(niceMaximum / step) + 1},
    (_, index) => index * step);
  return {maximum: niceMaximum, ticks};
}

function drawLocationBarChart(containerId, rows, options) {
  const container = $(containerId);
  const methods = [...new Set(rows.map((row) => row.method))].slice(0, SERIES_SLOTS);
  const locations = [...new Set(rows.map((row) => String(row.geo_value)))];
  const nullMedians = new Map(rows.filter((row) => row.method === 'baseline_null')
    .map((row) => [String(row.geo_value), Number(row.median)]));
  const locationMedians = new Map(locations.map((location) => {
    const values = rows.filter((row) => String(row.geo_value) === location)
      .map((row) => Number(row.median)).filter(Number.isFinite);
    return [location, values.reduce((sum, value) => sum + value, 0) / values.length];
  }));
  locations.sort((a, b) => (nullMedians.get(b) ?? locationMedians.get(b))
    - (nullMedians.get(a) ?? locationMedians.get(a)) || a.localeCompare(b));

  const observedMaximum = Math.max(...rows.map((row) => Number(row.q90)), 0) || 1;
  const yAxis = Number.isFinite(options.yMax)
    ? {maximum: options.yMax, ticks: options.yTicks || niceAxis(options.yMax).ticks}
    : niceAxis(observedMaximum);
  const width = Math.max(720, locations.length * Math.max(50, methods.length * 15 + 18));
  const height = 350;
  const pad = {top: 14, right: 18, bottom: 82, left: 76};
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const groupWidth = plotWidth / Math.max(1, locations.length);
  const barWidth = Math.min(13, Math.max(5, (groupWidth - 10) / Math.max(1, methods.length)));
  const occupiedWidth = barWidth * methods.length;
  const y = (value) => pad.top + plotHeight
    - Math.max(0, Number(value)) / yAxis.maximum * plotHeight;
  const lookup = new Map(rows.map((row) =>
    [`${String(row.geo_value)}\u0000${row.method}`, row]));

  const grid = yAxis.ticks.map((tick) => `<line x1="${pad.left}" x2="${width - pad.right}"
      y1="${y(tick).toFixed(1)}" y2="${y(tick).toFixed(1)}" stroke="var(--border)"/>
    <text x="${pad.left - 10}" y="${(y(tick) + 5).toFixed(1)}" text-anchor="end"
      font-size="15" font-weight="550" fill="var(--text-primary)">${compactNumber(tick)}</text>`).join('');
  const bars = locations.flatMap((location, locationIndex) => methods.map((method, methodIndex) => {
    const row = lookup.get(`${location}\u0000${method}`);
    if (!row) return '';
    const median = Number(row.median);
    const q10 = Number(row.q10);
    const q90 = Number(row.q90);
    const center = pad.left + locationIndex * groupWidth + groupWidth / 2
      - occupiedWidth / 2 + methodIndex * barWidth + barWidth / 2;
    const top = y(median);
    const bottom = y(0);
    const errorTop = y(q90);
    const errorBottom = y(q10);
    const color = `var(--series-${methodIndex + 1})`;
    const detail = `${methodLabel(method)} · ${location}: median ${compactNumber(median)}, 10th–90th percentile ${compactNumber(q10)}–${compactNumber(q90)}, n=${row.n}`;
    return `<g><title>${escapeHtml(detail)}</title>
      <rect x="${(center - barWidth / 2 + 1).toFixed(1)}" y="${top.toFixed(1)}"
        width="${Math.max(2, barWidth - 2).toFixed(1)}" height="${Math.max(0, bottom - top).toFixed(1)}"
        fill="${color}" opacity="0.82"/>
      <line x1="${center.toFixed(1)}" x2="${center.toFixed(1)}" y1="${errorTop.toFixed(1)}"
        y2="${errorBottom.toFixed(1)}" stroke="var(--text-primary)" stroke-width="1.2"/>
      <line x1="${(center - 4).toFixed(1)}" x2="${(center + 4).toFixed(1)}"
        y1="${errorTop.toFixed(1)}" y2="${errorTop.toFixed(1)}" stroke="var(--text-primary)" stroke-width="1.2"/>
      <line x1="${(center - 4).toFixed(1)}" x2="${(center + 4).toFixed(1)}"
        y1="${errorBottom.toFixed(1)}" y2="${errorBottom.toFixed(1)}" stroke="var(--text-primary)" stroke-width="1.2"/>
    </g>`;
  })).join('');
  const xLabels = locations.map((location, index) => {
    const x = pad.left + index * groupWidth + groupWidth / 2;
    return `<text x="${x.toFixed(1)}" y="${height - pad.bottom + 18}" text-anchor="end"
      transform="rotate(-55 ${x.toFixed(1)} ${height - pad.bottom + 18})"
      font-size="13" font-weight="550" fill="var(--text-primary)">${escapeHtml(location)}</text>`;
  }).join('');
  const legend = methods.map((method, index) => `<li><span class="swatch" style="background:var(--series-${index + 1})"></span>
    ${escapeHtml(methodLabel(method))}</li>`).join('');
  container.innerHTML = `<ul class="legend">${legend}</ul>
    <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="${escapeHtml(options.yLabel)} by location for reporting lags through 14 days">
      ${grid}${bars}${xLabels}
      <line x1="${pad.left}" x2="${width - pad.right}" y1="${pad.top + plotHeight}"
        y2="${pad.top + plotHeight}" stroke="var(--border-strong)"/>
      <text x="20" y="${pad.top + plotHeight / 2}" text-anchor="middle"
        transform="rotate(-90 20 ${pad.top + plotHeight / 2})"
        font-size="16" font-weight="600" fill="var(--text-primary)">${escapeHtml(options.yLabel)}</text>
    </svg>`;
}

function drawQuantileBandChart(containerId, rows, options) {
  const container = $(containerId);
  if (!rows.length) {
    container.innerHTML = '<p class="hint empty-chart">No eligible values.</p>';
    return;
  }
  const width = 960;
  const height = options.compact ? 237 : 362;
  const pad = {top: 14, right: 20, bottom: 64, left: 88};
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const keys = options.xKeys || [...new Set(rows.map((row) => row[options.xField]))]
    .sort(options.xKind === 'number' ? (a, b) => Number(a) - Number(b) : undefined);
  const methods = [...new Set(rows.map((row) => row.method))];
  const keyIndex = new Map(keys.map((key, index) => [key, index]));
  const series = methods.slice(0, SERIES_SLOTS).map((method, index) => ({
    method, color: `var(--series-${index + 1})`,
    points: rows.filter((row) => row.method === method)
      .sort((a, b) => keyIndex.get(a[options.xField]) - keyIndex.get(b[options.xField])),
  })).filter((entry) => entry.points.length);
  const observedMaximum = Math.max(...rows.flatMap((row) =>
    [Number(row.q90), Number(row.median)]), 0) || 1;
  const yAxis = Number.isFinite(options.yMax)
    ? {maximum: options.yMax, ticks: options.yTicks || niceAxis(options.yMax).ticks}
    : niceAxis(observedMaximum);
  const maximum = yAxis.maximum;
  const x = (key) => pad.left + (keys.length === 1 ? plotWidth / 2
    : keyIndex.get(key) / (keys.length - 1) * plotWidth);
  const y = (value) => pad.top + plotHeight - Number(value) / maximum * plotHeight;
  const ticks = yAxis.ticks;
  const grid = ticks.map((tick) => `<line x1="${pad.left}" x2="${width - pad.right}"
      y1="${y(tick).toFixed(1)}" y2="${y(tick).toFixed(1)}" stroke="var(--border)"/>
    <text x="${pad.left - 12}" y="${(y(tick) + 5).toFixed(1)}" text-anchor="end"
      font-size="15" font-weight="550" fill="var(--text-primary)">${compactNumber(tick)}</text>`).join('');
  const shapes = series.map((entry) => {
    const upper = entry.points.map((point) => `${x(point[options.xField]).toFixed(1)},${y(point.q90).toFixed(1)}`);
    const lower = [...entry.points].reverse().map((point) => `${x(point[options.xField]).toFixed(1)},${y(point.q10).toFixed(1)}`);
    const median = entry.points.map((point, index) => `${index ? 'L' : 'M'}${x(point[options.xField]).toFixed(1)},${y(point.median).toFixed(1)}`).join(' ');
    return `<polygon points="${upper.concat(lower).join(' ')}" fill="${entry.color}" opacity="0.13"/>
      <path d="${median}" fill="none" stroke="${entry.color}" stroke-width="2"
        stroke-linejoin="round" stroke-linecap="round"/>`;
  }).join('');
  const labelCount = options.compact ? 4 : 6;
  const labelIndexes = new Set([0, keys.length - 1]);
  const step = Math.max(1, Math.ceil((keys.length - 1) / (labelCount - 1)));
  for (let index = 0; index < keys.length; index += step) labelIndexes.add(index);
  const xLabels = [...labelIndexes].filter((index) => index >= 0).sort((a, b) => a - b)
    .map((index) => {
      const raw = String(keys[index]);
      const label = options.xKind === 'date' ? raw.slice(0, 7) : raw;
      return `<text x="${x(keys[index]).toFixed(1)}" y="${pad.top + plotHeight + 30}" text-anchor="middle"
        font-size="15" font-weight="550" fill="var(--text-primary)">${escapeHtml(label)}</text>`;
    }).join('');
  const legend = series.map((entry) => `<li><span class="swatch" style="background:${entry.color}"></span>
    ${escapeHtml(methodLabel(entry.method))}</li>`).join('');
  container.innerHTML = `<ul class="legend">${legend}</ul>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(options.yLabel)} by ${escapeHtml(options.xLabel)}">
      ${grid}${shapes}
      <line x1="${pad.left}" x2="${width - pad.right}" y1="${pad.top + plotHeight}"
        y2="${pad.top + plotHeight}" stroke="var(--border-strong)"/>
      ${xLabels}
      <text x="${pad.left + plotWidth / 2}" y="${pad.top + plotHeight + 52}" text-anchor="middle"
        font-size="16" font-weight="600" fill="var(--text-primary)">${escapeHtml(options.xLabel)}</text>
      <text x="20" y="${pad.top + plotHeight / 2}" text-anchor="middle"
        transform="rotate(-90 20 ${pad.top + plotHeight / 2})"
        font-size="16" font-weight="600" fill="var(--text-primary)">${escapeHtml(options.yLabel)}</text>
    </svg>`;
}

function compactNumber(value) {
  // Axis ticks on a count scale: toPrecision alone turns 1100 into "1.10e+3".
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e4) return `${Math.round(value / 1e3)}k`;
  if (magnitude >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  if (magnitude >= 1) {
    const digits = magnitude >= 100 || Number.isInteger(value) ? 0 : 1;
    return value.toFixed(digits).replace(/\.0$/, '');
  }
  return value === 0 ? '0' : value.toPrecision(2);
}

/* --- go --------------------------------------------------------------- */

window.addEventListener('error', (event) => showFailure(
  `${event.message} (${event.filename || 'app.js'}:${event.lineno})`));
window.addEventListener('unhandledrejection', (event) => showFailure(
  (event.reason && event.reason.message) || String(event.reason)));

loadEnvironment();
initWizard();
initDropzone();
$('stop-run').addEventListener('click', stopRun);
$('resume-run').addEventListener('click', resumeRun);
restoreLastRun();
