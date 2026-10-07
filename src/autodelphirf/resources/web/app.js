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
  confirmLag: false,        // true when that answer overrides the recommendation
  environment: null,        // R/DelphiRF readiness, as reported at startup
  schedule: null,           // the retraining calendar last previewed by the server
  results: null,            // the finished run's summary, as /api/results returned it
  polling: null,
};

/* --- plumbing --------------------------------------------------------- */

async function api(path, {method = 'GET', body = null, raw = false} = {}) {
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
         <code>Rscript -e 'remotes::install_github("cmu-delphi/DelphiRF@refactor-clean")'</code><br>
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
    columnSelect('col-reference', 'Reference date <span class="hint">when it happened</span>',
                 guesses.reference),
    columnSelect('col-report', 'Report date <span class="hint">when it was published</span>',
                 guesses.report),
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

async function runDiagnosis(userTargetLag = null, acceptCustomLag = false) {
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
    if (acceptCustomLag) {
      acceptLag(
        Number(userTargetLag),
        Number(userTargetLag) !== Number(state.diagnosis.recommended_target_lag)
      );
      return;
    }
    // A re-diagnosis is an answer to the target-lag question, so scroll to the
    // answer buttons rather than back to the top of the card: the recommendation
    // and the curve are already read by then, and landing above them hides the
    // one control that unlocks the rest of the page.
    scrollIntoView(userTargetLag != null ? 'lag-answers' : 'card-diagnosis');
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

function fact(term, value) {
  return `<div><dt>${term}</dt><dd>${value}</dd></div>`;
}

function renderDiagnosis(report, userTargetLag) {
  $('diagnosis-facts').innerHTML = [
    fact('Rows', report.n_rows.toLocaleString()),
    fact('Locations', report.n_locations.toLocaleString()),
    fact('Reference dates', `${report.reference_date_min} → ${report.reference_date_max}`),
    fact('Reference-date cadence', report.reference_axis_resolution),
    fact('Report-date cadence', report.report_axis_resolution),
    fact('Genuine revisions', `${(report.genuine_event_rate * 100).toFixed(1)}% of rows`),
    fact('Revision feature lags', (report.reference_axis_feature_lags || []).join(', ') || '—'),
    fact('Training window', `${report.training_window_days} days`),
  ].join('');

  const notes = (report.notes || []).map((note) => notice('warn', escapeHtml(note))).join('');
  $('diagnosis-notes').innerHTML = notes;

  drawCompletionCurve(report);
  renderLagQuestion(report, userTargetLag);
  if (report.retraining) initSchedule(report.retraining);
}

function renderLagQuestion(report, userTargetLag) {
  const recommended = report.recommended_target_lag;
  const body = $('lag-question-body');
  const answers = $('lag-answers');
  const custom = $('custom-lag');

  body.innerHTML = `AutoDelphiRF forecasts the target value reported at a prespecified
    target lag. From your archive's historical revisions, the recommendation is
    <strong>${recommended} days</strong> &mdash; the first candidate target lag at which the
    relative error is within 10% for at least 90% of location and reference-date
    combinations. The chart below still
    shows median relative error as supplementary diagnostic information.`;

  if (report.target_lag_confirmation_prompt) {
    body.insertAdjacentHTML('afterend', '');
    const prompt = escapeHtml(report.target_lag_confirmation_prompt);
    answers.innerHTML = `<div style="width:100%">${notice('warn', prompt)}</div>`;
  } else {
    answers.innerHTML = '';
  }

  // The two real answers, plus the escape hatch. When the user has supplied a
  // value the server's resolution already picked a default; the buttons make
  // both options explicit so the default is chosen, not merely accepted by
  // silence.
  if (userTargetLag != null) {
    $('custom-lag-note').textContent =
      `Re-diagnosed with ${userTargetLag} days. Choose which target lag to use from the `
      + 'buttons above, and the retraining schedule opens next.';
  }

  const options = [];
  options.push({
    id: 'accept',
    label: `Use ${recommended} days`,
    sub: 'the recommendation',
    lag: recommended,
    confirm: false,
  });
  if (userTargetLag != null && userTargetLag !== recommended) {
    options.push({
      id: 'mine',
      label: `Use ${userTargetLag} days`,
      sub: 'my value, overriding the recommendation',
      lag: userTargetLag,
      confirm: true,
    });
  }

  answers.insertAdjacentHTML('beforeend', options.map((option) => `
    <button class="secondary" data-lag="${option.lag}" data-confirm="${option.confirm}"
            data-id="${option.id}">
      <span class="layer-name">${escapeHtml(option.label)}</span><br>
      <span class="layer-desc">${escapeHtml(option.sub)}</span>
    </button>`).join('') + `
    <button class="secondary" id="choose-other">
      <span class="layer-name">Use a different value</span><br>
      <span class="layer-desc">I know this stream's target lag</span>
    </button>`);

  answers.querySelectorAll('button[data-lag]').forEach((button) => {
    button.addEventListener('click', () => {
      answers.querySelectorAll('button').forEach((other) => other.classList.remove('chosen'));
      button.classList.add('chosen');
      custom.hidden = true;
      acceptLag(Number(button.dataset.lag), button.dataset.confirm === 'true');
    });
  });
  $('choose-other').addEventListener('click', () => {
    custom.hidden = false;
    $('custom-lag-input').value = userTargetLag != null ? userTargetLag : recommended;
    $('custom-lag-note').textContent =
      'AutoDelphiRF will re-diagnose with your value and show you how the two compare.';
    $('custom-lag-input').focus();
  });
  $('custom-lag-apply').onclick = () => {
    const value = Number($('custom-lag-input').value);
    if (!Number.isFinite(value) || value < 1) {
      $('custom-lag-note').textContent = 'Enter a whole number of days.';
      return;
    }
    runDiagnosis(value, true);
  };
}

function acceptLag(lag, confirm) {
  state.answeredLag = lag;
  state.confirmLag = confirm;
  show('card-schedule');
  loadLayers().then(() => {
    show('card-methods');
    scrollIntoView('card-methods');
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
    + `which reports from ${retraining.report_date_min} to ${retraining.report_date_max}`;
  renderSchedule(retraining);

  $('retrain-days').oninput = previewSchedule;
  $('first-origin-date').onchange = previewSchedule;
  $('schedule-reset').onclick = () => {
    $('retrain-days').value = retraining.default_retrain_days;
    $('first-origin-date').value = retraining.first_origin;
    previewSchedule();
  };
  $('schedule-continue').onclick = () => {
    loadLayers().then(() => {
      show('card-methods');
      scrollIntoView('card-methods');
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
        break. A longer interval, or a later first origin, shortens it.`)
    : '';
  $('schedule-preview').innerHTML = `
    <dl class="facts">
      ${fact('Retrainings', `<strong>${origins}</strong>`)}
      ${fact('First origin', `${preview.first_origin}
        <span class="hint">${preview.history_days_at_first_origin} days of history behind it</span>`)}
      ${fact('Last origin', preview.last_origin)}
      ${fact('Interval', `${preview.retrain_days} days`)}
    </dl>${cost}`;
}

/* --- the completion curve --------------------------------------------- */

function drawCompletionCurve(report) {
  const curve = report.target_lag_completion_curve || {};
  const band = report.target_lag_completion_band || {};
  const points = Object.keys(curve)
    .map((lag) => ({
      lag: Number(lag),
      error: Number(curve[lag]),
      q10: Number((band[lag] || {}).q10),
      q90: Number((band[lag] || {}).q90),
    }))
    .filter((point) => Number.isFinite(point.lag) && Number.isFinite(point.error))
    .sort((a, b) => a.lag - b.lag);

  const container = $('completion-chart');
  if (points.length < 2) {
    container.innerHTML = `<p class="hint">Not enough candidate target lags to plot a curve.</p>`;
    $('completion-table').innerHTML = '';
    return;
  }

  const width = 640;
  const height = 240;
  const pad = {top: 14, right: 18, bottom: 34, left: 66};
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const maxLag = points[points.length - 1].lag;
  const bandPoints = points.filter((point) => Number.isFinite(point.q10) && Number.isFinite(point.q90));
  const maxError = Math.max(0.12, ...points.map((point) => point.error),
                            ...bandPoints.map((point) => point.q90));
  const x = (lag) => pad.left + (lag / maxLag) * plotWidth;
  const y = (error) => pad.top + plotHeight - (error / maxError) * plotHeight;

  const recommended = report.recommended_target_lag;
  const recommendedPoint = points.find((point) => point.lag === recommended);

  // Y gridlines at round shares.
  const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0].filter((t) => t <= maxError);
  const gridlines = ticks.map((tick) => `
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${y(tick).toFixed(1)}"
          y2="${y(tick).toFixed(1)}" stroke="var(--border)" stroke-width="1"/>
    <text x="${pad.left - 8}" y="${(y(tick) + 4).toFixed(1)}" text-anchor="end"
          font-size="11" fill="var(--text-muted)">${(tick * 100).toFixed(0)}%</text>`).join('');

  const xTicks = points.filter((_, index) => index % Math.ceil(points.length / 7) === 0);
  const xLabels = xTicks.map((point) => `
    <text x="${x(point.lag).toFixed(1)}" y="${height - 12}" text-anchor="middle"
          font-size="11" fill="var(--text-muted)">${point.lag}</text>`).join('');

  const line = points.map((point, index) =>
    `${index === 0 ? 'M' : 'L'}${x(point.lag).toFixed(1)},${y(point.error).toFixed(1)}`).join(' ');

  const shadedBand = bandPoints.length === points.length ? `
    <path d="${[
      ...bandPoints.map((point, index) =>
        `${index === 0 ? 'M' : 'L'}${x(point.lag).toFixed(1)},${y(point.q90).toFixed(1)}`),
      ...bandPoints.slice().reverse().map((point) =>
        `L${x(point.lag).toFixed(1)},${y(point.q10).toFixed(1)}`),
      'Z',
    ].join(' ')}" fill="var(--series-1)" opacity="0.18" stroke="none"/>
    <text x="${width - pad.right}" y="${pad.top + 12}" text-anchor="end"
          font-size="11" fill="var(--text-muted)">10th–90th percentile</text>` : '';

  // The 10% tolerance the rule is defined against, as a reference line.
  const tolerance = 0.10;
  const toleranceLine = tolerance <= maxError ? `
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${y(tolerance).toFixed(1)}"
          y2="${y(tolerance).toFixed(1)}" stroke="var(--text-secondary)" stroke-width="2"
          stroke-dasharray="5 4"/>
    <text x="${width - pad.right}" y="${(y(tolerance) - 7).toFixed(1)}" text-anchor="end"
          font-size="11.5" font-weight="600" fill="var(--text-secondary)">10% tolerance</text>` : '';

  const markers = points.map((point) => `
    <circle cx="${x(point.lag).toFixed(1)}" cy="${y(point.error).toFixed(1)}" r="4"
            fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2"
            data-lag="${point.lag}" data-error="${point.error}"
            data-q10="${point.q10}" data-q90="${point.q90}"/>`).join('');

  // The recommendation: a vertical rule and a direct label, not a second hue.
  const recommendedMark = recommendedPoint ? `
    <line x1="${x(recommended).toFixed(1)}" x2="${x(recommended).toFixed(1)}"
          y1="${pad.top}" y2="${pad.top + plotHeight}" stroke="var(--series-1)"
          stroke-width="2" stroke-dasharray="3 3" opacity="0.55"/>
    <circle cx="${x(recommended).toFixed(1)}" cy="${y(recommendedPoint.error).toFixed(1)}" r="6.5"
            fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2.5"/>
    <text x="${x(recommended).toFixed(1)}" y="${pad.top - 2}" text-anchor="middle"
          font-size="12" font-weight="700" fill="var(--text-primary)">${recommended} days</text>` : '';

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img"
         aria-label="Median relative error by candidate target lag with a 10th to 90th
                     percentile band. Recommended target lag ${recommended} days.">
      ${gridlines}
      ${toleranceLine}
      ${shadedBand}
      <path d="${line}" fill="none" stroke="var(--series-1)" stroke-width="2"
            stroke-linejoin="round" stroke-linecap="round"/>
      <text x="${pad.left + 8}" y="${pad.top + 12}" font-size="11"
            font-weight="600" fill="var(--text-secondary)">Median</text>
      ${markers}
      ${recommendedMark}
      <line x1="${pad.left}" x2="${width - pad.right}" y1="${pad.top + plotHeight}"
            y2="${pad.top + plotHeight}" stroke="var(--border-strong)" stroke-width="1"/>
      <text x="${pad.left + plotWidth / 2}" y="${height - 1}" text-anchor="middle"
            font-size="11.5" fill="var(--text-secondary)">candidate target lag (days after reference date)</text>
      <text x="15" y="${pad.top + plotHeight / 2}" text-anchor="middle"
            transform="rotate(-90 15 ${pad.top + plotHeight / 2})"
            font-size="11.5" fill="var(--text-secondary)">relative error</text>
    </svg>
    <div class="tooltip" id="chart-tooltip"></div>`;

  attachChartHover(container, points, recommended);
  renderCompletionTable(points, recommended);
}

function attachChartHover(container, points, recommended) {
  const tooltip = $('chart-tooltip');
  container.querySelectorAll('circle[data-lag]').forEach((marker) => {
    const showTip = () => {
      const lag = marker.dataset.lag;
      const error = (Number(marker.dataset.error) * 100).toFixed(1);
      const q10 = Number(marker.dataset.q10);
      const q90 = Number(marker.dataset.q90);
      const interval = Number.isFinite(q10) && Number.isFinite(q90)
        ? ` · 10th–90th: ${(q10 * 100).toFixed(1)}%–${(q90 * 100).toFixed(1)}%`
        : '';
      tooltip.innerHTML = `${lag} days · ${error}% median error${interval}` +
        (Number(lag) === recommended ? ' · recommended' : '');
      const box = marker.getBoundingClientRect();
      const parent = container.getBoundingClientRect();
      tooltip.style.left = `${box.left - parent.left + box.width / 2}px`;
      tooltip.style.top = `${box.top - parent.top}px`;
      tooltip.classList.add('visible');
    };
    // A bigger hit target than the 4px mark.
    marker.setAttribute('pointer-events', 'all');
    marker.style.cursor = 'crosshair';
    marker.addEventListener('mouseenter', showTip);
    marker.addEventListener('focus', showTip);
    marker.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));
  });
  container.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));
}

function renderCompletionTable(points, recommended) {
  $('completion-table').innerHTML = `
    <table>
      <thead><tr><th>Candidate target lag</th><th>Median relative error</th></tr></thead>
      <tbody>${points.map((point) => `
        <tr class="${point.lag === recommended ? 'recommended' : ''}">
          <td>${point.lag} days${point.lag === recommended ? ' (recommended)' : ''}</td>
          <td>${(point.error * 100).toFixed(1)}%</td>
        </tr>`).join('')}</tbody>
    </table>`;
}

/* --- 5. methods ------------------------------------------------------- */

let layersLoaded = false;

const METHOD_LABELS = {
  baseline_null: 'Latest reported value',
  red: 'Revision-pattern matching',
  delphirf: 'RevRoute pooling + DelphiRF',
  naive_delphirf: 'Separate DelphiRF by location',
  similarity_weighted_delphirf: 'Similarity-weighted DelphiRF',
  global_delphirf: 'One DelphiRF model for all locations',
};

function methodLabel(name) {
  return METHOD_LABELS[name] || String(name).replace(/_/g, ' ');
}

function comparisonLabel(comparison) {
  return String(comparison).split(' - ').map(methodLabel).join(' compared with ');
}

async function loadLayers() {
  if (layersLoaded) return;
  const {layers} = await api('/api/layers');
  $('layer-list').innerHTML = layers.map((layer) => `
    <label class="layer">
      <input type="checkbox" value="${escapeHtml(layer.name)}"${layer.default ? ' checked' : ''}>
      <span>
        <span class="layer-name">${escapeHtml(layer.label || methodLabel(layer.name))}</span><br>
        <span class="layer-desc">${escapeHtml(layer.description)}</span>
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
    scrollIntoView('card-run');
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
    scrollIntoView('card-run');
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
    }
    if (snapshot.state === 'idle') return;
    show('card-run');
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
    $('results-headline').innerHTML = notice('warn',
      `The run finished, but its tables could not be read: ${escapeHtml(error.message)}`);
    show('card-results');
    return;
  }
  state.results = results;
  const tables = results.tables || {};
  renderHeadline(tables.headline || [], tables.availability || []);
  renderByOrigin(tables.by_origin || []);
  renderPooling(tables.revroute_pools || [], tables.revroute_pool_profile || []);
  $('results-links').innerHTML = `
    ${results.report_available
      ? '<a class="report-link" href="/report/report/report.html" target="_blank" rel="noopener">Open the full report</a>'
      : notice('warn', 'The run finished but no report file was written.')}
    <p class="outputs">Every table above, and a great many more, were written to
      <code>${escapeHtml(results.output_dir)}</code>.</p>`;
  show('card-results');
  scrollIntoView('card-results');
}

function compactNumber(value) {
  // Axis ticks on a count scale: toPrecision alone turns 1100 into "1.10e+3".
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e4) return `${Math.round(value / 1e3)}k`;
  if (magnitude >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  if (magnitude >= 1) return value.toFixed(magnitude >= 100 ? 0 : 1);
  return value === 0 ? '0' : value.toPrecision(2);
}

function percent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : '\u2014';
}

function signed(value) {
  if (!Number.isFinite(value)) return '\u2014';
  return `${value > 0 ? '+' : ''}${value.toPrecision(3)}`;
}

function renderHeadline(headline, availability) {
  const available = new Map(availability.map((row) => [row.method, row.availability]));
  if (!headline.length) {
    $('results-headline').innerHTML = notice('warn',
      'No pairwise comparison was written, so there is nothing to score against the baseline.');
    return;
  }
  // A negative mean delta is less loss than the baseline, so the method won.
  const rows = headline.map((row) => {
    const better = row.mean_delta < 0;
    return `<tr>
      <td>${escapeHtml(comparisonLabel(row.comparison))}</td>
      <td>${escapeHtml(String(row.loss || '').replace(/_/g, ' '))}</td>
      <td class="num">${row.n}</td>
      <td class="num ${better ? 'good' : 'bad'}">${signed(row.mean_delta)}</td>
      <td class="num">${percent(row.win_probability)}</td>
    </tr>`;
  }).join('');
  const thin = [...available].filter(([, share]) => share < 1);
  $('results-headline').innerHTML = `
    <p class="hint">Each row is one method against the reference, pooled over every
      retraining origin. A negative mean change is less error than the reference;
      &ldquo;wins&rdquo; is the share of individual cases it beat.</p>
    <table class="result-table">
      <thead><tr><th>Comparison</th><th>Loss</th><th class="num">Cases</th>
        <th class="num">Mean change in error</th><th class="num">Wins</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${thin.length ? notice('warn', `Not every method produced a prediction everywhere: `
      + thin.map(([m, share]) => `${escapeHtml(methodLabel(m))} ${percent(share)}`).join(', ')
      + '. Thin early history is the usual reason.') : ''}`;
}

function renderByOrigin(byOrigin) {
  const figure = $('origin-figure');
  if (!byOrigin.length) {
    figure.hidden = true;
    return;
  }
  figure.hidden = false;
  const methods = [...new Set(byOrigin.map((row) => row.method))];
  const cutoffs = [...new Set(byOrigin.map((row) => row.cutoff))].sort();
  $('origin-fig-sub').textContent =
    `${cutoffs.length} retraining origins, ${methods.length} methods. `
    + 'Each point is one origin: the model refitted there, then scored on what it predicted next.';
  renderOriginShortfall(cutoffs.length);
  drawOriginChart(byOrigin, methods, cutoffs);
  $('origin-table').innerHTML = `
    <table class="result-table">
      <thead><tr><th>Retraining date</th>${methods.map((m) =>
        `<th class="num">${escapeHtml(methodLabel(m))}</th>`).join('')}</tr></thead>
      <tbody>${cutoffs.map((cutoff) => {
        const cells = methods.map((method) => {
          const row = byOrigin.find((r) => r.cutoff === cutoff && r.method === method);
          return `<td class="num">${row ? row.mean_ae.toPrecision(4) : '\u2014'}</td>`;
        }).join('');
        return `<tr><td>${escapeHtml(cutoff)}</td>${cells}</tr>`;
      }).join('')}</tbody>
    </table>`;
}

function renderOriginShortfall(evaluated) {
  // The newest origins do not have their target values yet: scoring one needs
  // L days of follow-up the archive does not have, so the schedule asks for more
  // origins than the replay can score. Said plainly rather than quietly
  // showing the smaller number.
  const scheduled = state.results ? state.results.origins_scheduled : 0;
  const note = $('origin-shortfall');
  if (!scheduled || scheduled <= evaluated) {
    note.innerHTML = '';
    return;
  }
  note.innerHTML = notice('warn',
    `The schedule asked for <strong>${scheduled}</strong> retrainings but
     <strong>${evaluated}</strong> could be scored. The rest are too recent: scoring an origin
     needs the target value at lag L, and those values are not available yet.`);
}

function renderPooling(pools, profile) {
  const container = $('results-pooling');
  if (!pools.length) {
    container.innerHTML = '';
    return;
  }
  // RevRoute relearns its pools at every origin, so these are per-retraining
  // results, not one property of the run.
  const byFold = new Map();
  profile.forEach((row) => {
    if (!byFold.has(row.cutoff)) byFold.set(row.cutoff, []);
    byFold.get(row.cutoff).push(row);
  });
  const rows = pools.map((pool) => {
    const detail = (byFold.get(pool.cutoff) || [])
      .map((row) => `group ${row.pool}: ${row.tasks} location-lag combination(s) across ${row.locations} location(s)`)
      .join('; ');
    return `<tr>
      <td>${escapeHtml(String(pool.cutoff).slice(0, 10))}</td>
      <td class="num">${pool.represented_tasks ?? pool.tasks ?? '\u2014'}</td>
      <td class="num">${pool.pool_count ?? pool.k_selected ?? '\u2014'}</td>
      <td class="num">${pool.largest_pool ?? '\u2014'}</td>
      <td class="num">${pool.singleton_pools ?? '\u2014'}</td>
      <td class="pool-detail">${escapeHtml(detail)}</td>
    </tr>`;
  }).join('');
  container.innerHTML = `
    <h3>How similar location and reporting-lag combinations were grouped</h3>
    <p class="hint">The method relearns which location and reporting-lag combinations can
      share one quantile regression every time it retrains. A group count that changes over
      time reflects changes in the revision process. Groups containing only one combination
      indicate that the archive did not support combining it with another.</p>
    <details class="table-view" open>
      <summary>${pools.length} retrainings</summary>
      <table class="result-table">
        <thead><tr><th>Retraining date</th><th class="num">Location-lag combinations</th><th class="num">Groups</th>
          <th class="num">Largest</th><th class="num">Singletons</th>
          <th>Composition</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </details>`;
}

/* The per-origin chart. Change over time, several methods -> a multi-series
 * line. Colour is assigned by method in fixed slot order and never cycled;
 * identity is carried by a legend and the table view as well as the hue,
 * which is also what the light-mode contrast relief on slots 3-5 requires.
 */
const SERIES_SLOTS = 8;

function drawOriginChart(rows, methods, cutoffs) {
  const container = $('origin-chart');
  const width = 720;
  const height = 280;
  const pad = {top: 14, right: 18, bottom: 38, left: 62};
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const series = methods.slice(0, SERIES_SLOTS).map((method, index) => ({
    method,
    color: `var(--series-${index + 1})`,
    points: cutoffs.map((cutoff, position) => {
      const row = rows.find((r) => r.cutoff === cutoff && r.method === method);
      return row && Number.isFinite(row.mean_ae) ? {position, cutoff, value: row.mean_ae} : null;
    }).filter(Boolean),
  })).filter((entry) => entry.points.length);

  const values = series.flatMap((entry) => entry.points.map((point) => point.value));
  if (!values.length || cutoffs.length < 2) {
    container.innerHTML = `<p class="hint">Not enough retraining origins to plot a line.</p>`;
    return;
  }
  // A method that is exactly right everywhere would otherwise divide by zero.
  const maxValue = Math.max(...values) || 1;
  const x = (position) => pad.left + (cutoffs.length === 1 ? plotWidth / 2
    : (position / (cutoffs.length - 1)) * plotWidth);
  const y = (value) => pad.top + plotHeight - (value / maxValue) * plotHeight;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((share) => share * maxValue);
  const gridlines = ticks.map((tick) => `
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${y(tick).toFixed(1)}"
          y2="${y(tick).toFixed(1)}" stroke="var(--border)" stroke-width="1"/>
    <text x="${pad.left - 8}" y="${(y(tick) + 4).toFixed(1)}" text-anchor="end"
          font-size="11" fill="var(--text-muted)">${compactNumber(tick)}</text>`).join('');

  // At most seven dated labels, however many origins there are -- and the last
  // origin always gets one, so the axis does not appear to stop short of where
  // the line actually ends. Its neighbour is dropped if they would collide.
  const step = Math.ceil(cutoffs.length / 7);
  const last = cutoffs.length - 1;
  const labelled = new Set(cutoffs.map((_, position) => position).filter((p) => p % step === 0));
  labelled.delete(last - (last % step));
  if (last - Math.max(...labelled, 0) < step / 2) labelled.delete(Math.max(...labelled, 0));
  labelled.add(last);
  const xLabels = [...labelled].sort((a, b) => a - b).map((position) => `
    <text x="${x(position).toFixed(1)}" y="${height - 16}" text-anchor="middle"
          font-size="11" fill="var(--text-muted)">${escapeHtml(String(cutoffs[position]).slice(0, 7))}</text>`)
    .join('');

  // Markers only when the origins are few enough that they do not merge into
  // a solid band; the line carries the shape either way.
  const showMarkers = cutoffs.length <= 40;
  const paths = series.map((entry) => {
    const line = entry.points.map((point, index) =>
      `${index === 0 ? 'M' : 'L'}${x(point.position).toFixed(1)},${y(point.value).toFixed(1)}`).join(' ');
    const markers = showMarkers ? entry.points.map((point) => `
      <circle cx="${x(point.position).toFixed(1)}" cy="${y(point.value).toFixed(1)}" r="4"
              fill="${entry.color}" stroke="var(--surface-1)" stroke-width="2"/>`).join('') : '';
    return `<path d="${line}" fill="none" stroke="${entry.color}" stroke-width="2"
                  stroke-linejoin="round" stroke-linecap="round"/>${markers}`;
  }).join('');

  const legend = series.map((entry) => `
    <li><span class="swatch" style="background:${entry.color}"></span>
      ${escapeHtml(methodLabel(entry.method))}</li>`).join('');

  container.innerHTML = `
    <ul class="legend">${legend}</ul>
    <svg viewBox="0 0 ${width} ${height}" role="img"
         aria-label="Mean absolute error at each of ${cutoffs.length} retraining origins,
                     for ${series.map((entry) => entry.method).join(', ')}.">
      ${gridlines}
      ${paths}
      <line x1="${pad.left}" x2="${width - pad.right}" y1="${pad.top + plotHeight}"
            y2="${pad.top + plotHeight}" stroke="var(--border-strong)" stroke-width="1"/>
      ${xLabels}
      <text x="${pad.left + plotWidth / 2}" y="${height - 1}" text-anchor="middle"
            font-size="11.5" fill="var(--text-secondary)">retraining origin</text>
    </svg>`;
}

/* --- go --------------------------------------------------------------- */

window.addEventListener('error', (event) => showFailure(
  `${event.message} (${event.filename || 'app.js'}:${event.lineno})`));
window.addEventListener('unhandledrejection', (event) => showFailure(
  (event.reason && event.reason.message) || String(event.reason)));

loadEnvironment();
initDropzone();
$('stop-run').addEventListener('click', stopRun);
$('resume-run').addEventListener('click', resumeRun);
restoreLastRun();
