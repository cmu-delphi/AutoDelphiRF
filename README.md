# AutoDelphiRF

AutoDelphiRF uses the revision history of one surveillance indicator to estimate
how its recently reported values may change. Give it a raw revision-history
file, and it guides you from data checks through retrospective forecasts and
evaluation. You do not need to explore or preprocess the data first.

Each indicator is forecast from its own reporting and revision history. This
keeps revision adjustment independent from any later analysis that combines
information across indicators.

```text
revision archive → diagnosis → forecasting → assessment tables
```

When setting up a new indicator retrospectively, AutoDelphiRF moves through
past report dates and recreates the forecasts that could have been made at each
date. Future reports are kept hidden, so the evaluation reflects historical
real-time use. When later releases arrive, the same analysis can be updated
without discarding its saved models or previously issued forecasts.

See the [static interface demonstration](https://cmu-delphi.github.io/AutoDelphiRF/demo.html)
without installing the package. The demonstration is illustrative and does not
upload data or run models.

## Requirements

- Python 3.10 or newer
- R and `Rscript`
- The R package [DelphiRF](https://github.com/cmu-delphi/DelphiRF/tree/refactor-clean)
- The R packages `jsonlite` and `data.table`
- The R package `arrow` when using the default Parquet triangle format

Install the R dependencies:

```bash
Rscript -e 'install.packages(c("remotes", "jsonlite", "data.table", "arrow"), repos="https://cloud.r-project.org"); remotes::install_github("cmu-delphi/DelphiRF@refactor-clean", dependencies=TRUE)'
```

DelphiRF declares its own modeling dependencies, including `quantreg`, so
`remotes::install_github(..., dependencies=TRUE)` installs them without
AutoDelphiRF maintaining a duplicate list. If installing R `arrow` is not
practical, omit it and use `--triangle-format csv`; Python `pyarrow` and R
`arrow` are separate packages, and the default Parquet workflow needs both.

Create a Python virtual environment and install AutoDelphiRF from GitHub:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "autodelphirf[parquet] @ git+https://github.com/cmu-delphi/AutoDelphiRF.git@dev"
```

Use an installed Python 3.10 or newer to create the environment (replace
`python3.10` in these commands if your interpreter has another name).
Activate the environment again with `source .venv/bin/activate` before using
AutoDelphiRF in a new terminal session.

Or install an editable copy for development:

```bash
git clone -b dev https://github.com/cmu-delphi/AutoDelphiRF.git
cd AutoDelphiRF
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[parquet]"
```

## Retrospective nowcasting setup and diagnosis for a new indicator

To set up a new indicator, provide one revision-history file containing
multiple releases of the same indicator. Each row records the value for a
location and reference date as it appeared on a particular report date.

The easiest way to begin is the local web application. It asks for the file and
settings step by step. The same analysis can also be run from one terminal
command.

### Recommended: use the local web application

Start the application:

```bash
autodelphirf web
```

It opens <http://127.0.0.1:8765> in a browser. The page and the uploaded archive
remain on your computer.

The application leads you through five steps:

1. **Describe the archive.** Select the columns containing the reference date,
   report date, location, and reported value.
2. **Review the data diagnosis.** Check the date ranges, reporting frequency,
   and evidence about how quickly values stabilize.
3. **Choose a target lag.** Decide how mature a reported value should be before
   AutoDelphiRF treats it as the training target.
4. **Recreate historical forecasts.** Choose the historical period, training
   window, retraining frequency, and forecasting methods.
5. **Review the results.** Compare forecast errors across reporting lags,
   locations, and time.

The web application saves score tables and reassessment results.

Two dates and two lags appear throughout the workflow:

- **Reference date:** when the measured event or surveillance period occurred.
- **Report date:** when that value was released or updated.
- **Reporting lag:** how old an observation was when a forecast was made.
- **Target lag:** the maturity point used to define the training outcome.

Multiple report dates for the same location and reference date show how a value
was revised.

To use another port or working directory:

```bash
autodelphirf web --port 9000 --out ~/autodelphirf_work
```

Use `autodelphirf web --no-browser` to start the server without opening a browser
automatically.

### Alternative: run the same workflow from the terminal

For your own archive using the default column names shown below, replace
`path/to/your_archive.csv` with the path to your file and run:

```bash
autodelphirf run \
  --archive path/to/your_archive.csv \
  --name mydata \
  --target-lag 28
```

This single command checks the archive, prepares the data, recreates the
historical forecasts, and writes predictions and assessment tables. Results
are written to `autodelphirf_work` by default.

If you have not chosen a target lag, inspect the evidence first:

```bash
autodelphirf diagnose --archive path/to/your_archive.csv
```

AutoDelphiRF shows a suggested range rather than choosing the target lag for
you. A shorter target lag uses more recent revision information but may define
targets that are still changing. A longer target lag uses more stable targets
but excludes more recent information. After reviewing this tradeoff, supply
your choice with `--target-lag` or enter it in the web application.

The suggested range starts at the earliest available lag where at least 90% of
historical records are within 10% relative error and ends at the earliest lag
where at least 90% are within 1% relative error. Diagnosis writes no files and
does not fit a model.

## Input data

AutoDelphiRF reads CSV and gzipped CSV revision archives. Each row is one version
of a value as it was known on a particular report date:

| geo_value | reference_date | report_date | value |
|---|---|---|---:|
| ak | 2024-03-04 | 2024-03-06 | 118 |
| ak | 2024-03-04 | 2024-03-13 | 142 |
| ak | 2024-03-04 | 2024-03-20 | 147 |
| ak | 2024-03-05 | 2024-03-06 | 95 |

- `reference_date` is when the measured event or surveillance period occurred.
- `report_date` is when that value was released or updated.
- `value` is the value available on the report date.
- `geo_value` identifies the location or time series.

The web application can map differently named columns. AutoDelphiRF supports
counts, precomputed rates or percentages, and fractions supplied as numerator
and denominator columns. A file without a geography column can be explicitly
treated as a single-location dataset; it can pool information across lags but
not across locations.

## Additional command-line options

When they are not supplied by the user, AutoDelphiRF diagnoses the reporting
cadence, revision feature lags, training-window length, and retraining calendar.
The selected target lag is always supplied or confirmed by the user.

For custom column names, replace the example file and column names below with
the names in your archive:

```bash
autodelphirf run \
  --archive path/to/your_archive.csv \
  --name mydata \
  --geo-col location_id \
  --reference-col event_date \
  --report-col issue_date \
  --value-col count \
  --value-type count \
  --target-lag 28
```

For a single-location CSV without a geography column, use `--no-geo` explicitly:

```bash
autodelphirf run \
  --archive single_series.csv \
  --name single_series \
  --no-geo \
  --target-lag 28
```

AutoDelphiRF reports that it is treating the archive as one synthetic location.
Cross-location pooling is then unavailable, while pooling across lags remains
available.

### Save the settings in one JSON file

For a reproducible analysis, the raw archive path, column mappings,
preprocessing choices, target lag, rolling calendar, and model methods can be
stored in one JSON file. First create the starter file:

```bash
autodelphirf init --kind raw --name mydata --output mydata-raw.json
```

Open `mydata-raw.json` and replace the example archive path, output path, and
column names with values for your data. Then inspect the archive:

```bash
autodelphirf diagnose --raw-config mydata-raw.json
```

After reviewing the diagnosis, enter your chosen target lag in
`preprocessing.target_lag` in the JSON file. Weekday groups are detected from the
archive's report dates unless you set `preprocessing.weekday_groups` yourself,
using ISO weekday numbers (1 = Monday ... 7 = Sunday), for example
`{"monday": [1], "weekend": [6, 7]}`. Weekdays you leave out form the
reference category, at least one must be left out, and `null` keeps the
automatic detection. Run the complete analysis with:

```bash
autodelphirf run --raw-config mydata-raw.json
```

Three top-level fields control RevRoute training and forecasting:

- `"genuine_training": true` trains only on actual releases, the rows where a
  value was reported at that lag.
- `"genuine_testing"` chooses the rows that are forecast and scored. `true`
  forecasts only actual releases; `false` also forecasts the rows the daily
  grid carries forward from an earlier release.
- `"revroute_min_pool_rows": 200` merges RevRoute pools with fewer than 200
  training rows into their nearest structural neighbour. Set it to `0` to
  keep the similarity-derived pools without a minimum size.

The genuine-event fields default to `true`, and the minimum pool size defaults
to `200`. Every prediction includes a `genuine_event` column
(`true` for an actual release), so a run with `"genuine_testing": false` can
later be evaluated on actual releases, on carried-forward rows, or on both.
These choices are recorded in `preparation.json` and stay fixed for later
`autodelphirf update` runs. Preparations made by older versions without a
recorded minimum retain their original behavior, equivalent to a value of `0`.

Paths inside the raw configuration are resolved relative to the JSON file.
Every analysis starts from a raw revision archive. `autodelphirf prepare` (or
`run --archive`) builds the prepared triangle through DelphiRF and writes a
dataset JSON next to it. To rerun that prepared dataset without rebuilding the
triangle, pass the JSON to `autodelphirf run --config`; the rerun uses only the
prepared artifacts and does not reread the raw archive. A prepared triangle
built outside AutoDelphiRF is not supported.

The repository includes a runnable version of this configuration for
`examples/my_archive.csv`. From the repository root, run:

```bash
autodelphirf diagnose --raw-config examples/my_archive.json
autodelphirf run --raw-config examples/my_archive.json
```

The checked-in example is a compact CHNG outpatient fraction archive with
three pseudonymized locations (`AB`, `CD`, and `EF`). It fixes the target lag
at 60 days and runs three monthly historical forecast origins from October 29
through December 28, 2021. The numerator counts were transformed with fixed,
location-specific multipliers; the location labels are replaced, while the
dates, denominators, and revision timing are retained. The checked-in archive
is self-contained. Results are written under `autodelphirf_work/readme_example`.

For the checked-in archive, the assessment scores 14,981 genuine forecast cases.
Its overall mean absolute error is 0.000333 for DelphiRF and 0.000437 for the
last-report baseline (23.8% lower for DelphiRF); DelphiRF also has lower mean
absolute error at each of the three origins. The paired bootstrap estimate for
`DelphiRF - baseline` is -0.000103 (95% interval -0.000128 to -0.000078).
These numbers document this fixed example rather than promising the same
improvement for other archives.

## Continue nowcasting for an existing indicator

Use this workflow when the first retrospective run is complete and a newer
release of the same revision archive becomes available. The new file must be a
complete archive containing both its earlier records and the newly reported or
revised records; do not supply only the latest rows.

Keep the original raw JSON configuration and its working directory. Replace
the archive at the path named by `archive` with each newer complete export and
run the update command every time a raw release arrives:

```bash
autodelphirf update --raw-config examples/my_archive.json
```

For your own analysis, replace `examples/my_archive.json` with the raw JSON
file used for its first run. AutoDelphiRF then:

1. verifies that the target lag, feature lags, weekday groups, training window,
   and other preparation settings agree with the first run;
2. rebuilds the prepared data from the updated complete raw archive;
3. preserves every previously issued forecast without regenerating it;
4. fills in outcomes and scores for old forecasts whose target has become
   observable;
5. fits one new pooling plan and model set only when a new retraining origin is
   due; otherwise it loads only the latest applicable models;
6. forecasts only newly received records whose target is not yet observable;
7. writes prospective calibration and PIT diagnostics for those new
   forecasts; and
8. atomically appends the new forecasts to the cumulative ledger.

Update cadence and retraining cadence are deliberately different. Invoke
`autodelphirf update` for **every raw release**, including releases between
scheduled model origins. `retraining.retrain_days` controls only when a new
pooling plan and model set are fitted. On intervening releases, AutoDelphiRF
loads the latest deployed models and forecasts the newly arrived records. If a
release is skipped, AutoDelphiRF cannot retroactively claim that forecasts were
issued when that release originally arrived.

On a scheduled retraining date with no new raw release, run the command with
`--as-of-date YYYY-MM-DD`; the model is refreshed even when there are no new
forecast rows. In an automated deployment, trigger ordinary updates from the
raw-data ingestion job and schedule the quiet-date command separately.

Realized accuracy and coverage assessments cannot be calculated for an
unresolved forecast. They are filled in by a later update after its target
becomes observable. The update-specific `calibration.csv.gz` and
`forecast_assessment_status.json` therefore describe
only forecasts issued by the latest update.

If the archive has not changed, the command exits without rebuilding the data
or fitting models. Keep the entire working directory: the prepared-data
contract, models, pooling decisions, and forecast ledger are all required for
future updates.

Run `autodelphirf run --help`, `autodelphirf update --help`, or
`autodelphirf diagnose --help` for all options.

## Results

A complete run writes its main outputs under the working directory:

```text
autodelphirf_work/
├── mydata.json
├── mydata/
│   ├── preparation.json
│   ├── inventory.csv
│   ├── test_dates.csv
│   └── prepared data files
└── results/
    ├── models/
    ├── analysis_contract.json
    ├── forecast_ledger.csv.gz
    ├── model_manifest.json
    ├── web_tables/
    ├── reassessment_decisions.csv.gz
    ├── reassessment_by_age.csv
    ├── predictions.csv.gz
    ├── resolved_config.json
    └── run_manifest.json
```

The important files for a new indicator are:

- `results/web_tables/`: score summaries for the web results view;
- `results/reassessment_decisions.csv.gz`: forecast-to-forecast decisions;
- `results/reassessment_by_age.csv`: evaluation by days since forecast issue;
- `results/predictions.csv.gz`: row-level forecasts produced by the latest
  successful run or update;
- `results/forecast_ledger.csv.gz`: forecasts preserved across later archive
  updates, with outcomes and scores added when they become available;
- `results/resolved_config.json`: the settings actually used by the run; and
- `results/run_manifest.json`: the run and input provenance record.

The R training output for each invocation is saved under `results/logs/` as a
`delphirf_*.log` file. The run manifest records its exact path under
`delphirf_provenance.training_log_file`. Failed fits and missing-predictor rows
also remain available as structured `.fit_failures.csv` and
`.dropped_test_rows.csv` files beside the model predictions. DelphiRF fitting
shows only aggregate warnings with paths to those files in the console.

The saved prediction and assessment tables support comparisons over reporting
lags, locations, and historical forecast origins.

Calibration is model-free post-processing. The existing asymmetric endpoint
correction remains the `endpoint` baseline; AutoDelphiRF also evaluates
prospective monotone quantile mapping using the location/lag hierarchy and,
when `revroute_pools.csv` is present, the current origin's RevRoute task pool.
No strategy triggers model fitting. Their coverage, width, miss direction and
WIS are compared in `calibration_strategy_comparison.csv`.

The standard run writes `reassessment_decisions.csv.gz` and
`reassessment_by_age.csv` from original DelphiRF forecasts in the forecast
ledger. Each new release compares its median with earlier forecasts for the
same location and target issued 1–14 days ago. A flag means the new median
lies outside the old central 80% interval; the normalized median jump uses
the model's working scale. Target values only grade earlier intervals after
they become observable. The age summary uses 1,000 release-date bootstrap
resamples. Run the assessment independently from saved forecasts:

```bash
autodelphirf assess --results autodelphirf_work/results
```

For a minimal test case, suppose a genuine forecast issued on January 1 has
working-scale 10th, 50th, and 90th percentiles of 1.0, 1.5, and 2.0. On
January 8, a new forecast for the same location and reference date has median
2.3. The new median exceeds the earlier 90th percentile, so the 7-day decision
is flagged; its normalized jump is `(2.3 - 1.5) / (2.0 - 1.0) = 0.8`.
On a `log1p` working scale, the earlier reporting-scale interval is approximately
`[1.72, 6.39]`. A later target value of 7 makes its retrospective interval-miss
label 1. The target value does not enter the flag or jump calculation:

```python
import numpy as np
import pandas as pd
from autodelphirf.post_assessment.reassessment import reassess

forecasts = pd.DataFrame({
    "geo_value": ["ca", "ca"],
    "reference_date": ["2023-12-31", "2023-12-31"],
    "report_date": ["2024-01-01", "2024-01-08"],
    "target_date": ["2024-01-28", "2024-01-28"],
    "q0.1_working": [1.0, 1.8],
    "q0.5_working": [1.5, 2.3],
    "q0.9_working": [2.0, 2.8],
    "miss": [1.0, np.nan],  # grade the old forecast after truth is available
})
decision = reassess(forecasts).iloc[0]
assert decision.age == 7 and decision.flag == 1
assert np.isclose(decision.jump, 0.8) and decision.miss == 1
```

Standalone assessment uses genuine test events by default. Rows with
`genuine_event == false` (including carried-forward pseudo-origins) are removed
before calibration and affect neither its historical strata nor its metrics.
Selecting `genuine` or `carried` requires the saved predictions to contain the
`genuine_event` column; AutoDelphiRF fails explicitly if that provenance is
unavailable.

`quantile_reliability.csv` and `pit_diagnostics.csv.gz` are diagnostic add-ons.
Because DelphiRF exports nine quantiles rather than a continuous predictive
CDF, PIT is reported as the probability bin containing the observation and
its midpoint. When the truth ties one or more forecast quantiles within the
same relative boundary tolerance used for interval coverage, `pit_lower` and
`pit_upper` span the full tied probability range; `pit_tied` and
`pit_tied_quantiles` make those cases explicit. This avoids assigning tied or
discrete forecasts to a rounding-dependent single bin. An existing completed
run can generate these optional artifacts without refitting models.

PIT diagnostics can run independently on the original forecast quantiles:

```bash
autodelphirf assess \
  --results autodelphirf_work/results \
  --pit-only
```

To run endpoint recalibration, the two quantile-mapping strategies, and their
empirical coverage/PIT comparisons without bootstrap inference, use:

```bash
autodelphirf assess \
  --results autodelphirf_work/results \
  --recalibration-only
```

The default reassessment accepts genuine forecasts only. For optional PIT and
recalibration, use `--event-filter carried` to assess only carried-forward
forecasts or `--event-filter all` to include both.

PIT-only assessment writes `quantile_reliability.csv` and
`pit_diagnostics.csv.gz`. It reads only the columns required from
`predictions.csv.gz`; it does not load `predictions_wide.csv.gz`, pool
assignments, runtime tables, or plotting dependencies. Its assessment manifest
therefore lists only the inputs actually consumed by PIT diagnostics.
Recalibration-only assessment additionally writes
`calibration.csv.gz`,
`calibration_variants.csv.gz`, `calibration_strategy_comparison.csv`,
and `interval_calibration.csv`. Files go into a new timestamped assessment
directory; the original predictions are not changed. `endpoint` only expands interval
endpoints; `quantile_map_location_lag` and `quantile_map_pool` are symmetric
quantile mappings that may widen or narrow the distribution. If a prospective
stratum lacks sufficient completed history, its original quantiles are kept
and the fallback source and history count are recorded.

Normal retrospective runs generate them with the rest of the assessment. An
update for an existing indicator refreshes the diagnostics from the cumulative
forecast ledger as outcomes become observable and applies prospective calibration to
new genuine forecasts using only genuine outcomes available at their issue
time. Carried-forward predictions remain in the prediction ledger, but an
update containing only carried-forward rows intentionally emits
no calibrated intervals for that update.

## Development

Install development dependencies and run the tests:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Python 3.10 or newer is supported. The slower integration tests
also require R and a compatible DelphiRF installation.

## License

AutoDelphiRF is distributed under the MIT License. See `LICENSE`.
