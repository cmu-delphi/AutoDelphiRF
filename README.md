# AutoDelphiRF

AutoDelphiRF uses the revision history of one surveillance indicator to estimate
how its recently reported values may change. Give it a raw revision-history
file, and it guides you from data checks through retrospective forecasts and
evaluation. You do not need to explore or preprocess the data first.

Each indicator is forecast from its own reporting and revision history. This
keeps revision adjustment independent from any later analysis that combines
information across indicators.

```text
revision archive → diagnosis → forecasting → evaluation report
```

The current application is designed for a **one-time retrospective analysis**.
It moves through past report dates and recreates the forecasts that could have
been made at each date. Future reports are kept hidden, so the evaluation
reflects historical real-time use.

See the [static interface demonstration](https://cmu-delphi.github.io/AutoDelphiRF/demo.html)
without installing the package. The demonstration is illustrative and does not
upload data or run models.

## Requirements

- Python 3.10 or newer
- R and `Rscript`
- The R package [DelphiRF](https://github.com/cmu-delphi/DelphiRF/tree/refactor-clean)
- The R package `quantreg`

Install the R dependencies:

```bash
Rscript -e 'install.packages(c("remotes", "quantreg"), repos="https://cloud.r-project.org"); remotes::install_github("cmu-delphi/DelphiRF@refactor-clean")'
```

Create a Python virtual environment and install AutoDelphiRF from GitHub:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "autodelphirf[parquet] @ git+https://github.com/cmu-delphi/AutoDelphiRF.git@dev"
```

The `python3` used to create the environment must be version 3.10 or newer.
Activate the environment again with `source .venv/bin/activate` before using
AutoDelphiRF in a new terminal session.

Or install an editable copy for development:

```bash
git clone -b dev https://github.com/cmu-delphi/AutoDelphiRF.git
cd AutoDelphiRF
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[parquet]"
```

## First-time retrospective workflow

A first-time user needs one revision-history file containing multiple releases
of the same indicator. Each row records the value for a location and reference
date as it appeared on a particular report date.

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

For an archive using the default column names shown below, replace
`my_archive.csv` with the path to your file and run:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --target-lag 28
```

This single command checks the archive, prepares the data, recreates the
historical forecasts, and writes the predictions and evaluation report. Results
are written to `autodelphirf_work` by default.

If you have not chosen a target lag, inspect the evidence first:

```bash
autodelphirf diagnose --archive my_archive.csv
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
  --archive my_archive.csv \
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
`preprocessing.target_lag` in the JSON file. Run the complete analysis with:

```bash
autodelphirf run --raw-config mydata-raw.json
```

Paths inside the raw configuration are resolved relative to the JSON file.
`autodelphirf init --kind prepared` remains available for the less common case
of registering an existing prepared triangle.

Run `autodelphirf run --help` or `autodelphirf diagnose --help` for all options.

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
    ├── report/report.html
    ├── predictions.csv.gz
    ├── resolved_config.json
    └── run_manifest.json
```

The important files for a first-time user are:

- `results/report/report.html`: the evaluation report;
- `results/predictions.csv.gz`: complete row-level forecasts and scores;
- `results/resolved_config.json`: the settings actually used by the run; and
- `results/run_manifest.json`: the run and input provenance record.

The report compares nowcast accuracy and, where available, probabilistic
forecast performance over reporting lags, locations, and historical forecast
origins.

## Development

Install development dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The repository's slower integration tests require R and a compatible DelphiRF
installation.

## License

AutoDelphiRF is distributed under the MIT License. See `LICENSE`.
