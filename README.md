# AutoDelphiRF

AutoDelphiRF is an end-to-end pipeline for data that are revised after first
publication. It can inspect a revision archive, prepare the modeling data,
train and test forecasting models retrospectively, evaluate their predictions,
run post-forecast assessments, and produce a local HTML report.

You can use AutoDelphiRF from the terminal or run the complete workflow in its
local web application.

## Requirements

- Python 3.10 or newer
- R and `Rscript`
- The R package [DelphiRF](https://github.com/cmu-delphi/DelphiRF/tree/refactor-clean),
  installed from its `refactor-clean` branch
- The R package `quantreg`

Install the R dependencies:

```bash
Rscript -e 'install.packages(c("remotes", "quantreg")); remotes::install_github("cmu-delphi/DelphiRF@refactor-clean")'
```

This branch of AutoDelphiRF needs DelphiRF's `refactor-clean` API. The DelphiRF
on `main` does not have it yet, so `remotes::install_github("cmu-delphi/DelphiRF")`
installs a DelphiRF that AutoDelphiRF reports as too old.

Pre-diagnosis runs entirely in Python. R and DelphiRF are required when the
pipeline prepares data or trains a DelphiRF model.

AutoDelphiRF always uses the DelphiRF installed in R's library, for both data
preparation and model training; it never loads a DelphiRF source checkout. After
reinstalling DelphiRF, reload the `autodelphirf web` page to pick it up.

## Install AutoDelphiRF

Install directly from GitHub:

```bash
python -m pip install "autodelphirf[parquet] @ git+https://github.com/cmu-delphi/AutoDelphiRF.git@dev"
```

Or install from a clone:

```bash
git clone -b dev https://github.com/cmu-delphi/AutoDelphiRF.git
cd AutoDelphiRF
python -m pip install -e ".[parquet]"
```

The `parquet` extra installs `pyarrow`, which is needed for the default
prepared-data format. Use `--triangle-format csv` if you do not want Parquet.

Confirm the installation:

```bash
autodelphirf --version
autodelphirf --help
```

## Input data

AutoDelphiRF accepts CSV, gzipped CSV, and Parquet files. The input is a long
revision archive with one row for every published value:

| geo_value | reference_date | report_date | value |
|---|---|---|---:|
| ak | 2024-03-04 | 2024-03-06 | 118 |
| ak | 2024-03-04 | 2024-03-13 | 142 |
| ak | 2024-03-04 | 2024-03-20 | 147 |
| ak | 2024-03-05 | 2024-03-06 | 95 |

- `reference_date`: the date the measurement describes.
- `report_date`: the date that version of the value was published.
- `value`: the value available on that report date.
- `geo_value`: the location or series identifier.

Column names can be changed with command-line options. A single-location file
without a location column can use `--no-geo`.

### Counts and one-column fractions

For a count or a fraction stored in one value column:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --value-col value \
  --value-type count \
  --out autodelphirf_work
```

Use `--value-type fraction` if the one column represents a fraction.

### Fractions with numerator and denominator

Pass `--value-col` twice, numerator first and denominator second:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --value-type fraction \
  --value-col numerator \
  --value-col denominator \
  --out autodelphirf_work
```

### Custom column names

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --geo-col location_id \
  --reference-col event_date \
  --report-col issue_date \
  --value-col count \
  --out autodelphirf_work
```

## Pre-diagnosis only

Use `diagnose` to inspect the data before preparing or training anything:

```bash
autodelphirf diagnose --archive my_archive.csv
```

This reports the archive size, location count, date ranges, reference-date and
report-date cadence, revision rate, suggested feature lags, recommended target
lag, and training-window length. It writes no files and does not require R.

Use the same column options that you would use for a full run:

```bash
autodelphirf diagnose \
  --archive my_archive.csv \
  --geo-col location_id \
  --reference-col event_date \
  --report-col issue_date \
  --value-col count
```

For machine-readable output:

```bash
autodelphirf diagnose --archive my_archive.csv --json
```

## Run the complete terminal pipeline

The following command performs pre-diagnosis, preprocessing, retrospective
training and testing, evaluation, post-forecast assessment, and report
generation:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --out autodelphirf_work
```

The default model selection is `baseline_null,delphirf`. Select additional
available models with `--methods`:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --methods baseline_null,delphirf,naive_delphirf,global_delphirf \
  --out autodelphirf_work
```

Run `autodelphirf run --help` to see the model identifiers and all available
options.

### Choose the retraining schedule

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --first-origin-date 2024-01-01 \
  --retrain-days 30 \
  --out autodelphirf_work
```

You can also provide an explicit start and end date:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --start-date 2024-01-01 \
  --end-date 2025-01-01 \
  --retrain-days 30 \
  --out autodelphirf_work
```

### Choose the target lag

AutoDelphiRF recommends an operational target lag from the archive. To provide one:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --target-lag 28 \
  --out autodelphirf_work
```

If the selected value is shorter than the diagnosed recommendation, add
`--confirm-target-lag` to explicitly retain it:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --target-lag 28 \
  --confirm-target-lag \
  --out autodelphirf_work
```

## Prepare first and run later

Build the prepared dataset and configuration without fitting models:

```bash
autodelphirf prepare \
  --archive my_archive.csv \
  --name mydata \
  --out autodelphirf_work
```

Then run the saved configuration:

```bash
autodelphirf run --config autodelphirf_work/mydata.json
```

## Rerun post-forecast assessment

A completed run freezes its effective configuration in
`results/resolved_config.json`. Rerun calibration, reliability, validation, and
report generation from the saved predictions without invoking R or fitting a
model:

```bash
autodelphirf assess --results autodelphirf_work/results
```

By default each assessment is written to a new UTC-stamped directory under
`results/assessments/`. Use `--out` to choose another new directory. An
assessment refuses to overwrite a nonempty directory and records SHA-256
hashes of its prediction/configuration inputs in `assessment_manifest.json`.

Validate the prepared input without model fitting:

```bash
autodelphirf run \
  --config autodelphirf_work/mydata.json \
  --diagnosis-only
```

## Use the web application

Start the local web application:

```bash
autodelphirf web
```

AutoDelphiRF opens `http://127.0.0.1:8765` in the default browser. The web
workflow lets you:

1. Select a revision archive.
2. Identify the date, location, and value columns.
3. Review the pre-diagnosis.
4. Confirm the target lag.
5. Choose models and retraining settings.
6. Start, stop, or resume the run.
7. Review the results and generated report.

Use another port or output directory if needed:

```bash
autodelphirf web --port 9000 --out ~/autodelphirf_work
```

Start the server without opening a browser automatically:

```bash
autodelphirf web --no-browser
```

The web application runs locally. Files selected in the page are handled by
the AutoDelphiRF process on the same computer.

## Output files

A complete run writes results under the selected working directory:

```text
autodelphirf_work/
├── mydata.json
├── mydata/
│   ├── preparation.json
│   ├── inventory.csv
│   ├── test_dates.csv
│   └── prepared data files
└── results/
    ├── report/
    │   ├── report.html
    │   ├── figures/
    │   └── tables/
    ├── predictions.csv.gz
    ├── predictions_wide.csv.gz
    ├── resolved_config.json
    ├── assessment_inputs.json
    ├── run_manifest.json
    ├── target_lag_resolution.json
    ├── runtime_by_method.csv
    └── runtime_by_origin.csv
```

The HTML report is normally located at:

```text
autodelphirf_work/results/report/report.html
```

Predictions and evaluation values are written on the original data scale.

## Generate a small example archive

```bash
python examples/make_example_archive.py --out example_archive.csv
autodelphirf diagnose --archive example_archive.csv
```

To run the full example:

```bash
autodelphirf run \
  --archive example_archive.csv \
  --name example \
  --out autodelphirf_work
```

## Environment variables

| Variable | Purpose |
|---|---|
| `AUTODELPHIRF_RSCRIPT` | Path to the `Rscript` executable to use. |
| `AUTODELPHIRF_CONFIG_DIR` | Directory containing optional user configuration files. |
| `AUTODELPHIRF_TARGET_LAG_PARAMS` | Path to an optional target-lag configuration file. |
| `AUTODELPHIRF_RAW_ARCHIVE_MANIFEST` | Path to an optional raw-archive manifest. |

## Development installation

```bash
git clone https://github.com/cmu-delphi/AutoDelphiRF.git
cd AutoDelphiRF
python -m pip install -e ".[dev]"
python -m pytest -m "not slow"
```

The slow tests require R and a compatible DelphiRF installation.

## License

AutoDelphiRF is distributed under the MIT License. See `LICENSE`.
