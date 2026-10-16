# AutoDelphiRF

AutoDelphiRF forecasts how recently reported values will be revised. It takes a
revision archive, diagnoses its revision behavior, runs retrospective forecasts,
and produces predictions and an evaluation report.

```text
revision archive → diagnosis → forecasting → evaluation report
```

The local web application is the easiest way to try AutoDelphiRF. A command-line
interface is also available for scripted analyses.

## Requirements

- Python 3.10 or newer
- R and `Rscript`
- The R package [DelphiRF](https://github.com/cmu-delphi/DelphiRF/tree/refactor-clean)
- The R package `quantreg`

Install the R dependencies:

```bash
Rscript -e 'install.packages(c("remotes", "quantreg")); remotes::install_github("cmu-delphi/DelphiRF@refactor-clean")'
```

Install AutoDelphiRF from GitHub:

```bash
python -m pip install "autodelphirf[parquet] @ git+https://github.com/cmu-delphi/AutoDelphiRF.git@dev"
```

Or install an editable copy for development:

```bash
git clone -b dev https://github.com/cmu-delphi/AutoDelphiRF.git
cd AutoDelphiRF
python -m pip install -e ".[parquet]"
```

## Start with the web application

Run:

```bash
autodelphirf web
```

AutoDelphiRF opens a local page at <http://127.0.0.1:8765>. The guided workflow
helps you:

1. Select a revision-archive CSV file.
2. Identify its date, location, and value columns.
3. Review the automatic diagnosis.
4. Confirm the target lag and forecasting settings.
5. Run the analysis.
6. Open the generated evaluation report.

The application and your data remain on your computer. To use another port or
working directory:

```bash
autodelphirf web --port 9000 --out ~/autodelphirf_work
```

Use `autodelphirf web --no-browser` to start the server without opening a browser
automatically.

## Input data

AutoDelphiRF reads CSV and gzipped CSV revision archives. Each row is one version
of a value as it was known on a particular report date:

| geo_value | reference_date | report_date | value |
|---|---|---|---:|
| ak | 2024-03-04 | 2024-03-06 | 118 |
| ak | 2024-03-04 | 2024-03-13 | 142 |
| ak | 2024-03-04 | 2024-03-20 | 147 |
| ak | 2024-03-05 | 2024-03-06 | 95 |

- `reference_date` is the date the measurement describes.
- `report_date` is the date that version was published.
- `value` is the value available on the report date.
- `geo_value` identifies the location or time series.

The web application can map differently named columns. AutoDelphiRF supports
counts, precomputed rates or percentages, and fractions supplied as numerator
and denominator columns. A file without a geography column can be explicitly
treated as a single-location dataset; it can pool information across lags but
not across locations.

## Basic command-line usage

For a CSV using the default column names above, run:

```bash
autodelphirf run \
  --archive my_archive.csv \
  --name mydata \
  --target-lag 28
```

Run `autodelphirf diagnose --archive my_archive.csv` first to inspect the
target-lag evidence. AutoDelphiRF reports the area from 90% of records being
within 10% relative error to 90% being within 1%; it does not choose the target
lag. Preparation and training wait until you supply `--target-lag`. Reporting
cadence, feature lags, training window, and retraining calendar are diagnosed
when not specified. Results are written to `autodelphirf_work` by default.

For custom column names:

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

To inspect an archive without preparing data or fitting models:

```bash
autodelphirf diagnose --archive my_archive.csv
```

The same raw archive, column mappings, preprocessing choices, retraining
calendar, and model methods can be stored in one JSON file. Create a starter
file and use it with any raw-data command:

```bash
autodelphirf init --kind raw --name mydata --output mydata-raw.json
autodelphirf diagnose --raw-config mydata-raw.json
autodelphirf prepare --raw-config mydata-raw.json
autodelphirf run --raw-config mydata-raw.json
```

Paths inside the raw configuration are resolved relative to the JSON file.
`autodelphirf init --kind prepared` remains available for the less common case
of registering an existing prepared triangle.

Diagnosis runs in Python, writes no files, and does not require R. Run
`autodelphirf run --help` or `autodelphirf diagnose --help` for all options.

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

Open `results/report/report.html` to review forecast accuracy, uncertainty,
calibration, pooling, and runtime. The complete row-level forecasts are in
`results/predictions.csv.gz`.

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
