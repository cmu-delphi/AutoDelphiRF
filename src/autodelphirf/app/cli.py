"""The ``autodelphirf`` command line.

Seven subcommands, in the order a new user meets them:

  ``autodelphirf diagnose``  read a raw archive and print what AutoDelphiRF would do
                         with it. Pure Python, no R, nothing written.
  ``autodelphirf prepare``   build the prepared triangle, rolling schedule, and
                         dataset JSON from a raw archive, by calling DelphiRF.
  ``autodelphirf run``       train, test, evaluate, and produce the final report.
                         Given ``--archive`` it prepares first; given ``--config``
                         it reruns a dataset that ``autodelphirf prepare`` built.
  ``autodelphirf update``    ingest a later release of the same raw archive,
                         reuse the frozen preparation contract and saved models,
                         and append newly issued forecasts.
  ``autodelphirf assess``    rerun post-forecast assessment from saved predictions;
                         never fits a model.
  ``autodelphirf init``      write a starter raw-archive configuration JSON.
  ``autodelphirf web``       do all of the above in a browser: drag a file in and
                         answer the questions on a page.

``autodelphirf run --archive`` is the one-line path:

    autodelphirf run --archive my_archive.csv --name mydata --out work/

which diagnoses the archive, preprocesses it through DelphiRF, runs rolling
training and testing, and writes ``work/results/report/report.html``.

``autodelphirf web`` is the same workflow with the questions asked on a page.
For target lag, diagnosis supplies an evidence range and waits for the user's
explicit choice.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import pandas as pd

from .. import __version__
from ..core.config import load_dataset_config
from ..core.io import describe_skipped_locations, read_skipped_locations
from ..pre_diagnosis.preparation import (DEFAULT_MIN_LOCATION_ROWS, DEFAULT_RETRAIN_DAYS,
                                         PreparationError, build_spec, diagnose_archive,
                                         iso_weekday_groups, prepare_dataset)
from ..core.registry import DEFAULT_LAYERS, PREDICTION_LAYERS
from ..core.resources import resource_path




def _layers(value: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated ``--layers`` value."""
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("--layers needs at least one layer name")
    unknown = [name for name in names if name not in PREDICTION_LAYERS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown prediction layer(s) {unknown}. Available: {sorted(PREDICTION_LAYERS)}. "
            "A comparator imported from a file is named in the dataset JSON instead.")
    return names


def _load_raw_config(arguments: argparse.Namespace) -> argparse.Namespace:
    """Populate raw-archive arguments from a self-contained JSON config."""
    path = Path(arguments.raw_config).resolve()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("raw dataset config must contain a JSON object")
    allowed = {"name", "archive", "work_dir", "columns", "value_type",
               "preprocessing", "retraining", "model_methods",
               "genuine_training", "genuine_testing"}
    unknown = {key for key in raw if not key.startswith("_")}.difference(allowed)
    if unknown:
        raise ValueError(f"unknown raw dataset config fields: {sorted(unknown)}")
    missing = {key for key in ("name", "archive") if not raw.get(key)}
    if missing:
        raise ValueError(f"raw dataset config missing: {sorted(missing)}")

    def resolve(value):
        target = Path(value)
        return target if target.is_absolute() else (path.parent / target).resolve()

    columns = raw.get("columns") or {}
    if not isinstance(columns, dict):
        raise ValueError("raw dataset config 'columns' must be an object")
    allowed_columns = {"reference_date", "report_date", "geo_value", "values"}
    unknown_columns = set(columns).difference(allowed_columns)
    if unknown_columns:
        raise ValueError(f"unknown raw column fields: {sorted(unknown_columns)}")
    values = columns.get("values", ["value"])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values or not all(isinstance(x, str) for x in values):
        raise ValueError("raw dataset config columns.values must be a nonempty string list")

    preprocessing = raw.get("preprocessing") or {}
    retraining = raw.get("retraining") or {}
    if not isinstance(preprocessing, dict) or not isinstance(retraining, dict):
        raise ValueError("raw dataset config preprocessing and retraining must be objects")
    preprocessing_fields = {
        "target_lag", "confirm_target_lag", "temporal_resolution", "lag_terms",
        "training_days", "smoothed", "smoothed_target", "weekday_groups",
        "min_location_rows", "triangle_format"}
    retraining_fields = {"first_origin_date", "retrain_days", "start_date", "end_date"}
    if set(preprocessing).difference(preprocessing_fields):
        raise ValueError(f"unknown preprocessing fields: "
                         f"{sorted(set(preprocessing).difference(preprocessing_fields))}")
    if set(retraining).difference(retraining_fields):
        raise ValueError(f"unknown retraining fields: "
                         f"{sorted(set(retraining).difference(retraining_fields))}")

    geo = columns.get("geo_value", "geo_value")
    methods = raw.get("model_methods", list(DEFAULT_LAYERS))
    if not isinstance(methods, list) or not all(isinstance(x, str) for x in methods):
        raise ValueError("raw dataset config model_methods must be a string list")
    arguments.archive = resolve(raw["archive"])
    arguments.name = str(raw["name"])
    arguments.out = resolve(raw.get("work_dir", "autodelphirf_work"))
    arguments.reference_col = columns.get("reference_date", "reference_date")
    arguments.report_col = columns.get("report_date", "report_date")
    arguments.geo_col = geo or "geo_value"
    arguments.no_geo = geo is None
    arguments.value_cols = values
    arguments.value_type = raw.get("value_type", "count")
    arguments.target_lag = preprocessing.get("target_lag")
    arguments.confirm_target_lag = bool(preprocessing.get("confirm_target_lag", False))
    arguments.temporal_resolution = preprocessing.get("temporal_resolution")
    arguments.lag_terms = preprocessing.get("lag_terms")
    arguments.training_days = preprocessing.get("training_days")
    arguments.smoothed = preprocessing.get("smoothed")
    arguments.smoothed_target = preprocessing.get("smoothed_target")
    # ISO weekday numbers per group (1 = Monday ... 7 = Sunday); null or absent
    # means the groups are detected from the archive's report dates.
    arguments.weekday_groups = preprocessing.get("weekday_groups")
    arguments.min_location_rows = preprocessing.get("min_location_rows", DEFAULT_MIN_LOCATION_ROWS)
    arguments.triangle_format = preprocessing.get("triangle_format", "parquet")
    arguments.first_origin_date = retraining.get("first_origin_date")
    arguments.testing_days = retraining.get("retrain_days")
    arguments.start_date = retraining.get("start_date")
    arguments.end_date = retraining.get("end_date")
    arguments.layers = _layers(",".join(methods))
    for flag in ("genuine_training", "genuine_testing"):
        value = raw.get(flag, True)
        if not isinstance(value, bool):
            raise ValueError(f"raw dataset config {flag} must be true or false")
        setattr(arguments, flag, value)
    return arguments


def preparation_options(include_archive: bool = True) -> argparse.ArgumentParser:
    """The options that describe a raw archive and how to preprocess it.

    Returned as a parent parser so ``diagnose``, ``prepare`` and ``run`` share
    one definition. ``--archive``/``--name`` are never marked required here:
    ``run`` accepts ``--config`` instead, so the requirement is per-subcommand
    and checked in :func:`_require`.
    """
    parser = argparse.ArgumentParser(add_help=False)
    source = parser.add_argument_group(
        "input archive", "One row per (location, reference date, report date, value).")
    if include_archive:
        raw_source = source.add_mutually_exclusive_group()
        raw_source.add_argument("--archive", type=Path, metavar="CSV",
                                help="raw revision archive: .csv, .csv.gz, or .parquet")
        raw_source.add_argument("--raw-config", type=Path, metavar="JSON",
                                help="complete raw-archive configuration JSON")
    source.add_argument("--name", help="dataset name, used for output paths")
    source.add_argument("--out", type=Path, default=Path("autodelphirf_work"), metavar="DIR",
                        help="working directory for the triangle, config, and results "
                             "(default: ./autodelphirf_work)")
    source.add_argument("--reference-col", default="reference_date",
                        help="column holding the event/reference date (default: reference_date)")
    source.add_argument("--report-col", default="report_date",
                        help="column holding the report/issue date (default: report_date)")
    source.add_argument("--value-col", action="append", dest="value_cols", metavar="COL",
                        help="value column; give it twice for a fraction signal, numerator "
                             "then denominator (default: value)")
    source.add_argument("--geo-col", default="geo_value",
                        help="location column; pass --no-geo for a single-location archive")
    source.add_argument("--no-geo", action="store_true",
                        help="the archive covers one location and has no location column")
    source.add_argument("--value-type", choices=("count", "fraction"), default="count",
                        help="count models the value; fraction models numerator/denominator")

    diagnosed = parser.add_argument_group(
        "preprocessing overrides",
        "Each of these is diagnosed from the archive when omitted. Anything set here is "
        "recorded as a user override in preparation.json.")
    diagnosed.add_argument("--target-lag", type=int, metavar="L",
                           help="explicit target lag in days; always used as supplied")
    diagnosed.add_argument("--confirm-target-lag", action="store_true",
                           help="deprecated compatibility flag; an explicit --target-lag already wins")
    diagnosed.add_argument("--temporal-resolution", choices=("daily", "weekly"),
                           help="reporting cadence")
    diagnosed.add_argument("--lag-terms", metavar="N,N",
                           help="comma-separated reference-axis feature lags, e.g. 7,14")
    diagnosed.add_argument("--training-days", type=int,
                           help="rolling training window length in days")
    diagnosed.add_argument("--smoothed", dest="smoothed", action="store_true", default=None,
                           help="the archive values are already smoothed: DelphiRF does no "
                                "further averaging and the target is the archive value")
    diagnosed.add_argument("--smoothed-target", dest="smoothed_target", action="store_true",
                           default=None,
                           help="predict the 7-day-average target at lag L (for a fraction, the "
                                "7-day mean numerator over the 7-day mean denominator) from "
                                "the matching 7-day-average provisional value")
    diagnosed.add_argument("--no-smoothed", dest="smoothed", action="store_false",
                           help="(default) predict the unsmoothed target value; a daily "
                                "stream's features use 7-day trailing averages")

    calendar = parser.add_argument_group(
        "retraining calendar",
        "Every test origin is a retraining origin: the models are refitted on the history "
        "visible at that date. Omitted, origins run from 60 days after the archive's first "
        "report date to its last, one every --retrain-days.")
    calendar.add_argument("--first-origin-date", metavar="YYYY-MM-DD",
                          help="first retraining origin (default: 60 days into the archive)")
    calendar.add_argument("--retrain-days", type=int, dest="testing_days", metavar="N",
                          help="days between retraining origins (default: diagnosed -- "
                               f"{DEFAULT_RETRAIN_DAYS['weekly']} for a weekly stream, "
                               f"{DEFAULT_RETRAIN_DAYS['daily']} for a daily one)")
    calendar.add_argument("--start-date", help="first test origin, YYYY-MM-DD (with --end-date, "
                                               "overrides the rolling calendar)")
    calendar.add_argument("--end-date", help="last test origin, YYYY-MM-DD")

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument("--min-location-rows", type=int, default=DEFAULT_MIN_LOCATION_ROWS,
                          metavar="N",
                          help=f"skip locations with fewer archive rows "
                               f"(default: {DEFAULT_MIN_LOCATION_ROWS})")
    advanced.add_argument("--triangle-format", choices=("parquet", "csv"), default="parquet",
                          help="prepared-triangle file format (default: parquet)")
    return parser


def _require(arguments: argparse.Namespace, *names: str) -> None:
    """Fail the way argparse would for options this subcommand needs."""
    missing = [f"--{name.replace('_', '-')}" for name in names
               if getattr(arguments, name, None) is None]
    if missing:
        raise PreparationError(f"the following arguments are required: {', '.join(missing)}")


def _spec_from_arguments(arguments: argparse.Namespace):
    """Turn parsed arguments into a preparation specification."""
    _require(arguments, "archive", "name")
    value_cols = tuple(arguments.value_cols or ("value",))
    lag_terms = None
    if arguments.lag_terms:
        try:
            parts = (arguments.lag_terms.split(",") if isinstance(arguments.lag_terms, str)
                     else arguments.lag_terms)
            lag_terms = tuple(int(part) for part in parts)
        except ValueError as error:
            raise PreparationError(
                f"--lag-terms must be comma-separated integers, got {arguments.lag_terms!r}"
            ) from error
    return build_spec(
        arguments.name, arguments.archive, Path(arguments.out) / arguments.name,
        reference_col=arguments.reference_col, report_col=arguments.report_col,
        value_cols=value_cols, geo_col=None if arguments.no_geo else arguments.geo_col,
        value_type=arguments.value_type, smoothed=arguments.smoothed,
        smoothed_target=getattr(arguments, "smoothed_target", None),
        target_lag=arguments.target_lag, confirm_target_lag=arguments.confirm_target_lag,
        temporal_resol=arguments.temporal_resolution,
        lag_terms=lag_terms, training_days=arguments.training_days,
        testing_days=arguments.testing_days, first_origin_date=arguments.first_origin_date,
        start_date=arguments.start_date,
        end_date=arguments.end_date, triangle_format=arguments.triangle_format,
        min_location_rows=arguments.min_location_rows,
        weekday_groups=getattr(arguments, "weekday_groups", None),
        as_of_date=getattr(arguments, "as_of_date", None),
        genuine_training=getattr(arguments, "genuine_training", True),
        genuine_testing=getattr(arguments, "genuine_testing", True))


def _report_diagnosis(diagnosis: dict) -> None:
    """Print the parts of a diagnosis a user acts on."""
    print(f"  rows                {diagnosis['n_rows']:,}")
    print(f"  locations           {diagnosis['n_locations']:,}")
    print(f"  reference dates     {diagnosis['reference_date_min']} .. "
          f"{diagnosis['reference_date_max']}")
    print(f"  cadence             {diagnosis['temporal_resolution']} "
          f"(reference axis {diagnosis['reference_axis_resolution']}, "
          f"report axis {diagnosis['report_axis_resolution']})")
    print(f"  feature lags        {diagnosis['reference_axis_feature_lags']}")
    bounds = diagnosis.get("target_lag_suggested_range") or {}
    if bounds.get("skipped"):
        print("  suggested L area    skipped (target lag specified)")
    else:
        print(f"  suggested L area    {bounds.get('lower')} .. {bounds.get('upper')} days "
              "(90% within 10% .. 1% relative error)")
    resolved = diagnosis.get("resolved_target_lag")
    print(f"  selected L          {resolved if resolved is not None else 'awaiting user decision'} "
          f"({diagnosis['target_lag_resolution']})")
    print(f"  training window     {diagnosis['training_window_days']} days")
    if diagnosis.get("target_lag_confirmation_prompt"):
        print(f"\n  CONFIRM: {diagnosis['target_lag_confirmation_prompt']}")
    for note in diagnosis.get("notes") or ():
        print(f"  note: {note}")


def command_diagnose(arguments: argparse.Namespace) -> int:
    if arguments.raw_config is not None:
        arguments = _load_raw_config(arguments)
    _require(arguments, "archive")
    value_cols = tuple(arguments.value_cols or ("value",))
    diagnosis = diagnose_archive(
        arguments.archive, geo_col=None if arguments.no_geo else arguments.geo_col,
        reference_col=arguments.reference_col, report_col=arguments.report_col,
        value_cols=value_cols, value_type=arguments.value_type,
        user_target_lag=arguments.target_lag)
    if arguments.json:
        print(json.dumps(diagnosis, indent=2, default=str))
        return 0
    print(f"Diagnosis of {arguments.archive}:\n")
    _report_diagnosis(diagnosis)
    print("\nNothing was written. Run `autodelphirf prepare` with the same options to build "
          "the prepared triangle.")
    return 0


def command_prepare(arguments: argparse.Namespace) -> int:
    if arguments.raw_config is not None:
        arguments = _load_raw_config(arguments)
    spec = _spec_from_arguments(arguments)
    print(f"Diagnosis of {spec.raw_csv}:\n")
    _report_diagnosis(spec.diagnosis)
    if spec.overrides:
        print(f"\n  user overrides      {', '.join(spec.overrides)}")
    print(f"\nBuilding prepared triangle in {spec.output_dir} (DelphiRF)\n")
    config_path = prepare_dataset(spec, prediction_layers=arguments.layers,
                                  config_path=Path(arguments.out) / f"{spec.name}.json")
    skipped = read_skipped_locations(spec.output_dir)
    if not skipped.empty:
        print(f"\n{len(skipped)} location(s) were skipped during preparation and will not be "
              "forecast or evaluated:")
        print("\n".join(describe_skipped_locations(skipped)))
    print(f"\nWrote dataset config: {config_path}")
    print(f"Run it with: autodelphirf run --config {config_path}")
    return 0


def command_run(arguments: argparse.Namespace) -> int:
    # Imported here, not at module scope: `autodelphirf diagnose` and `--help`
    # should not pay for the pipeline's scipy/matplotlib import chain.
    from ..workflow.run import run_pipeline

    if arguments.raw_config is not None:
        arguments = _load_raw_config(arguments)
    if arguments.archive is not None:
        spec = _spec_from_arguments(arguments)
        print(f"Diagnosis of {spec.raw_csv}:\n")
        _report_diagnosis(spec.diagnosis)
        if spec.overrides:
            print(f"\n  user overrides      {', '.join(spec.overrides)}")
        print(f"\nBuilding prepared triangle in {spec.output_dir} (DelphiRF)\n")
        config_file = prepare_dataset(spec, prediction_layers=arguments.layers,
                                      config_path=Path(arguments.out) / f"{spec.name}.json")
    else:
        config_file = arguments.config

    config = load_dataset_config(config_file)
    print(f"\nRunning {config.name}: layers {list(config.prediction_layers)}\n")
    summary = run_pipeline(config, diagnosis_only=arguments.diagnosis_only,
                           post_assessment=not arguments.no_post_assessment)
    if arguments.diagnosis_only:
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nInput diagnosis written to {config.output_dir / 'diagnostics'}")
        return 0
    if arguments.no_post_assessment:
        print(f"\nDone. Evaluated predictions: {config.output_dir / 'predictions.csv.gz'} "
              "(post-assessment skipped)")
        return 0
    print(f"\nDone. Report: {config.output_dir / 'report' / 'report.html'}")
    return 0


def _apply_frozen_update_settings(arguments: argparse.Namespace) -> argparse.Namespace:
    """Fill diagnosis-derived settings from the first successful preparation."""
    preparation = Path(arguments.out) / arguments.name / "preparation.json"
    contract = Path(arguments.out) / "results" / "analysis_contract.json"
    if not preparation.is_file() or not contract.is_file():
        raise FileNotFoundError(
            "returning-user update needs an earlier completed run in the configured work_dir")
    recorded = json.loads(preparation.read_text())
    resolved = recorded["resolved_preprocessing"]
    frozen = json.loads(contract.read_text())["settings"]
    arguments.target_lag = int(frozen["ref_lag"])
    arguments.confirm_target_lag = True
    arguments.temporal_resolution = frozen["temporal_resol"]
    arguments.lag_terms = tuple(frozen["lag_terms"])
    arguments.training_days = int(frozen["training_days"])
    arguments.smoothed = bool(frozen["smoothed"])
    arguments.smoothed_target = str(frozen.get("target_column", "")).endswith("_7dav")
    arguments.testing_days = int(frozen["testing_days"])
    arguments.first_origin_date = resolved["start_date"]
    arguments.start_date = None
    arguments.end_date = None
    # The frozen groups are reused exactly (in ISO numbers), keeping whether
    # they were originally detected or chosen by the user.
    arguments.weekday_groups = iso_weekday_groups(frozen.get("weekday_groups") or {})
    arguments.weekday_groups_source = resolved.get("weekday_groups_source")
    # Case selection is frozen too; a preparation that predates the record used
    # the defaults (both True).
    selection = recorded.get("case_selection", {})
    for flag in ("genuine_training", "genuine_testing"):
        frozen_value = bool(selection.get(flag, True))
        given = getattr(arguments, flag, frozen_value)
        if given != frozen_value:
            raise ValueError(f"{flag} is frozen at {frozen_value} by the first run; "
                             f"the update config says {given}")
        setattr(arguments, flag, frozen_value)
    return arguments


def command_update(arguments: argparse.Namespace) -> int:
    """Refresh preparation and forecast state from a later archive release."""
    from ..workflow.run import run_pipeline
    from ..workflow.operational import raw_archive_state

    arguments = _load_raw_config(arguments)
    prepared_dir = Path(arguments.out) / arguments.name
    contract = Path(arguments.out) / "results" / "analysis_contract.json"
    initialized = (prepared_dir / "preparation.json").is_file() and contract.is_file()
    if initialized:
        arguments = _apply_frozen_update_settings(arguments)
        preview = raw_archive_state(
            prepared_dir, Path(arguments.out) / "results", commit=False)
        if not preview["changed"] and arguments.as_of_date is None:
            print("The raw archive is unchanged; models and forecasts are already current.")
            return 0
    if arguments.as_of_date is not None:
        first_origin = arguments.first_origin_date
        if first_origin is None:
            raise PreparationError("--as-of-date requires a configured first_origin_date")
        if pd.Timestamp(arguments.as_of_date) < pd.Timestamp(first_origin):
            raise PreparationError("--as-of-date cannot precede the first retraining origin")
        arguments.start_date = first_origin
        arguments.end_date = arguments.as_of_date
        arguments.first_origin_date = None
    spec = _spec_from_arguments(arguments)
    if getattr(arguments, "weekday_groups_source", None):
        spec = replace(spec, weekday_groups_source=arguments.weekday_groups_source)
    config_file = prepare_dataset(
        spec, prediction_layers=arguments.layers,
        config_path=Path(arguments.out) / f"{spec.name}.json")
    config = load_dataset_config(config_file)
    action = "Updating" if initialized else "Initializing real-time deployment for"
    print(f"\n{action} {config.name} from {arguments.archive}\n")
    summary = run_pipeline(config, model_mode=arguments.model_mode,
                           incremental_update=True)
    print(f"\nUpdate complete. Forecast ledger: "
          f"{summary['forecast_ledger']['path']}")
    return 0


def command_assess(arguments: argparse.Namespace) -> int:
    from ..post_assessment.run import assess_results
    output = assess_results(arguments.results, arguments.out)
    print(f"Assessment complete. Report: {output / 'report' / 'report.html'}")
    return 0


def command_web(arguments: argparse.Namespace) -> int:
    # Imported here so `autodelphirf --help` does not pay for the server module.
    from .web import serve

    serve(host=arguments.host, port=arguments.port, work_dir=arguments.out,
          open_browser=not arguments.no_browser, verbose=arguments.verbose)
    return 0


def command_init(arguments: argparse.Namespace) -> int:
    destination = arguments.output or Path(f"{arguments.name}.json")
    if destination.exists() and not arguments.force:
        print(f"{destination} already exists; pass --force to overwrite.", file=sys.stderr)
        return 1
    template = json.loads(resource_path("raw_dataset_template.json").read_text())
    template["name"] = arguments.name
    template["archive"] = template["archive"].replace("your_dataset_name", arguments.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(template, indent=2) + "\n")
    print(f"Wrote {destination}. Fill in the raw archive and column mapping, then:\n"
          f"  autodelphirf diagnose --raw-config {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autodelphirf", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"autodelphirf {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser(
        "diagnose", parents=[preparation_options()],
        help="inspect a raw archive without writing anything",
        description="Read a raw revision archive and report what AutoDelphiRF would do with it: "
                    "cadence, genuine-revision rate, feature lags, and the recommended "
                    "target lag L. Pure Python -- R and DelphiRF are not needed.")
    diagnose.add_argument("--json", action="store_true", help="print the full report as JSON")
    diagnose.set_defaults(handler=command_diagnose, layers=DEFAULT_LAYERS)

    prepare = subparsers.add_parser(
        "prepare", parents=[preparation_options()],
        help="build a prepared triangle from a raw archive (calls DelphiRF)",
        description="Diagnose a raw archive and build the prepared triangle, rolling schedule, "
                    "and dataset JSON. Needs R and DelphiRF.")
    prepare.add_argument("--methods", "--layers", dest="layers", type=_layers, default=DEFAULT_LAYERS,
                         help="prediction layers to write into the dataset JSON "
                              f"(default: {','.join(DEFAULT_LAYERS)})")
    prepare.set_defaults(handler=command_prepare)

    # `run` owns --archive itself so it can sit in the mutually exclusive
    # group opposite --config; every other preparation option is inherited.
    run = subparsers.add_parser(
        "run", parents=[preparation_options(include_archive=False)],
        help="train, test, evaluate, and report; prepare the data first if needed",
        description="Run the rolling AutoDelphiRF pipeline. Pass --archive to go from a "
                    "raw archive to a report in one command, or --config to rerun a dataset that "
                    "autodelphirf prepare built.")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, metavar="JSON",
                        help="dataset JSON written by autodelphirf prepare (a triangle built "
                             "elsewhere is not supported)")
    source.add_argument("--archive", type=Path, metavar="CSV",
                        help="raw revision archive to prepare and then run")
    source.add_argument("--raw-config", type=Path, metavar="JSON",
                        help="complete raw-archive configuration JSON to prepare and run")
    run.add_argument("--methods", "--layers", dest="layers", type=_layers, default=DEFAULT_LAYERS,
                     help="prediction layers, when preparing from --archive "
                          f"(default: {','.join(DEFAULT_LAYERS)})")
    run.add_argument("--diagnosis-only", action="store_true",
                     help="validate the input and stop before fitting anything")
    run.add_argument("--no-post-assessment", action="store_true",
                     help="fit, predict and evaluate, then stop before calibration, "
                          "reliability, bootstrap and the report")
    run.set_defaults(handler=command_run)

    update = subparsers.add_parser(
        "update", parents=[preparation_options(include_archive=False)],
        help="process a later release using the first run's frozen settings and saved models",
        description="Run this for every new complete raw-archive release. Rebuild the prepared "
                    "triangle, keep the first run's preprocessing choices, reuse the deployed "
                    "models between retraining origins, fit a new model only when due, and "
                    "append newly issued forecasts to the persistent ledger.")
    update.add_argument("--raw-config", type=Path, required=True, metavar="JSON",
                        help="the same raw configuration used for the first run")
    update.add_argument("--as-of-date", metavar="YYYY-MM-DD",
                        help="calendar date of this operational update; use it for a scheduled "
                             "retraining date with no new raw release (ordinary release-driven "
                             "updates infer the current horizon from the archive)")
    update.set_defaults(model_mode="auto")
    update.set_defaults(handler=command_update, layers=DEFAULT_LAYERS)

    assess = subparsers.add_parser(
        "assess", help="rerun assessment and reporting without fitting models",
        description="Read frozen predictions from a completed result directory and write a "
                    "new post-forecast assessment. This never invokes R or refits a model.")
    assess.add_argument("--results", type=Path, required=True, metavar="DIR",
                        help="completed result directory containing predictions and resolved_config.json")
    assess.add_argument("--out", type=Path, metavar="DIR",
                        help="new assessment directory (default: RESULTS/assessments/<UTC timestamp>)")
    assess.set_defaults(handler=command_assess)

    web = subparsers.add_parser(
        "web", help="open the browser UI: drag a file in, answer the questions",
        description="Start a local web UI on this machine. Drop a raw archive onto the page, "
                    "answer the questions AutoDelphiRF cannot decide on its own (above all the "
                    "target lag), and watch the run. Nothing is uploaded anywhere: the "
                    "page talks only to this process.")
    web.add_argument("--port", type=int, default=8765,
                     help="port to listen on (default: 8765; 0 picks a free one)")
    web.add_argument("--host", default="127.0.0.1",
                     help="address to bind (default: 127.0.0.1). This server runs programs and "
                          "writes files on this machine -- only change this for a network you "
                          "control.")
    web.add_argument("--out", type=Path, default=Path("autodelphirf_work"), metavar="DIR",
                     help="working directory for uploads, triangles, and results "
                          "(default: ./autodelphirf_work)")
    web.add_argument("--no-browser", action="store_true",
                     help="do not open a browser window automatically")
    web.add_argument("--verbose", action="store_true", help="log every HTTP request")
    web.set_defaults(handler=command_web)

    initialize = subparsers.add_parser(
        "init", help="write a starter raw-archive configuration JSON",
        description="Write a raw-archive configuration template to fill in by hand. It can "
                    "drive diagnosis, preparation, and a complete run.")
    initialize.add_argument("--name", required=True, help="dataset name")
    initialize.add_argument("--kind", choices=("raw",), default="raw",
                            help="configuration to write; only raw-archive configurations are "
                                 "supported (default: raw)")
    initialize.add_argument("--output", type=Path, help="where to write it (default: <name>.json)")
    initialize.add_argument("--force", action="store_true", help="overwrite an existing file")
    initialize.set_defaults(handler=command_init)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status rather than raising.

    Every failure mode a user can reach by supplying the wrong data or the
    wrong option is reported as one line on stderr and status 2. An unexpected
    exception is deliberately NOT caught: a traceback from inside the pipeline
    is diagnostic information, and swallowing it would make a real bug look
    like a usage error.
    """
    parser = build_parser()
    supplied = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(supplied)
    try:
        if getattr(arguments, "raw_config", None) is not None:
            raw_options = {
                "--archive", "--name", "--out", "--reference-col", "--report-col",
                "--value-col", "--geo-col", "--no-geo", "--value-type", "--target-lag",
                "--confirm-target-lag", "--temporal-resolution", "--lag-terms",
                "--training-days", "--smoothed", "--smoothed-target", "--no-smoothed",
                "--first-origin-date",
                "--retrain-days", "--start-date", "--end-date",
                "--min-location-rows", "--triangle-format", "--methods", "--layers"}
            mixed = sorted({token.split("=", 1)[0] for token in supplied}.intersection(raw_options))
            if mixed:
                raise ValueError(
                    f"--raw-config is a complete input and cannot be combined with: {mixed}")
        return arguments.handler(arguments)
    except (PreparationError, FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
