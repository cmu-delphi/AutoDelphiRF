"""The ``autodelphirf`` command line.

Six subcommands, in the order a new user meets them:

  ``autodelphirf diagnose``  read a raw archive and print what AutoDelphiRF would do
                         with it. Pure Python, no R, nothing written.
  ``autodelphirf prepare``   build the prepared triangle, rolling schedule, and
                         dataset JSON from a raw archive, by calling DelphiRF.
  ``autodelphirf run``       train, test, evaluate, and produce the final report.
                         Given ``--archive`` it prepares first; given ``--config``
                         it runs an already-prepared dataset.
  ``autodelphirf assess``    rerun post-forecast assessment from saved predictions;
                         never fits a model.
  ``autodelphirf init``      write a starter dataset JSON to fill in by hand.
  ``autodelphirf web``       do all of the above in a browser: drag a file in and
                         answer the questions on a page.

``autodelphirf run --archive`` is the one-line path:

    autodelphirf run --archive my_archive.csv --name mydata --out work/

which diagnoses the archive, preprocesses it through DelphiRF, runs rolling
training and testing, and writes ``work/results/report/report.html``.

``autodelphirf web`` is the same workflow with the questions asked on a page
rather than through flags -- above all the target lag, where the
diagnosis makes a recommendation the user is meant to accept or override.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import __version__
from .config import load_dataset_config
from .prepare import (DEFAULT_MIN_LOCATION_ROWS, DEFAULT_RETRAIN_DAYS, PreparationError, build_spec,
                      diagnose_archive, prepare_dataset)
from .registry import DEFAULT_LAYERS, PREDICTION_LAYERS
from .resources import resource_path




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
        source.add_argument("--archive", type=Path, metavar="CSV",
                            help="raw revision archive: .csv, .csv.gz, or .parquet")
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
                           help="target lag in days; reconciled with the recommended one")
    diagnosed.add_argument("--confirm-target-lag", action="store_true",
                           help="use --target-lag even when the diagnosis disagrees. Without "
                                "this, a value shorter than the recommended target lag loses to "
                                "the diagnosis because its target value may be less stable.")
    diagnosed.add_argument("--temporal-resolution", choices=("daily", "weekly"),
                           help="reporting cadence")
    diagnosed.add_argument("--lag-terms", metavar="N,N",
                           help="comma-separated reference-axis feature lags, e.g. 7,14")
    diagnosed.add_argument("--training-days", type=int,
                           help="rolling training window length in days")
    diagnosed.add_argument("--smoothed", dest="smoothed", action="store_true", default=None,
                           help="model the 7-day trailing average")
    diagnosed.add_argument("--no-smoothed", dest="smoothed", action="store_false",
                           help="model the unsmoothed value")

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
    advanced.add_argument("--target-lag-lower-tolerance", type=int, default=0, metavar="N",
                          help="days before L at which a target may be read (default: 0)")
    advanced.add_argument("--target-lag-upper-tolerance", type=int, default=0, metavar="N",
                          help="days after L at which a target may be read (default: 0)")
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
    """Turn parsed arguments into a :class:`~autodelphirf.prepare.PreparationSpec`."""
    _require(arguments, "archive", "name")
    value_cols = tuple(arguments.value_cols or ("value",))
    lag_terms = None
    if arguments.lag_terms:
        try:
            lag_terms = tuple(int(part) for part in arguments.lag_terms.split(","))
        except ValueError as error:
            raise PreparationError(
                f"--lag-terms must be comma-separated integers, got {arguments.lag_terms!r}"
            ) from error
    return build_spec(
        arguments.name, arguments.archive, Path(arguments.out) / arguments.name,
        reference_col=arguments.reference_col, report_col=arguments.report_col,
        value_cols=value_cols, geo_col=None if arguments.no_geo else arguments.geo_col,
        value_type=arguments.value_type, smoothed=arguments.smoothed,
        target_lag=arguments.target_lag, confirm_target_lag=arguments.confirm_target_lag,
        temporal_resol=arguments.temporal_resolution,
        lag_terms=lag_terms, training_days=arguments.training_days,
        testing_days=arguments.testing_days, first_origin_date=arguments.first_origin_date,
        lower=arguments.target_lag_lower_tolerance,
        upper=arguments.target_lag_upper_tolerance, start_date=arguments.start_date,
        end_date=arguments.end_date, triangle_format=arguments.triangle_format,
        min_location_rows=arguments.min_location_rows)


def _report_diagnosis(diagnosis: dict) -> None:
    """Print the parts of a diagnosis a user acts on."""
    print(f"  rows                {diagnosis['n_rows']:,}")
    print(f"  locations           {diagnosis['n_locations']:,}")
    print(f"  reference dates     {diagnosis['reference_date_min']} .. "
          f"{diagnosis['reference_date_max']}")
    print(f"  cadence             {diagnosis['temporal_resolution']} "
          f"(reference axis {diagnosis['reference_axis_resolution']}, "
          f"report axis {diagnosis['report_axis_resolution']})")
    print(f"  genuine revisions   {diagnosis['genuine_event_rate']:.1%} of archived rows")
    print(f"  feature lags        {diagnosis['reference_axis_feature_lags']}")
    print(f"  suggested L         {diagnosis['recommended_target_lag']} days")
    print(f"  resolved L          {diagnosis['resolved_target_lag']} days "
          f"({diagnosis['target_lag_resolution']})")
    print(f"  training window     {diagnosis['training_window_days']} days")
    if diagnosis.get("target_lag_confirmation_prompt"):
        print(f"\n  CONFIRM: {diagnosis['target_lag_confirmation_prompt']}")
    for note in diagnosis.get("notes") or ():
        print(f"  note: {note}")


def command_diagnose(arguments: argparse.Namespace) -> int:
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
    spec = _spec_from_arguments(arguments)
    print(f"Diagnosis of {spec.raw_csv}:\n")
    _report_diagnosis(spec.diagnosis)
    if spec.overrides:
        print(f"\n  user overrides      {', '.join(spec.overrides)}")
    print(f"\nBuilding prepared triangle in {spec.output_dir} (DelphiRF)\n")
    config_path = prepare_dataset(spec, prediction_layers=arguments.layers,
                                  config_path=Path(arguments.out) / f"{spec.name}.json")
    print(f"\nWrote dataset config: {config_path}")
    print(f"Run it with: autodelphirf run --config {config_path}")
    return 0


def command_run(arguments: argparse.Namespace) -> int:
    # Imported here, not at module scope: `autodelphirf diagnose` and `--help`
    # should not pay for pipeline's sklearn/matplotlib import chain.
    from .pipeline import run_pipeline

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
    summary = run_pipeline(config, diagnosis_only=arguments.diagnosis_only)
    if arguments.diagnosis_only:
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nInput diagnosis written to {config.output_dir / 'diagnostics'}")
        return 0
    print(f"\nDone. Report: {config.output_dir / 'report' / 'report.html'}")
    return 0


def command_assess(arguments: argparse.Namespace) -> int:
    from .assessment import assess_results
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
    template = json.loads(resource_path("dataset_template.json").read_text())
    template["name"] = arguments.name
    for key in ("prepared_dir", "output_dir"):
        template[key] = template[key].replace("your_dataset_name", arguments.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(template, indent=2) + "\n")
    print(f"Wrote {destination}. Fill in prepared_dir/output_dir, then:\n"
          f"  autodelphirf run --config {destination} --diagnosis-only")
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
                    "raw archive to a report in one command, or --config to run a dataset that "
                    "is already prepared.")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, metavar="JSON",
                        help="dataset config of an already-prepared dataset")
    source.add_argument("--archive", type=Path, metavar="CSV",
                        help="raw revision archive to prepare and then run")
    run.add_argument("--methods", "--layers", dest="layers", type=_layers, default=DEFAULT_LAYERS,
                     help="prediction layers, when preparing from --archive "
                          f"(default: {','.join(DEFAULT_LAYERS)})")
    run.add_argument("--diagnosis-only", action="store_true",
                     help="validate the input and stop before fitting anything")
    run.set_defaults(handler=command_run)

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
        "init", help="write a starter dataset JSON",
        description="Write a dataset JSON template to fill in by hand, for a prepared triangle "
                    "you built yourself.")
    initialize.add_argument("--name", required=True, help="dataset name")
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
    arguments = parser.parse_args(argv)
    try:
        return arguments.handler(arguments)
    except (PreparationError, FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
