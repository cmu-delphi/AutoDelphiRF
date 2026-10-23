"""Read and write shared AutoDelphiRF data files."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd


#: Per-location triangle file suffixes, in preference order. Parquet is what
#: DelphiRF's preprocessing writes; CSV is
#: accepted so a triangle can be inspected, hand-built, or loaded on a host
#: without a working Arrow install, which is otherwise a hard stop.
TRIANGLE_SUFFIXES = (".parquet", ".csv.gz", ".csv")

#: Files in a prepared directory that are never per-location triangles.
#: ``inventory.csv``/``test_dates.csv`` are written alongside the triangles by
#: the preprocessing step, and ``genuine_rows.csv.gz`` is a convenience export
#: of rows already present in them -- concatenating any of these would
#: duplicate or corrupt the triangle.
NON_TRIANGLE_STEMS = frozenset({"inventory", "test_dates", "genuine_rows", "raw_source",
                                "skipped_locations", "cleaned_releases"})


def _rscript_executable() -> str:
    """Use the same R installation selected for preparation and training."""
    executable = os.environ.get("AUTODELPHIRF_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        raise RuntimeError(
            "Rscript was not found on PATH. Install R or set AUTODELPHIRF_RSCRIPT "
            "to the Rscript executable used by AutoDelphiRF.")
    return executable


def read_csv(path, *, geo_col: str | None = "geo_value", **options) -> pd.DataFrame:
    """``pandas.read_csv`` that keeps the location column as text.

    A location is a label, never a number: "01" and "1" are different
    locations, and a numeric parse would silently merge or rename them.
    """
    if geo_col:
        options["dtype"] = {**(options.get("dtype") or {}), geo_col: str}
    return pd.read_csv(path, **options)


def triangle_files(prepared_dir: Path) -> list[Path]:
    """The per-location triangle files in ``prepared_dir``, one format only.

    Formats are not mixed: a directory holding both ``ak.parquet`` and a
    stale ``ak.csv`` would otherwise load that location twice, so the first
    suffix in :data:`TRIANGLE_SUFFIXES` that matches anything wins outright.
    """
    directory = Path(prepared_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"prepared_dir is not a directory: {directory}")
    for suffix in TRIANGLE_SUFFIXES:
        files = sorted(path for path in directory.glob(f"*{suffix}")
                       if path.name[: -len(suffix)] not in NON_TRIANGLE_STEMS)
        if files:
            return files
    return []


def read_skipped_locations(prepared_dir: Path) -> pd.DataFrame:
    """Locations preparation could not build a triangle for, with the reason."""
    path = Path(prepared_dir) / "skipped_locations.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["geo_value", "raw_rows", "reason"])
    return read_csv(path)


def describe_skipped_locations(skipped: pd.DataFrame) -> list[str]:
    """One line per skipped location, for terminal output."""
    return [f"  {row.geo_value}: {row.reason}" for row in skipped.itertuples(index=False)]


def load_prepared_triangles(prepared_dir: Path) -> pd.DataFrame:
    """Load full per-location triangles, with an R-arrow fallback on this host."""
    files = triangle_files(prepared_dir)
    if not files:
        raise FileNotFoundError(
            f"no prepared triangles found in {prepared_dir}. Expected one file per location "
            f"with a {' / '.join(TRIANGLE_SUFFIXES)} suffix, as written by "
            "`autodelphirf prepare`.")
    if files[0].suffix != ".parquet":
        return pd.concat([read_csv(path) for path in files], ignore_index=True)
    try:
        return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    except ImportError:
        with tempfile.NamedTemporaryFile(suffix=".csv.gz") as temporary:
            script = ("args<-commandArgs(trailingOnly=TRUE);f<-list.files(args[1],pattern='parquet$',"
                      "full.names=TRUE);data.table::fwrite(data.table::rbindlist(lapply(f,arrow::read_parquet)),args[2])")
            subprocess.run([_rscript_executable(), "-e", script,
                            str(prepared_dir), temporary.name], check=True)
            return read_csv(temporary.name)


def write_parquet_or_csv(frame: pd.DataFrame, destination: Path) -> Path:
    """Write a genuine Parquet artifact, with the same R-arrow fallback."""
    try:
        frame.to_parquet(destination, index=False)
        return destination
    except (ImportError, ValueError):
        csv = destination.with_suffix(".conversion.csv")
        frame.to_csv(csv, index=False)
        script = ("args<-commandArgs(trailingOnly=TRUE);"
                  "x<-data.table::fread(args[1]);arrow::write_parquet(x,args[2])")
        subprocess.run([_rscript_executable(), "-e", script,
                        str(csv), str(destination)], check=True)
        csv.unlink(missing_ok=True)
        return destination
