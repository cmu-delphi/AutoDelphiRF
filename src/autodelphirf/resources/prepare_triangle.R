#!/usr/bin/env Rscript

# Build an AutoDelphiRF prepared triangle from a raw revision archive by calling
# DelphiRF's data_preprocessing().
#
#   usage: prepare_triangle.R SPEC_JSON
#
# RevRoute does NOT reimplement DelphiRF's feature engineering: DelphiRF is a
# prerequisite (https://github.com/cmu-delphi/DelphiRF) and this script is the
# bridge to it. Every modelling decision below -- smoothing, lagged terms,
# target construction, weekday encoding -- is made inside data_preprocessing().
#
# Every input setting comes from SPEC_JSON, which autodelphirf/prepare.py
# writes from the user's column mapping and the archive diagnosis.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: prepare_triangle.R SPEC_JSON", call. = FALSE)
spec <- fromJSON(args[[1]], simplifyVector = TRUE)

# --- DelphiRF -------------------------------------------------------------
# AutoDelphiRF's native methods may use the explicitly recorded local checkout for
# preprocessing while an older installed DelphiRF is being updated. The path
# is preserved in preparation.json, so the source of the triangle is auditable.
if (!is.null(spec$delphirf_dir) && nzchar(spec$delphirf_dir)) {
  if (!requireNamespace("pkgload", quietly = TRUE)) {
    stop("loading the local DelphiRF checkout needs the R package 'pkgload'.", call. = FALSE)
  }
  suppressPackageStartupMessages(pkgload::load_all(spec$delphirf_dir, quiet = TRUE))
} else {
  if (!requireNamespace("DelphiRF", quietly = TRUE)) {
    stop("DelphiRF is not installed and no local source checkout was configured.", call. = FALSE)
  }
  suppressPackageStartupMessages(library(DelphiRF))
}

preprocess <- get("data_preprocessing")
required_arguments <- c("ref_lag", "lagged_term_list", "value_type", "temporal_resol",
                        "smoothed", "target_lag_lower_tolerance",
                        "target_lag_upper_tolerance", "onehot_weekdays")
missing_arguments <- setdiff(required_arguments, names(formals(preprocess)))
if (length(missing_arguments) > 0) {
  stop("the available DelphiRF::data_preprocessing() does not accept: ",
       paste(missing_arguments, collapse = ", "),
       ". This is an older DelphiRF than RevRoute requires; update it with ",
       "remotes::install_github('cmu-delphi/DelphiRF').", call. = FALSE)
}

normalize_weekly <- NULL
if (identical(spec$temporal_resol, "weekly")) {
  normalize_weekly <- tryCatch(get("normalize_weekly_observations"),
                               error = function(e) NULL)
  if (is.null(normalize_weekly)) {
    normalize_weekly <- tryCatch(
      if ("DelphiRF" %in% loadedNamespaces())
        getFromNamespace("normalize_weekly_observations", "DelphiRF") else NULL,
      error = function(e) NULL)
  }
  if (is.null(normalize_weekly)) {
    stop("a weekly archive needs DelphiRF's normalize_weekly_observations(), which this ",
         "DelphiRF install does not provide. Install a current DelphiRF release.",
         call. = FALSE)
  }
}

# --- Raw archive ----------------------------------------------------------
read_archive <- function(path) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(data.table::fread(path, showProgress = FALSE)))
  }
  utils::read.csv(path, stringsAsFactors = FALSE)
}

message("Reading ", spec$raw_csv)
raw_all <- read_archive(spec$raw_csv)

reference_col <- spec$reference_col
report_col <- spec$report_col
value_cols <- as.character(spec$value_cols)

missing_columns <- setdiff(c(reference_col, report_col, value_cols), names(raw_all))
if (length(missing_columns) > 0) {
  stop("raw archive is missing column(s): ", paste(missing_columns, collapse = ", "),
       ". Present: ", paste(utils::head(names(raw_all), 40), collapse = ", "), call. = FALSE)
}

raw_all[[reference_col]] <- as.Date(raw_all[[reference_col]])
raw_all[[report_col]] <- as.Date(raw_all[[report_col]])
raw_all$lag <- as.integer(raw_all[[report_col]] - raw_all[[reference_col]])

if (!is.null(spec$geo_col) && nzchar(spec$geo_col)) {
  raw_all$geo_value <- as.character(raw_all[[spec$geo_col]])
} else if (!("geo_value" %in% names(raw_all))) {
  raw_all$geo_value <- spec$constant_geo
}

raw_all <- raw_all[!is.na(raw_all[[reference_col]]) & !is.na(raw_all[[report_col]]), , drop = FALSE]
# A negative lag is a report filed before its own reference date: not a
# revision, so it is dropped rather than modelled. The upper bound keeps the
# follow-up needed for the target plus a margin, and nothing beyond it.
raw_all <- raw_all[raw_all$lag >= 0 & raw_all$lag <= spec$ref_lag + 100, , drop = FALSE]
if (nrow(raw_all) == 0) {
  stop("no rows left after dropping negative lags and lags beyond ref_lag + 100 (ref_lag=",
       spec$ref_lag, "). Check that reference/report columns are not swapped.", call. = FALSE)
}

# --- Experiment calendar --------------------------------------------------
# Either the caller fixed the calendar, or it is derived from the archive's own
# span so an unfamiliar dataset still produces a schedule without the user
# having to pick dates.
if (!is.null(spec$start_date) && nzchar(spec$start_date)) {
  test_dates <- seq(as.Date(spec$start_date), as.Date(spec$end_date), by = spec$testing_days)
  # The last fold is evaluated from its own origin up to experiment_end_date,
  # so when the caller states one (the rolling retraining calendar does) that
  # fold spans a full retraining interval like every other fold, instead of
  # whatever remainder the range happened to leave.
  experiment_end_date <- if (!is.null(spec$experiment_end_date) && nzchar(spec$experiment_end_date)) {
    as.Date(spec$experiment_end_date)
  } else {
    as.Date(spec$end_date) + 1
  }
} else {
  report_dates <- sort(unique(raw_all[[report_col]]))
  quantile_positions <- floor(c(0.60, 0.72, 0.84) * length(report_dates))
  quantile_positions <- unique(pmax(quantile_positions, 1))
  test_dates <- report_dates[quantile_positions]
  experiment_end_date <- max(test_dates) + 30
}
test_dates <- sort(unique(test_dates))
if (length(test_dates) == 0) {
  stop("the experiment calendar is empty: the archive has too few distinct report dates ",
       "to place a test origin. Supply --start-date/--end-date explicitly.", call. = FALSE)
}

# Keep a common target lag plus enough earlier reference dates for
# every training window; this shrinks the grid without dropping eligible rows.
min_reference <- min(test_dates) - spec$training_days - spec$ref_lag - spec$upper - 35
max_report <- max(test_dates) + spec$ref_lag
raw_all <- raw_all[raw_all[[reference_col]] >= min_reference & raw_all[[report_col]] <= max_report, ,
                   drop = FALSE]
if (nrow(raw_all) == 0) {
  stop("no rows fall inside the experiment window (reference >= ", min_reference,
       ", report <= ", max_report, "). The calendar and the archive do not overlap.",
       call. = FALSE)
}

# --- Per-location preprocessing ------------------------------------------
output_dir <- spec$output_dir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

weekdays_spec <- spec$onehot_weekdays
if (is.null(weekdays_spec)) weekdays_spec <- list()
weekdays_spec <- lapply(weekdays_spec, as.character)

write_triangle <- function(frame, stem) {
  if (identical(spec$triangle_format, "csv")) {
    path <- file.path(output_dir, paste0(stem, ".csv.gz"))
    utils::write.csv(frame, gzfile(path), row.names = FALSE)
    return(path)
  }
  path <- file.path(output_dir, paste0(stem, ".parquet"))
  arrow::write_parquet(frame, path, compression = "zstd")
  path
}
if (!identical(spec$triangle_format, "csv") && !requireNamespace("arrow", quietly = TRUE)) {
  stop("writing Parquet triangles needs the 'arrow' R package; install it, or pass ",
       "--triangle-format csv.", call. = FALSE)
}

locations <- sort(unique(raw_all$geo_value))
inventory <- list()
skipped <- list()
for (i in seq_along(locations)) {
  location <- locations[[i]]
  raw <- raw_all[raw_all$geo_value == location, , drop = FALSE]
  # Too few rows to build lagged features and a target from; recorded in the
  # inventory rather than silently dropped.
  if (nrow(raw) < spec$min_location_rows) {
    skipped[[length(skipped) + 1]] <- data.frame(
      geo_value = location, raw_rows = nrow(raw),
      reason = paste0("fewer than ", spec$min_location_rows, " archive rows"))
    next
  }
  raw <- raw[order(raw[[reference_col]], raw[[report_col]]), , drop = FALSE]
  if (identical(spec$temporal_resol, "weekly")) {
    raw <- normalize_weekly(raw, reference_col, "lag")
  }

  prepared <- tryCatch(
    preprocess(raw, value_col = value_cols, refd_col = reference_col, lag_col = "lag",
               ref_lag = spec$ref_lag, lagged_term_list = as.numeric(spec$lag_terms),
               value_type = spec$value_type, temporal_resol = spec$temporal_resol,
               smoothed = spec$smoothed,
               target_lag_lower_tolerance = spec$lower,
               target_lag_upper_tolerance = spec$upper,
               target_as_of_date = NULL, onehot_weekdays = weekdays_spec),
    error = function(e) {
      message("  ", location, ": data_preprocessing() failed -- ", conditionMessage(e))
      NULL
    })
  if (is.null(prepared) || nrow(prepared) == 0) {
    skipped[[length(skipped) + 1]] <- data.frame(
      geo_value = location, raw_rows = nrow(raw),
      reason = "data_preprocessing() produced no rows")
    next
  }
  prepared$geo_value <- location

  write_triangle(prepared, location)
  inventory[[length(inventory) + 1]] <- data.frame(
    dataset = spec$name, geo_value = location, raw_rows = nrow(raw),
    prepared_rows = nrow(prepared), genuine_rows = sum(prepared$genuine_event, na.rm = TRUE),
    min_report = min(prepared$report_date), max_report = max(prepared$report_date))
  if (i %% 5 == 0) message("Prepared ", i, " of ", length(locations), " locations")
}

if (length(inventory) == 0) {
  stop("no location produced a prepared triangle. ",
       if (length(skipped) > 0) paste0("All ", length(skipped),
                                       " location(s) were skipped; the most common reason was: ",
                                       skipped[[1]]$reason, ".") else "",
       call. = FALSE)
}

inventory <- do.call(rbind, inventory)
utils::write.csv(inventory, file.path(output_dir, "inventory.csv"), row.names = FALSE)
if (length(skipped) > 0) {
  utils::write.csv(do.call(rbind, skipped), file.path(output_dir, "skipped_locations.csv"),
                   row.names = FALSE)
}

utils::write.csv(
  data.frame(dataset = spec$name, test_date = as.Date(test_dates),
             target_lag = spec$ref_lag, training_days = spec$training_days,
             testing_days = spec$testing_days,
             experiment_end_date = as.Date(experiment_end_date),
             target_column = spec$target_column,
             target_lag_lower_tolerance = spec$lower,
             target_lag_upper_tolerance = spec$upper,
             temporal_resolution = spec$temporal_resol),
  file.path(output_dir, "test_dates.csv"), row.names = FALSE)
writeLines(normalizePath(spec$raw_csv), file.path(output_dir, "raw_source.txt"))

message("Finished ", spec$name, ": ", nrow(inventory), " location(s), ",
        sum(inventory$prepared_rows), " prepared rows")
