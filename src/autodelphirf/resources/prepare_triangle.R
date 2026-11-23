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
# Always the DelphiRF installed in R's library, never a source checkout, so the
# triangle is built by the same package version that trains the models.
if (!requireNamespace("DelphiRF", quietly = TRUE)) {
  stop("DelphiRF is not installed; install it as the AutoDelphiRF README describes, e.g. ",
       "remotes::install_github('cmu-delphi/DelphiRF@refactor-clean').", call. = FALSE)
}
suppressPackageStartupMessages(library(DelphiRF))

preprocess <- DelphiRF::data_preprocessing
# revision_start_lag: revision magnitudes are measured from the initial report
# (l_min, Ch. 2), not from the lag-0 row that grid completion fills in.
required_arguments <- c("ref_lag", "lagged_term_list", "value_type", "temporal_resol",
                        "smoothed", "revision_start_lag")
missing_arguments <- setdiff(required_arguments, names(formals(preprocess)))
if (length(missing_arguments) > 0) {
  stop("the available DelphiRF::data_preprocessing() does not accept: ",
       paste(missing_arguments, collapse = ", "),
       ". This is an older DelphiRF than AutoDelphiRF requires; reinstall it as the ",
       "AutoDelphiRF README describes.", call. = FALSE)
}

normalize_weekly <- NULL
if (identical(spec$temporal_resol, "weekly")) {
  normalize_weekly <- tryCatch(getFromNamespace("normalize_weekly_observations", "DelphiRF"),
                               error = function(e) NULL)
  if (is.null(normalize_weekly)) {
    stop("a weekly archive needs DelphiRF's normalize_weekly_observations(), which this ",
         "DelphiRF install does not provide. Install a current DelphiRF release.",
         call. = FALSE)
  }
}

# --- Raw archive ----------------------------------------------------------
# A location is a label, never a number: "01" and "1" are different locations,
# so the geography column is always read as text.
read_archive <- function(path, geo_col = NULL) {
  text <- if (!is.null(geo_col) && nzchar(geo_col)) setNames(list(geo_col), "character") else NULL
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(data.table::fread(path, showProgress = FALSE, colClasses = text)))
  }
  utils::read.csv(path, stringsAsFactors = FALSE,
                  colClasses = if (is.null(text)) NA else setNames("character", geo_col))
}

message("Reading ", spec$raw_csv)
raw_all <- read_archive(spec$raw_csv, spec$geo_col)

reference_col <- spec$reference_col
report_col <- spec$report_col
value_cols <- as.character(spec$value_cols)

missing_columns <- setdiff(c(reference_col, report_col, value_cols), names(raw_all))
if (length(missing_columns) > 0) {
  stop("raw archive is missing column(s): ", paste(missing_columns, collapse = ", "),
       ". Present: ", paste(utils::head(names(raw_all), 40), collapse = ", "), call. = FALSE)
}

# The shared raw-archive contract (resources/raw_archive_contract.json). This
# script implements version 2 and refuses any other.
SUPPORTED_RAW_CONTRACT_VERSION <- 2L
if (is.null(spec$raw_contract_version) ||
    as.integer(spec$raw_contract_version) != SUPPORTED_RAW_CONTRACT_VERSION) {
  stop("unsupported raw-archive contract version ", format(spec$raw_contract_version),
       "; this preparation script implements version ", SUPPORTED_RAW_CONTRACT_VERSION,
       call. = FALSE)
}

# A date is the calendar date written at the start of the value: 2024-12-06
# and 2024-12-06T00:00:00Z are the same day, and a time of day or UTC offset
# is ignored, never converted. Anything else is invalid (removed below), not
# an error. Python's calendar_dates() applies the same rule.
calendar_dates <- function(values) {
  text <- trimws(as.character(values))
  text[!grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}($|[T ])", text)] <- NA_character_
  as.Date(substr(text, 1L, 10L), format = "%Y-%m-%d")
}
raw_all[[reference_col]] <- calendar_dates(raw_all[[reference_col]])
raw_all[[report_col]] <- calendar_dates(raw_all[[report_col]])
raw_all$lag <- as.integer(raw_all[[report_col]] - raw_all[[reference_col]])


# Cleaning, in the order the Python diagnosis applies the same rules
# (pre_diagnosis/archive.py clean_raw_archive), with this stage's scope: the
# as-of date and the feature horizon (target lag + 100 days, at most two
# years). Every removal is counted in raw_cleaning_audit.json.
audit <- list(scope = "feature_preparation",
              contract_version = as.integer(spec$raw_contract_version),
              input_rows = nrow(raw_all))
keep_rows <- function(frame, mask, name) {
  mask[is.na(mask)] <- FALSE
  audit[[name]] <<- sum(!mask)
  frame[mask, , drop = FALSE]
}

if (!is.null(spec$geo_col) && nzchar(spec$geo_col)) {
  raw_all$geo_value <- as.character(raw_all[[spec$geo_col]])
  # A declared geography column must name a location on every row: missing or
  # blank labels are invalid and dropped, as in the Python diagnosis.
  raw_all <- keep_rows(raw_all, !(is.na(raw_all$geo_value) | trimws(raw_all$geo_value) == ""),
                       "missing_location_rows")
  if (nrow(raw_all) == 0) {
    stop("the geography column '", spec$geo_col, "' has no valid location on any row; ",
         "check --geo-col, or use --no-geo for a single-location archive", call. = FALSE)
  }
} else {
  # --no-geo is an explicit declaration that this is one series. Force the
  # synthetic label even if an unrelated column happens to be named
  # geo_value, so the choice can never be silently reinterpreted.
  raw_all$geo_value <- spec$constant_geo
  audit$missing_location_rows <- 0L
}

raw_all <- keep_rows(raw_all, !is.na(raw_all[[reference_col]]) & !is.na(raw_all[[report_col]]),
                     "invalid_date_rows")

# The archive's as-of date is the last report date anyone could have seen. An
# explicit as-of date replays an earlier state of the archive, so nothing
# reported after it may enter the triangle.
explicit_as_of <- !is.null(spec$as_of_date) && nzchar(spec$as_of_date)
as_of_date <- if (explicit_as_of) as.Date(spec$as_of_date) else max(raw_all[[report_col]])
audit$effective_as_of_date <- format(as_of_date)
audit$as_of_source <- if (explicit_as_of) "explicit" else "inferred"
raw_all <- keep_rows(raw_all, raw_all[[report_col]] <= as_of_date, "future_report_rows")
audit$max_visible_report_date <- format(max(raw_all[[report_col]]))

# A row without a usable value is not an observation. For a numerator and
# denominator pair the value is their ratio, undefined unless both are present
# and the denominator is positive. Value columns become numeric here.
for (column in value_cols) raw_all[[column]] <- suppressWarnings(as.numeric(raw_all[[column]]))
observed_value <- if (length(value_cols) == 2L) {
  !is.na(raw_all[[value_cols[[1]]]]) & !is.na(raw_all[[value_cols[[2]]]]) &
    raw_all[[value_cols[[2]]]] > 0
} else {
  !is.na(raw_all[[value_cols[[1]]]])
}
raw_all <- keep_rows(raw_all, observed_value, "unusable_value_rows")

# A negative lag is a report filed before its own reference date: not a
# revision. The upper bound keeps the follow-up needed for the target plus a
# margin, without exceeding the two-year horizon of the diagnosis.
effective_max_lag <- min(spec$ref_lag + 100, spec$max_observed_lag_days)
audit$lag_upper_bound <- as.integer(effective_max_lag)
raw_all <- keep_rows(raw_all, raw_all$lag >= 0, "negative_lag_rows")
raw_all <- keep_rows(raw_all, raw_all$lag <= effective_max_lag, "over_lag_limit_rows")

# Several rows for one (location, reference date, report date) are one release
# reported more than once. Keep the last in file order, the same rule as the
# Python diagnosis (drop_duplicates(keep="last")), so DelphiRF sees one value.
raw_all <- keep_rows(raw_all, !duplicated(raw_all[, c("geo_value", reference_col, report_col)],
                                          fromLast = TRUE),
                     "duplicate_rows_removed")
audit$retained_rows <- nrow(raw_all)
audit$retained_locations <- length(unique(raw_all$geo_value))
dir.create(spec$output_dir, recursive = TRUE, showWarnings = FALSE)
audit_fields <- c("scope", "contract_version", "effective_as_of_date", "as_of_source",
                  "max_visible_report_date", "lag_upper_bound", "input_rows",
                  "missing_location_rows", "invalid_date_rows", "future_report_rows",
                  "unusable_value_rows", "negative_lag_rows", "over_lag_limit_rows",
                  "duplicate_rows_removed", "retained_rows", "retained_locations")
write_audit <- function() {
  jsonlite::write_json(audit[intersect(c(audit_fields, "experiment_window",
                                         "outside_experiment_window_rows", "min_location_rows",
                                         "skipped_location_rows"), names(audit))],
                       file.path(spec$output_dir, "raw_cleaning_audit.json"),
                       auto_unbox = TRUE, pretty = TRUE)
}
write_audit()
removed <- unlist(audit[grepl("_rows(_removed)?$", names(audit)) &
                         !names(audit) %in% c("input_rows", "retained_rows")])
if (any(removed > 0)) {
  message("Raw-archive cleaning removed: ",
          paste(sprintf("%s=%d", names(removed)[removed > 0], removed[removed > 0]),
                collapse = ", "))
}
if (isTRUE(spec$write_cleaned_releases)) {
  utils::write.csv(raw_all[, c("geo_value", reference_col, report_col, value_cols)],
                   file.path(spec$output_dir, "cleaned_releases.csv"), row.names = FALSE)
}
if (nrow(raw_all) == 0) {
  stop("no rows left after keeping lags from 0 through ", effective_max_lag,
       " days (target lag=", spec$ref_lag, ", archive lag cap=",
       spec$max_observed_lag_days,
       "). Check that reference/report columns are not swapped.", call. = FALSE)
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
# Every release up to the as-of date is kept for the forecast windows, and the
# target of the last window's cases needs releases through its end plus L.
min_reference <- min(test_dates) - spec$training_days - spec$ref_lag - 35
max_report <- min(as_of_date, as.Date(experiment_end_date) + spec$ref_lag)
# Stage-specific scope beyond the shared contract: the experiment window.
in_window <- raw_all[[reference_col]] >= min_reference & raw_all[[report_col]] <= max_report
audit$experiment_window <- list(min_reference_date = format(min_reference),
                                max_report_date = format(as.Date(max_report)))
audit$outside_experiment_window_rows <- sum(!in_window)
raw_all <- raw_all[in_window, ,
                   drop = FALSE]
if (nrow(raw_all) == 0) {
  stop("no rows fall inside the experiment window (reference >= ", min_reference,
       ", report <= ", max_report, "). The calendar and the archive do not overlap.",
       call. = FALSE)
}

# --- Per-location preprocessing ------------------------------------------
output_dir <- spec$output_dir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

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

# DelphiRF's daily mode lays reference dates out day by day. Days absent from
# the archive are synthetic episodes (all zero) and are removed. Forward-filled
# rows of real episodes are kept: they carry each real episode's trajectory.
drop_synthetic_reference_dates <- function(prepared, raw) {
  real <- unique(as.Date(raw[[reference_col]]))
  prepared[as.Date(prepared$reference_date) %in% real, , drop = FALSE]
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
               target_as_of_date = max_report),
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
  prepared <- drop_synthetic_reference_dates(prepared, raw)

  write_triangle(prepared, location)
  inventory[[length(inventory) + 1]] <- data.frame(
    dataset = spec$name, geo_value = location, raw_rows = nrow(raw),
    prepared_rows = nrow(prepared), genuine_rows = sum(prepared$genuine_event, na.rm = TRUE),
    min_report = min(prepared$report_date), max_report = max(prepared$report_date))
  if (i %% 5 == 0) message("Prepared ", i, " of ", length(locations), " locations")
}

# Stage-specific scope: locations too small to prepare (or that DelphiRF could
# not prepare) are skipped; their rows are recorded in the audit.
audit$min_location_rows <- as.integer(spec$min_location_rows)
audit$skipped_location_rows <- if (length(skipped)) {
  as.integer(sum(vapply(skipped, function(row) as.numeric(row$raw_rows), numeric(1))))
} else 0L
write_audit()

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
             target_as_of_date = as.Date(max_report),
             temporal_resolution = spec$temporal_resol),
  file.path(output_dir, "test_dates.csv"), row.names = FALSE)
writeLines(normalizePath(spec$raw_csv), file.path(output_dir, "raw_source.txt"))

message("Finished ", spec$name, ": ", nrow(inventory), " location(s), ",
        sum(inventory$prepared_rows), " prepared rows")
