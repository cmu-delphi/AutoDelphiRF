#!/usr/bin/env Rscript
#
# Fit the requested AutoDelphiRF model choices through the installed DelphiRF
# package and write one prediction row for every scheduled case and method.
#
# Usage:
#   Rscript train_delphirf.R PREPARED_DIR RAW_CSV \
#       SCHEDULE_CSV OUTPUT_CSV [--flags...]
#
# PREPARED_DIR contains the prepared files created by prepare_triangle.R.
# RAW_CSV is the original revision archive.
# SCHEDULE_CSV contains the rolling training and testing dates.
# OUTPUT_CSV receives the long prediction table.

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(DelphiRF)
})

script_argument <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_argument[grepl("^--file=", script_argument)][1])
source(file.path(dirname(normalizePath(script_path)), "similarity_pooling.R"), local = TRUE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) {
  stop("usage: train_delphirf.R PREPARED_DIR RAW_CSV SCHEDULE_CSV ",
       "OUTPUT_CSV [--flags...]")
}
positional <- args[1:4]
flag_args <- if (length(args) > 4L) args[5:length(args)] else character(0)

parse_flags <- function(flag_args) {
  flags <- list()
  i <- 1L
  while (i <= length(flag_args)) {
    key <- flag_args[[i]]
    if (!startsWith(key, "--")) stop("Expected a --flag, got: ", key)
    name <- sub("^--", "", key)
    if (i == length(flag_args)) stop("Flag --", name, " is missing a value")
    flags[[name]] <- flag_args[[i + 1L]]
    i <- i + 2L
  }
  flags
}
flags <- parse_flags(flag_args)
get_flag <- function(name, default) if (is.null(flags[[name]])) default else flags[[name]]
get_bool_flag <- function(name, default) {
  value <- tolower(as.character(get_flag(name, if (default) "true" else "false")))
  if (!value %in% c("true", "false")) {
    stop("--", name, " must be true or false")
  }
  identical(value, "true")
}

prepared_dir <- normalizePath(positional[[1]])
raw_path     <- normalizePath(positional[[2]])
schedule_path <- normalizePath(positional[[3]])
output_path  <- positional[[4]]
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

# Training deliberately uses the installed DelphiRF package. RevRoute must not
# silently substitute its own quantile-regression implementation or a source
# checkout whose uncommitted state is absent from the run manifest.

# DelphiRF already knows how to save and reload fitted models. AutoDelphiRF
# supplies a persistent directory so a retrospective run also becomes the
# reusable model history for later releases of the same archive.
model_save_dir <- normalizePath(
  get_flag("model-save-dir", tempfile("delphirf_models_")), mustWork = FALSE)
dir.create(model_save_dir, recursive = TRUE, showWarnings = FALSE)
model_mode <- match.arg(get_flag("model-mode", "train"), c("train", "auto"))

ref_col    <- get_flag("ref-col", "reference_date")
report_col <- get_flag("report-col", "report_date")
geo_col    <- get_flag("geo-col", "geo_value")
constant_geo <- get_flag("constant-geo", "")
value_col  <- get_flag("value-col", "value")
# Per-dataset, not per-script: "count" vs "fraction" is a property of the
# dataset's own reporting triangle, recorded at preparation.
value_type <- match.arg(get_flag("value-type", "count"), c("count", "fraction"))
genuine_training <- get_bool_flag("genuine-training", TRUE)
genuine_testing <- get_bool_flag("genuine-testing", TRUE)
# Optional denominator column, used only to build a fraction-aware revision
# profile for the similarity-weighted pooling routes. Datasets whose raw value
# column is already a rate/percentage (e.g. nssp_*) are value_type="fraction"
# with no denominator and need no denom-col.
denom_col <- get_flag("denom-col", "")
# Working scale of the prepared response: log(num + 1) - log(denom + 1) when the
# triangle was built from numerator and denominator columns, else log(value + 1).
working_scale <- if (nzchar(denom_col)) "log_ratio" else "log_value"
lag_terms  <- as.integer(strsplit(get_flag("lag-terms", "7,14"), ",")[[1]])
weekday_groups <- jsonlite::fromJSON(get_flag("weekdays-json", "{}"), simplifyVector = FALSE)
if (is.null(weekday_groups)) weekday_groups <- list()
requested_methods <- trimws(strsplit(
  get_flag("methods", "Naive DelphiRF,Similarity-weighted DelphiRF"), ","
)[[1]])
allowed_methods <- c("RevRoute DelphiRF", "Naive DelphiRF",
                     "Similarity-weighted DelphiRF", "Global DelphiRF")
unknown_methods <- setdiff(requested_methods, allowed_methods)
if (length(unknown_methods)) {
  stop("Unsupported --methods entr(y/ies): ", paste(unknown_methods, collapse = ", "),
       ". This script only produces: ", paste(allowed_methods, collapse = ", "))
}
revision_args <- names(formals(DelphiRF::revision_forecast))
required_revision_args <- c("model_backend", "working_scale")
missing_revision_args <- setdiff(required_revision_args, revision_args)
for (helper in c("create_weight_anchors", "incomplete_test_rows")) {
  if (!exists(helper, envir = asNamespace("DelphiRF"))) {
    missing_revision_args <- c(missing_revision_args, paste0(helper, "()"))
  }
}
if (length(missing_revision_args)) {
  stop("Installed DelphiRF is too old for this pipeline; revision_forecast() is missing: ",
       paste(missing_revision_args, collapse = ", "), ". Install the current DelphiRF release.")
}
model_backend <- match.arg(get_flag("model-backend", "quantreg"), c("quantreg", "quantgen"))
lp_solver <- get_flag("lp-solver", "glpk")
experiment_taus <- sort(as.numeric(strsplit(get_flag("taus", "0.5"), "[,;[:space:]]+")[[1]]))
if (!length(experiment_taus) || any(!is.finite(experiment_taus)) ||
    any(experiment_taus <= 0 | experiment_taus >= 1) || anyDuplicated(experiment_taus)) {
  stop("--taus must contain unique quantiles strictly between 0 and 1.")
}
if (!any(abs(experiment_taus - 0.5) < 1e-10)) {
  stop("--taus must include 0.5 (comparators are reported at the median).")
}
only_fold <- suppressWarnings(as.integer(get_flag("fold", NA_character_)))
# A location is a label, never a number: "01" and "1" are different locations.
read_with_text_geo <- function(path, column = "geo_value") {
  # fread needs R.utils for gzipped files; base R reads them without it.
  if (grepl("\\.gz$", path) && !requireNamespace("R.utils", quietly = TRUE)) {
    header <- names(utils::read.csv(gzfile(path), nrows = 1L, check.names = FALSE))
    classes <- if (column %in% header) setNames("character", column) else NA
    return(as.data.table(utils::read.csv(gzfile(path), colClasses = classes,
                                         check.names = FALSE, stringsAsFactors = FALSE)))
  }
  header <- names(fread(path, nrows = 0L))
  if (column %in% header) fread(path, colClasses = setNames(list(column), "character"))
  else fread(path)
}
pool_assignments_path <- get_flag("revroute-pools", "")
pool_assignments <- if (nzchar(pool_assignments_path)) read_with_text_geo(pool_assignments_path) else NULL
issued_forecasts_path <- get_flag("issued-forecasts", "")
issued_forecasts <- if (nzchar(issued_forecasts_path) && file.exists(issued_forecasts_path)) {
  read_with_text_geo(issued_forecasts_path)
} else NULL
unresolved_only <- identical(tolower(get_flag("unresolved-only", "false")), "true")

schedule <- fread(schedule_path)
schedule[, test_date := as.Date(test_date)]
if (!nrow(schedule)) stop("Schedule is empty: ", schedule_path)
# Fold k is the k-th origin in calendar order, exactly as Python numbers it.
setorder(schedule, test_date)
if (!is.na(only_fold) && (only_fold < 1L || only_fold > nrow(schedule))) {
  stop("--fold ", only_fold, " is outside the schedule's ", nrow(schedule), " origins")
}
temporal_resol <- tolower(as.character(schedule$temporal_resolution[[1]]))
lag_groups <- if (temporal_resol == "weekly") {
  as.character(DelphiRF:::TEST_LAG_GROUPS_WEEKLY)
} else {
  as.character(DelphiRF:::TEST_LAG_GROUPS_DAILY)
}
parse_group <- function(x) {
  z <- as.integer(strsplit(x, "-", fixed = TRUE)[[1]])
  if (length(z) == 1L) c(z, z) else z
}
# Location-specific comparators forecast every lag below the target lag. Keep
# DelphiRF's own lag groups up to L - 1 and continue past its last group in
# blocks as wide as its widest daily group (14 days) or one week.
cover_lags_below_target <- function(groups, target_lag, block) {
  bounds <- lapply(groups, parse_group)
  keep <- vapply(bounds, function(b) b[[1]] <= target_lag - 1L, logical(1))
  bounds <- lapply(bounds[keep], function(b) c(b[[1]], min(b[[2]], target_lag - 1L)))
  start <- if (length(bounds)) max(vapply(bounds, `[[`, integer(1), 2L)) + 1L else 0L
  while (start <= target_lag - 1L) {
    bounds[[length(bounds) + 1L]] <- c(start, min(start + block - 1L, target_lag - 1L))
    start <- start + block
  }
  vapply(bounds, function(b) if (b[[1]] == b[[2]]) as.character(b[[1]])
         else paste0(b[[1]], "-", b[[2]]), character(1))
}
lag_groups <- cover_lags_below_target(lag_groups, as.integer(schedule$target_lag[[1]]),
                                      if (temporal_resol == "weekly") 7L else 14L)

location_files <- list.files(prepared_dir, pattern = "\\.(parquet|csv|csv\\.gz)$", full.names = TRUE)
location_files <- location_files[!grepl("/(inventory|test_dates|genuine_rows|skipped_locations|cleaned_releases)\\.csv", location_files)]
if (!length(location_files)) stop("No prepared triangles found in ", prepared_dir)
all_data <- rbindlist(lapply(location_files, function(path) {
  d <- if (grepl("\\.parquet$", path)) {
    if (!requireNamespace("arrow", quietly = TRUE)) stop("Reading Parquet needs R package 'arrow'.")
    as.data.frame(arrow::read_parquet(path))
  } else as.data.frame(read_with_text_geo(path))
  d$reference_date <- as.Date(d$reference_date)
  d$report_date <- as.Date(d$report_date)
  d$target_date <- as.Date(d$target_date)
  if (!("geo_value" %in% names(d))) d$geo_value <- tools::file_path_sans_ext(basename(path))
  d
}), fill = TRUE)
setDT(all_data)
setorder(all_data, geo_value, report_date, reference_date)

# DelphiRF keeps a canonical indicator for every weekday in the prepared
# triangle. AutoDelphiRF's pre-diagnosis chooses the contrasts used for
# fitting. Days not represented by a group form the reference category.
canonical_weekdays <- c("Mon", "Tue", "Wed", "Thurs", "Fri", "Sat", "Sun")
add_weekday_groups <- function(frame, groups) {
  if (!length(groups)) return(frame)
  # Every group is computed from the canonical columns before any is written,
  # so a group labelled with a weekday name cannot overwrite a column that a
  # later group still reads. The same groups define the reference-date (_ref)
  # and report-date (_issue) weekday features.
  indicators <- list()
  for (group_name in names(groups)) {
    members <- as.character(groups[[group_name]])
    for (axis in c("ref", "issue")) {
      sources <- paste0(members, "_", axis)
      missing <- setdiff(sources, names(frame))
      if (length(missing)) {
        stop("prepared triangle is missing canonical weekday column(s): ",
             paste(missing, collapse = ", "))
      }
      indicators[[paste0(group_name, "_", axis)]] <- as.integer(
        rowSums(as.data.frame(frame)[, sources, drop = FALSE], na.rm = FALSE) > 0)
    }
  }
  for (column in names(indicators)) frame[[column]] <- indicators[[column]]
  frame
}
all_data <- add_weekday_groups(all_data, weekday_groups)

# Calendar indicators (weekday groups on the reference and report dates, and
# week of month) that never vary carry no information. Screened twice: on the
# prepared triangle's genuine observations, and again within each training
# sample, where a further indicator can be constant.
is_calendar_indicator <- function(columns) grepl("_(ref|issue)$", columns)
constant_columns <- function(frame, columns) {
  columns[vapply(columns, function(column) {
    if (!(column %in% names(frame))) return(FALSE)
    values <- frame[[column]]
    values <- values[!is.na(values)]
    length(unique(values)) < 2L
  }, logical(1))]
}
calendar_candidates <- function() {
  base <- DelphiRF:::create_params_list(all_data[0], lag_terms, temporal_resol)
  grouped <- if (temporal_resol == "daily" && length(weekday_groups)) {
    c(paste0(names(weekday_groups), "_ref"), paste0(names(weekday_groups), "_issue"))
  } else character(0)
  unique(c(base, grouped))[is_calendar_indicator(unique(c(base, grouped)))]
}
genuine_rows <- all_data[genuine_event == TRUE]
constant_calendar_in_triangle <- constant_columns(as.data.frame(genuine_rows),
                                                  calendar_candidates())
if (length(constant_calendar_in_triangle)) {
  message("Calendar indicators constant in the prepared triangle, excluded: ",
          paste(constant_calendar_in_triangle, collapse = ", "))
}

model_params <- function(train) {
  base <- DelphiRF:::create_params_list(train, lag_terms, temporal_resol)
  canonical <- c(paste0(canonical_weekdays, "_ref"),
                 paste0(canonical_weekdays, "_issue"))
  base <- setdiff(base, canonical)
  grouped <- if (temporal_resol == "daily" && length(weekday_groups)) {
    c(paste0(names(weekday_groups), "_ref"),
      paste0(names(weekday_groups), "_issue"))
  } else character(0)
  params <- unique(c(base, grouped))
  calendar <- params[is_calendar_indicator(params)]
  excluded <- union(constant_calendar_in_triangle,
                    constant_columns(as.data.frame(train), calendar))
  setdiff(params, excluded)
}
target_column <- as.character(schedule$target_column[[1]])
# DelphiRF's `smoothed_target` picks the response column: TRUE ->
# log_value_target_7dav, FALSE -> log_value_target. Deriving it from the
# schedule's own target_column guarantees the comparator predicts exactly the
# column the RevRoute run is evaluated against, instead of being hardcoded to
# the smoothed one. That distinction is real: the chng FRACTION build predicts
# the smoothed target and the chng COUNT build the unsmoothed one, matching
# DelphiRF's chng_fraction_config.R / chng_count_config.R.
smoothed_target <- grepl("_7dav$", target_column)
smoothed_flag <- get_flag("smoothed-target", NA_character_)
if (!is.na(smoothed_flag)) smoothed_target <- toupper(smoothed_flag) %in% c("TRUE","T","YES","1")
message("smoothed_target = ", smoothed_target, " (from target_column '", target_column, "')")
if (!(target_column %in% names(all_data))) {
  stop("Schedule's target_column '", target_column, "' is not in the prepared data.")
}

# The prepared triangle is the one definition of each target's date:
# prepare_triangle.R writes target_date = reference_date + L, and it is used
# here as stored. A triangle that disagrees was built by an older version.
rebuild <- paste0("Rebuild it with the current version: `autodelphirf prepare` ",
                  "(or `autodelphirf run --archive ...`).")
if (!("target_as_of_date" %in% names(schedule)) || all(is.na(schedule$target_as_of_date))) {
  stop("the prepared schedule records no target_as_of_date. ", rebuild, call. = FALSE)
}
target_as_of <- max(as.Date(schedule$target_as_of_date), na.rm = TRUE)
mismatched <- sum(all_data$target_date != all_data$reference_date +
                    as.integer(schedule$target_lag[[1]]), na.rm = TRUE) +
  sum(is.na(all_data$target_date))
if (mismatched) {
  stop(mismatched, " prepared row(s) have a target_date other than reference_date + ",
       schedule$target_lag[[1]], " days; the prepared dataset is incompatible. ", rebuild,
       call. = FALSE)
}
# Before the recorded as-of date reaches t + L the forward-filled value is
# still provisional, so it is not a target (prepare_triangle.R already
# withholds it; this keeps training independent of how the file was written).
target_value_columns <- grep("^(log_)?value_target", names(all_data), value = TRUE)
for (column in target_value_columns) {
  set(all_data, which(all_data$target_date > target_as_of), column, NA)
}

# Raw profile rows, used only for similarity-weighted pooling routes.
profile_rows <- NULL
if ("Similarity-weighted DelphiRF" %in% requested_methods) {
  raw_input <- if (nzchar(constant_geo)) fread(raw_path) else read_with_text_geo(raw_path, geo_col)
  if (nzchar(constant_geo)) raw_input[[geo_col]] <- constant_geo
  if (!(geo_col %in% names(raw_input))) stop("--geo-col '", geo_col, "' is not a column of ", raw_path)
  profile_value <- as.numeric(raw_input[[value_col]])
  if (nzchar(denom_col)) {
    if (!(denom_col %in% names(raw_input))) stop("--denom-col '", denom_col, "' is not a column of ", raw_path)
    denom <- as.numeric(raw_input[[denom_col]])
    profile_value <- ifelse(is.finite(denom) & denom > 0, profile_value / denom, NA_real_)
  }
  profile_rows <- data.table(
    geo_value = as.character(raw_input[[geo_col]]),
    reference_date = as.Date(raw_input[[ref_col]]),
    report_date = as.Date(raw_input[[report_col]]), value_raw = profile_value
  )[is.finite(value_raw)]
  if (!nrow(profile_rows)) stop("No usable raw profile rows after applying configured columns.")
  rm(raw_input)
}
profile_lag_breaks <- c(-Inf, 7, 14, 28, Inf)

route_weights <- function(plan, target_geo, lag_value, sources) {
  band <- as.character(cut(lag_value, profile_lag_breaks, include.lowest = TRUE, right = TRUE))
  z <- plan$routes[plan$routes$target_location == target_geo & plan$routes$lag_band == band, , drop = FALSE]
  w <- setNames(rep(0, length(sources)), sources)
  matched <- match(sources, z$source_location)
  w[!is.na(matched)] <- z$weight[matched[!is.na(matched)]]
  w[!is.finite(w) | w < 0] <- 0
  if (!any(w > 0, na.rm = TRUE)) {
    local_rows <- !is.na(sources) & sources == target_geo
    if (any(local_rows)) w[local_rows] <- 1 else w[] <- 1
  }
  positive_mean <- mean(w[w > 0], na.rm = TRUE)
  if (!is.finite(positive_mean) || positive_mean <= 0) return(rep(1, length(w)))
  w / positive_mean
}


# ---------------------------------------------------------------------------
# Global fixed-effects Delphi-RF
# ---------------------------------------------------------------------------
# One penalized Delphi-RF fit over ALL fold rows with available target values -- every
# location and every lag together -- retaining additive location and lag
# effects:
#     Q_tau(Y* | x, i, l) = alpha_tau + gamma_{i,tau} + eta_{l,tau} + x' beta_tau.
#
# gamma_i enters as location indicator columns, one level dropped as the
# reference so they are not collinear with the intercept. eta_l is carried by
# Delphi-RF's own `inv_log_lag`, which is retained rather than replaced by a
# full set of per-lag indicators: those 11-14 extra dummies pushed the design
# to ~76-79 parameters against ~2,850 genuine training rows and quantreg's
# interior-point solve failed on 29 of 35 nhsn28 folds. `inv_log_lag` is a
# deterministic function of lag, so the two forms cannot coexist -- keeping it
# is the smooth-offset reading of the same additive lag effect.
#
# `revision_forecast` uses `test_lag_group` only to name the cached model, not
# to filter rows, so pooling is simply a matter of not subsetting.
add_fixed_effects <- function(train, test) {
  train <- as.data.frame(train); test <- as.data.frame(test)
  locations <- sort(unique(c(as.character(train$geo_value), as.character(test$geo_value))))
  columns <- character(0)
  if (length(locations) > 1L) {
    for (loc in locations[-1]) {
      name <- paste0("fe_loc_", make.names(loc))
      train[[name]] <- as.integer(as.character(train$geo_value) == loc)
      test[[name]] <- as.integer(as.character(test$geo_value) == loc)
      columns <- c(columns, name)
    }
  }
  list(train = train, test = test, columns = columns)
}

independent_design <- function(train, base_params, columns, response, anchors) {
  # Prune the pooled design to a full-rank set of columns.
  #
  # The redundancy is NOT confined to the location indicators. Pooling every
  # location makes Delphi-RF's own weekday indicators degenerate: nhsn28
  # reports on Wed/Fri only, so Wed_ref + Fri_ref is identically 1 and exactly
  # collinear with the intercept. A base-only pooled fit with 12 parameters
  # fails the same way a 67-parameter one does, so base covariates must be
  # eligible for pruning too.
  #
  # Rank selection runs on exactly the rows revision_forecast will fit:
  #   add_weights_related -> subset to selected columns -> drop_na
  candidates <- c(base_params, columns)
  prepared <- DelphiRF::add_weights_related(as.data.frame(train), anchors)
  selected <- unique(c("reference_date", "lag", "report_date", candidates,
                       "value_slope_diff", "value_7dav_diff", response))
  selected <- selected[selected %in% names(prepared)]
  frame <- prepared[stats::complete.cases(prepared[, selected, drop = FALSE]), , drop = FALSE]
  if (!nrow(frame)) return(candidates)
  ordered <- candidates[candidates %in% names(frame)]
  design <- cbind(`(intercept)` = 1, as.matrix(frame[, ordered, drop = FALSE]))
  decomposition <- qr(design, tol = 1e-7)
  if (decomposition$rank >= ncol(design)) return(ordered)
  kept <- colnames(design)[decomposition$pivot[seq_len(decomposition$rank)]]
  dropped <- setdiff(ordered, kept)
  if (length(dropped)) {
    message("  [Global DelphiRF] dropped ", length(dropped),
            " rank-deficient column(s): ", paste(head(dropped, 6), collapse = ", "),
            if (length(dropped) > 6) " ..." else "")
  }
  intersect(ordered, kept)
}

fit_failures <- list()
record_failure <- function(method, fold, group, pool_id, test_rows, reason, train_only) {
  fit_failures[[length(fit_failures) + 1L]] <<- data.frame(
    method = method, fold = as.integer(fold), group = as.character(group),
    pool_id = as.character(pool_id), expected_test_rows = as.integer(test_rows),
    error = reason, kind = if (train_only) "train_only" else "prediction",
    stringsAsFactors = FALSE)
  message("  [", method, " fold ", fold, " group ", group, " geo ", pool_id, "] ",
          if (train_only) "train-only fit failed: " else "fit failed: ", reason)
  NULL
}

# Test rows DelphiRF could not forecast because a predictor is missing (for
# example a lagged feature whose earlier reference date has no report). They
# are recorded next to the fit failures rather than lost silently.
dropped_test_rows <- list()
record_dropped <- function(dropped, method, fold, group, pool_id) {
  dropped <- as.data.frame(dropped)
  dropped_test_rows[[length(dropped_test_rows) + 1L]] <<- data.frame(
    method = method,
    fold = if ("fold_id" %in% names(schedule)) schedule$fold_id[[fold]] else fold,
    cutoff = as.Date(schedule$test_date[[fold]]), group = as.character(group),
    pool_id = as.character(pool_id), dropped, stringsAsFactors = FALSE)
}

fit_one <- function(train, test, method, fold, group, pool_id, anchors,
                    observation_weights = NULL, params_list = NULL, train_only = FALSE) {
  if (!nrow(test)) return(NULL)
  if (nrow(train) < 10L) {
    return(record_failure(method, fold, group, pool_id, if (train_only) 0L else nrow(test),
                          paste0("fewer than 10 training rows (", nrow(train), ")"), train_only))
  }
  test_meta <- data.frame(
    geo_value = as.character(test$geo_value), reference_date = test$reference_date,
    report_date = test$report_date, lag = test$lag, target_date = test$target_date,
    stringsAsFactors = FALSE
  )
  # Similarity weights compare each training row with its location's
  # reporting context at this origin (one anchor per location and lag).
  train <- DelphiRF::add_weights_related(as.data.frame(train), anchors)
  if (!is.null(observation_weights)) {
    weights <- as.numeric(observation_weights)
    if (length(weights) != nrow(train) || any(!is.finite(weights)) ||
        any(weights < 0) || !any(weights > 0)) {
      stop("similarity weights must be finite, non-negative, and match training rows")
    }
    # Keep DelphiRF's API model-only. Apply similarity weights in this adapter
    # through deterministic weighted resampling, then give the resulting
    # training sample to the ordinary DelphiRF fit.
    seed <- 2026L + as.integer(fold) * 1000L +
      sum(utf8ToInt(paste(group, pool_id, sep = "_")))
    set.seed(seed)
    train <- train[sample.int(nrow(train), nrow(train), replace = TRUE,
                              prob = weights), , drop = FALSE]
  }
  # Reproduce revision_forecast()'s test-row completeness filter before the
  # call. DelphiRF intentionally omits rows whose predictors are unavailable,
  # and its returned frame omits geo_value. Keeping the exact surviving rows
  # here lets pooled predictions be mapped back to locations without treating
  # an ordinary unavailable forecast as a fatal row-count mismatch.
  alignment_params <- params_list
  if (is.null(alignment_params)) {
    alignment_params <- model_params(train)
  }
  alignment_test <- as.data.frame(test)
  alignment_columns <- unique(c(alignment_params, "reference_date", "lag", "report_date"))
  alignment_columns <- alignment_columns[alignment_columns %in% names(alignment_test)]
  aligned_test <- test[stats::complete.cases(alignment_test[, alignment_columns, drop = FALSE]), ]
  basic_columns <- c("reference_date", "lag", "report_date")
  basic_test <- as.data.frame(test)[, basic_columns, drop = FALSE]
  basic_aligned_test <- test[
    stats::complete.cases(basic_test), , drop = FALSE]
  # Calendar origin, rather than the schedule row number, is the stable model
  # identity across an archive that grows over time.
  origin_key <- format(as.Date(schedule$test_date[[fold]]), "%Y%m%d")
  key <- paste(method, "origin", origin_key, "g", gsub("-", "_", group),
               "p", make.names(pool_id), sep = "_")
  result <- tryCatch(
    DelphiRF::revision_forecast(
      train_data = as.data.frame(train), test_data = as.data.frame(test),
      params_list = alignment_params,
      taus = experiment_taus, smoothed_target = smoothed_target, lagged_term_list = lag_terms,
      temporal_resol = temporal_resol, lambda = .1, gamma = .1, lp_solver = lp_solver,
      test_lag_group = group, geo = key, value_type = value_type,
      working_scale = working_scale,
      model_save_dir = model_save_dir,
      indicator = "revroute_comparator", signal = method,
      training_end_date = as.character(schedule$test_date[[fold]]),
      training_days = schedule$training_days[[fold]],
      train_models = identical(model_mode, "train"),
      # A train-only fit saves the model without forecasting.
      make_predictions = !train_only,
      model_backend = model_backend
    ), error = function(e) {
      record_failure(method, fold, group, pool_id, if (train_only) 0L else nrow(test),
                     conditionMessage(e), train_only)
    }
  )
  if (train_only) return(NULL)
  dropped <- attr(result, "dropped_test_rows")
  if (!is.null(dropped) && nrow(dropped)) record_dropped(dropped, method, fold, group, pool_id)
  if (is.null(result) || !nrow(result)) return(NULL)
  tryCatch(collect_predictions(
    result,
    list(feature_complete = aligned_test, basic_complete = basic_aligned_test),
    test_meta, method, fold),
           error = function(e) record_failure(method, fold, group, pool_id, nrow(test),
                                              conditionMessage(e), FALSE))
}

# Map DelphiRF's returned rows back to the test cases and keep every quantile.
collect_predictions <- function(result, alignment_candidates, test_meta, method, fold) {
  # Include geo_value in the join key whenever the prediction frame carries it.
  # A per-location fit has one location so the date/lag triple is unique, but a
  # pooled fit spans every location and the triple repeats once per location --
  # joining on it alone fans each predicted row out across all of them.
  # revision_forecast keeps only basic_cols + covariates + response, so
  # geo_value is dropped. A per-location fit does not care -- its date/lag
  # triple is unique -- but a pooled fit spans every location and the triple
  # repeats once per location, so joining on it alone fans each predicted row
  # out across all of them. The returned frame is the test frame with
  # prediction columns appended, in the order of the filtered test frame.
  if (!("geo_value" %in% names(result))) {
    key_columns <- c("reference_date", "report_date", "lag")
    same_ordered_keys <- function(candidate) {
      if (nrow(result) != nrow(candidate)) return(FALSE)
      result_keys <- lapply(as.data.frame(result)[, key_columns, drop = FALSE], as.character)
      candidate_keys <- lapply(as.data.frame(candidate)[, key_columns, drop = FALSE], as.character)
      identical(result_keys, candidate_keys)
    }
    valid_candidates <- alignment_candidates[
      vapply(alignment_candidates, same_ordered_keys, logical(1))]
    if (length(valid_candidates) == 0L) {
      expected <- paste(vapply(alignment_candidates, nrow, integer(1)), collapse = " or ")
      stop("cannot align predictions to DelphiRF's complete test rows: returned ",
           nrow(result), " rows for ", expected,
           " valid test rows, or the ordered date/lag keys differ")
    }
    # Ordinary fits use feature_complete. Constant-response fits do not need
    # predictors and therefore use DelphiRF's basic-key completeness path.
    matched_test <- valid_candidates[[1L]]
    result$geo_value <- as.character(matched_test$geo_value)
  } else {
    result$geo_value <- as.character(result$geo_value)
  }
  result <- merge(result, test_meta, by = c("reference_date", "report_date", "lag", "geo_value"),
                  all.x = TRUE, sort = FALSE)
  if (anyDuplicated(result[, c("geo_value", "reference_date", "report_date", "lag")])) {
    stop("duplicated prediction rows after the test-metadata join for ", method)
  }
  out <- data.frame(
    method = method,
    fold = if ("fold_id" %in% names(schedule)) schedule$fold_id[[fold]] else fold,
    cutoff = as.Date(schedule$test_date[[fold]]),
    geo_value = as.character(result$geo_value), reference_date = result$reference_date,
    report_date = result$report_date, lag = result$lag, target_date = result$target_date,
    prediction = result$predicted_tau0.5,
    prediction_working = result$predicted_tau0.5_working,
    stringsAsFactors = FALSE
  )
  # Carry EVERY fitted quantile, not only the median. `prediction` stays the
  # median so the existing point-comparator contract is unchanged, and each
  # requested tau is added as its own `q<tau>` column. Without this the other
  # eight quantile regressions are fitted at full cost and then discarded, so
  # an imported comparator could never be scored on WIS or coverage.
  for (tau in experiment_taus) {
    source_col <- paste0("predicted_tau", format(tau, trim = TRUE, scientific = FALSE))
    if (!source_col %in% names(result)) {
      stop("DelphiRF returned no column '", source_col, "'; requested taus and ",
           "returned quantile columns disagree.")
    }
    out[[paste0("q", format(tau, trim = TRUE, scientific = FALSE))]] <- result[[source_col]]
    working_col <- paste0(source_col, "_working")
    if (!working_col %in% names(result)) {
      stop("DelphiRF returned no column '", working_col, "'; reporting- and ",
           "working-scale predictions must be emitted together.")
    }
    out[[paste0("q", format(tau, trim = TRUE, scientific = FALSE), "_working")]] <-
      result[[working_col]]
  }
  out
}

out <- list()
fold_ids <- if (is.na(only_fold)) seq_len(nrow(schedule)) else only_fold

# Wall-clock fitting time per retraining origin. A fold may end early with
# `next`, so each fold is closed when the next one starts (and after the loop).
fold_timings <- list()
open_fold <- NULL
close_fold <- function() {
  if (is.null(open_fold)) return(invisible())
  fold_timings[[length(fold_timings) + 1L]] <<- data.frame(
    fold = if ("fold_id" %in% names(schedule)) schedule$fold_id[[open_fold$fold]] else open_fold$fold,
    cutoff = as.Date(schedule$test_date[[open_fold$fold]]),
    seconds = proc.time()[["elapsed"]] - open_fold$started)
  open_fold <<- NULL
}

for (fold in fold_ids) {
  close_fold()
  open_fold <- list(fold = fold, started = proc.time()[["elapsed"]])
  cutoff <- schedule$test_date[[fold]]
  next_cutoff <- if ("testing_days" %in% names(schedule) &&
                     is.finite(schedule$testing_days[[fold]])) {
    cutoff + as.integer(schedule$testing_days[[fold]])
  } else if (fold < nrow(schedule)) schedule$test_date[[fold + 1L]] else
    as.Date(schedule$experiment_end_date[[fold]])
  visible <- all_data[report_date < next_cutoff]
  # The forecast context of this origin: for every location and lag, the
  # latest row reported on or before the cutoff, whether or not its target has
  # matured or it is a genuine release. Training weights are measured from it.
  anchors <- DelphiRF::create_weight_anchors(as.data.frame(all_data), cutoff,
                                             by = "geo_value")
  training <- visible[
    # The update ingests the current day's release before fitting. Targets
    # observable on the origin date are therefore valid training outcomes;
    # report-state feature rows released on the origin are also observable.
    report_date <= cutoff & target_date <= cutoff &
      target_date > cutoff - schedule$training_days[[fold]] &
      !is.na(get(target_column)) & is.finite(value_7dav)
  ]
  if (genuine_training) training <- training[genuine_event == TRUE]
  # A reference date whose trailing 7-day window is still incomplete has no
  # 7-day-average features (DelphiRF leaves them missing), so it can neither
  # train nor be forecast; build_cases drops the same rows.
  testing <- visible[report_date >= cutoff & report_date < next_cutoff &
                       is.finite(value_7dav)]
  if (genuine_testing) testing <- testing[genuine_event == TRUE]
  if (unresolved_only) {
    # Unresolved: the stored target date is after the as-of date recorded at
    # preparation (not the latest report seen, which can be earlier than it).
    testing <- testing[target_date > target_as_of]
    if (nrow(testing)) testing[, (target_column) := NA_real_]
  }
  if (!is.null(issued_forecasts) && nrow(testing)) {
    issued <- copy(issued_forecasts)
    issued[, cutoff := as.Date(cutoff)]
    issued <- issued[cutoff == as.Date(schedule$test_date[[fold]])]
    if (nrow(issued)) {
      issued[, reference_date := as.Date(reference_date)]
      issued[, report_date := as.Date(report_date)]
      issued[, lag := as.integer(lag)]
      issued[, geo_value := as.character(geo_value)]
      testing <- testing[!issued, on = .(geo_value, reference_date, report_date, lag)]
    }
  }
  if (!nrow(training)) next
  # RevRoute persists every pool at its scheduled origin even when no report
  # is released that day.  An empty test set therefore means train-only for
  # RevRoute, while comparator methods with no test cases have nothing to do.
  if (!nrow(testing) && !("RevRoute DelphiRF" %in% requested_methods)) next

  similarity_plan <- if ("Similarity-weighted DelphiRF" %in% requested_methods) {
    prospective_pooling_plan(
      as.data.frame(profile_rows), cutoff, method = "similarity_weighted",
      lag_breaks = profile_lag_breaks, reference_date_col = "reference_date",
      report_date_col = "report_date", value_col = "value_raw", location_col = "geo_value"
    )
  } else NULL

  if ("RevRoute DelphiRF" %in% requested_methods) {
    if (is.null(pool_assignments)) {
      stop("RevRoute DelphiRF requires --revroute-pools generated by the RevRoute stage.")
    }
    pool_cutoff <- as.Date(schedule$test_date[[fold]])
    fold_pools <- if ("cutoff" %in% names(pool_assignments)) {
      pool_assignments[as.Date(cutoff) == pool_cutoff]
    } else {
      pool_assignments[pool_assignments[["fold"]] == fold]
    }
    if (!nrow(fold_pools)) stop("No RevRoute task pools for fold ", fold)
    test_tasks <- unique(testing[, .(geo_value = as.character(geo_value), lag = as.integer(lag))])
    learned_pools <- copy(fold_pools)
    missing_tasks <- test_tasks[!fold_pools, on = .(geo_value, lag)]
    if (nrow(missing_tasks)) {
      fallback_rows <- lapply(seq_len(nrow(missing_tasks)), function(i) {
        task <- missing_tasks[i]
        candidates <- fold_pools[geo_value == task$geo_value]
        if (!nrow(candidates)) candidates <- fold_pools
        chosen <- candidates[which.min(abs(lag - task$lag))]
        data.table(geo_value = task$geo_value, lag = task$lag, pool = chosen$pool,
                   threshold = chosen$threshold)
      })
      fold_pools <- unique(rbindlist(c(list(fold_pools), fallback_rows), fill = TRUE),
                           by = c("geo_value", "lag"))
    }
    # Training rows come from learned pool members only (A.2.2: assigning an
    # unrepresented task does not change the learned pools).
    train_pooled <- merge(training, learned_pools[, .(geo_value, lag, pool)],
                          by = c("geo_value", "lag"), all.x = FALSE, all.y = FALSE)
    test_pooled <- merge(testing, fold_pools[, .(geo_value, lag, pool)],
                         by = c("geo_value", "lag"), all.x = FALSE, all.y = FALSE)
    for (pool_id in sort(unique(test_pooled$pool))) {
      pool_train <- train_pooled[pool == pool_id]
      pool_test <- test_pooled[pool == pool_id]
      if (nrow(pool_train) < 10L) pool_train <- training[lag %in% unique(pool_test$lag)]
      if (nrow(pool_train) < 10L) pool_train <- training
      z <- fit_one(pool_train, pool_test, "RevRoute DelphiRF", fold, "pool",
                   paste0("revroute_", pool_id), anchors = anchors)
      if (!is.null(z)) out[[length(out) + 1L]] <- z
    }
    # A real-time origin must leave a complete model set behind. Some learned
    # pools have no forecast row in the archive yet; train those pools now so
    # a later release can remain prediction-only instead of fitting a model
    # after the origin at which its parameters were supposed to be frozen.
    untested_pools <- setdiff(sort(unique(fold_pools$pool)),
                              sort(unique(test_pooled$pool)))
    for (pool_id in untested_pools) {
      pool_train <- train_pooled[pool == pool_id]
      pool_lags <- unique(fold_pools[pool == pool_id]$lag)
      if (nrow(pool_train) < 10L) pool_train <- training[lag %in% pool_lags]
      if (nrow(pool_train) < 10L) pool_train <- training
      # revision_forecast takes a test frame even when predictions are
      # disabled; a training row supplies it and is never forecast.
      fit_one(pool_train, pool_train[1], "RevRoute DelphiRF", fold, "pool",
              paste0("revroute_", pool_id), anchors = anchors, train_only = TRUE)
    }
  }

  if ("Global DelphiRF" %in% requested_methods) {
    fe <- add_fixed_effects(training, testing)
    base_params <- model_params(fe$train)
    global_params <- independent_design(fe$train, base_params, fe$columns, target_column,
                                        anchors)
    z <- fit_one(fe$train, fe$test, "Global DelphiRF", fold, "all", "global",
                 anchors = anchors, params_list = global_params)
    if (!is.null(z)) out[[length(out) + 1L]] <- z
  }

  for (group in lag_groups) {
    bounds <- parse_group(group)
    train_group <- training[lag >= bounds[[1]] - 1L & lag <= bounds[[2]] + 1L]
    test_group <- testing[lag >= bounds[[1]] & lag <= bounds[[2]]]
    if (!nrow(test_group)) next
    for (geo in sort(unique(as.character(test_group$geo_value)))) {
      test_geo <- test_group[geo_value == geo]
      local_train <- train_group[geo_value == geo]

      if ("Naive DelphiRF" %in% requested_methods) {
        z <- fit_one(local_train, test_geo, "Naive DelphiRF", fold, group, paste0("local_", geo),
                     anchors = anchors)
        if (!is.null(z)) out[[length(out) + 1L]] <- z
      }
      if ("Similarity-weighted DelphiRF" %in% requested_methods) {
        source_geo <- as.character(train_group$geo_value)
        lag_value <- mean(bounds)
        sim_w <- route_weights(similarity_plan, geo, lag_value, source_geo)
        z <- fit_one(train_group, test_geo, "Similarity-weighted DelphiRF", fold, group,
                     paste0("similarity_", geo), anchors = anchors,
                     observation_weights = sim_w)
        if (!is.null(z)) out[[length(out) + 1L]] <- z
      }
    }
  }
  message("DelphiRF comparators: fold ", fold, "/", nrow(schedule), " complete")
}
close_fold()
fold_timing_table <- if (length(fold_timings)) rbindlist(fold_timings) else
  data.table(fold = integer(), cutoff = as.Date(character()), seconds = numeric())
fold_timing_path <- paste0(output_path, ".fold_seconds.csv")
fwrite(fold_timing_table, fold_timing_path)

if (!length(out) && !unresolved_only) {
  stop("No comparator predictions were produced -- check the schedule and prepared data.")
}
if (!length(out)) {
  result <- data.table(
    method = character(), fold = integer(), cutoff = as.Date(character()),
    geo_value = character(), reference_date = as.Date(character()),
    report_date = as.Date(character()), lag = integer(),
    target_date = as.Date(character()), prediction = numeric())
  for (tau in experiment_taus) {
    result[[paste0("q", format(tau, trim = TRUE, scientific = FALSE))]] <- numeric()
  }
} else {
  result <- rbindlist(out, fill = TRUE)
}
if (grepl("\\.gz$", output_path)) {
  fwrite(result, output_path, compress = "gzip")
} else {
  fwrite(result, output_path)
}
message("Wrote ", nrow(result), " comparator rows to ", output_path)
failure_table <- if (length(fit_failures)) rbindlist(fit_failures, fill = TRUE) else
  data.table(method = character(), fold = integer(), group = character(),
             pool_id = character(), expected_test_rows = integer(), error = character(),
             kind = character())
failure_path <- paste0(output_path, ".fit_failures.csv")
fwrite(failure_table, failure_path)
dropped_table <- if (length(dropped_test_rows)) rbindlist(dropped_test_rows, fill = TRUE) else
  data.table(method = character(), fold = integer(), cutoff = as.Date(character()),
             group = character(), pool_id = character(), geo_value = character(),
             reference_date = as.Date(character()), report_date = as.Date(character()),
             lag = numeric(), missing_features = character(), reason = character())
dropped_path <- paste0(output_path, ".dropped_test_rows.csv")
fwrite(dropped_table, dropped_path)
if (nrow(dropped_table)) {
  message("DelphiRF left ", nrow(dropped_table), " test row(s) unforecast because a predictor ",
          "is missing; details: ", dropped_path)
}
prediction_failures <- sum(failure_table$kind == "prediction")
train_only_failures <- sum(failure_table$kind == "train_only")
if (prediction_failures) {
  warning("DelphiRF completed with ", prediction_failures,
          " failed fit(s); details: ", failure_path, call. = FALSE)
}
jsonlite::write_json(list(
  backend = "installed DelphiRF",
  similarity_planner = "autodelphirf package-owned prospective_pooling_plan",
  similarity_weight_application = "deterministic weighted resampling before ordinary DelphiRF fit",
  prediction_scale = "working scale; Python export performs the dataset-specific inverse transform",
  package_version = as.character(utils::packageVersion("DelphiRF")),
  package_path = find.package("DelphiRF"),
  methods = requested_methods,
  genuine_training = genuine_training,
  genuine_testing = genuine_testing,
  model_backend = model_backend,
  model_save_dir = model_save_dir,
  model_mode = model_mode,
  fit_status = if (prediction_failures) "partial_failure" else "complete",
  fit_failure_count = prediction_failures,
  train_only_failure_count = train_only_failures,
  fit_failures_file = failure_path,
  fit_failures = as.data.frame(failure_table),
  dropped_test_row_count = nrow(dropped_table),
  dropped_test_rows_by_reason = as.list(table(dropped_table$reason)),
  dropped_test_rows_file = dropped_path,
  fitting_seconds_total = sum(fold_timing_table$seconds),
  fold_seconds_file = fold_timing_path,
  weekday_groups = weekday_groups,
  calendar_indicators_constant_in_triangle = I(constant_calendar_in_triangle),
  effective_base_params = model_params(as.data.frame(all_data)),
  taus = experiment_taus
), paste0(output_path, ".provenance.json"), auto_unbox = TRUE, pretty = TRUE)
