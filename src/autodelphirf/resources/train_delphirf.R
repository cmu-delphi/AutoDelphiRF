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
  library(dplyr)
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

prepared_dir <- normalizePath(positional[[1]])
raw_path     <- normalizePath(positional[[2]])
schedule_path <- normalizePath(positional[[3]])
output_path  <- positional[[4]]
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

# Training deliberately uses the installed DelphiRF package. RevRoute must not
# silently substitute its own quantile-regression implementation or a source
# checkout whose uncommitted state is absent from the run manifest.

# DelphiRF writes each fitted model under model_save_dir/<indicator_signal_...>/
# via create_dir_not_exist(), which calls dir.create() WITHOUT recursive=TRUE.
# So model_save_dir itself must already exist, or saveRDS() fails with
# "cannot open the connection" and every fit is silently dropped. Create one
# run-scoped cache dir up front (mirrors run_nhsn28_pooling_candidates.R's
# dir.create(model_dir, recursive = TRUE)). Model filenames already encode
# training_end_date + geo key + lag group + tau + hyperparams, and our geo key
# encodes method/fold/group/pool, so a single shared dir cannot collide.
model_save_dir <- tempfile("delphirf_comparator_models_")
dir.create(model_save_dir, recursive = TRUE, showWarnings = FALSE)

ref_col    <- get_flag("ref-col", "reference_date")
report_col <- get_flag("report-col", "report_date")
geo_col    <- get_flag("geo-col", "geo_value")
value_col  <- get_flag("value-col", "value")
# Per-dataset, not per-script: "count" vs "fraction" is a property of the
# dataset's own reporting triangle (see full_package_pooling_preprocess.R's
# `configs` list, mirrored in raw_archive_manifest.json). Previously hardcoded
# to "count", which was only correct for the nhsn* datasets.
value_type <- match.arg(get_flag("value-type", "count"), c("count", "fraction"))
# Optional denominator column, used only to build a fraction-aware revision
# profile for the similarity-weighted pooling routes. Datasets whose raw value
# column is already a rate/percentage (e.g. nssp_*) are value_type="fraction"
# with no denominator and need no denom-col.
denom_col <- get_flag("denom-col", "")
lag_terms  <- as.integer(strsplit(get_flag("lag-terms", "7,14"), ",")[[1]])
weekdays   <- jsonlite::fromJSON(get_flag("weekdays-json", "{}"), simplifyVector = TRUE)
if (!is.list(weekdays)) weekdays <- as.list(weekdays)
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
required_revision_args <- "model_backend"
missing_revision_args <- setdiff(required_revision_args, revision_args)
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
pool_assignments_path <- get_flag("revroute-pools", "")
pool_assignments <- if (nzchar(pool_assignments_path)) fread(pool_assignments_path) else NULL

schedule <- fread(schedule_path)
schedule[, test_date := as.Date(test_date)]
if (!nrow(schedule)) stop("Schedule is empty: ", schedule_path)
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

location_files <- list.files(prepared_dir, pattern = "\\.(parquet|csv|csv\\.gz)$", full.names = TRUE)
location_files <- location_files[!grepl("/(inventory|test_dates|genuine_rows|skipped_locations)\\.csv", location_files)]
if (!length(location_files)) stop("No prepared triangles found in ", prepared_dir)
all_data <- rbindlist(lapply(location_files, function(path) {
  d <- if (grepl("\\.parquet$", path)) {
    if (!requireNamespace("arrow", quietly = TRUE)) stop("Reading Parquet needs R package 'arrow'.")
    as.data.frame(arrow::read_parquet(path))
  } else as.data.frame(data.table::fread(path))
  d$reference_date <- as.Date(d$reference_date)
  d$report_date <- as.Date(d$report_date)
  d$target_date <- as.Date(d$target_date)
  if (!("geo_value" %in% names(d))) d$geo_value <- tools::file_path_sans_ext(basename(path))
  d
}), fill = TRUE)
setDT(all_data)
setorder(all_data, geo_value, report_date, reference_date)
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

# Raw profile rows, used only for similarity-weighted pooling routes.
profile_rows <- NULL
if ("Similarity-weighted DelphiRF" %in% requested_methods) {
  raw_input <- fread(raw_path)
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

independent_design <- function(train, base_params, columns, response) {
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
  #   add_weights_related -> add_sqrtscale -> subset to selected columns
  #   -> drop_na
  # kept_bins take part in that drop_na even though they are not covariates,
  # so the surviving row set cannot be reproduced without them.
  candidates <- c(base_params, columns)
  prepared <- DelphiRF:::add_weights_related(as.data.frame(train))
  scaled <- DelphiRF:::add_sqrtscale(prepared, sqrt(max(prepared$value_7dav, na.rm = TRUE)))
  prepared <- scaled$data
  selected <- unique(c("reference_date", "lag", "report_date", candidates,
                       "value_slope_diff", "value_7dav_diff", scaled$kept_bins, response))
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

fit_one <- function(train, test, method, fold, group, pool_id, observation_weights = NULL,
                    params_list = NULL) {
  if (nrow(train) < 10L || !nrow(test)) return(NULL)
  test_meta <- data.frame(
    geo_value = as.character(test$geo_value), reference_date = test$reference_date,
    report_date = test$report_date, lag = test$lag, target_date = test$target_date,
    stringsAsFactors = FALSE
  )
  train <- DelphiRF:::add_weights_related(as.data.frame(train))
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
    alignment_params <- DelphiRF:::create_params_list(
      train, lag_terms, temporal_resol, onehot_weekdays = weekdays)
  }
  alignment_scale <- sqrt(max(train$value_7dav, na.rm = TRUE))
  alignment_train <- DelphiRF:::add_sqrtscale(train, alignment_scale)
  alignment_test <- DelphiRF:::add_sqrtscale_test(
    as.data.frame(test), alignment_scale, alignment_train$kept_bins)
  alignment_columns <- unique(c(alignment_params, alignment_train$kept_bins,
                                "reference_date", "lag", "report_date"))
  alignment_columns <- alignment_columns[alignment_columns %in% names(alignment_test)]
  aligned_test <- test[stats::complete.cases(alignment_test[, alignment_columns, drop = FALSE]), ]
  key <- paste(method, "f", fold, "g", gsub("-", "_", group), "p", make.names(pool_id), sep = "_")
  result <- tryCatch(
    DelphiRF::revision_forecast(
      train_data = as.data.frame(train), test_data = as.data.frame(test),
      params_list = params_list,
      taus = experiment_taus, smoothed_target = smoothed_target, lagged_term_list = lag_terms,
      temporal_resol = temporal_resol, lambda = .1, gamma = .1, lp_solver = lp_solver,
      test_lag_group = group, geo = key, value_type = value_type,
      model_save_dir = model_save_dir,
      indicator = "revroute_comparator", signal = method,
      training_end_date = as.character(schedule$test_date[[fold]]),
      training_days = schedule$training_days[[fold]], train_models = TRUE,
      make_predictions = TRUE,
      onehot_weekdays = weekdays, model_backend = model_backend
    ), error = function(e) {
      message("  [", method, " fold ", fold, " group ", group, " geo ",
              pool_id, "] fit failed: ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(result) || !nrow(result)) return(NULL)
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
    if (nrow(result) != nrow(aligned_test)) {
      stop("cannot align predictions to DelphiRF's complete test rows: returned ",
           nrow(result), " rows for ", nrow(aligned_test), " complete test rows")
    }
    result$geo_value <- as.character(aligned_test$geo_value)
  } else {
    result$geo_value <- as.character(result$geo_value)
  }
  result <- merge(result, test_meta, by = c("reference_date", "report_date", "lag", "geo_value"),
                  all.x = TRUE, sort = FALSE)
  if (anyDuplicated(result[, c("geo_value", "reference_date", "report_date", "lag")])) {
    stop("duplicated prediction rows after the test-metadata join for ", method)
  }
  out <- data.frame(
    method = method, fold = fold, cutoff = as.Date(schedule$test_date[[fold]]),
    geo_value = as.character(result$geo_value), reference_date = result$reference_date,
    report_date = result$report_date, lag = result$lag, target_date = result$target_date,
    prediction = result$predicted_tau0.5, stringsAsFactors = FALSE
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
  }
  out
}

out <- list()
fold_ids <- if (is.na(only_fold)) seq_len(nrow(schedule)) else only_fold

for (fold in fold_ids) {
  cutoff <- schedule$test_date[[fold]]
  next_cutoff <- if ("testing_days" %in% names(schedule) &&
                     is.finite(schedule$testing_days[[fold]])) {
    cutoff + as.integer(schedule$testing_days[[fold]])
  } else if (fold < nrow(schedule)) schedule$test_date[[fold + 1L]] else
    as.Date(schedule$experiment_end_date[[fold]])
  visible <- all_data[report_date < next_cutoff]
  training <- visible[
    report_date < cutoff & target_date < cutoff &
      target_date > cutoff - schedule$training_days[[fold]] & genuine_event == TRUE
  ]
  testing <- visible[report_date >= cutoff & report_date < next_cutoff & genuine_event == TRUE]
  if (!nrow(training) || !nrow(testing)) next

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
    current_fold <- fold
    fold_pools <- pool_assignments[pool_assignments[["fold"]] == current_fold]
    if (!nrow(fold_pools)) stop("No RevRoute task pools for fold ", fold)
    test_tasks <- unique(testing[, .(geo_value = as.character(geo_value), lag = as.integer(lag))])
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
    train_pooled <- merge(training, fold_pools[, .(geo_value, lag, pool)],
                          by = c("geo_value", "lag"), all.x = FALSE, all.y = FALSE)
    test_pooled <- merge(testing, fold_pools[, .(geo_value, lag, pool)],
                         by = c("geo_value", "lag"), all.x = FALSE, all.y = FALSE)
    for (pool_id in sort(unique(test_pooled$pool))) {
      pool_train <- train_pooled[pool == pool_id]
      pool_test <- test_pooled[pool == pool_id]
      if (nrow(pool_train) < 10L) pool_train <- training[lag %in% unique(pool_test$lag)]
      if (nrow(pool_train) < 10L) pool_train <- training
      z <- fit_one(pool_train, pool_test, "RevRoute DelphiRF", fold, "pool",
                   paste0("revroute_", pool_id))
      if (!is.null(z)) out[[length(out) + 1L]] <- z
    }
  }

  if ("Global DelphiRF" %in% requested_methods) {
    fe <- add_fixed_effects(training, testing)
    base_params <- DelphiRF:::create_params_list(
      DelphiRF:::add_weights_related(fe$train), lag_terms, temporal_resol,
      onehot_weekdays = weekdays)
    global_params <- independent_design(fe$train, base_params, fe$columns, target_column)
    z <- fit_one(fe$train, fe$test, "Global DelphiRF", fold, "all", "global",
                 params_list = global_params)
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
        z <- fit_one(local_train, test_geo, "Naive DelphiRF", fold, group, paste0("local_", geo))
        if (!is.null(z)) out[[length(out) + 1L]] <- z
      }
      if ("Similarity-weighted DelphiRF" %in% requested_methods) {
        source_geo <- as.character(train_group$geo_value)
        lag_value <- mean(bounds)
        sim_w <- route_weights(similarity_plan, geo, lag_value, source_geo)
        z <- fit_one(train_group, test_geo, "Similarity-weighted DelphiRF", fold, group,
                     paste0("similarity_", geo), observation_weights = sim_w)
        if (!is.null(z)) out[[length(out) + 1L]] <- z
      }
    }
  }
  message("DelphiRF comparators: fold ", fold, "/", nrow(schedule), " complete")
}

if (!length(out)) stop("No comparator predictions were produced -- check the schedule and prepared data.")
result <- rbindlist(out, fill = TRUE)
if (grepl("\\.gz$", output_path)) {
  fwrite(result, output_path, compress = "gzip")
} else {
  fwrite(result, output_path)
}
message("Wrote ", nrow(result), " comparator rows to ", output_path)
jsonlite::write_json(list(
  backend = "installed DelphiRF",
  similarity_planner = "autodelphirf package-owned prospective_pooling_plan",
  similarity_weight_application = "deterministic weighted resampling before ordinary DelphiRF fit",
  prediction_scale = "working scale; Python export performs the dataset-specific inverse transform",
  package_version = as.character(utils::packageVersion("DelphiRF")),
  package_path = find.package("DelphiRF"),
  methods = requested_methods,
  model_backend = model_backend,
  taus = experiment_taus
), paste0(output_path, ".provenance.json"), auto_unbox = TRUE, pretty = TRUE)
