# Package-owned prospective similarity routing.
# Extracted from the earlier DelphiRF prediagnosis prototype so optional
# similarity pooling does not extend or depend on the DelphiRF model API.
.standardize_revision_triangle <- function(
    raw_data, cutoff_date, reference_date_col, report_date_col,
    value_col, location_col, start_report_date = NULL,
    revision_tolerance = sqrt(.Machine$double.eps)) {
  required <- c(reference_date_col, report_date_col, value_col)
  missing <- setdiff(required, names(raw_data))
  if (length(missing)) stop("Missing revision-process columns: ", paste(missing, collapse = ", "))
  x <- as.data.frame(raw_data)
  x$.reference_date <- as.Date(x[[reference_date_col]])
  x$.report_date <- as.Date(x[[report_date_col]])
  x$.value <- suppressWarnings(as.numeric(x[[value_col]]))
  x$.location <- if (location_col %in% names(x)) as.character(x[[location_col]]) else "all"
  cutoff_date <- as.Date(cutoff_date)
  keep <- !is.na(x$.reference_date) & !is.na(x$.report_date) &
    is.finite(x$.value) & x$.report_date <= cutoff_date
  x <- x[keep, , drop = FALSE]
  x <- x[order(x$.location, x$.reference_date, x$.report_date), , drop = FALSE]
  if (!nrow(x)) return(x)
  key <- interaction(x$.location, x$.reference_date, drop = TRUE)
  x$.previous <- ave(x$.value, key, FUN = function(z) c(NA_real_, head(z, -1L)))
  x$.revision <- x$.value - x$.previous
  x$.genuine <- !is.na(x$.previous) & abs(x$.revision) > revision_tolerance
  x$.lag <- as.numeric(x$.report_date - x$.reference_date)
  if (!is.null(start_report_date)) {
    x <- x[x$.report_date >= as.Date(start_report_date), , drop = FALSE]
  }
  x
}

#' Build cutoff-valid location--lag revision-process profiles
#'
#' The profile uses released data only and therefore remains available before
#' finalized targets arrive. It summarizes cadence, genuine-event frequency,
#' signed and absolute revision magnitude, revision stability, and row count.
#'
#' @param raw_data A reporting triangle.
#' @param cutoff_date Latest observable report date.
#' @param lag_breaks Breaks defining lag bands in days.
#' @param recent_days Optional trailing report-date window.
#' @param reference_date_col,report_date_col,value_col,location_col Column names.
#' @param revision_tolerance Absolute tolerance for a genuine revision.
#' @return One row per location and observed lag band.
#' @export
revision_process_profiles <- function(
    raw_data, cutoff_date, lag_breaks = c(-Inf, 7, 14, 28, Inf),
    recent_days = NULL, reference_date_col = "reference_date",
    report_date_col = "report_date", value_col = "value",
    location_col = "geo_value",
    revision_tolerance = sqrt(.Machine$double.eps)) {
  start <- if (is.null(recent_days)) NULL else as.Date(cutoff_date) - as.integer(recent_days) + 1L
  x <- .standardize_revision_triangle(
    raw_data, cutoff_date, reference_date_col, report_date_col,
    value_col, location_col, start, revision_tolerance
  )
  if (!nrow(x)) return(data.frame())
  x$.lag_band <- cut(x$.lag, lag_breaks, include.lowest = TRUE, right = TRUE)
  report_spacing <- function(d) {
    d <- sort(unique(as.Date(d)))
    if (length(d) < 2L) return(NA_real_)
    stats::median(as.numeric(diff(d)))
  }
  groups <- split(x, list(x$.location, x$.lag_band), drop = TRUE)
  rows <- lapply(groups, function(z) {
    revisions <- z$.revision[z$.genuine & is.finite(z$.revision)]
    data.frame(
      geo_value = z$.location[[1]],
      lag_band = as.character(z$.lag_band[[1]]),
      reporting_cadence_days = report_spacing(z$.report_date),
      genuine_event_rate = mean(z$.genuine),
      median_revision = if (length(revisions)) stats::median(revisions) else 0,
      median_abs_revision = if (length(revisions)) stats::median(abs(revisions)) else 0,
      revision_mad = if (length(revisions) > 1L) stats::mad(revisions, constant = 1) else 0,
      available_rows = nrow(z),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

.profile_matrix <- function(profiles) {
  metrics <- c("reporting_cadence_days", "genuine_event_rate", "median_revision",
               "median_abs_revision", "revision_mad", "available_rows")
  z <- as.matrix(profiles[metrics])
  for (j in seq_len(ncol(z))) {
    center <- stats::median(z[, j], na.rm = TRUE)
    spread <- stats::mad(z[, j], constant = 1, na.rm = TRUE)
    if (!is.finite(center)) center <- 0
    if (!is.finite(spread) || spread <= 0) spread <- 1
    z[!is.finite(z[, j]), j] <- center
    z[, j] <- (z[, j] - center) / spread
  }
  z
}

.stable_row_softmax <- function(log_weight) {
  log_weight <- as.matrix(log_weight)
  row_max <- apply(log_weight, 1L, max, na.rm = TRUE)
  row_max[!is.finite(row_max)] <- 0
  shifted <- log_weight - row_max
  shifted[!is.finite(shifted)] <- -Inf
  weight <- exp(shifted)
  denominator <- rowSums(weight)
  bad <- !is.finite(denominator) | denominator <= 0
  if (any(bad)) {
    weight[bad, ] <- 1
    denominator[bad] <- ncol(weight)
  }
  weight / denominator
}

#' Construct a prospective information-sharing plan
#'
#' Similarity-weighted routing assigns peer weights that decay smoothly with
#' profile distance. Soft clustering assigns fractional membership in a small
#' set of revision-process clusters and derives peer weights from shared
#' membership. Both plans use reports released by `cutoff_date` only.
#'
#' @param raw_data A reporting triangle.
#' @param cutoff_date Latest observable report date.
#' @param method `"similarity_weighted"` or `"soft_cluster"`.
#' @param lag_breaks Lag-band boundaries.
#' @param bandwidth Positive similarity bandwidth.
#' @param n_clusters Number of soft clusters.
#' @param seed Random seed for cluster centers.
#' @param ... Column arguments passed to [revision_process_profiles()].
#' @return A `revision_pooling_plan` object.
#' @export
prospective_pooling_plan <- function(
    raw_data, cutoff_date,
    method = c("similarity_weighted", "soft_cluster"),
    lag_breaks = c(-Inf, 7, 14, 28, Inf), bandwidth = 1,
    n_clusters = 3L, seed = 2026L, ...) {
  method <- match.arg(method)
  if (!is.finite(bandwidth) || bandwidth <= 0) stop("bandwidth must be positive.")
  profiles <- revision_process_profiles(raw_data, cutoff_date, lag_breaks, ...)
  if (!nrow(profiles)) stop("No profiles are observable at cutoff_date.")
  routes <- list()
  memberships <- list()
  for (band in unique(profiles$lag_band)) {
    p <- profiles[profiles$lag_band == band, , drop = FALSE]
    z <- .profile_matrix(p)
    ids <- p$geo_value
    if (method == "similarity_weighted" || nrow(p) == 1L) {
      distance <- as.matrix(stats::dist(z))
      weight <- exp(-(distance^2) / (2 * bandwidth^2))
    } else {
      distinct <- nrow(unique(as.data.frame(z)))
      k <- max(1L, min(as.integer(n_clusters), nrow(p), distinct))
      if (k == 1L) {
        membership <- matrix(1, nrow = nrow(p), ncol = 1L)
      } else {
        set.seed(seed)
        fit <- stats::kmeans(z, centers = k, nstart = 20)
        d2 <- vapply(seq_len(k), function(j) rowSums((z -
          matrix(fit$centers[j, ], nrow(z), ncol(z), byrow = TRUE))^2), numeric(nrow(z)))
        d2 <- as.matrix(d2)
        membership <- .stable_row_softmax(-d2 / (2 * bandwidth^2))
      }
      memberships[[band]] <- data.frame(
        geo_value = rep(ids, each = ncol(membership)), lag_band = band,
        cluster = rep(seq_len(ncol(membership)), times = length(ids)),
        membership = as.vector(t(membership)), stringsAsFactors = FALSE
      )
      weight <- membership %*% t(membership)
    }
    weight <- .stable_row_softmax(log(pmax(weight, .Machine$double.xmin)))
    routes[[band]] <- data.frame(
      target_location = rep(ids, each = length(ids)),
      source_location = rep(ids, times = length(ids)),
      lag_band = band,
      weight = as.vector(t(weight)), stringsAsFactors = FALSE
    )
  }
  out <- list(
    cutoff_date = as.Date(cutoff_date), method = method,
    profiles = profiles, routes = do.call(rbind, routes),
    memberships = if (length(memberships)) do.call(rbind, memberships) else NULL,
    parameters = list(lag_breaks = lag_breaks, bandwidth = bandwidth,
                      n_clusters = as.integer(n_clusters), seed = as.integer(seed))
  )
  class(out) <- "revision_pooling_plan"
  out
}
