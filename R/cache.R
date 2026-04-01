# Formula/data cache for repeated fits.
#
# When fitting the same model on the same data repeatedly (bootstrap,
# simulation, cross-validation), the formula parsing and matrix
# construction are identical each time. This cache stores the parsed
# result keyed by (formula, data_fingerprint) and returns it instantly
# on subsequent calls.

# Package-level cache environment
.fastmlm_cache <- new.env(parent = emptyenv())
.fastmlm_cache$entries <- list()
.fastmlm_cache$max_size <- 5L  # max cached formulas

#' Compute a fast fingerprint of a data frame
#'
#' Uses dimensions + column names + first/last values for a quick hash.
#' Not cryptographic — just enough to detect different datasets.
#' @keywords internal
data_fingerprint <- function(data) {
  paste0(
    nrow(data), "x", ncol(data), ":",
    paste(names(data), collapse = ","), ":",
    paste(vapply(data, function(col) {
      if (length(col) == 0) return("empty")
      paste0(col[1], ".", col[length(col)])
    }, character(1)), collapse = ";")
  )
}

#' Cache key from formula + data fingerprint
#' @keywords internal
cache_key <- function(formula, data) {
  paste0(deparse(formula, width.cutoff = 500), "|", data_fingerprint(data))
}

#' Get cached parse result, or NULL if not cached
#' @keywords internal
cache_get <- function(formula, data) {
  key <- cache_key(formula, data)
  .fastmlm_cache$entries[[key]]
}

#' Store a parse result in the cache
#' @keywords internal
cache_put <- function(formula, data, result) {
  key <- cache_key(formula, data)

  # Evict oldest if at capacity
  if (length(.fastmlm_cache$entries) >= .fastmlm_cache$max_size) {
    .fastmlm_cache$entries[[1]] <- NULL
  }

  .fastmlm_cache$entries[[key]] <- result
}

#' Clear the formula cache
#'
#' Removes all cached formula parse results. Call this if you've
#' modified your data in place and want to ensure fresh parsing.
#'
#' @export
fastmlm_clear_cache <- function() {
  .fastmlm_cache$entries <- list()
  invisible()
}
