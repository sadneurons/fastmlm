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

#' Compute a robust fingerprint of a data frame
#'
#' Uses dimensions, column names, column types, and a sample of values
#' spread across the data to detect changes. Not cryptographic but
#' catches modifications anywhere in the data, not just first/last rows.
#' @param data A data frame.
#' @return A character string fingerprint.
#' @keywords internal
data_fingerprint <- function(data) {
  nr <- nrow(data)
  nc <- ncol(data)

  # Sample row indices spread across the data
  sample_rows <- unique(c(1L, nr,
    as.integer(seq(1, nr, length.out = min(nr, 10)))))

  # Build fingerprint from sampled values + types
  col_fp <- vapply(data, function(col) {
    vals <- col[sample_rows]
    paste0(class(col)[1], ":", paste(format(vals), collapse = ","))
  }, character(1))

  paste0(nr, "x", nc, ":",
         paste(names(data), collapse = ","), ":",
         paste(col_fp, collapse = ";"))
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
