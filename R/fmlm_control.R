#' Control parameters for fmlm model fitting
#'
#' @param optimizer Character; optimisation method. Currently \code{"L-BFGS-B"}
#'   (default) or \code{"bobyqa"} (via \pkg{nloptr}).
#' @param maxiter Integer; maximum number of optimiser iterations.
#' @param ftol Numeric; function tolerance for convergence.
#' @param gtol Numeric; projected gradient norm tolerance for convergence.
#' @param pcg_threshold Integer; random-effects dimension above which the PCG
#'   solver is used for crossed random effects instead of direct Cholesky.
#' @param verbose Integer; verbosity level (0 = silent, 1 = progress, 2 = debug).
#' @param nthreads Integer; number of threads for OpenMP operations.
#'   0 (default) uses auto-detection.
#' @param use_gpu Character; GPU usage: \code{"auto"}, \code{"yes"}, or
#'   \code{"no"}.
#'
#' @return A named list of control parameters.
#' @export
fmlm_control <- function(optimizer = "L-BFGS-B",
                         maxiter = 300L,
                         ftol = 1e-10,
                         gtol = 1e-8,
                         pcg_threshold = 5000L,
                         verbose = 0L,
                         nthreads = 0L,
                         use_gpu = "auto") {
  list(
    optimizer     = match.arg(optimizer, c("L-BFGS-B", "bobyqa")),
    maxiter       = as.integer(maxiter),
    ftol          = as.double(ftol),
    gtol          = as.double(gtol),
    pcg_threshold = as.integer(pcg_threshold),
    verbose       = as.integer(verbose),
    nthreads      = as.integer(nthreads),
    use_gpu       = match.arg(use_gpu, c("auto", "yes", "no"))
  )
}
