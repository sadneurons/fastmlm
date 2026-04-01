.onLoad <- function(libname, pkgname) {
  # Register emmeans methods if emmeans is available
  if (requireNamespace("emmeans", quietly = TRUE)) {
    emmeans::.emm_register("fmlmMod", pkgname = "fastmlm")
  }
}

#' Report BLAS and system information
#'
#' Returns information about the linked BLAS library, Eigen version,
#' OpenMP availability, and thread counts.
#'
#' @return A list with components:
#' \describe{
#'   \item{eigen_version}{Eigen C++ library version}
#'   \item{blas_library}{Detected BLAS library (OpenBLAS, MKL, or Reference)}
#'   \item{blas_threads}{Number of BLAS threads}
#'   \item{has_openmp}{Logical; is OpenMP available?}
#'   \item{omp_threads}{Number of OpenMP threads}
#'   \item{r_blas}{R's La_library() path}
#' }
#' @export
fastmlm_blas_info <- function() {
  info <- C_fastmlm_blas_info()
  info$r_blas <- La_library()
  info
}

#' Set thread counts for BLAS and OpenMP
#'
#' @param blas Integer; number of BLAS threads (OpenBLAS or MKL).
#'   Use 0 to leave unchanged.
#' @param omp Integer; number of OpenMP threads for fastmlm operations.
#'   Use 0 to leave unchanged.
#' @export
fastmlm_set_threads <- function(blas = 0L, omp = 0L) {
  C_fastmlm_set_threads(as.integer(blas), as.integer(omp))
  invisible()
}

#' Benchmark BLAS performance
#'
#' Times a dense matrix multiplication (dgemm) to verify BLAS performance.
#'
#' @param n Integer; matrix dimension for the benchmark.
#' @return Time in seconds for an n x n matrix multiplication.
#' @export
fastmlm_benchmark_blas <- function(n = 500L) {
  C_fastmlm_benchmark_blas(as.integer(n))
}
