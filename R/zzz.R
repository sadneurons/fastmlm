.onLoad <- function(libname, pkgname) {
  # Register our S3 methods with lme4's generics if lme4 is loaded
  # (so fixef(fastmlm_obj) works regardless of load order)
  if (isNamespaceLoaded("lme4")) {
    registerS3method("fixef", "fmlmMod", fixef.fmlmMod,
                     envir = asNamespace("lme4"))
    registerS3method("ranef", "fmlmMod", ranef.fmlmMod,
                     envir = asNamespace("lme4"))
    registerS3method("VarCorr", "fmlmMod", VarCorr.fmlmMod,
                     envir = asNamespace("lme4"))
    registerS3method("fixef", "fglmmMod", fixef.fglmmMod,
                     envir = asNamespace("lme4"))
    registerS3method("ranef", "fglmmMod", ranef.fglmmMod,
                     envir = asNamespace("lme4"))
    registerS3method("VarCorr", "fglmmMod", VarCorr.fglmmMod,
                     envir = asNamespace("lme4"))
  }

  # Register emmeans methods if emmeans is available
  if (requireNamespace("emmeans", quietly = TRUE)) {
    emmeans::.emm_register("fmlmMod", pkgname = "fastmlm")
  }

  # Register broom/generics methods if available
  register_if <- function(generic, class, method) {
    for (pkg in c("generics", "broom", "broom.mixed")) {
      if (isNamespaceLoaded(pkg)) {
        registerS3method(generic, class, method, envir = asNamespace(pkg))
      }
    }
  }
  register_if("tidy", "fmlmMod", tidy.fmlmMod)
  register_if("glance", "fmlmMod", glance.fmlmMod)
  register_if("augment", "fmlmMod", augment.fmlmMod)
}

# Also register when lme4 is loaded after us
.onAttach <- function(libname, pkgname) {
  setHook(packageEvent("lme4", "onLoad"), function(...) {
    registerS3method("fixef", "fmlmMod", fixef.fmlmMod,
                     envir = asNamespace("lme4"))
    registerS3method("ranef", "fmlmMod", ranef.fmlmMod,
                     envir = asNamespace("lme4"))
    registerS3method("VarCorr", "fmlmMod", VarCorr.fmlmMod,
                     envir = asNamespace("lme4"))
    registerS3method("fixef", "fglmmMod", fixef.fglmmMod,
                     envir = asNamespace("lme4"))
    registerS3method("ranef", "fglmmMod", ranef.fglmmMod,
                     envir = asNamespace("lme4"))
    registerS3method("VarCorr", "fglmmMod", VarCorr.fglmmMod,
                     envir = asNamespace("lme4"))
  })
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
