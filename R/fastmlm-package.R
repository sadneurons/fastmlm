#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @importFrom methods new setClass setGeneric setMethod is show validObject
#' @importFrom Matrix t sparseMatrix
#' @importFrom stats optim na.omit model.frame model.matrix model.response AIC BIC
#'   sigma nobs df.residual predict confint terms delete.response reformulate
#'   qnorm pt coef fitted logLik residuals vcov printCoefmat cov2cor formula
#'   simulate update anova
# Define our own S3 generics for fixef/ranef/VarCorr
# so they work without lme4 loaded

#' Extract fixed effects
#'
#' Generic function to extract fixed-effect coefficients from a
#' fitted mixed model.
#'
#' @param object A fitted model object.
#' @param ... Additional arguments.
#' @return Named numeric vector of fixed-effect coefficients.
#' @rdname fixef
#' @export
fixef <- function(object, ...) UseMethod("fixef")

#' Extract random effects
#' @param object A fitted model object.
#' @param ... Additional arguments.
#' @return A list of data frames, one per grouping factor.
#' @export
ranef <- function(object, ...) UseMethod("ranef")

#' Extract variance-covariance of random effects
#' @param x A fitted model object.
#' @param sigma Residual standard deviation (used for scaling).
#' @param ... Additional arguments.
#' @return A list of variance-covariance matrices.
#' @export
VarCorr <- function(x, sigma = 1, ...) UseMethod("VarCorr")
#' @useDynLib fastmlm, .registration = TRUE
## usethis namespace: end
NULL

# Suppress R CMD check notes for package-level environment
utils::globalVariables(".fastmlm_cache")
