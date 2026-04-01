#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @importFrom methods new setClass setGeneric setMethod is show validObject
#' @importFrom Matrix t sparseMatrix
#' @importFrom stats optim na.omit model.frame model.matrix model.response AIC BIC
#'   sigma nobs df.residual predict confint terms delete.response reformulate
#'   qnorm pt coef fitted logLik residuals vcov printCoefmat cov2cor formula
#' @importFrom lme4 lFormula lmerControl fixef ranef VarCorr
#'
#' @name fixef
#' @rdname reexports
#' @keywords internal
#' @export
NULL

#' @name ranef
#' @rdname reexports
#' @keywords internal
#' @export
NULL

#' @name VarCorr
#' @rdname reexports
#' @keywords internal
#' @export
NULL

#' Re-exported functions from lme4
#'
#' These generics are imported from \pkg{lme4} and re-exported so that
#' \code{fixef()}, \code{ranef()}, and \code{VarCorr()} work without
#' explicitly loading lme4.
#'
#' @name reexports
#' @keywords internal
NULL
#' @useDynLib fastmlm, .registration = TRUE
## usethis namespace: end
NULL

# Suppress R CMD check notes for package-level environment
utils::globalVariables(".fastmlm_cache")
