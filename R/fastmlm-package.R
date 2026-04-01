#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @importFrom methods new setClass setGeneric setMethod is show validObject
#' @importFrom Matrix t sparseMatrix
#' @importFrom stats optim na.omit model.frame model.matrix model.response AIC BIC
#'   sigma nobs df.residual predict confint terms delete.response reformulate
#'   qnorm pt coef fitted logLik residuals vcov printCoefmat cov2cor
#' @importFrom lme4 lFormula lmerControl fixef ranef VarCorr
#' @export fixef
#' @export ranef
#' @export VarCorr
#' @useDynLib fastmlm, .registration = TRUE
## usethis namespace: end
NULL

# Suppress R CMD check notes for package-level environment
utils::globalVariables(".fastmlm_cache")
