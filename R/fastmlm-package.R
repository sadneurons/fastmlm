#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @importFrom methods new setClass setGeneric setMethod is show validObject
#' @importFrom Matrix t sparseMatrix
#' @importFrom stats optim na.omit model.frame model.matrix model.response AIC BIC
#' @importFrom lme4 lFormula lmerControl fixef ranef VarCorr
#' @useDynLib fastmlm, .registration = TRUE
## usethis namespace: end
NULL
