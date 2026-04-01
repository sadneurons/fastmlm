#' S4 class for fastmlm model fit
#'
#' @slot call The matched call.
#' @slot formula The model formula.
#' @slot frame The model frame.
#' @slot flist List of grouping factors.
#' @slot cnms List of random-effect term column names.
#' @slot Gp Integer vector of group pointers.
#' @slot lower Numeric vector of lower bounds on theta.
#' @slot theta Numeric vector of fitted variance parameters.
#' @slot beta Numeric vector of fixed-effect coefficients.
#' @slot u Numeric vector of spherical random effects.
#' @slot sigma Residual standard deviation.
#' @slot deviance Profiled deviance at convergence.
#' @slot REML Logical; was REML estimation used?
#' @slot vcov_beta Variance-covariance matrix of fixed effects.
#' @slot optinfo List with optimiser convergence info.
#' @slot X Fixed-effects model matrix.
#' @slot Zt Sparse transpose of random-effects design matrix.
#' @slot Lambdat Sparse relative covariance factor.
#' @slot Lind Integer vector mapping theta to Lambdat entries.
#' @slot pp_xptr External pointer to C++ deviance function container.
#'
#' @exportClass fmlmMod
setClass("fmlmMod", representation(
  call      = "call",
  formula   = "formula",
  frame     = "data.frame",
  flist     = "list",
  cnms      = "list",
  Gp        = "integer",
  lower     = "numeric",
  theta     = "numeric",
  beta      = "numeric",
  u         = "numeric",
  sigma     = "numeric",
  deviance  = "numeric",
  REML      = "logical",
  vcov_beta = "matrix",
  optinfo   = "list",
  X         = "matrix",
  Zt        = "dgCMatrix",
  Lambdat   = "dgCMatrix",
  Lind      = "integer",
  pp_xptr   = "externalptr"
))
