#' Fit a fast generalised linear mixed model
#'
#' Fits a GLMM using Penalised Iteratively Reweighted Least Squares (PIRLS)
#' with a Laplace approximation, optimised in C++. Uses the same formula
#' syntax as \code{\link[lme4]{glmer}}.
#'
#' @param formula A two-sided formula with random effects in bar notation.
#' @param data A data frame.
#' @param family A family object (e.g., \code{binomial()}, \code{poisson()})
#'   or a string (\code{"binomial"}, \code{"poisson"}, \code{"Gamma"}).
#' @param control A list of control parameters; see \code{\link{fmlm_control}}.
#' @param weights Optional prior weights.
#' @param na.action Function for handling NAs.
#' @param subset Optional subset expression.
#' @param contrasts Contrast specifications for factors.
#' @param verbose Integer verbosity level.
#'
#' @return An object of class \code{\linkS4class{fglmmMod}}.
#'
#' @examples
#' \dontrun{
#' # Binomial GLMM
#' m <- fglmm(cbind(incidence, size - incidence) ~ period + (1 | herd),
#'             data = lme4::cbpp, family = binomial())
#' summary(m)
#'
#' # Poisson GLMM
#' m2 <- fglmm(count ~ treatment + (1 | site), data = mydata,
#'              family = poisson())
#' }
#'
#' @export
fglmm <- function(formula, data, family = binomial(),
                   control = fmlm_control(),
                   weights = NULL, na.action = na.omit,
                   subset = NULL, contrasts = NULL,
                   verbose = 0L) {

  mc <- match.call()
  verbose <- as.integer(verbose)

  # Parse family
  if (is.character(family)) family <- get(family, mode = "function")()
  if (is.function(family)) family <- family()
  fam_name <- family$family
  link_name <- family$link

  # Parse formula (reuse our fast parser)
  cached <- cache_get(formula, data)
  if (!is.null(cached)) {
    lf <- cached
  } else {
    lf <- tryCatch(
      fast_lFormula(formula, data, na.action = na.action, contrasts = contrasts),
      error = function(e) {
        if (!requireNamespace("lme4", quietly = TRUE)) {
          stop("Formula parsing failed: ", conditionMessage(e), call. = FALSE)
        }
        lf_raw <- lme4::glFormula(
          formula = formula, data = data, family = family,
          na.action = na.action,
          control = lme4::glmerControl(
            check.nobs.vs.nlev = "ignore",
            check.nobs.vs.nRE = "ignore"
          )
        )
        list(
          fr = lf_raw$fr, X = lf_raw$X,
          Zt = lf_raw$reTrms$Zt, Lambdat = lf_raw$reTrms$Lambdat,
          Lind = as.integer(lf_raw$reTrms$Lind),
          theta = as.numeric(lf_raw$reTrms$theta),
          lower = as.numeric(lf_raw$reTrms$lower),
          flist = lf_raw$reTrms$flist, cnms = lf_raw$reTrms$cnms,
          Gp = as.integer(lf_raw$reTrms$Gp)
        )
      }
    )
    cache_put(formula, data, lf)
  }

  # Extract components
  X <- as.matrix(lf$X)
  Zt <- lf$Zt
  Lambdat <- lf$Lambdat
  Lind <- as.integer(lf$Lind)
  theta0 <- as.numeric(lf$theta)
  lower <- as.numeric(lf$lower)
  Gp <- as.integer(lf$Gp)

  # Extract response — handle cbind for binomial
  n_trials <- NULL
  resp_var <- lf$fr[, 1]
  if (is.matrix(resp_var) && ncol(resp_var) == 2) {
    n_trials <- rowSums(resp_var)
    y <- as.numeric(resp_var[, 1] / n_trials)
  } else {
    y <- as.numeric(resp_var)
  }

  # Prior weights: n_trials for binomial, user weights otherwise, or ones
  if (!is.null(n_trials)) {
    pw <- as.numeric(n_trials)
  } else if (!is.null(weights)) {
    pw <- as.numeric(weights)
  } else {
    pw <- rep(1.0, length(y))
  }

  # Fit GLMM in C++
  result <- C_fastmlm_fit_glmm(
    y, X, Zt, Lambdat, Lind, theta0, lower, Gp, pw,
    fam_name, link_name,
    control$maxiter, control$ftol, control$gtol, verbose
  )

  # Name beta
  names_beta <- colnames(X)
  if (!is.null(names_beta)) {
    names(result$beta) <- names_beta
    rownames(result$vcov_beta) <- names_beta
    colnames(result$vcov_beta) <- names_beta
  }

  optinfo <- list(
    optimizer   = "L-BFGS-B (C++)",
    theta       = as.numeric(result$theta),
    deviance    = result$deviance,
    convergence = result$convergence,
    message     = result$message,
    fn_evals    = result$fn_evaluations,
    iterations  = result$iterations,
    pirls_iter  = result$pirls_iterations
  )

  methods::new("fglmmMod",
    call      = mc,
    formula   = formula,
    frame     = lf$fr,
    flist     = lf$flist,
    cnms      = lf$cnms,
    Gp        = Gp,
    lower     = lower,
    theta     = as.numeric(result$theta),
    beta      = as.numeric(result$beta),
    u         = as.numeric(result$u),
    eta       = as.numeric(result$eta),
    mu        = as.numeric(result$mu),
    deviance  = as.numeric(result$deviance),
    vcov_beta = as.matrix(result$vcov_beta),
    optinfo   = optinfo,
    family    = unclass(family),
    X         = X,
    Zt        = Zt,
    Lambdat   = Lambdat,
    Lind      = Lind
  )
}
