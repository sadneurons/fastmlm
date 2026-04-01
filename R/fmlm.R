#' Fit a fast multilevel linear model
#'
#' Fits a linear mixed-effects model using an optimised C++ backend with
#' sparse Cholesky factorisation, L-BFGS-B optimisation with numerical
#' gradients, and direct BLAS linkage. Uses the same formula syntax as
#' \code{\link[lme4]{lmer}}.
#'
#' @param formula A two-sided formula with random effects specified using
#'   bar notation: \code{y ~ x1 + x2 + (1 + x1 | group)}.
#' @param data A data frame containing the variables in the formula.
#' @param REML Logical; use REML (default \code{TRUE}) or ML estimation.
#' @param control A list of control parameters; see \code{\link{fmlm_control}}.
#' @param weights Optional prior weights.
#' @param na.action Function for handling \code{NA}s.
#' @param subset Optional subset expression.
#' @param contrasts Contrast specifications for factors.
#' @param verbose Integer verbosity level.
#'
#' @return An object of class \code{\linkS4class{fmlmMod}}.
#'
#' @examples
#' \dontrun{
#' library(lme4)
#' m <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)
#' summary(m)
#' fixef(m)
#' ranef(m)
#' }
#'
#' @export
fmlm <- function(formula, data, REML = TRUE,
                 control = fmlm_control(),
                 weights = NULL, na.action = na.omit,
                 subset = NULL, contrasts = NULL,
                 verbose = 0L) {

  mc <- match.call()
  verbose <- as.integer(verbose)

  # --- Parse formula via lme4::lFormula ---
  lf_args <- list(
    formula  = formula,
    data     = data,
    REML     = REML,
    na.action = na.action,
    control   = lme4::lmerControl(
      check.nobs.vs.nlev  = "ignore",
      check.nobs.vs.nRE   = "ignore",
      check.nlev.gtr.1    = "ignore",
      check.nobs.vs.rankZ = "ignore"
    )
  )
  if (!is.null(subset))    lf_args$subset    <- subset
  if (!is.null(weights))   lf_args$weights   <- weights
  if (!is.null(contrasts)) lf_args$contrasts <- contrasts

  lf <- do.call(lme4::lFormula, lf_args)

  # Extract components
  y       <- as.numeric(lf$fr[, 1])
  X       <- as.matrix(lf$X)
  Zt      <- lf$reTrms$Zt
  Lambdat <- lf$reTrms$Lambdat
  Lind    <- as.integer(lf$reTrms$Lind)
  theta0  <- as.numeric(lf$reTrms$theta)
  lower   <- as.numeric(lf$reTrms$lower)
  Gp      <- as.integer(lf$reTrms$Gp)

  # PCG threshold from control
  pcg_threshold <- if (!is.null(control$pcg_threshold)) {
    as.integer(control$pcg_threshold)
  } else {
    5000L
  }

  # --- Fit: dispatch to C++ or R optimizer ---
  if (control$optimizer == "L-BFGS-B") {
    # Phase 2/3: Full C++ L-BFGS-B with crossed RE support
    result <- C_fastmlm_fit(
      y, X, Zt, Lambdat, Lind, theta0, lower, Gp, REML,
      control$maxiter, control$ftol, control$gtol, pcg_threshold, verbose
    )

    # Name beta with column names from X
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
      grad_evals  = result$grad_evaluations,
      iterations  = result$iterations,
      is_crossed  = isTRUE(result$is_crossed),
      using_pcg   = isTRUE(result$using_pcg)
    )

    # Package into S4 object
    methods::new("fmlmMod",
      call      = mc,
      formula   = formula,
      frame     = lf$fr,
      flist     = lf$reTrms$flist,
      cnms      = lf$reTrms$cnms,
      Gp        = as.integer(lf$reTrms$Gp),
      lower     = lower,
      theta     = as.numeric(result$theta),
      beta      = as.numeric(result$beta),
      u         = as.numeric(result$u),
      sigma     = as.numeric(result$sigma),
      deviance  = as.numeric(result$deviance),
      REML      = REML,
      vcov_beta = as.matrix(result$vcov_beta),
      optinfo   = optinfo,
      X         = X,
      Zt        = Zt,
      Lambdat   = Lambdat,
      Lind      = Lind,
      pp_xptr   = new("externalptr")
    )
  } else {
    # Phase 1 fallback: R-side optim with C++ deviance callback
    pp <- C_fastmlm_create(y, X, Zt, Lambdat, Lind, lower, REML)

    devfun <- function(theta) {
      C_fastmlm_deviance(pp, theta)
    }

    opt <- if (requireNamespace("nloptr", quietly = TRUE)) {
      res <- nloptr::bobyqa(
        x0    = theta0,
        fn    = devfun,
        lower = lower,
        upper = rep(Inf, length(theta0)),
        control = list(maxeval = control$maxiter, xtol_rel = 1e-8)
      )
      list(par = res$par, value = res$value, convergence = res$convergence,
           counts = c("function" = res$iter), message = res$message)
    } else {
      warning("nloptr not available; using Nelder-Mead (ignoring lower bounds)")
      stats::optim(par = theta0, fn = devfun, method = "Nelder-Mead",
                   control = list(maxit = control$maxiter))
    }

    devfun(opt$par)
    result <- C_fastmlm_result(pp)

    names_beta <- colnames(X)
    if (!is.null(names_beta)) {
      names(result$beta) <- names_beta
      rownames(result$vcov_beta) <- names_beta
      colnames(result$vcov_beta) <- names_beta
    }

    optinfo <- list(
      optimizer   = control$optimizer,
      theta       = opt$par,
      deviance    = opt$value,
      convergence = opt$convergence,
      message     = if (!is.null(opt$message)) opt$message else "converged",
      fn_evals    = if (!is.null(opt$counts)) opt$counts["function"] else NA_integer_
    )

    methods::new("fmlmMod",
      call      = mc,
      formula   = formula,
      frame     = lf$fr,
      flist     = lf$reTrms$flist,
      cnms      = lf$reTrms$cnms,
      Gp        = as.integer(lf$reTrms$Gp),
      lower     = lower,
      theta     = as.numeric(opt$par),
      beta      = as.numeric(result$beta),
      u         = as.numeric(result$u),
      sigma     = as.numeric(result$sigma),
      deviance  = as.numeric(result$deviance),
      REML      = REML,
      vcov_beta = as.matrix(result$vcov_beta),
      optinfo   = optinfo,
      X         = X,
      Zt        = Zt,
      Lambdat   = Lambdat,
      Lind      = Lind,
      pp_xptr   = pp
    )
  }
}
