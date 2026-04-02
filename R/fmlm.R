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

  # --- Parse formula: try cache → fast parser → lme4 fallback ---
  cached <- cache_get(formula, data)
  if (!is.null(cached)) {
    lf <- cached
  } else {
    lf <- tryCatch(
      fast_lFormula(formula, data, REML = REML,
                    na.action = na.action, contrasts = contrasts),
      error = function(e) {
        if (!requireNamespace("lme4", quietly = TRUE)) {
          stop("Formula parsing failed and lme4 is not installed for fallback: ",
               conditionMessage(e), call. = FALSE)
        }
        if (verbose > 0L) {
          message("fastmlm: fast parser failed (", conditionMessage(e),
                  "), falling back to lme4::lFormula")
        }
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
        lf_raw <- do.call(lme4::lFormula, lf_args)
        # Normalise to same structure as fast_lFormula output
        list(
          fr      = lf_raw$fr,
          X       = lf_raw$X,
          Zt      = lf_raw$reTrms$Zt,
          Lambdat = lf_raw$reTrms$Lambdat,
          Lind    = as.integer(lf_raw$reTrms$Lind),
          theta   = as.numeric(lf_raw$reTrms$theta),
          lower   = as.numeric(lf_raw$reTrms$lower),
          flist   = lf_raw$reTrms$flist,
          cnms    = lf_raw$reTrms$cnms,
          Gp      = as.integer(lf_raw$reTrms$Gp)
        )
      }
    )
    cache_put(formula, data, lf)
  }

  # Extract components
  y       <- as.numeric(lf$fr[, 1])
  X       <- as.matrix(lf$X)

  # Apply prior weights if specified
  if (!is.null(weights)) {
    if (is.character(weights)) weights <- lf$fr[[weights]]
    w <- as.numeric(weights)
    if (length(w) != length(y)) stop("weights must have same length as data")
    if (any(w < 0)) stop("weights must be non-negative")
    sqrtw <- sqrt(w)
    y <- y * sqrtw
    X <- X * sqrtw
    # Scale Z columns too
    # Zt is q x n, scale columns (observations) by sqrtw
    Zt_scaled <- lf$Zt
    for (j in seq_along(sqrtw)) {
      # Scale column j of Z (= row j of Zt... but Zt is CSC so column j)
      # Actually Zt is stored as dgCMatrix (CSC). Column j of Zt corresponds
      # to observation j. We need to scale the values in column j.
      idx_start <- Zt_scaled@p[j] + 1L
      idx_end <- Zt_scaled@p[j + 1L]
      if (idx_end >= idx_start) {
        Zt_scaled@x[idx_start:idx_end] <- Zt_scaled@x[idx_start:idx_end] * sqrtw[j]
      }
    }
    lf$Zt <- Zt_scaled
  }
  Zt      <- lf$Zt
  Lambdat <- lf$Lambdat
  Lind    <- as.integer(lf$Lind)
  theta0  <- as.numeric(lf$theta)
  lower   <- as.numeric(lf$lower)
  Gp      <- as.integer(lf$Gp)

  # Extract rcs knots from the model matrix for use in predict()
  rcs_knots <- list()
  if (requireNamespace("rms", quietly = TRUE)) {
    trm_labels <- attr(stats::terms(remove_bars(formula), data = lf$fr),
                       "term.labels")
    for (lab in trm_labels) {
      m <- regmatches(lab, regexec("^rcs\\(([^,]+),\\s*([0-9]+)\\)$", lab))[[1]]
      if (length(m) == 3) {
        varname <- m[2]
        # Get raw variable from the original data (not model frame)
        raw_var <- tryCatch(data[[varname]], error = function(e) NULL)
        if (!is.null(raw_var) && is.numeric(raw_var)) {
          knots <- tryCatch({
            b <- rms::rcs(raw_var, as.integer(m[3]))
            attr(b, "parms")
          }, error = function(e) NULL)
          if (!is.null(knots)) rcs_knots[[varname]] <- knots
        }
      }
    }
  }

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
      using_pcg   = isTRUE(result$using_pcg),
      rcs_knots   = rcs_knots
    )

    # Package into S4 object
    methods::new("fmlmMod",
      call      = mc,
      formula   = formula,
      frame     = lf$fr,
      flist     = lf$flist,
      cnms      = lf$cnms,
      Gp        = as.integer(lf$Gp),
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
      fn_evals    = if (!is.null(opt$counts)) opt$counts["function"] else NA_integer_,
      rcs_knots   = rcs_knots
    )

    methods::new("fmlmMod",
      call      = mc,
      formula   = formula,
      frame     = lf$fr,
      flist     = lf$flist,
      cnms      = lf$cnms,
      Gp        = as.integer(lf$Gp),
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
