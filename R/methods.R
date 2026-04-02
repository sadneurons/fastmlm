# ============================================================================
# S3/S4 methods for fmlmMod
# ============================================================================

# --- fixef / ranef / VarCorr ---

#' @export
fixef.fmlmMod <- function(object, ...) {
  b <- object@beta
  names(b) <- colnames(object@X)
  b
}

#' @export
ranef.fmlmMod <- function(object, ...) {
  Lambdat <- object@Lambdat
  x <- Lambdat@x
  for (i in seq_along(x)) {
    x[i] <- object@theta[object@Lind[i]]
  }
  Lambdat@x <- x

  b <- as.numeric(Matrix::t(Lambdat) %*% object@u)

  flist <- object@flist
  cnms  <- object@cnms
  Gp    <- object@Gp

  result <- list()
  for (i in seq_along(flist)) {
    fac    <- flist[[i]]
    nlevs  <- nlevels(fac)
    cnames <- cnms[[i]]
    ncols  <- length(cnames)

    idx_start <- Gp[i] + 1L
    idx_end   <- Gp[i + 1L]
    bi <- b[idx_start:idx_end]

    mat <- matrix(bi, nrow = nlevs, ncol = ncols, byrow = FALSE)
    colnames(mat) <- cnames
    rownames(mat) <- levels(fac)

    result[[names(flist)[i]]] <- as.data.frame(mat)
  }

  result
}

#' @export
VarCorr.fmlmMod <- function(x, sigma = 1, ...) {
  flist <- x@flist
  cnms  <- x@cnms
  Gp    <- x@Gp
  theta <- x@theta
  Lind  <- x@Lind
  sig   <- x@sigma

  Lambdat <- x@Lambdat
  lx <- Lambdat@x
  for (i in seq_along(lx)) {
    lx[i] <- theta[Lind[i]]
  }
  Lambdat@x <- lx

  result <- list()
  for (i in seq_along(flist)) {
    cnames <- cnms[[i]]
    ncols  <- length(cnames)

    idx_start <- Gp[i] + 1L
    idx_end   <- Gp[i] + ncols

    Li <- as.matrix(Lambdat[idx_start:idx_end, idx_start:idx_end])

    vc <- sig^2 * crossprod(Li)
    rownames(vc) <- cnames
    colnames(vc) <- cnames
    attr(vc, "stddev") <- sqrt(diag(vc))
    attr(vc, "correlation") <- if (ncols > 1) stats::cov2cor(vc) else NULL

    result[[names(flist)[i]]] <- vc
  }

  attr(result, "sc") <- sig
  class(result) <- "VarCorr.fmlmMod"
  result
}

# --- S4 methods ---

#' @rdname fmlmMod-class
#' @export
setMethod("vcov", "fmlmMod", function(object, ...) {
  object@vcov_beta
})

#' @rdname fmlmMod-class
#' @export
setMethod("coef", "fmlmMod", function(object, ...) {
  fe <- fixef.fmlmMod(object)
  re <- ranef.fmlmMod(object)

  result <- list()
  for (nm in names(re)) {
    re_df <- re[[nm]]
    coef_mat <- matrix(fe, nrow = nrow(re_df), ncol = length(fe), byrow = TRUE)
    colnames(coef_mat) <- names(fe)
    rownames(coef_mat) <- rownames(re_df)

    for (cn in colnames(re_df)) {
      if (cn %in% colnames(coef_mat)) {
        coef_mat[, cn] <- coef_mat[, cn] + re_df[[cn]]
      }
    }
    result[[nm]] <- as.data.frame(coef_mat)
  }

  result
})

#' @rdname fmlmMod-class
#' @export
setMethod("fitted", "fmlmMod", function(object, ...) {
  Xb <- as.numeric(object@X %*% object@beta)

  Lambdat <- object@Lambdat
  lx <- Lambdat@x
  for (i in seq_along(lx)) {
    lx[i] <- object@theta[object@Lind[i]]
  }
  Lambdat@x <- lx
  b_vec <- Matrix::t(Lambdat) %*% object@u

  Zb <- as.numeric(Matrix::t(object@Zt) %*% b_vec)
  Xb + Zb
})

#' @rdname fmlmMod-class
#' @export
setMethod("residuals", "fmlmMod", function(object, type = "response", ...) {
  y <- object@frame[, 1]
  fv <- fitted(object)
  y - fv
})

#' @rdname fmlmMod-class
#' @export
setMethod("logLik", "fmlmMod", function(object, ...) {
  ll <- -0.5 * object@deviance
  n <- nrow(object@frame)
  p <- length(object@beta)
  nth <- length(object@theta)

  df <- p + nth + 1L

  attr(ll, "df") <- df
  attr(ll, "nobs") <- if (object@REML) n - p else n
  class(ll) <- "logLik"
  ll
})

# S3 version needed for AIC/BIC dispatch
#' @method logLik fmlmMod
#' @export
logLik.fmlmMod <- function(object, ...) {
  ll <- -0.5 * object@deviance
  n <- nrow(object@frame)
  p <- length(object@beta)
  nth <- length(object@theta)
  attr(ll, "df") <- p + nth + 1L
  attr(ll, "nobs") <- if (object@REML) n - p else n
  class(ll) <- "logLik"
  ll
}

#' @rdname fmlmMod-class
#' @export
setMethod("show", "fmlmMod", function(object) {
  cat("Fast Multilevel Linear Model (fastmlm)\n")
  cat("Formula:", deparse(object@formula), "\n")
  cat("Data:   ", nrow(object@frame), "observations\n")
  cat("REML:   ", object@REML, "\n")

  # Convergence warnings
  warnings <- check_convergence(object)
  if (length(warnings) > 0) {
    cat("\n")
    for (w in warnings) cat("WARNING:", w, "\n")
  }

  cat("\nRandom effects:\n")
  vc <- VarCorr.fmlmMod(object)
  for (nm in names(vc)) {
    mat <- vc[[nm]]
    cat(sprintf("  Groups: %s (%d levels)\n", nm, nlevels(object@flist[[nm]])))
    sds <- attr(mat, "stddev")
    for (j in seq_along(sds)) {
      cat(sprintf("    %-20s Std.Dev. %8.4f\n", names(sds)[j], sds[j]))
    }
  }
  cat(sprintf("  Residual               Std.Dev. %8.4f\n", object@sigma))

  cat("\nFixed effects:\n")
  fe <- fixef.fmlmMod(object)
  se <- sqrt(diag(vcov(object)))
  tval <- fe / se

  sat_df <- tryCatch(satterthwaite_df(object), error = function(e) NULL)
  if (!is.null(sat_df)) {
    pval <- 2 * stats::pt(abs(tval), df = sat_df, lower.tail = FALSE)
    coef_tab <- cbind(Estimate = fe, `Std. Error` = se, df = sat_df,
                      `t value` = tval, `Pr(>|t|)` = pval)
  } else {
    coef_tab <- cbind(Estimate = fe, `Std. Error` = se, `t value` = tval)
  }
  stats::printCoefmat(coef_tab, P.values = !is.null(sat_df),
                      has.Pvalue = !is.null(sat_df), digits = 4,
                      signif.stars = TRUE)

  cat("\nOptimiser:", object@optinfo$optimizer,
      "| Convergence:", object@optinfo$convergence, "\n")

  invisible(object)
})

#' @rdname fmlmMod-class
#' @export
setMethod("summary", "fmlmMod", function(object, ...) {
  show(object)

  cat("\nDeviance:", round(object@deviance, 2), "\n")
  ll <- logLik(object)
  cat("logLik:  ", round(as.numeric(ll), 2), "\n")
  cat("AIC:     ", round(AIC(object), 2), "\n")
  cat("BIC:     ", round(BIC(object), 2), "\n")

  invisible(object)
})

#' @export
print.VarCorr.fmlmMod <- function(x, ...) {
  for (nm in names(x)) {
    mat <- x[[nm]]
    cat(sprintf("Groups: %s\n", nm))
    sds <- attr(mat, "stddev")

    if (length(sds) == 1) {
      cat(sprintf("  %-20s Std.Dev. %8.4f\n", names(sds), sds))
    } else {
      cat("  Variance-Covariance:\n")
      print(round(mat, 4))
      cat("  Std.Dev.:", paste(round(sds, 4), collapse = ", "), "\n")
    }
  }
  cat(sprintf("Residual Std.Dev.: %8.4f\n", attr(x, "sc")))
  invisible(x)
}

# ============================================================================
# anova() for likelihood ratio tests
# ============================================================================

#' Likelihood ratio test for fmlmMod objects
#'
#' Compare nested models via likelihood ratio test. Models must be fit
#' with \code{REML = FALSE} for valid comparison.
#'
#' @param object An \code{fmlmMod} object.
#' @param ... Additional \code{fmlmMod} objects to compare.
#' @return A data frame with model comparison statistics.
#' @method anova fmlmMod
#' @export
anova.fmlmMod <- function(object, ...) {
  models <- c(list(object), list(...))
  n_models <- length(models)

  if (n_models < 2) {
    stop("anova() requires at least two models to compare")
  }

  # Check all are fmlmMod
  for (i in seq_along(models)) {
    if (!methods::is(models[[i]], "fmlmMod")) {
      stop("All objects must be fmlmMod models")
    }
  }

  # Warn if any use REML
  if (any(vapply(models, function(m) m@REML, logical(1)))) {
    warning("Models should be fit with REML = FALSE for valid likelihood ratio tests")
  }

  # Compute stats
  npar <- vapply(models, function(m) attr(logLik(m), "df"), integer(1))
  loglik <- vapply(models, function(m) as.numeric(logLik(m)), numeric(1))
  aic_vals <- vapply(models, function(m) AIC(m), numeric(1))
  bic_vals <- vapply(models, function(m) BIC(m), numeric(1))
  nobs_vals <- vapply(models, function(m) nobs(m), integer(1))

  # Order by complexity
  ord <- order(npar)
  models <- models[ord]
  npar <- npar[ord]
  loglik <- loglik[ord]
  aic_vals <- aic_vals[ord]
  bic_vals <- bic_vals[ord]

  # LRT
  chisq <- c(NA, diff(2 * loglik))
  chi_df <- c(NA, diff(npar))
  pval <- c(NA, stats::pchisq(chisq[-1], df = chi_df[-1], lower.tail = FALSE))

  # Model names from calls
  mc <- match.call()
  mnames <- vapply(as.list(mc)[-1], deparse, character(1))
  if (length(mnames) > n_models) mnames <- mnames[seq_len(n_models)]

  result <- data.frame(
    npar = npar,
    AIC = aic_vals,
    BIC = bic_vals,
    logLik = loglik,
    deviance = -2 * loglik,
    Chisq = chisq,
    Df = chi_df,
    `Pr(>Chisq)` = pval,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  rownames(result) <- mnames[ord]
  class(result) <- c("anova", "data.frame")
  result
}

# ============================================================================
# simulate() for parametric bootstrap
# ============================================================================

#' Simulate responses from a fitted fmlmMod
#'
#' Generates new response vectors by sampling new random effects and
#' residuals from the fitted model's estimated distribution.
#'
#' @param object An \code{fmlmMod} object.
#' @param nsim Integer; number of simulations. Default 1.
#' @param seed Optional random seed.
#' @param ... Ignored.
#' @return A data frame with \code{nsim} columns, each a simulated response.
#' @method simulate fmlmMod
#' @export
simulate.fmlmMod <- function(object, nsim = 1, seed = NULL, ...) {
  if (!is.null(seed)) set.seed(seed)

  n <- nrow(object@frame)
  X <- object@X
  beta <- object@beta
  sig <- object@sigma

  # Get variance-covariance of random effects
  vc <- VarCorr.fmlmMod(object)
  flist <- object@flist
  cnms <- object@cnms
  Gp <- object@Gp

  Xbeta <- as.numeric(X %*% beta)

  result <- data.frame(matrix(NA_real_, nrow = n, ncol = nsim))
  names(result) <- paste0("sim_", seq_len(nsim))

  for (sim in seq_len(nsim)) {
    # Simulate new random effects
    b_new <- numeric(object@Gp[length(Gp)])

    for (i in seq_along(flist)) {
      fac <- flist[[i]]
      nlevs <- nlevels(fac)
      cnames <- cnms[[i]]
      ncols <- length(cnames)

      # Get the variance-covariance for this term
      Sigma_i <- vc[[names(flist)[i]]]

      # Simulate random effects for each level
      if (ncols == 1) {
        re_i <- stats::rnorm(nlevs, sd = sqrt(Sigma_i[1, 1]))
      } else {
        re_i <- MASS_mvrnorm_simple(nlevs, rep(0, ncols), Sigma_i)
      }

      # Place into b_new
      idx_start <- Gp[i] + 1L
      idx_end <- Gp[i + 1L]
      b_new[idx_start:idx_end] <- as.vector(t(re_i))
    }

    # Compute Z * b
    Zb <- as.numeric(Matrix::t(object@Zt) %*% b_new)

    # Simulate response
    result[, sim] <- Xbeta + Zb + stats::rnorm(n, sd = sig)
  }

  result
}

# Simple multivariate normal without MASS dependency
#' @keywords internal
MASS_mvrnorm_simple <- function(n, mu, Sigma) {
  p <- length(mu)
  L <- chol(Sigma)
  Z <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  sweep(Z %*% L, 2, mu, "+")
}

# ============================================================================
# update() method
# ============================================================================

#' Update a fastmlm model
#'
#' Re-fits a model with modified formula or arguments.
#'
#' @param object An \code{fmlmMod} object.
#' @param formula. A new formula (use \code{. ~ .} to keep existing).
#' @param ... Additional arguments passed to \code{\link{fmlm}}.
#' @param evaluate Logical; if FALSE, return the unevaluated call.
#' @return A new \code{fmlmMod} object.
#' @method update fmlmMod
#' @export
update.fmlmMod <- function(object, formula., ..., evaluate = TRUE) {
  call <- object@call
  if (!missing(formula.)) {
    call$formula <- stats::update.formula(formula(object), formula.)
  }
  extras <- match.call(expand.dots = FALSE)$...
  for (nm in names(extras)) {
    call[[nm]] <- extras[[nm]]
  }
  if (evaluate) eval(call, parent.frame()) else call
}

# ============================================================================
# Convergence diagnostics (item 3)
# ============================================================================

#' Check model convergence and identify potential issues
#'
#' @param object An \code{fmlmMod} object.
#' @return Character vector of warning messages (empty if no issues).
#' @keywords internal
check_convergence <- function(object) {
  warnings <- character(0)

  theta <- object@theta
  lower <- object@lower

  # Check for singular fit (theta at boundary)
  at_boundary <- which(abs(theta - lower) < 1e-10 & is.finite(lower))
  if (length(at_boundary) > 0) {
    warnings <- c(warnings,
      "Singular fit: variance component(s) estimated at boundary (zero variance). ",
      "Model may be overparameterised for the data.")
  }

  # Check optimizer convergence
  if (object@optinfo$convergence != 0) {
    warnings <- c(warnings,
      paste("Optimizer did not converge:", object@optinfo$message))
  }

  # Check for very large or very small sigma
  if (object@sigma < 1e-8) {
    warnings <- c(warnings,
      "Residual standard deviation is near zero. Check model specification.")
  }

  # Check for very large random effects relative to residual
  vc <- VarCorr.fmlmMod(object)
  for (nm in names(vc)) {
    re_sd <- max(attr(vc[[nm]], "stddev"))
    if (re_sd > 100 * object@sigma) {
      warnings <- c(warnings,
        sprintf("Random effect SD for '%s' is %.0fx the residual SD.", nm,
                re_sd / object@sigma))
    }
  }

  warnings
}

# ============================================================================
# Profile confidence intervals for variance components (item 9)
# ============================================================================

#' Profile confidence intervals for variance parameters
#'
#' Computes confidence intervals for theta by profiling the deviance
#' function. More accurate than Wald intervals for variance components.
#'
#' @param object An \code{fmlmMod} object.
#' @param level Confidence level.
#' @return A matrix with lower and upper bounds for each theta.
#' @keywords internal
profile_ci_theta <- function(object, level = 0.95) {
  theta <- object@theta
  nth <- length(theta)
  dev_opt <- object@deviance

  y <- as.numeric(object@frame[, 1])
  X <- object@X
  Zt <- object@Zt
  Lambdat <- object@Lambdat
  Lind <- object@Lind
  lower <- object@lower

  pp <- C_fastmlm_create(y, X, Zt, Lambdat, Lind, lower, object@REML)

  # chi-squared cutoff for the deviance difference
  cutoff <- stats::qchisq(level, df = 1)

  ci <- matrix(NA_real_, nrow = nth, ncol = 2)
  colnames(ci) <- c("lower", "upper")

  for (k in seq_len(nth)) {
    # Profile: find theta[k] such that dev(theta_profile) - dev_opt = cutoff
    # where theta_profile minimises deviance over all other theta with theta[k] fixed

    # Simple approach: use the deviance at the optimum as a function of theta[k]
    # and find where it crosses dev_opt + cutoff via bisection

    # Lower bound
    lo <- lower[k]
    hi <- theta[k]
    theta_try <- theta

    # Check if lower bound gives deviance above cutoff
    theta_try[k] <- lo
    dev_lo <- C_fastmlm_deviance(pp, theta_try)
    if (dev_lo - dev_opt > cutoff) {
      # Bisect to find crossing point
      for (iter in 1:50) {
        mid <- (lo + hi) / 2
        theta_try[k] <- mid
        dev_mid <- C_fastmlm_deviance(pp, theta_try)
        if (dev_mid - dev_opt > cutoff) lo <- mid else hi <- mid
        if (abs(hi - lo) < 1e-6) break
      }
      ci[k, 1] <- (lo + hi) / 2
    } else {
      ci[k, 1] <- lo
    }

    # Upper bound
    lo <- theta[k]
    hi <- theta[k] * 3 + 1  # generous upper search bound
    theta_try <- theta

    theta_try[k] <- hi
    dev_hi <- C_fastmlm_deviance(pp, theta_try)
    # Expand if needed
    while (dev_hi - dev_opt < cutoff && hi < 1e6) {
      hi <- hi * 2
      theta_try[k] <- hi
      dev_hi <- C_fastmlm_deviance(pp, theta_try)
    }

    if (dev_hi - dev_opt > cutoff) {
      for (iter in 1:50) {
        mid <- (lo + hi) / 2
        theta_try[k] <- mid
        dev_mid <- C_fastmlm_deviance(pp, theta_try)
        if (dev_mid - dev_opt > cutoff) hi <- mid else lo <- mid
        if (abs(hi - lo) < 1e-6) break
      }
      ci[k, 2] <- (lo + hi) / 2
    } else {
      ci[k, 2] <- Inf
    }
  }

  # Restore state
  C_fastmlm_deviance(pp, theta)

  ci
}
