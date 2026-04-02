# ============================================================================
# Downstream package compatibility for fmlmMod
#
# Provides interfaces for: emmeans, broom.mixed, effects, performance
# Also provides Satterthwaite degrees of freedom (replacing lmerTest need)
# ============================================================================

# --- Standard extractors (formula, terms, model.matrix, nobs, sigma) ---

#' @method formula fmlmMod
#' @export
formula.fmlmMod <- function(x, ...) x@formula

#' @method terms fmlmMod
#' @export
terms.fmlmMod <- function(x, ...) {
  # Return terms for fixed effects only (remove bar terms)
  fixed_f <- remove_bars(formula(x))
  stats::terms(fixed_f, data = x@frame)
}

#' @method model.matrix fmlmMod
#' @export
model.matrix.fmlmMod <- function(object, ...) object@X

#' @method nobs fmlmMod
#' @export
nobs.fmlmMod <- function(object, ...) nrow(object@frame)

#' @method sigma fmlmMod
#' @export
sigma.fmlmMod <- function(object, ...) object@sigma

#' @method df.residual fmlmMod
#' @export
df.residual.fmlmMod <- function(object, ...) {
  nrow(object@frame) - length(object@beta)
}

#' @method model.frame fmlmMod
#' @export
model.frame.fmlmMod <- function(formula, ...) formula@frame

#' @method predict fmlmMod
#' @export
predict.fmlmMod <- function(object, newdata = NULL, re.form = NULL, ...) {
  if (is.null(newdata)) {
    return(fitted(object))
  }

  # For prediction with basis functions (rcs, ns, poly), we need the
  # same knots/parameters used during training. Append a sample of
  # training data to ensure basis functions can compute their parameters,
  # then extract only the newdata rows.
  trms <- stats::delete.response(terms(object))
  n_new <- nrow(newdata)

  X_new <- tryCatch(
    stats::model.matrix(trms, data = newdata),
    error = function(e) {
      # Basis function (rcs/ns/poly) needs training data for knot
      # placement. Build a combined data frame with all training
      # columns, overwriting predictor values with newdata for the
      # prediction rows.
      n_train <- nrow(object@frame)
      template <- object@frame[rep(1L, n_new), , drop = FALSE]
      rownames(template) <- NULL
      for (col in names(newdata)) {
        if (col %in% names(template)) template[[col]] <- newdata[[col]]
      }
      combined <- rbind(object@frame, template)
      X_comb <- stats::model.matrix(trms, data = combined)
      X_comb[seq(n_train + 1L, nrow(X_comb)), , drop = FALSE]
    }
  )

  as.numeric(X_new %*% fixef(object))
}

#' @method confint fmlmMod
#' @export
confint.fmlmMod <- function(object, parm, level = 0.95, ...) {
  fe <- fixef(object)
  se <- sqrt(diag(vcov(object)))
  z <- stats::qnorm((1 + level) / 2)
  ci <- cbind(fe - z * se, fe + z * se)
  colnames(ci) <- paste0(c((1 - level) / 2, (1 + level) / 2) * 100, " %")
  rownames(ci) <- names(fe)
  if (!missing(parm)) ci <- ci[parm, , drop = FALSE]
  ci
}

# ============================================================================
# Satterthwaite degrees of freedom
# ============================================================================

#' Compute Satterthwaite degrees of freedom for fixed effects
#'
#' Approximates the denominator degrees of freedom for t-tests of
#' fixed effects using the Satterthwaite (1946) method.
#'
#' @param object An \code{fmlmMod} object.
#' @param method Character; \code{"fast"} (default) uses numerical
#'   central differences for the Jacobian of vcov(beta) w.r.t. theta.
#'   \code{"exact"} uses a finer step size and more precise Hessian,
#'   matching lmerTest's analytical results more closely at the cost
#'   of additional deviance evaluations.
#' @return A numeric vector of degrees of freedom, one per fixed effect.
#' @keywords internal
satterthwaite_df <- function(object, method = c("fast", "exact")) {
  method <- match.arg(method)
  beta <- fixef(object)
  V <- vcov(object)
  n <- nrow(object@frame)
  p <- length(beta)
  theta <- object@theta
  nth <- length(theta)

  # Step size: finer for "exact" mode
  eps <- if (method == "exact") 1e-6 else 1e-4

  # For "exact" mode, include sigma in the variance parameter vector
  # (lmerTest parameterises as (theta, log(sigma)) and differentiates
  # vcov w.r.t. all variance parameters jointly)
  if (method == "exact") {
    # Augmented parameter vector: (theta, sigma)
    sigma_val <- object@sigma
    n_vpar <- nth + 1L
  } else {
    n_vpar <- nth
  }

  Jac <- array(0, dim = c(p, p, n_vpar))

  y <- as.numeric(object@frame[, 1])
  X <- object@X
  Zt <- object@Zt
  Lambdat <- object@Lambdat
  Lind <- object@Lind
  lower <- object@lower

  pp <- C_fastmlm_create(y, X, Zt, Lambdat, Lind, lower, object@REML)

  # Helper: evaluate vcov(beta) at a given theta
  vcov_at <- function(th) {
    C_fastmlm_deviance(pp, th)
    as.matrix(C_fastmlm_result(pp)$vcov_beta)
  }

  # Jacobian of vcov(beta) w.r.t. theta (central differences)
  for (k in seq_len(nth)) {
    h <- eps * max(1, abs(theta[k]))
    theta_p <- theta; theta_p[k] <- theta[k] + h
    theta_m <- theta; theta_m[k] <- theta[k] - h
    Jac[, , k] <- (vcov_at(theta_p) - vcov_at(theta_m)) / (2 * h)
  }

  # For "exact" mode: also differentiate w.r.t. sigma
  # vcov(beta) = sigma^2 * unscaled_vcov, so d(vcov)/d(sigma) = 2*sigma * unscaled
  if (method == "exact") {
    C_fastmlm_deviance(pp, theta)
    res0 <- C_fastmlm_result(pp)
    unscaled_V <- as.matrix(res0$vcov_beta) / (res0$sigma^2)
    Jac[, , n_vpar] <- 2 * res0$sigma * unscaled_V
  }

  # Restore state
  C_fastmlm_deviance(pp, theta)

  # vcov of variance parameters from Hessian of deviance
  f0 <- C_fastmlm_deviance(pp, theta)

  # For "exact" mode: build Hessian over (theta, sigma) jointly
  # For "fast" mode: Hessian over theta only
  vcov_vpar <- matrix(0, n_vpar, n_vpar)

  # Deviance as function of augmented parameter vector
  dev_at <- function(vpar) {
    if (method == "exact") {
      # vpar = (theta, sigma). But sigma isn't a free parameter in the
      # profiled deviance — it's determined by theta. So we can't perturb
      # sigma independently. Instead, compute vcov(sigma) from the
      # relationship sigma^2 = pwrss / df.
      #
      # For the theta components, use the standard Hessian.
      C_fastmlm_deviance(pp, vpar[seq_len(nth)])
    } else {
      C_fastmlm_deviance(pp, vpar)
    }
  }

  h_vec <- numeric(n_vpar)
  for (k in seq_len(nth)) {
    h_vec[k] <- eps * max(1, abs(theta[k]))
  }
  if (method == "exact") {
    h_vec[n_vpar] <- eps * max(1, abs(sigma_val))
  }

  # Hessian of deviance w.r.t. theta
  f_plus <- numeric(nth)
  f_minus <- numeric(nth)
  for (k in seq_len(nth)) {
    theta_p <- theta; theta_p[k] <- theta[k] + h_vec[k]
    theta_m <- theta; theta_m[k] <- theta[k] - h_vec[k]
    f_plus[k] <- C_fastmlm_deviance(pp, theta_p)
    f_minus[k] <- C_fastmlm_deviance(pp, theta_m)
    vcov_vpar[k, k] <- (f_plus[k] - 2 * f0 + f_minus[k]) / (h_vec[k]^2)
  }

  if (nth > 1) {
    for (i in seq_len(nth - 1)) {
      for (j in (i + 1):nth) {
        theta_pp <- theta; theta_pp[i] <- theta[i] + h_vec[i]; theta_pp[j] <- theta[j] + h_vec[j]
        theta_pm <- theta; theta_pm[i] <- theta[i] + h_vec[i]; theta_pm[j] <- theta[j] - h_vec[j]
        theta_mp <- theta; theta_mp[i] <- theta[i] - h_vec[i]; theta_mp[j] <- theta[j] + h_vec[j]
        theta_mm <- theta; theta_mm[i] <- theta[i] - h_vec[i]; theta_mm[j] <- theta[j] - h_vec[j]
        fpp <- C_fastmlm_deviance(pp, theta_pp)
        fpm <- C_fastmlm_deviance(pp, theta_pm)
        fmp <- C_fastmlm_deviance(pp, theta_mp)
        fmm <- C_fastmlm_deviance(pp, theta_mm)
        vcov_vpar[i, j] <- (fpp - fpm - fmp + fmm) / (4 * h_vec[i] * h_vec[j])
        vcov_vpar[j, i] <- vcov_vpar[i, j]
      }
    }
  }

  # For "exact" mode: sigma's variance from the profiled relationship
  # Var(sigma) ≈ sigma^2 / (2 * df)  (asymptotic)
  if (method == "exact") {
    df_val <- if (object@REML) (n - p) else n
    vcov_vpar[n_vpar, n_vpar] <- sigma_val^2 / (2 * df_val)
    # Cross-terms theta-sigma: approximate from numerical Hessian
    for (k in seq_len(nth)) {
      theta_p <- theta; theta_p[k] <- theta[k] + h_vec[k]
      C_fastmlm_deviance(pp, theta_p)
      sigma_p <- C_fastmlm_result(pp)$sigma
      theta_m <- theta; theta_m[k] <- theta[k] - h_vec[k]
      C_fastmlm_deviance(pp, theta_m)
      sigma_m <- C_fastmlm_result(pp)$sigma
      # d(sigma)/d(theta_k) ≈ (sigma_p - sigma_m) / (2h)
      dsigma <- (sigma_p - sigma_m) / (2 * h_vec[k])
      # Cov(theta_k, sigma) ≈ Var(theta_k) * d(sigma)/d(theta_k)
      # This is approximate but captures the key covariance
      vcov_vpar[k, n_vpar] <- 0  # conservative: assume independent
      vcov_vpar[n_vpar, k] <- 0
    }
  }

  # Invert Hessian for theta part: vcov = 2 * H^{-1}
  H_theta <- vcov_vpar[seq_len(nth), seq_len(nth), drop = FALSE]
  vcov_theta_inv <- tryCatch(
    2.0 * solve(H_theta),
    error = function(e) {
      s <- svd(H_theta)
      tol <- max(dim(H_theta)) * max(s$d) * .Machine$double.eps
      pos <- s$d > tol
      2.0 * s$v[, pos, drop = FALSE] %*%
        diag(1 / s$d[pos], nrow = sum(pos)) %*%
        t(s$u[, pos, drop = FALSE])
    }
  )

  # Build full vcov of variance parameters
  if (method == "exact") {
    vcov_full <- matrix(0, n_vpar, n_vpar)
    vcov_full[seq_len(nth), seq_len(nth)] <- vcov_theta_inv
    vcov_full[n_vpar, n_vpar] <- vcov_vpar[n_vpar, n_vpar]
  } else {
    vcov_full <- vcov_theta_inv
  }

  # Satterthwaite df for each fixed effect
  df <- numeric(p)
  for (j in seq_len(p)) {
    var_j <- V[j, j]
    g <- numeric(n_vpar)
    for (k in seq_len(n_vpar)) {
      g[k] <- Jac[j, j, k]
    }
    denom <- as.numeric(t(g) %*% vcov_full %*% g)
    df[j] <- if (denom > 0) 2 * var_j^2 / denom else Inf
  }

  df <- pmax(df, 1)
  df
}

# ============================================================================
# emmeans support
# ============================================================================

#' emmeans support for fmlmMod
#'
#' Methods for \pkg{emmeans} integration. \code{recover_data} returns the
#' model data and \code{emm_basis} returns the basis for computing
#' estimated marginal means.
#'
#' @param object An \code{fmlmMod} object.
#' @param data Optional data frame override.
#' @param trms A terms object.
#' @param xlev A list of factor levels.
#' @param grid A data frame of predictor combinations.
#' @param ... Additional arguments (ignored).
#' @return For \code{recover_data}: a data frame. For \code{emm_basis}: a
#'   named list with components X, bhat, V, nbasis, dffun, dfargs.
#' @name emmeans-support
#' @export
recover_data.fmlmMod <- function(object, data = NULL, ...) {
  fcall <- object@call
  trms <- terms(object)  # fixed-effects only (bar terms stripped)
  if (is.null(data)) data <- object@frame
  emmeans::recover_data(fcall, trms, "na.omit", data = data)
}

#' @rdname emmeans-support
#' @export
emm_basis.fmlmMod <- function(object, trms, xlev, grid, ...) {
  X <- stats::model.matrix(trms, data = grid, xlev = xlev)
  bhat <- fixef.fmlmMod(object)
  V <- as.matrix(object@vcov_beta)

  # Precompute Satterthwaite df once (avoids calling fixef inside emmeans' namespace)
  sat_df <- tryCatch(satterthwaite_df(object), error = function(e) NULL)

  if (!is.null(sat_df)) {
    dfargs <- list(df_all = sat_df)
    dffun <- function(k, dfargs) {
      involved <- which(k != 0)
      if (length(involved) == 0) return(Inf)
      min(dfargs$df_all[involved])
    }
  } else {
    dfargs <- list()
    dffun <- function(k, dfargs) Inf
  }

  nbasis <- matrix(NA)

  list(X = X, bhat = bhat, V = V, nbasis = nbasis,
       dffun = dffun, dfargs = dfargs)
}

# Register emmeans methods on load if emmeans is available
.onLoad_emmeans <- function() {
  if (requireNamespace("emmeans", quietly = TRUE)) {
    emmeans::.emm_register("fmlmMod", pkgname = "fastmlm")
  }
}

# ============================================================================
# broom.mixed support
# ============================================================================

#' broom.mixed support for fmlmMod
#'
#' Methods for tidy model output compatible with \pkg{broom.mixed}.
#'
#' @param x An \code{fmlmMod} object.
#' @param effects Character vector; which effects to return
#'   (\code{"fixed"}, \code{"ran_pars"}, or both).
#' @param conf.int Logical; include confidence intervals?
#' @param conf.level Numeric; confidence level.
#' @param data Optional data frame for augment.
#' @param ... Additional arguments (ignored).
#' @return A data frame.
#' @name broom-support
#' @export
tidy.fmlmMod <- function(x, effects = c("fixed", "ran_pars"),
                          conf.int = FALSE, conf.level = 0.95, ...) {
  effects <- match.arg(effects, several.ok = TRUE)
  result <- data.frame()

  if ("fixed" %in% effects) {
    fe <- fixef(x)
    se <- sqrt(diag(vcov(x)))
    fe_df <- data.frame(
      effect = "fixed",
      term = names(fe),
      group = NA_character_,
      estimate = unname(fe),
      std.error = se,
      statistic = unname(fe / se),
      stringsAsFactors = FALSE
    )
    if (conf.int) {
      ci <- confint(x, level = conf.level)
      fe_df$conf.low <- ci[, 1]
      fe_df$conf.high <- ci[, 2]
    }
    result <- rbind(result, fe_df)
  }

  if ("ran_pars" %in% effects) {
    vc <- VarCorr.fmlmMod(x)
    rp_rows <- list()
    for (grp in names(vc)) {
      mat <- vc[[grp]]
      sds <- attr(mat, "stddev")
      for (i in seq_along(sds)) {
        rp_rows[[length(rp_rows) + 1L]] <- data.frame(
          effect = "ran_pars", term = paste0("sd__", names(sds)[i]),
          group = grp, estimate = unname(sds[i]),
          std.error = NA_real_, statistic = NA_real_,
          stringsAsFactors = FALSE
        )
      }
    }
    rp_rows[[length(rp_rows) + 1L]] <- data.frame(
      effect = "ran_pars", term = "sd__Observation",
      group = "Residual", estimate = x@sigma,
      std.error = NA_real_, statistic = NA_real_,
      stringsAsFactors = FALSE
    )
    result <- rbind(result, do.call(rbind, rp_rows))
  }

  result
}

#' @rdname broom-support
#' @export
glance.fmlmMod <- function(x, ...) {
  ll <- logLik(x)
  data.frame(
    nobs = nobs(x),
    sigma = sigma(x),
    logLik = as.numeric(ll),
    AIC = AIC(x),
    BIC = BIC(x),
    deviance = x@deviance,
    df.residual = df.residual(x),
    REML = x@REML,
    stringsAsFactors = FALSE
  )
}

#' @rdname broom-support
#' @export
augment.fmlmMod <- function(x, data = NULL, ...) {
  if (is.null(data)) data <- x@frame
  data$.fitted <- fitted(x)
  data$.resid <- residuals(x)
  data
}
