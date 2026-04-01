# ============================================================================
# Downstream package compatibility for fmlmMod
#
# Provides interfaces for: emmeans, broom.mixed, effects, performance
# Also provides Satterthwaite degrees of freedom (replacing lmerTest need)
# ============================================================================

# --- Standard extractors (formula, terms, model.matrix, nobs, sigma) ---

#' @export
formula.fmlmMod <- function(x, ...) x@formula

#' @export
terms.fmlmMod <- function(x, ...) {
  # Return terms for fixed effects only (remove bar terms)
  fixed_f <- fastmlm:::remove_bars(formula(x))
  stats::terms(fixed_f, data = x@frame)
}

#' @export
model.matrix.fmlmMod <- function(object, ...) object@X

#' @export
nobs.fmlmMod <- function(object, ...) nrow(object@frame)

#' @export
sigma.fmlmMod <- function(object, ...) object@sigma

#' @export
df.residual.fmlmMod <- function(object, ...) {
  nrow(object@frame) - length(object@beta)
}

#' @export
model.frame.fmlmMod <- function(formula, ...) formula@frame

#' @export
predict.fmlmMod <- function(object, newdata = NULL, re.form = NULL, ...) {
  if (is.null(newdata)) {
    return(fitted(object))
  }
  # Fixed effects only prediction for new data
  X_new <- stats::model.matrix(
    stats::delete.response(terms(object)),
    data = newdata
  )
  as.numeric(X_new %*% fixef(object))
}

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
#' @return A numeric vector of degrees of freedom, one per fixed effect.
#' @keywords internal
satterthwaite_df <- function(object) {
  beta <- fixef(object)
  V <- vcov(object)
  p <- length(beta)
  theta <- object@theta
  nth <- length(theta)

  # We need: d(vcov(beta)) / d(theta_k) for each k
  # Use central differences for better accuracy
  eps <- 1e-4
  Jac <- array(0, dim = c(p, p, nth))

  y <- as.numeric(object@frame[, 1])
  X <- object@X
  Zt <- object@Zt
  Lambdat <- object@Lambdat
  Lind <- object@Lind
  lower <- object@lower

  pp <- C_fastmlm_create(y, X, Zt, Lambdat, Lind, lower, object@REML)

  for (k in seq_len(nth)) {
    h <- eps * max(1, abs(theta[k]))

    # Forward
    theta_p <- theta
    theta_p[k] <- theta[k] + h
    C_fastmlm_deviance(pp, theta_p)
    V_p <- as.matrix(C_fastmlm_result(pp)$vcov_beta)

    # Backward
    theta_m <- theta
    theta_m[k] <- theta[k] - h
    C_fastmlm_deviance(pp, theta_m)
    V_m <- as.matrix(C_fastmlm_result(pp)$vcov_beta)

    Jac[, , k] <- (V_p - V_m) / (2 * h)
  }

  # Restore state
  C_fastmlm_deviance(pp, theta)

  # vcov(theta) from full Hessian of deviance (including off-diagonals)
  f0 <- C_fastmlm_deviance(pp, theta)
  vcov_theta <- matrix(0, nth, nth)

  # Diagonal: (f(x+h) - 2f(x) + f(x-h)) / h^2
  f_plus <- numeric(nth)
  f_minus <- numeric(nth)
  h_vec <- numeric(nth)
  for (k in seq_len(nth)) {
    h_vec[k] <- eps * max(1, abs(theta[k]))
    theta_p <- theta; theta_p[k] <- theta[k] + h_vec[k]
    theta_m <- theta; theta_m[k] <- theta[k] - h_vec[k]
    f_plus[k] <- C_fastmlm_deviance(pp, theta_p)
    f_minus[k] <- C_fastmlm_deviance(pp, theta_m)
    vcov_theta[k, k] <- (f_plus[k] - 2 * f0 + f_minus[k]) / (h_vec[k]^2)
  }

  # Off-diagonal: (f(+i,+j) - f(+i,-j) - f(-i,+j) + f(-i,-j)) / (4*hi*hj)
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
        vcov_theta[i, j] <- (fpp - fpm - fmp + fmm) / (4 * h_vec[i] * h_vec[j])
        vcov_theta[j, i] <- vcov_theta[i, j]
      }
    }
  }

  # Invert Hessian: vcov = 2 * H^{-1} (deviance = -2 logLik)
  vcov_theta <- tryCatch(
    2.0 * solve(vcov_theta),
    error = function(e) {
      # If Hessian is singular, use pseudoinverse
      s <- svd(vcov_theta)
      tol <- max(dim(vcov_theta)) * max(s$d) * .Machine$double.eps
      pos <- s$d > tol
      2.0 * s$v[, pos, drop = FALSE] %*% diag(1 / s$d[pos], nrow = sum(pos)) %*% t(s$u[, pos, drop = FALSE])
    }
  )

  # Satterthwaite df for each fixed effect
  df <- numeric(p)
  for (j in seq_len(p)) {
    var_j <- V[j, j]
    # Gradient of var(beta_j) w.r.t. theta
    g <- numeric(nth)
    for (k in seq_len(nth)) {
      g[k] <- Jac[j, j, k]
    }
    denom <- sum(g * (vcov_theta %*% g))
    df[j] <- if (denom > 0) 2 * var_j^2 / denom else Inf
  }

  # Clamp to reasonable range
  df <- pmax(df, 1)
  df
}

# ============================================================================
# emmeans support
# ============================================================================

#' @export
recover_data.fmlmMod <- function(object, data = NULL, ...) {
  fcall <- object@call
  trms <- terms(object)  # fixed-effects only (bar terms stripped)
  if (is.null(data)) data <- object@frame
  emmeans::recover_data(fcall, trms, "na.omit", data = data)
}

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

#' @export
augment.fmlmMod <- function(x, data = NULL, ...) {
  if (is.null(data)) data <- x@frame
  data$.fitted <- fitted(x)
  data$.resid <- residuals(x)
  data
}
