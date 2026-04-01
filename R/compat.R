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
  stats::terms(formula(x), data = x@frame)
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
  # Approximate numerically by perturbing theta and refitting vcov
  eps <- 1e-4
  Jac <- array(0, dim = c(p, p, nth))

  # Refit at perturbed theta to get gradient of vcov(beta) w.r.t. theta
  # Use the cached C++ deviance function via create/deviance/result API
  y <- as.numeric(object@frame[, 1])
  X <- object@X
  Zt <- object@Zt
  Lambdat <- object@Lambdat
  Lind <- object@Lind
  lower <- object@lower

  pp <- C_fastmlm_create(y, X, Zt, Lambdat, Lind, lower, object@REML)

  for (k in seq_len(nth)) {
    theta_p <- theta
    h <- eps * max(1, abs(theta[k]))
    theta_p[k] <- theta[k] + h

    C_fastmlm_deviance(pp, theta_p)
    res_p <- C_fastmlm_result(pp)
    V_p <- res_p$sigma^2 * res_p$vcov_beta / res_p$sigma^2
    # Actually: vcov_beta from C_fastmlm_result already includes sigma^2
    V_p <- as.matrix(res_p$vcov_beta)

    Jac[, , k] <- (V_p - V) / h
  }

  # Restore state
  C_fastmlm_deviance(pp, theta)

  # vcov(theta) — approximate from optinfo or compute
  # Use simple Hessian-based estimate
  vcov_theta <- matrix(0, nth, nth)
  for (k in seq_len(nth)) {
    theta_p <- theta; theta_m <- theta
    h <- eps * max(1, abs(theta[k]))
    theta_p[k] <- theta[k] + h
    theta_m[k] <- theta[k] - h
    fp <- C_fastmlm_deviance(pp, theta_p)
    fm <- C_fastmlm_deviance(pp, theta_m)
    f0 <- C_fastmlm_deviance(pp, theta)
    vcov_theta[k, k] <- 2.0 / ((fp - 2 * f0 + fm) / (h * h))
  }

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
recover_data.fmlmMod <- function(object, ...) {
  fcall <- object@call
  trms <- terms(object)
  fr <- object@frame

  attr(fr, "call") <- fcall
  attr(fr, "terms") <- trms
  attr(fr, "predictors") <- setdiff(all.vars(formula(object)[[3]]),
                                     names(object@flist))
  attr(fr, "responses") <- all.vars(formula(object)[[2]])
  fr
}

#' @export
emm_basis.fmlmMod <- function(object, trms, xlev, grid, ...) {
  X <- stats::model.matrix(trms, data = grid, xlev = xlev)
  bhat <- fixef(object)
  V <- vcov(object)

  # Satterthwaite df
  dfargs <- list(object = object)
  dffun <- function(k, dfargs) {
    obj <- dfargs$object
    df_all <- tryCatch(
      satterthwaite_df(obj),
      error = function(e) rep(Inf, length(fixef(obj)))
    )
    # For a contrast k, use the minimum df of involved coefficients
    involved <- which(k != 0)
    if (length(involved) == 0) return(Inf)
    min(df_all[involved])
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
    vc <- VarCorr(x)
    for (grp in names(vc)) {
      mat <- vc[[grp]]
      sds <- attr(mat, "stddev")
      rp_df <- data.frame(
        effect = "ran_pars",
        group = grp,
        term = paste0("sd__", names(sds)),
        estimate = unname(sds),
        stringsAsFactors = FALSE
      )
      result <- rbind(result, rp_df)
    }
    # Residual
    result <- rbind(result, data.frame(
      effect = "ran_pars",
      group = "Residual",
      term = "sd__Observation",
      estimate = x@sigma,
      stringsAsFactors = FALSE
    ))
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
