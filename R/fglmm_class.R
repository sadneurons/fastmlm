#' S4 class for fastmlm GLMM fit
#'
#' @param object An \code{fglmmMod} object.
#' @param type Character; type of prediction or residuals.
#' @param ... Additional arguments.
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
#' @slot eta Numeric vector of linear predictor values.
#' @slot mu Numeric vector of fitted means (response scale).
#' @slot deviance The deviance at convergence.
#' @slot vcov_beta Variance-covariance matrix of fixed effects.
#' @slot optinfo List with optimiser convergence info.
#' @slot family The GLM family object.
#' @slot X Fixed-effects model matrix.
#' @slot Zt Sparse transpose of random-effects design matrix.
#' @slot Lambdat Sparse relative covariance factor.
#' @slot Lind Integer vector mapping theta to Lambdat entries.
#'
#' @exportClass fglmmMod
setClass("fglmmMod", representation(
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
  eta       = "numeric",
  mu        = "numeric",
  deviance  = "numeric",
  vcov_beta = "matrix",
  optinfo   = "list",
  family    = "list",
  X         = "matrix",
  Zt        = "dgCMatrix",
  Lambdat   = "dgCMatrix",
  Lind      = "integer"
))

# --- S3 methods ---

#' @export
fixef.fglmmMod <- function(object, ...) {
  b <- object@beta
  names(b) <- colnames(object@X)
  b
}

#' @export
ranef.fglmmMod <- function(object, ...) {
  Lambdat <- object@Lambdat
  x <- Lambdat@x
  for (i in seq_along(x)) x[i] <- object@theta[object@Lind[i]]
  Lambdat@x <- x
  b <- as.numeric(Matrix::t(Lambdat) %*% object@u)
  flist <- object@flist
  cnms <- object@cnms
  Gp <- object@Gp
  result <- list()
  for (i in seq_along(flist)) {
    fac <- flist[[i]]
    nlevs <- nlevels(fac)
    cnames <- cnms[[i]]
    ncols <- length(cnames)
    bi <- b[(Gp[i] + 1L):Gp[i + 1L]]
    mat <- matrix(bi, nrow = nlevs, ncol = ncols, byrow = TRUE)
    colnames(mat) <- cnames
    rownames(mat) <- levels(fac)
    result[[names(flist)[i]]] <- as.data.frame(mat)
  }
  result
}

#' @export
VarCorr.fglmmMod <- function(x, sigma = 1, ...) {
  # GLMMs don't have a residual sigma — variance is determined by the family.
  # Extract theta and build VarCorr directly (sigma = 1 for scaling).
  flist <- x@flist
  cnms <- x@cnms
  Gp <- x@Gp
  theta <- x@theta
  Lind <- x@Lind

  Lambdat <- x@Lambdat
  lx <- Lambdat@x
  for (i in seq_along(lx)) lx[i] <- theta[Lind[i]]
  Lambdat@x <- lx

  result <- list()
  for (i in seq_along(flist)) {
    cnames <- cnms[[i]]
    ncols <- length(cnames)
    idx_start <- Gp[i] + 1L
    idx_end <- Gp[i] + ncols
    Li <- as.matrix(Lambdat[idx_start:idx_end, idx_start:idx_end])
    vc <- crossprod(Li)  # no sigma scaling for GLMMs
    rownames(vc) <- cnames
    colnames(vc) <- cnames
    attr(vc, "stddev") <- sqrt(diag(vc))
    attr(vc, "correlation") <- if (ncols > 1) stats::cov2cor(vc) else NULL
    result[[names(flist)[i]]] <- vc
  }

  attr(result, "sc") <- NA_real_  # no residual SD for GLMMs
  class(result) <- "VarCorr.fmlmMod"
  result
}

#' @method logLik fglmmMod
#' @export
logLik.fglmmMod <- function(object, ...) {
  ll <- -0.5 * object@deviance
  nth <- length(object@theta)
  p <- length(object@beta)
  attr(ll, "df") <- p + nth
  attr(ll, "nobs") <- nrow(object@frame)
  class(ll) <- "logLik"
  ll
}

#' @method formula fglmmMod
#' @export
formula.fglmmMod <- function(x, ...) x@formula

#' @method family fglmmMod
#' @export
family.fglmmMod <- function(object, ...) {
  fam <- object@family
  class(fam) <- "family"
  fam
}

#' @method predict fglmmMod
#' @export
predict.fglmmMod <- function(object, newdata = NULL,
                              type = c("link", "response"), ...) {
  type <- match.arg(type)
  if (is.null(newdata)) {
    eta <- object@eta
  } else {
    X_new <- stats::model.matrix(
      stats::delete.response(terms.fmlmMod(object)),
      data = newdata
    )
    eta <- as.numeric(X_new %*% fixef(object))
  }
  if (type == "response") {
    object@family$linkinv(eta)
  } else {
    eta
  }
}

# --- S4 methods ---

#' @rdname fglmmMod-class
#' @export
setMethod("vcov", "fglmmMod", function(object, ...) object@vcov_beta)

#' @rdname fglmmMod-class
#' @export
setMethod("fitted", "fglmmMod", function(object, ...) object@mu)

#' @rdname fglmmMod-class
#' @export
setMethod("residuals", "fglmmMod", function(object, type = "deviance", ...) {
  y <- object@frame[, 1]
  if (is.matrix(y)) y <- y[, 1] / rowSums(y)
  mu <- object@mu
  type <- match.arg(type, c("deviance", "response", "pearson"))
  switch(type,
    response = y - mu,
    pearson = (y - mu) / sqrt(object@family$variance(mu)),
    deviance = {
      d <- numeric(length(y))
      for (i in seq_along(y)) {
        mu_i <- max(1e-10, min(1 - 1e-10, mu[i]))
        dr <- object@family$dev.resids(y[i], mu_i, 1)
        d[i] <- sign(y[i] - mu_i) * sqrt(abs(dr))
      }
      d
    }
  )
})

#' @rdname fglmmMod-class
#' @export
setMethod("show", "fglmmMod", function(object) {
  cat("Fast Generalised Linear Mixed Model (fastmlm)\n")
  cat("Family: ", object@family$family, "(", object@family$link, ")\n")
  cat("Formula:", deparse(object@formula), "\n")
  cat("Data:   ", nrow(object@frame), "observations\n")

  # Convergence warnings
  cat("\nRandom effects:\n")
  vc <- VarCorr.fglmmMod(object)
  for (nm in names(vc)) {
    mat <- vc[[nm]]
    cat(sprintf("  Groups: %s (%d levels)\n", nm, nlevels(object@flist[[nm]])))
    sds <- attr(mat, "stddev")
    for (j in seq_along(sds)) {
      cat(sprintf("    %-20s Std.Dev. %8.4f\n", names(sds)[j], sds[j]))
    }
  }

  cat("\nFixed effects:\n")
  fe <- fixef.fglmmMod(object)
  se <- sqrt(diag(vcov(object)))
  zval <- fe / se
  pval <- 2 * stats::pnorm(abs(zval), lower.tail = FALSE)
  coef_tab <- cbind(Estimate = fe, `Std. Error` = se,
                    `z value` = zval, `Pr(>|z|)` = pval)
  stats::printCoefmat(coef_tab, digits = 4, signif.stars = TRUE)

  cat("\nDeviance:", round(object@deviance, 2),
      "| PIRLS iter:", object@optinfo$pirls_iter,
      "| Convergence:", object@optinfo$convergence, "\n")

  invisible(object)
})

#' @rdname fglmmMod-class
#' @export
setMethod("summary", "fglmmMod", function(object, ...) {
  show(object)
  ll <- logLik(object)
  cat("\nAIC:", round(AIC(object), 2),
      "| BIC:", round(BIC(object), 2),
      "| logLik:", round(as.numeric(ll), 2), "\n")
  invisible(object)
})
