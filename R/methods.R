# ============================================================================
# S3/S4 methods for fmlmMod
# ============================================================================

# --- fixef / ranef / VarCorr ---
# These are S3 generics in lme4, so we define S3 methods.

#' @export
fixef.fmlmMod <- function(object, ...) {
  b <- object@beta
  names(b) <- colnames(object@X)
  b
}

#' @export
ranef.fmlmMod <- function(object, ...) {
  Lambdat <- object@Lambdat
  # Update Lambdat with fitted theta
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

# --- S4 methods for standard generics ---

#' @export
setMethod("vcov", "fmlmMod", function(object, ...) {
  object@vcov_beta
})

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

#' @export
setMethod("residuals", "fmlmMod", function(object, type = "response", ...) {
  y <- object@frame[, 1]
  fv <- fitted(object)
  y - fv
})

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

#' @export
setMethod("show", "fmlmMod", function(object) {
  cat("Fast Multilevel Linear Model (fastmlm)\n")
  cat("Formula:", deparse(object@formula), "\n")
  cat("Data:   ", nrow(object@frame), "observations\n")
  cat("REML:   ", object@REML, "\n\n")

  cat("Random effects:\n")
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

  # Try Satterthwaite df for p-values
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
