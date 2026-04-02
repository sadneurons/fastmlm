#' Lightweight formula parser for mixed models
#'
#' Parses lme4-style formulas without depending on lme4::lFormula.
#' Extracts fixed effects, random effects bar terms, builds model
#' matrices X (dense) and Z/Lambdat/Lind (sparse) directly.
#'
#' @param formula A two-sided formula with bar notation for random effects.
#' @param data A data frame.
#' @param REML Logical; REML or ML (used for naming only).
#' @param na.action Function for NA handling.
#' @param contrasts Optional contrast specifications.
#' @return A list with components: fr (model frame), X, Zt, Lambdat, Lind,
#'   theta, lower, flist, cnms, Gp.
#' @keywords internal
fast_lFormula <- function(formula, data, REML = TRUE,
                          na.action = na.omit,
                          contrasts = NULL) {

  # --- Step 1: Split formula into fixed and random parts ---
  bars <- extract_bars(formula)
  fixed_formula <- remove_bars(formula)

  # --- Step 2: Build model frame and fixed-effects matrix ---
  # Combine all variables needed (fixed + random grouping + random terms)
  all_vars <- all.vars(formula)
  mf <- stats::model.frame(
    stats::reformulate(all_vars[-1], response = all_vars[1]),
    data = data, na.action = na.action
  )
  y <- stats::model.response(mf)

  # Build X from the fixed-effects-only formula
  X <- stats::model.matrix(fixed_formula, data = mf, contrasts.arg = contrasts)

  # --- Step 3: Build random-effects structures via C++ ---
  re_info <- build_re_structures(bars, mf)

  list(
    fr       = mf,
    X        = X,
    Zt       = re_info$Zt,
    Lambdat  = re_info$Lambdat,
    Lind     = re_info$Lind,
    theta    = re_info$theta,
    lower    = re_info$lower,
    flist    = re_info$flist,
    cnms     = re_info$cnms,
    Gp       = re_info$Gp
  )
}

#' Extract bar terms from a formula
#'
#' Finds all (... | group) terms in a formula.
#'
#' @param formula A formula potentially containing bar terms.
#' @return A list of lists, each with \code{terms} (character vector of
#'   variable names on the LHS of |) and \code{group} (RHS of |).
#' @keywords internal
extract_bars <- function(formula) {
  # Walk the formula tree to find all | terms
  bars <- list()

  find_bars <- function(expr) {
    if (is.call(expr)) {
      if (identical(expr[[1]], as.name("|"))) {
        lhs <- expr[[2]]
        rhs <- expr[[3]]
        # Check for nested syntax: (1 | a/b) => (1 | a) + (1 | a:b)
        if (is.call(rhs) && identical(rhs[[1]], as.name("/"))) {
          a <- rhs[[2]]
          b <- rhs[[3]]
          bars[[length(bars) + 1L]] <<- list(
            lhs = lhs, rhs = a, group = deparse(a), uncorrelated = FALSE)
          interaction_expr <- call(":", a, b)
          bars[[length(bars) + 1L]] <<- list(
            lhs = lhs, rhs = interaction_expr,
            group = deparse(interaction_expr), uncorrelated = FALSE)
        } else {
          bars[[length(bars) + 1L]] <<- list(
            lhs = lhs, rhs = rhs, group = deparse(rhs), uncorrelated = FALSE)
        }
      } else if (identical(expr[[1]], as.name("||"))) {
        # Double bar: uncorrelated random slopes
        # (1 + x || group) expands to (1 | group) + (0 + x | group)
        lhs <- expr[[2]]
        rhs <- expr[[3]]
        grp <- deparse(rhs)

        # Parse LHS terms
        lhs_str <- deparse(lhs)
        lhs_terms <- trimws(strsplit(lhs_str, "\\+")[[1]])

        for (term in lhs_terms) {
          if (term %in% c("1", "")) {
            bars[[length(bars) + 1L]] <<- list(
              lhs = quote(1), rhs = rhs, group = grp, uncorrelated = TRUE)
          } else {
            bars[[length(bars) + 1L]] <<- list(
              lhs = str2lang(paste0("0 + ", term)),
              rhs = rhs, group = grp, uncorrelated = TRUE)
          }
        }
      } else if (identical(expr[[1]], as.name("/"))) {
        # Nested: (1 | a/b) => (1 | a) + (1 | a:b)
        # This is handled at the | level when we detect / in the RHS
        # Just recurse normally
        for (i in seq_along(expr)[-1]) {
          find_bars(expr[[i]])
        }
      } else {
        for (i in seq_along(expr)[-1]) {
          find_bars(expr[[i]])
        }
      }
    }
  }

  find_bars(formula[[3]])  # RHS of the two-sided formula
  bars
}

#' Remove bar terms from a formula
#'
#' Returns a formula with all (... | group) terms removed, suitable for
#' building the fixed-effects model matrix.
#'
#' @param formula A formula potentially containing bar terms.
#' @return A formula with bar terms removed.
#' @keywords internal
remove_bars <- function(formula) {
  remove_bar_expr <- function(expr) {
    if (is.call(expr)) {
      # If this is a ( ... ) wrapping a bar term, remove it
      if (identical(expr[[1]], as.name("("))) {
        inner <- expr[[2]]
        if (is.call(inner) && identical(inner[[1]], as.name("|"))) {
          return(NULL)
        }
      }

      # For + and other operators, recurse and rebuild
      op <- expr[[1]]
      if (identical(op, as.name("+")) && length(expr) == 3) {
        left <- remove_bar_expr(expr[[2]])
        right <- remove_bar_expr(expr[[3]])
        if (is.null(left) && is.null(right)) return(NULL)
        if (is.null(left)) return(right)
        if (is.null(right)) return(left)
        return(call("+", left, right))
      }

      if (identical(op, as.name("-")) && length(expr) == 3) {
        left <- remove_bar_expr(expr[[2]])
        right <- remove_bar_expr(expr[[3]])
        if (is.null(left) && is.null(right)) return(NULL)
        if (is.null(left)) return(right)
        if (is.null(right)) return(left)
        return(call("-", left, right))
      }

      # Other calls: recurse on arguments
      for (i in seq_along(expr)[-1]) {
        expr[[i]] <- remove_bar_expr(expr[[i]])
      }
      return(expr)
    }
    expr
  }

  rhs <- remove_bar_expr(formula[[3]])
  if (is.null(rhs)) rhs <- 1  # intercept-only if all terms were bars

  stats::reformulate(deparse(rhs), response = deparse(formula[[2]]))
}

#' Build random-effects structures from parsed bar terms
#'
#' Constructs Zt (sparse), Lambdat, Lind, theta, lower, flist, cnms, Gp
#' from the parsed bar terms and model frame.
#'
#' @param bars List of bar term info from \code{extract_bars}.
#' @param mf Model frame.
#' @return A list with Zt, Lambdat, Lind, theta, lower, flist, cnms, Gp.
#' @keywords internal
build_re_structures <- function(bars, mf) {
  n <- nrow(mf)
  n_terms <- length(bars)

  flist <- list()
  cnms <- list()
  Zt_blocks <- list()
  Lambdat_blocks <- list()
  Lind_parts <- list()
  theta_parts <- list()
  lower_parts <- list()
  Gp <- integer(n_terms + 1L)
  Gp[1] <- 0L

  theta_offset <- 0L  # running offset into theta for Lind

  for (t in seq_len(n_terms)) {
    bar <- bars[[t]]
    grp_name <- bar$group

    # Get the grouping factor
    grp <- as.factor(mf[[grp_name]])
    nlevs <- nlevels(grp)
    flist[[grp_name]] <- grp

    # Parse the LHS of the bar term to get the design matrix for this RE term
    lhs_str <- deparse(bar$lhs)

    # Build the within-group model matrix
    if (lhs_str == "1") {
      # Random intercept only
      Zi <- Matrix::sparseMatrix(
        i = as.integer(grp),
        j = seq_len(n),
        x = 1.0,
        dims = c(nlevs, n)
      )
      term_names <- "(Intercept)"
    } else if (lhs_str == "0 + 1" || lhs_str == "1 + 0") {
      Zi <- Matrix::sparseMatrix(
        i = as.integer(grp),
        j = seq_len(n),
        x = 1.0,
        dims = c(nlevs, n)
      )
      term_names <- "(Intercept)"
    } else {
      # Random slopes: build model matrix for the LHS
      # Handle "1 + x", "x", "0 + x", etc.
      lhs_formula <- stats::reformulate(lhs_str, intercept = TRUE)
      # Check if intercept is suppressed
      if (grepl("^0\\s*\\+", lhs_str) || grepl("\\+\\s*0$", lhs_str) ||
          lhs_str == "0") {
        lhs_formula <- stats::reformulate(
          gsub("(^0\\s*\\+\\s*|\\s*\\+\\s*0$)", "", lhs_str),
          intercept = FALSE
        )
      }
      Zmat <- stats::model.matrix(lhs_formula, data = mf)
      term_names <- colnames(Zmat)
      ncols <- ncol(Zmat)

      # Build the sparse Zt block: for each level of grp, ncols rows in Zt
      # Row indices: for observation j in group g, rows are
      #   (g-1)*ncols + 1 : g*ncols
      q_t <- nlevs * ncols

      # Build using triplets
      grp_int <- as.integer(grp)
      ii <- integer(0)
      jj <- integer(0)
      xx <- numeric(0)

      for (k in seq_len(ncols)) {
        row_idx <- (grp_int - 1L) * ncols + k
        nz <- which(Zmat[, k] != 0)
        ii <- c(ii, row_idx[nz])
        jj <- c(jj, nz)
        xx <- c(xx, Zmat[nz, k])
      }

      Zi <- Matrix::sparseMatrix(
        i = ii, j = jj, x = xx,
        dims = c(q_t, n)
      )
    }

    q_t <- nrow(Zi)
    ncols <- if (exists("ncols", inherits = FALSE)) ncols else 1L
    if (lhs_str == "1") ncols <- 1L
    cnms[[grp_name]] <- term_names

    Zt_blocks[[t]] <- Zi

    # Build Lambdat block: lower-triangular ncols x ncols, replicated nlevs times
    # For random intercept: 1x1 identity per level
    # For random slope: ncols x ncols lower-triangular per level
    if (ncols == 1L) {
      # Diagonal: one theta per term
      Lambdat_t <- Matrix::Diagonal(q_t, x = 1.0)
      Lind_t <- rep(theta_offset + 1L, q_t)
      theta_t <- 1.0
      lower_t <- 0.0
    } else {
      # Lower-triangular block for each level
      # Number of free params in lower triangle: ncols*(ncols+1)/2
      n_params <- ncols * (ncols + 1L) %/% 2L

      # Build the template lower-triangular matrix
      Lt <- matrix(0, ncols, ncols)
      param_idx <- integer(ncols * ncols)
      k <- 0L
      for (j in seq_len(ncols)) {
        for (i in j:ncols) {
          k <- k + 1L
          Lt[i, j] <- 1.0
          param_idx[(j - 1L) * ncols + i] <- k
        }
      }

      # Replicate across all levels: block-diagonal
      ii <- integer(0); jj <- integer(0); xx <- numeric(0)
      lind_t <- integer(0)

      for (lev in seq_len(nlevs)) {
        offset <- (lev - 1L) * ncols
        for (j in seq_len(ncols)) {
          for (i in j:ncols) {
            ii <- c(ii, offset + i)
            jj <- c(jj, offset + j)
            xx <- c(xx, if (i == j) 1.0 else 0.0)
            lind_t <- c(lind_t, theta_offset + param_idx[(j - 1L) * ncols + i])
          }
        }
      }

      Lambdat_t <- Matrix::sparseMatrix(
        i = ii, j = jj, x = xx,
        dims = c(q_t, q_t)
      )
      Lind_t <- lind_t

      # Starting theta: 1 on diagonal, 0 on off-diagonal
      theta_t <- numeric(n_params)
      lower_t <- numeric(n_params)
      k <- 0L
      for (j in seq_len(ncols)) {
        for (i in j:ncols) {
          k <- k + 1L
          if (i == j) {
            theta_t[k] <- 1.0
            lower_t[k] <- 0.0
          } else {
            theta_t[k] <- 0.0
            lower_t[k] <- -Inf
          }
        }
      }
    }

    Lambdat_blocks[[t]] <- Lambdat_t
    Lind_parts[[t]] <- Lind_t
    theta_parts[[t]] <- theta_t
    lower_parts[[t]] <- lower_t

    Gp[t + 1L] <- Gp[t] + q_t
    theta_offset <- theta_offset + length(theta_t)

    # Clean up ncols for next iteration
    if (exists("ncols", inherits = FALSE)) rm(ncols)
  }

  # Stack Zt blocks vertically
  Zt <- do.call(rbind, Zt_blocks)

  # Build block-diagonal Lambdat
  Lambdat <- Matrix::bdiag(Lambdat_blocks)
  # Ensure it's dgCMatrix
  Lambdat <- methods::as(Lambdat, "dgCMatrix")

  list(
    Zt      = Zt,
    Lambdat = Lambdat,
    Lind    = as.integer(unlist(Lind_parts)),
    theta   = as.numeric(unlist(theta_parts)),
    lower   = as.numeric(unlist(lower_parts)),
    flist   = flist,
    cnms    = cnms,
    Gp      = as.integer(Gp)
  )
}
