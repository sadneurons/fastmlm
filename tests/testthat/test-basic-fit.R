test_that("fmlm fits a random intercept model", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  expect_s4_class(m, "fmlmMod")
  expect_length(fixef(m), 2)
  expect_named(fixef(m), c("(Intercept)", "Days"))
  expect_true(m@sigma > 0)
  expect_true(is.finite(m@deviance))
  expect_equal(m@optinfo$convergence, 0)
})

test_that("fmlm fits a random slope model", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)

  expect_s4_class(m, "fmlmMod")
  expect_length(fixef(m), 2)
  expect_length(m@theta, 3)  # 3 variance params for (Days | Subject)
  expect_true(m@sigma > 0)
})

test_that("fmlm methods work", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  # fixef
  fe <- fixef(m)
  expect_true(is.numeric(fe))
  expect_length(fe, 2)

 # ranef
  re <- ranef(m)
  expect_true(is.list(re))
  expect_true("Subject" %in% names(re))
  expect_equal(nrow(re$Subject), 18)

  # vcov
  vc <- vcov(m)
  expect_true(is.matrix(vc))
  expect_equal(dim(vc), c(2, 2))
  expect_true(all(diag(vc) > 0))

  # fitted & residuals
  fv <- fitted(m)
  expect_length(fv, nrow(sleepstudy))
  res <- residuals(m)
  expect_length(res, nrow(sleepstudy))
  expect_equal(fv + res, sleepstudy$Reaction, tolerance = 1e-10)

  # logLik
  ll <- logLik(m)
  expect_true(is.finite(as.numeric(ll)))

  # VarCorr
  varcor <- VarCorr(m)
  expect_true(is.list(varcor))

  # coef
  co <- coef(m)
  expect_true(is.list(co))
})

test_that("fmlm handles formula with multiple fixed effects", {
  skip_if_not_installed("lme4")
  library(lme4)

  set.seed(42)
  n <- 200
  grp <- factor(rep(1:20, each = 10))
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  re <- rnorm(20)[grp]
  y <- 1 + 2 * x1 - 0.5 * x2 + re + rnorm(n)
  df <- data.frame(y = y, x1 = x1, x2 = x2, grp = grp)

  m <- fmlm(y ~ x1 + x2 + (1 | grp), data = df)
  expect_s4_class(m, "fmlmMod")
  expect_length(fixef(m), 3)
})
