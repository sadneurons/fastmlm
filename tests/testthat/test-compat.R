test_that("standard extractors work", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)

  expect_equal(deparse(formula(m)), "Reaction ~ Days + (Days | Subject)")
  expect_true(inherits(terms(m), "terms"))
  expect_equal(ncol(model.matrix(m)), 2)
  expect_equal(nrow(model.frame(m)), 180)
  expect_equal(nobs(m), 180L)
  expect_true(sigma(m) > 0)
  expect_equal(df.residual(m), 178)
})

test_that("predict works with new data", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  nd <- data.frame(Days = c(0, 5, 10))
  preds <- predict(m, newdata = nd)
  expect_length(preds, 3)
  expect_true(all(is.finite(preds)))
  expect_true(preds[3] > preds[1])  # reaction time increases with days
})

test_that("confint produces valid intervals", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  ci <- confint(m)
  expect_equal(nrow(ci), 2)
  expect_equal(ncol(ci), 2)
  expect_true(all(ci[, 1] < ci[, 2]))  # lower < upper
  expect_true(all(fixef(m) > ci[, 1] & fixef(m) < ci[, 2]))  # estimates inside CI
})

test_that("logLik, AIC, BIC work", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  ll <- logLik(m)
  expect_true(is.finite(as.numeric(ll)))
  expect_true(attr(ll, "df") > 0)
  expect_true(is.finite(AIC(m)))
  expect_true(is.finite(BIC(m)))
  expect_true(BIC(m) > AIC(m))  # BIC penalises more
})

test_that("Satterthwaite df are reasonable", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)

  df <- fastmlm:::satterthwaite_df(m)
  expect_length(df, 2)
  expect_true(all(df > 1))
  expect_true(all(df < nobs(m)))
  # Should be around 17-20 for sleepstudy (18 subjects)
  expect_true(all(df > 10 & df < 30))
})

test_that("emmeans integration works", {
  skip_if_not_installed("emmeans")
  skip_if_not_installed("lme4")
  library(lme4)
  library(emmeans)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  em <- emmeans(m, ~ Days, at = list(Days = c(0, 5)))
  expect_s4_class(em, "emmGrid")
  s <- summary(em)
  expect_equal(nrow(s), 2)
  expect_true(all(is.finite(s$emmean)))
})

test_that("tidy/glance/augment work", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  m <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)

  td <- fastmlm:::tidy.fmlmMod(m)
  expect_true(is.data.frame(td))
  expect_true(nrow(td) > 0)
  expect_true("estimate" %in% names(td))

  gl <- fastmlm:::glance.fmlmMod(m)
  expect_true(is.data.frame(gl))
  expect_equal(nrow(gl), 1)
  expect_true("AIC" %in% names(gl))

  ag <- fastmlm:::augment.fmlmMod(m)
  expect_true(is.data.frame(ag))
  expect_true(".fitted" %in% names(ag))
  expect_true(".resid" %in% names(ag))
  expect_equal(nrow(ag), 180)
})

test_that("cache works correctly", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")
  fastmlm_clear_cache()

  m1 <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)
  m2 <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)  # cached

  expect_equal(fixef(m1), fixef(m2))
  expect_equal(sigma(m1), sigma(m2))

  fastmlm_clear_cache()
})

test_that("fastmlm_blas_info returns valid info", {
  info <- fastmlm_blas_info()
  expect_true(is.list(info))
  expect_true("blas_library" %in% names(info))
  expect_true("has_openmp" %in% names(info))
  expect_true(is.logical(info$has_openmp))
})
