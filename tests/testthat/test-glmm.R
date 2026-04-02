test_that("binomial GLMM: cbpp dataset", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(cbpp, package = "lme4")

  m1 <- fglmm(cbind(incidence, size - incidence) ~ period + (1 | herd),
               data = cbpp, family = binomial())
  m2 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
              data = cbpp, family = binomial())

  expect_s4_class(m1, "fglmmMod")
  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 0.05)
  expect_equal(sqrt(VarCorr(m1)[[1]][1, 1]),
               as.numeric(attr(lme4::VarCorr(m2)$herd, "stddev")),
               tolerance = 0.05)
})

test_that("Poisson GLMM: simulated count data", {
  skip_if_not_installed("lme4")
  library(lme4)

  set.seed(42)
  n <- 200
  grp <- factor(rep(1:20, each = 10))
  x <- rnorm(n)
  re <- rnorm(20, sd = 0.5)[grp]
  y <- rpois(n, lambda = exp(1 + 0.5 * x + re))
  df <- data.frame(y = y, x = x, grp = grp)

  m1 <- fglmm(y ~ x + (1 | grp), data = df, family = poisson())
  m2 <- glmer(y ~ x + (1 | grp), data = df, family = poisson())

  expect_s4_class(m1, "fglmmMod")
  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 0.1)
})

test_that("fglmm summary and show work", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(cbpp, package = "lme4")

  m <- fglmm(cbind(incidence, size - incidence) ~ period + (1 | herd),
              data = cbpp, family = binomial())

  expect_output(show(m), "Generalised Linear Mixed Model")
  expect_output(show(m), "binomial")
  expect_output(summary(m), "AIC")
})

test_that("fglmm predict works", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(cbpp, package = "lme4")

  m <- fglmm(cbind(incidence, size - incidence) ~ period + (1 | herd),
              data = cbpp, family = binomial())

  # Link scale
  p_link <- predict(m, type = "link")
  expect_length(p_link, nrow(cbpp))
  expect_true(all(is.finite(p_link)))

  # Response scale
  p_resp <- predict(m, type = "response")
  expect_true(all(p_resp > 0 & p_resp < 1))  # probabilities
})

test_that("fglmm methods: fixef, ranef, VarCorr, vcov", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(cbpp, package = "lme4")

  m <- fglmm(cbind(incidence, size - incidence) ~ period + (1 | herd),
              data = cbpp, family = binomial())

  fe <- fixef(m)
  expect_length(fe, 4)
  expect_true(all(is.finite(fe)))

  re <- ranef(m)
  expect_true("herd" %in% names(re))
  expect_equal(nrow(re$herd), 15)

  vc <- VarCorr(m)
  expect_true(is.list(vc))
  expect_true(attr(vc[[1]], "stddev") > 0)

  V <- vcov(m)
  expect_equal(dim(V), c(4, 4))
  expect_true(all(diag(V) > 0))
})

test_that("fglmm family accessor works", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(cbpp, package = "lme4")

  m <- fglmm(cbind(incidence, size - incidence) ~ period + (1 | herd),
              data = cbpp, family = binomial())

  fam <- family(m)
  expect_equal(fam$family, "binomial")
  expect_equal(fam$link, "logit")
})

test_that("binomial GLMM with 0/1 response", {
  skip_if_not_installed("lme4")
  library(lme4)

  set.seed(42)
  n <- 300
  grp <- factor(rep(1:30, each = 10))
  x <- rnorm(n)
  re <- rnorm(30, sd = 1)[grp]
  p <- plogis(0.5 + x + re)
  y <- rbinom(n, 1, p)
  df <- data.frame(y = y, x = x, grp = grp)

  m1 <- fglmm(y ~ x + (1 | grp), data = df, family = binomial())
  m2 <- glmer(y ~ x + (1 | grp), data = df, family = binomial())

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 0.15)
})
