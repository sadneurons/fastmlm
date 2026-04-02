test_that("rcs() from rms works in fmlm formula", {
  skip_if_not_installed("rms")
  skip_if_not_installed("lme4")
  library(rms)
  library(lme4)

  set.seed(42)
  n <- 300
  subj <- factor(rep(1:30, each = 10))
  age <- runif(n, 20, 80)
  re <- rnorm(30, sd = 3)[subj]
  y <- 5 + 0.3 * (age - 50) + 0.005 * (age - 50)^2 + re + rnorm(n, sd = 2)
  d <- data.frame(y = y, age = age, subj = subj)

  m <- fmlm(y ~ rcs(age, 4) + (1 | subj), data = d)
  expect_s4_class(m, "fmlmMod")
  expect_length(fixef(m), 4)  # intercept + 3 spline terms
  expect_true(m@sigma > 0)
})

test_that("rcs() predict works with new data", {
  skip_if_not_installed("rms")
  library(rms)

  set.seed(42)
  n <- 300
  subj <- factor(rep(1:30, each = 10))
  age <- runif(n, 20, 80)
  y <- 5 + 0.01 * (age - 50)^2 + rnorm(30, sd = 3)[subj] + rnorm(n, sd = 2)
  d <- data.frame(y = y, age = age, subj = subj)

  m <- fmlm(y ~ rcs(age, 4) + (1 | subj), data = d)
  nd <- data.frame(age = c(30, 50, 70))

  p <- predict(m, newdata = nd)
  expect_length(p, 3)
  expect_true(all(is.finite(p)))
  # Predictions should vary (not constant)
  expect_true(max(p) - min(p) > 0.1)
})

test_that("anova detects nonlinearity with rcs", {
  skip_if_not_installed("rms")
  library(rms)

  set.seed(42)
  n <- 500
  subj <- factor(rep(1:50, each = 10))
  age <- runif(n, 20, 80)
  y <- 5 + 0.01 * (age - 50)^2 + rnorm(50, sd = 2)[subj] + rnorm(n, sd = 2)
  d <- data.frame(y = y, age = age, subj = subj)

  m_lin <- fmlm(y ~ age + (1 | subj), data = d, REML = FALSE)
  m_rcs <- fmlm(y ~ rcs(age, 4) + (1 | subj), data = d, REML = FALSE)

  a <- anova(m_lin, m_rcs)
  expect_s3_class(a, "anova")
  # Nonlinear term should be significant (quadratic DGP)
  expect_true(a$`Pr(>Chisq)`[2] < 0.05)
})

test_that("ns() from splines works in fmlm formula", {
  set.seed(42)
  n <- 200
  subj <- factor(rep(1:20, each = 10))
  x <- runif(n, 0, 10)
  y <- sin(x) + rnorm(20, sd = 1)[subj] + rnorm(n, sd = 0.5)
  d <- data.frame(y = y, x = x, subj = subj)

  m <- fmlm(y ~ splines::ns(x, 5) + (1 | subj), data = d)
  expect_s4_class(m, "fmlmMod")
  expect_length(fixef(m), 6)  # intercept + 5 basis functions
})

test_that("multiple rcs terms work", {
  skip_if_not_installed("rms")
  library(rms)

  set.seed(42)
  n <- 400
  subj <- factor(rep(1:40, each = 10))
  age <- runif(n, 20, 80)
  bmi <- runif(n, 18, 35)
  y <- 5 + 0.01 * (age - 50)^2 - 0.5 * bmi +
       rnorm(40, sd = 2)[subj] + rnorm(n, sd = 2)
  d <- data.frame(y = y, age = age, bmi = bmi, subj = subj)

  m <- fmlm(y ~ rcs(age, 4) + rcs(bmi, 3) + (1 | subj), data = d)
  expect_s4_class(m, "fmlmMod")
  # 4 knots = 3 terms, 3 knots = 2 terms, plus intercept = 6
  expect_length(fixef(m), 6)
})

test_that("rcs with GLMM (fglmm) works", {
  skip_if_not_installed("rms")
  skip_if_not_installed("lme4")
  library(rms)

  set.seed(42)
  n <- 300
  subj <- factor(rep(1:30, each = 10))
  age <- runif(n, 20, 80)
  eta <- -2 + 0.03 * age + rnorm(30, sd = 0.5)[subj]
  y <- rbinom(n, 1, plogis(eta))
  d <- data.frame(y = y, age = age, subj = subj)

  m <- fglmm(y ~ rcs(age, 3) + (1 | subj), data = d, family = binomial())
  expect_s4_class(m, "fglmmMod")
  expect_length(fixef(m), 3)
})
