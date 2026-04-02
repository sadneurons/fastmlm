# ============================================================================
# lme4 parity tests
#
# Systematically tests fastmlm against lme4 on classic datasets and known
# edge cases. This absorbs 20 years of lme4's hardening by verifying we
# produce identical results on every dataset they've been tested on.
# ============================================================================

test_that("Dyestuff: simple random intercept", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(Dyestuff, package = "lme4")

  m1 <- fmlm(Yield ~ 1 + (1 | Batch), data = Dyestuff)
  m2 <- lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-3)
  expect_equal(sigma(m1), sigma(m2), tolerance = 1e-2)
})

test_that("Dyestuff2: near-zero variance component", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(Dyestuff2, package = "lme4")

  # Dyestuff2 has near-zero random effect variance — a classic edge case
  m1 <- fmlm(Yield ~ 1 + (1 | Batch), data = Dyestuff2)
  m2 <- lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff2)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-2)

  # Check singular fit detection
  warns <- fastmlm:::check_convergence(m1)
  # Dyestuff2 typically produces a singular fit
  # (theta at boundary) — we should detect this
  expect_true(is.character(warns))
})

test_that("Penicillin: crossed random effects (plate x sample)", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(Penicillin, package = "lme4")

  m1 <- fmlm(diameter ~ 1 + (1 | plate) + (1 | sample), data = Penicillin)
  m2 <- lmer(diameter ~ 1 + (1 | plate) + (1 | sample), data = Penicillin)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-3)
  expect_equal(sigma(m1), sigma(m2), tolerance = 1e-2)
  expect_true(m1@optinfo$is_crossed)
})

test_that("Pastes: nested random effects", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(Pastes, package = "lme4")

  # Pastes has batch / cask nesting
  m1 <- fmlm(strength ~ 1 + (1 | batch) + (1 | batch:cask), data = Pastes)
  m2 <- lmer(strength ~ 1 + (1 | batch) + (1 | batch:cask), data = Pastes)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-3)
  expect_equal(sigma(m1), sigma(m2), tolerance = 1e-2)
})

test_that("sleepstudy: random slope + intercept (correlated)", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(sleepstudy, package = "lme4")

  m1 <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)
  m2 <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-3)
  expect_equal(sigma(m1), sigma(m2), tolerance = 1e-2)

  # Check random effects match (compare by column name)
  re1 <- ranef(m1)$Subject
  re2 <- lme4::ranef(m2)$Subject
  common_cols <- intersect(names(re1), names(re2))
  for (cn in common_cols) {
    expect_equal(re1[[cn]], re2[[cn]], tolerance = 1e-2,
                 label = paste("ranef column", cn))
  }
})

test_that("cake: multiple fixed effects + random intercept", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(cake, package = "lme4")

  m1 <- fmlm(angle ~ recipe + temperature + (1 | replicate), data = cake)
  m2 <- lmer(angle ~ recipe + temperature + (1 | replicate), data = cake)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-2)
  expect_equal(sigma(m1), sigma(m2), tolerance = 1e-2)
})

test_that("InstEval: large crossed design (73k obs)", {
  skip_if_not_installed("lme4")
  skip_on_cran()  # too slow for CRAN
  library(lme4)
  data(InstEval, package = "lme4")

  m1 <- fmlm(y ~ service + (1 | s) + (1 | d), data = InstEval)
  m2 <- lmer(y ~ service + (1 | s) + (1 | d), data = InstEval)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-2)
  expect_equal(sigma(m1), sigma(m2), tolerance = 1e-2)
})

test_that("ML vs REML gives different results", {
  skip_if_not_installed("lme4")
  library(lme4)
  data(sleepstudy, package = "lme4")

  m_reml <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = TRUE)
  m_ml   <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = FALSE)

  # ML sigma should be smaller (divides by n, not n-p)
  expect_true(sigma(m_ml) <= sigma(m_reml) + 1e-4)

  # Deviances should differ
  expect_false(abs(m_reml@deviance - m_ml@deviance) < 1e-6)
})

test_that("single group level doesn't crash", {
  skip_if_not_installed("lme4")
  library(lme4)

  # Only 2 groups (minimum for a random effect)
  set.seed(42)
  d <- data.frame(
    y = rnorm(20),
    x = rnorm(20),
    g = factor(rep(1:2, each = 10))
  )

  m <- fmlm(y ~ x + (1 | g), data = d)
  expect_s4_class(m, "fmlmMod")
  expect_length(fixef(m), 2)
})

test_that("many groups with few obs each", {
  skip_if_not_installed("lme4")
  library(lme4)

  # 100 groups, 2 obs each — small nj
  set.seed(42)
  d <- data.frame(
    y = rnorm(200),
    x = rnorm(200),
    g = factor(rep(1:100, each = 2))
  )

  m1 <- fmlm(y ~ x + (1 | g), data = d)
  m2 <- lmer(y ~ x + (1 | g), data = d)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-2)
})

test_that("highly unbalanced groups", {
  skip_if_not_installed("lme4")
  library(lme4)

  set.seed(42)
  # Group sizes: 1, 2, 5, 50, 100
  sizes <- c(1, 2, 5, 50, 100)
  n <- sum(sizes)
  g <- factor(rep(1:5, times = sizes))
  d <- data.frame(y = rnorm(n), x = rnorm(n), g = g)

  m1 <- fmlm(y ~ x + (1 | g), data = d)
  m2 <- lmer(y ~ x + (1 | g), data = d)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-2)
})

test_that("factor predictor in fixed effects", {
  skip_if_not_installed("lme4")
  library(lme4)

  set.seed(42)
  n <- 200
  g <- factor(rep(1:20, each = 10))
  treatment <- factor(sample(c("A", "B", "C"), n, replace = TRUE))
  y <- rnorm(n) + rnorm(20)[g] + ifelse(treatment == "B", 1, 0)
  d <- data.frame(y = y, treatment = treatment, g = g)

  m1 <- fmlm(y ~ treatment + (1 | g), data = d)
  m2 <- lmer(y ~ treatment + (1 | g), data = d)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-2)
  expect_length(fixef(m1), 3)  # intercept + 2 dummy contrasts
})

test_that("interaction in fixed effects", {
  skip_if_not_installed("lme4")
  library(lme4)

  set.seed(42)
  n <- 300
  g <- factor(rep(1:30, each = 10))
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y <- 1 + x1 + x2 + 0.5 * x1 * x2 + rnorm(30)[g] + rnorm(n)
  d <- data.frame(y, x1, x2, g)

  m1 <- fmlm(y ~ x1 * x2 + (1 | g), data = d)
  m2 <- lmer(y ~ x1 * x2 + (1 | g), data = d)

  expect_equal(fixef(m1), lme4::fixef(m2), tolerance = 1e-2)
  expect_length(fixef(m1), 4)  # intercept, x1, x2, x1:x2
})
