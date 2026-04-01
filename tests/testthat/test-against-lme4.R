test_that("fmlm agrees with lme4::lmer on sleepstudy random intercept", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")

  m_fast <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)
  m_lme4 <- lme4::lmer(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  # Fixed effects should agree closely
  expect_equal(fixef(m_fast), lme4::fixef(m_lme4), tolerance = 1e-4)

  # Residual SD should agree
  expect_equal(m_fast@sigma, sigma(m_lme4), tolerance = 1e-3)

  # Random effects should agree
  re_fast <- ranef(m_fast)$Subject[, 1]
  re_lme4 <- lme4::ranef(m_lme4)$Subject[, 1]
  expect_equal(re_fast, re_lme4, tolerance = 1e-3)
})

test_that("fmlm agrees with lme4::lmer on sleepstudy random slope", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")

  m_fast <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)
  m_lme4 <- lme4::lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy)

  # Fixed effects
  expect_equal(fixef(m_fast), lme4::fixef(m_lme4), tolerance = 1e-3)

  # Sigma
  expect_equal(m_fast@sigma, sigma(m_lme4), tolerance = 1e-2)

  # Deviance (REML criterion)
  expect_equal(m_fast@deviance, deviance(m_lme4, REML = TRUE),
               tolerance = 0.1)
})

test_that("fmlm ML agrees with lme4 ML", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")

  m_fast <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = FALSE)
  m_lme4 <- lme4::lmer(Reaction ~ Days + (1 | Subject), data = sleepstudy,
                        REML = FALSE)

  expect_equal(fixef(m_fast), lme4::fixef(m_lme4), tolerance = 1e-4)
  expect_equal(m_fast@sigma, sigma(m_lme4), tolerance = 1e-3)
})
