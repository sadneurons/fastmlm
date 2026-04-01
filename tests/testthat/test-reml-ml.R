test_that("fmlm supports both REML and ML estimation", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")

  m_reml <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = TRUE)
  m_ml   <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = FALSE)

  expect_true(m_reml@REML)
  expect_false(m_ml@REML)

  # REML and ML should give different deviances
  expect_false(m_reml@deviance == m_ml@deviance)

  # Fixed effects should be similar but not identical
  expect_equal(fixef(m_reml), fixef(m_ml), tolerance = 1.0)

  # Both should converge

  expect_equal(m_reml@optinfo$convergence, 0)
  expect_equal(m_ml@optinfo$convergence, 0)
})

test_that("ML sigma is typically smaller than REML sigma", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")

  m_reml <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = TRUE)
  m_ml   <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = FALSE)

  # ML divides by n, REML by n-p, so ML sigma should be <= REML sigma
  expect_true(m_ml@sigma <= m_reml@sigma + 1e-6)
})
