test_that("fmlm fits a crossed random effects model", {
  skip_if_not_installed("lme4")
  library(lme4)

  # Simulate crossed data: items x subjects
  set.seed(123)
  n_subj <- 30
  n_item <- 20
  n <- n_subj * n_item

  subj <- factor(rep(1:n_subj, each = n_item))
  item <- factor(rep(1:n_item, times = n_subj))
  x <- rnorm(n)
  re_subj <- rnorm(n_subj, sd = 1.5)[subj]
  re_item <- rnorm(n_item, sd = 1.0)[item]
  y <- 2 + 0.5 * x + re_subj + re_item + rnorm(n)

  df <- data.frame(y = y, x = x, subj = subj, item = item)

  # Fit with fastmlm (uses Cholesky path since q < pcg_threshold)
  m <- fmlm(y ~ x + (1 | subj) + (1 | item), data = df)

  expect_s4_class(m, "fmlmMod")
  expect_length(fixef(m), 2)
  expect_true(m@sigma > 0)
  expect_equal(m@optinfo$convergence, 0)
  expect_true(m@optinfo$is_crossed)
  expect_false(m@optinfo$using_pcg)  # q = 50 < 5000

  # Compare with lme4
  m2 <- lmer(y ~ x + (1 | subj) + (1 | item), data = df)
  expect_equal(fixef(m), lme4::fixef(m2), tolerance = 1e-3)
  expect_equal(m@sigma, sigma(m2), tolerance = 1e-2)
})

test_that("fmlm detects nested vs crossed correctly", {
  skip_if_not_installed("lme4")
  library(lme4)

  data(sleepstudy, package = "lme4")

  # sleepstudy has only nested RE (Days | Subject)
  m_nested <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)
  expect_false(m_nested@optinfo$is_crossed)

  # Simulate crossed data
  set.seed(42)
  n <- 200
  a <- factor(rep(1:10, each = 20))
  b <- factor(rep(1:20, times = 10))
  y <- rnorm(n) + rnorm(10)[a] + rnorm(20)[b]
  x <- rnorm(n)
  df <- data.frame(y = y, x = x, a = a, b = b)

  m_crossed <- fmlm(y ~ x + (1 | a) + (1 | b), data = df)
  expect_true(m_crossed@optinfo$is_crossed)
})

test_that("crossed RE model with multiple fixed effects works", {
  skip_if_not_installed("lme4")
  library(lme4)

  set.seed(99)
  n_subj <- 20
  n_item <- 15
  n <- n_subj * n_item

  subj <- factor(rep(1:n_subj, each = n_item))
  item <- factor(rep(1:n_item, times = n_subj))
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  re_subj <- rnorm(n_subj, sd = 2.0)[subj]
  re_item <- rnorm(n_item, sd = 1.0)[item]
  y <- 1 + 0.5 * x1 - 0.3 * x2 + re_subj + re_item + rnorm(n, sd = 0.5)

  df <- data.frame(y = y, x1 = x1, x2 = x2, subj = subj, item = item)

  m_fast <- fmlm(y ~ x1 + x2 + (1 | subj) + (1 | item), data = df)
  m_lme4 <- lmer(y ~ x1 + x2 + (1 | subj) + (1 | item), data = df)

  expect_equal(fixef(m_fast), lme4::fixef(m_lme4), tolerance = 1e-3)
  expect_equal(m_fast@sigma, sigma(m_lme4), tolerance = 1e-2)
})

test_that("fmlm handles three-way crossed RE", {
  skip_if_not_installed("lme4")
  library(lme4)

  set.seed(77)
  n <- 300
  a <- factor(sample(1:10, n, replace = TRUE))
  b <- factor(sample(1:8, n, replace = TRUE))
  c <- factor(sample(1:5, n, replace = TRUE))
  x <- rnorm(n)
  y <- 1 + x + rnorm(10)[a] + rnorm(8)[b] + rnorm(5)[c] + rnorm(n)
  df <- data.frame(y = y, x = x, a = a, b = b, c = c)

  m <- fmlm(y ~ x + (1 | a) + (1 | b) + (1 | c), data = df)
  expect_s4_class(m, "fmlmMod")
  expect_length(fixef(m), 2)
  expect_true(m@optinfo$is_crossed)
  expect_equal(m@optinfo$convergence, 0)

  # Compare with lme4
  m2 <- lmer(y ~ x + (1 | a) + (1 | b) + (1 | c), data = df)
  expect_equal(fixef(m), lme4::fixef(m2), tolerance = 1e-2)
})
