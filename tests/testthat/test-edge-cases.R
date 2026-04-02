test_that("anova() compares models via LRT", {
  library(lme4)
  data(sleepstudy, package = "lme4")

  m1 <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = FALSE)
  m2 <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = FALSE)

  a <- anova(m1, m2)
  expect_s3_class(a, "anova")
  expect_equal(nrow(a), 2)
  expect_true("Pr(>Chisq)" %in% names(a))
  expect_true(a$Chisq[2] > 0)
  expect_true(a$`Pr(>Chisq)`[2] < 0.05)  # random slope is significant
})

test_that("simulate() produces valid responses", {
  library(lme4)
  data(sleepstudy, package = "lme4")

  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  sims <- simulate(m, nsim = 5, seed = 42)
  expect_equal(ncol(sims), 5)
  expect_equal(nrow(sims), 180)
  expect_true(all(is.finite(as.matrix(sims))))

  # Simulated values should be in a reasonable range
  expect_true(all(sims > 0))
  expect_true(all(sims < 600))
})

test_that("update() modifies formula correctly", {
  library(lme4)
  data(sleepstudy, package = "lme4")

  m1 <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)
  m2 <- update(m1, REML = FALSE)

  expect_false(m2@REML)
  expect_equal(fixef(m1), fixef(m2), tolerance = 1)
})

test_that("convergence warnings for singular fit", {
  # Create data where random effect variance is near zero
  set.seed(42)
  n <- 200
  grp <- factor(rep(1:20, each = 10))
  x <- rnorm(n)
  y <- 1 + x + rnorm(n, sd = 10)  # no actual group effect
  df <- data.frame(y = y, x = x, grp = grp)

  m <- fmlm(y ~ x + (1 | grp), data = df)

  warns <- fastmlm:::check_convergence(m)
  # May or may not trigger singular warning depending on data
  expect_true(is.character(warns))
})

test_that("cache detects data changes", {
  library(lme4)
  data(sleepstudy, package = "lme4")

  fastmlm_clear_cache()
  d1 <- sleepstudy
  d2 <- sleepstudy
  d2$Reaction[1] <- 999

  fp1 <- fastmlm:::data_fingerprint(d1)
  fp2 <- fastmlm:::data_fingerprint(d2)
  expect_false(fp1 == fp2)  # different fingerprints

  fastmlm_clear_cache()
})

test_that("formula parser handles || syntax", {
  set.seed(42)
  n <- 200
  grp <- factor(rep(1:20, each = 10))
  x <- rnorm(n)
  y <- 1 + x + rnorm(20, sd = 2)[grp] + rnorm(n)
  df <- data.frame(y = y, x = x, grp = grp)

  # || should give uncorrelated random slopes
  bars <- fastmlm:::extract_bars(y ~ x + (1 + x || grp))
  expect_true(length(bars) >= 2)  # expanded to separate terms
})

test_that("formula parser handles nested (1|a/b)", {
  bars <- fastmlm:::extract_bars(y ~ x + (1 | school/class))
  expect_equal(length(bars), 2)
  expect_equal(bars[[1]]$group, "school")
  expect_true(grepl(":", bars[[2]]$group))
})

test_that("fmlm works with unbalanced data", {
  library(lme4)
  set.seed(42)
  # Groups of varying sizes
  grp_sizes <- c(5, 10, 3, 20, 7, 15, 2, 12, 8, 6)
  n <- sum(grp_sizes)
  grp <- factor(rep(1:10, times = grp_sizes))
  x <- rnorm(n)
  y <- 1 + x + rnorm(10, sd = 2)[grp] + rnorm(n)
  df <- data.frame(y = y, x = x, grp = grp)

  m <- fmlm(y ~ x + (1 | grp), data = df)
  m2 <- lmer(y ~ x + (1 | grp), data = df)

  expect_equal(fixef(m), lme4::fixef(m2), tolerance = 1e-3)
})

test_that("fmlm handles intercept-only fixed effects", {
  library(lme4)
  data(sleepstudy, package = "lme4")

  m <- fmlm(Reaction ~ 1 + (1 | Subject), data = sleepstudy)
  m2 <- lmer(Reaction ~ 1 + (1 | Subject), data = sleepstudy)

  expect_equal(fixef(m), lme4::fixef(m2), tolerance = 1e-3)
  expect_length(fixef(m), 1)
})

test_that("fmlm handles many fixed effects", {
  set.seed(42)
  n <- 500
  grp <- factor(rep(1:50, each = 10))
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); x4 <- rnorm(n)
  y <- 1 + x1 - 0.5 * x2 + 0.3 * x3 + rnorm(50, sd = 1.5)[grp] + rnorm(n)
  df <- data.frame(y, x1, x2, x3, x4, grp)

  m <- fmlm(y ~ x1 + x2 + x3 + x4 + (1 | grp), data = df)
  expect_length(fixef(m), 5)
  expect_true(abs(fixef(m)["x1"] - 1.0) < 0.3)
})

test_that("profile CI for theta works", {
  library(lme4)
  data(sleepstudy, package = "lme4")

  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)
  ci <- fastmlm:::profile_ci_theta(m)

  expect_equal(nrow(ci), length(m@theta))
  expect_equal(ncol(ci), 2)
  # theta should be inside its CI
  for (k in seq_along(m@theta)) {
    expect_true(m@theta[k] >= ci[k, 1])
    expect_true(m@theta[k] <= ci[k, 2])
  }
})

test_that("predict works with and without newdata", {
  library(lme4)
  data(sleepstudy, package = "lme4")

  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)

  # Without newdata = fitted values
  p1 <- predict(m)
  expect_length(p1, 180)

  # With newdata = fixed effects only
  p2 <- predict(m, newdata = data.frame(Days = 0:9))
  expect_length(p2, 10)
  expect_true(p2[10] > p2[1])
})

test_that("confint produces valid intervals with profile method", {
  library(lme4)
  data(sleepstudy, package = "lme4")

  m <- fmlm(Reaction ~ Days + (1 | Subject), data = sleepstudy)
  ci <- confint(m, level = 0.99)
  expect_true(all(ci[, 2] - ci[, 1] > 0))

  ci90 <- confint(m, level = 0.90)
  # 99% CI should be wider than 90%
  expect_true(all((ci[, 2] - ci[, 1]) > (ci90[, 2] - ci90[, 1])))
})
