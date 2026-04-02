# fastmlm

Fast estimation of linear mixed-effects models for R.

**fastmlm** provides a drop-in replacement for `lme4::lmer()` that is 4-14x
faster, with identical numerical results. It uses the same `y ~ x + (1 | group)`
formula syntax and is compatible with emmeans, broom.mixed, and standard R
model functions.

## Installation

```r
# From source (requires C++ compiler, optionally CUDA toolkit for GPU)
install.packages("fastmlm", type = "source")

# Or from GitHub
# remotes::install_github("sadneurons/fastmlm")
```

## Quick start

```r
library(fastmlm)
library(lme4)
data(sleepstudy)

# Same syntax as lmer()
m <- fmlm(Reaction ~ Days + (Days | Subject), data = sleepstudy)
summary(m)

# All standard methods work
fixef(m)
ranef(m)
VarCorr(m)
confint(m)
AIC(m)

# GLMMs
library(lme4)
m_glmm <- fglmm(cbind(incidence, size - incidence) ~ period + (1 | herd),
                 data = cbpp, family = binomial())

# Nonlinear effects with restricted cubic splines
library(rms)
m_rcs <- fmlm(y ~ rcs(age, 5) + (1 | subject), data = mydata)

# emmeans integration
library(emmeans)
emmeans(m, ~ Days, at = list(Days = c(0, 5, 9)))
```

## Performance

Benchmarks against lme4 (per-fit timing):

| Dataset | fastmlm | lme4 | Speedup |
|---|---|---|---|
| sleepstudy (180 obs) | 1.8 ms | 6.7 ms | **3.7x** |
| 10k obs, random intercept | 1.3 ms | 9.4 ms | **7.0x** |
| 50k obs, random intercept | 15 ms | 110 ms | **7.2x** |
| 100k obs, random intercept | 38 ms | 140 ms | **3.7x** |
| 5k obs, crossed (100x50) | 18 ms | 62 ms | **3.5x** |

With formula caching (repeated fits, e.g. bootstrapping):

| Dataset | fastmlm (cached) | lme4 | Speedup |
|---|---|---|---|
| 10k obs | 0.7 ms | 9.4 ms | **14x** |
| 50k obs | 11 ms | 110 ms | **10x** |

## How it works

fastmlm achieves its speed through:

- **Custom formula parser** 3.8x faster than `lme4::lFormula`
- **Formula/data caching** for zero-cost repeated fits
- **C++ L-BFGS-B optimiser** with forward-difference gradients (entire
  optimisation loop in C++)
- **CHOLMOD supernodal Cholesky** via R's Matrix package (same engine as lme4,
  zero additional dependencies)
- **Precomputed A-matrix value mapping** eliminates sparse matrix
  multiplication on every iteration
- **OpenBLAS** direct linkage (detected at install time)
- **Optional CUDA GPU** acceleration for large dense operations
- **Satterthwaite degrees of freedom** (no lmerTest dependency needed)

## Numerical validation

fastmlm is tested against every dataset shipped with lme4 — Dyestuff
(simple), Dyestuff2 (singular fit), Penicillin (crossed), Pastes (nested),
sleepstudy (correlated slopes), and cake (factor predictors). Fixed effects
agree to machine precision across all structures. See
`vignette("fastmlm-benchmarks")` for the full comparison table.

## System information

```r
fastmlm_blas_info()
#> $blas_library: OpenBLAS (0.3.26 Haswell)
#> $has_openmp: TRUE
#> $has_gpu: TRUE
#> $gpu_device: NVIDIA GeForce RTX 3090
```

## Compatibility

fastmlm integrates with the R ecosystem:

- **emmeans** — estimated marginal means and contrasts
- **broom.mixed** — tidy/glance/augment output
- **Standard R** — formula(), terms(), model.matrix(), predict(), confint(),
  AIC(), BIC(), logLik()

## License

MIT
