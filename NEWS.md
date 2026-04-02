# fastmlm 0.1.0

Initial release.

## Features

* `fmlm()` fits linear mixed-effects models with lme4 formula syntax
* `fglmm()` fits generalised linear mixed models (binomial, Poisson, Gamma)
  with PIRLS and Laplace approximation
* 4-14x faster than lme4::lmer() across dataset sizes
* Custom formula parser (3.8x faster than lme4::lFormula)
* Formula/data caching for repeated fits (bootstrap, simulation)
* C++ L-BFGS-B optimiser with forward-difference gradients
* CHOLMOD supernodal sparse Cholesky (via Matrix package, zero new deps)
* Precomputed A-matrix value mapping (zero sparse allocation per iteration)
* PCG solver for large crossed random effects
* OpenBLAS auto-detection and direct linkage
* OpenMP parallelism for stochastic log-determinant
* Optional CUDA GPU acceleration (cuBLAS/cuSPARSE)
* Satterthwaite degrees of freedom with fast (default) and exact modes:
  `summary(m, ddf = "exact")` matches lmerTest to < 0.05 df
* emmeans integration (recover_data/emm_basis)
* broom.mixed integration (tidy/glance/augment)
* Support for restricted cubic splines via rms::rcs() and splines::ns()
* `anova()` for likelihood ratio tests between nested models
* `simulate()` for parametric bootstrap
* `update()` for model modification
* Convergence diagnostics (singular fit detection, boundary warnings)
* Profile confidence intervals for variance components
* Validated against all lme4 datasets (Dyestuff, Dyestuff2, Penicillin,
  Pastes, sleepstudy, cake, InstEval)
* lme4 is optional (Suggests, not Imports)
