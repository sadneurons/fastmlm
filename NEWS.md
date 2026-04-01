# fastmlm 0.1.0

Initial release.

## Features

* `fmlm()` fits linear mixed-effects models with lme4 formula syntax
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
* Satterthwaite degrees of freedom and p-values
* emmeans integration (recover_data/emm_basis)
* broom.mixed integration (tidy/glance/augment)
* Full set of standard R model methods
