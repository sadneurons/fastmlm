#ifndef FASTMLM_PARALLEL_OPS_H
#define FASTMLM_PARALLEL_OPS_H

#include "fastmlm_types.h"

namespace fastmlm {
namespace parallel {

// Parallel sparse cross-product: Z^T Z computed in column blocks.
// Uses OpenMP to parallelise the outer product accumulation.
// Falls back to serial if OpenMP is unavailable.
SpMatd parallel_ZtZ(const SpMatd& Zt, int nthreads = 0);

// Parallel residual computation for large n.
// resid = y - X*beta - Z^T' * b
VectorXd parallel_residuals(const VectorXd& y,
                             const MatrixXd& X,
                             const VectorXd& beta,
                             const SpMatd& Zt,
                             const VectorXd& b,
                             int nthreads = 0);

// Parallel stochastic log-determinant probes.
// Each probe vector is independent — embarrassingly parallel.
double parallel_stochastic_logdet(const SpMatd& A,
                                   int n_probes = 30,
                                   int lanczos_steps = 30,
                                   int nthreads = 0,
                                   unsigned int seed = 42);

// Get the effective number of threads (respects user setting and OMP_NUM_THREADS)
int effective_threads(int requested = 0);

} // namespace parallel
} // namespace fastmlm

#endif // FASTMLM_PARALLEL_OPS_H
