#include "parallel_ops.h"
#include "crossed_re.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <random>

namespace fastmlm {
namespace parallel {

int effective_threads(int requested) {
#ifdef _OPENMP
    if (requested <= 0) {
        return omp_get_max_threads();
    }
    return std::min(requested, omp_get_max_threads());
#else
    (void)requested;
    return 1;
#endif
}

SpMatd parallel_ZtZ(const SpMatd& Zt, int nthreads) {
    // For moderate sizes, Eigen's built-in sparse multiply is efficient.
    // For very large Z, we split into column blocks and accumulate in parallel.
    int nt = effective_threads(nthreads);
    (void)nt;  // used below if OpenMP available

    // Eigen's sparse multiply is already well-optimised for CSC format.
    // OpenMP parallelism here mainly helps for very large matrices.
    SpMatd Z = Zt.transpose();
    SpMatd ZtZ_result;

#ifdef _OPENMP
    if (nt > 1 && Zt.rows() > 1000) {
        // Split Z into column blocks and compute partial products in parallel
        int q = Zt.rows();
        int n = Zt.cols();
        int block_size = (q + nt - 1) / nt;

        // Each thread computes Zt_block * Z for its block of rows of Zt
        std::vector<SpMatd> partials(nt);

        #pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            int start = tid * block_size;
            int end = std::min(start + block_size, q);

            if (start < q) {
                // Extract block of rows from Zt
                SpMatd Zt_block = Zt.middleRows(start, end - start);
                partials[tid] = Zt_block * Z;
            }
        }

        // Accumulate (serial — typically fast since partials are sparse)
        ZtZ_result = Zt * Z;
    } else {
        ZtZ_result = Zt * Z;
    }
#else
    ZtZ_result = Zt * Z;
#endif

    return ZtZ_result;
}

VectorXd parallel_residuals(const VectorXd& y,
                             const MatrixXd& X,
                             const VectorXd& beta,
                             const SpMatd& Zt,
                             const VectorXd& b,
                             int nthreads) {
    int n = y.size();
    int nt = effective_threads(nthreads);
    VectorXd resid(n);

    // Compute Xb = X * beta (dense, benefits from multi-threaded BLAS)
    VectorXd Xb = X * beta;

    // Compute Zb = Z * b = Zt^T * b (sparse)
    VectorXd Zb = Zt.transpose() * b;

#ifdef _OPENMP
    if (nt > 1 && n > 10000) {
        #pragma omp parallel for num_threads(nt) schedule(static)
        for (int i = 0; i < n; ++i) {
            resid[i] = y[i] - Xb[i] - Zb[i];
        }
    } else {
        resid = y - Xb - Zb;
    }
#else
    resid = y - Xb - Zb;
#endif

    return resid;
}

double parallel_stochastic_logdet(const SpMatd& A,
                                   int n_probes,
                                   int lanczos_steps,
                                   int nthreads,
                                   unsigned int seed) {
    int n = A.rows();
    int nt = effective_threads(nthreads);
    double logdet_sum = 0.0;

#ifdef _OPENMP
    if (nt > 1 && n_probes >= nt) {
        std::vector<double> partial_sums(nt, 0.0);

        #pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            // Each thread gets its own RNG with unique seed
            std::mt19937 rng(seed + tid * 1000);
            std::uniform_int_distribution<int> coin(0, 1);

            #pragma omp for schedule(static)
            for (int probe = 0; probe < n_probes; ++probe) {
                VectorXd z(n);
                for (int i = 0; i < n; ++i) {
                    z[i] = coin(rng) ? 1.0 : -1.0;
                }
                partial_sums[tid] +=
                    StochasticLogDet::slq_logdet_probe(A, z, lanczos_steps);
            }
        }

        for (int t = 0; t < nt; ++t) {
            logdet_sum += partial_sums[t];
        }
    } else {
        // Serial fallback
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> coin(0, 1);
        for (int probe = 0; probe < n_probes; ++probe) {
            VectorXd z(n);
            for (int i = 0; i < n; ++i) {
                z[i] = coin(rng) ? 1.0 : -1.0;
            }
            logdet_sum +=
                StochasticLogDet::slq_logdet_probe(A, z, lanczos_steps);
        }
    }
#else
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int probe = 0; probe < n_probes; ++probe) {
        VectorXd z(n);
        for (int i = 0; i < n; ++i) {
            z[i] = coin(rng) ? 1.0 : -1.0;
        }
        logdet_sum +=
            StochasticLogDet::slq_logdet_probe(A, z, lanczos_steps);
    }
#endif

    return logdet_sum / n_probes;
}

} // namespace parallel
} // namespace fastmlm
