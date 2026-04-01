#ifndef FASTMLM_CROSSED_RE_H
#define FASTMLM_CROSSED_RE_H

#include "model_data.h"
#include <random>

namespace fastmlm {

// Block-diagonal preconditioner for PCG.
//
// For crossed random effects, A = Lambdat * ZtZ * Lambdat^T + I is not
// block-diagonal, but its diagonal blocks (one per RE term) provide a
// good preconditioner. Each block is factored independently via dense
// Cholesky, then M^{-1} v is computed by solving each block separately.
class BlockDiagonalPreconditioner {
public:
    BlockDiagonalPreconditioner() : computed_(false) {}

    // Compute the preconditioner from the diagonal blocks of A
    void compute(const SpMatd& A, const VectorXi& block_starts,
                 const VectorXi& block_sizes);

    // Apply preconditioner: solve M z = r
    VectorXd solve(const VectorXd& r) const;

    bool is_computed() const { return computed_; }

private:
    std::vector<Eigen::LLT<MatrixXd>> block_factors_;
    VectorXi block_starts_;
    VectorXi block_sizes_;
    bool computed_;
};

// Preconditioned Conjugate Gradient solver for the system
//   A x = b
// where A = Lambdat * ZtZ * Lambdat^T + I.
//
// For crossed random effects, A is sparse but has significant fill-in
// under Cholesky factorisation. PCG avoids this by using only matrix-
// vector products with A, which preserve sparsity.
class PCGSolver {
public:
    struct Result {
        VectorXd x;          // solution
        int iterations;      // PCG iterations used
        double residual_norm;// final residual norm
        bool converged;
    };

    // Solve A x = b using PCG with block-diagonal preconditioner
    static Result solve(const SpMatd& A,
                        const VectorXd& b,
                        const BlockDiagonalPreconditioner& precond,
                        double tol = 1e-10,
                        int maxiter = 500);

    // Solve A X = B (multiple right-hand sides)
    static MatrixXd solve_multi(const SpMatd& A,
                                const MatrixXd& B,
                                const BlockDiagonalPreconditioner& precond,
                                double tol = 1e-10,
                                int maxiter = 500);
};

// Stochastic log-determinant estimation via Hutchinson's trace estimator.
//
// log|A| = trace(log(A))
//        ≈ (1/n_probes) * sum_i z_i^T log(A) z_i
//
// where z_i are Rademacher random vectors (+1/-1 with equal probability).
//
// We approximate log(A) z_i using a Chebyshev polynomial expansion of
// the matrix logarithm applied via matrix-vector products with A.
//
// For small-to-moderate q (<= pcg_threshold), we fall back to direct
// sparse Cholesky which gives the exact log-determinant.
class StochasticLogDet {
public:
    // Estimate log|A| using n_probes Rademacher vectors
    // and chebyshev_order polynomial terms.
    //
    // The PCG solver is used to compute A^{-1} z needed for the
    // Hutchinson estimator of trace(A^{-1}) which relates to
    // d/dt log|A + tI| at t=0.
    //
    // For better accuracy, we use the identity:
    //   log|A| = sum of log of eigenvalues
    // approximated via stochastic Lanczos quadrature (SLQ).
    static double estimate(const SpMatd& A,
                           const BlockDiagonalPreconditioner& precond,
                           int n_probes = 30,
                           int lanczos_steps = 30,
                           unsigned int seed = 42);

    // Stochastic Lanczos Quadrature for log-determinant
    // Runs Lanczos iteration to tridiagonalise A using random start vector,
    // then computes log-determinant contribution from eigenvalues of
    // the tridiagonal matrix.
    static double slq_logdet_probe(const SpMatd& A, const VectorXd& z,
                                   int lanczos_steps);
};

// High-level crossed RE solver that integrates PCG + stochastic logdet
// into the profiled deviance computation.
class CrossedRESolver {
public:
    CrossedRESolver(const MLMData& data, int pcg_threshold = 5000);

    // Should we use the PCG path?
    bool use_pcg() const { return use_pcg_; }

    // Compute the PCG-based deviance components:
    //   - Solve for cu = A^{-1} Lambdat Zty
    //   - Solve for RZX columns = A^{-1} Lambdat ZtX
    //   - Estimate log|A|
    struct DevianceComponents {
        VectorXd cu;          // q-vector
        MatrixXd RZX;         // q x p matrix (NOT L^{-1} form, but A^{-1} form)
        double logdet_A;      // log|A| (stochastic estimate)
        int pcg_iterations;   // total PCG iterations across all solves
    };

    DevianceComponents compute(const SpMatd& A,
                               const VectorXd& LamtZty,
                               const MatrixXd& LamtZtX,
                               int n_probes = 30);

private:
    const MLMData& data_;
    int pcg_threshold_;
    bool use_pcg_;
    BlockDiagonalPreconditioner precond_;
};

} // namespace fastmlm

#endif // FASTMLM_CROSSED_RE_H
