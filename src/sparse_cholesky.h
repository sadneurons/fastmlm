#ifndef FASTMLM_SPARSE_CHOLESKY_H
#define FASTMLM_SPARSE_CHOLESKY_H

#include "fastmlm_types.h"
#include "cholmod_wrapper.h"

namespace fastmlm {

// Sparse Cholesky manager with automatic backend selection.
//
// For small matrices (q <= threshold): uses Eigen's SimplicialLLT
//   - Header-only, zero dependencies, works with AD
//   - Adequate performance for typical MLM sizes
//
// For large matrices (q > threshold): uses CHOLMOD supernodal via
//   R's Matrix package
//   - Supernodal Cholesky with dense BLAS kernels on interior blocks
//   - Same engine lme4 uses internally
//   - Significantly faster for q > ~500
//
// Both backends expose the same interface: analyze (once) + factorize (many).
class SparseCholeskyManager {
public:
    SparseCholeskyManager(int cholmod_threshold = 300);

    // Symbolic analysis of sparsity pattern — call once
    void analyze(const SpMatd& pattern);

    // Numeric factorisation — call each optimizer iteration
    // Returns 2 * log|L| (the log-determinant contribution)
    double factorize(const SpMatd& A);

    // Solve L x = P b (lower triangular)
    VectorXd solve_L(const VectorXd& b) const;

    // Solve L^T x = b, then apply P^T
    VectorXd solve_Lt(const VectorXd& b) const;

    // Solve A x = b (full: L L^T x = b)
    VectorXd solve(const VectorXd& b) const;

    // Solve A X = B (multiple RHS)
    MatrixXd solve(const MatrixXd& B) const;

    bool is_analyzed() const { return analyzed_; }
    bool using_cholmod() const { return use_cholmod_; }

private:
    int cholmod_threshold_;
    bool use_cholmod_;
    bool analyzed_;

    // Backend A: Eigen SimplicialLLT
    Eigen::SimplicialLLT<SpMatd, Eigen::Lower, Eigen::AMDOrdering<int>> eigen_solver_;

    // Backend B: CHOLMOD supernodal
    CholmodWrapper cholmod_solver_;
};

} // namespace fastmlm

#endif // FASTMLM_SPARSE_CHOLESKY_H
