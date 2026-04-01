#ifndef FASTMLM_SPARSE_CHOLESKY_H
#define FASTMLM_SPARSE_CHOLESKY_H

#include "fastmlm_types.h"

namespace fastmlm {

// Wrapper around Eigen's SimplicialLLT that separates symbolic analysis
// (expensive, done once) from numeric factorisation (cheap, done every iteration).
class SparseCholeskyManager {
public:
    SparseCholeskyManager() : analyzed_(false) {}

    // Symbolic analysis of sparsity pattern — call once
    void analyze(const SpMatd& pattern);

    // Numeric factorisation — call each optimizer iteration
    // Returns 2 * log|L| (the log-determinant contribution)
    double factorize(const SpMatd& A);

    // Solve L x = b (lower triangular)
    VectorXd solve_L(const VectorXd& b) const;

    // Solve L^T x = b (upper triangular)
    VectorXd solve_Lt(const VectorXd& b) const;

    // Solve A x = b (full: L L^T x = b)
    VectorXd solve(const VectorXd& b) const;

    // Solve A X = B (multiple RHS)
    MatrixXd solve(const MatrixXd& B) const;

    // Permutation vector (for applying to RHS before/after solve)
    const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>&
    permutation() const;

    bool is_analyzed() const { return analyzed_; }

    // Access the internal solver for advanced use
    const Eigen::SimplicialLLT<SpMatd, Eigen::Lower, Eigen::AMDOrdering<int>>&
    solver() const { return solver_; }

private:
    Eigen::SimplicialLLT<SpMatd, Eigen::Lower, Eigen::AMDOrdering<int>> solver_;
    bool analyzed_;
};

} // namespace fastmlm

#endif // FASTMLM_SPARSE_CHOLESKY_H
