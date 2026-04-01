#include "sparse_cholesky.h"
#include <cmath>
#include <stdexcept>

namespace fastmlm {

void SparseCholeskyManager::analyze(const SpMatd& pattern) {
    solver_.analyzePattern(pattern);
    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky: symbolic analysis failed");
    }
    analyzed_ = true;
}

double SparseCholeskyManager::factorize(const SpMatd& A) {
    if (!analyzed_) {
        throw std::runtime_error("Sparse Cholesky: must call analyze() before factorize()");
    }

    solver_.factorize(A);
    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky: numeric factorisation failed "
                                 "(matrix may not be positive definite)");
    }

    // Compute 2 * log|L| = sum of log of diagonal entries of L
    // L is lower triangular, so |A| = |L|^2 and log|A| = 2 * sum(log(L_ii))
    const SpMatd& L = solver_.matrixL();
    double ldL2 = 0.0;
    for (int j = 0; j < L.cols(); ++j) {
        // Diagonal of L in column j
        // For SimplicialLLT, the diagonal entry is at the start of each column
        double Ljj = L.coeff(j, j);
        ldL2 += std::log(Ljj);
    }
    ldL2 *= 2.0;

    return ldL2;
}

VectorXd SparseCholeskyManager::solve_L(const VectorXd& b) const {
    // Solve L x = P b where P is the fill-reducing permutation
    const SpMatd& L = solver_.matrixL();
    VectorXd Pb = solver_.permutationP() * b;
    // Forward substitution
    return L.triangularView<Eigen::Lower>().solve(Pb);
}

VectorXd SparseCholeskyManager::solve_Lt(const VectorXd& b) const {
    // Solve L^T x = b, then apply P^T
    const SpMatd& L = solver_.matrixL();
    VectorXd x = L.transpose().triangularView<Eigen::Upper>().solve(b);
    return solver_.permutationPinv() * x;
}

VectorXd SparseCholeskyManager::solve(const VectorXd& b) const {
    return solver_.solve(b);
}

MatrixXd SparseCholeskyManager::solve(const MatrixXd& B) const {
    return solver_.solve(B);
}

const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>&
SparseCholeskyManager::permutation() const {
    return solver_.permutationP();
}

} // namespace fastmlm
