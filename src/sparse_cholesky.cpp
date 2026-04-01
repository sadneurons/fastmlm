#include "sparse_cholesky.h"
#include <cmath>
#include <stdexcept>

namespace fastmlm {

SparseCholeskyManager::SparseCholeskyManager(int cholmod_threshold)
    : cholmod_threshold_(cholmod_threshold),
      use_cholmod_(false),
      analyzed_(false)
{
    // Default threshold of 300: CHOLMOD's supernodal Cholesky wins for
    // larger matrices, but the Eigen→CHOLMOD conversion overhead makes it
    // slower for small q. Threshold of ~300 is the crossover point.
}

void SparseCholeskyManager::analyze(const SpMatd& pattern) {
    int q = pattern.rows();
    use_cholmod_ = (q > cholmod_threshold_);

    if (use_cholmod_) {
        cholmod_solver_.analyze(pattern);
    } else {
        eigen_solver_.analyzePattern(pattern);
        if (eigen_solver_.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky: symbolic analysis failed");
        }
    }

    analyzed_ = true;
}

double SparseCholeskyManager::factorize(const SpMatd& A) {
    if (!analyzed_) {
        throw std::runtime_error("Sparse Cholesky: must call analyze() before factorize()");
    }

    if (use_cholmod_) {
        return cholmod_solver_.factorize(A);
    }

    // Eigen path
    eigen_solver_.factorize(A);
    if (eigen_solver_.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky: numeric factorisation failed");
    }

    const SpMatd& L = eigen_solver_.matrixL();
    double ldL2 = 0.0;
    for (int j = 0; j < L.cols(); ++j) {
        ldL2 += std::log(L.coeff(j, j));
    }
    ldL2 *= 2.0;
    return ldL2;
}

VectorXd SparseCholeskyManager::solve_L(const VectorXd& b) const {
    if (use_cholmod_) {
        return cholmod_solver_.solve_L(b);
    }

    const SpMatd& L = eigen_solver_.matrixL();
    VectorXd Pb = eigen_solver_.permutationP() * b;
    return L.triangularView<Eigen::Lower>().solve(Pb);
}

VectorXd SparseCholeskyManager::solve_Lt(const VectorXd& b) const {
    if (use_cholmod_) {
        return cholmod_solver_.solve_Lt(b);
    }

    const SpMatd& L = eigen_solver_.matrixL();
    VectorXd x = L.transpose().triangularView<Eigen::Upper>().solve(b);
    return eigen_solver_.permutationPinv() * x;
}

VectorXd SparseCholeskyManager::solve(const VectorXd& b) const {
    if (use_cholmod_) {
        return cholmod_solver_.solve(b);
    }
    return eigen_solver_.solve(b);
}

MatrixXd SparseCholeskyManager::solve(const MatrixXd& B) const {
    if (use_cholmod_) {
        return cholmod_solver_.solve(B);
    }
    return eigen_solver_.solve(B);
}

} // namespace fastmlm
