#include "profiled_deviance.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fastmlm {

ProfiledDeviance::ProfiledDeviance(MLMData& data, bool REML,
                                   int pcg_threshold, int n_probes)
    : data_(data), REML_(REML),
      chol_initialized_(false),
      crossed_solver_(data, pcg_threshold),
      n_probes_(n_probes),
      pattern_initialized_(false),
      ztx_initialized_(false),
      beta_(VectorXd::Zero(data.p)),
      u_(VectorXd::Zero(data.q)),
      sigma2_(0.0), deviance_(0.0),
      ldL2_(0.0), ldRX2_(0.0), pwrss_(0.0)
{
    use_pcg_ = crossed_solver_.use_pcg();

    // Precompute ZtX once (never changes)
    ZtX_ = data_.Zt * data_.X;  // q x p, sparse * dense
    ztx_initialized_ = true;
}

double ProfiledDeviance::operator()(const VectorXd& theta) {
    if (use_pcg_) {
        return eval_pcg(theta);
    } else {
        return eval_cholesky(theta);
    }
}

// ============================================================================
// Pattern initialisation and fast A update
// ============================================================================

void ProfiledDeviance::init_pattern(const SpMatd& A) {
    // Store a copy of A with its sparsity pattern
    A_pattern_ = A;
    pattern_initialized_ = true;
}

void ProfiledDeviance::update_A_fast(const VectorXd& theta) {
    // Update Lambdat values from theta
    data_.update_Lambdat(theta);
    const SpMatd& Lamt = data_.Lambdat;

    if (!pattern_initialized_) {
        // First call: compute A the slow way and cache the pattern
        SpMatd LZZLt = Lamt * data_.ZtZ * Lamt.transpose();
        A_pattern_ = LZZLt;
        // Add identity to diagonal
        for (int j = 0; j < data_.q; ++j) {
            A_pattern_.coeffRef(j, j) += 1.0;
        }
        pattern_initialized_ = true;
        return;
    }

    // Fast path: recompute A values using the precomputed pattern.
    // A = Lamt * ZtZ * Lamt^T + I
    //
    // Since Lambdat is block-diagonal and its pattern is fixed,
    // we can compute the product efficiently:
    //   A(i,j) = sum_k sum_l Lamt(i,k) * ZtZ(k,l) * Lamt(j,l) + (i==j)
    //
    // For the product Lamt * ZtZ * Lamt^T, we compute it as:
    //   tmp = Lamt * ZtZ   (sparse * sparse, but Lamt is very sparse — diagonal or block-diag)
    //   A = tmp * Lamt^T + I
    //
    // Key insight: Lamt is block-diagonal with tiny blocks (1x1 or 2x2 etc.),
    // so Lamt * ZtZ has the SAME sparsity pattern as ZtZ (just scaled rows).
    // This means the product Lamt * ZtZ is just a row-scaling of ZtZ.

    // Compute Lamt * ZtZ as a row-scaled version of ZtZ
    // For each nonzero ZtZ(k, l), the product Lamt * ZtZ has entry:
    //   (Lamt * ZtZ)(i, l) = sum_k Lamt(i, k) * ZtZ(k, l)
    // Since Lamt is block-diagonal, only Lamt(i, k) where k is in the same block as i is nonzero.

    // For efficiency: form A = Lamt * ZtZ * Lamt^T + I directly
    // using sparse operations but reusing the pattern
    SpMatd tmp = Lamt * data_.ZtZ;
    SpMatd LZZLt = tmp * Lamt.transpose();

    // Copy values into pre-allocated pattern (preserving structure)
    // Add identity to diagonal
    double* Ax = A_pattern_.valuePtr();
    const double* Px = LZZLt.valuePtr();
    int nnz = A_pattern_.nonZeros();

    // If the sparsity pattern matches exactly, just copy + add I
    if (LZZLt.nonZeros() == nnz) {
        std::memcpy(Ax, Px, nnz * sizeof(double));
        // Add 1 to diagonal
        for (int j = 0; j < data_.q; ++j) {
            A_pattern_.coeffRef(j, j) += 1.0;
        }
    } else {
        // Pattern changed (shouldn't happen, but fallback)
        A_pattern_ = LZZLt;
        for (int j = 0; j < data_.q; ++j) {
            A_pattern_.coeffRef(j, j) += 1.0;
        }
    }
}

// ============================================================================
// Path A: Direct sparse Cholesky
// ============================================================================

double ProfiledDeviance::eval_cholesky(const VectorXd& theta) {
    const int n = data_.n;
    const int p = data_.p;
    const int q = data_.q;

    // Step 1-2: Update Lambdat and form A efficiently
    update_A_fast(theta);
    const SpMatd& Lamt = data_.Lambdat;

    // Step 3: Cholesky (symbolic analysis once)
    if (!chol_initialized_) {
        L_chol_.analyze(A_pattern_);
        chol_initialized_ = true;
    }
    ldL2_ = L_chol_.factorize(A_pattern_);

    // Step 4: cu = L^{-1} (Lambdat * Zty)
    // Lamt * Zty: Lamt is block-diagonal, so this is just row-scaling of Zty
    VectorXd LamtZty = Lamt * data_.Zty;
    cu_ = L_chol_.solve_L(LamtZty);

    // Step 5: RZX = L^{-1} (Lambdat * ZtX)
    // Lamt * ZtX: row-scaling of precomputed ZtX_
    MatrixXd LamtZtX = Lamt * ZtX_;  // sparse * dense, Lamt very sparse
    RZX_ = MatrixXd(q, p);
    for (int j = 0; j < p; ++j) {
        RZX_.col(j) = L_chol_.solve_L(LamtZtX.col(j));
    }

    // Step 6: RXtRX = XtX - RZX^T * RZX
    MatrixXd RXtRX = data_.XtX - RZX_.transpose() * RZX_;

    // Step 7: Dense Cholesky of RXtRX
    RX_chol_.compute(RXtRX);
    if (RX_chol_.info() != Eigen::Success) {
        deviance_ = std::numeric_limits<double>::infinity();
        return deviance_;
    }

    ldRX2_ = 0.0;
    MatrixXd RX = RX_chol_.matrixL();
    for (int j = 0; j < p; ++j) {
        ldRX2_ += std::log(RX(j, j));
    }
    ldRX2_ *= 2.0;

    // Step 8: beta
    VectorXd rhs_beta = data_.Xty - RZX_.transpose() * cu_;
    beta_ = RX_chol_.solve(rhs_beta);

    // Step 9: u
    VectorXd cu_minus_RZXb = cu_ - RZX_ * beta_;
    u_ = L_chol_.solve_Lt(cu_minus_RZXb);

    // Step 10: pwrss
    VectorXd b = Lamt.transpose() * u_;
    VectorXd resid = data_.y - data_.X * beta_ - data_.Zt.transpose() * b;
    pwrss_ = resid.squaredNorm() + u_.squaredNorm();

    // Step 11: deviance
    int df = REML_ ? (n - p) : n;
    sigma2_ = pwrss_ / df;
    deviance_ = df * (1.0 + std::log(2.0 * M_PI * sigma2_)) + ldL2_;
    if (REML_) {
        deviance_ += ldRX2_;
    }

    return deviance_;
}

// ============================================================================
// Path B: PCG for large crossed random effects
// ============================================================================

double ProfiledDeviance::eval_pcg(const VectorXd& theta) {
    const int n = data_.n;
    const int p = data_.p;
    const int q = data_.q;

    // Update A
    update_A_fast(theta);
    const SpMatd& Lamt = data_.Lambdat;

    // PCG solves + stochastic log-det
    VectorXd LamtZty = Lamt * data_.Zty;
    MatrixXd LamtZtX = Lamt * ZtX_;

    CrossedRESolver::DevianceComponents dc =
        crossed_solver_.compute(A_pattern_, LamtZty, LamtZtX, n_probes_);

    cu_ = dc.cu;
    RZX_ = dc.RZX;
    ldL2_ = dc.logdet_A;

    MatrixXd RXtRX = data_.XtX - LamtZtX.transpose() * dc.RZX;

    RX_chol_.compute(RXtRX);
    if (RX_chol_.info() != Eigen::Success) {
        deviance_ = std::numeric_limits<double>::infinity();
        return deviance_;
    }

    ldRX2_ = 0.0;
    MatrixXd RX = RX_chol_.matrixL();
    for (int j = 0; j < p; ++j) {
        ldRX2_ += std::log(RX(j, j));
    }
    ldRX2_ *= 2.0;

    VectorXd rhs_beta = data_.Xty - LamtZtX.transpose() * dc.cu;
    beta_ = RX_chol_.solve(rhs_beta);

    VectorXd rhs_u = LamtZty - LamtZtX * beta_;
    BlockDiagonalPreconditioner precond;
    precond.compute(A_pattern_, data_.block_starts, data_.block_sizes);
    PCGSolver::Result u_result = PCGSolver::solve(A_pattern_, rhs_u, precond);
    u_ = u_result.x;

    VectorXd b = Lamt.transpose() * u_;
    VectorXd resid = data_.y - data_.X * beta_ - data_.Zt.transpose() * b;
    pwrss_ = resid.squaredNorm() + u_.squaredNorm();

    int df = REML_ ? (n - p) : n;
    sigma2_ = pwrss_ / df;
    deviance_ = df * (1.0 + std::log(2.0 * M_PI * sigma2_)) + ldL2_;
    if (REML_) {
        deviance_ += ldRX2_;
    }

    return deviance_;
}

MatrixXd ProfiledDeviance::vcov_beta_unscaled() const {
    MatrixXd I_p = MatrixXd::Identity(data_.p, data_.p);
    return RX_chol_.solve(I_p);
}

} // namespace fastmlm
