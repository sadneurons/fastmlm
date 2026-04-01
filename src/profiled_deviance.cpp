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

void ProfiledDeviance::init_pattern(const SpMatd& /* unused */) {}

void ProfiledDeviance::update_A_fast(const VectorXd& theta) {
    data_.update_Lambdat(theta);
    const SpMatd& Lamt = data_.Lambdat;
    const SpMatd& ZtZ = data_.ZtZ;
    const int q = data_.q;

    if (!pattern_initialized_) {
        // First call: compute A via sparse multiply to get the sparsity pattern
        SpMatd LZZLt = Lamt * ZtZ * Lamt.transpose();
        A_pattern_ = LZZLt;
        for (int j = 0; j < q; ++j) {
            A_pattern_.coeffRef(j, j) += 1.0;
        }
        A_pattern_.makeCompressed();

        // Check if all RE terms are random-intercept-only (Lambdat is diagonal)
        is_all_intercept_ = (Lamt.nonZeros() == q);

        // Build the precomputed mapping for fast updates
        const int* Ap = A_pattern_.outerIndexPtr();
        const int* Ai = A_pattern_.innerIndexPtr();
        int nnz = A_pattern_.nonZeros();
        a_map_.resize(nnz);

        for (int col = 0; col < q; ++col) {
            for (int idx = Ap[col]; idx < Ap[col + 1]; ++idx) {
                int row = Ai[idx];
                AMapping& m = a_map_[idx];
                m.is_diag = (row == col);

                if (is_all_intercept_) {
                    // For diagonal Lambdat: A(i,j) = Lamt(i,i)*ZtZ(i,j)*Lamt(j,j)
                    // Find ZtZ(row, col) index by scanning ZtZ's column 'col'
                    m.lamt_row = row;  // Lambdat diagonal position
                    m.lamt_col = col;
                    m.ztz_idx = -1;
                    for (SpMatd::InnerIterator it(ZtZ, col); it; ++it) {
                        if (it.row() == row) {
                            // Store the raw index into ZtZ.valuePtr()
                            m.ztz_idx = static_cast<int>(&it.value() - ZtZ.valuePtr());
                            break;
                        }
                    }
                }
            }
        }

        pattern_initialized_ = true;
        return;
    }

    // === Fast path: update A values with NO sparse matrix multiply ===

    double* Ax = A_pattern_.valuePtr();
    int nnz = A_pattern_.nonZeros();

    if (is_all_intercept_) {
        // Random-intercept fast path: A(i,j) = Lamt(i,i)*ZtZ(i,j)*Lamt(j,j) + delta(i,j)
        // Lamt diagonal values are directly in Lamt.valuePtr() at positions [0..q-1]
        // since diagonal Lambdat has exactly one entry per column.
        const double* Lx = Lamt.valuePtr();
        const double* Zx = ZtZ.valuePtr();

        for (int idx = 0; idx < nnz; ++idx) {
            const AMapping& m = a_map_[idx];
            double val = (m.is_diag) ? 1.0 : 0.0;
            if (m.ztz_idx >= 0) {
                val += Lx[m.lamt_row] * Zx[m.ztz_idx] * Lx[m.lamt_col];
            }
            Ax[idx] = val;
        }
    } else {
        // General case: fall back to sparse triple product
        // (still much faster than before since we reuse A_pattern_ storage)
        SpMatd LZZLt = Lamt * ZtZ * Lamt.transpose();

        if (LZZLt.nonZeros() == nnz) {
            const double* Px = LZZLt.valuePtr();
            std::memcpy(Ax, Px, nnz * sizeof(double));
            for (int j = 0; j < q; ++j) {
                A_pattern_.coeffRef(j, j) += 1.0;
            }
        } else {
            A_pattern_ = LZZLt;
            for (int j = 0; j < q; ++j) {
                A_pattern_.coeffRef(j, j) += 1.0;
            }
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
