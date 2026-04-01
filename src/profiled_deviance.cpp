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
      beta_(VectorXd::Zero(data.p)),
      u_(VectorXd::Zero(data.q)),
      sigma2_(0.0), deviance_(0.0),
      ldL2_(0.0), ldRX2_(0.0), pwrss_(0.0)
{
    use_pcg_ = crossed_solver_.use_pcg();
}

double ProfiledDeviance::operator()(const VectorXd& theta) {
    if (use_pcg_) {
        return eval_pcg(theta);
    } else {
        return eval_cholesky(theta);
    }
}

// ============================================================================
// Path A: Direct sparse Cholesky (nested RE or small crossed RE)
// ============================================================================

double ProfiledDeviance::eval_cholesky(const VectorXd& theta) {
    const int n = data_.n;
    const int p = data_.p;
    const int q = data_.q;

    // Step 1: Update Lambdat from theta
    data_.update_Lambdat(theta);
    const SpMatd& Lamt = data_.Lambdat;

    // Step 2: Form A = Lambdat * ZtZ * Lambdat^T + I
    SpMatd I_q(q, q);
    I_q.setIdentity();
    LamtZtZLamt_ = Lamt * data_.ZtZ * Lamt.transpose();
    SpMatd A = LamtZtZLamt_ + I_q;

    // Step 3: Cholesky (symbolic analysis once, numeric each iteration)
    if (!chol_initialized_) {
        L_chol_.analyze(A);
        chol_initialized_ = true;
    }
    ldL2_ = L_chol_.factorize(A);

    // Step 4: cu = L^{-1} (Lambdat * Zty)
    VectorXd LamtZty = Lamt * data_.Zty;
    cu_ = L_chol_.solve_L(LamtZty);

    // Step 5: RZX = L^{-1} (Lambdat * ZtX)
    MatrixXd LamtZtX = Lamt * (data_.Zt * data_.X);
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

    // ldRX2 = 2 * sum(log(diag(RX)))
    ldRX2_ = 0.0;
    MatrixXd RX = RX_chol_.matrixL();
    for (int j = 0; j < p; ++j) {
        ldRX2_ += std::log(RX(j, j));
    }
    ldRX2_ *= 2.0;

    // Step 8: beta = RXtRX^{-1} (Xty - RZX^T cu)
    VectorXd rhs_beta = data_.Xty - RZX_.transpose() * cu_;
    beta_ = RX_chol_.solve(rhs_beta);

    // Step 9: u = L^{-T} (cu - RZX * beta)
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

    // Step 1: Update Lambdat
    data_.update_Lambdat(theta);
    const SpMatd& Lamt = data_.Lambdat;

    // Step 2: Form A
    SpMatd I_q(q, q);
    I_q.setIdentity();
    LamtZtZLamt_ = Lamt * data_.ZtZ * Lamt.transpose();
    SpMatd A = LamtZtZLamt_ + I_q;

    // Steps 3-5: PCG solves + stochastic log-det
    VectorXd LamtZty = Lamt * data_.Zty;
    MatrixXd LamtZtX = Lamt * (data_.Zt * data_.X);

    CrossedRESolver::DevianceComponents dc =
        crossed_solver_.compute(A, LamtZty, LamtZtX, n_probes_);

    // For PCG path:
    //   cu = A^{-1} LamtZty  (NOT L^{-1}, but full solve)
    //   RZX = A^{-1} LamtZtX (NOT L^{-1}, but full solve)
    //   ldL2 = logdet_A (stochastic estimate of log|A|)
    //
    // The profiled deviance in terms of A^{-1} (instead of L^{-1}):
    //   RXtRX = XtX - LamtZtX^T A^{-1} LamtZtX
    //   beta = RXtRX^{-1} (Xty - LamtZtX^T A^{-1} LamtZty)
    //   u = A^{-1} (LamtZty - LamtZtX beta) ... but we already have A^{-1} applied

    cu_ = dc.cu;
    RZX_ = dc.RZX;
    ldL2_ = dc.logdet_A;

    // Step 6: RXtRX = XtX - LamtZtX^T * A^{-1} * LamtZtX
    MatrixXd RXtRX = data_.XtX - LamtZtX.transpose() * dc.RZX;

    // Step 7: Dense Cholesky of RXtRX
    RX_chol_.compute(RXtRX);
    if (RX_chol_.info() != Eigen::Success) {
        deviance_ = std::numeric_limits<double>::infinity();
        return deviance_;
    }

    // ldRX2
    ldRX2_ = 0.0;
    MatrixXd RX = RX_chol_.matrixL();
    for (int j = 0; j < p; ++j) {
        ldRX2_ += std::log(RX(j, j));
    }
    ldRX2_ *= 2.0;

    // Step 8: beta
    VectorXd rhs_beta = data_.Xty - LamtZtX.transpose() * dc.cu;
    beta_ = RX_chol_.solve(rhs_beta);

    // Step 9: u via PCG solve of A u = LamtZty - LamtZtX * beta
    VectorXd rhs_u = LamtZty - LamtZtX * beta_;
    BlockDiagonalPreconditioner precond;
    precond.compute(A, data_.block_starts, data_.block_sizes);
    PCGSolver::Result u_result = PCGSolver::solve(A, rhs_u, precond);
    u_ = u_result.x;

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
// Common
// ============================================================================

MatrixXd ProfiledDeviance::vcov_beta_unscaled() const {
    MatrixXd I_p = MatrixXd::Identity(data_.p, data_.p);
    return RX_chol_.solve(I_p);
}

} // namespace fastmlm
