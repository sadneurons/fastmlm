#ifndef FASTMLM_PROFILED_DEVIANCE_H
#define FASTMLM_PROFILED_DEVIANCE_H

#include "model_data.h"
#include "sparse_cholesky.h"
#include "crossed_re.h"

namespace fastmlm {

// Evaluates the profiled REML or ML deviance as a function of theta.
// This is the hot loop — every optimizer iteration calls operator().
//
// Two computational paths:
//   (A) Direct sparse Cholesky — for nested RE or small crossed RE (default)
//   (B) PCG + stochastic log-det — for large crossed RE (q > pcg_threshold)
//
// Algorithm (Bates et al. 2015, JSS):
//   1. Update Lambdat from theta
//   2. Form A = Lambdat * ZtZ * Lambdat^T + I
//   3. Decompose A (Cholesky or PCG)
//   4. Solve for cu, RZX
//   5. Form RXtRX = XtX - RZX^T RZX (path A) or XtX - LamtZtX^T A^{-1} LamtZtX (path B)
//   6. Dense Cholesky of RXtRX → beta
//   7. Compute u, pwrss, deviance
class ProfiledDeviance {
public:
    ProfiledDeviance(MLMData& data, bool REML = true,
                     int pcg_threshold = 5000, int n_probes = 30);

    // Evaluate profiled deviance at theta
    double operator()(const VectorXd& theta);

    // Accessors (valid after a call to operator())
    const VectorXd& beta() const { return beta_; }
    const VectorXd& u() const { return u_; }
    double sigma2() const { return sigma2_; }
    double deviance() const { return deviance_; }
    double ldL2() const { return ldL2_; }
    double ldRX2() const { return ldRX2_; }
    double pwrss() const { return pwrss_; }

    // Variance-covariance of beta (unscaled)
    MatrixXd vcov_beta_unscaled() const;

    bool is_REML() const { return REML_; }
    bool using_pcg() const { return use_pcg_; }

private:
    MLMData& data_;
    bool REML_;
    bool use_pcg_;
    int n_probes_;

    // === Path A: Direct Cholesky ===
    SparseCholeskyManager L_chol_;
    bool chol_initialized_;

    // === Path B: PCG for crossed RE ===
    CrossedRESolver crossed_solver_;

    // Dense Cholesky for RXtRX (p x p), used by both paths
    Eigen::LLT<MatrixXd> RX_chol_;

    // Cached results
    VectorXd beta_;
    VectorXd u_;
    double sigma2_;
    double deviance_;
    double ldL2_;
    double ldRX2_;
    double pwrss_;

    // Working storage
    SpMatd LamtZtZLamt_;
    MatrixXd RZX_;
    VectorXd cu_;

    // Path dispatch
    double eval_cholesky(const VectorXd& theta);
    double eval_pcg(const VectorXd& theta);
};

} // namespace fastmlm

#endif // FASTMLM_PROFILED_DEVIANCE_H
