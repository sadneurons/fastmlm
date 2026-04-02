#ifndef FASTMLM_GLMM_DEVIANCE_H
#define FASTMLM_GLMM_DEVIANCE_H

#include "model_data.h"
#include "sparse_cholesky.h"

namespace fastmlm {

// GLM family specification
struct GLMFamily {
    enum Type { BINOMIAL, POISSON, GAMMA };
    enum Link { LOGIT, LOG, PROBIT, INVERSE, IDENTITY };

    Type family;
    Link link;

    // Link function: eta = g(mu)
    double link_fun(double mu) const;
    // Inverse link: mu = g^{-1}(eta)
    double linkinv(double eta) const;
    // Derivative of inverse link: d(mu)/d(eta)
    double mu_eta(double eta) const;
    // Variance function: V(mu)
    double variance(double mu) const;
    // Deviance residual contribution for one observation
    double dev_resid(double y, double mu, double wt) const;
};

// PIRLS (Penalised Iteratively Reweighted Least Squares) deviance
// for GLMMs with Laplace approximation.
//
// The Laplace-approximated deviance is:
//   dev(theta) = sum(dev_resid(y, mu)) + ||u||^2 + log|L|^2
//
// where mu = linkinv(X*beta + Z*b), b = Lambdat^T * u,
// and L is the Cholesky factor of the penalised working system.
class PIRLSDeviance {
public:
    PIRLSDeviance(MLMData& data, const GLMFamily& family,
                  const VectorXd& prior_weights = VectorXd(),
                  int pirls_maxiter = 30, double pirls_tol = 1e-10);

    // Evaluate GLMM deviance at theta
    double operator()(const VectorXd& theta);

    // Accessors (valid after evaluation)
    const VectorXd& beta() const { return beta_; }
    const VectorXd& u() const { return u_; }
    const VectorXd& eta() const { return eta_; }
    const VectorXd& mu() const { return mu_; }
    double deviance() const { return deviance_; }
    double ldL2() const { return ldL2_; }
    int pirls_iterations() const { return pirls_iter_; }

    // Variance-covariance of beta (approximate, from last PIRLS iteration)
    MatrixXd vcov_beta() const;

private:
    MLMData& data_;
    GLMFamily family_;
    VectorXd prior_weights_;
    int pirls_maxiter_;
    double pirls_tol_;

    // Cholesky for the penalised system
    SparseCholeskyManager L_chol_;
    bool chol_initialized_;

    // Cached results
    VectorXd beta_;
    VectorXd u_;
    VectorXd eta_;      // linear predictor
    VectorXd mu_;       // fitted means
    double deviance_;
    double ldL2_;
    int pirls_iter_;

    // Working storage
    Eigen::LLT<MatrixXd> RX_chol_;

    // Precomputed
    MatrixXd ZtX_;
    bool ztx_initialized_;


    // Run PIRLS iterations for fixed theta
    void pirls(const SpMatd& Lamt);
};

} // namespace fastmlm

#endif // FASTMLM_GLMM_DEVIANCE_H
