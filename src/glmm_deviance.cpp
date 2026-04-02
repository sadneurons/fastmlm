#include "glmm_deviance.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fastmlm {

// ============================================================================
// GLMFamily functions
// ============================================================================

double GLMFamily::link_fun(double mu) const {
    switch (link) {
        case LOGIT:    return std::log(mu / (1.0 - mu));
        case LOG:      return std::log(mu);
        case PROBIT:   return 0.0; // TODO: qnorm
        case INVERSE:  return 1.0 / mu;
        case IDENTITY: return mu;
    }
    return mu;
}

double GLMFamily::linkinv(double eta) const {
    switch (link) {
        case LOGIT:    return 1.0 / (1.0 + std::exp(-eta));
        case LOG:      return std::exp(eta);
        case PROBIT:   return 0.0; // TODO: pnorm
        case INVERSE:  return 1.0 / eta;
        case IDENTITY: return eta;
    }
    return eta;
}

double GLMFamily::mu_eta(double eta) const {
    switch (link) {
        case LOGIT: {
            double p = 1.0 / (1.0 + std::exp(-eta));
            return p * (1.0 - p);
        }
        case LOG:      return std::exp(eta);
        case PROBIT:   return 0.0; // TODO
        case INVERSE:  return -1.0 / (eta * eta);
        case IDENTITY: return 1.0;
    }
    return 1.0;
}

double GLMFamily::variance(double mu) const {
    switch (family) {
        case BINOMIAL: return mu * (1.0 - mu);
        case POISSON:  return mu;
        case GAMMA:    return mu * mu;
    }
    return 1.0;
}

double GLMFamily::dev_resid(double y, double mu, double wt) const {
    switch (family) {
        case BINOMIAL: {
            // 2 * wt * (y*log(y/mu) + (1-y)*log((1-y)/(1-mu)))
            double r = 0.0;
            if (y > 0.0) r += y * std::log(y / mu);
            if (y < 1.0) r += (1.0 - y) * std::log((1.0 - y) / (1.0 - mu));
            return 2.0 * wt * r;
        }
        case POISSON: {
            // 2 * wt * (y*log(y/mu) - (y - mu))
            double r = (y > 0.0) ? y * std::log(y / mu) : 0.0;
            return 2.0 * wt * (r - (y - mu));
        }
        case GAMMA: {
            // 2 * wt * (-log(y/mu) + (y - mu)/mu)
            return 2.0 * wt * (-std::log(y / mu) + (y - mu) / mu);
        }
    }
    return 0.0;
}

// ============================================================================
// PIRLSDeviance
// ============================================================================

PIRLSDeviance::PIRLSDeviance(MLMData& data, const GLMFamily& family,
                             const VectorXd& prior_weights,
                             int pirls_maxiter, double pirls_tol)
    : data_(data), family_(family), prior_weights_(prior_weights),
      pirls_maxiter_(pirls_maxiter), pirls_tol_(pirls_tol),
      chol_initialized_(false), ztx_initialized_(false),
      beta_(VectorXd::Zero(data.p)),
      u_(VectorXd::Zero(data.q)),
      eta_(VectorXd::Zero(data.n)),
      mu_(VectorXd::Zero(data.n)),
      deviance_(0.0), ldL2_(0.0), pirls_iter_(0)
{
    ZtX_ = data_.Zt * data_.X;
    ztx_initialized_ = true;

    if (prior_weights_.size() == 0) {
        prior_weights_ = VectorXd::Ones(data_.n);
    }
}

double PIRLSDeviance::operator()(const VectorXd& theta) {
    data_.update_Lambdat(theta);
    const SpMatd& Lamt = data_.Lambdat;

    // Warm start: keep beta from previous theta, reset u.
    u_.setZero();

    // Run PIRLS to find optimal (beta, u) for this theta
    pirls(Lamt);

    // Laplace-approximated deviance:
    // dev = sum(dev_resid_i) + ||u||^2 + ldL2
    double dev_sum = 0.0;
    for (int i = 0; i < data_.n; ++i) {
        double mu_i = mu_[i];
        // Clamp mu to valid range for the family
        if (family_.family == GLMFamily::BINOMIAL) {
            mu_i = std::max(1e-10, std::min(1.0 - 1e-10, mu_i));
        } else {
            mu_i = std::max(1e-10, mu_i);
        }
        dev_sum += family_.dev_resid(data_.y[i], mu_i, prior_weights_[i]);
    }

    deviance_ = dev_sum + u_.squaredNorm() + ldL2_;
    return deviance_;
}

void PIRLSDeviance::pirls(const SpMatd& Lamt) {
    const int n = data_.n;
    const int p = data_.p;
    const int q = data_.q;

    // Compute eta from current beta and u
    VectorXd b = Lamt.transpose() * u_;
    eta_ = data_.X * beta_ + data_.Zt.transpose() * b;

    // If this is the first call (eta near zero), initialise from the mean
    if (eta_.squaredNorm() < 1e-20) {
        double y_mean = data_.y.mean();
        if (family_.family == GLMFamily::BINOMIAL) {
            y_mean = std::max(0.01, std::min(0.99, y_mean));
        } else {
            y_mean = std::max(0.01, y_mean);
        }
        double eta0 = family_.link_fun(y_mean);
        eta_.setConstant(eta0);
    }

    for (pirls_iter_ = 0; pirls_iter_ < pirls_maxiter_; ++pirls_iter_) {
        // Compute mu = linkinv(eta)
        for (int i = 0; i < n; ++i) {
            mu_[i] = family_.linkinv(eta_[i]);
        }

        // Compute working weights: W_ii = (d mu/d eta)^2 / Var(mu)
        // and working response: z_i = eta_i + (y_i - mu_i) / (d mu/d eta)
        VectorXd w(n), z(n);
        for (int i = 0; i < n; ++i) {
            double dmu = family_.mu_eta(eta_[i]);
            double var_mu = family_.variance(mu_[i]);

            // Clamp to avoid division by zero
            dmu = std::max(std::abs(dmu), 1e-10);
            var_mu = std::max(var_mu, 1e-10);

            w[i] = prior_weights_[i] * (dmu * dmu) / var_mu;
            z[i] = eta_[i] + (data_.y[i] - mu_[i]) / dmu;
        }

        // Solve the weighted penalised least squares problem:
        //   minimise ||sqrt(W)(z - X*beta - Z*Lamt'*u)||^2 + ||u||^2
        //
        // This is equivalent to the LMM profiled deviance with
        // y replaced by z and a diagonal weight matrix W.

        // Compute weighted cross-products
        // W^{1/2} * Zt^T = Zt * diag(sqrt(w)) conceptually
        // But we need Zt_w = diag(sqrt(w)) * Z^T... no.
        // ZtWZ = Zt * diag(w) * Z = sum over i: w_i * z_i * z_i^T
        // More efficient: scale Zt columns by sqrt(w)

        VectorXd sqrtw = w.array().sqrt().matrix();

        // Scale Z: Zt_w[, i] = Zt[, i] * sqrtw[i]
        // Form A = Lamt * Zt_w * Zt_w^T * Lamt^T + I
        //        = Lamt * Zt * diag(w) * Z * Lamt^T + I

        // Build A = Lamt * ZtWZ * Lamt^T + I
        // ZtWZ = Zt * diag(w) * Z. Compute via:
        // For each column j of Zt (= observation j):
        //   ZtWZ += w[j] * Zt.col(j) * Zt.col(j)^T
        // This is a weighted outer product sum.
        //
        // Efficient implementation: use Zt * diag(sqrt(w)) then multiply
        // by its own transpose. We build the product via triplets.

        // Build ZtWZ by scaling ZtZ entries.
        // ZtWZ(i,j) = sum_k w_k * Z(k,i) * Z(k,j)
        // For random intercept: Z is an indicator matrix, so
        // ZtZ(i,j) = #{obs in both groups i and j}.
        // ZtWZ(i,j) = sum of w_k for obs in both groups.
        //
        // General approach: build ZtWZ from scratch via dense accumulation
        // for the small q typical in GLMM.
        MatrixXd ZtWZ_dense = MatrixXd::Zero(q, q);
        // Zt is q x n. Column j = observation j.
        // ZtWZ = sum_j w_j * Zt.col(j) * Zt.col(j)^T
        for (int j = 0; j < n; ++j) {
            VectorXd zt_j(q);
            zt_j.setZero();
            for (SpMatd::InnerIterator it(data_.Zt, j); it; ++it) {
                zt_j[it.row()] = it.value();
            }
            ZtWZ_dense.noalias() += w[j] * zt_j * zt_j.transpose();
        }
        SpMatd ZtWZ = ZtWZ_dense.sparseView();

        SpMatd A = Lamt * ZtWZ * Lamt.transpose();
        for (int j = 0; j < q; ++j) {
            A.coeffRef(j, j) += 1.0;
        }

        // Eigen SimplicialLLT
        Eigen::SimplicialLLT<SpMatd, Eigen::Lower, Eigen::AMDOrdering<int>> llt;
        llt.analyzePattern(A);
        llt.factorize(A);
        if (llt.info() != Eigen::Success) {
            deviance_ = std::numeric_limits<double>::infinity();
            return;
        }

        ldL2_ = 0.0;
        const SpMatd& L_factor = llt.matrixL();
        for (int j = 0; j < q; ++j) {
            ldL2_ += std::log(L_factor.coeff(j, j));
        }
        ldL2_ *= 2.0;

        // Weighted cross-products
        VectorXd wz = w.array() * z.array();
        MatrixXd WX(n, p);
        for (int j = 0; j < p; ++j) {
            WX.col(j) = w.array() * data_.X.col(j).array();
        }

        // cu = A^{-1} Lamt * Zt * (w .* z)
        VectorXd LamtZtwz = Lamt * (data_.Zt * wz);
        VectorXd cu = llt.solve(LamtZtwz);

        // Solve for RZX columns
        MatrixXd LamtZtWX = Lamt * (data_.Zt * WX);
        MatrixXd RZX(q, p);
        for (int j = 0; j < p; ++j) {
            RZX.col(j) = llt.solve(VectorXd(LamtZtWX.col(j)));
        }

        // RXtRX = XtWX - LamtZtWX^T * A^{-1} * LamtZtWX
        MatrixXd XtWX = data_.X.transpose() * WX;
        MatrixXd RXtRX = XtWX - LamtZtWX.transpose() * RZX;

        RX_chol_.compute(RXtRX);
        if (RX_chol_.info() != Eigen::Success) {
            deviance_ = std::numeric_limits<double>::infinity();
            return;
        }

        // beta_new
        VectorXd XtWz = data_.X.transpose() * wz;
        VectorXd rhs_beta = XtWz - LamtZtWX.transpose() * cu;
        VectorXd beta_new = RX_chol_.solve(rhs_beta);

        // u_new
        VectorXd rhs_u = LamtZtwz - LamtZtWX * beta_new;
        VectorXd u_new = llt.solve(rhs_u);

        // Update eta
        VectorXd b_new = Lamt.transpose() * u_new;
        VectorXd eta_new = data_.X * beta_new + data_.Zt.transpose() * b_new;

        // Check convergence
        double delta = (eta_new - eta_).squaredNorm() /
                       std::max(1.0, eta_new.squaredNorm());

        beta_ = beta_new;
        u_ = u_new;
        eta_ = eta_new;

        if (pirls_iter_ >= 2 && delta < pirls_tol_ * pirls_tol_) {
            break;
        }
    }

    // Final mu
    for (int i = 0; i < n; ++i) {
        mu_[i] = family_.linkinv(eta_[i]);
    }
}

MatrixXd PIRLSDeviance::vcov_beta() const {
    MatrixXd I_p = MatrixXd::Identity(data_.p, data_.p);
    return RX_chol_.solve(I_p);
}

} // namespace fastmlm
