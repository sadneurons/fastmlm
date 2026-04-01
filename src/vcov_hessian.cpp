#include "vcov_hessian.h"
#include <cmath>
#include <stdexcept>

namespace fastmlm {

DevianceHessian::DevianceHessian(ProfiledDeviance& devfun, double eps)
    : devfun_(devfun), eps_(eps), eval_count_(0) {}

MatrixXd DevianceHessian::compute(const VectorXd& theta) {
    int nth = theta.size();
    MatrixXd H(nth, nth);

    // Compute step sizes (adaptive)
    VectorXd h(nth);
    for (int k = 0; k < nth; ++k) {
        h[k] = eps_ * std::max(1.0, std::abs(theta[k]));
    }

    // Diagonal elements: (f(x+h) - 2*f(x) + f(x-h)) / h^2
    double f0 = devfun_(theta);
    eval_count_++;

    VectorXd theta_mod = theta;

    for (int i = 0; i < nth; ++i) {
        theta_mod[i] = theta[i] + h[i];
        double fpi = devfun_(theta_mod);
        eval_count_++;

        theta_mod[i] = theta[i] - h[i];
        double fmi = devfun_(theta_mod);
        eval_count_++;

        H(i, i) = (fpi - 2.0 * f0 + fmi) / (h[i] * h[i]);

        theta_mod[i] = theta[i];  // restore
    }

    // Off-diagonal elements: (f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)) / (4*hi*hj)
    for (int i = 0; i < nth; ++i) {
        for (int j = i + 1; j < nth; ++j) {
            theta_mod = theta;

            theta_mod[i] = theta[i] + h[i];
            theta_mod[j] = theta[j] + h[j];
            double fpp = devfun_(theta_mod);

            theta_mod[j] = theta[j] - h[j];
            double fpm = devfun_(theta_mod);

            theta_mod[i] = theta[i] - h[i];
            double fmm = devfun_(theta_mod);

            theta_mod[j] = theta[j] + h[j];
            double fmp = devfun_(theta_mod);

            eval_count_ += 4;

            H(i, j) = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j]);
            H(j, i) = H(i, j);
        }
    }

    // Restore devfun state
    devfun_(theta);
    eval_count_++;

    return H;
}

MatrixXd DevianceHessian::vcov_theta(const VectorXd& theta) {
    MatrixXd H = compute(theta);

    // vcov(theta) = 2 * H^{-1}
    // deviance = -2 * logLik, so Hessian of deviance = 2 * Fisher info
    Eigen::LLT<MatrixXd> llt(H);
    if (llt.info() != Eigen::Success) {
        // Hessian not positive definite — use pseudoinverse
        Eigen::JacobiSVD<MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
        MatrixXd Hinv = svd.solve(MatrixXd::Identity(H.rows(), H.cols()));
        return 2.0 * Hinv;
    }

    MatrixXd I = MatrixXd::Identity(H.rows(), H.cols());
    return 2.0 * llt.solve(I);
}

} // namespace fastmlm
