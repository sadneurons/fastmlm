#ifndef FASTMLM_VCOV_HESSIAN_H
#define FASTMLM_VCOV_HESSIAN_H

#include "profiled_deviance.h"

namespace fastmlm {

// Compute the Hessian of the profiled deviance at theta via
// central finite differences. Used to obtain the variance-covariance
// matrix of theta at convergence.
//
// The Hessian is computed as:
//   H[i,j] = (f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)) / (4*hi*hj)
//
// Cost: O(nth^2) deviance evaluations (for nth=3, that's ~9 evals).
// Compare to lme4's approach: 2*nth^2 - nth + 1 evaluations.
class DevianceHessian {
public:
    explicit DevianceHessian(ProfiledDeviance& devfun, double eps = 1e-4);

    // Compute Hessian at theta
    MatrixXd compute(const VectorXd& theta);

    // Compute variance-covariance of theta = 2 * H^{-1}
    // (factor of 2 because deviance = -2 * logLik)
    MatrixXd vcov_theta(const VectorXd& theta);

    int eval_count() const { return eval_count_; }

private:
    ProfiledDeviance& devfun_;
    double eps_;
    int eval_count_;
};

} // namespace fastmlm

#endif // FASTMLM_VCOV_HESSIAN_H
