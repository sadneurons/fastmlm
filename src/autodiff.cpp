#include "autodiff.h"
#include <cmath>

namespace fastmlm {

DevianceGradient::DevianceGradient(ProfiledDeviance& devfun, double eps)
    : devfun_(devfun), eps_(eps), eval_count_(0) {}

double DevianceGradient::compute(const VectorXd& theta, VectorXd& gradient) {
    int nth = theta.size();
    gradient.resize(nth);

    // Use forward differences for speed: (f(x+h) - f(x)) / h
    // Only nth+1 evaluations instead of 2*nth+2 for central differences.
    // For the low-dimensional theta typical in MLMs, the O(h) accuracy
    // of forward differences is sufficient for L-BFGS-B convergence.
    double f0 = devfun_(theta);
    eval_count_++;

    VectorXd theta_p = theta;

    for (int k = 0; k < nth; ++k) {
        double h = eps_ * std::max(1.0, std::abs(theta[k]));

        theta_p[k] = theta[k] + h;
        double fp = devfun_(theta_p);
        eval_count_++;

        gradient[k] = (fp - f0) / h;

        theta_p[k] = theta[k];  // restore
    }

    // No need to re-evaluate at theta — the optimizer will do this
    // at the next step via the objective function. The devfun's internal
    // state (beta, u) is stale but won't be read until after the next
    // devfun(theta_new) call in the line search.

    return f0;
}

} // namespace fastmlm
