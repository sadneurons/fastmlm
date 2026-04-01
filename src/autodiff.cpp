#include "autodiff.h"
#include <cmath>

namespace fastmlm {

DevianceGradient::DevianceGradient(ProfiledDeviance& devfun, double eps)
    : devfun_(devfun), eps_(eps), eval_count_(0) {}

double DevianceGradient::compute(const VectorXd& theta, VectorXd& gradient) {
    int nth = theta.size();
    gradient.resize(nth);

    // Evaluate at the current point
    double f0 = devfun_(theta);
    eval_count_++;

    // Central differences: (f(x+h) - f(x-h)) / (2h)
    // More accurate than forward differences (O(h^2) vs O(h) error)
    VectorXd theta_p = theta;
    VectorXd theta_m = theta;

    for (int k = 0; k < nth; ++k) {
        // Adaptive step size based on parameter magnitude
        double h = eps_ * std::max(1.0, std::abs(theta[k]));

        theta_p[k] = theta[k] + h;
        theta_m[k] = theta[k] - h;

        double fp = devfun_(theta_p);
        double fm = devfun_(theta_m);
        eval_count_ += 2;

        gradient[k] = (fp - fm) / (2.0 * h);

        // Restore
        theta_p[k] = theta[k];
        theta_m[k] = theta[k];
    }

    // Re-evaluate at theta to restore the devfun's internal state
    // (beta, u, sigma etc. should correspond to the actual theta, not theta±h)
    devfun_(theta);
    eval_count_++;

    return f0;
}

} // namespace fastmlm
