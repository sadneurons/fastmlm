#ifndef FASTMLM_AUTODIFF_H
#define FASTMLM_AUTODIFF_H

#include "profiled_deviance.h"

namespace fastmlm {

// Gradient computation for the profiled deviance.
//
// Phase 2 uses central finite differences computed entirely in C++.
// This avoids any R↔C++ overhead and gives us exact-enough gradients
// for L-BFGS-B convergence. The cost is 2*nth deviance evaluations
// per gradient call, which is cheap since nth is typically 1-10.
//
// Future: replace with analytical gradients via matrix calculus
// (implicit differentiation through the Cholesky decomposition).
class DevianceGradient {
public:
    explicit DevianceGradient(ProfiledDeviance& devfun, double eps = 1e-7);

    // Compute gradient at theta via central differences.
    // Returns the deviance value f(theta) as a side effect.
    double compute(const VectorXd& theta, VectorXd& gradient);

    // Combined function: evaluates deviance and gradient.
    // This is the interface expected by LBFGSB::GradFun.
    double operator()(const VectorXd& theta, VectorXd& gradient) {
        return compute(theta, gradient);
    }

    // Number of deviance evaluations so far
    int eval_count() const { return eval_count_; }

private:
    ProfiledDeviance& devfun_;
    double eps_;
    int eval_count_;
};

} // namespace fastmlm

#endif // FASTMLM_AUTODIFF_H
