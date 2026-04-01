#ifndef FASTMLM_THETA_OPTIMIZER_H
#define FASTMLM_THETA_OPTIMIZER_H

#include "profiled_deviance.h"
#include "lbfgsb.h"
#include "autodiff.h"

namespace fastmlm {

// Container holding the model data and deviance function.
// Used as an external pointer from R for the Phase 1 API.
struct DevFunContainer {
    MLMData data;
    ProfiledDeviance devfun;

    DevFunContainer(const VectorXd& y, const MatrixXd& X,
                    const SpMatd& Zt, const SpMatd& Lambdat,
                    const VectorXi& Lind, const VectorXd& lower,
                    bool REML)
        : data(y, X, Zt, Lambdat, Lind, lower),
          devfun(data, REML) {}
};

struct ThetaOptimizerControl {
    int maxiter = 300;
    double ftol = 1e-10;
    double gtol = 1e-8;
    double grad_eps = 1e-7;     // step size for numerical gradient
    double hessian_eps = 1e-4;  // step size for Hessian
    int verbose = 0;
    bool use_gradient = true;   // true = L-BFGS-B, false = Nelder-Mead style
};

// Phase 2: Full C++ optimisation of theta.
// Runs L-BFGS-B with numerical gradients entirely in C++,
// eliminating all R↔C++ callback overhead.
class ThetaOptimizer {
public:
    using Control = ThetaOptimizerControl;

    // Optimise theta for the given deviance function.
    // Returns the LBFGSB::Result with optimal theta and convergence info.
    static LBFGSB::Result optimize(
        ProfiledDeviance& devfun,
        const VectorXd& theta_start,
        const VectorXd& lower,
        const Control& control = Control());
};

} // namespace fastmlm

#endif // FASTMLM_THETA_OPTIMIZER_H
