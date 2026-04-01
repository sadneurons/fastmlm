#include "theta_optimizer.h"

namespace fastmlm {

LBFGSB::Result ThetaOptimizer::optimize(
    ProfiledDeviance& devfun,
    const VectorXd& theta_start,
    const VectorXd& lower,
    const Control& control)
{
    // Objective function: just evaluate deviance
    LBFGSB::ObjFun obj_fn = [&devfun](const VectorXd& theta) -> double {
        return devfun(theta);
    };

    // Gradient function: evaluate deviance + numerical gradient
    DevianceGradient grad_computer(devfun, control.grad_eps);

    LBFGSB::GradFun grad_fn = [&grad_computer](
        const VectorXd& theta, VectorXd& g) -> double {
        return grad_computer.compute(theta, g);
    };

    // Set up L-BFGS-B options
    LBFGSB::Options opts;
    opts.max_iterations = control.maxiter;
    opts.ftol = control.ftol;
    opts.gtol = control.gtol;
    opts.verbose = control.verbose;

    // Run the optimizer
    LBFGSB::Result result = LBFGSB::minimize(
        obj_fn, grad_fn, theta_start, lower, opts);

    // Final evaluation at the optimum to populate devfun's cached results
    devfun(result.x);

    return result;
}

} // namespace fastmlm
