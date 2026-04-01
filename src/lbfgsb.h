#ifndef FASTMLM_LBFGSB_H
#define FASTMLM_LBFGSB_H

#include "fastmlm_types.h"
#include <functional>
#include <deque>

namespace fastmlm {

// L-BFGS-B: Limited-memory BFGS with box constraints.
//
// Implements the projected-gradient L-BFGS method for minimising
// f(x) subject to lower <= x <= upper (component-wise).
//
// Key properties:
//   - O(m*n) memory and per-iteration cost (m = history size, n = dimension)
//   - Superlinear convergence with exact gradients
//   - Handles box constraints via gradient projection
struct LBFGSBOptions {
    int max_iterations = 300;
    int history_size = 6;        // number of (s, y) pairs to store
    double ftol = 1e-10;         // function value tolerance
    double gtol = 1e-8;          // gradient norm tolerance (projected)
    double step_min = 1e-20;     // minimum step size in line search
    double step_max = 1e20;      // maximum step size
    int max_linesearch = 20;     // max line search iterations
    double armijo_c1 = 1e-4;     // sufficient decrease parameter
    double wolfe_c2 = 0.9;       // curvature condition parameter
    int verbose = 0;
};

class LBFGSB {
public:
    using Options = LBFGSBOptions;

    struct Result {
        VectorXd x;
        double f;
        VectorXd gradient;
        int iterations;
        int fn_evaluations;
        int grad_evaluations;
        int convergence;    // 0 = converged, 1 = max iter, 2 = line search fail
        std::string message;
    };

    // Objective function type: takes x, returns f(x)
    using ObjFun = std::function<double(const VectorXd&)>;

    // Gradient function type: takes x, writes gradient into g, returns f(x)
    using GradFun = std::function<double(const VectorXd&, VectorXd&)>;

    // Minimise f(x) subject to lower <= x (upper = +inf)
    static Result minimize(
        ObjFun fn,
        GradFun grad_fn,
        const VectorXd& x0,
        const VectorXd& lower,
        const Options& opts = Options()
    );

private:
    // Project x onto the feasible set [lower, +inf)
    static void project(VectorXd& x, const VectorXd& lower);

    // Compute projected gradient norm (for convergence check)
    static double projected_gradient_norm(
        const VectorXd& x, const VectorXd& g, const VectorXd& lower);

    // L-BFGS two-loop recursion to compute search direction
    static VectorXd lbfgs_direction(
        const VectorXd& g,
        const std::deque<VectorXd>& s_history,
        const std::deque<VectorXd>& y_history,
        double gamma  // initial Hessian scaling
    );

    // Projected line search with Armijo condition
    static double line_search(
        ObjFun& fn,
        GradFun& grad_fn,
        const VectorXd& x,
        const VectorXd& d,
        double f0,
        const VectorXd& g0,
        const VectorXd& lower,
        const Options& opts,
        VectorXd& x_new,
        VectorXd& g_new,
        double& f_new,
        int& fn_evals,
        int& grad_evals
    );
};

} // namespace fastmlm

#endif // FASTMLM_LBFGSB_H
