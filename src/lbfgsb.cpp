#include "lbfgsb.h"
#include <cmath>
#include <algorithm>

namespace fastmlm {

void LBFGSB::project(VectorXd& x, const VectorXd& lower) {
    for (int i = 0; i < x.size(); ++i) {
        if (std::isfinite(lower[i]) && x[i] < lower[i]) {
            x[i] = lower[i];
        }
    }
}

double LBFGSB::projected_gradient_norm(
    const VectorXd& x, const VectorXd& g, const VectorXd& lower)
{
    double norm = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        double gi = g[i];
        // If at lower bound and gradient points into the constraint, zero it
        if (std::isfinite(lower[i]) && x[i] <= lower[i] && gi > 0.0) {
            gi = 0.0;
        }
        norm += gi * gi;
    }
    return std::sqrt(norm);
}

VectorXd LBFGSB::lbfgs_direction(
    const VectorXd& g,
    const std::deque<VectorXd>& s_history,
    const std::deque<VectorXd>& y_history,
    double gamma)
{
    int m = static_cast<int>(s_history.size());
    if (m == 0) {
        return -gamma * g;
    }

    VectorXd q = g;
    std::vector<double> alpha(m);
    std::vector<double> rho(m);

    // Forward pass
    for (int i = m - 1; i >= 0; --i) {
        rho[i] = 1.0 / s_history[i].dot(y_history[i]);
        alpha[i] = rho[i] * s_history[i].dot(q);
        q -= alpha[i] * y_history[i];
    }

    // Initial Hessian approximation
    VectorXd r = gamma * q;

    // Backward pass
    for (int i = 0; i < m; ++i) {
        double beta = rho[i] * y_history[i].dot(r);
        r += s_history[i] * (alpha[i] - beta);
    }

    return -r;
}

double LBFGSB::line_search(
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
    int& grad_evals)
{
    double dg0 = g0.dot(d);
    if (dg0 >= 0.0) {
        // Not a descent direction; use steepest descent
        return 0.0;
    }

    double alpha = 1.0;
    double alpha_lo = 0.0;
    double alpha_hi = opts.step_max;

    for (int iter = 0; iter < opts.max_linesearch; ++iter) {
        x_new = x + alpha * d;
        project(x_new, lower);

        f_new = grad_fn(x_new, g_new);
        fn_evals++;
        grad_evals++;

        // Armijo (sufficient decrease) condition
        double actual_step_dot = (x_new - x).dot(g0);
        if (f_new <= f0 + opts.armijo_c1 * actual_step_dot) {
            // Wolfe curvature condition
            double dg_new = g_new.dot(d);
            if (dg_new >= opts.wolfe_c2 * dg0) {
                return alpha;
            }
            // Curvature not satisfied — increase step
            alpha_lo = alpha;
            alpha = (alpha_hi >= opts.step_max)
                    ? 2.0 * alpha
                    : 0.5 * (alpha_lo + alpha_hi);
        } else {
            // Armijo not satisfied — decrease step
            alpha_hi = alpha;
            alpha = 0.5 * (alpha_lo + alpha_hi);
        }

        if (alpha < opts.step_min) {
            return alpha;
        }
    }

    return alpha;
}

LBFGSB::Result LBFGSB::minimize(
    ObjFun fn,
    GradFun grad_fn,
    const VectorXd& x0,
    const VectorXd& lower,
    const Options& opts)
{
    int n = x0.size();
    Result result;
    result.iterations = 0;
    result.fn_evaluations = 0;
    result.grad_evaluations = 0;

    VectorXd x = x0;
    project(x, lower);

    VectorXd g(n);
    double f = grad_fn(x, g);
    result.fn_evaluations++;
    result.grad_evaluations++;

    std::deque<VectorXd> s_history;
    std::deque<VectorXd> y_history;

    double pgnorm = projected_gradient_norm(x, g, lower);

    if (pgnorm < opts.gtol) {
        result.x = x;
        result.f = f;
        result.gradient = g;
        result.convergence = 0;
        result.message = "converged (initial point is optimal)";
        return result;
    }

    for (int iter = 0; iter < opts.max_iterations; ++iter) {
        result.iterations = iter + 1;

        // Compute initial Hessian scaling
        double gamma = 1.0;
        if (!s_history.empty()) {
            const VectorXd& s_last = s_history.back();
            const VectorXd& y_last = y_history.back();
            double sy = s_last.dot(y_last);
            double yy = y_last.dot(y_last);
            if (yy > 0.0) {
                gamma = sy / yy;
            }
        }

        // Compute search direction via L-BFGS two-loop recursion
        VectorXd d = lbfgs_direction(g, s_history, y_history, gamma);

        // Projected line search
        VectorXd x_new(n), g_new(n);
        double f_new;
        int ls_fn = 0, ls_gr = 0;

        double alpha = line_search(
            fn, grad_fn, x, d, f, g, lower, opts,
            x_new, g_new, f_new, ls_fn, ls_gr);

        result.fn_evaluations += ls_fn;
        result.grad_evaluations += ls_gr;

        if (alpha <= 0.0 || ls_fn == 0) {
            // Line search failed — try steepest descent with projection
            d = -g;
            for (int i = 0; i < n; ++i) {
                if (std::isfinite(lower[i]) && x[i] <= lower[i] && d[i] < 0.0) {
                    d[i] = 0.0;
                }
            }
            alpha = line_search(
                fn, grad_fn, x, d, f, g, lower, opts,
                x_new, g_new, f_new, ls_fn, ls_gr);
            result.fn_evaluations += ls_fn;
            result.grad_evaluations += ls_gr;

            if (alpha <= 0.0 || ls_fn == 0) {
                result.x = x;
                result.f = f;
                result.gradient = g;
                result.convergence = 2;
                result.message = "line search failed";
                return result;
            }
            // Reset L-BFGS history after steepest descent step
            s_history.clear();
            y_history.clear();
        }

        // Update L-BFGS history
        VectorXd s = x_new - x;
        VectorXd y = g_new - g;
        double sy = s.dot(y);

        if (sy > 1e-20) {
            s_history.push_back(std::move(s));
            y_history.push_back(std::move(y));

            if (static_cast<int>(s_history.size()) > opts.history_size) {
                s_history.pop_front();
                y_history.pop_front();
            }
        }

        // Check convergence (before updating f)
        double fdiff = std::abs(f_new - f);

        x = x_new;
        f = f_new;
        g = g_new;

        pgnorm = projected_gradient_norm(x, g, lower);

        if (pgnorm < opts.gtol) {
            result.x = x;
            result.f = f;
            result.gradient = g;
            result.convergence = 0;
            result.message = "converged (projected gradient norm < gtol)";
            return result;
        }

        if (result.iterations > 1 && fdiff < opts.ftol) {
            result.x = x;
            result.f = f;
            result.gradient = g;
            result.convergence = 0;
            result.message = "converged (function change < ftol)";
            return result;
        }
    }

    result.x = x;
    result.f = f;
    result.gradient = g;
    result.convergence = 1;
    result.message = "maximum iterations reached";
    return result;
}

} // namespace fastmlm
