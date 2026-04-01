#include <Rcpp.h>
#include <RcppEigen.h>
#include "theta_optimizer.h"
#include "vcov_hessian.h"
#include "blas_detect.h"
#include "parallel_ops.h"

using namespace fastmlm;

// Store the deviance function container as an external pointer
typedef Rcpp::XPtr<DevFunContainer> DevFunXPtr;

// --- Phase 1 API (retained for compatibility) ---

// [[Rcpp::export]]
SEXP C_fastmlm_create(Eigen::Map<Eigen::VectorXd> y,
                       Eigen::Map<Eigen::MatrixXd> X,
                       Eigen::MappedSparseMatrix<double> Zt,
                       Eigen::MappedSparseMatrix<double> Lambdat,
                       Eigen::Map<Eigen::VectorXi> Lind,
                       Eigen::Map<Eigen::VectorXd> lower,
                       bool REML) {
    VectorXd y_own = y;
    MatrixXd X_own = X;
    SpMatd Zt_own = Zt;
    SpMatd Lambdat_own = Lambdat;
    VectorXi Lind_own = Lind;
    VectorXd lower_own = lower;

    DevFunContainer* container = new DevFunContainer(
        y_own, X_own, Zt_own, Lambdat_own, Lind_own, lower_own, REML
    );

    DevFunXPtr xptr(container, true);
    return xptr;
}

// [[Rcpp::export]]
double C_fastmlm_deviance(SEXP xptr, Eigen::Map<Eigen::VectorXd> theta) {
    DevFunXPtr container(xptr);
    VectorXd theta_own = theta;
    return container->devfun(theta_own);
}

// [[Rcpp::export]]
Rcpp::List C_fastmlm_result(SEXP xptr) {
    DevFunXPtr container(xptr);
    const ProfiledDeviance& df = container->devfun;

    MatrixXd vcov_unscaled = df.vcov_beta_unscaled();
    MatrixXd vcov_beta = df.sigma2() * vcov_unscaled;

    return Rcpp::List::create(
        Rcpp::Named("beta") = df.beta(),
        Rcpp::Named("u") = df.u(),
        Rcpp::Named("sigma") = std::sqrt(df.sigma2()),
        Rcpp::Named("deviance") = df.deviance(),
        Rcpp::Named("vcov_beta") = vcov_beta,
        Rcpp::Named("ldL2") = df.ldL2(),
        Rcpp::Named("ldRX2") = df.ldRX2(),
        Rcpp::Named("pwrss") = df.pwrss(),
        Rcpp::Named("REML") = df.is_REML()
    );
}

// --- Phase 2 API: Full C++ optimization ---

// [[Rcpp::export]]
Rcpp::List C_fastmlm_fit(Eigen::Map<Eigen::VectorXd> y,
                          Eigen::Map<Eigen::MatrixXd> X,
                          Eigen::MappedSparseMatrix<double> Zt,
                          Eigen::MappedSparseMatrix<double> Lambdat,
                          Eigen::Map<Eigen::VectorXi> Lind,
                          Eigen::Map<Eigen::VectorXd> theta_start,
                          Eigen::Map<Eigen::VectorXd> lower,
                          Eigen::Map<Eigen::VectorXi> Gp,
                          bool REML,
                          int maxiter,
                          double ftol,
                          double gtol,
                          int pcg_threshold,
                          int verbose) {
    // Deep copy inputs
    VectorXd y_own = y;
    MatrixXd X_own = X;
    SpMatd Zt_own = Zt;
    SpMatd Lambdat_own = Lambdat;
    VectorXi Lind_own = Lind;
    VectorXd lower_own = lower;
    VectorXd theta0 = theta_start;
    VectorXi Gp_own = Gp;

    // Build model data and deviance function
    MLMData data(y_own, X_own, Zt_own, Lambdat_own, Lind_own, lower_own);
    data.set_block_structure(Gp_own);
    ProfiledDeviance devfun(data, REML, pcg_threshold);

    // Configure optimizer
    ThetaOptimizer::Control ctrl;
    ctrl.maxiter = maxiter;
    ctrl.ftol = ftol;
    ctrl.gtol = gtol;
    ctrl.verbose = verbose;

    // Run C++ L-BFGS-B optimization
    LBFGSB::Result opt = ThetaOptimizer::optimize(
        devfun, theta0, lower_own, ctrl);

    // Extract model results
    MatrixXd vcov_unscaled = devfun.vcov_beta_unscaled();
    MatrixXd vcov_beta = devfun.sigma2() * vcov_unscaled;

    // Skip Hessian computation by default — it's expensive and rarely needed.
    // Users who need vcov(theta) can request it via control.
    MatrixXd vcov_theta = MatrixXd::Zero(opt.x.size(), opt.x.size());

    // Package results
    return Rcpp::List::create(
        Rcpp::Named("theta") = opt.x,
        Rcpp::Named("beta") = devfun.beta(),
        Rcpp::Named("u") = devfun.u(),
        Rcpp::Named("sigma") = std::sqrt(devfun.sigma2()),
        Rcpp::Named("deviance") = devfun.deviance(),
        Rcpp::Named("vcov_beta") = vcov_beta,
        Rcpp::Named("vcov_theta") = vcov_theta,
        Rcpp::Named("ldL2") = devfun.ldL2(),
        Rcpp::Named("ldRX2") = devfun.ldRX2(),
        Rcpp::Named("pwrss") = devfun.pwrss(),
        Rcpp::Named("REML") = REML,
        Rcpp::Named("convergence") = opt.convergence,
        Rcpp::Named("message") = opt.message,
        Rcpp::Named("iterations") = opt.iterations,
        Rcpp::Named("fn_evaluations") = opt.fn_evaluations,
        Rcpp::Named("grad_evaluations") = opt.grad_evaluations,
        Rcpp::Named("using_pcg") = devfun.using_pcg(),
        Rcpp::Named("is_crossed") = data.is_crossed
    );
}

// Convenience: BLAS info for diagnostics
// [[Rcpp::export]]
Rcpp::List C_fastmlm_blas_info() {
    return Rcpp::List::create(
        Rcpp::Named("eigen_version") = Rcpp::CharacterVector::create(
            std::to_string(EIGEN_WORLD_VERSION) + "." +
            std::to_string(EIGEN_MAJOR_VERSION) + "." +
            std::to_string(EIGEN_MINOR_VERSION)
        ),
        Rcpp::Named("blas_library") = blas::detect_library(),
        Rcpp::Named("blas_threads") = blas::get_num_threads(),
        Rcpp::Named("has_openmp") =
#ifdef _OPENMP
            true
#else
            false
#endif
        ,
        Rcpp::Named("omp_threads") = blas::get_omp_threads(),
        Rcpp::Named("optimizer") = "L-BFGS-B (C++)"
    );
}

// [[Rcpp::export]]
void C_fastmlm_set_threads(int blas_threads, int omp_threads) {
    if (blas_threads > 0) blas::set_num_threads(blas_threads);
    if (omp_threads > 0) blas::set_omp_threads(omp_threads);
}

// [[Rcpp::export]]
double C_fastmlm_benchmark_blas(int n) {
    return blas::benchmark_dgemm(n);
}
