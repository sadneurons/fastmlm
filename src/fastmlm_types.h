#ifndef FASTMLM_TYPES_H
#define FASTMLM_TYPES_H

#include <RcppEigen.h>

namespace fastmlm {

// Dense types
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;
using VectorXi = Eigen::VectorXi;

// Sparse types
using SpMatd = Eigen::SparseMatrix<double>;
using SpMatdRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;

// Index type
using Index = Eigen::Index;

// Optimization result
struct OptResult {
    VectorXd theta;
    double deviance;
    int iterations;
    int fn_evaluations;
    int convergence;  // 0 = converged
    std::string message;
};

// Model fit result (returned to R)
struct FitResult {
    VectorXd theta;
    VectorXd beta;
    VectorXd u;
    double sigma;
    double deviance;
    MatrixXd vcov_beta;
    OptResult optinfo;
    bool REML;
};

} // namespace fastmlm

#endif // FASTMLM_TYPES_H
