// CHOLMOD wrapper using Matrix package's CHOLMOD.
//
// The actual CHOLMOD stub functions (M_cholmod_*) are compiled in
// cholmod_stubs.c (pure C). Here we just declare them as extern "C"
// and call them.

#include <Rcpp.h>

// Include CHOLMOD types (but NOT stubs.c — that's in cholmod_stubs.c)
// We need the cholmod_common, cholmod_sparse, etc. typedefs.
// Use R_NO_REMAP to prevent macro conflicts with Rcpp.
#define R_NO_REMAP
#include <Matrix/cholmod.h>
#undef R_NO_REMAP

// Undo any conflicting macros from cholmod.h
#undef length
#undef FREE

#include "cholmod_wrapper.h"
#include <cmath>
#include <stdexcept>

// Declare the Matrix stub functions (defined in cholmod_stubs.c)
extern "C" {
    cholmod_factor* M_cholmod_analyze(cholmod_sparse*, cholmod_common*);
    int M_cholmod_factorize(cholmod_sparse*, cholmod_factor*, cholmod_common*);
    cholmod_dense* M_cholmod_solve(int, cholmod_factor*, cholmod_dense*, cholmod_common*);
    cholmod_sparse* M_cholmod_allocate_sparse(size_t, size_t, size_t, int, int, int, int, cholmod_common*);
    int M_cholmod_free_sparse(cholmod_sparse**, cholmod_common*);
    int M_cholmod_free_dense(cholmod_dense**, cholmod_common*);
    int M_cholmod_free_factor(cholmod_factor**, cholmod_common*);
    int M_cholmod_start(cholmod_common*);
    int M_cholmod_finish(cholmod_common*);
    double M_cholmod_factor_ldetA(cholmod_factor*);
}

namespace fastmlm {

CholmodWrapper::CholmodWrapper() : factor_(nullptr) {
    M_cholmod_start(&common());
    common().supernodal = CHOLMOD_SUPERNODAL;
}

CholmodWrapper::~CholmodWrapper() {
    if (factor_) {
        M_cholmod_free_factor(&factor_, &common());
    }
    M_cholmod_finish(&common());
}

cholmod_sparse* CholmodWrapper::eigen_to_cholmod(const SpMatd& A) const {
    int n = A.rows();
    int nnz = A.nonZeros();

    cholmod_sparse* C = M_cholmod_allocate_sparse(
        n, n, nnz,
        1,  // sorted
        1,  // packed
        0,  // stype=0 (set below)
        CHOLMOD_REAL,
        const_cast<cholmod_common*>(&common())
    );

    if (!C) return nullptr;

    int* Cp = static_cast<int*>(C->p);
    int* Ci = static_cast<int*>(C->i);
    double* Cx = static_cast<double*>(C->x);

    const int* Ap = A.outerIndexPtr();
    const int* Ai = A.innerIndexPtr();
    const double* Ax = A.valuePtr();

    for (int j = 0; j <= n; ++j) Cp[j] = Ap[j];
    for (int k = 0; k < nnz; ++k) {
        Ci[k] = Ai[k];
        Cx[k] = Ax[k];
    }

    // Mark as symmetric upper for analyze/factorize
    C->stype = 1;

    return C;
}

void CholmodWrapper::free_cholmod_sparse(cholmod_sparse* A) const {
    M_cholmod_free_sparse(&A, const_cast<cholmod_common*>(&common()));
}

void CholmodWrapper::analyze(const SpMatd& pattern) {
    if (factor_) {
        M_cholmod_free_factor(&factor_, &common());
        factor_ = nullptr;
    }

    cholmod_sparse* C = eigen_to_cholmod(pattern);
    if (!C) throw std::runtime_error("CHOLMOD: failed to convert matrix");

    factor_ = M_cholmod_analyze(C, &common());
    free_cholmod_sparse(C);

    if (!factor_) throw std::runtime_error("CHOLMOD: symbolic analysis failed");
}

double CholmodWrapper::factorize(const SpMatd& A) {
    if (!factor_) throw std::runtime_error("CHOLMOD: must call analyze() first");

    cholmod_sparse* C = eigen_to_cholmod(A);
    if (!C) throw std::runtime_error("CHOLMOD: failed to convert matrix");

    int ok = M_cholmod_factorize(C, factor_, &common());
    free_cholmod_sparse(C);

    if (!ok || common().status != CHOLMOD_OK) {
        throw std::runtime_error("CHOLMOD: numeric factorisation failed");
    }

    return M_cholmod_factor_ldetA(factor_);
}

VectorXd CholmodWrapper::solve(const VectorXd& b) const {
    if (!factor_) throw std::runtime_error("CHOLMOD: no factorisation");

    int n = b.size();
    cholmod_dense b_chm;
    b_chm.nrow = n; b_chm.ncol = 1; b_chm.nzmax = n; b_chm.d = n;
    b_chm.x = const_cast<double*>(b.data());
    b_chm.z = nullptr;
    b_chm.xtype = CHOLMOD_REAL;
    b_chm.dtype = CHOLMOD_DOUBLE;

    cholmod_dense* x_chm = M_cholmod_solve(
        CHOLMOD_A, factor_, &b_chm, const_cast<cholmod_common*>(&common()));
    if (!x_chm) throw std::runtime_error("CHOLMOD: solve failed");

    VectorXd x(n);
    std::memcpy(x.data(), x_chm->x, n * sizeof(double));
    M_cholmod_free_dense(&x_chm, const_cast<cholmod_common*>(&common()));
    return x;
}

MatrixXd CholmodWrapper::solve(const MatrixXd& B) const {
    int n = B.rows(), nrhs = B.cols();
    MatrixXd X(n, nrhs);
    for (int j = 0; j < nrhs; ++j) {
        X.col(j) = solve(VectorXd(B.col(j)));
    }
    return X;
}

VectorXd CholmodWrapper::solve_L(const VectorXd& b) const {
    if (!factor_) throw std::runtime_error("CHOLMOD: no factorisation");

    int n = b.size();
    cholmod_dense b_chm;
    b_chm.nrow = n; b_chm.ncol = 1; b_chm.nzmax = n; b_chm.d = n;
    b_chm.x = const_cast<double*>(b.data());
    b_chm.z = nullptr;
    b_chm.xtype = CHOLMOD_REAL;
    b_chm.dtype = CHOLMOD_DOUBLE;

    // P b
    cholmod_dense* Pb = M_cholmod_solve(
        CHOLMOD_P, factor_, &b_chm, const_cast<cholmod_common*>(&common()));
    if (!Pb) throw std::runtime_error("CHOLMOD: P solve failed");

    // L^{-1} P b
    cholmod_dense* x_chm = M_cholmod_solve(
        CHOLMOD_L, factor_, Pb, const_cast<cholmod_common*>(&common()));
    M_cholmod_free_dense(&Pb, const_cast<cholmod_common*>(&common()));
    if (!x_chm) throw std::runtime_error("CHOLMOD: L solve failed");

    VectorXd x(n);
    std::memcpy(x.data(), x_chm->x, n * sizeof(double));
    M_cholmod_free_dense(&x_chm, const_cast<cholmod_common*>(&common()));
    return x;
}

VectorXd CholmodWrapper::solve_Lt(const VectorXd& b) const {
    if (!factor_) throw std::runtime_error("CHOLMOD: no factorisation");

    int n = b.size();
    cholmod_dense b_chm;
    b_chm.nrow = n; b_chm.ncol = 1; b_chm.nzmax = n; b_chm.d = n;
    b_chm.x = const_cast<double*>(b.data());
    b_chm.z = nullptr;
    b_chm.xtype = CHOLMOD_REAL;
    b_chm.dtype = CHOLMOD_DOUBLE;

    // L^{-T} b
    cholmod_dense* y_chm = M_cholmod_solve(
        CHOLMOD_Lt, factor_, &b_chm, const_cast<cholmod_common*>(&common()));
    if (!y_chm) throw std::runtime_error("CHOLMOD: Lt solve failed");

    // P^T result
    cholmod_dense* x_chm = M_cholmod_solve(
        CHOLMOD_Pt, factor_, y_chm, const_cast<cholmod_common*>(&common()));
    M_cholmod_free_dense(&y_chm, const_cast<cholmod_common*>(&common()));
    if (!x_chm) throw std::runtime_error("CHOLMOD: Pt solve failed");

    VectorXd x(n);
    std::memcpy(x.data(), x_chm->x, n * sizeof(double));
    M_cholmod_free_dense(&x_chm, const_cast<cholmod_common*>(&common()));
    return x;
}

} // namespace fastmlm
