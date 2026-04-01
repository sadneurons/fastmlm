#include <Rcpp.h>

#define R_NO_REMAP
#include <Matrix/cholmod.h>
#undef R_NO_REMAP
#undef length
#undef FREE

#include "cholmod_wrapper.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

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

CholmodWrapper::CholmodWrapper()
    : factor_(nullptr), cached_sparse_(nullptr), cached_n_(0), cached_nnz_(0)
{
    M_cholmod_start(&common());
    common().supernodal = CHOLMOD_SUPERNODAL;
}

CholmodWrapper::~CholmodWrapper() {
    if (factor_) M_cholmod_free_factor(&factor_, &common());
    if (cached_sparse_) M_cholmod_free_sparse(&cached_sparse_, &common());
    M_cholmod_finish(&common());
}

void CholmodWrapper::update_cholmod_values(const SpMatd& A) {
    int n = A.rows();
    int nnz = A.nonZeros();

    // Allocate or reallocate if dimensions changed
    if (!cached_sparse_ || cached_n_ != n || cached_nnz_ != nnz) {
        if (cached_sparse_) M_cholmod_free_sparse(&cached_sparse_, &common());

        cached_sparse_ = M_cholmod_allocate_sparse(
            n, n, nnz, 1, 1, 0, CHOLMOD_REAL, &common());
        if (!cached_sparse_) throw std::runtime_error("CHOLMOD: alloc failed");

        cached_n_ = n;
        cached_nnz_ = nnz;

        // Copy structure (column pointers and row indices — stable across iterations)
        int* Cp = static_cast<int*>(cached_sparse_->p);
        int* Ci = static_cast<int*>(cached_sparse_->i);
        const int* Ap = A.outerIndexPtr();
        const int* Ai = A.innerIndexPtr();
        std::memcpy(Cp, Ap, (n + 1) * sizeof(int));
        std::memcpy(Ci, Ai, nnz * sizeof(int));
    }

    // Only copy numerical values (this is the fast path for repeat factorizations)
    std::memcpy(cached_sparse_->x, A.valuePtr(), nnz * sizeof(double));
    cached_sparse_->stype = 1;  // symmetric, upper triangle
}

void CholmodWrapper::analyze(const SpMatd& pattern) {
    if (factor_) {
        M_cholmod_free_factor(&factor_, &common());
        factor_ = nullptr;
    }

    update_cholmod_values(pattern);
    factor_ = M_cholmod_analyze(cached_sparse_, &common());
    if (!factor_) throw std::runtime_error("CHOLMOD: analysis failed");
}

double CholmodWrapper::factorize(const SpMatd& A) {
    if (!factor_) throw std::runtime_error("CHOLMOD: must call analyze() first");

    // Update only the numerical values (structure unchanged)
    update_cholmod_values(A);

    int ok = M_cholmod_factorize(cached_sparse_, factor_, &common());
    if (!ok || common().status != CHOLMOD_OK) {
        throw std::runtime_error("CHOLMOD: factorisation failed");
    }

    return M_cholmod_factor_ldetA(factor_);
}

VectorXd CholmodWrapper::solve(const VectorXd& b) const {
    if (!factor_) throw std::runtime_error("CHOLMOD: no factorisation");
    int n = b.size();

    cholmod_dense b_chm;
    b_chm.nrow = n; b_chm.ncol = 1; b_chm.nzmax = n; b_chm.d = n;
    b_chm.x = const_cast<double*>(b.data()); b_chm.z = nullptr;
    b_chm.xtype = CHOLMOD_REAL; b_chm.dtype = CHOLMOD_DOUBLE;

    cholmod_dense* x_chm = M_cholmod_solve(
        CHOLMOD_A, factor_, &b_chm, const_cast<cholmod_common*>(&common()));
    if (!x_chm) throw std::runtime_error("CHOLMOD: solve failed");

    VectorXd x(n);
    std::memcpy(x.data(), x_chm->x, n * sizeof(double));
    M_cholmod_free_dense(&x_chm, const_cast<cholmod_common*>(&common()));
    return x;
}

MatrixXd CholmodWrapper::solve(const MatrixXd& B) const {
    int nrhs = B.cols();
    MatrixXd X(B.rows(), nrhs);
    for (int j = 0; j < nrhs; ++j) X.col(j) = solve(VectorXd(B.col(j)));
    return X;
}

VectorXd CholmodWrapper::solve_L(const VectorXd& b) const {
    if (!factor_) throw std::runtime_error("CHOLMOD: no factorisation");
    int n = b.size();

    cholmod_dense b_chm;
    b_chm.nrow = n; b_chm.ncol = 1; b_chm.nzmax = n; b_chm.d = n;
    b_chm.x = const_cast<double*>(b.data()); b_chm.z = nullptr;
    b_chm.xtype = CHOLMOD_REAL; b_chm.dtype = CHOLMOD_DOUBLE;

    cholmod_dense* Pb = M_cholmod_solve(
        CHOLMOD_P, factor_, &b_chm, const_cast<cholmod_common*>(&common()));
    if (!Pb) throw std::runtime_error("CHOLMOD: P solve failed");

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
    b_chm.x = const_cast<double*>(b.data()); b_chm.z = nullptr;
    b_chm.xtype = CHOLMOD_REAL; b_chm.dtype = CHOLMOD_DOUBLE;

    cholmod_dense* y_chm = M_cholmod_solve(
        CHOLMOD_Lt, factor_, &b_chm, const_cast<cholmod_common*>(&common()));
    if (!y_chm) throw std::runtime_error("CHOLMOD: Lt solve failed");

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
