#include "model_data.h"

namespace fastmlm {

MLMData::MLMData(const VectorXd& y_,
                 const MatrixXd& X_,
                 const SpMatd& Zt_,
                 const SpMatd& Lambdat_,
                 const VectorXi& Lind_,
                 const VectorXd& lower_)
    : y(y_), X(X_), Zt(Zt_), Lambdat(Lambdat_),
      Lind(Lind_), lower(lower_),
      n_re_terms(1), is_crossed(false)
{
    n = y.size();
    p = X.cols();
    q = Zt.rows();
    nth = lower.size();

    // Default: single block covering all q columns
    block_sizes.resize(1);
    block_sizes[0] = q;
    block_starts.resize(1);
    block_starts[0] = 0;

    precompute();
}

void MLMData::set_block_structure(const VectorXi& Gp) {
    // Gp is the group pointer vector from lme4: length = n_re_terms + 1
    // Gp[i] is the starting column index of RE term i in Z
    // Gp[n_re_terms] = q
    n_re_terms = Gp.size() - 1;

    block_sizes.resize(n_re_terms);
    block_starts.resize(n_re_terms);

    for (int i = 0; i < n_re_terms; ++i) {
        block_starts[i] = Gp[i];
        block_sizes[i] = Gp[i + 1] - Gp[i];
    }

    detect_crossed();
}

void MLMData::precompute() {
    // Z^T Z — sparse cross product, computed once
    SpMatd Z = Zt.transpose();
    ZtZ = Zt * Z;

    // Dense cross-products for fixed effects
    XtX = X.transpose() * X;
    Xty = X.transpose() * y;
    Zty = Zt * y;
}

void MLMData::detect_crossed() {
    if (n_re_terms <= 1) {
        is_crossed = false;
        return;
    }

    // Check if RE terms are crossed by examining if ZtZ has nonzero
    // off-diagonal blocks. If block (i, j) of ZtZ (for i != j) has
    // any nonzeros, the corresponding RE terms share observations.
    is_crossed = false;

    for (int t1 = 0; t1 < n_re_terms && !is_crossed; ++t1) {
        int start1 = block_starts[t1];
        int end1 = start1 + block_sizes[t1];

        for (int t2 = t1 + 1; t2 < n_re_terms && !is_crossed; ++t2) {
            int start2 = block_starts[t2];
            int end2 = start2 + block_sizes[t2];

            // Check if the (t1, t2) block of ZtZ has nonzeros
            for (int col = start2; col < end2 && !is_crossed; ++col) {
                for (SpMatd::InnerIterator it(ZtZ, col); it; ++it) {
                    if (it.row() >= start1 && it.row() < end1) {
                        is_crossed = true;
                        break;
                    }
                }
            }
        }
    }
}

void MLMData::update_Lambdat(const VectorXd& theta) {
    double* x = Lambdat.valuePtr();
    int nnz = Lambdat.nonZeros();
    for (int i = 0; i < nnz; ++i) {
        x[i] = theta[Lind[i] - 1];
    }
}

} // namespace fastmlm
