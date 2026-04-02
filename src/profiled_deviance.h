#ifndef FASTMLM_PROFILED_DEVIANCE_H
#define FASTMLM_PROFILED_DEVIANCE_H

#include "model_data.h"
#include "sparse_cholesky.h"
#include "crossed_re.h"

namespace fastmlm {

// Evaluates the profiled REML or ML deviance as a function of theta.
//
// Key optimization: the sparsity pattern of A = Lamt * ZtZ * Lamt^T + I
// is fixed across iterations (only values change). We precompute the
// pattern once and maintain a mapping from (theta, Lind) to the
// numerical values of A, avoiding redundant sparse matrix multiplications.
class ProfiledDeviance {
public:
    ProfiledDeviance(MLMData& data, bool REML = true,
                     int pcg_threshold = 5000, int n_probes = 30);

    double operator()(const VectorXd& theta);

    const VectorXd& beta() const { return beta_; }
    const VectorXd& u() const { return u_; }
    double sigma2() const { return sigma2_; }
    double deviance() const { return deviance_; }
    double ldL2() const { return ldL2_; }
    double ldRX2() const { return ldRX2_; }
    double pwrss() const { return pwrss_; }

    MatrixXd vcov_beta_unscaled() const;

    bool is_REML() const { return REML_; }
    bool using_pcg() const { return use_pcg_; }
    MLMData& data() { return data_; }

private:
    MLMData& data_;
    bool REML_;
    bool use_pcg_;
    int n_probes_;

    // Sparse Cholesky
    SparseCholeskyManager L_chol_;
    bool chol_initialized_;

    // PCG for crossed RE
    CrossedRESolver crossed_solver_;

    // Dense Cholesky for RXtRX
    Eigen::LLT<MatrixXd> RX_chol_;

    // Cached results
    VectorXd beta_;
    VectorXd u_;
    double sigma2_, deviance_, ldL2_, ldRX2_, pwrss_;

    // Working storage (reused across iterations)
    VectorXd cu_;               // L^{-1} Lambdat Zty
    MatrixXd RZX_;              // L^{-1} Lambdat ZtX

    // === Precomputed structure for fast A update ===
    // A = Lamt * ZtZ * Lamt^T + I
    // We store A's sparsity pattern and a mapping that lets us
    // recompute A's values directly from theta without forming
    // the sparse triple product.
    SpMatd A_pattern_;          // pre-allocated sparse matrix (pattern fixed)
    bool pattern_initialized_;

    // Precomputed mapping: for each nonzero in A, how to compute its
    // value from Lambdat diagonal entries and ZtZ values.
    // Handles random intercept (k=1) with a fast scalar multiply,
    // and random slopes (k>1) via small dense block products.
    struct AMapping {
        int ztz_idx;    // index into ZtZ.valuePtr() (-1 if no ZtZ contribution)
        int lamt_row;   // Lambdat diagonal index for the row (for k=1 case)
        int lamt_col;   // Lambdat diagonal index for the col (for k=1 case)
        bool is_diag;   // true if on diagonal (add 1.0)
    };
    std::vector<AMapping> a_map_;
    bool is_all_intercept_;  // true if all RE terms are random-intercept-only (k=1)
    VectorXi diag_indices_;  // position of diagonal entries in A.valuePtr()

    // Precomputed sparse-dense products
    MatrixXd ZtX_;              // Zt * X, computed once (q x p)
    bool ztx_initialized_;

    // Path dispatch
    double eval_cholesky(const VectorXd& theta);
    double eval_pcg(const VectorXd& theta);

    // Fast update of A = Lamt * ZtZ * Lamt^T + I using precomputed pattern
    void update_A_fast(const VectorXd& theta);

    // Initialise pattern on first call
    void init_pattern(const SpMatd& A);
};

} // namespace fastmlm

#endif // FASTMLM_PROFILED_DEVIANCE_H
