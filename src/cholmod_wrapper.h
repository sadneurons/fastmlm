#ifndef FASTMLM_CHOLMOD_WRAPPER_H
#define FASTMLM_CHOLMOD_WRAPPER_H

#include "fastmlm_types.h"

// Forward-declare CHOLMOD types
struct cholmod_common_struct;
struct cholmod_factor_struct;
struct cholmod_sparse_struct;
typedef struct cholmod_common_struct cholmod_common;
typedef struct cholmod_factor_struct cholmod_factor;
typedef struct cholmod_sparse_struct cholmod_sparse;

namespace fastmlm {

class CholmodWrapper {
public:
    CholmodWrapper();
    ~CholmodWrapper();

    CholmodWrapper(const CholmodWrapper&) = delete;
    CholmodWrapper& operator=(const CholmodWrapper&) = delete;

    void analyze(const SpMatd& pattern);
    double factorize(const SpMatd& A);

    VectorXd solve(const VectorXd& b) const;
    MatrixXd solve(const MatrixXd& B) const;
    VectorXd solve_L(const VectorXd& b) const;
    VectorXd solve_Lt(const VectorXd& b) const;

    bool is_analyzed() const { return factor_ != nullptr; }

private:
    cholmod_factor* factor_;

    // Pre-allocated CHOLMOD sparse matrix (reused across factorize calls)
    cholmod_sparse* cached_sparse_;
    int cached_n_;
    int cached_nnz_;

    // Aligned storage for cholmod_common (~2680 bytes on 64-bit)
    alignas(16) char common_storage_[4096];

    cholmod_common& common() { return *reinterpret_cast<cholmod_common*>(common_storage_); }
    const cholmod_common& common() const { return *reinterpret_cast<const cholmod_common*>(common_storage_); }

    // Update cached_sparse_ values from Eigen matrix (no allocation)
    void update_cholmod_values(const SpMatd& A);
};

} // namespace fastmlm

#endif // FASTMLM_CHOLMOD_WRAPPER_H
