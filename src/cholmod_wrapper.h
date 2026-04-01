#ifndef FASTMLM_CHOLMOD_WRAPPER_H
#define FASTMLM_CHOLMOD_WRAPPER_H

#include "fastmlm_types.h"

// Forward-declare CHOLMOD types to avoid including cholmod.h here
// (cholmod.h defines macros that conflict with C++ standard library).
// The actual #include <Matrix/cholmod.h> happens only in cholmod_wrapper.cpp.
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
    cholmod_common* common_ptr_;
    cholmod_factor* factor_;

    // Raw storage for cholmod_common (avoid including cholmod.h)
    // cholmod_common is ~720 bytes on 64-bit; we use aligned storage
    alignas(16) char common_storage_[1024];

    cholmod_common& common() { return *reinterpret_cast<cholmod_common*>(common_storage_); }
    const cholmod_common& common() const { return *reinterpret_cast<const cholmod_common*>(common_storage_); }

    cholmod_sparse* eigen_to_cholmod(const SpMatd& A) const;
    void free_cholmod_sparse(cholmod_sparse* A) const;
};

} // namespace fastmlm

#endif // FASTMLM_CHOLMOD_WRAPPER_H
