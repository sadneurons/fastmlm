// CPU fallback stubs when CUDA is not available.
// All functions either return false/empty or fall back to CPU computation.

#include "gpu_backend.h"

namespace fastmlm {
namespace gpu {

bool is_available() { return false; }

MatrixXd gpu_XtX(const MatrixXd& X) {
    // CPU fallback: standard Eigen multiply
    return X.transpose() * X;
}

VectorXd gpu_Xty(const MatrixXd& X, const VectorXd& y) {
    return X.transpose() * y;
}

VectorXd gpu_spmv(const SpMatd& A, const VectorXd& x) {
    return A * x;
}

MatrixXd gpu_cholesky(const MatrixXd& A) {
    Eigen::LLT<MatrixXd> llt(A);
    return llt.matrixL();
}

std::string device_name() { return "none (CPU fallback)"; }
size_t device_memory_bytes() { return 0; }

} // namespace gpu
} // namespace fastmlm
