#ifndef FASTMLM_GPU_BACKEND_H
#define FASTMLM_GPU_BACKEND_H

#include "fastmlm_types.h"

namespace fastmlm {
namespace gpu {

// Check GPU availability at runtime
bool is_available();

// Dense cross-products on GPU (for large n)
// Returns X^T X computed on GPU via cuBLAS GEMM
MatrixXd gpu_XtX(const MatrixXd& X);

// Returns X^T y computed on GPU
VectorXd gpu_Xty(const MatrixXd& X, const VectorXd& y);

// Sparse matrix-vector product on GPU (for PCG iterations)
// Uses cuSPARSE
VectorXd gpu_spmv(const SpMatd& A, const VectorXd& x);

// Dense Cholesky on GPU (for large p x p RXtRX)
MatrixXd gpu_cholesky(const MatrixXd& A);

// GPU device info
std::string device_name();
size_t device_memory_bytes();

} // namespace gpu
} // namespace fastmlm

#endif // FASTMLM_GPU_BACKEND_H
