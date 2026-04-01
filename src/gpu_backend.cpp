// CUDA GPU backend for fastmlm.
// Uses cuBLAS for dense linear algebra and cuSPARSE for sparse MatVec.
// Only compiled when configure detects CUDA (FASTMLM_HAS_CUDA defined).

#ifdef FASTMLM_HAS_CUDA

#include "gpu_backend.h"
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstring>

namespace fastmlm {
namespace gpu {

// RAII handle managers
static cublasHandle_t cublas_handle = nullptr;
static cusolverDnHandle_t cusolver_handle = nullptr;
static cusparseHandle_t cusparse_handle = nullptr;
static bool initialized = false;

static void ensure_init() {
    if (initialized) return;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        initialized = true;  // mark as attempted
        return;
    }

    cudaSetDevice(0);

    cublasCreate(&cublas_handle);
    cusolverDnCreate(&cusolver_handle);
    cusparseCreate(&cusparse_handle);

    initialized = true;
}

bool is_available() {
    ensure_init();
    return cublas_handle != nullptr;
}

std::string device_name() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return "none";
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "unknown";
    return std::string(prop.name);
}

size_t device_memory_bytes() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    return prop.totalGlobalMem;
}

// Helper: allocate + copy to device
static double* to_device(const double* host, size_t n) {
    double* d_ptr;
    cudaMalloc(&d_ptr, n * sizeof(double));
    cudaMemcpy(d_ptr, host, n * sizeof(double), cudaMemcpyHostToDevice);
    return d_ptr;
}

// Helper: copy from device + free
static void from_device(double* d_ptr, double* host, size_t n) {
    cudaMemcpy(host, d_ptr, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_ptr);
}

MatrixXd gpu_XtX(const MatrixXd& X) {
    ensure_init();
    if (!cublas_handle) return X.transpose() * X;  // CPU fallback

    int n = X.rows();
    int p = X.cols();

    // Only worth GPU for large matrices
    if (n < 10000 || p < 10) {
        return X.transpose() * X;
    }

    // X is column-major (Eigen default). cuBLAS also expects column-major.
    // X^T X = DSYRK: C = alpha * A^T * A + beta * C
    double* d_X = to_device(X.data(), n * p);

    double* d_XtX;
    cudaMalloc(&d_XtX, p * p * sizeof(double));
    cudaMemset(d_XtX, 0, p * p * sizeof(double));

    double alpha = 1.0, beta = 0.0;
    // DSYRK: C = alpha * A^T * A (A is n×p, C is p×p)
    cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                p, n, &alpha, d_X, n, &beta, d_XtX, p);

    MatrixXd XtX(p, p);
    from_device(d_XtX, XtX.data(), p * p);
    cudaFree(d_X);

    // Fill lower triangle from upper
    for (int j = 0; j < p; ++j)
        for (int i = j + 1; i < p; ++i)
            XtX(i, j) = XtX(j, i);

    return XtX;
}

VectorXd gpu_Xty(const MatrixXd& X, const VectorXd& y) {
    ensure_init();
    if (!cublas_handle) return X.transpose() * y;

    int n = X.rows();
    int p = X.cols();

    if (n < 10000) return X.transpose() * y;

    double* d_X = to_device(X.data(), n * p);
    double* d_y = to_device(y.data(), n);

    double* d_Xty;
    cudaMalloc(&d_Xty, p * sizeof(double));

    double alpha = 1.0, beta = 0.0;
    // DGEMV: y = alpha * A^T * x + beta * y
    cublasDgemv(cublas_handle, CUBLAS_OP_T,
                n, p, &alpha, d_X, n, d_y, 1, &beta, d_Xty, 1);

    VectorXd Xty(p);
    from_device(d_Xty, Xty.data(), p);
    cudaFree(d_X);
    cudaFree(d_y);

    return Xty;
}

VectorXd gpu_spmv(const SpMatd& A, const VectorXd& x) {
    ensure_init();
    if (!cusparse_handle) return A * x;

    int m = A.rows();
    int n = A.cols();
    int nnz = A.nonZeros();

    // Only worth GPU for large sparse matrices
    if (m < 50000) return A * x;

    // CSC format → cuSPARSE expects CSR or CSC
    // Eigen stores in CSC. cuSPARSE can do CSC via transpose trick:
    // A * x in CSC = A^T * x in CSR (with A^T being the CSR view of A)
    // Actually cuSPARSE can handle CSC directly.

    // Copy to device
    double* d_vals = to_device(A.valuePtr(), nnz);
    int* d_outerIdx;
    int* d_innerIdx;
    cudaMalloc(&d_outerIdx, (n + 1) * sizeof(int));
    cudaMalloc(&d_innerIdx, nnz * sizeof(int));
    cudaMemcpy(d_outerIdx, A.outerIndexPtr(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_innerIdx, A.innerIndexPtr(), nnz * sizeof(int), cudaMemcpyHostToDevice);

    double* d_x = to_device(x.data(), n);
    double* d_y;
    cudaMalloc(&d_y, m * sizeof(double));

    // Create cuSPARSE descriptors
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    cusparseCreateCsc(&matA, m, n, nnz,
                      d_outerIdx, d_innerIdx, d_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F);

    double alpha = 1.0, beta = 0.0;
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    void* d_buffer;
    cudaMalloc(&d_buffer, bufferSize);

    cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);

    VectorXd y_out(m);
    cudaMemcpy(y_out.data(), d_y, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cudaFree(d_buffer);
    cudaFree(d_vals);
    cudaFree(d_outerIdx);
    cudaFree(d_innerIdx);
    cudaFree(d_x);
    cudaFree(d_y);

    return y_out;
}

MatrixXd gpu_cholesky(const MatrixXd& A) {
    ensure_init();
    if (!cusolver_handle) {
        Eigen::LLT<MatrixXd> llt(A);
        return llt.matrixL();
    }

    int n = A.rows();
    if (n < 500) {
        Eigen::LLT<MatrixXd> llt(A);
        return llt.matrixL();
    }

    // Copy A to device (column-major)
    double* d_A = to_device(A.data(), n * n);

    // Workspace query
    int lwork = 0;
    cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                 n, d_A, n, &lwork);

    double* d_work;
    cudaMalloc(&d_work, lwork * sizeof(double));

    int* d_info;
    cudaMalloc(&d_info, sizeof(int));

    // Cholesky factorisation
    cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                      n, d_A, n, d_work, lwork, d_info);

    MatrixXd L(n, n);
    from_device(d_A, L.data(), n * n);

    // Zero upper triangle (cuSOLVER writes lower)
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < j; ++i)
            L(i, j) = 0.0;

    cudaFree(d_work);
    cudaFree(d_info);

    return L;
}

} // namespace gpu
} // namespace fastmlm

#else
// If CUDA not available at compile time, use the stub
#include "gpu_backend.h"

namespace fastmlm {
namespace gpu {
bool is_available() { return false; }
MatrixXd gpu_XtX(const MatrixXd& X) { return X.transpose() * X; }
VectorXd gpu_Xty(const MatrixXd& X, const VectorXd& y) { return X.transpose() * y; }
VectorXd gpu_spmv(const SpMatd& A, const VectorXd& x) { return A * x; }
MatrixXd gpu_cholesky(const MatrixXd& A) {
    Eigen::LLT<MatrixXd> llt(A);
    return llt.matrixL();
}
std::string device_name() { return "none (CPU fallback)"; }
size_t device_memory_bytes() { return 0; }
} // namespace gpu
} // namespace fastmlm
#endif
