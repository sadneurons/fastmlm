#include "blas_detect.h"
#include <RcppEigen.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// OpenBLAS thread control (weak symbols — resolve only if OpenBLAS is linked)
extern "C" {
    void openblas_set_num_threads(int num_threads) __attribute__((weak));
    int openblas_get_num_threads(void) __attribute__((weak));
    char* openblas_get_config(void) __attribute__((weak));
    char* openblas_get_corename(void) __attribute__((weak));
}

// MKL thread control (weak symbols)
extern "C" {
    void mkl_set_num_threads(int num_threads) __attribute__((weak));
    int mkl_get_max_threads(void) __attribute__((weak));
}

namespace fastmlm {
namespace blas {

std::string detect_library() {
    // Check for OpenBLAS
    if (openblas_get_config) {
        std::string info = "OpenBLAS";
        char* config = openblas_get_config();
        if (config) {
            info += " (";
            info += config;
            info += ")";
        }
        if (openblas_get_corename) {
            char* core = openblas_get_corename();
            if (core) {
                info += " [";
                info += core;
                info += "]";
            }
        }
        return info;
    }

    // Check for MKL
    if (mkl_get_max_threads) {
        return "Intel MKL";
    }

#ifdef FASTMLM_HAS_OPENBLAS
    return "OpenBLAS (compile-time linked)";
#else
    return "Reference BLAS (R default)";
#endif
}

int get_num_threads() {
    if (openblas_get_num_threads) {
        return openblas_get_num_threads();
    }
    if (mkl_get_max_threads) {
        return mkl_get_max_threads();
    }
    return 1;
}

void set_num_threads(int n) {
    if (n < 1) n = 1;
    if (openblas_set_num_threads) {
        openblas_set_num_threads(n);
    }
    if (mkl_set_num_threads) {
        mkl_set_num_threads(n);
    }
}

int get_omp_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void set_omp_threads(int n) {
#ifdef _OPENMP
    if (n < 1) n = 1;
    omp_set_num_threads(n);
#endif
    (void)n;
}

double benchmark_dgemm(int n) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);

    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd C = A * B;
    auto end = std::chrono::high_resolution_clock::now();

    // Prevent optimiser from eliding the multiplication
    volatile double sink = C(0, 0);
    (void)sink;

    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

} // namespace blas
} // namespace fastmlm
