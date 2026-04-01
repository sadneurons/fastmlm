#ifndef FASTMLM_BLAS_DETECT_H
#define FASTMLM_BLAS_DETECT_H

#include <string>

namespace fastmlm {

// Runtime BLAS detection and thread control.
//
// Detects which BLAS library is linked at runtime by timing a small
// dgemm and checking for OpenBLAS-specific symbols. Also provides
// thread count control for OpenBLAS and MKL.
namespace blas {

    // Detect the linked BLAS library name
    std::string detect_library();

    // Get/set number of BLAS threads (OpenBLAS or MKL)
    int get_num_threads();
    void set_num_threads(int n);

    // Get/set number of OpenMP threads for fastmlm
    int get_omp_threads();
    void set_omp_threads(int n);

    // Quick benchmark: time a small dgemm to verify BLAS is working
    double benchmark_dgemm(int n = 500);

} // namespace blas
} // namespace fastmlm

#endif // FASTMLM_BLAS_DETECT_H
