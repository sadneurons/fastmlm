AC_INIT([fastmlm], [0.0.1])

## ---- OpenBLAS detection ----
AC_MSG_CHECKING([for OpenBLAS])
OPENBLAS_LIBS=""
OPENBLAS_CPPFLAGS=""

# Try pkg-config first
if pkg-config --exists openblas 2>/dev/null; then
    OPENBLAS_LIBS=$(pkg-config --libs openblas)
    OPENBLAS_CPPFLAGS="-DFASTMLM_HAS_OPENBLAS"
    AC_MSG_RESULT([yes (pkg-config: ${OPENBLAS_LIBS})])
else
    # Try direct library check
    save_LIBS="$LIBS"
    LIBS="-lopenblas $LIBS"
    AC_LANG([C])
    AC_LINK_IFELSE(
        [AC_LANG_PROGRAM([[extern void dgemm_();]], [[dgemm_();]])],
        [
            OPENBLAS_LIBS="-lopenblas"
            OPENBLAS_CPPFLAGS="-DFASTMLM_HAS_OPENBLAS"
            AC_MSG_RESULT([yes (-lopenblas)])
        ],
        [
            OPENBLAS_LIBS=""
            AC_MSG_RESULT([no, using R default BLAS])
        ]
    )
    LIBS="$save_LIBS"
fi

# Fall back to R's BLAS if no OpenBLAS
if test -z "$OPENBLAS_LIBS"; then
    OPENBLAS_LIBS='$(BLAS_LIBS)'
fi

## ---- OpenMP detection ----
AC_MSG_CHECKING([for OpenMP])
OPENMP_CXXFLAGS='$(SHLIB_OPENMP_CXXFLAGS)'
OPENMP_LIBS='$(SHLIB_OPENMP_CXXFLAGS)'
OPENMP_CPPFLAGS=""

# Check if the compiler supports -fopenmp
AC_LANG([C])
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -fopenmp"
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([], [])],
    [
        OPENMP_CPPFLAGS="-DFASTMLM_HAS_OPENMP"
        AC_MSG_RESULT([yes])
    ],
    [
        OPENMP_CXXFLAGS=""
        OPENMP_LIBS=""
        AC_MSG_RESULT([no])
    ]
)
CFLAGS="$save_CFLAGS"

## ---- CUDA detection (Phase 5 placeholder) ----
AC_MSG_CHECKING([for CUDA/nvcc])
CUDA_CPPFLAGS=""
CUDA_LIBS=""
GPU_OBJECT="gpu_backend_stub"

AC_CHECK_PROG([NVCC], [nvcc], [yes], [no])
if test "x$NVCC" = "xyes"; then
    CUDA_CPPFLAGS="-DFASTMLM_HAS_CUDA"
    CUDA_LIBS="-lcublas -lcusolver -lcusparse"
    GPU_OBJECT="gpu_backend"
    AC_MSG_RESULT([yes])
else
    AC_MSG_RESULT([no])
fi

## ---- Output ----
AC_SUBST(OPENBLAS_LIBS)
AC_SUBST(OPENBLAS_CPPFLAGS)
AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_LIBS)
AC_SUBST(OPENMP_CPPFLAGS)
AC_SUBST(CUDA_CPPFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(GPU_OBJECT)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
