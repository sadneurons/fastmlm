/* Pure C file that includes Matrix's CHOLMOD stubs.
 * This must be compiled as C (not C++) to avoid macro conflicts
 * between R's Rinternals.h and C++ standard library headers.
 *
 * The stubs define inline functions like M_cholmod_analyze() that
 * resolve CHOLMOD symbols at runtime from Matrix.so via R_GetCCallable.
 */

#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#define R_MATRIX_INLINE
#include <Matrix/cholmod.h>
#include <Matrix/stubs.c>
