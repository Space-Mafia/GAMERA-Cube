#ifndef GAMERA_CS_HALO_FASTPATH_H
#define GAMERA_CS_HALO_FASTPATH_H
#include <math.h>
#include <stddef.h>
/* Zero-weight terms may only be omitted when every donor value is finite.
 * In particular, retain the old 0*Inf/NaN behavior and nearest-state fallback.
 * Nonzero terms must retain their original p-major/q-minor summation order. */
static inline int cs_halo_values_finite(const double *values, size_t count) {
  for (size_t i = 0; i < count; ++i)
    if (!isfinite(values[i])) return 0;
  return 1;
}
static inline int cs_halo_skip_zero(double weight, int all_finite) {
  return all_finite && weight == 0.0;
}
/* Geometry-only cache; keep the exact original p-major/q-minor ordering.
 * The full 64 donors remain in the exchange for nonfinite detection/fallback. */
static inline unsigned char cs_halo_nonzero_terms(
    const double weights[2][8], unsigned char indices[64]) {
  unsigned char count = 0;
  for (int p = 0; p < 8; ++p)
    for (int q = 0; q < 8; ++q)
      if (weights[0][p] * weights[1][q] != 0.0)
        indices[count++] = (unsigned char)(p * 8 + q);
  return count;
}
#endif
