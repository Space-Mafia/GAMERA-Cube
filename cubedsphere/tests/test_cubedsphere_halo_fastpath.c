#include "cubedsphere_halo_fastpath.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void accumulate(const double *values, const double *weights,
                       int compact, double result[16]) {
  const int finite = compact && cs_halo_values_finite(values, 64U * 16U);
  memset(result, 0, 16U * sizeof(double));
  for (int item = 0; item < 64; ++item) {
    if (cs_halo_skip_zero(weights[item], finite)) continue;
    for (int channel = 0; channel < 16; ++channel)
      result[channel] += weights[item] * values[item * 16 + channel];
  }
}
int main(void) {
  uint32_t seed = 123456789U;
  double values[64 * 16], weights[64], axis_weights[2][8];
  double reference[16], compact[16], cached[16];
  unsigned char indices[64];
  for (int trial = 0; trial < 2048; ++trial) {
    const int bp = trial % 7, bq = (trial / 7) % 7;
    const double x = (trial % 17) / 8.0 - 0.5;
    const double y = (trial % 13) / 6.0 - 0.5;
    for (int p = 0; p < 8; ++p) {
      axis_weights[0][p] = p == bp ? 1.0-x : p == bp+1 ? x : 0.0;
      axis_weights[1][p] = p == bq ? 1.0-y : p == bq+1 ? y : 0.0;
    }
    /* Also exercise arbitrary dense stencils and the all-zero stencil. */
    if (trial % 31 == 0)
      for (int p = 0; p < 8; ++p)
        axis_weights[0][p] = axis_weights[1][p] = (p + 1.0) / 36.0;
    if (trial % 37 == 0) memset(axis_weights, 0, sizeof(axis_weights));
    for (int p = 0; p < 8; ++p)
      for (int q = 0; q < 8; ++q) {
        const double wp = axis_weights[0][p];
        const double wq = axis_weights[1][q];
        weights[p * 8 + q] = wp * wq;
      }
    for (int i = 0; i < 64 * 16; ++i) {
      seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
      values[i] = ((double)seed - 2147483648.0) / 1024.0;
      if (trial % 8 == 0) values[i] = (i % 2) ? -0.0 : 0.0;
      if (trial % 8 == 1) values[i] *= DBL_MIN;
      if (trial % 8 == 2) values[i] *= 1e300;
    }
    if (trial % 8 >= 3 && trial % 8 <= 5) {
      int zero = 0;
      while (zero < 63 && weights[zero] != 0.0) ++zero;
      values[zero * 16 + trial % 16] = trial % 8 == 3 ?
          INFINITY : trial % 8 == 4 ? -INFINITY : NAN;
      if (cs_halo_values_finite(values, 64U * 16U)) return 1;
    }
    accumulate(values, weights, 0, reference);
    accumulate(values, weights, 1, compact);
    const unsigned char count = cs_halo_nonzero_terms(axis_weights, indices);
    int expected = 0;
    for (int item = 0; item < 64; ++item)
      if (weights[item] != 0.0) {
        if (expected >= count || indices[expected++] != item) return 1;
      }
    if (expected != count) return 1;
    const int finite = cs_halo_values_finite(values, 64U * 16U);
    memset(cached, 0, sizeof(cached));
    for (int term = 0; term < (finite ? count : 64); ++term) {
      const int item = finite ? indices[term] : term;
      for (int channel = 0; channel < 16; ++channel)
        cached[channel] += weights[item] * values[item * 16 + channel];
    }
    for (int channel = 0; channel < 16; ++channel)
      if (memcmp(&reference[channel], &compact[channel], sizeof(double)) != 0 &&
          !(isnan(reference[channel]) && isnan(compact[channel]))) {
        fprintf(stderr, "halo fastpath mismatch trial=%d channel=%d\n", trial, channel);
        return 1;
      }
    for (int channel = 0; channel < 16; ++channel)
      if (memcmp(&reference[channel], &cached[channel], sizeof(double)) != 0 &&
          !(isnan(reference[channel]) && isnan(cached[channel]))) return 1;
  }
  puts("HALO_FASTPATH_FINITE_NONFINITE_INDEX_CACHE_PARITY_PASS trials=2048");
  return 0;
}
