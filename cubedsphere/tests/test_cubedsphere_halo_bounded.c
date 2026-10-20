#include "cubedsphere_halo.h"
#include <math.h>
#include <stdio.h>

/* Interpolation is convex; half-center cube-corner extension is bounded.
 * The original valid-state/nearest-donor fallback remains required.
 */
int main(void) {
  const size_t sizes[] = {16, 24, 32, 64, 256};
  const int dims[3] = {2, 2, 2};
  const gamera_no_vec3 points[] = {
      {{1,1,1}}, {{1,-1,1}}, {{-1,1,1}}, {{-1,-1,-1}},
      {{1,1.01,0.99}}, {{1,0.3,-1.02}}, {{0.1,1,0.2}}};
  for (size_t n=0; n<sizeof(sizes)/sizeof(sizes[0]); ++n) {
    const size_t shape[3] = {32,sizes[n],sizes[n]};
    for (size_t p=0; p<sizeof(points)/sizeof(points[0]); ++p) {
      for (int magnetic=0; magnetic<2; ++magnetic) {
        gamera_cs_tiled_donor_stencil s;
        if (gamera_cs_tiled_donor_stencil_create(points[p],shape,dims,19,magnetic,&s)) return 1;
        double sum=0, norm=0;
        for (int j=0; j<8; ++j) for (int k=0; k<8; ++k) {
          double w=s.weight[0][j]*s.weight[1][k];
          if (!isfinite(w)) return 2;
          sum+=w;
          norm+=fabs(w);
        }
        if (fabs(sum-1)>1e-14 || norm>4.0+1e-12) return 3;
      }
    }
  }
  puts("cube halo linear weights, bounded corner extension, generic sizes PASS");
  return 0;
}
