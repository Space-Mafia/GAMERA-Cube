#ifndef GAMERA_CUBEDSPHERE_DECOMPOSITION_H
#define GAMERA_CUBEDSPHERE_DECOMPOSITION_H

#include "cubedsphere_mesh.h"

#include <stddef.h>

/* Rank order matches the accepted Yin--Yang patch/cartesian convention:
 * one contiguous Cartesian rank block per cubed-sphere face.  This is
 * topology only; no MHD reconstruction or interface scheme lives here. */
typedef struct {
  int face;
  int coord[3];
  int proc_dims[3];
  size_t global_active[3];
  size_t local_active[3];
  size_t global_start[3];
} gamera_cs_decomposition;

int gamera_cs_decomposition_init(const size_t global_active[3],
                                 const int proc_dims[3], int world_size,
                                 int world_rank,
                                 gamera_cs_decomposition *result);

int gamera_cs_decomposition_rank(int face, const int coord[3],
                                 const int proc_dims[3], int *world_rank);

/* Resolve a global active-cell index to its unique tile and local index.
 * Cube-face interpolation requests use global donor coordinates; the local
 * tile width must never be mistaken for the full-face angular resolution. */
int gamera_cs_decomposition_cell_owner(
    const size_t global_active[3], const int proc_dims[3], int face,
    const size_t global_index[3], int *world_rank,
    size_t local_index[3]);

#endif
