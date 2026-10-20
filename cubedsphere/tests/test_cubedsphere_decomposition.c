#include "cubedsphere_decomposition.h"
#include "cubedsphere_halo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_layout(const size_t global[3], const int dims[3]) {
  const int count = GAMERA_CS_FACE_COUNT * dims[0] * dims[1] * dims[2];
  for (int rank = 0; rank < count; ++rank) {
    gamera_cs_decomposition tile;
    int inverse = -1;
    if (gamera_cs_decomposition_init(global, dims, count, rank, &tile) != 0 ||
        gamera_cs_decomposition_rank(tile.face, tile.coord, dims,
                                     &inverse) != 0 || inverse != rank)
      return -1;
    for (int axis = 0; axis < 3; ++axis)
      if (tile.global_start[axis] + tile.local_active[axis] > global[axis] ||
          tile.local_active[axis] * (size_t)dims[axis] != global[axis])
        return -1;
    for (int sample = 0; sample < 3; ++sample) {
      size_t global_cell[3], local_cell[3];
      int owner = -1;
      for (int axis = 0; axis < 3; ++axis)
        global_cell[axis] = tile.global_start[axis] +
            (sample == 0 ? 0 : sample == 1 ? tile.local_active[axis] / 2 :
             tile.local_active[axis] - 1);
      if (gamera_cs_decomposition_cell_owner(
              global, dims, tile.face, global_cell, &owner,
              local_cell) != 0 || owner != rank) return -1;
      for (int axis = 0; axis < 3; ++axis)
        if (local_cell[axis] !=
            global_cell[axis] - tile.global_start[axis]) return -1;
    }
  }
  return 0;
}

static int check_tile_vertices(void) {
  const size_t global[3] = {72, 32, 32};
  const int dims[3] = {2, 2, 2};
  const size_t ghost = 4;
  const size_t full_extent[3] = {81, 41, 41};
  gamera_no_radial_map map;
  if (gamera_no_radial_map_init(
          &map, GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
          2.5, 200.0, 3.0) != 0) return -1;
  gamera_no_vec3 *full = calloc(gamera_no_element_count3(full_extent),
                                sizeof(*full));
  if (full == NULL) return -1;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    if (gamera_cs_generate_vertices_radial_map(
            face, global, ghost, &map, full) != 0) goto failed;
    for (int rank = face * 8; rank < (face + 1) * 8; ++rank) {
      gamera_cs_decomposition tile;
      if (gamera_cs_decomposition_init(global, dims, 48, rank, &tile) != 0)
        goto failed;
      const size_t extent[3] = {
          tile.local_active[0] + 2 * ghost + 1,
          tile.local_active[1] + 2 * ghost + 1,
          tile.local_active[2] + 2 * ghost + 1};
      gamera_no_vec3 *local = calloc(gamera_no_element_count3(extent),
                                     sizeof(*local));
      if (local == NULL || gamera_cs_generate_vertices_radial_map_tile(
              face, global, tile.global_start, tile.local_active,
              ghost, &map, local) != 0) {
        free(local);
        goto failed;
      }
      for (size_t i = 0; i < extent[0]; ++i)
        for (size_t j = 0; j < extent[1]; ++j)
          for (size_t k = 0; k < extent[2]; ++k) {
            const size_t f = gamera_no_index3(full_extent,
                tile.global_start[0] + i, tile.global_start[1] + j,
                tile.global_start[2] + k);
            const size_t t = gamera_no_index3(extent, i, j, k);
            for (int axis = 0; axis < 3; ++axis)
              if (fabs(full[f].value[axis] - local[t].value[axis]) > 2e-13) {
                fprintf(stderr,
                        "tile vertex mismatch face=%d rank=%d i=%zu j=%zu k=%zu\n",
                        face, rank, i, j, k);
                free(local);
                goto failed;
              }
          }
      free(local);
    }
  }
  free(full);
  return 0;
failed:
  free(full);
  return -1;
}

/* A high-resolution mesh must be checked without allocating the full globe.
 * This also exercises unequal local angular widths on one face. */
static int check_sampled_tile(const size_t global[3], const int dims[3],
                              const int coord[3]) {
  const size_t ghost = 4;
  int rank = -1;
  const int face = GAMERA_CS_POS_Z;
  const int world_size = GAMERA_CS_FACE_COUNT * dims[0] * dims[1] * dims[2];
  gamera_cs_decomposition tile;
  gamera_no_radial_map map;
  if (gamera_cs_decomposition_rank(face, coord, dims, &rank) != 0 ||
      gamera_cs_decomposition_init(global, dims, world_size, rank,
                                   &tile) != 0 ||
      gamera_no_radial_map_init(&map,
          GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
          2.5, 200.0, 3.0) != 0) return -1;
  const size_t extent[3] = {
      tile.local_active[0] + 2 * ghost + 1,
      tile.local_active[1] + 2 * ghost + 1,
      tile.local_active[2] + 2 * ghost + 1};
  gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(extent),
                                     sizeof(*vertices));
  if (vertices == NULL || gamera_cs_generate_vertices_radial_map_tile(
          face, global, tile.global_start, tile.local_active,
          ghost, &map, vertices) != 0) {
    free(vertices);
    return -1;
  }
  const double quarter_pi = 0.25 * acos(-1.0);
  const size_t sample[3][3] = {
      {0, ghost, extent[0] - 1},
      {0, ghost, extent[1] - 1},
      {0, ghost, extent[2] - 1}};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) {
        const size_t i = sample[0][a], j = sample[1][b], k = sample[2][c];
        const double radial_fraction =
            ((double)tile.global_start[0] + (double)i - ghost) / global[0];
        const double angle_a = -quarter_pi +
            ((double)tile.global_start[1] + (double)j - ghost) *
                (2.0 * quarter_pi / global[1]);
        const double angle_b = -quarter_pi +
            ((double)tile.global_start[2] + (double)k - ghost) *
                (2.0 * quarter_pi / global[2]);
        double radius;
        gamera_no_vec3 direction;
        if (gamera_no_radial_map_forward(&map, radial_fraction, &radius) != 0 ||
            gamera_cs_direction(face, angle_a, angle_b, &direction) != 0) {
          free(vertices);
          return -1;
        }
        const gamera_no_vec3 vertex = vertices[
            gamera_no_index3(extent, i, j, k)];
        for (int axis = 0; axis < 3; ++axis)
          if (fabs(vertex.value[axis] - radius * direction.value[axis]) >
              2e-13 * fmax(1.0, radius)) {
            free(vertices);
            return -1;
          }
      }
  free(vertices);
  return 0;
}

static int check_sparse_stencil(const size_t global[3], const int dims[3],
                                double a, double b, int crosses_face) {
  const double quarter_pi = 0.25 * acos(-1.0);
  gamera_no_vec3 point;
  gamera_cs_tiled_donor_stencil stencil;
  if (gamera_cs_direction(GAMERA_CS_POS_X, a, b, &point) != 0 ||
      gamera_cs_tiled_donor_stencil_create(
          point, global, dims, 71, 0, &stencil) != 0 ||
      (stencil.donor_face != GAMERA_CS_POS_X) != crosses_face) return -1;
  double sum_a = 0.0, sum_b = 0.0;
  int distinct[64], distinct_count = 0;
  const int world_size = GAMERA_CS_FACE_COUNT * dims[0] * dims[1] * dims[2];
  for (size_t p = 0; p < 8; ++p) {
    sum_a += stencil.weight[0][p];
    sum_b += stencil.weight[1][p];
    for (size_t q = 0; q < 8; ++q) {
      const int owner = stencil.rank[p][q];
      gamera_cs_decomposition tile;
      if (gamera_cs_decomposition_init(global, dims, world_size,
                                       owner, &tile) != 0 ||
          tile.face != stencil.donor_face ||
          tile.global_start[0] + stencil.local_cell[p][q][0] != 71 ||
          tile.global_start[1] + stencil.local_cell[p][q][1] !=
              stencil.start[0] + p ||
          tile.global_start[2] + stencil.local_cell[p][q][2] !=
              stencil.start[1] + q) return -1;
      int seen = 0;
      for (int n = 0; n < distinct_count; ++n)
        seen |= distinct[n] == owner;
      if (!seen) distinct[distinct_count++] = owner;
    }
  }
  if (fabs(sum_a - 1.0) > 2e-13 || fabs(sum_b - 1.0) > 2e-13 ||
      (!crosses_face && distinct_count < 4)) return -1;
  /* Exercise both accepted cell-centroid and magnetic-face operation order. */
  return gamera_cs_tiled_donor_stencil_create(
      point, global, dims, 71, 1, &stencil);
}

int main(void) {
  const size_t global[3] = {72, 32, 32};
  const int face_only[3] = {1, 1, 1};
  const int radial[3] = {2, 1, 1};
  const int angular[3] = {1, 2, 1};
  const int radial_angular[3] = {2, 2, 2};
  const size_t medium[3] = {96, 64, 64};
  const int rectangular_tiles[3] = {3, 2, 4};
  const int medium_coord[3] = {1, 1, 2};
  const size_t large[3] = {256, 256, 256};
  const int large_tiles[3] = {16, 16, 16};
  const int large_coord[3] = {5, 7, 11};
  const double quarter_pi = 0.25 * acos(-1.0);
  const double da = 2.0 * quarter_pi / (double)large[1];
  const size_t outside[3] = {256, 0, 0};
  size_t local_outside[3];
  int outside_rank = -1;
  if (check_layout(global, face_only) != 0 ||
      check_layout(global, radial) != 0 ||
      check_layout(global, angular) != 0 ||
      check_layout(global, radial_angular) != 0 ||
      check_layout(medium, rectangular_tiles) != 0 ||
      check_layout(large, large_tiles) != 0 ||
      check_tile_vertices() != 0 ||
      check_sampled_tile(medium, rectangular_tiles, medium_coord) != 0 ||
      check_sampled_tile(large, large_tiles, large_coord) != 0 ||
      check_sparse_stencil(large, large_tiles,
          -quarter_pi + 32.3 * da, -quarter_pi + 48.3 * da, 0) != 0 ||
      check_sparse_stencil(large, large_tiles,
          quarter_pi + 0.01, 0.1, 1) != 0 ||
      gamera_cs_decomposition_cell_owner(
          large, large_tiles, 0, outside, &outside_rank,
          local_outside) == 0) return 1;
  puts("cubed-sphere face/radial/angular MPI tile topology and Earth mesh PASS");
  return 0;
}
