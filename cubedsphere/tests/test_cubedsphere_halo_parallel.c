#define _POSIX_C_SOURCE 200809L
#include "cubedsphere_mpi_tile.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void initialize(gamera_no_storage *s, int rank, int sample) {
  const size_t n = gamera_no_element_count3(s->cell_extent);
  for (size_t c = 0; c < n; ++c) {
    const double x = 0.001 * rank + 0.00001 * c + 0.03 * sample;
    for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
      double v = m == GAMERA_NO_FLUX_DENSITY ? 2.0 + x :
          m == GAMERA_NO_FLUX_ENERGY ? 10.0 + x : 0.001 * (m + 1) + x;
      if (sample == 1 && m == GAMERA_NO_FLUX_DENSITY) v = -v;
      if (sample == 3 && m != GAMERA_NO_FLUX_DENSITY &&
          m != GAMERA_NO_FLUX_ENERGY) v = -0.0;
      if (sample == 4 && m == GAMERA_NO_FLUX_ENERGY) v = INFINITY;
      s->conserved[c * GAMERA_NO_FLUX_COUNT + m] = v;
      s->old_conserved[c * GAMERA_NO_FLUX_COUNT + m] = v * 1.125;
    }
    for (int a = 0; a < 3; ++a) {
      double v = 0.2 + x + 0.01 * a;
      if (sample == 2 && a == 1) v = NAN;
      if (sample == 3) v = -0.0;
      s->cell_magnetic[c].value[a] = v;
      s->old_cell_magnetic[c].value[a] = v * 1.25;
    }
  }
  for (int a = 0; a < 3; ++a) {
    const size_t faces = gamera_no_element_count3(s->face_extent[a]);
    for (size_t c = 0; c < faces; ++c) {
      s->face_flux[a][c] = -777.0;
      s->old_face_flux[a][c] = -888.0;
    }
  }
}

static int same_bits(const gamera_no_storage *a, const gamera_no_storage *b) {
  const size_t n = gamera_no_element_count3(a->cell_extent);
  if (memcmp(a->conserved, b->conserved, n * GAMERA_NO_FLUX_COUNT * sizeof(double)) ||
      memcmp(a->old_conserved, b->old_conserved, n * GAMERA_NO_FLUX_COUNT * sizeof(double)) ||
      memcmp(a->cell_magnetic, b->cell_magnetic, n * sizeof(gamera_no_vec3)) ||
      memcmp(a->old_cell_magnetic, b->old_cell_magnetic, n * sizeof(gamera_no_vec3))) return 0;
  for (int d = 0; d < 3; ++d) {
    const size_t bytes = gamera_no_element_count3(a->face_extent[d]) * sizeof(double);
    if (memcmp(a->face_flux[d], b->face_flux[d], bytes) ||
        memcmp(a->old_face_flux[d], b->old_face_flux[d], bytes)) return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int threads = 1;
#ifdef _OPENMP
  omp_set_dynamic(0);
#pragma omp parallel
  {
#pragma omp single
    threads = omp_get_num_threads();
  }
  const char *expected = getenv("OMP_NUM_THREADS");
  if (expected && threads != atoi(expected)) MPI_Abort(MPI_COMM_WORLD, 6);
#endif
  if (argc != 4) MPI_Abort(MPI_COMM_WORLD, 2);
  const int dims[3] = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
  const size_t global[3] = {16, 16, 16}, ghost = 4;
  gamera_cs_decomposition tile;
  if (gamera_cs_decomposition_init(global, dims, ranks, rank, &tile))
    MPI_Abort(MPI_COMM_WORLD, 3);
  const size_t extent[3] = {tile.local_active[0] + 2 * ghost,
      tile.local_active[1] + 2 * ghost, tile.local_active[2] + 2 * ghost};
  const size_t vertex_extent[3] = {extent[0]+1, extent[1]+1, extent[2]+1};
  gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(vertex_extent), sizeof(*vertices));
  gamera_no_radial_map map;
  gamera_no_grid grid = {0};
  gamera_no_storage s[2] = {{0}, {0}};
  if (!vertices || gamera_no_radial_map_init(&map,
          GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4, 2.5, 200.0, 3.0) ||
      gamera_cs_generate_vertices_radial_map_tile(tile.face, global,
          tile.global_start, tile.local_active, ghost, &map, vertices) ||
      gamera_no_grid_create(extent, vertices, &grid) ||
      gamera_no_storage_create(extent, &s[0]) ||
      gamera_no_storage_create(extent, &s[1])) MPI_Abort(MPI_COMM_WORLD, 4);
  free(vertices);
  int failed = 0;
  for (int mode = 0; mode < 2; ++mode) {
    setenv("GAMERA_CS_HALO_COMPACT", mode ? "1" : "0", 1);
    setenv("GAMERA_CS_HALO_INDEX_CACHE", mode ? "1" : "0", 1);
    gamera_cs_tiled_cell_halo_plan *cells[2] = {NULL, NULL};
    gamera_cs_tiled_magnetic_halo_plan *magnetic[2] = {NULL, NULL};
    for (int variant = 0; variant < 2; ++variant) {
      setenv("GAMERA_CS_HALO_BLOCK_ACCESS", "1", 1);
      setenv("GAMERA_CS_HALO_PARALLEL", variant ? "1" : "0", 1);
      if (gamera_cs_tiled_cell_halo_plan_create(&grid, &tile, ghost,
              MPI_COMM_WORLD, &cells[variant]) ||
          gamera_cs_tiled_magnetic_halo_plan_create(&grid, &tile, ghost,
              MPI_COMM_WORLD, &magnetic[variant])) MPI_Abort(MPI_COMM_WORLD, 5);
    }
    /* Repeated exchanges reuse plans with different donor values, including
     * nonfinite values (full stencil), invalid hydro (nearest fallback), and
     * signed zero. Compare every output byte, including untouched ghosts. */
    for (int repetition = 0; repetition < 4; ++repetition)
    for (int sample = 0; sample < 5; ++sample) {
      int status[2][2];
      for (int v = 0; v < 2; ++v) {
        initialize(&s[v], rank, sample);
        status[v][0] = gamera_cs_tiled_cell_halo_fill(cells[v], &s[v]);
        status[v][1] = gamera_cs_tiled_magnetic_halo_fill(magnetic[v], &s[v]);
      }
      if (memcmp(status[0], status[1], sizeof(status[0])) ||
          status[0][0] != 0 || status[0][1] != 0 || !same_bits(&s[0], &s[1])) {
        fprintf(stderr, "HALO_PARALLEL_BIT_FAIL rank=%d mode=%d sample=%d\n", rank, mode, sample);
        failed = 1;
      }
    }
    for (int v = 0; v < 2; ++v) {
      gamera_cs_tiled_cell_halo_plan_destroy(cells[v]);
      gamera_cs_tiled_magnetic_halo_plan_destroy(magnetic[v]);
    }
  }
  int any_failed;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  for (int v = 0; v < 2; ++v) gamera_no_storage_destroy(&s[v]);
  gamera_no_grid_destroy(&grid);
  if (!rank && !any_failed)
    printf("HALO_PARALLEL_DOUBLE_BIT_PASS modes=2 states=5 repetitions=4 actual_threads=%d\n", threads);
  MPI_Finalize();
  return any_failed ? 1 : 0;
}
