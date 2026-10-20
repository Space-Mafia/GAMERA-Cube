#include "cubedsphere_decomposition.h"
#include "cubedsphere_earth_startup.h"
#include "cubedsphere_mesh.h"
#include "cubedsphere_mi_dipole.h"
#include "cubedsphere_mi_dipole_mpi.h"
#include "nonorthogonal_background.h"
#include "nonorthogonal_grid.h"
#include "nonorthogonal_radial_map.h"
#include "nonorthogonal_storage.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_residual(const gamera_no_grid *grid,
                         gamera_no_storage *storage,
                         const size_t active[3], size_t ghost) {
  for (size_t i = ghost; i < ghost + active[0]; ++i)
    for (size_t j = ghost; j < ghost + active[1]; ++j)
      for (size_t k = ghost; k < ghost + active[2]; ++k) {
        const size_t cell = gamera_no_index3(grid->cell_extent, i, j, k);
        const gamera_no_vec3 point = grid->cell[cell].centroid;
        const double r2 = point.value[0] * point.value[0] +
                          point.value[1] * point.value[1] +
                          point.value[2] * point.value[2];
        storage->cell_magnetic[cell] = (gamera_no_vec3){{
            0.03 * point.value[1] / r2,
            -0.02 * point.value[0] / r2,
            0.01 * point.value[0] * point.value[2] / r2}};
      }
}

static int make_face(int face, const size_t global[3],
                     const size_t start[3], const size_t active[3],
                     size_t ghost, const gamera_no_radial_map *map,
                     const gamera_cs_earth_startup *startup,
                     gamera_no_grid *grid, gamera_no_storage *storage,
                     gamera_no_background_data *background) {
  const size_t extent[3] = {active[0] + 2U * ghost,
      active[1] + 2U * ghost, active[2] + 2U * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1U,
      extent[1] + 1U, extent[2] + 1U};
  gamera_no_vec3 *vertices = calloc(
      gamera_no_element_count3(vertex_extent), sizeof(*vertices));
  int bad = vertices == NULL ||
      gamera_cs_generate_vertices_radial_map_tile(face, global,
          start, active, ghost, map, vertices) != 0 ||
      gamera_no_grid_create(extent, vertices, grid) != 0 ||
      gamera_cs_calibrate_face_frames(grid) != 0 ||
      gamera_no_storage_create(extent, storage) != 0 ||
      gamera_no_background_create(grid, gamera_cs_earth_startup_background,
          (void *)startup, background) != 0;
  free(vertices);
  if (!bad) set_residual(grid, storage, active, ghost);
  return bad ? -1 : 0;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = -1, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int dims[3] = {0};
  int bad = argc != 4 && argc != 5;
  if (!bad)
    for (int d = 0; d < 3; ++d) {
      dims[d] = atoi(argv[d + 1]);
      if (dims[d] <= 0) bad = 1;
    }
  const size_t global[3] = {16U, 16U, 16U};
  const size_t start0[3] = {0U, 0U, 0U};
  const size_t ghost = 4U;
  gamera_cs_decomposition tile;
  if (!bad && gamera_cs_decomposition_init(global, dims, ranks,
          rank, &tile) != 0) bad = 1;
  const gamera_cs_earth_startup startup = {
      .inner_radius = 2.5, .density = 5.0, .speed = 4.0,
      .mach = 10.0, .gamma = 5.0 / 3.0,
      .ambient_density = 1.0, .ambient_pressure = 0.001,
      .front_x = -30.0, .front_width = 1.5,
      .cut_radius = 16.0, .cut_length = 8.0,
      .dipole = {{{0.0, 0.0, -6761.730688424097}}}};
  gamera_cs_mi_dipole_config config = {
      .azimuth_points = 192U, .colatitude_points = 22U,
      .current_shell = 1U, .ionosphere_radius_re = 1.0,
      .pedersen_siemens = 5.0, .coupling_interval_s = 10.0,
      .length_norm_m = 6371000.0, .speed_norm_m_s = 100000.0,
      .mass_density_norm_kg_m3 = 1.67262192369e-21,
      .maximum_iterations = 4000, .relative_tolerance = 1e-8};
  if (!bad && argc == 5) {
    const int shell = atoi(argv[4]);
    if (shell < 1 || shell + 1 >= (int)global[0]) bad = 1;
    else config.current_shell = (size_t)shell;
  }
  gamera_no_radial_map map;
  if (!bad && gamera_no_radial_map_init(&map,
          GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
          2.5, 200.0, 3.0) != 0) bad = 1;
  gamera_no_grid local_grid = {0};
  gamera_no_storage local_storage = {0};
  gamera_no_background_data local_background = {0};
  if (!bad && make_face(tile.face, global, tile.global_start,
          tile.local_active, ghost, &map, &startup, &local_grid,
          &local_storage, &local_background) != 0) bad = 1;
  gamera_cs_mi_dipole_mpi *distributed = NULL;
  int any_bad = 0;
  MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (!any_bad && gamera_cs_mi_dipole_mpi_create(&tile, ghost,
          &map, &startup, &config, MPI_COMM_WORLD, &distributed) != 0)
    bad = 1;
  MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (!any_bad && gamera_cs_mi_dipole_mpi_maybe_update(
          distributed, &local_storage, 0.0) != 1) bad = 1;
  MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  gamera_no_grid full_grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage full_storage[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_background_data full_background[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_background_field full_field[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_cs_mi_dipole *oracle = NULL;
  if (!any_bad && rank == 0) {
    for (int face = 0; face < GAMERA_CS_FACE_COUNT && !bad; ++face) {
      if (make_face(face, global, start0, global, ghost, &map,
              &startup, &full_grid[face], &full_storage[face],
              &full_background[face]) != 0) bad = 1;
      full_field[face] = full_background[face].field;
    }
    if (!bad && (gamera_cs_mi_dipole_create(full_grid, full_storage,
            full_field, &startup, global, ghost, &config, &oracle) != 0 ||
        gamera_cs_mi_dipole_maybe_update(oracle, 0.0) != 1)) bad = 1;
  }
  const size_t count = 2U * config.azimuth_points * config.colatitude_points;
  double *test_potential = calloc(count, sizeof(double));
  double *oracle_potential = rank == 0 ? calloc(count, sizeof(double)) : NULL;
  double test_epoch = 0.0, test_next = 0.0;
  if (!any_bad && (test_potential == NULL || (rank == 0 &&
          oracle_potential == NULL) ||
      gamera_cs_mi_dipole_export_potential(
          gamera_cs_mi_dipole_mpi_state(distributed), test_potential,
          count, &test_epoch, &test_next) != 0)) bad = 1;
  if (!bad && rank == 0) {
    double epoch = 0.0, next = 0.0;
    if (gamera_cs_mi_dipole_export_potential(oracle, oracle_potential,
            count, &epoch, &next) != 0 ||
        memcmp(test_potential, oracle_potential,
            count * sizeof(double)) != 0 ||
        test_epoch != epoch || test_next != next) bad = 1;
  }
  MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0 && any_bad)
    fputs("distributed three-layer MI != six-face oracle\n", stderr);
  free(test_potential);
  free(oracle_potential);
  gamera_cs_mi_dipole_destroy(oracle);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_background_destroy(&full_background[face]);
    gamera_no_storage_destroy(&full_storage[face]);
    gamera_no_grid_destroy(&full_grid[face]);
  }
  gamera_cs_mi_dipole_mpi_destroy(distributed);
  gamera_no_background_destroy(&local_background);
  gamera_no_storage_destroy(&local_storage);
  gamera_no_grid_destroy(&local_grid);
  if (rank == 0 && !any_bad)
    puts("distributed three-layer MI vs six-face oracle: BITWISE PASS");
  MPI_Finalize();
  return any_bad ? 1 : 0;
}
