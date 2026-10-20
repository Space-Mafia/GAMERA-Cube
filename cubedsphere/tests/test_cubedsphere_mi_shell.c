#include "cubedsphere_earth_startup.h"
#include "cubedsphere_mesh.h"
#include "cubedsphere_mi_dipole.h"
#include "nonorthogonal_background.h"
#include "nonorthogonal_grid.h"
#include "nonorthogonal_radial_map.h"
#include "nonorthogonal_storage.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The unchanged MI curl needs three active radial cells around shell 1. */
static int make_face(int face, const size_t global[3],
                     const size_t local[3], size_t ghost,
                     const gamera_no_radial_map *map,
                     const gamera_cs_earth_startup *startup,
                     gamera_no_grid *grid, gamera_no_storage *storage,
                     gamera_no_background_data *background) {
  const size_t extent[3] = {local[0] + 2U * ghost,
      local[1] + 2U * ghost, local[2] + 2U * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1U,
      extent[1] + 1U, extent[2] + 1U};
  gamera_no_vec3 *vertices = calloc(
      gamera_no_element_count3(vertex_extent), sizeof(*vertices));
  const size_t start[3] = {0, 0, 0};
  int ok = vertices != NULL;
  if (ok && gamera_cs_generate_vertices_radial_map_tile(face, global,
          start, local, ghost, map, vertices) != 0) {
    fprintf(stderr, "vertex generation failed radial=%zu\n", local[0]);
    ok = 0;
  }
  if (ok && gamera_no_grid_create(extent, vertices, grid) != 0) {
    fprintf(stderr, "grid creation failed radial=%zu\n", local[0]);
    ok = 0;
  }
  if (ok && gamera_cs_calibrate_face_frames(grid) != 0) {
    fprintf(stderr, "frame calibration failed radial=%zu\n", local[0]);
    ok = 0;
  }
  if (ok && gamera_no_storage_create(extent, storage) != 0) {
    fprintf(stderr, "storage creation failed radial=%zu\n", local[0]);
    ok = 0;
  }
  if (ok && gamera_no_background_create(grid,
          gamera_cs_earth_startup_background, (void *)startup,
          background) != 0) {
    fprintf(stderr, "background creation failed radial=%zu\n", local[0]);
    ok = 0;
  }
  free(vertices);
  if (!ok) return -1;
  for (size_t i = ghost; i < ghost + local[0]; ++i)
    for (size_t j = ghost; j < ghost + local[1]; ++j)
      for (size_t k = ghost; k < ghost + local[2]; ++k) {
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
  return 0;
}

int main(void) {
  const size_t global[3] = {16U, 16U, 16U};
  const size_t shell[3] = {3U, 16U, 16U};
  const size_t ghost = 4U;
  const gamera_cs_earth_startup startup = {
      .inner_radius = 2.5, .density = 5.0, .speed = 4.0,
      .mach = 10.0, .gamma = 5.0 / 3.0,
      .ambient_density = 1.0, .ambient_pressure = 0.001,
      .front_x = -30.0, .front_width = 1.5,
      .cut_radius = 16.0, .cut_length = 8.0,
      .dipole = {{{0.0, 0.0, -6761.730688424097}}}};
  const gamera_cs_mi_dipole_config config = {
      .azimuth_points = 192U, .colatitude_points = 22U,
      .current_shell = 1U, .ionosphere_radius_re = 1.0,
      .pedersen_siemens = 5.0, .coupling_interval_s = 10.0,
      .length_norm_m = 6371000.0, .speed_norm_m_s = 100000.0,
      .mass_density_norm_kg_m3 = 1.67262192369e-21,
      .maximum_iterations = 4000, .relative_tolerance = 1e-8};
  gamera_no_radial_map map;
  gamera_no_grid full_grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_grid shell_grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage full_storage[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage shell_storage[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_background_data full_background[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_background_data shell_background[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_background_field full_field[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_background_field shell_field[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_cs_mi_dipole *full_mi = NULL, *shell_mi = NULL, *view = NULL;
  int failed = gamera_no_radial_map_init(&map,
      GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
      2.5, 200.0, 3.0) != 0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT && !failed; ++face) {
    if (make_face(face, global, global, ghost, &map, &startup,
            &full_grid[face], &full_storage[face],
            &full_background[face]) != 0 ||
        make_face(face, global, shell, ghost, &map, &startup,
            &shell_grid[face], &shell_storage[face],
            &shell_background[face]) != 0) {
      fprintf(stderr, "MI test face setup failed: %d\n", face);
      failed = 1;
    }
    full_field[face] = full_background[face].field;
    shell_field[face] = shell_background[face].field;
  }
  if (!failed && gamera_cs_mi_dipole_create(full_grid, full_storage,
          full_field, &startup, global, ghost, &config, &full_mi) != 0) {
    fputs("full MI create failed\n", stderr); failed = 1;
  }
  if (!failed && gamera_cs_mi_dipole_create(shell_grid, shell_storage,
          shell_field, &startup, shell, ghost, &config, &shell_mi) != 0) {
    fputs("shell MI create failed\n", stderr); failed = 1;
  }
  if (!failed && gamera_cs_mi_dipole_create_potential_view(
          &startup, &config, &view) != 0) {
    fputs("MI view create failed\n", stderr); failed = 1;
  }
  if (!failed && gamera_cs_mi_dipole_maybe_update(full_mi, 0.0) != 1) {
    fputs("full MI solve failed\n", stderr); failed = 1;
  }
  if (!failed && gamera_cs_mi_dipole_maybe_update(shell_mi, 0.0) != 1) {
    fputs("shell MI solve failed\n", stderr); failed = 1;
  }
  const size_t count = 2U * config.azimuth_points * config.colatitude_points;
  double *full_potential = calloc(count, sizeof(double));
  double *shell_potential = calloc(count, sizeof(double));
  double full_epoch = 0.0, full_next = 0.0;
  double shell_epoch = 0.0, shell_next = 0.0;
  if (!failed && (full_potential == NULL || shell_potential == NULL ||
      gamera_cs_mi_dipole_export_potential(full_mi, full_potential,
          count, &full_epoch, &full_next) != 0 ||
      gamera_cs_mi_dipole_export_potential(shell_mi, shell_potential,
          count, &shell_epoch, &shell_next) != 0)) {
    fputs("MI potential export failed\n", stderr); failed = 1;
  }
  if (!failed && (memcmp(full_potential, shell_potential,
          count * sizeof(double)) != 0 ||
      full_epoch != shell_epoch || full_next != shell_next)) {
    fputs("MI potential mismatch\n", stderr); failed = 1;
  }
  if (!failed && gamera_cs_mi_dipole_import_potential(
          view, shell_potential, count, shell_epoch, shell_next) != 0) {
    fputs("MI potential import failed\n", stderr); failed = 1;
  }
  if (!failed) {
    double fa, fb, pa, pb, sf_a, sf_b, sp_a, sp_b;
    gamera_mi_solver_stats full_solve[2], shell_solve[2];
    if (gamera_cs_mi_dipole_stats(full_mi, &fa, &fb, &pa, &pb,
            full_solve) != 0 ||
        gamera_cs_mi_dipole_stats(shell_mi, &sf_a, &sf_b,
            &sp_a, &sp_b, shell_solve) != 0 ||
        fa != sf_a || fb != sf_b || pa != sp_a || pb != sp_b ||
        fmax(fabs(fa), fabs(fb)) <= 0.0 ||
        fmax(fabs(pa), fabs(pb)) <= 0.0) {
      fputs("nonzero MI FAC/potential statistics mismatch\n", stderr);
      failed = 1;
    }
  }
  if (!failed) {
    const double theta = 0.2;
    for (int hemi = -1; hemi <= 1; hemi += 2)
      for (int q = 0; q < 4; ++q) {
        const double phi = (double)q * 0.5 * acos(-1.0);
        const gamera_no_vec3 point = {{
            2.5 * sin(theta) * cos(phi),
            2.5 * sin(theta) * sin(phi),
            (double)hemi * 2.5 * cos(theta)}};
        double a = 0.0, b = 0.0;
        if (gamera_cs_mi_dipole_potential(full_mi, point, &a) != 0 ||
            gamera_cs_mi_dipole_potential(view, point, &b) != 0 ||
            a != b) {
          fprintf(stderr, "MI view potential mismatch hemi=%d q=%d "
                  "full=%.17g view=%.17g\n", hemi, q, a, b);
          failed = 1;
        }
      }
  }
  free(full_potential);
  free(shell_potential);
  gamera_cs_mi_dipole_destroy(full_mi);
  gamera_cs_mi_dipole_destroy(shell_mi);
  gamera_cs_mi_dipole_destroy(view);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_background_destroy(&full_background[face]);
    gamera_no_background_destroy(&shell_background[face]);
    gamera_no_storage_destroy(&full_storage[face]);
    gamera_no_storage_destroy(&shell_storage[face]);
    gamera_no_grid_destroy(&full_grid[face]);
    gamera_no_grid_destroy(&shell_grid[face]);
  }
  if (failed) {
    fprintf(stderr, "full-face vs three-layer MI shell parity failed\n");
    return 1;
  }
  puts("full-face vs three-layer MI shell and potential view: BITWISE PASS");
  return 0;
}
