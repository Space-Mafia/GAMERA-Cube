#include "cubedsphere_step.h"
#include "nonorthogonal_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  const int steps = argc > 1 ? atoi(argv[1]) : 30;
  if (steps < 1) {
    fprintf(stderr, "usage: %s [steps>=1]\n", argv[0]);
    return EXIT_FAILURE;
  }
  const size_t ghost = 4;
  const size_t active[3] = {12, 9, 9};
  const size_t extent[3] = {active[0] + 2 * ghost,
                            active[1] + 2 * ghost,
                            active[2] + 2 * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1,
                                   extent[1] + 1, extent[2] + 1};
  const gamera_no_vec3 magnetic = {{0.1, 0.08, -0.03}};
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage storage[GAMERA_CS_FACE_COUNT] = {{0}};
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(vertex_extent),
                                      sizeof(*vertices));
    if (vertices == NULL || gamera_cs_generate_vertices(
            face, active, ghost, 2.0, 3.0, vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid[face]) != 0 ||
        gamera_cs_calibrate_face_frames(&grid[face]) != 0 ||
        gamera_no_storage_create(extent, &storage[face]) != 0) {
      fprintf(stderr, "FAIL shock setup face %d\n", face);
      return EXIT_FAILURE;
    }
    free(vertices);
    const size_t cells = gamera_no_element_count3(extent);
    for (size_t c = 0; c < cells; ++c) {
      const gamera_no_vec3 p = grid[face].cell[c].centroid;
      const double r = sqrt(p.value[0] * p.value[0] +
                            p.value[1] * p.value[1] +
                            p.value[2] * p.value[2]);
      const gamera_no_primitive primitive =
          r < 2.5 ? (gamera_no_primitive){1.0, {{0.0, 0.0, 0.0}}, 1.0} :
                    (gamera_no_primitive){0.125, {{0.0, 0.0, 0.0}}, 0.1};
      double conserved[GAMERA_NO_FLUX_COUNT];
      if (gamera_no_primitive_to_conserved(
              &primitive, 5.0 / 3.0, 1e-12, 1e-12, conserved) != 0)
        return EXIT_FAILURE;
      memcpy(&storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
             conserved, sizeof(conserved));
      memcpy(&storage[face].old_conserved[c * GAMERA_NO_FLUX_COUNT],
             conserved, sizeof(conserved));
      storage[face].cell_magnetic[c] = magnetic;
      storage[face].old_cell_magnetic[c] = magnetic;
    }
    for (int d = 0; d < 3; ++d) {
      const size_t faces = gamera_no_element_count3(grid[face].face[d].extent);
      for (size_t f = 0; f < faces; ++f) {
        const gamera_no_vec3 area = grid[face].face[d].value[f].area_vector;
        double flux = 0.0;
        for (int m = 0; m < 3; ++m) flux += magnetic.value[m] * area.value[m];
        storage[face].face_flux[d][f] = flux;
        storage[face].old_face_flux[d][f] = flux;
      }
    }
  }
  const gamera_no_advance_options options = {
      .stress = {5.0 / 3.0, 1e-12, 1e-12, 1.0, 0.25, 0.25, 100.0,
                 true, true, false, false},
      .emf = {1e-12, 1.0, 0.5, 100.0, 0.3, 0.001, false, false},
      .update = {5.0 / 3.0, 1e-12, 1e-12, 100.0, true, false, false},
      .source_time = 0.0};
  for (int step = 0; step < steps; ++step) {
    if (gamera_cs_advance_six(grid, storage, active, ghost,
                              0.5, 0.001, &options) != 0) {
      fprintf(stderr, "FAIL shock MHD step %d\n", step);
      return EXIT_FAILURE;
    }
  }
  double min_density = INFINITY, min_pressure = INFINITY;
  double max_density = 0.0, max_divergence = 0.0;
  double max_density_change = 0.0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t c = gamera_no_index3(extent, i, j, k);
          gamera_no_primitive p;
          if (gamera_no_conserved_to_primitive(
                  &storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
                  5.0 / 3.0, 1e-12, 1e-12, &p) != 0) return EXIT_FAILURE;
          min_density = fmin(min_density, p.density);
          min_pressure = fmin(min_pressure, p.pressure);
          max_density = fmax(max_density, p.density);
          const gamera_no_vec3 point = grid[face].cell[c].centroid;
          const double radius = sqrt(point.value[0] * point.value[0] +
                                     point.value[1] * point.value[1] +
                                     point.value[2] * point.value[2]);
          max_density_change = fmax(max_density_change,
              fabs(p.density - (radius < 2.5 ? 1.0 : 0.125)));
          const double div = fabs(gamera_no_cell_net_flux(
              &storage[face], i, j, k)) / grid[face].cell[c].volume;
          max_divergence = fmax(max_divergence, div);
        }
      }
    }
  }
  printf("six-face radial MHD shock: steps=%d rho [%.6e, %.6e], min p %.6e, "
         "max density change %.6e, max divB %.6e\n",
         steps, min_density, max_density, min_pressure,
         max_density_change, max_divergence);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_storage_destroy(&storage[face]);
    gamera_no_grid_destroy(&grid[face]);
  }
  if (!isfinite(min_density) || !isfinite(min_pressure) ||
      !isfinite(max_density) || !isfinite(max_divergence) ||
      min_density <= 0.0 || min_pressure <= 0.0 ||
      max_density < 0.9 || max_density_change < 1e-4 ||
      max_divergence > 1e-9)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
