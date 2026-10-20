#include "cubedsphere_step.h"
#include "nonorthogonal_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  const size_t ghost = 4;
  const size_t active[3] = {6, 9, 9};
  const size_t extent[3] = {active[0] + 2 * ghost,
                            active[1] + 2 * ghost,
                            active[2] + 2 * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1,
                                   extent[1] + 1, extent[2] + 1};
  const gamera_no_vec3 magnetic = {{0.2, 0.2, 0.0}};
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage storage[GAMERA_CS_FACE_COUNT] = {{0}};
  const double center = 2.25 / sqrt(2.0);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(vertex_extent),
                                      sizeof(*vertices));
    if (vertices == NULL || gamera_cs_generate_vertices(
            face, active, ghost, 2.0, 2.5, vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid[face]) != 0 ||
        gamera_cs_calibrate_face_frames(&grid[face]) != 0 ||
        gamera_no_storage_create(extent, &storage[face]) != 0) {
      fprintf(stderr, "FAIL blast setup face %d\n", face);
      return EXIT_FAILURE;
    }
    free(vertices);
    const size_t cells = gamera_no_element_count3(extent);
    for (size_t c = 0; c < cells; ++c) {
      const gamera_no_vec3 p = grid[face].cell[c].centroid;
      const double distance_square =
          (p.value[0] - center) * (p.value[0] - center) +
          (p.value[1] - center) * (p.value[1] - center) +
          p.value[2] * p.value[2];
      const gamera_no_primitive primitive = {
          1.0, {{0.0, 0.0, 0.0}},
          0.8 + 0.6 * exp(-distance_square / 0.12)};
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
        const double flux = magnetic.value[0] * area.value[0] +
                            magnetic.value[1] * area.value[1];
        storage[face].face_flux[d][f] = flux;
        storage[face].old_face_flux[d][f] = flux;
      }
    }
  }
  const gamera_no_advance_options options = {
      .stress = {5.0 / 3.0, 1e-12, 1e-12, 1.0, 0.25, 0.25, 100.0,
                 true, true, false, false},
      .emf = {1e-12, 1.0, 0.5, 100.0, 0.3, 0.002, false, false},
      .update = {5.0 / 3.0, 1e-12, 1e-12, 100.0, true, false, false},
      .source_time = 0.0};
  for (int step = 0; step < 20; ++step) {
    if (gamera_cs_advance_six(grid, storage, active, ghost,
                              0.5, 0.002, &options) != 0) {
      fprintf(stderr, "FAIL blast MHD step %d\n", step);
      return EXIT_FAILURE;
    }
  }
  double max_divergence = 0.0;
  double min_density = INFINITY, min_pressure = INFINITY;
  double max_symmetry = 0.0, max_pressure = 0.0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t c = gamera_no_index3(extent, i, j, k);
          gamera_no_primitive primitive;
          if (gamera_no_conserved_to_primitive(
                  &storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
                  5.0 / 3.0, 1e-12, 1e-12, &primitive) != 0) {
            fprintf(stderr, "FAIL nonphysical blast cell\n");
            return EXIT_FAILURE;
          }
          min_density = fmin(min_density, primitive.density);
          min_pressure = fmin(min_pressure, primitive.pressure);
          max_pressure = fmax(max_pressure, primitive.pressure);
          const double div = fabs(gamera_no_cell_net_flux(
              &storage[face], i, j, k)) / grid[face].cell[c].volume;
          max_divergence = fmax(max_divergence, div);
          if (face == GAMERA_CS_POS_X) {
            const size_t mirrored_j =
                ghost + active[1] - 1U - (j - ghost);
            const size_t mirror = gamera_no_index3(
                extent, i, mirrored_j, k);
            gamera_no_primitive other;
            if (gamera_no_conserved_to_primitive(
                    &storage[GAMERA_CS_POS_Y].conserved[
                        mirror * GAMERA_NO_FLUX_COUNT],
                    5.0 / 3.0, 1e-12, 1e-12, &other) != 0)
              return EXIT_FAILURE;
            max_symmetry = fmax(max_symmetry,
                fabs(primitive.pressure - other.pressure));
          }
        }
      }
    }
  }
  printf("seam-centered magnetized blast: min rho %.6e, min p %.6e, "
         "max p %.6e, max divB %.6e, x/y symmetry %.6e\n",
         min_density, min_pressure, max_pressure,
         max_divergence, max_symmetry);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_storage_destroy(&storage[face]);
    gamera_no_grid_destroy(&grid[face]);
  }
  if (!isfinite(min_density) || !isfinite(min_pressure) ||
      !isfinite(max_pressure) || !isfinite(max_divergence) ||
      !isfinite(max_symmetry) || min_density <= 0.0 || min_pressure <= 0.0 ||
      max_pressure < 0.81 || max_divergence > 1e-9 ||
      max_symmetry > 2e-2) return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
