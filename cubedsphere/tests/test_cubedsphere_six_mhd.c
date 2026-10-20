#include "cubedsphere_step.h"
#include "nonorthogonal_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  const size_t ghost = 4;
  const size_t active[3] = {5, 9, 9};
  const size_t extent[3] = {active[0] + 2 * ghost,
                            active[1] + 2 * ghost,
                            active[2] + 2 * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1, extent[1] + 1,
                                   extent[2] + 1};
  const gamera_no_primitive primitive = {1.2, {{0.23, -0.17, 0.09}}, 0.74};
  const gamera_no_vec3 magnetic = {{0.38, -0.21, 0.16}};
  double expected[GAMERA_NO_FLUX_COUNT];
  if (gamera_no_primitive_to_conserved(&primitive, 5.0 / 3.0,
                                       1e-12, 1e-12, expected) != 0)
    return EXIT_FAILURE;
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage storage[GAMERA_CS_FACE_COUNT] = {{0}};
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(vertex_extent),
                                      sizeof(*vertices));
    if (vertices == NULL || gamera_cs_generate_vertices(
            face, active, ghost, 2.0, 2.5, vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid[face]) != 0 ||
        gamera_cs_calibrate_face_frames(&grid[face]) != 0 ||
        gamera_no_storage_create(extent, &storage[face]) != 0) {
      fprintf(stderr, "FAIL face %d setup\n", face);
      return EXIT_FAILURE;
    }
    free(vertices);
    const size_t cells = gamera_no_element_count3(extent);
    for (size_t c = 0; c < cells; ++c) {
      memcpy(&storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
             expected, sizeof(expected));
      memcpy(&storage[face].old_conserved[c * GAMERA_NO_FLUX_COUNT],
             expected, sizeof(expected));
      storage[face].cell_magnetic[c] = magnetic;
      storage[face].old_cell_magnetic[c] = magnetic;
    }
    for (int d = 0; d < 3; ++d) {
      const size_t count = gamera_no_element_count3(grid[face].face[d].extent);
      for (size_t f = 0; f < count; ++f) {
        const gamera_no_vec3 a = grid[face].face[d].value[f].area_vector;
        double flux = 0.0;
        for (int m = 0; m < 3; ++m) flux += magnetic.value[m] * a.value[m];
        storage[face].face_flux[d][f] = flux;
        storage[face].old_face_flux[d][f] = flux;
      }
    }
  }
  const gamera_no_advance_options options = {
      .stress = {5.0 / 3.0, 1e-12, 1e-12, 1.0, 0.0, 0.0, 100.0,
                 true, false, false, false},
      .emf = {1e-12, 1.0, 0.0, 100.0, 0.8, 0.01, false, false},
      .update = {5.0 / 3.0, 1e-12, 1e-12, 100.0, true, false, false},
      .source_time = 0.0};
  for (int step = 0; step < 3; ++step) {
    if (gamera_cs_advance_six(grid, storage, active, ghost,
                              0.5, 0.01, &options) != 0) {
      fprintf(stderr, "FAIL six-block MHD step %d\n", step);
      return EXIT_FAILURE;
    }
  }
  double max_state = 0.0, max_divergence = 0.0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t c = gamera_no_index3(extent, i, j, k);
          for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
            const double error = fabs(storage[face].conserved[
                c * GAMERA_NO_FLUX_COUNT + (size_t)m] - expected[m]);
            if (error > max_state) max_state = error;
          }
          const double div = fabs(gamera_no_cell_net_flux(
              &storage[face], i, j, k)) / grid[face].cell[c].volume;
          if (div > max_divergence) max_divergence = div;
        }
      }
    }
  }
  printf("six-block three-step MHD: max state error %.6e, divB %.6e\n",
         max_state, max_divergence);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_storage_destroy(&storage[face]);
    gamera_no_grid_destroy(&grid[face]);
  }
  if (!isfinite(max_state) || !isfinite(max_divergence) ||
      max_state > 5e-6 || max_divergence > 1e-10) return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
