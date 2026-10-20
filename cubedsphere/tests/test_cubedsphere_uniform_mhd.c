#include "cubedsphere_mesh.h"
#include "nonorthogonal_advance.h"
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
  const size_t lower[3] = {ghost, ghost, ghost};
  const size_t upper[3] = {ghost + active[0], ghost + active[1],
                           ghost + active[2]};
  const gamera_no_primitive primitive = {1.2, {{0.23, -0.17, 0.09}}, 0.74};
  const gamera_no_vec3 magnetic = {{0.38, -0.21, 0.16}};
  double conserved[GAMERA_NO_FLUX_COUNT];
  if (gamera_no_primitive_to_conserved(&primitive, 5.0 / 3.0, 1e-12, 1e-12,
                                       conserved) != 0) {
    return EXIT_FAILURE;
  }
  const gamera_no_advance_options options = {
      .stress = {5.0 / 3.0, 1e-12, 1e-12, 1.0, 0.0, 0.0, 100.0,
                 true, false, false, false},
      .emf = {1e-12, 1.0, 0.0, 100.0, 0.8, 0.01, false, false},
      .update = {5.0 / 3.0, 1e-12, 1e-12, 100.0, true, false, false},
      .source_time = 0.0};
  double worst_state = 0.0;
  double worst_divergence = 0.0;
  double worst_recovery = 0.0;
  int worst_face = -1;
  int worst_component = -1;
  size_t worst_i = 0, worst_j = 0, worst_k = 0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(vertex_extent),
                                      sizeof(*vertices));
    gamera_no_grid grid = {0};
    gamera_no_storage storage = {0};
    if (vertices == NULL || gamera_cs_generate_vertices(
            face, active, ghost, 2.0, 2.5, vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid) != 0 ||
        gamera_cs_calibrate_face_frames(&grid) != 0 ||
        gamera_no_storage_create(extent, &storage) != 0) {
      fprintf(stderr, "FAIL face %d setup\n", face);
      return EXIT_FAILURE;
    }
    free(vertices);
    const size_t cells = gamera_no_element_count3(extent);
    for (size_t c = 0; c < cells; ++c) {
      memcpy(&storage.conserved[c * GAMERA_NO_FLUX_COUNT], conserved,
             sizeof(conserved));
      memcpy(&storage.old_conserved[c * GAMERA_NO_FLUX_COUNT], conserved,
             sizeof(conserved));
      storage.cell_magnetic[c] = magnetic;
      storage.old_cell_magnetic[c] = magnetic;
    }
    for (int d = 0; d < GAMERA_NO_DIM; ++d) {
      const size_t count = gamera_no_element_count3(grid.face[d].extent);
      for (size_t f = 0; f < count; ++f) {
        const gamera_no_vec3 area = grid.face[d].value[f].area_vector;
        const double flux = magnetic.value[0] * area.value[0] +
                            magnetic.value[1] * area.value[1] +
                            magnetic.value[2] * area.value[2];
        storage.face_flux[d][f] = flux;
        storage.old_face_flux[d][f] = flux;
      }
    }
    const double *fluxes[3] = {storage.face_flux[0], storage.face_flux[1],
                               storage.face_flux[2]};
    if (gamera_no_recover_magnetic_field(&grid, fluxes,
                                         storage.cell_magnetic) != 0) {
      fprintf(stderr, "FAIL face %d magnetic recovery\n", face);
      return EXIT_FAILURE;
    }
    for (size_t c = 0; c < cells; ++c) {
      for (int d = 0; d < 3; ++d) {
        const double error = fabs(storage.cell_magnetic[c].value[d] -
                                  magnetic.value[d]);
        if (error > worst_recovery) worst_recovery = error;
      }
    }
    if (gamera_no_advance(&storage, &grid, lower, upper, 0.5, 0.01,
                          &options, NULL) != 0) {
      fprintf(stderr, "FAIL face %d GAMERA MHD advance\n", face);
      return EXIT_FAILURE;
    }
    for (size_t i = lower[0]; i < upper[0]; ++i) {
      for (size_t j = lower[1]; j < upper[1]; ++j) {
        for (size_t k = lower[2]; k < upper[2]; ++k) {
          const size_t c = gamera_no_index3(extent, i, j, k);
          for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
            const double error = fabs(storage.conserved[
                c * GAMERA_NO_FLUX_COUNT + (size_t)m] - conserved[m]);
            if (error > worst_state) {
              worst_state = error;
              worst_face = face;
              worst_component = m;
              worst_i = i;
              worst_j = j;
              worst_k = k;
            }
          }
          const double divergence = fabs(gamera_no_cell_net_flux(
              &storage, i, j, k)) / grid.cell[c].volume;
          if (divergence > worst_divergence) worst_divergence = divergence;
        }
      }
    }
    gamera_no_storage_destroy(&storage);
    gamera_no_grid_destroy(&grid);
  }
  printf("six-face uniform MHD: max conserved error %.6e; "
         "max divB %.6e\n", worst_state, worst_divergence);
  printf("worst face=%d component=%d cell=(%zu,%zu,%zu)\n",
         worst_face, worst_component, worst_i, worst_j, worst_k);
  printf("max B recovery error %.6e\n", worst_recovery);
  if (!isfinite(worst_state) || !isfinite(worst_divergence) ||
      worst_state > 1e-8 || worst_divergence > 1e-10) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
