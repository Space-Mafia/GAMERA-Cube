#include "cubedsphere_step.h"
#include "nonorthogonal_initialization.h"
#include "nonorthogonal_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DEFAULT_STEPS = 20 };
static const double amplitude = 0.05;
static const double wavenumber = 0.7;

static gamera_no_primitive wave_at(gamera_no_vec3 point, double time) {
  const double phase = wavenumber * (point.value[2] - time);
  return (gamera_no_primitive){
      1.0, {{-amplitude * cos(phase), -amplitude * sin(phase), 0.0}}, 0.8};
}

static gamera_no_vec3 magnetic_at(gamera_no_vec3 point, double time) {
  const double phase = wavenumber * (point.value[2] - time);
  return (gamera_no_vec3){{amplitude * cos(phase),
                            amplitude * sin(phase), 1.0}};
}

static int wave_primitive(gamera_no_vec3 point, void *context,
                          gamera_no_primitive *primitive) {
  (void)context;
  *primitive = wave_at(point, 0.0);
  return 0;
}

static int analytic_radial_ghosts(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, double current_time,
    double previous_time, void *context) {
  (void)context;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    const size_t *extent = grid[face].cell_extent;
    for (size_t i = 0; i < extent[0]; ++i) {
      if (i >= ghost && i < ghost + active[0]) continue;
      for (size_t j = 0; j < extent[1]; ++j) {
        for (size_t k = 0; k < extent[2]; ++k) {
          const size_t c = gamera_no_index3(extent, i, j, k);
          const gamera_no_vec3 point = grid[face].cell[c].centroid;
          const gamera_no_primitive current = wave_at(point, current_time);
          const gamera_no_primitive previous = wave_at(point, previous_time);
          if (gamera_no_primitive_to_conserved(&current, 5.0 / 3.0,
                  1e-12, 1e-12,
                  &storage[face].conserved[c * GAMERA_NO_FLUX_COUNT]) != 0 ||
              gamera_no_primitive_to_conserved(&previous, 5.0 / 3.0,
                  1e-12, 1e-12,
                  &storage[face].old_conserved[c * GAMERA_NO_FLUX_COUNT]) != 0)
            return -1;
          storage[face].cell_magnetic[c] =
              magnetic_at(point, current_time);
          storage[face].old_cell_magnetic[c] =
              magnetic_at(point, previous_time);
        }
      }
    }
    for (int direction = 0; direction < 3; ++direction) {
      const size_t *face_extent = grid[face].face[direction].extent;
      for (size_t i = 0; i < face_extent[0]; ++i) {
        const int radial_ghost = direction == 0
            ? i < ghost || i > ghost + active[0]
            : i < ghost || i >= ghost + active[0];
        if (!radial_ghost) continue;
        for (size_t j = 0; j < face_extent[1]; ++j) {
          for (size_t k = 0; k < face_extent[2]; ++k) {
            const size_t f = gamera_no_index3(face_extent, i, j, k);
            const gamera_no_face_geometry geometry =
                grid[face].face[direction].value[f];
            const gamera_no_vec3 current = magnetic_at(
                geometry.centroid, current_time);
            const gamera_no_vec3 previous = magnetic_at(
                geometry.centroid, previous_time);
            double current_flux = 0.0, previous_flux = 0.0;
            for (int d = 0; d < 3; ++d) {
              current_flux += current.value[d] * geometry.area_vector.value[d];
              previous_flux += previous.value[d] * geometry.area_vector.value[d];
            }
            storage[face].face_flux[direction][f] = current_flux;
            storage[face].old_face_flux[direction][f] = previous_flux;
          }
        }
      }
    }
  }
  return 0;
}

static int wave_potential(gamera_no_vec3 point, void *context,
                           gamera_no_vec3 *potential) {
  (void)context;
  const double phase = wavenumber * point.value[2];
  *potential = (gamera_no_vec3){{
      -amplitude * cos(phase) / wavenumber,
      point.value[0] - amplitude * sin(phase) / wavenumber,
      0.0}};
  return 0;
}

int main(int argc, char **argv) {
  const size_t angular_cells = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 9;
  const int steps = argc > 2 ? atoi(argv[2]) : DEFAULT_STEPS;
  const double dt = argc > 3 ? strtod(argv[3], NULL) : 0.002;
  const int analytic_boundary = argc > 4 && strcmp(argv[4], "analytic") == 0;
  if (angular_cells < 9 || steps < 1 || !isfinite(dt) || dt <= 0.0) {
    fprintf(stderr, "usage: %s [angular_cells>=9] [steps>=1] [dt>0]\n", argv[0]);
    return EXIT_FAILURE;
  }
  const size_t ghost = 4;
  const size_t active[3] = {2 * angular_cells / 3, angular_cells,
                            angular_cells};
  const size_t extent[3] = {active[0] + 2 * ghost,
                            active[1] + 2 * ghost,
                            active[2] + 2 * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1,
                                   extent[1] + 1, extent[2] + 1};
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage storage[GAMERA_CS_FACE_COUNT] = {{0}};
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(vertex_extent),
                                      sizeof(*vertices));
    if (vertices == NULL || gamera_cs_generate_vertices(
            face, active, ghost, 2.0, 2.5, vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid[face]) != 0 ||
        gamera_cs_calibrate_face_frames(&grid[face]) != 0 ||
        gamera_no_storage_create(extent, &storage[face]) != 0 ||
        gamera_no_initialize_primitives(&grid[face], &storage[face],
            wave_primitive, NULL, false, 5.0 / 3.0, 1e-12, 1e-12) != 0 ||
        gamera_no_initialize_flux_from_vector_potential(
            &grid[face], &storage[face], wave_potential, NULL) != 0) {
      fprintf(stderr, "FAIL Alfvén setup face %d\n", face);
      return EXIT_FAILURE;
    }
    free(vertices);
    const size_t cells = gamera_no_element_count3(extent);
    memcpy(storage[face].old_conserved, storage[face].conserved,
           cells * GAMERA_NO_FLUX_COUNT * sizeof(double));
    memcpy(storage[face].old_cell_magnetic, storage[face].cell_magnetic,
           cells * sizeof(gamera_no_vec3));
    for (int d = 0; d < 3; ++d) {
      const size_t count = gamera_no_element_count3(
          grid[face].face[d].extent);
      memcpy(storage[face].old_face_flux[d], storage[face].face_flux[d],
             count * sizeof(double));
    }
  }
  const gamera_no_advance_options options = {
      .stress = {5.0 / 3.0, 1e-12, 1e-12, 1.0, 0.0, 0.0, 100.0,
                 true, false, false, false},
      .emf = {1e-12, 1.0, 0.01, 100.0, 0.8, dt, false, false},
      .update = {5.0 / 3.0, 1e-12, 1e-12, 100.0, true, false, false},
      .source_time = 0.0};
  for (int step = 0; step < steps; ++step) {
    const double current_time = (double)step * dt;
    const double previous_time = step == 0 ? 0.0 : (double)(step - 1) * dt;
    if (gamera_cs_advance_six_with_boundary(grid, storage, active, ghost,
                              0.5, dt, &options,
                              analytic_boundary ? analytic_radial_ghosts : NULL,
                              current_time, previous_time, NULL) != 0) {
      fprintf(stderr, "FAIL Alfvén MHD step %d\n", step);
      return EXIT_FAILURE;
    }
  }
  double squared_error = 0.0;
  double interior_squared_error = 0.0;
  double maximum_divergence = 0.0;
  double maximum_change = 0.0;
  size_t sample_count = 0;
  size_t interior_sample_count = 0;
  const double time = (double)steps * dt;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t c = gamera_no_index3(extent, i, j, k);
          const gamera_no_vec3 point = grid[face].cell[c].centroid;
          gamera_no_primitive p;
          if (gamera_no_conserved_to_primitive(
                  &storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
                  5.0 / 3.0, 1e-12, 1e-12, &p) != 0) return EXIT_FAILURE;
          const double phase = wavenumber * (point.value[2] - time);
          const double exact_vx = -amplitude * cos(phase);
          const double exact_vy = -amplitude * sin(phase);
          const double cell_squared_error =
                           (p.velocity.value[0] - exact_vx) *
                           (p.velocity.value[0] - exact_vx) +
                           (p.velocity.value[1] - exact_vy) *
                           (p.velocity.value[1] - exact_vy) +
                           (storage[face].cell_magnetic[c].value[0] +
                            exact_vx) *
                           (storage[face].cell_magnetic[c].value[0] +
                            exact_vx) +
                           (storage[face].cell_magnetic[c].value[1] +
                            exact_vy) *
                            (storage[face].cell_magnetic[c].value[1] +
                            exact_vy);
          squared_error += cell_squared_error;
          if (i >= ghost + 2 && i + 2 < ghost + active[0]) {
            interior_squared_error += cell_squared_error;
            ++interior_sample_count;
          }
          maximum_change = fmax(maximum_change,
              fabs(p.velocity.value[0] +
                   amplitude * cos(wavenumber * point.value[2])));
          const double div = fabs(gamera_no_cell_net_flux(
              &storage[face], i, j, k)) / grid[face].cell[c].volume;
          maximum_divergence = fmax(maximum_divergence, div);
          ++sample_count;
        }
      }
    }
  }
  const double rms = sqrt(squared_error / (4.0 * (double)sample_count));
  const double interior_rms = interior_sample_count > 0
      ? sqrt(interior_squared_error / (4.0 * (double)interior_sample_count))
      : NAN;
  printf("six-face circular Alfvén wave: n=%zu nr=%zu steps=%d dt=%.6e "
         "radial=%s "
         "RMS %.6e, interior RMS %.6e, max change %.6e, max divB %.6e\n",
         angular_cells, active[0], steps, dt,
         analytic_boundary ? "analytic" : "outflow", rms, interior_rms,
         maximum_change, maximum_divergence);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_storage_destroy(&storage[face]);
    gamera_no_grid_destroy(&grid[face]);
  }
  if (!isfinite(rms) || !isfinite(maximum_change) ||
      !isfinite(maximum_divergence) || rms > 0.03 ||
      maximum_change < 2.5e-4 * time || maximum_divergence > 1e-9)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
