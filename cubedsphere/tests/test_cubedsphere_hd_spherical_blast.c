#include "cubedsphere_step.h"
#include "nonorthogonal_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* An angular pressure blast on a six-face spherical shell. Initial data are
 * independent of radius, so this tests the 2-D spherical propagation in the
 * existing 3-D HD solver. Radial outflow is retained; this is not a strictly
 * constrained 2-D Euler equation. */
static const double gamma_gas = 5.0 / 3.0;
static const double blast_center = 0.57735026918962576451;
static int face_center_case = 0;
static int magnetized_case = 0;
static const gamera_no_vec3 imposed_magnetic = {{0.5, 0.0, 0.5}};

static double initial_pressure(gamera_no_vec3 point) {
  const double radius = sqrt(point.value[0] * point.value[0] +
                             point.value[1] * point.value[1] +
                             point.value[2] * point.value[2]);
  const double cosine = fmax(-1.0, fmin(1.0,
      face_center_case ? point.value[0] / radius :
      blast_center * (point.value[0] + point.value[1] + point.value[2]) /
      radius));
  const double angle = acos(cosine);
  if (face_center_case) {
    /* Exactly compact support on the +X face: the outer angular edge is
     * 0.22 rad, well inside the nearest cube-face seam at pi/4 rad. */
    if (angle <= 0.14) return 20.0;
    if (angle >= 0.22) return 1.0;
    const double s = (angle - 0.14) / 0.08;
    return 1.0 + 19.0 * (1.0 - 3.0 * s * s + 2.0 * s * s * s);
  }
  return 1.0 + 19.0 / (1.0 + exp((angle - 0.18) / 0.025));
}

static int snapshot(FILE *output, double time,
                    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
                    const gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
                    const size_t active[3], size_t ghost) {
  if (output == NULL) return 0;
  const size_t i = ghost + active[0] / 2U;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    const size_t *extent = grid[face].cell_extent;
    for (size_t j = 0; j < active[1]; ++j) {
      for (size_t k = 0; k < active[2]; ++k) {
        const size_t c = gamera_no_index3(extent, i, ghost + j, ghost + k);
        const gamera_no_vec3 point = grid[face].cell[c].centroid;
        gamera_no_primitive primitive;
        if (gamera_no_conserved_to_primitive(
                &storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
                gamma_gas, 1e-12, 1e-12, &primitive) != 0)
          return -1;
        if (fprintf(output, "%.17g,%d,%zu,%zu,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    time, face, j, k, point.value[0], point.value[1],
                    point.value[2], primitive.density,
                    primitive.pressure) < 0) return -1;
      }
    }
  }
  return 0;
}

static int diagnostics(double time,
                       const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
                       const gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
                       const size_t active[3], size_t ghost,
                       double *min_density, double *min_pressure,
                       double *max_radial_speed, double max_change[6],
                       double *mass, double *max_divergence) {
  *min_density = INFINITY;
  *min_pressure = INFINITY;
  *max_radial_speed = 0.0;
  *mass = 0.0;
  *max_divergence = 0.0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    max_change[face] = 0.0;
    const size_t *extent = grid[face].cell_extent;
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t c = gamera_no_index3(extent, i, j, k);
          gamera_no_primitive p;
          if (gamera_no_conserved_to_primitive(
                  &storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
                  gamma_gas, 1e-12, 1e-12, &p) != 0) return -1;
          const gamera_no_vec3 x = grid[face].cell[c].centroid;
          const double radius = sqrt(x.value[0] * x.value[0] +
                                     x.value[1] * x.value[1] +
                                     x.value[2] * x.value[2]);
          double radial_speed = 0.0;
          for (int d = 0; d < 3; ++d)
            radial_speed += p.velocity.value[d] * x.value[d] / radius;
          *min_density = fmin(*min_density, p.density);
          *min_pressure = fmin(*min_pressure, p.pressure);
          *max_radial_speed = fmax(*max_radial_speed, fabs(radial_speed));
          *max_divergence = fmax(*max_divergence,
              fabs(gamera_no_cell_net_flux(&storage[face], i, j, k)) /
              grid[face].cell[c].volume);
          max_change[face] = fmax(max_change[face],
                                  fabs(p.pressure - initial_pressure(x)));
          *mass += p.density * grid[face].cell[c].volume;
        }
      }
    }
  }
  if (!isfinite(*min_density) || !isfinite(*min_pressure) ||
      !isfinite(*max_radial_speed) || !isfinite(*mass) ||
      !isfinite(*max_divergence) ||
      *min_density <= 0.0 || *min_pressure <= 0.0) return -1;
  printf("t=%.5f min_rho=%.8g min_p=%.8g max_vrad=%.8g "
         "max_divB=%.8g mass=%.10g "
         "max_dp_faces=", time, *min_density, *min_pressure,
         *max_radial_speed, *max_divergence, *mass);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    printf("%s%.5g", face == 0 ? "" : ",", max_change[face]);
  putchar('\n');
  return 0;
}

static double pressure_at(const gamera_no_storage *storage, size_t i,
                          size_t j, size_t k) {
  const size_t c = gamera_no_index3(storage->cell_extent, i, j, k);
  gamera_no_primitive p;
  if (gamera_no_conserved_to_primitive(
          &storage->conserved[c * GAMERA_NO_FLUX_COUNT], gamma_gas,
          1e-12, 1e-12, &p) != 0) return NAN;
  return p.pressure;
}

static int seam_metrics(const gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
                        const size_t active[3], size_t ghost,
                        double *max_jump, double *max_gradient_ratio) {
  *max_jump = 0.0;
  *max_gradient_ratio = 0.0;
  const size_t i = ghost + active[0] / 2U;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
      gamera_cs_neighbor neighbor;
      if (gamera_cs_edge_neighbor(face, edge, &neighbor) != 0) return -1;
      if (face >= neighbor.face) continue;
      for (size_t along = 0; along < active[1]; ++along) {
        const size_t mapped = neighbor.reversed ?
            active[1] - 1U - along : along;
        size_t source[3] = {i, ghost + along, ghost + along};
        size_t target[3] = {i, ghost + mapped, ghost + mapped};
        size_t source_inner[3], target_inner[3];
        const int source_axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
        const int target_axis = neighbor.edge < GAMERA_CS_B_LOWER ? 1 : 2;
        source[source_axis] = edge % 2 == 0 ? ghost :
                              ghost + active[source_axis] - 1U;
        target[target_axis] = neighbor.edge % 2 == 0 ? ghost :
                              ghost + active[target_axis] - 1U;
        memcpy(source_inner, source, sizeof(source));
        memcpy(target_inner, target, sizeof(target));
        if (edge % 2 == 0) ++source_inner[source_axis];
        else --source_inner[source_axis];
        if (neighbor.edge % 2 == 0) ++target_inner[target_axis];
        else --target_inner[target_axis];
        const double p_source = pressure_at(&storage[face],
            source[0], source[1], source[2]);
        const double p_target = pressure_at(&storage[neighbor.face],
            target[0], target[1], target[2]);
        const double p_source_inner = pressure_at(&storage[face],
            source_inner[0], source_inner[1], source_inner[2]);
        const double p_target_inner = pressure_at(&storage[neighbor.face],
            target_inner[0], target_inner[1], target_inner[2]);
        if (!isfinite(p_source) || !isfinite(p_target) ||
            !isfinite(p_source_inner) || !isfinite(p_target_inner)) return -1;
        const double jump = fabs(p_source - p_target);
        const double local_gradient = 0.5 *
            (fabs(p_source - p_source_inner) +
             fabs(p_target - p_target_inner));
        *max_jump = fmax(*max_jump, jump);
        if (local_gradient > 1e-3)
          *max_gradient_ratio = fmax(*max_gradient_ratio,
                                     jump / local_gradient);
      }
    }
  }
  return 0;
}

/* On a rejected long MHD run, identify the physical cell that sets CFL.
 * This is diagnostic only and does not alter the advance. */
static void report_minimum_cfl_cell(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    const gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, double cfl) {
  double minimum = INFINITY;
  int minimum_face = -1;
  size_t minimum_i = 0, minimum_j = 0, minimum_k = 0;
  gamera_no_primitive minimum_primitive = {0};
  gamera_no_vec3 minimum_b = {{0}}, minimum_x = {{0}};
  double minimum_length = 0.0, minimum_cs = 0.0, minimum_va = 0.0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t c = gamera_no_index3(grid[face].cell_extent, i, j, k);
          gamera_no_primitive p;
          if (gamera_no_conserved_to_primitive(
                  &storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
                  gamma_gas, 1e-12, 1e-12, &p) != 0) continue;
          const gamera_no_vec3 b = storage[face].cell_magnetic[c];
          const double speed = hypot(p.velocity.value[0],
                                    hypot(p.velocity.value[1],
                                          p.velocity.value[2]));
          const double field = hypot(b.value[0], hypot(b.value[1], b.value[2]));
          double length = grid[face].cell[c].cfl_length[0];
          for (int d = 1; d < 3; ++d)
            length = fmin(length, grid[face].cell[c].cfl_length[d]);
          const double cs = sqrt(gamma_gas * p.pressure / p.density);
          const double va = field / sqrt(p.density);
          const double candidate = cfl * length / (speed + hypot(cs, va));
          if (candidate >= minimum) continue;
          minimum = candidate;
          minimum_face = face;
          minimum_i = i;
          minimum_j = j;
          minimum_k = k;
          minimum_primitive = p;
          minimum_b = b;
          minimum_x = grid[face].cell[c].centroid;
          minimum_length = length;
          minimum_cs = cs;
          minimum_va = va;
        }
      }
    }
  }
  if (minimum_face >= 0)
    fprintf(stderr,
        "minimum CFL cell: face=%d ijk=%zu,%zu,%zu xyz=%.9g,%.9g,%.9g "
        "dt=%.9g length=%.9g rho=%.9g p=%.9g "
        "v=%.9g,%.9g,%.9g B=%.9g,%.9g,%.9g cs=%.9g va=%.9g\n",
        minimum_face, minimum_i, minimum_j, minimum_k,
        minimum_x.value[0], minimum_x.value[1], minimum_x.value[2],
        minimum, minimum_length, minimum_primitive.density,
        minimum_primitive.pressure, minimum_primitive.velocity.value[0],
        minimum_primitive.velocity.value[1], minimum_primitive.velocity.value[2],
        minimum_b.value[0], minimum_b.value[1], minimum_b.value[2],
        minimum_cs, minimum_va);
}

int main(int argc, char **argv) {
  const char *csv_path = argc > 1 && strcmp(argv[1], "-") != 0
      ? argv[1] : NULL;
  const size_t angular = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 16U;
  const double end_time = argc > 3 ? strtod(argv[3], NULL) : 0.3;
  const char *center_mode = argc > 4 ? argv[4] : "corner";
  const double snapshot_cadence = argc > 5 ? strtod(argv[5], NULL) : 0.2;
  if (argc > 6 || (strcmp(center_mode, "corner") != 0 &&
                   strcmp(center_mode, "face-center") != 0 &&
                   strcmp(center_mode, "face-center-mhd") != 0) ||
      !isfinite(snapshot_cadence) || snapshot_cadence <= 0.0 ||
      snapshot_cadence > end_time ||
      angular < 9U || angular > 64U || !isfinite(end_time) ||
      end_time <= 0.0 || end_time > 3.0) return EXIT_FAILURE;
  face_center_case = strcmp(center_mode, "corner") != 0;
  magnetized_case = strcmp(center_mode, "face-center-mhd") == 0;
  const size_t ghost = 4U;
  const size_t active[3] = {8U, angular, angular};
  const size_t extent[3] = {active[0] + 2U * ghost,
                            active[1] + 2U * ghost,
                            active[2] + 2U * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1U,
                                   extent[1] + 1U, extent[2] + 1U};
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage storage[GAMERA_CS_FACE_COUNT] = {{0}};
  FILE *output = NULL;
  if (csv_path != NULL) {
    output = fopen(csv_path, "w");
    if (output == NULL || fprintf(output,
        "time,face,j,k,x,y,z,density,pressure\n") < 0) return EXIT_FAILURE;
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(vertex_extent),
                                      sizeof(*vertices));
    if (vertices == NULL ||
        gamera_cs_generate_vertices(face, active, ghost,
                                     2.0, 2.8, vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid[face]) != 0 ||
        gamera_cs_calibrate_face_frames(&grid[face]) != 0 ||
        gamera_no_storage_create(extent, &storage[face]) != 0) {
      fprintf(stderr, "FAIL spherical blast setup face %d\n", face);
      free(vertices);
      return EXIT_FAILURE;
    }
    free(vertices);
    const size_t cells = gamera_no_element_count3(extent);
    for (size_t c = 0; c < cells; ++c) {
      const gamera_no_primitive p = {1.0, {{0.0, 0.0, 0.0}},
                                      initial_pressure(grid[face].cell[c].centroid)};
      double state[GAMERA_NO_FLUX_COUNT];
      if (gamera_no_primitive_to_conserved(&p, gamma_gas,
             1e-12, 1e-12, state) != 0) return EXIT_FAILURE;
      memcpy(&storage[face].conserved[c * GAMERA_NO_FLUX_COUNT],
             state, sizeof(state));
      memcpy(&storage[face].old_conserved[c * GAMERA_NO_FLUX_COUNT],
             state, sizeof(state));
      if (magnetized_case) {
        storage[face].cell_magnetic[c] = imposed_magnetic;
        storage[face].old_cell_magnetic[c] = imposed_magnetic;
      }
    }
    if (magnetized_case) for (int direction = 0; direction < 3; ++direction) {
      const size_t faces = gamera_no_element_count3(
          grid[face].face[direction].extent);
      for (size_t f = 0; f < faces; ++f) {
        const gamera_no_vec3 area =
            grid[face].face[direction].value[f].area_vector;
        double flux = 0.0;
        for (int component = 0; component < 3; ++component)
          flux += imposed_magnetic.value[component] * area.value[component];
        storage[face].face_flux[direction][f] = flux;
        storage[face].old_face_flux[direction][f] = flux;
      }
    }
  }
  const gamera_no_advance_options options = {
      .stress = {gamma_gas, 1e-12, 1e-12, 1.0,
                 magnetized_case ? 0.25 : 0.0,
                 magnetized_case ? 0.25 : 0.0, 100.0,
                 magnetized_case, magnetized_case, false, false},
      .emf = {1e-12, 1.0, magnetized_case ? 0.5 : 0.0,
              100.0, magnetized_case ? 0.3 : 0.8, 0.0, false, false},
      .update = {gamma_gas, 1e-12, 1e-12, 100.0, magnetized_case, false, false},
      .source_time = 0.0};
  const gamera_no_timestep_options timestep = {
      gamma_gas, 1e-12, 1e-12, magnetized_case ? 0.3 : 0.25,
      100.0, magnetized_case, false, false};
  const size_t lower[3] = {ghost, ghost, ghost};
  const size_t upper[3] = {ghost + active[0], ghost + active[1],
                           ghost + active[2]};
  double time = 0.0, previous_dt = 0.0;
  int steps = 0;
  double min_density, min_pressure, max_radial_speed, mass, max_divergence;
  double face_change[GAMERA_CS_FACE_COUNT];
  if (snapshot(output, time, grid, storage, active, ghost) != 0 ||
      diagnostics(time, grid, storage, active, ghost, &min_density,
                  &min_pressure, &max_radial_speed, face_change, &mass,
                  &max_divergence) != 0)
    return EXIT_FAILURE;
  double next_snapshot = snapshot_cadence;
  while (time < end_time && steps < 2000) {
    double dt = INFINITY;
    for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
      double local_dt = INFINITY;
      if (gamera_no_local_timestep(&grid[face], storage[face].conserved,
              storage[face].cell_magnetic, lower, upper, &timestep,
              NULL, &local_dt) != 0) return EXIT_FAILURE;
      dt = fmin(dt, local_dt);
    }
    dt = fmin(dt, end_time - time);
    if (!isfinite(dt) || dt <= 1e-12 ||
        gamera_cs_advance_six(grid, storage, active, ghost,
              steps == 0 ? 0.0 : 0.5 * dt / previous_dt,
              dt, &options) != 0) {
      fprintf(stderr, "FAIL spherical blast step %d at t=%.9g dt=%.9g\n",
              steps, time, dt);
      if (magnetized_case)
        report_minimum_cfl_cell(grid, storage, active, ghost, timestep.cfl);
      return EXIT_FAILURE;
    }
    time += dt;
    previous_dt = dt;
    ++steps;
    if (time >= next_snapshot || time >= end_time - 1e-12) {
      if (snapshot(output, time, grid, storage, active, ghost) != 0 ||
          diagnostics(time, grid, storage, active, ghost, &min_density,
                      &min_pressure, &max_radial_speed, face_change,
                      &mass, &max_divergence) != 0) return EXIT_FAILURE;
      next_snapshot += snapshot_cadence;
    }
  }
  printf("%s spherical blast complete: steps=%d final_time=%.9g\n",
         magnetized_case ? "MHD" : "HD", steps, time);
  double max_seam_jump, max_seam_ratio;
  if (seam_metrics(storage, active, ghost, &max_seam_jump,
                   &max_seam_ratio) != 0) return EXIT_FAILURE;
  printf("mid-shell seam pressure: max jump=%.8g "
         "max jump/local-gradient=%.8g\n",
         max_seam_jump, max_seam_ratio);
  if (angular == 24U && end_time >= 1.5) {
    for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
      if (face_change[face] <= 0.1) return EXIT_FAILURE;
    if (max_seam_ratio > 5.0) return EXIT_FAILURE;
  }
  /* The local-gradient ratio is ill-conditioned in smooth, near-flat cells.
   * Bound the dimensional jump as well as CT divergence for this blast. */
  if (magnetized_case && (max_divergence > 1e-9 ||
                          max_seam_jump > 0.05)) return EXIT_FAILURE;
  if (output != NULL && fclose(output) != 0) return EXIT_FAILURE;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_storage_destroy(&storage[face]);
    gamera_no_grid_destroy(&grid[face]);
  }
  return time >= end_time - 1e-12 && min_density > 0.0 &&
         min_pressure > 0.0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
