#include "cubedsphere_mi_dipole.h"

#include "nonorthogonal_state.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CS_MI_NORTH = 0, CS_MI_SOUTH = 1, CS_MI_HEMISPHERES = 2 };

struct gamera_cs_mi_dipole {
  const gamera_no_grid *grid;
  gamera_no_storage *storage;
  const gamera_no_background_field *background;
  gamera_cs_earth_startup startup;
  gamera_cs_mi_dipole_config config;
  size_t active[3];
  size_t ghost;
  size_t source_stride;
  gamera_no_vec3 *mi_magnetic[GAMERA_CS_FACE_COUNT];
  double *shell_fac[GAMERA_CS_FACE_COUNT];
  double *fac[CS_MI_HEMISPHERES];
  double *potential[CS_MI_HEMISPHERES];
  gamera_mi_solver_stats solve[CS_MI_HEMISPHERES];
  double sample_radius_re;
  double maximum_colatitude;
  double sample_maximum_colatitude;
  double magnetic_norm_t;
  double potential_norm_v;
  double current_norm_a_m2;
  double next_update_s;
  double epoch_s;
  int ready;
};

static double dot(gamera_no_vec3 a, gamera_no_vec3 b) {
  return a.value[0] * b.value[0] + a.value[1] * b.value[1] +
         a.value[2] * b.value[2];
}

static double magnitude(gamera_no_vec3 a) { return sqrt(dot(a, a)); }

static int valid_config(const gamera_cs_mi_dipole_config *c) {
  return c != NULL && c->azimuth_points >= 8U &&
         c->colatitude_points >= 3U && c->current_shell < 64U &&
         isfinite(c->ionosphere_radius_re) && c->ionosphere_radius_re > 0.0 &&
         isfinite(c->pedersen_siemens) && c->pedersen_siemens > 0.0 &&
         isfinite(c->coupling_interval_s) && c->coupling_interval_s > 0.0 &&
         isfinite(c->length_norm_m) && c->length_norm_m > 0.0 &&
         isfinite(c->speed_norm_m_s) && c->speed_norm_m_s > 0.0 &&
         isfinite(c->mass_density_norm_kg_m3) &&
         c->mass_density_norm_kg_m3 > 0.0 &&
         c->maximum_iterations > 0 && isfinite(c->relative_tolerance) &&
         c->relative_tolerance > 0.0;
}

static int initialize_potential_view(
    gamera_cs_mi_dipole *state, const gamera_cs_earth_startup *startup,
    const gamera_cs_mi_dipole_config *config) {
  if (state == NULL || startup == NULL || !valid_config(config) ||
      !(config->ionosphere_radius_re < startup->inner_radius) ||
      config->azimuth_points > SIZE_MAX / config->colatitude_points)
    return -1;
  state->startup = *startup;
  state->config = *config;
  state->maximum_colatitude = asin(sqrt(config->ionosphere_radius_re /
                                        startup->inner_radius));
  const double mu0 = 4.0e-7 * acos(-1.0);
  state->magnetic_norm_t = config->speed_norm_m_s *
      sqrt(mu0 * config->mass_density_norm_kg_m3);
  state->current_norm_a_m2 = state->magnetic_norm_t /
      (mu0 * config->length_norm_m);
  state->potential_norm_v = config->speed_norm_m_s *
      state->magnetic_norm_t * config->length_norm_m;
  if (!(state->potential_norm_v > 0.0) ||
      !isfinite(state->maximum_colatitude)) return -1;
  const size_t count = config->azimuth_points * config->colatitude_points;
  for (int hemi = 0; hemi < CS_MI_HEMISPHERES; ++hemi) {
    state->potential[hemi] = calloc(count, sizeof(double));
    if (state->potential[hemi] == NULL) return -1;
  }
  return 0;
}

void gamera_cs_mi_dipole_destroy(gamera_cs_mi_dipole *state) {
  if (state == NULL) return;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    free(state->mi_magnetic[face]);
    free(state->shell_fac[face]);
  }
  for (int hemi = 0; hemi < CS_MI_HEMISPHERES; ++hemi) {
    free(state->fac[hemi]);
    free(state->potential[hemi]);
  }
  free(state);
}

int gamera_cs_mi_dipole_create_potential_view(
    const gamera_cs_earth_startup *startup,
    const gamera_cs_mi_dipole_config *config,
    gamera_cs_mi_dipole **result) {
  if (result == NULL) return -1;
  *result = NULL;
  gamera_cs_mi_dipole *state = calloc(1, sizeof(*state));
  if (state == NULL) return -1;
  if (initialize_potential_view(state, startup, config) != 0) {
    gamera_cs_mi_dipole_destroy(state);
    return -1;
  }
  *result = state;
  return 0;
}

int gamera_cs_mi_dipole_export_potential(
    const gamera_cs_mi_dipole *state, double *values, size_t count,
    double *epoch_s, double *next_update_s) {
  if (!gamera_cs_mi_dipole_ready(state) || values == NULL ||
      epoch_s == NULL || next_update_s == NULL ||
      state->config.azimuth_points >
          SIZE_MAX / state->config.colatitude_points / CS_MI_HEMISPHERES)
    return -1;
  const size_t per_hemi = state->config.azimuth_points *
                          state->config.colatitude_points;
  if (count != CS_MI_HEMISPHERES * per_hemi) return -1;
  for (int hemi = 0; hemi < CS_MI_HEMISPHERES; ++hemi)
    memcpy(values + (size_t)hemi * per_hemi, state->potential[hemi],
           per_hemi * sizeof(double));
  *epoch_s = state->epoch_s;
  *next_update_s = state->next_update_s;
  return 0;
}

int gamera_cs_mi_dipole_import_potential(
    gamera_cs_mi_dipole *state, const double *values, size_t count,
    double epoch_s, double next_update_s) {
  if (state == NULL || values == NULL || !isfinite(epoch_s) ||
      !isfinite(next_update_s) || epoch_s < 0.0 ||
      next_update_s <= epoch_s || state->config.azimuth_points >
          SIZE_MAX / state->config.colatitude_points / CS_MI_HEMISPHERES)
    return -1;
  const size_t per_hemi = state->config.azimuth_points *
                          state->config.colatitude_points;
  if (count != CS_MI_HEMISPHERES * per_hemi) return -1;
  for (size_t q = 0; q < count; ++q)
    if (!isfinite(values[q])) return -1;
  for (int hemi = 0; hemi < CS_MI_HEMISPHERES; ++hemi)
    memcpy(state->potential[hemi], values + (size_t)hemi * per_hemi,
           per_hemi * sizeof(double));
  state->epoch_s = epoch_s;
  state->next_update_s = next_update_s;
  state->ready = 1;
  return 0;
}

int gamera_cs_mi_dipole_create(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_earth_startup *startup, const size_t active[3],
    size_t ghost, const gamera_cs_mi_dipole_config *config,
    gamera_cs_mi_dipole **result) {
  if (result == NULL) return -1;
  *result = NULL;
  if (grid == NULL || storage == NULL || background == NULL ||
      startup == NULL || active == NULL || ghost < 4U ||
      active[0] < 3U || active[1] < 8U || active[1] != active[2] ||
      !valid_config(config) || config->current_shell + 1U >= active[0] ||
      !(config->ionosphere_radius_re < startup->inner_radius) ||
      config->azimuth_points > SIZE_MAX / config->colatitude_points)
    return -1;
  gamera_cs_mi_dipole *state = calloc(1, sizeof(*state));
  if (state == NULL) return -1;
  if (initialize_potential_view(state, startup, config) != 0) goto failed;
  state->grid = grid;
  state->storage = storage;
  state->background = background;
  memcpy(state->active, active, sizeof(state->active));
  state->ghost = ghost;
  state->source_stride = active[1] + 2U;
  const size_t source_count = state->source_stride * state->source_stride;
  const size_t polar_count = config->azimuth_points * config->colatitude_points;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    state->mi_magnetic[face] = calloc(
        gamera_no_element_count3(grid[face].cell_extent),
        sizeof(*state->mi_magnetic[face]));
    state->shell_fac[face] = calloc(source_count, sizeof(double));
    if (state->mi_magnetic[face] == NULL ||
        state->shell_fac[face] == NULL) goto failed;
  }
  for (int hemi = 0; hemi < CS_MI_HEMISPHERES; ++hemi) {
    state->fac[hemi] = calloc(polar_count, sizeof(double));
    if (state->fac[hemi] == NULL) goto failed;
  }
  double radius_sum = 0.0;
  size_t radius_count = 0U;
  const size_t shell_i = ghost + config->current_shell;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    for (size_t j = ghost; j < ghost + active[1]; ++j)
      for (size_t k = ghost; k < ghost + active[2]; ++k) {
        const size_t cell = gamera_no_index3(grid[face].cell_extent,
                                             shell_i, j, k);
        radius_sum += magnitude(grid[face].cell[cell].centroid);
        ++radius_count;
      }
  if (radius_count == 0U) goto failed;
  state->sample_radius_re = radius_sum / (double)radius_count;
  state->sample_maximum_colatitude =
      asin(sqrt(config->ionosphere_radius_re / state->sample_radius_re));
  if (!isfinite(state->sample_radius_re) ||
      !(state->sample_radius_re > startup->inner_radius)) goto failed;
  *result = state;
  return 0;
failed:
  gamera_cs_mi_dipole_destroy(state);
  return -1;
}

/* The one-cell fringe is reconstructed from the same exchanged Cartesian
 * residual B used by the MHD stencil, then the unchanged GAMERA Stokes curl
 * is applied on the physical sample shell. */
static int fill_shell_fac(gamera_cs_mi_dipole *s) {
  const size_t i = s->ghost + s->config.current_shell;
  const size_t stride = s->source_stride;
  const double quarter_pi = 0.25 * acos(-1.0);
  /* Keep the FAC stencil in a private magnetic buffer. Neither the MHD
   * active field nor its reconstruction halos are altered by MI sampling. */
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    const size_t count = gamera_no_element_count3(s->grid[face].cell_extent);
    memcpy(s->mi_magnetic[face], s->storage[face].cell_magnetic,
           count * sizeof(*s->mi_magnetic[face]));
  }
  /* A local second-order physical-donor reconstruction avoids taking a
   * derivative through the unlimited eighth-order MHD halo polynomial at
   * cube corners. It is used only for the MI current diagnostic. */
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    for (size_t radial = i - 1U; radial <= i + 1U; ++radial)
      for (size_t local_j = 0; local_j < stride; ++local_j)
        for (size_t local_k = 0; local_k < stride; ++local_k) {
          if (local_j > 0 && local_j <= s->active[1] &&
              local_k > 0 && local_k <= s->active[2]) continue;
          const size_t j = s->ghost - 1U + local_j;
          const size_t k = s->ghost - 1U + local_k;
          const size_t target = gamera_no_index3(s->grid[face].cell_extent,
                                                 radial, j, k);
          int donor_face;
          double a, b;
          if (gamera_cs_inverse(s->grid[face].cell[target].centroid,
                                &donor_face, &a, &b) != 0 ||
              donor_face == face) return -1;
          const double x = (a + quarter_pi) *
              (double)s->active[1] / (2.0 * quarter_pi) - 0.5;
          const double y = (b + quarter_pi) *
              (double)s->active[2] / (2.0 * quarter_pi) - 0.5;
          const size_t j0 = (size_t)fmax(0.0, fmin(
              (double)s->active[1] - 2.0, floor(x)));
          const size_t k0 = (size_t)fmax(0.0, fmin(
              (double)s->active[2] - 2.0, floor(y)));
          const double wx = x - (double)j0;
          const double wy = y - (double)k0;
          const size_t *extent = s->grid[donor_face].cell_extent;
          gamera_no_vec3 field = {{0.0, 0.0, 0.0}};
          for (size_t dj = 0; dj < 2U; ++dj)
            for (size_t dk = 0; dk < 2U; ++dk) {
              const size_t source = gamera_no_index3(
                  extent, radial, s->ghost + j0 + dj,
                  s->ghost + k0 + dk);
              const double weight = (dj == 0U ? 1.0 - wx : wx) *
                                    (dk == 0U ? 1.0 - wy : wy);
              for (int axis = 0; axis < 3; ++axis)
                field.value[axis] += weight *
                    s->storage[donor_face].cell_magnetic[source].value[axis];
            }
          s->mi_magnetic[face][target] = field;
        }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    for (size_t local_j = 1; local_j <= s->active[1]; ++local_j)
      for (size_t local_k = 1; local_k <= s->active[2]; ++local_k) {
        const size_t j = s->ghost - 1U + local_j;
        const size_t k = s->ghost - 1U + local_k;
        const size_t cell = gamera_no_index3(s->grid[face].cell_extent,
                                             i, j, k);
        gamera_no_vec3 current;
        if (gamera_no_cell_current_from_residual(
                &s->grid[face], s->mi_magnetic[face],
                i, j, k, &current) != 0) return -1;
        const gamera_no_vec3 b0 = s->background[face].cell_magnetic[cell];
        const gamera_no_vec3 point = s->grid[face].cell[cell].centroid;
        const double radius = magnitude(point);
        const double bmag = magnitude(b0);
        if (!(radius > 0.0) || !(bmag > 0.0)) return -1;
        const double colatitude = acos(fmin(1.0,
            fabs(point.value[2]) / radius));
        const double ratio = gamera_mi_dipole_field_ratio(
            colatitude, radius, s->config.ionosphere_radius_re);
        const double fac = dot(current, b0) / bmag * ratio *
                           s->current_norm_a_m2;
        if (!isfinite(fac)) return -1;
        s->shell_fac[face][local_j * stride + local_k] = fac;
      }
  /* The Stokes curl is a derivative. Applying it to an interpolated
   * receiver-face magnetic ghost cell differentiates the interpolation
   * error, especially at the three-face cube corners. Compute the curl on
   * physical cells first, then exchange the scalar FAC fringe from its
   * physical owning face for the final MI-grid interpolation. */
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    for (size_t local_j = 0; local_j < stride; ++local_j)
      for (size_t local_k = 0; local_k < stride; ++local_k) {
        if (local_j > 0 && local_j <= s->active[1] &&
            local_k > 0 && local_k <= s->active[2]) continue;
        const size_t j = s->ghost - 1U + local_j;
        const size_t k = s->ghost - 1U + local_k;
        const size_t cell = gamera_no_index3(s->grid[face].cell_extent,
                                             i, j, k);
        int donor_face;
        double a, b;
        if (gamera_cs_inverse(s->grid[face].cell[cell].centroid,
                              &donor_face, &a, &b) != 0 ||
            donor_face == face) return -1;
        const double x = (a + quarter_pi) *
            (double)s->active[1] / (2.0 * quarter_pi) - 0.5;
        const double y = (b + quarter_pi) *
            (double)s->active[2] / (2.0 * quarter_pi) - 0.5;
        const size_t j0 = (size_t)fmax(0.0, fmin(
            (double)s->active[1] - 2.0, floor(x)));
        const size_t k0 = (size_t)fmax(0.0, fmin(
            (double)s->active[2] - 2.0, floor(y)));
        const double wx = x - (double)j0;
        const double wy = y - (double)k0;
        const double *source = s->shell_fac[donor_face];
        const size_t lower = (j0 + 1U) * stride + k0 + 1U;
        const double low = (1.0 - wy) * source[lower] +
                           wy * source[lower + 1U];
        const double high = (1.0 - wy) * source[lower + stride] +
                            wy * source[lower + stride + 1U];
        const double fac = (1.0 - wx) * low + wx * high;
        if (!isfinite(fac)) return -1;
        s->shell_fac[face][local_j * stride + local_k] = fac;
      }
  return 0;
}

static int sample_shell_fac(const gamera_cs_mi_dipole *s,
                            gamera_no_vec3 point, double *value) {
  int face;
  double a, b;
  if (value == NULL || gamera_cs_inverse(point, &face, &a, &b) != 0)
    return -1;
  const double quarter_pi = 0.25 * acos(-1.0);
  const double scale = (double)s->active[1] / (2.0 * quarter_pi);
  const double x = (a + quarter_pi) * scale - 0.5;
  const double y = (b + quarter_pi) * scale - 0.5;
  const long j = (long)floor(x);
  const long k = (long)floor(y);
  if (j < -1L || k < -1L || j >= (long)s->active[1] ||
      k >= (long)s->active[2]) return -1;
  const double wx = x - (double)j;
  const double wy = y - (double)k;
  const size_t row = (size_t)(j + 1L);
  const size_t col = (size_t)(k + 1L);
  const size_t stride = s->source_stride;
  const double *source = s->shell_fac[face];
  const double lower = (1.0 - wy) * source[row * stride + col] +
                       wy * source[row * stride + col + 1U];
  const double upper = (1.0 - wy) * source[(row + 1U) * stride + col] +
                       wy * source[(row + 1U) * stride + col + 1U];
  *value = (1.0 - wx) * lower + wx * upper;
  return isfinite(*value) ? 0 : -1;
}

int gamera_cs_mi_dipole_update_due(const gamera_cs_mi_dipole *s,
                                   double time_code) {
  if (s == NULL || !isfinite(time_code) || time_code < 0.0) return -1;
  const double time_s = time_code * s->config.length_norm_m /
                        s->config.speed_norm_m_s;
  return !s->ready || time_s + 1e-10 >= s->next_update_s;
}

int gamera_cs_mi_dipole_maybe_update(gamera_cs_mi_dipole *s,
                                     double time_code) {
  const int due = gamera_cs_mi_dipole_update_due(s, time_code);
  if (due <= 0) return due;
  if (s->grid == NULL || s->storage == NULL || s->background == NULL)
    return -1;
  const double time_s = time_code * s->config.length_norm_m /
                        s->config.speed_norm_m_s;
  if (fill_shell_fac(s) != 0) return -1;
  const size_t np = s->config.azimuth_points;
  const size_t nt = s->config.colatitude_points;
  const double dtheta = s->maximum_colatitude / (double)(nt - 1U);
  const double dphi = 2.0 * acos(-1.0) / (double)np;
  for (int hemi = 0; hemi < CS_MI_HEMISPHERES; ++hemi) {
    const double sign = hemi == CS_MI_NORTH ? 1.0 : -1.0;
    for (size_t t = 0; t < nt; ++t) {
      const double theta_i = (double)t * dtheta;
      if (theta_i > s->sample_maximum_colatitude + 4e-14) {
        memset(s->fac[hemi] + t * np, 0, np * sizeof(double));
        continue;
      }
      const double argument = fmin(1.0, sin(theta_i) * sqrt(
          s->sample_radius_re / s->config.ionosphere_radius_re));
      const double theta_mhd = asin(argument);
      for (size_t p = 0; p < np; ++p) {
        const double phi = (double)p * dphi;
        const gamera_no_vec3 point = {{
            s->sample_radius_re * sin(theta_mhd) * cos(phi),
            s->sample_radius_re * sin(theta_mhd) * sin(phi),
            sign * s->sample_radius_re * cos(theta_mhd)}};
        if (sample_shell_fac(s, point, &s->fac[hemi][t * np + p]) != 0)
          return -1;
      }
    }
    const gamera_mi_solver_config solver = {
        .longitude_count = np, .colatitude_count = nt,
        .maximum_colatitude = s->maximum_colatitude,
        .ionosphere_radius_m = s->config.ionosphere_radius_re *
                               s->config.length_norm_m,
        .pedersen_siemens = s->config.pedersen_siemens,
        .hall_siemens = 0.0, .low_latitude_potential_v = 0.0,
        .hemisphere = hemi == CS_MI_NORTH ? GAMERA_MI_NORTH : GAMERA_MI_SOUTH,
        .maximum_iterations = s->config.maximum_iterations,
        .relative_tolerance = s->config.relative_tolerance,
        .absolute_tolerance = 0.0};
    if (gamera_mi_solve_constant_pedersen(
            &solver, s->fac[hemi], s->potential[hemi],
            &s->solve[hemi]) != 0 || !s->solve[hemi].converged)
      return -1;
  }
  s->ready = 1;
  s->epoch_s = time_s;
  do {
    s->next_update_s += s->config.coupling_interval_s;
  } while (s->next_update_s <= time_s + 1e-10);
  return 1;
}

int gamera_cs_mi_dipole_ready(const gamera_cs_mi_dipole *s) {
  return s != NULL && s->ready;
}

int gamera_cs_mi_dipole_potential(const gamera_cs_mi_dipole *s,
                                  gamera_no_vec3 point,
                                  double *potential_v) {
  if (!gamera_cs_mi_dipole_ready(s) || potential_v == NULL) return -1;
  const double radius = magnitude(point);
  if (!(radius >= s->config.ionosphere_radius_re)) return -1;
  const int hemi = point.value[2] >= 0.0 ? CS_MI_NORTH : CS_MI_SOUTH;
  const double theta_mhd = acos(fmin(1.0,
      fabs(point.value[2]) / radius));
  const double theta_i = gamera_mi_mapped_colatitude(
      theta_mhd, radius, s->config.ionosphere_radius_re);
  if (!isfinite(theta_i)) return -1;
  if (theta_i >= s->maximum_colatitude) {
    *potential_v = 0.0;
    return 0;
  }
  const size_t np = s->config.azimuth_points;
  const size_t nt = s->config.colatitude_points;
  const double tq = theta_i / s->maximum_colatitude * (double)(nt - 1U);
  const size_t t = (size_t)floor(tq);
  const double wt = tq - (double)t;
  double phi = atan2(point.value[1], point.value[0]);
  if (phi < 0.0) phi += 2.0 * acos(-1.0);
  const double pq = phi * (double)np / (2.0 * acos(-1.0));
  const size_t p = (size_t)floor(pq) % np;
  const size_t q = (p + 1U) % np;
  const double wp = pq - floor(pq);
  const double *potential = s->potential[hemi];
  const double lower = (1.0 - wp) * potential[t * np + p] +
                       wp * potential[t * np + q];
  const double upper = (1.0 - wp) * potential[(t + 1U) * np + p] +
                       wp * potential[(t + 1U) * np + q];
  *potential_v = (1.0 - wt) * lower + wt * upper;
  return isfinite(*potential_v) ? 0 : -1;
}

int gamera_cs_mi_dipole_apply_inner_emf(
    const gamera_cs_mi_dipole *s, gamera_no_storage *storage,
    const gamera_no_grid *grid, size_t wall_i) {
  if (!gamera_cs_mi_dipole_ready(s) || storage == NULL || grid == NULL)
    return -1;
  for (int direction = GAMERA_NO_J; direction <= GAMERA_NO_K; ++direction)
    for (size_t j = 0; j < grid->edge[direction].extent[1]; ++j)
      for (size_t k = 0; k < grid->edge[direction].extent[2]; ++k) {
        const size_t edge = gamera_no_index3(
            grid->edge[direction].extent, wall_i, j, k);
        if (!grid->edge[direction].valid[edge]) continue;
        size_t end[3] = {wall_i, j, k};
        ++end[direction];
        const gamera_no_vec3 a = grid->vertex[gamera_no_index3(
            grid->vertex_extent, wall_i, j, k)];
        const gamera_no_vec3 b = grid->vertex[gamera_no_index3(
            grid->vertex_extent, end[0], end[1], end[2])];
        double phi_a, phi_b;
        if (gamera_cs_mi_dipole_potential(s, a, &phi_a) != 0 ||
            gamera_cs_mi_dipole_potential(s, b, &phi_b) != 0)
          return -1;
        storage->edge_emf[direction][edge] =
            -(phi_b - phi_a) / s->potential_norm_v;
      }
  return 0;
}

int gamera_cs_mi_dipole_ghost_velocity(
    const gamera_cs_mi_dipole *s, gamera_no_vec3 point,
    gamera_no_vec3 source_velocity, gamera_no_vec3 *ghost_velocity) {
  if (!gamera_cs_mi_dipole_ready(s) || ghost_velocity == NULL) return -1;
  const double radius = magnitude(point);
  if (!(radius > 0.0)) return -1;
  const double epsilon = 1e-4 * fmax(1.0, radius);
  gamera_no_vec3 electric = {{0.0, 0.0, 0.0}};
  for (int component = 0; component < GAMERA_NO_DIM; ++component) {
    gamera_no_vec3 plus = point, minus = point;
    plus.value[component] += epsilon;
    minus.value[component] -= epsilon;
    double phi_plus, phi_minus;
    if (gamera_cs_mi_dipole_potential(s, plus, &phi_plus) != 0 ||
        gamera_cs_mi_dipole_potential(s, minus, &phi_minus) != 0)
      return -1;
    electric.value[component] = -(phi_plus - phi_minus) /
        (2.0 * epsilon * s->config.length_norm_m *
         s->config.speed_norm_m_s * s->magnetic_norm_t);
  }
  gamera_no_vec3 background;
  if (gamera_cs_earth_startup_background(
          point, (void *)&s->startup, &background) != 0)
    return -1;
  const double b2 = dot(background, background);
  if (!(b2 > 0.0)) return -1;
  gamera_no_vec3 drift = {{
      (electric.value[1] * background.value[2] -
       electric.value[2] * background.value[1]) / b2,
      (electric.value[2] * background.value[0] -
       electric.value[0] * background.value[2]) / b2,
      (electric.value[0] * background.value[1] -
       electric.value[1] * background.value[0]) / b2}};
  const double radial = dot(drift, point) / (radius * radius);
  for (int c = 0; c < GAMERA_NO_DIM; ++c)
    drift.value[c] -= radial * point.value[c];
  return gamera_mi_compose_ghost_velocity(source_velocity, background,
                                          drift, ghost_velocity);
}

int gamera_cs_mi_dipole_stats(const gamera_cs_mi_dipole *s,
                              double *fac_min, double *fac_max,
                              double *potential_min_v,
                              double *potential_max_v,
                              gamera_mi_solver_stats solve[2]) {
  if (!gamera_cs_mi_dipole_ready(s) || fac_min == NULL || fac_max == NULL ||
      potential_min_v == NULL || potential_max_v == NULL || solve == NULL)
    return -1;
  const size_t count = s->config.azimuth_points *
                       s->config.colatitude_points;
  *fac_min = INFINITY;
  *fac_max = -INFINITY;
  *potential_min_v = INFINITY;
  *potential_max_v = -INFINITY;
  for (int hemi = 0; hemi < CS_MI_HEMISPHERES; ++hemi) {
    solve[hemi] = s->solve[hemi];
    for (size_t q = 0; q < count; ++q) {
      *fac_min = fmin(*fac_min, s->fac[hemi][q]);
      *fac_max = fmax(*fac_max, s->fac[hemi][q]);
      *potential_min_v = fmin(*potential_min_v, s->potential[hemi][q]);
      *potential_max_v = fmax(*potential_max_v, s->potential[hemi][q]);
    }
  }
  return 0;
}

int gamera_cs_mi_dipole_analysis_view(
    const gamera_cs_mi_dipole *s, gamera_cs_mi_analysis_view *view) {
  if (!gamera_cs_mi_dipole_ready(s) || view == NULL ||
      s->fac[0] == NULL || s->fac[1] == NULL) return -1;
  view->azimuth_points = s->config.azimuth_points;
  view->colatitude_points = s->config.colatitude_points;
  view->shell_j_cells = s->active[1];
  view->shell_k_cells = s->active[2];
  view->maximum_colatitude_rad = s->maximum_colatitude;
  view->epoch_s = s->epoch_s;
  view->sample_radius_re = s->sample_radius_re;
  view->pedersen_siemens = s->config.pedersen_siemens;
  for (int h = 0; h < CS_MI_HEMISPHERES; ++h) {
    view->fac_parallel_a_m2[h] = s->fac[h];
    view->potential_v[h] = s->potential[h];
  }
  return 0;
}

int gamera_cs_mi_dipole_export_shell_current(
    const gamera_cs_mi_dipole *s, double *jparallel_a_m2,
    double *mlat_deg, double *phi_deg, size_t count) {
  if (!gamera_cs_mi_dipole_ready(s) || !jparallel_a_m2 || !mlat_deg ||
      !phi_deg || count != GAMERA_CS_FACE_COUNT*s->active[1]*s->active[2])
    return -1;
  const size_t i=s->ghost+s->config.current_shell;
  const double pi=acos(-1.0);
  for(int face=0;face<GAMERA_CS_FACE_COUNT;++face)
    for(size_t j=0;j<s->active[1];++j)
      for(size_t k=0;k<s->active[2];++k) {
        size_t jj=j+s->ghost,kk=k+s->ghost;
        size_t q=((size_t)face*s->active[1]+j)*s->active[2]+k;
        size_t cell=gamera_no_index3(s->grid[face].cell_extent,i,jj,kk);
        gamera_no_vec3 current;
        if(gamera_no_cell_current_from_residual(
             &s->grid[face],s->mi_magnetic[face],i,jj,kk,&current)) return -1;
        gamera_no_vec3 b0=s->background[face].cell_magnetic[cell];
        gamera_no_vec3 point=s->grid[face].cell[cell].centroid;
        double radius=magnitude(point),bmag=magnitude(b0);
        if(!(radius>0.0)||!(bmag>0.0)) return -1;
        double phi=atan2(point.value[1],point.value[0])*180.0/pi;
        if(phi<0.0) phi+=360.0;
        mlat_deg[q]=asin(point.value[2]/radius)*180.0/pi;
        phi_deg[q]=phi;
        jparallel_a_m2[q]=dot(current,b0)/bmag*s->current_norm_a_m2;
      }
  return 0;
}

int gamera_cs_mi_dipole_write_csv(const gamera_cs_mi_dipole *s,
                                  const char *prefix) {
  if (!gamera_cs_mi_dipole_ready(s) || prefix == NULL) return -1;
  const size_t length = strlen(prefix);
  if (length > 4000U) return -1;
  char name[4096];
  const size_t np = s->config.azimuth_points;
  const size_t nt = s->config.colatitude_points;
  for (int hemi = 0; hemi < CS_MI_HEMISPHERES; ++hemi) {
    if (snprintf(name, sizeof(name), "%s.mi_%s.csv", prefix,
                 hemi == CS_MI_NORTH ? "north" : "south") >=
        (int)sizeof(name)) return -1;
    FILE *file = fopen(name, "w");
    if (file == NULL) return -1;
    fputs("time_s,hemisphere,theta_deg,phi_deg,fac_A_m2,potential_V\n", file);
    for (size_t t = 0; t < nt; ++t)
      for (size_t p = 0; p < np; ++p) {
        const size_t q = t * np + p;
        fprintf(file, "%.17g,%s,%.17g,%.17g,%.17g,%.17g\n",
                s->epoch_s, hemi == CS_MI_NORTH ? "north" : "south",
                s->maximum_colatitude * (double)t /
                    (double)(nt - 1U) * 180.0 / acos(-1.0),
                360.0 * (double)p / (double)np,
                s->fac[hemi][q], s->potential[hemi][q]);
      }
    if (fclose(file) != 0) return -1;
  }
  return 0;
}

int gamera_cs_mi_dipole_write_shell_current_csv(
    const gamera_cs_mi_dipole *s, const char *prefix) {
  if (!gamera_cs_mi_dipole_ready(s) || prefix == NULL) return -1;
  char path[4096];
  if (snprintf(path, sizeof(path), "%s.shell_j.csv", prefix) >=
      (int)sizeof(path)) return -1;
  FILE *file = fopen(path, "w");
  if (file == NULL) return -1;
  fputs("face,j,k,mlat_deg,phi_deg,jparallel_A_m2\n", file);
  const double pi = acos(-1.0);
  const size_t i = s->ghost + s->config.current_shell;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    for (size_t j = s->ghost; j < s->ghost + s->active[1]; ++j)
      for (size_t k = s->ghost; k < s->ghost + s->active[2]; ++k) {
        const size_t cell = gamera_no_index3(s->grid[face].cell_extent,
                                             i, j, k);
        gamera_no_vec3 current;
        if (gamera_no_cell_current_from_residual(
                &s->grid[face], s->mi_magnetic[face],
                i, j, k, &current) != 0) {
          fclose(file);
          return -1;
        }
        const gamera_no_vec3 b0 = s->background[face].cell_magnetic[cell];
        const gamera_no_vec3 point = s->grid[face].cell[cell].centroid;
        const double radius = magnitude(point);
        const double bmag = magnitude(b0);
        if (!(radius > 0.0) || !(bmag > 0.0)) {
          fclose(file);
          return -1;
        }
        double phi = atan2(point.value[1], point.value[0]) * 180.0 / pi;
        if (phi < 0.0) phi += 360.0;
        fprintf(file, "%d,%zu,%zu,%.17g,%.17g,%.17g\n",
                face, j - s->ghost, k - s->ghost,
                asin(point.value[2] / radius) * 180.0 / pi,
                phi, dot(current, b0) / bmag * s->current_norm_a_m2);
      }
  return fclose(file) == 0 ? 0 : -1;
}
