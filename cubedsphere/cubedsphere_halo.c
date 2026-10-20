#include "cubedsphere_halo.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int checked_add(size_t a, size_t b, size_t *out) {
  if (a > SIZE_MAX - b) return -1;
  *out = a + b;
  return 0;
}

static int checked_multiply(size_t a, size_t b, size_t *out) {
  if (b != 0 && a > SIZE_MAX / b) return -1;
  *out = a * b;
  return 0;
}

static size_t stencil_start(double x, size_t n) {
  long start = (long)floor(x) - 3L;
  if (start < 0) start = 0;
  if (start > (long)n - 8L) start = (long)n - 8L;
  return (size_t)start;
}

static void donor_linear_weights(double x, size_t start, double weight[8]) {
  /* Same base/fraction rule as production Yin-Yang interpolation_logical.
   * Retain the existing sparse-plan layout; only two donors have weight.
   * Unlike overlapping YY patches, a cube corner can lie half a cell beyond
   * donor centers. Continue the nearest two-center slope there rather than
   * using an eighth-order one-sided extrapolation or flattening the slope.
   * This geometrical extension is specific to the non-overlapping cube.
   * Patch-interior GAMERA reconstruction and flux algorithms are unchanged.
   */
  const double logical = x - (double)start;
  int base;
  double fraction;
  if (logical < 0.0) {
    base = 0;
    fraction = logical;
  } else if (logical >= 7.0) {
    base = 6;
    fraction = logical - 6.0;
  } else {
    base = (int)floor(logical);
    fraction = logical - (double)base;
  }
  for (int p = 0; p < 8; ++p) weight[p] = 0.0;
  weight[base] = 1.0 - fraction;
  weight[base + 1] = fraction;
}

int gamera_cs_tiled_donor_stencil_create(
    gamera_no_vec3 point, const size_t global_active[3],
    const int proc_dims[3], size_t global_radial_cell,
    int magnetic_face, gamera_cs_tiled_donor_stencil *result) {
  if (global_active == NULL || proc_dims == NULL || result == NULL ||
      global_active[1] < 8 || global_active[2] < 8 ||
      global_active[1] != global_active[2] ||
      global_radial_cell >= global_active[0]) return -1;
  int donor;
  double a, b;
  if (gamera_cs_inverse(point, &donor, &a, &b) != 0) return -1;
  const double quarter_pi = 0.25 * acos(-1.0);
  const double da = 2.0 * quarter_pi / (double)global_active[1];
  const double db = 2.0 * quarter_pi / (double)global_active[2];
  /* Match the operation order of the accepted cell/face halo plan. */
  const double x = magnetic_face ?
      (a + quarter_pi) * (double)global_active[1] / (2.0 * quarter_pi) - 0.5 :
      (a + quarter_pi) / da - 0.5;
  const double y = magnetic_face ?
      (b + quarter_pi) * (double)global_active[2] / (2.0 * quarter_pi) - 0.5 :
      (b + quarter_pi) / db - 0.5;
  result->donor_face = donor;
  result->start[0] = stencil_start(x, global_active[1]);
  result->start[1] = stencil_start(y, global_active[2]);
  result->nearest[0] = (size_t)fmax(0.0, fmin(
      (double)(global_active[1] - 1U), nearbyint(x))) - result->start[0];
  result->nearest[1] = (size_t)fmax(0.0, fmin(
      (double)(global_active[2] - 1U), nearbyint(y))) - result->start[1];
  if (result->nearest[0] >= 8 || result->nearest[1] >= 8) return -1;
  donor_linear_weights(x, result->start[0], result->weight[0]);
  donor_linear_weights(y, result->start[1], result->weight[1]);
  for (size_t p = 0; p < 8; ++p)
    for (size_t q = 0; q < 8; ++q) {
      const size_t global_cell[3] = {
          global_radial_cell, result->start[0] + p,
          result->start[1] + q};
      if (gamera_cs_decomposition_cell_owner(
              global_active, proc_dims, donor, global_cell,
              &result->rank[p][q], result->local_cell[p][q]) != 0)
        return -1;
    }
  return 0;
}

int gamera_cs_halo_valid_state(const double state[GAMERA_NO_FLUX_COUNT]) {
  const double rho = state[GAMERA_NO_FLUX_DENSITY];
  if (!isfinite(rho) || rho <= 0.0) return 0;
  double momentum_square = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double momentum = state[GAMERA_NO_FLUX_MOMENTUM_X + d];
    if (!isfinite(momentum)) return 0;
    momentum_square += momentum * momentum;
  }
  return isfinite(state[GAMERA_NO_FLUX_ENERGY]) &&
         state[GAMERA_NO_FLUX_ENERGY] >
             0.5 * momentum_square / rho;
}

int gamera_cs_fill_tangential_cell_halos(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost) {
  if (grid == NULL || storage == NULL || active == NULL || ghost < 4 ||
      active[1] < 8 || active[2] < 8 || active[1] != active[2]) {
    return -1;
  }
  const double quarter_pi = 0.25 * acos(-1.0);
  const double da = 2.0 * quarter_pi / (double)active[1];
  const double db = 2.0 * quarter_pi / (double)active[2];
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int d = 0; d < 3; ++d) {
      if (grid[face].cell_extent[d] != active[d] + 2U * ghost ||
          storage[face].cell_extent[d] != grid[face].cell_extent[d]) {
        return -1;
      }
    }
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    const size_t *extent = grid[face].cell_extent;
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = 0; j < extent[1]; ++j) {
        for (size_t k = 0; k < extent[2]; ++k) {
          if (j >= ghost && j < ghost + active[1] &&
              k >= ghost && k < ghost + active[2]) {
            continue;
          }
          const size_t target = gamera_no_index3(extent, i, j, k);
          const gamera_no_vec3 point = grid[face].cell[target].centroid;
          int donor_face;
          double a, b;
          if (gamera_cs_inverse(point, &donor_face, &a, &b) != 0 ||
              donor_face == face) {
            return -1;
          }
          const double x = (a + quarter_pi) / da - 0.5;
          const double y = (b + quarter_pi) / db - 0.5;
          const size_t start_x = stencil_start(x, active[1]);
          const size_t start_y = stencil_start(y, active[2]);
          double wx[8], wy[8];
          donor_linear_weights(x, start_x, wx);
          donor_linear_weights(y, start_y, wy);
          double current[GAMERA_NO_FLUX_COUNT] = {0.0};
          double old[GAMERA_NO_FLUX_COUNT] = {0.0};
          gamera_no_vec3 current_b = {{0.0, 0.0, 0.0}};
          gamera_no_vec3 old_b = {{0.0, 0.0, 0.0}};
          const gamera_no_storage *donor = &storage[donor_face];
          for (int p = 0; p < 8; ++p) {
            for (int q = 0; q < 8; ++q) {
              const size_t source = gamera_no_index3(
                  donor->cell_extent, i,
                  ghost + start_x + (size_t)p,
                  ghost + start_y + (size_t)q);
              const double weight = wx[p] * wy[q];
              for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
                current[m] += weight * donor->conserved[
                    source * GAMERA_NO_FLUX_COUNT + (size_t)m];
                old[m] += weight * donor->old_conserved[
                    source * GAMERA_NO_FLUX_COUNT + (size_t)m];
              }
              for (int d = 0; d < 3; ++d) {
                current_b.value[d] += weight *
                    donor->cell_magnetic[source].value[d];
                old_b.value[d] += weight *
                    donor->old_cell_magnetic[source].value[d];
              }
            }
          }
          /* A discontinuity can make any unlimited 8th-order interpolation
           * nonphysical. Use the nearest physical donor cell there; this is
           * a localized positivity fallback, not a global limiter. */
          if (!gamera_cs_halo_valid_state(current) ||
              !gamera_cs_halo_valid_state(old)) {
            const size_t near_x = (size_t)fmax(0.0, fmin(
                (double)(active[1] - 1U), nearbyint(x)));
            const size_t near_y = (size_t)fmax(0.0, fmin(
                (double)(active[2] - 1U), nearbyint(y)));
            const size_t source = gamera_no_index3(
                donor->cell_extent, i, ghost + near_x, ghost + near_y);
            memcpy(current, &donor->conserved[source * GAMERA_NO_FLUX_COUNT],
                   sizeof(current));
            memcpy(old, &donor->old_conserved[source * GAMERA_NO_FLUX_COUNT],
                   sizeof(old));
            current_b = donor->cell_magnetic[source];
            old_b = donor->old_cell_magnetic[source];
          }
          memcpy(&storage[face].conserved[target * GAMERA_NO_FLUX_COUNT],
                 current, sizeof(current));
          memcpy(&storage[face].old_conserved[target * GAMERA_NO_FLUX_COUNT],
                 old, sizeof(old));
          storage[face].cell_magnetic[target] = current_b;
          storage[face].old_cell_magnetic[target] = old_b;
        }
      }
    }
  }
  return 0;
}

static int fill_radial_outflow_cells_filtered(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, int receiver) {
  if (storage == NULL || active == NULL || ghost < 4 || active[0] == 0)
    return -1;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    if (receiver >= 0 && face != receiver) continue;
    gamera_no_storage *s = &storage[face];
    for (size_t i = 0; i < s->cell_extent[0]; ++i) {
      if (i >= ghost && i < ghost + active[0]) continue;
      const size_t donor_i = i < ghost ? ghost : ghost + active[0] - 1U;
      for (size_t j = 0; j < s->cell_extent[1]; ++j) {
        for (size_t k = 0; k < s->cell_extent[2]; ++k) {
          const size_t target = gamera_no_index3(s->cell_extent, i, j, k);
          const size_t donor = gamera_no_index3(s->cell_extent, donor_i, j, k);
          memcpy(&s->conserved[target * GAMERA_NO_FLUX_COUNT],
                 &s->conserved[donor * GAMERA_NO_FLUX_COUNT],
                 GAMERA_NO_FLUX_COUNT * sizeof(double));
          memcpy(&s->old_conserved[target * GAMERA_NO_FLUX_COUNT],
                 &s->old_conserved[donor * GAMERA_NO_FLUX_COUNT],
                 GAMERA_NO_FLUX_COUNT * sizeof(double));
          s->cell_magnetic[target] = s->cell_magnetic[donor];
          s->old_cell_magnetic[target] = s->old_cell_magnetic[donor];
        }
      }
    }
  }
  return 0;
}

int gamera_cs_fill_radial_outflow_cells(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost) {
  return fill_radial_outflow_cells_filtered(storage, active, ghost, -1);
}

int gamera_cs_fill_radial_outflow_cells_face(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, int receiver) {
  if (receiver < 0 || receiver >= GAMERA_CS_FACE_COUNT) return -1;
  return fill_radial_outflow_cells_filtered(storage, active, ghost, receiver);
}

static int interpolate_face_field(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    const gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, gamera_no_vec3 point,
    size_t radial_cell, gamera_no_vec3 *current,
    gamera_no_vec3 *old) {
  int donor_face;
  double a, b;
  if (gamera_cs_inverse(point, &donor_face, &a, &b) != 0) return -1;
  const double quarter_pi = 0.25 * acos(-1.0);
  const double x = (a + quarter_pi) * (double)active[1] /
                   (2.0 * quarter_pi) - 0.5;
  const double y = (b + quarter_pi) * (double)active[2] /
                   (2.0 * quarter_pi) - 0.5;
  const size_t start_x = stencil_start(x, active[1]);
  const size_t start_y = stencil_start(y, active[2]);
  double wx[8], wy[8];
  donor_linear_weights(x, start_x, wx);
  donor_linear_weights(y, start_y, wy);
  *current = (gamera_no_vec3){{0.0, 0.0, 0.0}};
  *old = *current;
  const size_t source_i = radial_cell < ghost ? ghost :
      radial_cell >= ghost + active[0] ? ghost + active[0] - 1U : radial_cell;
  for (int p = 0; p < 8; ++p) {
    for (int q = 0; q < 8; ++q) {
      const size_t source = gamera_no_index3(
          grid[donor_face].cell_extent, source_i,
          ghost + start_x + (size_t)p,
          ghost + start_y + (size_t)q);
      const double weight = wx[p] * wy[q];
      for (int d = 0; d < 3; ++d) {
        current->value[d] += weight *
            storage[donor_face].cell_magnetic[source].value[d];
        old->value[d] += weight *
            storage[donor_face].old_cell_magnetic[source].value[d];
      }
    }
  }
  return 0;
}

int gamera_cs_fill_magnetic_face_halos(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost) {
  if (grid == NULL || storage == NULL || active == NULL || ghost < 4 ||
      active[1] < 8 || active[1] != active[2]) return -1;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int direction = 0; direction < 3; ++direction) {
      const size_t *extent = grid[face].face[direction].extent;
      for (size_t i = 0; i < extent[0]; ++i) {
        for (size_t j = 0; j < extent[1]; ++j) {
          for (size_t k = 0; k < extent[2]; ++k) {
            const size_t coord[3] = {i, j, k};
            int physical = 1;
            for (int d = 0; d < 3; ++d) {
              const size_t upper = ghost + active[d] +
                                   (direction == d ? 1U : 0U);
              if (coord[d] < ghost || coord[d] >= upper) physical = 0;
            }
            if (physical) continue;
            const size_t index = gamera_no_index3(extent, i, j, k);
            const gamera_no_face_geometry *geometry =
                &grid[face].face[direction].value[index];
            gamera_no_vec3 current, old;
            if (interpolate_face_field(grid, storage, active, ghost,
                    geometry->centroid, i, &current, &old) != 0) return -1;
            double current_flux = 0.0, old_flux = 0.0;
            for (int d = 0; d < 3; ++d) {
              current_flux += current.value[d] *
                  geometry->area_vector.value[d];
              old_flux += old.value[d] *
                  geometry->area_vector.value[d];
            }
            storage[face].face_flux[direction][index] = current_flux;
            storage[face].old_face_flux[direction][index] = old_flux;
          }
        }
      }
    }
  }
  return 0;
}

typedef struct {
  int receiver, donor, direction;
  size_t target, source_i, start_x, start_y, nearest;
  size_t donor_row[8];
  double wx[8], wy[8];
} halo_plan_entry;

struct gamera_cs_halo_plan {
  size_t active[3], ghost, cell_count, face_count;
  unsigned char *donor_mask[GAMERA_CS_FACE_COUNT][GAMERA_CS_FACE_COUNT];
  size_t donor_count[GAMERA_CS_FACE_COUNT][GAMERA_CS_FACE_COUNT];
  size_t cell_begin[GAMERA_CS_FACE_COUNT], cell_end[GAMERA_CS_FACE_COUNT];
  size_t face_begin[GAMERA_CS_FACE_COUNT], face_end[GAMERA_CS_FACE_COUNT];
  const gamera_no_cell_geometry *cell_geometry[GAMERA_CS_FACE_COUNT];
  const gamera_no_face_geometry *face_geometry[GAMERA_CS_FACE_COUNT][3];
  halo_plan_entry *cell, *face;
};

void gamera_cs_halo_plan_destroy(gamera_cs_halo_plan *plan) {
  if (plan == NULL) return;
  for (int receiver = 0; receiver < GAMERA_CS_FACE_COUNT; ++receiver)
    for (int donor = 0; donor < GAMERA_CS_FACE_COUNT; ++donor)
      free(plan->donor_mask[receiver][donor]);
  free(plan->cell);
  free(plan->face);
  free(plan);
}

const unsigned char *gamera_cs_halo_plan_donor_mask(
    const gamera_cs_halo_plan *plan, int receiver, int donor,
    size_t *required_count) {
  if (plan == NULL || required_count == NULL || receiver < 0 || donor < 0 ||
      receiver >= GAMERA_CS_FACE_COUNT || donor >= GAMERA_CS_FACE_COUNT)
    return NULL;
  *required_count = plan->donor_count[receiver][donor];
  return plan->donor_mask[receiver][donor];
}

static void mark_donor_stencil(gamera_cs_halo_plan *plan,
                               const halo_plan_entry *entry,
                               int nearest_fallback) {
  unsigned char *mask = plan->donor_mask[entry->receiver][entry->donor];
  const size_t n = plan->active[2];
  for (size_t p = 0; p < 8; ++p)
    for (size_t q = 0; q < 8; ++q)
      mask[(entry->start_x + p) * n + entry->start_y + q] = 1;
  if (nearest_fallback) {
    const size_t stride = (plan->active[1] + 2 * plan->ghost) *
                          (plan->active[2] + 2 * plan->ghost);
    const size_t within = entry->nearest % stride;
    const size_t full_k = plan->active[2] + 2 * plan->ghost;
    const size_t j = within / full_k - plan->ghost;
    const size_t k = within % full_k - plan->ghost;
    mask[j * n + k] = 1;
  }
}

static int prepare_plan_entry(halo_plan_entry *entry,
                              const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
                              const size_t active[3], size_t ghost,
                              int receiver, int direction, size_t i,
                              size_t j, size_t k, int magnetic_face) {
  const size_t *extent = magnetic_face ?
      grid[receiver].face[direction].extent : grid[receiver].cell_extent;
  const size_t target = gamera_no_index3(extent, i, j, k);
  const gamera_no_vec3 point = magnetic_face ?
      grid[receiver].face[direction].value[target].centroid :
      grid[receiver].cell[target].centroid;
  int donor;
  double a, b;
  if (gamera_cs_inverse(point, &donor, &a, &b) != 0 ||
      (!magnetic_face && donor == receiver)) return -1;
  const double quarter_pi = 0.25 * acos(-1.0);
  /* Retain the two original expressions, including their operation order.
   * Cell and face centroids are sampled at their actual radial positions. */
  const double da = 2.0 * quarter_pi / (double)active[1];
  const double db = 2.0 * quarter_pi / (double)active[2];
  const double x = magnetic_face ?
      (a + quarter_pi) * (double)active[1] / (2.0 * quarter_pi) - 0.5 :
      (a + quarter_pi) / da - 0.5;
  const double y = magnetic_face ?
      (b + quarter_pi) * (double)active[2] / (2.0 * quarter_pi) - 0.5 :
      (b + quarter_pi) / db - 0.5;
  entry->receiver = receiver;
  entry->donor = donor;
  entry->direction = direction;
  entry->target = target;
  entry->source_i = magnetic_face ?
      (i < ghost ? ghost :
       i >= ghost + active[0] ? ghost + active[0] - 1U : i) : i;
  entry->start_x = stencil_start(x, active[1]);
  entry->start_y = stencil_start(y, active[2]);
  for (size_t p = 0; p < 8; ++p)
    entry->donor_row[p] = gamera_no_index3(grid[donor].cell_extent,
        entry->source_i, ghost + entry->start_x + p,
        ghost + entry->start_y);
  donor_linear_weights(x, entry->start_x, entry->wx);
  donor_linear_weights(y, entry->start_y, entry->wy);
  if (!magnetic_face) {
    const size_t near_x = (size_t)fmax(0.0, fmin(
        (double)(active[1] - 1U), nearbyint(x)));
    const size_t near_y = (size_t)fmax(0.0, fmin(
        (double)(active[2] - 1U), nearbyint(y)));
    entry->nearest = gamera_no_index3(grid[donor].cell_extent, i,
                                      ghost + near_x, ghost + near_y);
  }
  return 0;
}

int gamera_cs_halo_plan_create(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, gamera_cs_halo_plan **out) {
  if (grid == NULL || active == NULL || out == NULL || ghost < 4 ||
      active[0] == 0 || active[1] < 8 || active[1] != active[2]) return -1;
  if (ghost > (SIZE_MAX - 1U) / 2U) return -1;
  for (int d = 0; d < 3; ++d)
    if (active[d] > SIZE_MAX - 1U - 2U * ghost) return -1;
  *out = NULL;
  gamera_cs_halo_plan *plan = calloc(1, sizeof(*plan));
  if (plan == NULL) return -1;
  memcpy(plan->active, active, sizeof(plan->active));
  plan->ghost = ghost;
  for (int receiver = 0; receiver < GAMERA_CS_FACE_COUNT; ++receiver) {
    const gamera_no_grid *g = &grid[receiver];
    for (int d = 0; d < 3; ++d) {
      if (g->cell_extent[d] != active[d] + 2U * ghost ||
          g->cell == NULL || g->face[d].value == NULL) goto fail;
      plan->face_geometry[receiver][d] = g->face[d].value;
    }
    plan->cell_geometry[receiver] = g->cell;
    size_t angular_extent, angular_active, add_cells;
    if (checked_multiply(g->cell_extent[1], g->cell_extent[2],
                         &angular_extent) != 0 ||
        checked_multiply(active[1], active[2], &angular_active) != 0 ||
        angular_extent < angular_active ||
        checked_multiply(active[0], angular_extent - angular_active,
                         &add_cells) != 0 ||
        checked_add(plan->cell_count, add_cells,
                    &plan->cell_count) != 0) goto fail;
    for (int direction = 0; direction < 3; ++direction) {
      const size_t *extent = g->face[direction].extent;
      size_t physical, total, next;
      if (checked_multiply(active[0] + (direction == 0),
                           active[1] + (direction == 1), &physical) != 0 ||
          checked_multiply(physical, active[2] + (direction == 2),
                           &physical) != 0 ||
          checked_multiply(extent[0], extent[1], &total) != 0 ||
          checked_multiply(total, extent[2], &total) != 0 ||
          total < physical ||
          checked_add(plan->face_count, total - physical, &next) != 0)
        goto fail;
      plan->face_count = next;
    }
  }
  plan->cell = calloc(plan->cell_count, sizeof(*plan->cell));
  plan->face = calloc(plan->face_count, sizeof(*plan->face));
  if (plan->cell == NULL || plan->face == NULL) goto fail;
  size_t nc = 0, nf = 0;
  for (int receiver = 0; receiver < GAMERA_CS_FACE_COUNT; ++receiver) {
    plan->cell_begin[receiver] = nc;
    const size_t *extent = grid[receiver].cell_extent;
    for (size_t i = ghost; i < ghost + active[0]; ++i)
      for (size_t j = 0; j < extent[1]; ++j)
        for (size_t k = 0; k < extent[2]; ++k) {
          if (j >= ghost && j < ghost + active[1] &&
              k >= ghost && k < ghost + active[2]) continue;
          if (prepare_plan_entry(&plan->cell[nc++], grid, active, ghost,
                                 receiver, 0, i, j, k, 0) != 0) goto fail;
        }
    plan->cell_end[receiver] = nc;
  }
  for (int receiver = 0; receiver < GAMERA_CS_FACE_COUNT; ++receiver) {
    plan->face_begin[receiver] = nf;
    for (int direction = 0; direction < 3; ++direction) {
      const size_t *extent = grid[receiver].face[direction].extent;
      for (size_t i = 0; i < extent[0]; ++i)
        for (size_t j = 0; j < extent[1]; ++j)
          for (size_t k = 0; k < extent[2]; ++k) {
            const size_t coord[3] = {i, j, k};
            int physical = 1;
            for (int d = 0; d < 3; ++d) {
              const size_t upper = ghost + active[d] + (direction == d);
              if (coord[d] < ghost || coord[d] >= upper) physical = 0;
            }
            if (physical) continue;
            if (prepare_plan_entry(&plan->face[nf++], grid, active, ghost,
                                   receiver, direction, i, j, k, 1) != 0)
              goto fail;
          }
    }
    plan->face_end[receiver] = nf;
  }
  if (nc != plan->cell_count || nf != plan->face_count) goto fail;
  size_t angular_cells;
  if (checked_multiply(active[1], active[2], &angular_cells) != 0)
    goto fail;
  for (int receiver = 0; receiver < GAMERA_CS_FACE_COUNT; ++receiver)
    for (int donor = 0; donor < GAMERA_CS_FACE_COUNT; ++donor) {
      plan->donor_mask[receiver][donor] = calloc(angular_cells, 1);
      if (plan->donor_mask[receiver][donor] == NULL) goto fail;
    }
  for (size_t n = 0; n < nc; ++n)
    mark_donor_stencil(plan, &plan->cell[n], 1);
  for (size_t n = 0; n < nf; ++n)
    mark_donor_stencil(plan, &plan->face[n], 0);
  for (int receiver = 0; receiver < GAMERA_CS_FACE_COUNT; ++receiver)
    for (int donor = 0; donor < GAMERA_CS_FACE_COUNT; ++donor)
      for (size_t n = 0; n < angular_cells; ++n)
        plan->donor_count[receiver][donor] +=
            plan->donor_mask[receiver][donor][n] != 0;
  *out = plan;
  return 0;
fail:
  gamera_cs_halo_plan_destroy(plan);
  return -1;
}

static int plan_matches_grid(const gamera_cs_halo_plan *plan,
                             const gamera_no_grid grid[GAMERA_CS_FACE_COUNT]) {
  if (plan == NULL || grid == NULL) return 0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    if (plan->cell_geometry[face] != grid[face].cell) return 0;
    for (int d = 0; d < 3; ++d)
      if (plan->face_geometry[face][d] != grid[face].face[d].value) return 0;
  }
  return 1;
}

/* Each planned entry writes one distinct ghost target and reads only donor
 * active cells. Preserve a serial A/B path for regression and profiling. */
#ifdef _OPENMP
static int parallel_planned_halo_enabled(void) {
  const char *value = getenv("GAMERA_CS_PARALLEL_HALO");
  return omp_get_max_threads() > 1 &&
         (value == NULL || strcmp(value, "0") != 0);
}
#endif

static int fill_tangential_cell_halos_planned_filtered(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT], int receiver) {
  if (!plan_matches_grid(plan, grid) || storage == NULL) return -1;
  const size_t begin = receiver < 0 ? 0 : plan->cell_begin[receiver];
  const size_t end = receiver < 0 ? plan->cell_count : plan->cell_end[receiver];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_planned_halo_enabled())
#endif
  for (size_t n = begin; n < end; ++n) {
    const halo_plan_entry *e = &plan->cell[n];
    const gamera_no_storage *donor = &storage[e->donor];
    double current[GAMERA_NO_FLUX_COUNT] = {0.0};
    double old[GAMERA_NO_FLUX_COUNT] = {0.0};
    gamera_no_vec3 current_b = {{0.0, 0.0, 0.0}};
    gamera_no_vec3 old_b = {{0.0, 0.0, 0.0}};
    for (int p = 0; p < 8; ++p)
      for (int q = 0; q < 8; ++q) {
        const size_t source = e->donor_row[p] + (size_t)q;
        const double weight = e->wx[p] * e->wy[q];
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          current[m] += weight * donor->conserved[
              source * GAMERA_NO_FLUX_COUNT + (size_t)m];
          old[m] += weight * donor->old_conserved[
              source * GAMERA_NO_FLUX_COUNT + (size_t)m];
        }
        for (int d = 0; d < 3; ++d) {
          current_b.value[d] += weight * donor->cell_magnetic[source].value[d];
          old_b.value[d] += weight * donor->old_cell_magnetic[source].value[d];
        }
      }
    if (!gamera_cs_halo_valid_state(current) ||
        !gamera_cs_halo_valid_state(old)) {
      memcpy(current, &donor->conserved[e->nearest * GAMERA_NO_FLUX_COUNT],
             sizeof(current));
      memcpy(old, &donor->old_conserved[e->nearest * GAMERA_NO_FLUX_COUNT],
             sizeof(old));
      current_b = donor->cell_magnetic[e->nearest];
      old_b = donor->old_cell_magnetic[e->nearest];
    }
    gamera_no_storage *receiver = &storage[e->receiver];
    memcpy(&receiver->conserved[e->target * GAMERA_NO_FLUX_COUNT],
           current, sizeof(current));
    memcpy(&receiver->old_conserved[e->target * GAMERA_NO_FLUX_COUNT],
           old, sizeof(old));
    receiver->cell_magnetic[e->target] = current_b;
    receiver->old_cell_magnetic[e->target] = old_b;
  }
  return 0;
}

int gamera_cs_fill_tangential_cell_halos_planned(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT]) {
  return fill_tangential_cell_halos_planned_filtered(
      plan, grid, storage, -1);
}

int gamera_cs_fill_tangential_cell_halos_planned_face(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT], int receiver) {
  if (receiver < 0 || receiver >= GAMERA_CS_FACE_COUNT) return -1;
  return fill_tangential_cell_halos_planned_filtered(
      plan, grid, storage, receiver);
}

static int fill_magnetic_face_halos_planned_filtered(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT], int receiver) {
  if (!plan_matches_grid(plan, grid) || storage == NULL) return -1;
  const size_t begin = receiver < 0 ? 0 : plan->face_begin[receiver];
  const size_t end = receiver < 0 ? plan->face_count : plan->face_end[receiver];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_planned_halo_enabled())
#endif
  for (size_t n = begin; n < end; ++n) {
    const halo_plan_entry *e = &plan->face[n];
    const gamera_no_storage *donor = &storage[e->donor];
    gamera_no_vec3 current = {{0.0, 0.0, 0.0}};
    gamera_no_vec3 old = current;
    for (int p = 0; p < 8; ++p)
      for (int q = 0; q < 8; ++q) {
        const size_t source = e->donor_row[p] + (size_t)q;
        const double weight = e->wx[p] * e->wy[q];
        for (int d = 0; d < 3; ++d) {
          current.value[d] += weight * donor->cell_magnetic[source].value[d];
          old.value[d] += weight * donor->old_cell_magnetic[source].value[d];
        }
      }
    const gamera_no_face_geometry *geometry =
        &grid[e->receiver].face[e->direction].value[e->target];
    double current_flux = 0.0, old_flux = 0.0;
    for (int d = 0; d < 3; ++d) {
      current_flux += current.value[d] * geometry->area_vector.value[d];
      old_flux += old.value[d] * geometry->area_vector.value[d];
    }
    storage[e->receiver].face_flux[e->direction][e->target] = current_flux;
    storage[e->receiver].old_face_flux[e->direction][e->target] = old_flux;
  }
  return 0;
}

int gamera_cs_fill_magnetic_face_halos_planned(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT]) {
  return fill_magnetic_face_halos_planned_filtered(
      plan, grid, storage, -1);
}

int gamera_cs_fill_magnetic_face_halos_planned_face(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT], int receiver) {
  if (receiver < 0 || receiver >= GAMERA_CS_FACE_COUNT) return -1;
  return fill_magnetic_face_halos_planned_filtered(
      plan, grid, storage, receiver);
}
