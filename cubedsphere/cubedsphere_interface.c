#include "cubedsphere_interface.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int sync_faces(gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
                      const size_t active[3], size_t ghost, int magnetic) {
  if (storage == NULL || active == NULL || ghost < 4 || active[1] == 0 ||
      active[1] != active[2]) return -1;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
      gamera_cs_neighbor neighbor;
      if (gamera_cs_edge_neighbor(face, edge, &neighbor) != 0) return -1;
      if (face >= neighbor.face) continue;
      const int src_axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
      const int dst_axis = neighbor.edge < GAMERA_CS_B_LOWER ? 1 : 2;
      const size_t src_boundary = edge % 2 == 0 ? ghost :
                                  ghost + active[src_axis];
      const size_t dst_boundary = neighbor.edge % 2 == 0 ? ghost :
                                  ghost + active[dst_axis];
      const double orientation = edge % 2 == neighbor.edge % 2 ? -1.0 : 1.0;
      for (size_t i = ghost; i < ghost + active[0]; ++i) {
        for (size_t along = 0; along < active[1]; ++along) {
          const size_t donor_along = neighbor.reversed ?
              active[1] - 1U - along : along;
          size_t source[3] = {i, ghost + along, ghost + along};
          size_t target[3] = {i, ghost + donor_along, ghost + donor_along};
          source[src_axis] = src_boundary;
          target[dst_axis] = dst_boundary;
          const size_t s = gamera_no_index3(
              storage[face].face_extent[src_axis],
              source[0], source[1], source[2]);
          const size_t t = gamera_no_index3(
              storage[neighbor.face].face_extent[dst_axis],
              target[0], target[1], target[2]);
          if (magnetic) {
            const double common = 0.5 *
                (storage[face].face_flux[src_axis][s] +
                 orientation * storage[neighbor.face].face_flux[dst_axis][t]);
            storage[face].face_flux[src_axis][s] = common;
            storage[neighbor.face].face_flux[dst_axis][t] =
                orientation * common;
            const double old_common = 0.5 *
                (storage[face].old_face_flux[src_axis][s] +
                 orientation * storage[neighbor.face]
                     .old_face_flux[dst_axis][t]);
            storage[face].old_face_flux[src_axis][s] = old_common;
            storage[neighbor.face].old_face_flux[dst_axis][t] =
                orientation * old_common;
          } else {
            for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
              const double common = 0.5 *
                  (storage[face].fluid_face_flux[src_axis][
                       s * GAMERA_NO_FLUX_COUNT + (size_t)m] +
                   orientation * storage[neighbor.face]
                       .fluid_face_flux[dst_axis][
                           t * GAMERA_NO_FLUX_COUNT + (size_t)m]);
              storage[face].fluid_face_flux[src_axis][
                  s * GAMERA_NO_FLUX_COUNT + (size_t)m] = common;
              storage[neighbor.face].fluid_face_flux[dst_axis][
                  t * GAMERA_NO_FLUX_COUNT + (size_t)m] =
                  orientation * common;
            }
            for (int d = 0; d < 3; ++d) {
              const double common = 0.5 *
                  (storage[face].maxwell_face_flux[src_axis][s].value[d] +
                   orientation * storage[neighbor.face]
                       .maxwell_face_flux[dst_axis][t].value[d]);
              storage[face].maxwell_face_flux[src_axis][s].value[d] = common;
              storage[neighbor.face]
                  .maxwell_face_flux[dst_axis][t].value[d] =
                  orientation * common;
            }
          }
        }
      }
    }
  }
  return 0;
}

int gamera_cs_sync_magnetic_face_flux(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost) {
  return sync_faces(storage, active, ghost, 1);
}

int gamera_cs_sync_mhd_face_fluxes(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost) {
  return sync_faces(storage, active, ghost, 0);
}

typedef struct {
  int64_t endpoint[8];
  int face;
  int direction;
  size_t index;
  int sign;
} seam_edge;

struct gamera_cs_edge_sync_plan {
  seam_edge *entries;
  size_t count;
};

static int compare_point(const int64_t a[4], const int64_t b[4]) {
  for (int d = 0; d < 4; ++d) {
    if (a[d] < b[d]) return -1;
    if (a[d] > b[d]) return 1;
  }
  return 0;
}

static int compare_edge(const void *left, const void *right) {
  const seam_edge *a = left;
  const seam_edge *b = right;
  for (int d = 0; d < 8; ++d) {
    if (a->endpoint[d] < b->endpoint[d]) return -1;
    if (a->endpoint[d] > b->endpoint[d]) return 1;
  }
  return 0;
}

static int add_edge(seam_edge *entry, const gamera_no_grid *grid,
                    const size_t active[3], size_t ghost,
                    int face, int direction, size_t i, size_t j, size_t k) {
  size_t start_vertex[3] = {i - ghost, j - ghost, k - ghost};
  size_t end_vertex[3] = {
      start_vertex[0], start_vertex[1], start_vertex[2]};
  ++end_vertex[direction];
  int64_t start_key[4], end_key[4];
  if (gamera_cs_vertex_topology_key(face, active, start_vertex,
          start_key) != 0 ||
      gamera_cs_vertex_topology_key(face, active, end_vertex,
          end_key) != 0) return -1;
  const int orientation = compare_point(start_key, end_key);
  if (orientation == 0) return -1;
  entry->sign = orientation < 0 ? 1 : -1;
  for (int d = 0; d < 4; ++d) {
    entry->endpoint[d] = orientation < 0 ? start_key[d] : end_key[d];
    entry->endpoint[d + 4] = orientation < 0 ? end_key[d] : start_key[d];
  }
  entry->face = face;
  entry->direction = direction;
  entry->index = gamera_no_index3(grid->edge[direction].extent, i, j, k);
  return 0;
}

int gamera_cs_edge_sync_plan_create(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    gamera_cs_edge_sync_plan **out) {
  if (grid == NULL || active == NULL || out == NULL || ghost < 4 ||
      active[1] != active[2]) return -1;
  *out = NULL;
  const size_t n = active[1];
  const size_t capacity = 6U *
      (active[0] * (4U * n + 4U) +
       (active[0] + 1U) * 4U * n);
  gamera_cs_edge_sync_plan *plan = calloc(1, sizeof(*plan));
  if (plan == NULL) return -1;
  plan->entries = malloc(capacity * sizeof(*plan->entries));
  if (plan->entries == NULL) {
    free(plan);
    return -1;
  }
  seam_edge *entries = plan->entries;
  size_t count = 0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int d = 0; d < 3; ++d) {
      const size_t i_upper = ghost + active[0] + (d != 0);
      const size_t j_upper = ghost + active[1] + (d != 1);
      const size_t k_upper = ghost + active[2] + (d != 2);
      for (size_t i = ghost; i < i_upper; ++i) {
        for (size_t j = ghost; j < j_upper; ++j) {
          for (size_t k = ghost; k < k_upper; ++k) {
            const int on_a = j == ghost || j == ghost + n;
            const int on_b = k == ghost || k == ghost + n;
            const int shared = d == 0 ? (on_a || on_b) :
                               d == 1 ? on_b : on_a;
            if (!shared) continue;
            if (count >= capacity || add_edge(&entries[count],
                    &grid[face], active, ghost, face, d, i, j, k) != 0) {
              gamera_cs_edge_sync_plan_destroy(plan);
              return -1;
            }
            ++count;
          }
        }
      }
    }
  }
  qsort(entries, count, sizeof(*entries), compare_edge);
  plan->count = count;
  for (size_t first = 0; first < count;) {
    size_t last = first + 1U;
    while (last < count && compare_edge(&entries[first],
                                        &entries[last]) == 0) ++last;
    if (last - first < 2U || last - first > 3U) {
      gamera_cs_edge_sync_plan_destroy(plan);
      return -1;
    }
    first = last;
  }
  *out = plan;
  return 0;
}

void gamera_cs_edge_sync_plan_destroy(gamera_cs_edge_sync_plan *plan) {
  if (plan == NULL) return;
  free(plan->entries);
  free(plan);
}

int gamera_cs_sync_edge_emf_planned(
    const gamera_cs_edge_sync_plan *plan,
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT]) {
  if (plan == NULL || storage == NULL) return -1;
  const seam_edge *entries = plan->entries;
  const size_t count = plan->count;
  for (size_t first = 0; first < count;) {
    size_t last = first + 1U;
    while (last < count && compare_edge(&entries[first],
                                        &entries[last]) == 0) ++last;
    double common = 0.0;
    for (size_t t = first; t < last; ++t) {
      const seam_edge *entry = &entries[t];
      common += (double)entry->sign * storage[entry->face]
          .edge_emf[entry->direction][entry->index];
    }
    common /= (double)(last - first);
    for (size_t t = first; t < last; ++t) {
      const seam_edge *entry = &entries[t];
      storage[entry->face].edge_emf[entry->direction][entry->index] =
          (double)entry->sign * common;
    }
    first = last;
  }
  return 0;
}

int gamera_cs_sync_edge_emf(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost) {
  if (storage == NULL) return -1;
  gamera_cs_edge_sync_plan *plan = NULL;
  if (gamera_cs_edge_sync_plan_create(grid, active, ghost, &plan) != 0)
    return -1;
  const int status = gamera_cs_sync_edge_emf_planned(plan, storage);
  gamera_cs_edge_sync_plan_destroy(plan);
  return status;
}
