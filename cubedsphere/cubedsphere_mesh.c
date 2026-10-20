#include "cubedsphere_mesh.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* The signs are chosen so every local (r,a,b) Jacobian is positive. */
static const int frame[GAMERA_CS_FACE_COUNT][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
    {{0, 0, -1}, {1, 0, 0}, {0, -1, 0}}};

static double dot(gamera_no_vec3 a, const int b[3]) {
  return a.value[0] * (double)b[0] + a.value[1] * (double)b[1] +
         a.value[2] * (double)b[2];
}

int gamera_cs_face_frame(int face, gamera_no_vec3 *normal,
                         gamera_no_vec3 *tangent_a,
                         gamera_no_vec3 *tangent_b) {
  if (face < 0 || face >= GAMERA_CS_FACE_COUNT || normal == NULL ||
      tangent_a == NULL || tangent_b == NULL) {
    return -1;
  }
  for (int d = 0; d < 3; ++d) {
    normal->value[d] = (double)frame[face][0][d];
    tangent_a->value[d] = (double)frame[face][1][d];
    tangent_b->value[d] = (double)frame[face][2][d];
  }
  return 0;
}

int gamera_cs_vertex_topology_key(int face,
    const size_t global_active[3], const size_t global_vertex[3],
    int64_t key[4]) {
  if (face < 0 || face >= GAMERA_CS_FACE_COUNT ||
      global_active == NULL || global_vertex == NULL || key == NULL ||
      global_active[1] == 0 || global_active[1] != global_active[2] ||
      global_active[0] > (size_t)INT64_MAX ||
      global_active[1] > (size_t)INT64_MAX / 4U)
    return -1;
  for (int axis = 0; axis < 3; ++axis)
    if (global_vertex[axis] > global_active[axis]) return -1;
  const int64_t n = (int64_t)global_active[1];
  const int64_t a = 2 * (int64_t)global_vertex[1] - n;
  const int64_t b = 2 * (int64_t)global_vertex[2] - n;
  key[0] = (int64_t)global_vertex[0];
  for (int d = 0; d < 3; ++d)
    key[d + 1] = (int64_t)frame[face][0][d] * n +
                 (int64_t)frame[face][1][d] * a +
                 (int64_t)frame[face][2][d] * b;
  return 0;
}

int gamera_cs_direction(int face, double a, double b,
                        gamera_no_vec3 *direction) {
  const double quarter_pi = 0.25 * acos(-1.0);
  if (face < 0 || face >= GAMERA_CS_FACE_COUNT || direction == NULL ||
      !isfinite(a) || !isfinite(b) || fabs(a) >= 2.0 * quarter_pi ||
      fabs(b) >= 2.0 * quarter_pi) {
    return -1;
  }
  const double u = tan(a);
  const double v = tan(b);
  const double inv_length = 1.0 / sqrt(1.0 + u * u + v * v);
  for (int d = 0; d < 3; ++d) {
    direction->value[d] = inv_length *
        ((double)frame[face][0][d] + u * (double)frame[face][1][d] +
         v * (double)frame[face][2][d]);
  }
  return 0;
}

int gamera_cs_inverse(gamera_no_vec3 direction, int *face,
                      double *a, double *b) {
  if (face == NULL || a == NULL || b == NULL) {
    return -1;
  }
  for (int d = 0; d < 3; ++d) {
    if (!isfinite(direction.value[d])) {
      return -1;
    }
  }
  int selected = -1;
  double dominant = 0.0;
  for (int f = 0; f < GAMERA_CS_FACE_COUNT; ++f) {
    const double projection = dot(direction, frame[f][0]);
    if (projection > dominant) {
      dominant = projection;
      selected = f;
    }
  }
  if (selected < 0 || !isfinite(dominant)) {
    return -1;
  }
  *face = selected;
  *a = atan2(dot(direction, frame[selected][1]), dominant);
  *b = atan2(dot(direction, frame[selected][2]), dominant);
  return 0;
}

int gamera_cs_edge_neighbor(int face, int edge,
                            gamera_cs_neighbor *neighbor) {
  if (face < 0 || face >= GAMERA_CS_FACE_COUNT || edge < 0 ||
      edge >= GAMERA_CS_EDGE_COUNT || neighbor == NULL) {
    return -1;
  }
  const double quarter_pi = 0.25 * acos(-1.0);
  const double epsilon = 1.0e-7;
  double a = 0.0;
  double b = 0.0;
  if (edge == GAMERA_CS_A_LOWER || edge == GAMERA_CS_A_UPPER) {
    a = (edge == GAMERA_CS_A_LOWER ? -1.0 : 1.0) *
        (quarter_pi + epsilon);
  } else {
    b = (edge == GAMERA_CS_B_LOWER ? -1.0 : 1.0) *
        (quarter_pi + epsilon);
  }
  gamera_no_vec3 out;
  int target;
  double target_a, target_b;
  if (gamera_cs_direction(face, a, b, &out) != 0 ||
      gamera_cs_inverse(out, &target, &target_a, &target_b) != 0 ||
      target == face) {
    return -1;
  }
  const int donor_edge = fabs(target_a) > fabs(target_b)
                             ? (target_a < 0.0 ? GAMERA_CS_A_LOWER
                                               : GAMERA_CS_A_UPPER)
                             : (target_b < 0.0 ? GAMERA_CS_B_LOWER
                                               : GAMERA_CS_B_UPPER);
  const double along = 0.2;
  double src_a[2] = {0.0, 0.0};
  double src_b[2] = {0.0, 0.0};
  for (int n = 0; n < 2; ++n) {
    const double position = (n == 0 ? -along : along);
    if (edge == GAMERA_CS_A_LOWER || edge == GAMERA_CS_A_UPPER) {
      src_a[n] = edge == GAMERA_CS_A_LOWER ? -quarter_pi : quarter_pi;
      src_b[n] = position;
    } else {
      src_a[n] = position;
      src_b[n] = edge == GAMERA_CS_B_LOWER ? -quarter_pi : quarter_pi;
    }
  }
  double donor_along[2];
  for (int n = 0; n < 2; ++n) {
    if (gamera_cs_direction(face, src_a[n], src_b[n], &out) != 0) {
      return -1;
    }
    const double projection = dot(out, frame[target][0]);
    const double donor_a = atan2(dot(out, frame[target][1]), projection);
    const double donor_b = atan2(dot(out, frame[target][2]), projection);
    donor_along[n] = donor_edge < GAMERA_CS_B_LOWER ? donor_b : donor_a;
  }
  neighbor->face = target;
  neighbor->edge = donor_edge;
  neighbor->reversed = donor_along[1] < donor_along[0];
  return 0;
}

static int generate_vertices_window(
    int face, const size_t global_active[3],
    const size_t global_start[3], const size_t local_active[3],
    size_t ghost, double r_inner, double r_outer,
    const gamera_no_radial_map *map, gamera_no_vec3 *vertices) {
  if (face < 0 || face >= GAMERA_CS_FACE_COUNT ||
      global_active == NULL || global_start == NULL ||
      local_active == NULL || vertices == NULL || ghost < 4 ||
      global_active[0] == 0 || global_active[1] < 8 ||
      global_active[2] < 8 || global_active[1] != global_active[2] ||
      local_active[0] == 0 || local_active[1] < 8 || local_active[2] < 8 ||
      !isfinite(r_inner) || !isfinite(r_outer) ||
      r_inner <= 0.0 || r_outer <= r_inner ||
      (map == NULL &&
       r_inner <= (double)ghost *
           (r_outer - r_inner) / (double)global_active[0])) {
    return -1;
  }
  if (ghost > (SIZE_MAX - 1U) / 2U) return -1;
  for (int axis = 0; axis < 3; ++axis)
    if (global_start[axis] > global_active[axis] ||
        local_active[axis] > global_active[axis] - global_start[axis] ||
        local_active[axis] > SIZE_MAX - 1U - 2U * ghost)
      return -1;
  const double quarter_pi = 0.25 * acos(-1.0);
  const double da = 2.0 * quarter_pi / (double)global_active[1];
  const double db = 2.0 * quarter_pi / (double)global_active[2];
  if (quarter_pi + (double)ghost * fmax(da, db) >= 2.0 * quarter_pi) {
    return -1;
  }
  const size_t vertex_extent[3] = {local_active[0] + 2U * ghost + 1U,
                                   local_active[1] + 2U * ghost + 1U,
                                   local_active[2] + 2U * ghost + 1U};
  const double dr = (r_outer - r_inner) / (double)global_active[0];
  for (size_t i = 0; i < vertex_extent[0]; ++i) {
    const double logical_i =
        (double)global_start[0] + (double)i - (double)ghost;
    double r = r_inner + logical_i * dr;
    if (map != NULL &&
        gamera_no_radial_map_forward(
            map, logical_i / (double)global_active[0], &r) != 0)
      return -1;
    if (!(r > 0.0) || !isfinite(r)) return -1;
    for (size_t j = 0; j < vertex_extent[1]; ++j) {
      const double a = -quarter_pi +
          ((double)global_start[1] + (double)j - (double)ghost) * da;
      for (size_t k = 0; k < vertex_extent[2]; ++k) {
        const double b = -quarter_pi +
            ((double)global_start[2] + (double)k - (double)ghost) * db;
        gamera_no_vec3 direction;
        if (gamera_cs_direction(face, a, b, &direction) != 0) {
          return -1;
        }
        const size_t index = gamera_no_index3(vertex_extent, i, j, k);
        for (int d = 0; d < 3; ++d) {
          vertices[index].value[d] = r * direction.value[d];
        }
      }
    }
  }
  return 0;
}

int gamera_cs_generate_vertices(int face, const size_t active[3],
                                size_t ghost, double r_inner, double r_outer,
                                gamera_no_vec3 *vertices) {
  const size_t start[3] = {0, 0, 0};
  return generate_vertices_window(face, active, start, active, ghost,
                                  r_inner, r_outer, NULL, vertices);
}

int gamera_cs_generate_vertices_radial_map(
    int face, const size_t active[3], size_t ghost,
    const gamera_no_radial_map *map, gamera_no_vec3 *vertices) {
  if (map == NULL) return -1;
  const size_t start[3] = {0, 0, 0};
  return generate_vertices_window(face, active, start, active, ghost,
                                  map->inner_radius, map->outer_radius,
                                  map, vertices);
}

int gamera_cs_generate_vertices_radial_map_tile(
    int face, const size_t global_active[3],
    const size_t global_start[3], const size_t local_active[3],
    size_t ghost, const gamera_no_radial_map *map,
    gamera_no_vec3 *vertices) {
  if (map == NULL) return -1;
  return generate_vertices_window(face, global_active, global_start,
                                  local_active, ghost, map->inner_radius,
                                  map->outer_radius, map, vertices);
}

static int calibrate_face(gamera_no_face_geometry *face) {
  const gamera_no_vec3 area = face->area_vector;
  const double magnitude = sqrt(area.value[0] * area.value[0] +
                                area.value[1] * area.value[1] +
                                area.value[2] * area.value[2]);
  if (!isfinite(magnitude) || magnitude <= 0.0) return -1;
  gamera_no_vec3 normal;
  for (int d = 0; d < 3; ++d) normal.value[d] = area.value[d] / magnitude;
  const gamera_no_vec3 old_t2 = face->tangent2;
  double projection = 0.0;
  for (int d = 0; d < 3; ++d) projection += old_t2.value[d] * normal.value[d];
  gamera_no_vec3 t2;
  double t2_square = 0.0;
  for (int d = 0; d < 3; ++d) {
    t2.value[d] = old_t2.value[d] - projection * normal.value[d];
    t2_square += t2.value[d] * t2.value[d];
  }
  if (!isfinite(t2_square) || t2_square <= 0.0) return -1;
  for (int d = 0; d < 3; ++d) t2.value[d] /= sqrt(t2_square);
  const gamera_no_vec3 t1 = {{
      t2.value[1] * normal.value[2] - t2.value[2] * normal.value[1],
      t2.value[2] * normal.value[0] - t2.value[0] * normal.value[2],
      t2.value[0] * normal.value[1] - t2.value[1] * normal.value[0]}};
  face->area = magnitude;
  face->normal = normal;
  face->tangent1 = t1;
  face->tangent2 = t2;
  return 0;
}

int gamera_cs_calibrate_face_frames(gamera_no_grid *grid) {
  if (grid == NULL || grid->cell == NULL) return -1;
  const size_t cells = gamera_no_element_count3(grid->cell_extent);
  for (size_t c = 0; c < cells; ++c) {
    for (int d = 0; d < 3; ++d) {
      for (int side = 0; side < 2; ++side) {
        if (calibrate_face(&grid->cell[c].face[d][side]) != 0) return -1;
      }
    }
  }
  for (int d = 0; d < 3; ++d) {
    const size_t count = gamera_no_element_count3(grid->face[d].extent);
    for (size_t f = 0; f < count; ++f) {
      if (calibrate_face(&grid->face[d].value[f]) != 0) return -1;
    }
  }
  return 0;
}
