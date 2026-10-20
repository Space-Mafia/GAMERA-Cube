#include "cubedsphere_mesh.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double radius(gamera_no_vec3 point) {
  return sqrt(point.value[0] * point.value[0] +
              point.value[1] * point.value[1] +
              point.value[2] * point.value[2]);
}

int main(void) {
  const size_t ghost = 4;
  const size_t radial_cases[] = {72, 48, 36, 128, 144};
  for (size_t test = 0; test < sizeof(radial_cases) / sizeof(radial_cases[0]);
       ++test) {
  const size_t angular = test >= 3 ? 32U : 16U;
  const size_t active[3] = {radial_cases[test], angular, angular};
  const size_t extent[3] = {active[0] + 2 * ghost + 1,
                            active[1] + 2 * ghost + 1,
                            active[2] + 2 * ghost + 1};
  const size_t count = gamera_no_element_count3(extent);
  const size_t radial_index[3] = {ghost, ghost + 23, ghost + 72};
  const double expected[3] = {2.5, 14.0, 200.0};
  gamera_no_radial_map map;
  if ((active[0] == 48U
           ? gamera_no_radial_map_init_earth_v4_resolution(
                 &map, 2.5, 200.0, 48U, 23U)
           : gamera_no_radial_map_init(
                 &map, GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
                 2.5, 200.0, 3.0)) != 0) return 1;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(count, sizeof(*vertices));
    if (vertices == NULL ||
        gamera_cs_generate_vertices_radial_map(
            face, active, ghost, &map, vertices) != 0) {
      fprintf(stderr, "Earth v4 radial mesh failed on face %d\n", face);
      free(vertices);
      return 1;
    }
    const int exact_samples = test == 0 ? 3 : (test == 1 ? 2 : 1);
    for (int sample = 0; sample < exact_samples; ++sample) {
      const size_t index = gamera_no_index3(
          extent, radial_index[sample], ghost + 8, ghost + 8);
      if (fabs(radius(vertices[index]) - expected[sample]) > 2e-12) {
        fprintf(stderr, "Earth v4 radius mismatch face %d sample %d\n",
                face, sample);
        free(vertices);
        return 1;
      }
    }
    const size_t outer = gamera_no_index3(
        extent, ghost + active[0], ghost + 8, ghost + 8);
    if (fabs(radius(vertices[outer]) - 200.0) > 2e-12) {
      fprintf(stderr, "Earth v4 outer radius mismatch face %d\n", face);
      free(vertices);
      return 1;
    }
    if (active[0] == 48U) {
      for (size_t radial = 0; radial <= 23U; ++radial) {
        const size_t index = gamera_no_index3(
            extent, ghost + radial, ghost + 8, ghost + 8);
        if (fabs(radius(vertices[index]) - (2.5 + 0.5 * radial)) > 2e-12) {
          fprintf(stderr, "48-cell inner 0.5 RE spacing failed face %d\n",
                  face);
          free(vertices);
          return 1;
        }
      }
    }
    double previous = 0.0;
    for (size_t i = ghost; i <= ghost + active[0]; ++i) {
      const double current = radius(vertices[gamera_no_index3(
          extent, i, ghost + 8, ghost + 8)]);
      if (i > ghost && !(current > previous)) {
        fprintf(stderr, "Earth v4 nonmonotone radial mesh face %d\n", face);
        free(vertices);
        return 1;
      }
      previous = current;
    }
    if (test == 2) {
      const size_t before_knot = (size_t)((double)active[0] * 23.0 / 72.0);
      const double before = radius(vertices[gamera_no_index3(
          extent, ghost + before_knot, ghost + 8, ghost + 8)]);
      const double after = radius(vertices[gamera_no_index3(
          extent, ghost + before_knot + 1, ghost + 8, ghost + 8)]);
      if (!(before < 14.0 && after > 14.0)) {
        fprintf(stderr, "Earth v4 %zu-cell knot bracketing failed face %d\n",
                active[0], face);
        free(vertices);
        return 1;
      }
    }
    const size_t inner_ghost = gamera_no_index3(extent, 0, ghost + 8,
                                                 ghost + 8);
    if (!(radius(vertices[inner_ghost]) > 0.0) ||
        !(radius(vertices[inner_ghost]) < 2.5)) {
      fprintf(stderr, "Earth v4 inner ghost radius failed face %d\n", face);
      free(vertices);
      return 1;
    }
    if (test >= 3) {
      const size_t cell_extent[3] = {
          active[0] + 2 * ghost, active[1] + 2 * ghost,
          active[2] + 2 * ghost};
      gamera_no_grid grid = {0};
      if (gamera_no_grid_create(cell_extent, vertices, &grid) != 0 ||
          gamera_cs_calibrate_face_frames(&grid) != 0) {
        fprintf(stderr, "Earth v4 %zu x %zu geometry failed face %d\n",
                active[0], angular, face);
        gamera_no_grid_destroy(&grid);
        free(vertices);
        return 1;
      }
      gamera_no_grid_destroy(&grid);
    }
    free(vertices);
  }
  }
  puts("six-face Earth v4 radial maps: 72/48/36 x16 and 128/144 x32; "
       "positive inner ghosts and 200 RE outer face");
  return 0;
}
