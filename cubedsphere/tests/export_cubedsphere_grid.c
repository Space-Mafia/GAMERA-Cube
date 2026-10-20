#include "cubedsphere_mesh.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "usage: %s grid.csv [radial_vertex_index]\n", argv[0]);
    return 1;
  }
  const size_t ghost = 4;
  const size_t active[3] = {72, 32, 32};
  const size_t extent[3] = {active[0] + 2 * ghost + 1,
                            active[1] + 2 * ghost + 1,
                            active[2] + 2 * ghost + 1};
  char *end = NULL;
  const unsigned long requested_i = argc == 3 ? strtoul(argv[2], &end, 10) : 23;
  if (argc == 3 && (end == argv[2] || *end != '\0' || requested_i > active[0])) {
    fprintf(stderr, "radial_vertex_index must be in [0, %zu]\n", active[0]);
    return 1;
  }
  const size_t surface_i = ghost + (size_t)requested_i;
  gamera_no_radial_map map;
  if (gamera_no_radial_map_init(
          &map, GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
          2.5, 200.0, 3.0) != 0) return 1;
  FILE *output = fopen(argv[1], "w");
  if (output == NULL) return 1;
  fputs("kind,face,i,j,k,x,y,z,r\n", output);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(extent),
                                      sizeof(*vertices));
    if (vertices == NULL ||
        gamera_cs_generate_vertices_radial_map(
            face, active, ghost, &map, vertices) != 0) {
      free(vertices);
      fclose(output);
      return 1;
    }
    for (size_t j = ghost; j <= ghost + active[1]; ++j) {
      for (size_t k = ghost; k <= ghost + active[2]; ++k) {
        const gamera_no_vec3 point = vertices[gamera_no_index3(
            extent, surface_i, j, k)];
        const double r = sqrt(point.value[0] * point.value[0] +
                              point.value[1] * point.value[1] +
                              point.value[2] * point.value[2]);
        fprintf(output, "surface,%d,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g\n",
                face, surface_i - ghost, j - ghost, k - ghost,
                point.value[0], point.value[1], point.value[2], r);
      }
    }
    free(vertices);
  }
  for (size_t i = 0; i <= active[0]; ++i) {
    double radius;
    if (gamera_no_radial_map_forward(
            &map, (double)i / (double)active[0], &radius) != 0) {
      fclose(output);
      return 1;
    }
    fprintf(output, "radial,-1,%zu,-1,-1,0,0,0,%.17g\n", i, radius);
  }
  return fclose(output) == 0 ? 0 : 1;
}
