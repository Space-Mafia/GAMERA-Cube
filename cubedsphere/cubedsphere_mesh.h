#ifndef GAMERA_CUBEDSPHERE_MESH_H
#define GAMERA_CUBEDSPHERE_MESH_H

#include "nonorthogonal_grid.h"
#include "nonorthogonal_radial_map.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  GAMERA_CS_POS_X = 0,
  GAMERA_CS_NEG_X,
  GAMERA_CS_POS_Y,
  GAMERA_CS_NEG_Y,
  GAMERA_CS_POS_Z,
  GAMERA_CS_NEG_Z,
  GAMERA_CS_FACE_COUNT
};

enum {
  GAMERA_CS_A_LOWER = 0,
  GAMERA_CS_A_UPPER,
  GAMERA_CS_B_LOWER,
  GAMERA_CS_B_UPPER,
  GAMERA_CS_EDGE_COUNT
};

typedef struct {
  int face;
  int edge;
  int reversed;
} gamera_cs_neighbor;

/* Right-handed face frames: tangent_a cross tangent_b = outward normal. */
int gamera_cs_face_frame(int face, gamera_no_vec3 *normal,
                         gamera_no_vec3 *tangent_a,
                         gamera_no_vec3 *tangent_b);

/* Equiangular gnomonic map. a,b may extend into a four-cell ghost region. */
int gamera_cs_direction(int face, double a, double b,
                        gamera_no_vec3 *direction);

/* Identify the physical face of a direction and its two face coordinates. */
int gamera_cs_inverse(gamera_no_vec3 direction, int *face,
                      double *a, double *b);

/* Return exact topological adjacency, including edge-orientation reversal. */
int gamera_cs_edge_neighbor(int face, int edge,
                            gamera_cs_neighbor *neighbor);

/* Exact identity of a vertex shared by two or three cube faces.  The first
 * component is the global radial vertex index; the others are integer cube
 * coordinates derived from the existing right-handed face frames.  Unlike
 * rounded Cartesian positions, this is independent of radial resolution. */
int gamera_cs_vertex_topology_key(int face,
    const size_t global_active[3], const size_t global_vertex[3],
    int64_t key[4]);

/*
 * Generate vertices for a six-block shell patch. All six faces use identical
 * nr x na x nb active sizes and the same radial map. The returned array has
 * (nr+2*ghost+1) x (na+2*ghost+1) x (nb+2*ghost+1) vertices. Four ghosts
 * are required by GAMERA's eight-cell Up7/PDM reconstruction. These are
 * geometric extensions only; state/flux halo exchange is a separate step.
 */
int gamera_cs_generate_vertices(int face, const size_t active[3],
                                size_t ghost, double r_inner, double r_outer,
                                gamera_no_vec3 *vertices);

/* The accepted Earth radial map (including positive-radius ghost extension)
 * is evaluated at the same logical radial fractions on every cube face. */
int gamera_cs_generate_vertices_radial_map(
    int face, const size_t active[3], size_t ghost,
    const gamera_no_radial_map *map, gamera_no_vec3 *vertices);

/* The same accepted face map, restricted to a Cartesian MPI subdomain.
 * global_start is measured in active cells on the full face.  Interior
 * subdomain ghost vertices therefore coincide with neighboring active
 * vertices; physical radial and cube-face extensions remain unchanged. */
int gamera_cs_generate_vertices_radial_map_tile(
    int face, const size_t global_active[3],
    const size_t global_start[3], const size_t local_active[3],
    size_t ghost, const gamera_no_radial_map *map,
    gamera_no_vec3 *vertices);

/* Use the integrated area vector as the normal/area seen by kinetic fluxes.
 * This preserves a uniform Cartesian state on warped cubed-sphere faces. */
int gamera_cs_calibrate_face_frames(gamera_no_grid *grid);

#ifdef __cplusplus
}
#endif

#endif
