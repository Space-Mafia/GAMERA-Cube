#ifndef GAMERA_CUBEDSPHERE_HALO_H
#define GAMERA_CUBEDSPHERE_HALO_H

#include "cubedsphere_mesh.h"
#include "cubedsphere_decomposition.h"
#include "nonorthogonal_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gamera_cs_halo_plan gamera_cs_halo_plan;

/* Sparse routing for the accepted 8×8 donor interpolation. Coordinates and
 * Lagrange weights are computed on the GLOBAL face; each donor cell is then
 * mapped to its owning face/radial/angular tile. No MHD state is copied. */
typedef struct {
  int donor_face;
  size_t start[2];
  size_t nearest[2];
  double weight[2][8];
  int rank[8][8];
  size_t local_cell[8][8][3];
} gamera_cs_tiled_donor_stencil;

int gamera_cs_tiled_donor_stencil_create(
    gamera_no_vec3 point, const size_t global_active[3],
    const int proc_dims[3], size_t global_radial_cell,
    int magnetic_face, gamera_cs_tiled_donor_stencil *result);

/* Same positivity check used by the accepted full-face halo fallback. */
int gamera_cs_halo_valid_state(const double state[GAMERA_NO_FLUX_COUNT]);

/* Optional immutable geometry plan. Each actual cell/face ghost location gets
 * its own donor stencil, including its radial coordinate; only the geometry
 * is cached. The donor MHD state is still read afresh on every step. */
int gamera_cs_halo_plan_create(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, gamera_cs_halo_plan **out);
void gamera_cs_halo_plan_destroy(gamera_cs_halo_plan *plan);
/* Immutable donor mask indexed as j*active[2]+k, covering the existing
 * 8x8 ghost stencils and their positivity fallback. */
const unsigned char *gamera_cs_halo_plan_donor_mask(
    const gamera_cs_halo_plan *plan, int receiver, int donor,
    size_t *required_count);
int gamera_cs_fill_tangential_cell_halos_planned(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT]);
int gamera_cs_fill_magnetic_face_halos_planned(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT]);
/* Same immutable donor plan, but write only the receiver owned by this rank. */
int gamera_cs_fill_tangential_cell_halos_planned_face(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT], int receiver);
int gamera_cs_fill_magnetic_face_halos_planned_face(
    const gamera_cs_halo_plan *plan,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT], int receiver);

/*
 * Fill all tangential ghost cells from the physical owning face. The source
 * is sampled with an 8 x 8 tensor-product Lagrange stencil of donor active
 * cell centers, never by a same-index copy. This keeps GAMERA's eight-cell
 * Up7/PDM reconstruction available at every seam. Both time levels and
 * Cartesian B are exchanged; radial ghosts remain a physical-BC task.
 * Every face must have the same active dimensions and four or more ghosts.
 */
int gamera_cs_fill_tangential_cell_halos(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost);

/* Minimal outflow radial boundary for standard MHD tests. Magnetosphere
 * applications will replace this with their physical inner/outer BC. */
int gamera_cs_fill_radial_outflow_cells(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost);
int gamera_cs_fill_radial_outflow_cells_face(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, int receiver);

/* Populate nonphysical ghost magnetic faces by projecting donor Cartesian B
 * onto their local oriented area vectors. Active shared faces are untouched. */
int gamera_cs_fill_magnetic_face_halos(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost);

#ifdef __cplusplus
}
#endif

#endif
