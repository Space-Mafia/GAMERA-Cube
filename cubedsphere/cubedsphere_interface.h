#ifndef GAMERA_CUBEDSPHERE_INTERFACE_H
#define GAMERA_CUBEDSPHERE_INTERFACE_H

#include "cubedsphere_mesh.h"
#include "nonorthogonal_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reconcile the single physical face shared by two cube patches. */
int gamera_cs_sync_magnetic_face_flux(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost);
int gamera_cs_sync_mhd_face_fluxes(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost);

/* Average signed line-integrated EMF on every coincident physical edge.
 * The same operation handles two-face seam edges and three-face cube corners.
 */
int gamera_cs_sync_edge_emf(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost);

/* Cache only immutable geometric edge identities and their original sort
 * order. Applying the plan performs the same signed EMF average as above
 * using the current step's values. */
typedef struct gamera_cs_edge_sync_plan gamera_cs_edge_sync_plan;
int gamera_cs_edge_sync_plan_create(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    gamera_cs_edge_sync_plan **out);
void gamera_cs_edge_sync_plan_destroy(gamera_cs_edge_sync_plan *plan);
int gamera_cs_sync_edge_emf_planned(
    const gamera_cs_edge_sync_plan *plan,
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT]);

#ifdef __cplusplus
}
#endif

#endif
