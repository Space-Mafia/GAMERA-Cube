#ifndef GAMERA_CUBEDSPHERE_STEP_H
#define GAMERA_CUBEDSPHERE_STEP_H

#include "cubedsphere_mesh.h"
#include "cubedsphere_halo.h"
#include "nonorthogonal_advance.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Serial six-block standard-MHD step. Executes existing GAMERA Up7/PDM,
 * kinetic flux and CT kernels on each block, with conservative face-flux
 * reconciliation and a single line EMF at every seam/cube corner. No MI,
 * multifluid, sources, or planetary boundary conditions are implied. */
int gamera_cs_advance_six(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options);

/* Optional radial ghost override, applied after the standard radial outflow
 * and tangential/magnetic halo fills but before prediction. It may set only
 * nonphysical radial ghost cells/faces; active boundary faces remain owned by
 * the CT update. Both current and old time levels must be filled. */
typedef int (*gamera_cs_radial_boundary_function)(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, double current_time,
    double previous_time, void *context);

int gamera_cs_advance_six_with_boundary(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context);

/* Same six-block update with the existing non-orthogonal background-field
 * data supplied separately for each face. Pass NULL for a zero background.
 * This retains the GAMERA background stress, Boris, and CT kernels; only
 * cube-face synchronization is performed here. */
int gamera_cs_advance_six_with_background(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context);

/* Identical update with precomputed, per-geometry-point cube halo stencils.
 * The caller owns the plan for the lifetime of its fixed grid. */
int gamera_cs_advance_six_with_background_planned(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context);

#ifdef __cplusplus
}
#endif

#endif
