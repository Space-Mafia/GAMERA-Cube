#ifndef GAMERA_CUBEDSPHERE_MPI_STEP_H
#define GAMERA_CUBEDSPHERE_MPI_STEP_H

#include "cubedsphere_mpi.h"
#include "cubedsphere_mpi_tile.h"
#include "cubedsphere_step.h"

typedef int (*gamera_cs_tiled_radial_boundary_function)(
    const gamera_no_grid *grid, gamera_no_storage *local,
    const gamera_cs_decomposition *tile, size_t ghost,
    const gamera_no_background_field *background,
    double current_time, double previous_time, void *context);

/* Execute the unchanged GAMERA predictor, reconstruction, CT and active
 * update on one actual local tile. All interface plans are cached and use
 * global face coordinates; the callback owns only true radial boundaries. */
int gamera_cs_mpi_advance_tiled(
    const gamera_no_grid *grid, gamera_no_storage *local,
    const gamera_cs_decomposition *tile, size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field *background,
    gamera_cs_tiled_cell_halo_plan *cell_halo,
    gamera_cs_tiled_magnetic_halo_plan *magnetic_halo,
    gamera_cs_tiled_cube_face_sync_plan *face_plan,
    gamera_cs_tiled_edge_sync_plan *edge_plan,
    gamera_cs_tiled_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator);

/* One face per rank. The six grids and storage descriptors are present on
 * each rank for the existing exact donor/interfacial exchange; only the
 * owning rank computes a face's predictor, sweeps, CT, and active update.
 * This is a correctness-first distributed step, not yet a sparse-memory
 * high-resolution implementation. */
int gamera_cs_mpi_advance_six_with_background_planned(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator);

/* Identical update, using a precomputed immutable cube-edge identity plan
 * for the shared CT EMF average. */
int gamera_cs_mpi_advance_six_with_background_planned_cached_edges(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    const gamera_cs_edge_sync_plan *edge_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator);

/* For consecutive calls on the same storage only: the previous step's final
 * active-cell and magnetic-face synchronizations are still current, with no
 * intervening active-state writes. The first step must use the entry above. */
int gamera_cs_mpi_advance_six_with_background_planned_cached_edges_pre_synced(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    const gamera_cs_edge_sync_plan *edge_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator);

/* Same GAMERA update; only final donor-state transfer is sparse. */
int gamera_cs_mpi_advance_six_with_background_sparse_pre_synced(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    const gamera_cs_edge_sync_plan *edge_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator);

#endif
