#ifndef GAMERA_CUBEDSPHERE_MPI_TILE_H
#define GAMERA_CUBEDSPHERE_MPI_TILE_H

#include "cubedsphere_decomposition.h"
#include "cubedsphere_halo.h"
#include "nonorthogonal_storage.h"

#include <mpi.h>

/* Exchange the accepted GAMERA conserved and Cartesian magnetic channels
 * across Cartesian subdomains of one cube face.  Four active donor cells are
 * transferred in each direction; physical boundaries are left untouched for
 * the existing Earth and cube-face boundary routines. */
int gamera_cs_mpi_exchange_internal_cell_halos(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world);

/* The lower-coordinate owner supplies the one physical face or CT edge
 * shared by adjacent same-face tiles.  These operations reconcile only MPI
 * subdomain interfaces; cube-face topology is handled separately. */
int gamera_cs_mpi_sync_internal_magnetic_faces(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world);
/* After the shared normal face has its unique owner, exchange four further
 * current/old face-flux planes needed by the unchanged GAMERA EMF stencil.
 * Face orientation and transverse corner extents are derived from the tile. */
int gamera_cs_mpi_exchange_internal_magnetic_face_halos(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world);
int gamera_cs_mpi_sync_internal_mhd_faces(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world);
int gamera_cs_mpi_sync_internal_edge_emf(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world);

/* Reuse the accepted Yin--Yang sparse MPI donor plan for cube-face stencils.
 * Stencil reference order is [stencil][p][q], 64 values per stencil. */
typedef struct gamera_cs_tiled_sparse_plan gamera_cs_tiled_sparse_plan;
enum { GAMERA_CS_TILE_DONOR_CHANNELS = 2 * GAMERA_NO_FLUX_COUNT + 6 };

int gamera_cs_tiled_sparse_plan_create(
    const gamera_cs_tiled_donor_stencil *stencils, size_t stencil_count,
    const gamera_cs_decomposition *tile, MPI_Comm world,
    gamera_cs_tiled_sparse_plan **out);
int gamera_cs_tiled_sparse_plan_exchange(
    gamera_cs_tiled_sparse_plan *plan, const gamera_no_storage *local,
    const gamera_cs_decomposition *tile, size_t ghost);
int gamera_cs_tiled_sparse_plan_value(
    const gamera_cs_tiled_sparse_plan *plan, size_t stencil_index,
    size_t p, size_t q, int channel, double *value);
void gamera_cs_tiled_sparse_plan_destroy(gamera_cs_tiled_sparse_plan *plan);

/* Fill only true cube-face cell ghosts. Same-face internal tile ghosts are
 * filled by gamera_cs_mpi_exchange_internal_cell_halos; the 8×8 interpolation
 * and positivity fallback are identical to the full-face halo path. */
typedef struct gamera_cs_tiled_cell_halo_plan gamera_cs_tiled_cell_halo_plan;
int gamera_cs_tiled_cell_halo_plan_create(
    const gamera_no_grid *grid, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world, gamera_cs_tiled_cell_halo_plan **out);
int gamera_cs_tiled_cell_halo_fill(
    gamera_cs_tiled_cell_halo_plan *plan, gamera_no_storage *local);
void gamera_cs_tiled_cell_halo_plan_destroy(
    gamera_cs_tiled_cell_halo_plan *plan);

/* The accepted full-face magnetic ghost rule samples donor Cartesian B and
 * projects it onto the receiver's oriented face area. Only six B channels
 * are sent across ranks; no new CT or EMF formula is introduced. */
typedef struct gamera_cs_tiled_magnetic_halo_plan
    gamera_cs_tiled_magnetic_halo_plan;
int gamera_cs_tiled_magnetic_halo_plan_create(
    const gamera_no_grid *grid, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world,
    gamera_cs_tiled_magnetic_halo_plan **out);
int gamera_cs_tiled_magnetic_halo_fill(
    gamera_cs_tiled_magnetic_halo_plan *plan, gamera_no_storage *local);
void gamera_cs_tiled_magnetic_halo_plan_destroy(
    gamera_cs_tiled_magnetic_halo_plan *plan);

/* Owner-consistent physical cube-edge face-flux reconciliation. This is the
 * exact average/orientation rule of cubedsphere_interface.c, routed to the
 * owner of each global face cell; angular process counts may differ. */
int gamera_cs_mpi_sync_cube_magnetic_faces(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world);
int gamera_cs_mpi_sync_cube_mhd_faces(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world);

typedef struct gamera_cs_tiled_cube_face_sync_plan
    gamera_cs_tiled_cube_face_sync_plan;
int gamera_cs_tiled_cube_face_sync_plan_create(
    const gamera_cs_decomposition *tile, size_t ghost, MPI_Comm world,
    gamera_cs_tiled_cube_face_sync_plan **out);
int gamera_cs_tiled_cube_face_sync_magnetic(
    gamera_cs_tiled_cube_face_sync_plan *plan, gamera_no_storage *local);
int gamera_cs_tiled_cube_face_sync_mhd(
    gamera_cs_tiled_cube_face_sync_plan *plan, gamera_no_storage *local);
void gamera_cs_tiled_cube_face_sync_plan_destroy(
    gamera_cs_tiled_cube_face_sync_plan *plan);

/* The full-face CT interface matches physical edge endpoint keys, averages
 * signed EMFs, and writes the common value with each edge's sign. This tile
 * plan keeps that exact rule while distributing edge-key ownership over MPI
 * ranks. Same-face internal tile-edge replicas must be synchronized after
 * this call so they inherit their canonical tile's reconciled value. */
typedef struct gamera_cs_tiled_edge_sync_plan
    gamera_cs_tiled_edge_sync_plan;
int gamera_cs_tiled_edge_sync_plan_create(
    const gamera_no_grid *grid, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world, gamera_cs_tiled_edge_sync_plan **out);
int gamera_cs_tiled_edge_sync(
    gamera_cs_tiled_edge_sync_plan *plan, gamera_no_storage *local);
void gamera_cs_tiled_edge_sync_plan_destroy(
    gamera_cs_tiled_edge_sync_plan *plan);

#endif
