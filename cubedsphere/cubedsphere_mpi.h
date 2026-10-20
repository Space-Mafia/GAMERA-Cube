#ifndef GAMERA_CUBEDSPHERE_MPI_H
#define GAMERA_CUBEDSPHERE_MPI_H

#include "cubedsphere_mesh.h"
#include "cubedsphere_interface.h"
#include "cubedsphere_halo.h"
#include "nonorthogonal_storage.h"

#include <mpi.h>

/* Six-rank reference exchange: rank f owns face f. All six active cell
 * arrays are replicated after the collective so the existing donor
 * interpolation can be reused exactly. The separate mpi_step module now
 * uses this collective in a correctness-first distributed time step. */
int gamera_cs_mpi_exchange_active_cells(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator);

/* Transfer only active donor cells consumed by the immutable halo stencils.
 * Global MI/output diagnostics still require the full exchange above. */
int gamera_cs_mpi_exchange_active_cells_sparse(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    const gamera_cs_halo_plan *halo_plan, MPI_Comm communicator);

/* Interface exchanges send O(nr*n) seam data, then apply the unchanged
 * shared-flux/EMF reconciliation to those values. These routines are also
 * used by the distributed time step. */
int gamera_cs_mpi_sync_magnetic_face_flux(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator);
int gamera_cs_mpi_sync_mhd_face_fluxes(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator);
int gamera_cs_mpi_sync_edge_emf(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator);
int gamera_cs_mpi_sync_edge_emf_planned(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    const gamera_cs_edge_sync_plan *plan, MPI_Comm communicator);

#endif
