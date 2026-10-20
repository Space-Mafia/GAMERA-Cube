#ifndef GAMERA_CUBEDSPHERE_MI_DIPOLE_MPI_H
#define GAMERA_CUBEDSPHERE_MI_DIPOLE_MPI_H

#include "cubedsphere_decomposition.h"
#include "cubedsphere_mi_dipole.h"
#include "nonorthogonal_radial_map.h"

#include <mpi.h>

typedef struct gamera_cs_mi_dipole_mpi gamera_cs_mi_dipole_mpi;

/* Only the three physical radial layers required by the unchanged FAC curl
 * are assembled on rank zero. Other ranks retain the solved potential only. */
int gamera_cs_mi_dipole_mpi_create(
    const gamera_cs_decomposition *tile, size_t ghost,
    const gamera_no_radial_map *radial_map,
    const gamera_cs_earth_startup *startup,
    const gamera_cs_mi_dipole_config *config, MPI_Comm communicator,
    gamera_cs_mi_dipole_mpi **result);
int gamera_cs_mi_dipole_mpi_maybe_update(
    gamera_cs_mi_dipole_mpi *state, const gamera_no_storage *local,
    double time_code);
const gamera_cs_mi_dipole *gamera_cs_mi_dipole_mpi_state(
    const gamera_cs_mi_dipole_mpi *state);
void gamera_cs_mi_dipole_mpi_destroy(gamera_cs_mi_dipole_mpi *state);

#endif
