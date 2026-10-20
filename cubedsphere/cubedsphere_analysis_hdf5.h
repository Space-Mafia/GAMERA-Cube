#ifndef GAMERA_CUBEDSPHERE_ANALYSIS_HDF5_H
#define GAMERA_CUBEDSPHERE_ANALYSIS_HDF5_H

#include "cubedsphere_decomposition.h"
#include "cubedsphere_mi_dipole.h"
#include "nonorthogonal_background.h"
#include "nonorthogonal_grid.h"
#include "nonorthogonal_storage.h"

#include <mpi.h>

/* The Yin--Yang compact-analysis contract, with one collective HDF5 file
 * per cubed-sphere face.  This is analysis I/O, never a solver restart. */
typedef struct {
  MPI_Comm face_comm;
  gamera_cs_decomposition tile;
  char directory[4096];
  int face_rank;
  int world_rank;
  int grid_written;
} gamera_cs_analysis_hdf5;

int gamera_cs_analysis_hdf5_init(gamera_cs_analysis_hdf5 *output,
                                  const char *legacy_prefix,
                                  const gamera_cs_decomposition *tile,
                                  MPI_Comm world);
void gamera_cs_analysis_hdf5_close(gamera_cs_analysis_hdf5 *output);

/* All ranks participate.  Each face communicator writes its own global
 * face-shaped hyperslab directly, with no rank-zero field gather or CSV. */
int gamera_cs_analysis_hdf5_write(gamera_cs_analysis_hdf5 *output,
                                   long sequence, long step,
                                   double time_code, double time_seconds,
                                   const gamera_no_grid *grid,
                                   const gamera_no_storage *storage,
                                   const gamera_no_background_data *background,
                                   size_t ghost, double gamma,
                                   const gamera_cs_mi_dipole *mi,
                                   MPI_Comm world);

#endif
