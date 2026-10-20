#include "cubedsphere_halo.h"
#include "cubedsphere_interface.h"
#include "cubedsphere_mpi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void initialize_face(const gamera_no_grid *grid,
                            gamera_no_storage *storage,
                            const size_t active[3], size_t ghost) {
  const size_t *extent = storage->cell_extent;
  for (size_t i = ghost; i < ghost + active[0]; ++i) {
    for (size_t j = ghost; j < ghost + active[1]; ++j) {
      for (size_t k = ghost; k < ghost + active[2]; ++k) {
        const size_t c = gamera_no_index3(extent, i, j, k);
        const gamera_no_vec3 p = grid->cell[c].centroid;
        const double rho = 1.0 + 0.01 *
            (p.value[0] + 2.0 * p.value[1] - p.value[2]);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          storage->conserved[c * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              m == GAMERA_NO_FLUX_DENSITY ? rho :
              m == GAMERA_NO_FLUX_ENERGY ? 3.0 : 0.0;
          storage->old_conserved[c * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              m == GAMERA_NO_FLUX_DENSITY ? rho + 0.1 :
              m == GAMERA_NO_FLUX_ENERGY ? 3.1 : 0.0;
        }
        storage->cell_magnetic[c] = (gamera_no_vec3){{0.3, -0.2, 0.4}};
        storage->old_cell_magnetic[c] =
            (gamera_no_vec3){{0.31, -0.19, 0.39}};
      }
    }
  }
}

static void initialize_interfaces(const gamera_no_grid *grid,
                                  gamera_no_storage *storage, int face) {
  for (int d = 0; d < 3; ++d) {
    const size_t faces = gamera_no_element_count3(grid->face[d].extent);
    for (size_t f = 0; f < faces; ++f) {
      const double seed = 0.01 * (double)(face + 1) +
                          0.001 * (double)(d + 1) +
                          0.00001 * (double)(f % 101U);
      storage->face_flux[d][f] = seed;
      storage->old_face_flux[d][f] = seed + 0.1;
      for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
        storage->fluid_face_flux[d][f * GAMERA_NO_FLUX_COUNT + (size_t)m] =
            seed + 0.2 * (double)(m + 1);
      for (int c = 0; c < 3; ++c)
        storage->maxwell_face_flux[d][f].value[c] =
            seed + 0.3 * (double)(c + 1);
    }
    const size_t edges = gamera_no_element_count3(grid->edge[d].extent);
    for (size_t e = 0; e < edges; ++e)
      storage->edge_emf[d][e] = 0.02 * (double)(face + 1) +
          0.003 * (double)(d + 1) + 0.00001 * (double)(e % 103U);
  }
}

static int same_local_interfaces(const gamera_no_grid *grid,
                                 const gamera_no_storage *oracle,
                                 const gamera_no_storage *distributed) {
  for (int d = 0; d < 3; ++d) {
    const size_t faces = gamera_no_element_count3(grid->face[d].extent);
    const size_t edges = gamera_no_element_count3(grid->edge[d].extent);
    if (memcmp(oracle->face_flux[d], distributed->face_flux[d],
               faces * sizeof(double)) != 0 ||
        memcmp(oracle->old_face_flux[d], distributed->old_face_flux[d],
               faces * sizeof(double)) != 0 ||
        memcmp(oracle->fluid_face_flux[d], distributed->fluid_face_flux[d],
               faces * GAMERA_NO_FLUX_COUNT * sizeof(double)) != 0 ||
        memcmp(oracle->maxwell_face_flux[d],
               distributed->maxwell_face_flux[d],
               faces * sizeof(gamera_no_vec3)) != 0 ||
        memcmp(oracle->edge_emf[d], distributed->edge_emf[d],
               edges * sizeof(double)) != 0)
      return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = -1, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != GAMERA_CS_FACE_COUNT) {
    if (rank == 0) fprintf(stderr, "requires exactly six MPI ranks\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  const size_t ghost = 4;
  const size_t active[3] = {8, 10, 10};
  const size_t extent[3] = {16, 18, 18};
  const size_t vertex_extent[3] = {17, 19, 19};
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage oracle[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage distributed[GAMERA_CS_FACE_COUNT] = {{0}};
  int failure = 0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(
        gamera_no_element_count3(vertex_extent), sizeof(*vertices));
    if (vertices == NULL ||
        gamera_cs_generate_vertices(face, active, ghost, 2.0, 3.0,
                                    vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid[face]) != 0 ||
        gamera_no_storage_create(extent, &oracle[face]) != 0 ||
        gamera_no_storage_create(extent, &distributed[face]) != 0)
      failure = 1;
    free(vertices);
    if (failure) break;
    initialize_face(&grid[face], &oracle[face], active, ghost);
    initialize_interfaces(&grid[face], &oracle[face], face);
    if (face == rank)
      initialize_face(&grid[face], &distributed[face], active, ghost);
    if (face == rank)
      initialize_interfaces(&grid[face], &distributed[face], face);
  }
  int any_failure = 0;
  MPI_Allreduce(&failure, &any_failure, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (!any_failure) {
    if (gamera_cs_mpi_exchange_active_cells(
            distributed, active, ghost, MPI_COMM_WORLD) != 0 ||
        gamera_cs_fill_tangential_cell_halos(
            grid, oracle, active, ghost) != 0 ||
        gamera_cs_fill_tangential_cell_halos(
            grid, distributed, active, ghost) != 0)
      failure = 1;
  }
  if (!failure && !any_failure) {
    for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
      for (size_t i = ghost; i < ghost + active[0]; ++i) {
        for (size_t j = 0; j < extent[1]; ++j) {
          for (size_t k = 0; k < extent[2]; ++k) {
            const size_t c = gamera_no_index3(extent, i, j, k);
            if (memcmp(&distributed[face].conserved[
                           c * GAMERA_NO_FLUX_COUNT],
                       &oracle[face].conserved[
                           c * GAMERA_NO_FLUX_COUNT],
                       GAMERA_NO_FLUX_COUNT * sizeof(double)) != 0 ||
                memcmp(&distributed[face].old_conserved[
                           c * GAMERA_NO_FLUX_COUNT],
                       &oracle[face].old_conserved[
                           c * GAMERA_NO_FLUX_COUNT],
                       GAMERA_NO_FLUX_COUNT * sizeof(double)) != 0 ||
                memcmp(&distributed[face].cell_magnetic[c],
                       &oracle[face].cell_magnetic[c],
                       sizeof(gamera_no_vec3)) != 0 ||
                memcmp(&distributed[face].old_cell_magnetic[c],
                       &oracle[face].old_cell_magnetic[c],
                       sizeof(gamera_no_vec3)) != 0)
              failure = 1;
          }
        }
      }
    }
  }
  MPI_Allreduce(&failure, &any_failure, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (!any_failure) {
    if (gamera_cs_sync_magnetic_face_flux(oracle, active, ghost) != 0 ||
        gamera_cs_mpi_sync_magnetic_face_flux(distributed, active, ghost,
                                              MPI_COMM_WORLD) != 0 ||
        gamera_cs_sync_mhd_face_fluxes(oracle, active, ghost) != 0 ||
        gamera_cs_mpi_sync_mhd_face_fluxes(distributed, active, ghost,
                                           MPI_COMM_WORLD) != 0 ||
        gamera_cs_sync_edge_emf(grid, oracle, active, ghost) != 0 ||
        gamera_cs_mpi_sync_edge_emf(grid, distributed, active, ghost,
                                    MPI_COMM_WORLD) != 0 ||
        !same_local_interfaces(&grid[rank], &oracle[rank],
                               &distributed[rank]))
      failure = 1;
  }
  MPI_Allreduce(&failure, &any_failure, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (rank == 0) printf("six-rank cells/flux/EMF vs serial seams: %s\n",
                        any_failure ? "FAIL" : "BITWISE PASS");
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_storage_destroy(&oracle[face]);
    gamera_no_storage_destroy(&distributed[face]);
    gamera_no_grid_destroy(&grid[face]);
  }
  MPI_Finalize();
  return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
