#include "cubedsphere_mpi.h"
#include "cubedsphere_interface.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

enum { VALUES_PER_CELL = 2 * GAMERA_NO_FLUX_COUNT + 6 };

int gamera_cs_mpi_exchange_active_cells(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator) {
  int rank = -1, size = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS ||
      rank < 0 || size != GAMERA_CS_FACE_COUNT || storage == NULL ||
      active == NULL || ghost < 4 || active[1] != active[2]) return -1;
  if (active[0] == 0 || active[1] < 8 ||
      active[0] > SIZE_MAX / active[1] ||
      active[0] * active[1] > SIZE_MAX / active[2]) return -1;
  const size_t cells = active[0] * active[1] * active[2];
  if (cells > SIZE_MAX / ((size_t)size * VALUES_PER_CELL) ||
      cells > (size_t)INT_MAX / VALUES_PER_CELL) return -1;
  const size_t count = cells * VALUES_PER_CELL;
  int locally_valid = 1;
  for (int face = 0; face < size; ++face) {
    for (int d = 0; d < 3; ++d) {
      if (storage[face].cell_extent[d] != active[d] + 2 * ghost)
        locally_valid = 0;
    }
    if (storage[face].conserved == NULL ||
        storage[face].old_conserved == NULL ||
        storage[face].cell_magnetic == NULL ||
        storage[face].old_cell_magnetic == NULL) locally_valid = 0;
  }
  double *send = malloc(count * sizeof(*send));
  double *received = malloc((size_t)size * count * sizeof(*received));
  if (send == NULL || received == NULL) locally_valid = 0;
  int all_valid = 0;
  if (MPI_Allreduce(&locally_valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    communicator) != MPI_SUCCESS || !all_valid) {
    free(send);
    free(received);
    return -1;
  }
  size_t position = 0;
  const gamera_no_storage *local = &storage[rank];
  for (size_t i = ghost; i < ghost + active[0]; ++i) {
    for (size_t j = ghost; j < ghost + active[1]; ++j) {
      for (size_t k = ghost; k < ghost + active[2]; ++k) {
        const size_t cell = gamera_no_index3(local->cell_extent, i, j, k);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
          send[position++] = local->conserved[
              cell * GAMERA_NO_FLUX_COUNT + (size_t)m];
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
          send[position++] = local->old_conserved[
              cell * GAMERA_NO_FLUX_COUNT + (size_t)m];
        for (int d = 0; d < 3; ++d)
          send[position++] = local->cell_magnetic[cell].value[d];
        for (int d = 0; d < 3; ++d)
          send[position++] = local->old_cell_magnetic[cell].value[d];
      }
    }
  }
  if (MPI_Allgather(send, (int)count, MPI_DOUBLE, received, (int)count,
                    MPI_DOUBLE, communicator) != MPI_SUCCESS) {
    free(send);
    free(received);
    return -1;
  }
  for (int face = 0; face < size; ++face) {
    gamera_no_storage *destination = &storage[face];
    position = (size_t)face * count;
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t cell = gamera_no_index3(
              destination->cell_extent, i, j, k);
          for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
            destination->conserved[
                cell * GAMERA_NO_FLUX_COUNT + (size_t)m] =
                received[position++];
          for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
            destination->old_conserved[
                cell * GAMERA_NO_FLUX_COUNT + (size_t)m] =
                received[position++];
          for (int d = 0; d < 3; ++d)
            destination->cell_magnetic[cell].value[d] =
                received[position++];
          for (int d = 0; d < 3; ++d)
            destination->old_cell_magnetic[cell].value[d] =
                received[position++];
        }
      }
    }
  }
  free(send);
  free(received);
  return 0;
}

int gamera_cs_mpi_exchange_active_cells_sparse(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    const gamera_cs_halo_plan *halo_plan, MPI_Comm communicator) {
  int rank = -1, ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS ||
      rank < 0 || ranks != GAMERA_CS_FACE_COUNT) return -1;
  int send_count[GAMERA_CS_FACE_COUNT] = {0};
  int recv_count[GAMERA_CS_FACE_COUNT] = {0};
  int send_disp[GAMERA_CS_FACE_COUNT] = {0};
  int recv_disp[GAMERA_CS_FACE_COUNT] = {0};
  const unsigned char *send_mask[GAMERA_CS_FACE_COUNT] = {0};
  const unsigned char *recv_mask[GAMERA_CS_FACE_COUNT] = {0};
  size_t send_total = 0, recv_total = 0;
  int valid = storage != NULL && active != NULL && halo_plan != NULL &&
      ghost >= 4 && active[0] > 0 && active[1] >= 8 &&
      active[1] == active[2];
  if (valid) {
    for (int face = 0; face < ranks; ++face) {
      const gamera_no_storage *s = &storage[face];
      for (int d = 0; d < 3; ++d)
        if (s->cell_extent[d] != active[d] + 2 * ghost) valid = 0;
      if (s->conserved == NULL || s->old_conserved == NULL ||
          s->cell_magnetic == NULL || s->old_cell_magnetic == NULL)
        valid = 0;
    }
  }
  if (valid) {
    for (int peer = 0; peer < ranks; ++peer) {
      size_t send_cells = 0, recv_cells = 0;
      send_mask[peer] = gamera_cs_halo_plan_donor_mask(
          halo_plan, peer, rank, &send_cells);
      recv_mask[peer] = gamera_cs_halo_plan_donor_mask(
          halo_plan, rank, peer, &recv_cells);
      if (send_mask[peer] == NULL || recv_mask[peer] == NULL) valid = 0;
      if (peer == rank) continue;
      if (send_cells > (size_t)INT_MAX / VALUES_PER_CELL / active[0] ||
          recv_cells > (size_t)INT_MAX / VALUES_PER_CELL / active[0])
        valid = 0;
      if (!valid) continue;
      const size_t ns = send_cells * active[0] * VALUES_PER_CELL;
      const size_t nr = recv_cells * active[0] * VALUES_PER_CELL;
      if (send_total > (size_t)INT_MAX - ns ||
          recv_total > (size_t)INT_MAX - nr) valid = 0;
      if (!valid) continue;
      send_disp[peer] = (int)send_total;
      recv_disp[peer] = (int)recv_total;
      send_count[peer] = (int)ns;
      recv_count[peer] = (int)nr;
      send_total += ns;
      recv_total += nr;
    }
  }
  double *send = valid ? malloc((send_total ? send_total : 1) * sizeof(*send)) : NULL;
  double *received = valid ? malloc((recv_total ? recv_total : 1) * sizeof(*received)) : NULL;
  if (send == NULL || received == NULL) valid = 0;
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    communicator) != MPI_SUCCESS || !all_valid) {
    free(send);
    free(received);
    return -1;
  }
  const size_t angular = active[1] * active[2];
  const gamera_no_storage *local = &storage[rank];
  for (int receiver = 0; receiver < ranks; ++receiver) {
    if (receiver == rank) continue;
    size_t position = (size_t)send_disp[receiver];
    const unsigned char *mask = send_mask[receiver];
    for (size_t i = ghost; i < ghost + active[0]; ++i)
      for (size_t a = 0; a < angular; ++a) {
        if (!mask[a]) continue;
        const size_t cell = gamera_no_index3(local->cell_extent, i,
            ghost + a / active[2], ghost + a % active[2]);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
          send[position++] = local->conserved[
              cell * GAMERA_NO_FLUX_COUNT + (size_t)m];
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
          send[position++] = local->old_conserved[
              cell * GAMERA_NO_FLUX_COUNT + (size_t)m];
        for (int d = 0; d < 3; ++d)
          send[position++] = local->cell_magnetic[cell].value[d];
        for (int d = 0; d < 3; ++d)
          send[position++] = local->old_cell_magnetic[cell].value[d];
      }
    if (position != (size_t)send_disp[receiver] + (size_t)send_count[receiver]) {
      free(send);
      free(received);
      return -1;
    }
  }
  if (MPI_Alltoallv(send, send_count, send_disp, MPI_DOUBLE,
                    received, recv_count, recv_disp, MPI_DOUBLE,
                    communicator) != MPI_SUCCESS) {
    free(send);
    free(received);
    return -1;
  }
  for (int donor = 0; donor < ranks; ++donor) {
    if (donor == rank) continue;
    gamera_no_storage *destination = &storage[donor];
    const unsigned char *mask = recv_mask[donor];
    size_t position = (size_t)recv_disp[donor];
    for (size_t i = ghost; i < ghost + active[0]; ++i)
      for (size_t a = 0; a < angular; ++a) {
        if (!mask[a]) continue;
        const size_t cell = gamera_no_index3(destination->cell_extent, i,
            ghost + a / active[2], ghost + a % active[2]);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
          destination->conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              received[position++];
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
          destination->old_conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              received[position++];
        for (int d = 0; d < 3; ++d)
          destination->cell_magnetic[cell].value[d] = received[position++];
        for (int d = 0; d < 3; ++d)
          destination->old_cell_magnetic[cell].value[d] = received[position++];
      }
  }
  free(send);
  free(received);
  return 0;
}

static int rank_and_size(MPI_Comm communicator, int *rank) {
  int size = 0;
  if (MPI_Comm_rank(communicator, rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS ||
      *rank < 0 || size != GAMERA_CS_FACE_COUNT) return -1;
  return 0;
}

static size_t interface_face_index(const gamera_no_storage *storage,
                                   int edge, size_t i, size_t along,
                                   const size_t active[3], size_t ghost) {
  const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
  size_t coord[3] = {i, ghost + along, ghost + along};
  coord[axis] = edge % 2 == 0 ? ghost : ghost + active[axis];
  return gamera_no_index3(storage->face_extent[axis],
                          coord[0], coord[1], coord[2]);
}

static int exchange_interface_faces(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator,
    int magnetic) {
  int rank;
  if (storage == NULL || active == NULL || ghost < 4 ||
      active[0] == 0 || active[1] < 8 || active[1] != active[2] ||
      rank_and_size(communicator, &rank) != 0) return -1;
  const size_t values = magnetic ? 2U : GAMERA_NO_FLUX_COUNT + 3U;
  if (active[0] > SIZE_MAX / active[1] / (4U * values)) return -1;
  const size_t count = 4U * active[0] * active[1] * values;
  if (count > (size_t)INT_MAX || count > SIZE_MAX / 6U) return -1;
  double *send = malloc(count * sizeof(*send));
  double *received = malloc(6U * count * sizeof(*received));
  int locally_valid = send != NULL && received != NULL;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int axis = 1; axis < 3; ++axis) {
      if (magnetic) {
        if (storage[face].face_flux[axis] == NULL ||
            storage[face].old_face_flux[axis] == NULL) locally_valid = 0;
      } else if (storage[face].fluid_face_flux[axis] == NULL ||
                 storage[face].maxwell_face_flux[axis] == NULL) {
        locally_valid = 0;
      }
    }
  }
  int all_valid = 0;
  if (MPI_Allreduce(&locally_valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    communicator) != MPI_SUCCESS || !all_valid) {
    free(send);
    free(received);
    return -1;
  }
  size_t position = 0;
  for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
    const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t along = 0; along < active[1]; ++along) {
        const size_t f = interface_face_index(
            &storage[rank], edge, i, along, active, ghost);
        if (magnetic) {
          send[position++] = storage[rank].face_flux[axis][f];
          send[position++] = storage[rank].old_face_flux[axis][f];
        } else {
          for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
            send[position++] = storage[rank].fluid_face_flux[axis][
                f * GAMERA_NO_FLUX_COUNT + (size_t)m];
          for (int d = 0; d < 3; ++d)
            send[position++] = storage[rank].maxwell_face_flux[axis][f].value[d];
        }
      }
    }
  }
  const int status = MPI_Allgather(send, (int)count, MPI_DOUBLE,
                                   received, (int)count, MPI_DOUBLE,
                                   communicator);
  if (status != MPI_SUCCESS) {
    free(send);
    free(received);
    return -1;
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    position = (size_t)face * count;
    for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
      const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
      for (size_t i = ghost; i < ghost + active[0]; ++i) {
        for (size_t along = 0; along < active[1]; ++along) {
          const size_t f = interface_face_index(
              &storage[face], edge, i, along, active, ghost);
          if (magnetic) {
            storage[face].face_flux[axis][f] = received[position++];
            storage[face].old_face_flux[axis][f] = received[position++];
          } else {
            for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
              storage[face].fluid_face_flux[axis][
                  f * GAMERA_NO_FLUX_COUNT + (size_t)m] =
                  received[position++];
            for (int d = 0; d < 3; ++d)
              storage[face].maxwell_face_flux[axis][f].value[d] =
                  received[position++];
          }
        }
      }
    }
  }
  free(send);
  free(received);
  return magnetic ? gamera_cs_sync_magnetic_face_flux(storage, active, ghost)
                  : gamera_cs_sync_mhd_face_fluxes(storage, active, ghost);
}

int gamera_cs_mpi_sync_magnetic_face_flux(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator) {
  return exchange_interface_faces(storage, active, ghost, communicator, 1);
}

int gamera_cs_mpi_sync_mhd_face_fluxes(
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator) {
  return exchange_interface_faces(storage, active, ghost, communicator, 0);
}

static int shared_edge(int direction, size_t j, size_t k,
                       const size_t active[3], size_t ghost) {
  const int on_a = j == ghost || j == ghost + active[1];
  const int on_b = k == ghost || k == ghost + active[2];
  return direction == 0 ? (on_a || on_b) :
         direction == 1 ? on_b : on_a;
}

static int sync_edge_emf_impl(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    const gamera_cs_edge_sync_plan *plan, MPI_Comm communicator) {
  int rank;
  if (grid == NULL || storage == NULL || active == NULL || ghost < 4 ||
      active[0] == 0 || active[1] < 8 || active[1] != active[2] ||
      rank_and_size(communicator, &rank) != 0) return -1;
  if (active[0] > (SIZE_MAX / (4U * active[1]) - 1U) / 2U) return -1;
  const size_t count = 4U * active[1] * (2U * active[0] + 1U);
  if (count > (size_t)INT_MAX || count > SIZE_MAX / 6U) return -1;
  double *send = malloc(count * sizeof(*send));
  double *received = malloc(6U * count * sizeof(*received));
  int locally_valid = send != NULL && received != NULL;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    for (int d = 0; d < 3; ++d)
      if (storage[face].edge_emf[d] == NULL) locally_valid = 0;
  int all_valid = 0;
  if (MPI_Allreduce(&locally_valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    communicator) != MPI_SUCCESS || !all_valid) {
    free(send);
    free(received);
    return -1;
  }
  size_t position = 0;
  for (int d = 0; d < 3; ++d) {
    const size_t i_upper = ghost + active[0] + (d != 0);
    const size_t j_upper = ghost + active[1] + (d != 1);
    const size_t k_upper = ghost + active[2] + (d != 2);
    for (size_t i = ghost; i < i_upper; ++i)
      for (size_t j = ghost; j < j_upper; ++j)
        for (size_t k = ghost; k < k_upper; ++k) {
          if (!shared_edge(d, j, k, active, ghost)) continue;
          const size_t e = gamera_no_index3(
              storage[rank].edge_extent[d], i, j, k);
          send[position++] = storage[rank].edge_emf[d][e];
        }
  }
  if (position != count ||
      MPI_Allgather(send, (int)count, MPI_DOUBLE, received, (int)count,
                    MPI_DOUBLE, communicator) != MPI_SUCCESS) {
    free(send);
    free(received);
    return -1;
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    position = (size_t)face * count;
    for (int d = 0; d < 3; ++d) {
      const size_t i_upper = ghost + active[0] + (d != 0);
      const size_t j_upper = ghost + active[1] + (d != 1);
      const size_t k_upper = ghost + active[2] + (d != 2);
      for (size_t i = ghost; i < i_upper; ++i)
        for (size_t j = ghost; j < j_upper; ++j)
          for (size_t k = ghost; k < k_upper; ++k) {
            if (!shared_edge(d, j, k, active, ghost)) continue;
            const size_t e = gamera_no_index3(
                storage[face].edge_extent[d], i, j, k);
            storage[face].edge_emf[d][e] = received[position++];
          }
    }
  }
  free(send);
  free(received);
  return plan != NULL ? gamera_cs_sync_edge_emf_planned(plan, storage) :
                        gamera_cs_sync_edge_emf(grid, storage, active, ghost);
}

int gamera_cs_mpi_sync_edge_emf(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, MPI_Comm communicator) {
  return sync_edge_emf_impl(grid, storage, active, ghost, NULL, communicator);
}

int gamera_cs_mpi_sync_edge_emf_planned(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    const gamera_cs_edge_sync_plan *plan, MPI_Comm communicator) {
  if (plan == NULL) return -1;
  return sync_edge_emf_impl(grid, storage, active, ghost, plan, communicator);
}
