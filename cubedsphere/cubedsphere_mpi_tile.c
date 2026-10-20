#include "cubedsphere_mpi_tile.h"
#include "cubedsphere_profile.h"
#include "cubedsphere_halo_fastpath.h"

#include "nonorthogonal_grid.h"
#include "nonorthogonal_yinyang_sparse_plan.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { CHANNELS = 2 * GAMERA_NO_FLUX_COUNT + 6 };

static size_t slab_count(const gamera_cs_decomposition *tile,
                         size_t ghost, int axis) {
  size_t count = ghost;
  for (int d = 0; d < 3; ++d) {
    if (d == axis) continue;
    const size_t width = tile->local_active[d] + 2 * ghost;
    if (count > SIZE_MAX / width) return 0;
    count *= width;
  }
  if (count > SIZE_MAX / CHANNELS) return 0;
  return count * CHANNELS;
}

static void slab_range(const gamera_cs_decomposition *tile, size_t ghost,
                       int axis, size_t plane, size_t begin[3],
                       size_t end[3]) {
  for (int d = 0; d < 3; ++d) {
    if (d == axis) {
      begin[d] = plane;
      end[d] = plane + ghost;
    } else {
      begin[d] = 0;
      end[d] = tile->local_active[d] + 2 * ghost;
    }
  }
}

static void transfer_slab(gamera_no_storage *storage,
                          const gamera_cs_decomposition *tile,
                          size_t ghost, int axis, size_t plane,
                          double *buffer, int unpack) {
  size_t begin[3], end[3], position = 0;
  slab_range(tile, ghost, axis, plane, begin, end);
  for (size_t i = begin[0]; i < end[0]; ++i)
    for (size_t j = begin[1]; j < end[1]; ++j)
      for (size_t k = begin[2]; k < end[2]; ++k) {
        const size_t cell = gamera_no_index3(storage->cell_extent, i, j, k);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          double *value = &storage->conserved[
              cell * GAMERA_NO_FLUX_COUNT + (size_t)m];
          if (unpack) *value = buffer[position++];
          else buffer[position++] = *value;
        }
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          double *value = &storage->old_conserved[
              cell * GAMERA_NO_FLUX_COUNT + (size_t)m];
          if (unpack) *value = buffer[position++];
          else buffer[position++] = *value;
        }
        for (int d = 0; d < 3; ++d) {
          double *value = &storage->cell_magnetic[cell].value[d];
          if (unpack) *value = buffer[position++];
          else buffer[position++] = *value;
        }
        for (int d = 0; d < 3; ++d) {
          double *value = &storage->old_cell_magnetic[cell].value[d];
          if (unpack) *value = buffer[position++];
          else buffer[position++] = *value;
        }
      }
}

int gamera_cs_mpi_exchange_internal_cell_halos(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world) {
  int rank = -1, ranks = 0;
  if (MPI_Comm_rank(world, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(world, &ranks) != MPI_SUCCESS) return -1;
  int valid = local != NULL && tile != NULL && ghost >= 4 &&
      rank >= 0 && tile->face >= 0 &&
      ranks == GAMERA_CS_FACE_COUNT * tile->proc_dims[0] *
               tile->proc_dims[1] * tile->proc_dims[2];
  if (valid) {
    for (int d = 0; d < 3; ++d)
      if (local->cell_extent[d] != tile->local_active[d] + 2 * ghost ||
          tile->local_active[d] < ghost) valid = 0;
    if (local->conserved == NULL || local->old_conserved == NULL ||
        local->cell_magnetic == NULL || local->old_cell_magnetic == NULL)
      valid = 0;
  }
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;

  for (int axis = 0; axis < 3; ++axis) {
    if (tile->proc_dims[axis] == 1) continue;
    const size_t count = slab_count(tile, ghost, axis);
    if (count == 0 || count > INT_MAX) return -1;
    double *send = malloc(count * sizeof(*send));
    double *received = malloc(count * sizeof(*received));
    valid = send != NULL && received != NULL;
    if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                      world) != MPI_SUCCESS || !all_valid) {
      free(send);
      free(received);
      return -1;
    }
    int low = MPI_PROC_NULL, high = MPI_PROC_NULL;
    int neighbor[3] = {tile->coord[0], tile->coord[1], tile->coord[2]};
    if (tile->coord[axis] > 0) {
      --neighbor[axis];
      if (gamera_cs_decomposition_rank(tile->face, neighbor,
                                       tile->proc_dims, &low) != 0) return -1;
      ++neighbor[axis];
    }
    if (tile->coord[axis] + 1 < tile->proc_dims[axis]) {
      ++neighbor[axis];
      if (gamera_cs_decomposition_rank(tile->face, neighbor,
                                       tile->proc_dims, &high) != 0) return -1;
    }
    transfer_slab(local, tile, ghost, axis, ghost, send, 0);
    int status = MPI_Sendrecv(send, (int)count, MPI_DOUBLE, low,
                   2 * axis, received, (int)count, MPI_DOUBLE, high,
                   2 * axis, world, MPI_STATUS_IGNORE);
    if (status == MPI_SUCCESS && high != MPI_PROC_NULL)
      transfer_slab(local, tile, ghost, axis,
          ghost + tile->local_active[axis], received, 1);
    if (status != MPI_SUCCESS) {
      free(send);
      free(received);
      return -1;
    }
    transfer_slab(local, tile, ghost, axis,
                  tile->local_active[axis], send, 0);
    status = MPI_Sendrecv(send, (int)count, MPI_DOUBLE, high,
                 2 * axis + 1, received, (int)count, MPI_DOUBLE, low,
                 2 * axis + 1, world, MPI_STATUS_IGNORE);
    if (status == MPI_SUCCESS && low != MPI_PROC_NULL)
      transfer_slab(local, tile, ghost, axis, 0, received, 1);
    free(send);
    free(received);
    if (status != MPI_SUCCESS) return -1;
  }
  return 0;
}

static int tile_neighbor(const gamera_cs_decomposition *tile,
                         int axis, int delta, int *rank) {
  int coord[3] = {tile->coord[0], tile->coord[1], tile->coord[2]};
  coord[axis] += delta;
  if (coord[axis] < 0 || coord[axis] >= tile->proc_dims[axis]) {
    *rank = MPI_PROC_NULL;
    return 0;
  }
  return gamera_cs_decomposition_rank(tile->face, coord,
                                      tile->proc_dims, rank);
}

static size_t face_plane_values(const gamera_cs_decomposition *tile,
                                int axis, int magnetic) {
  size_t cells = 1;
  for (int d = 0; d < 3; ++d)
    if (d != axis) {
      if (cells > SIZE_MAX / tile->local_active[d]) return 0;
      cells *= tile->local_active[d];
    }
  const size_t channels = magnetic ? 2U : GAMERA_NO_FLUX_COUNT + 3U;
  return cells > SIZE_MAX / channels ? 0 : cells * channels;
}

static void transfer_face_plane(gamera_no_storage *s,
                                const gamera_cs_decomposition *tile,
                                size_t ghost, int axis, size_t plane,
                                int magnetic, double *buffer, int unpack) {
  size_t begin[3] = {ghost, ghost, ghost};
  size_t end[3] = {ghost + tile->local_active[0],
                   ghost + tile->local_active[1],
                   ghost + tile->local_active[2]};
  begin[axis] = plane;
  end[axis] = plane + 1U;
  size_t position = 0;
  for (size_t i = begin[0]; i < end[0]; ++i)
    for (size_t j = begin[1]; j < end[1]; ++j)
      for (size_t k = begin[2]; k < end[2]; ++k) {
        const size_t face = gamera_no_index3(s->face_extent[axis], i, j, k);
        if (magnetic) {
          double *value = &s->face_flux[axis][face];
          if (unpack) *value = buffer[position++];
          else buffer[position++] = *value;
          value = &s->old_face_flux[axis][face];
          if (unpack) *value = buffer[position++];
          else buffer[position++] = *value;
        } else {
          for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
            double *value = &s->fluid_face_flux[axis][
                face * GAMERA_NO_FLUX_COUNT + (size_t)m];
            if (unpack) *value = buffer[position++];
            else buffer[position++] = *value;
          }
          for (int d = 0; d < 3; ++d) {
            double *value = &s->maxwell_face_flux[axis][face].value[d];
            if (unpack) *value = buffer[position++];
            else buffer[position++] = *value;
          }
        }
      }
}

static int sync_faces(gamera_no_storage *local,
                      const gamera_cs_decomposition *tile,
                      size_t ghost, MPI_Comm world, int magnetic) {
  int valid = local != NULL && tile != NULL && ghost >= 4;
  if (valid)
    for (int axis = 0; axis < 3; ++axis)
      if (local->face_flux[axis] == NULL ||
          local->old_face_flux[axis] == NULL ||
          local->fluid_face_flux[axis] == NULL ||
          local->maxwell_face_flux[axis] == NULL)
        valid = 0;
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;
  for (int axis = 0; axis < 3; ++axis) {
    if (tile->proc_dims[axis] == 1) continue;
    const size_t count = face_plane_values(tile, axis, magnetic);
    if (count == 0 || count > INT_MAX) return -1;
    double *send = malloc(count * sizeof(*send));
    double *received = malloc(count * sizeof(*received));
    valid = send != NULL && received != NULL;
    if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                      world) != MPI_SUCCESS || !all_valid) {
      free(send);
      free(received);
      return -1;
    }
    int low, high;
    if (tile_neighbor(tile, axis, -1, &low) != 0 ||
        tile_neighbor(tile, axis, 1, &high) != 0) return -1;
    transfer_face_plane(local, tile, ghost, axis,
        ghost + tile->local_active[axis], magnetic, send, 0);
    const int status = MPI_Sendrecv(send, (int)count, MPI_DOUBLE, high,
        (magnetic ? 100 : 110) + axis,
        received, (int)count, MPI_DOUBLE, low,
        (magnetic ? 100 : 110) + axis, world, MPI_STATUS_IGNORE);
    if (status == MPI_SUCCESS && low != MPI_PROC_NULL)
      transfer_face_plane(local, tile, ghost, axis, ghost,
                          magnetic, received, 1);
    free(send);
    free(received);
    if (status != MPI_SUCCESS) return -1;
  }
  return 0;
}

int gamera_cs_mpi_sync_internal_magnetic_faces(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world) {
  return sync_faces(local, tile, ghost, world, 1);
}

static size_t magnetic_halo_values(const gamera_no_storage *storage,
                                   size_t ghost, int axis, int direction) {
  size_t count = 2U * ghost; /* current and old magnetic face flux */
  for (int d = 0; d < 3; ++d) {
    if (d == axis) continue;
    const size_t width = storage->face_extent[direction][d];
    if (width == 0 || count > SIZE_MAX / width) return 0;
    count *= width;
  }
  return count;
}

static void transfer_magnetic_halo(gamera_no_storage *storage,
                                   const gamera_cs_decomposition *tile,
                                   size_t ghost, int axis, int direction,
                                   size_t plane, double *buffer, int unpack) {
  size_t begin[3], end[3], position = 0;
  for (int d = 0; d < 3; ++d) {
    if (d == axis) {
      begin[d] = plane;
      end[d] = plane + ghost;
    } else {
      begin[d] = 0;
      end[d] = storage->face_extent[direction][d];
    }
  }
  for (size_t i = begin[0]; i < end[0]; ++i)
    for (size_t j = begin[1]; j < end[1]; ++j)
      for (size_t k = begin[2]; k < end[2]; ++k) {
        const size_t face = gamera_no_index3(
            storage->face_extent[direction], i, j, k);
        double *current = &storage->face_flux[direction][face];
        double *old = &storage->old_face_flux[direction][face];
        if (unpack) {
          *current = buffer[position++];
          *old = buffer[position++];
        } else {
          buffer[position++] = *current;
          buffer[position++] = *old;
        }
      }
}

int gamera_cs_mpi_exchange_internal_magnetic_face_halos(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world) {
  int valid = local != NULL && tile != NULL && ghost >= 4;
  if (valid)
    for (int d = 0; d < 3; ++d) {
      if (local->face_flux[d] == NULL || local->old_face_flux[d] == NULL)
        valid = 0;
      for (int a = 0; a < 3; ++a)
        if (local->face_extent[d][a] !=
            tile->local_active[a] + 2U * ghost + (a == d ? 1U : 0U) ||
            tile->local_active[a] < ghost) valid = 0;
    }
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;

  for (int axis = 0; axis < 3; ++axis) {
    if (tile->proc_dims[axis] == 1) continue;
    int low, high;
    if (tile_neighbor(tile, axis, -1, &low) != 0 ||
        tile_neighbor(tile, axis, 1, &high) != 0) return -1;
    for (int direction = 0; direction < 3; ++direction) {
      const size_t count = magnetic_halo_values(
          local, ghost, axis, direction);
      valid = count != 0 && count <= INT_MAX &&
          count <= SIZE_MAX / sizeof(double);
      if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                        world) != MPI_SUCCESS || !all_valid) return -1;
      double *send = malloc(count * sizeof(*send));
      double *received = malloc(count * sizeof(*received));
      valid = send != NULL && received != NULL;
      if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                        world) != MPI_SUCCESS || !all_valid) {
        free(send);
        free(received);
        return -1;
      }
      const int tag = 140 + 6 * axis + 2 * direction;
      transfer_magnetic_halo(local, tile, ghost, axis, direction,
          ghost + (direction == axis ? 1U : 0U), send, 0);
      int status = MPI_Sendrecv(send, (int)count, MPI_DOUBLE, low,
          tag, received, (int)count, MPI_DOUBLE, high,
          tag, world, MPI_STATUS_IGNORE);
      if (status == MPI_SUCCESS && high != MPI_PROC_NULL)
        transfer_magnetic_halo(local, tile, ghost, axis, direction,
            ghost + tile->local_active[axis] +
                (direction == axis ? 1U : 0U), received, 1);
      if (status == MPI_SUCCESS) {
        transfer_magnetic_halo(local, tile, ghost, axis, direction,
            tile->local_active[axis], send, 0);
        status = MPI_Sendrecv(send, (int)count, MPI_DOUBLE, high,
            tag + 1, received, (int)count, MPI_DOUBLE, low,
            tag + 1, world, MPI_STATUS_IGNORE);
        if (status == MPI_SUCCESS && low != MPI_PROC_NULL)
          transfer_magnetic_halo(local, tile, ghost, axis, direction,
              0, received, 1);
      }
      free(send);
      free(received);
      if (status != MPI_SUCCESS) return -1;
    }
  }
  return 0;
}

int gamera_cs_mpi_sync_internal_mhd_faces(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world) {
  return sync_faces(local, tile, ghost, world, 0);
}

static size_t edge_plane_values(const gamera_cs_decomposition *tile,
                                int axis) {
  size_t count = 0;
  for (int edge = 0; edge < 3; ++edge) {
    if (edge == axis) continue;
    size_t elements = 1;
    for (int d = 0; d < 3; ++d) {
      if (d == axis) continue;
      const size_t width = tile->local_active[d] + (d != edge);
      if (elements > SIZE_MAX / width) return 0;
      elements *= width;
    }
    if (count > SIZE_MAX - elements) return 0;
    count += elements;
  }
  return count;
}

static void transfer_edge_plane(gamera_no_storage *s,
                                const gamera_cs_decomposition *tile,
                                size_t ghost, int axis, size_t plane,
                                double *buffer, int unpack) {
  size_t position = 0;
  for (int edge = 0; edge < 3; ++edge) {
    if (edge == axis) continue;
    size_t begin[3] = {ghost, ghost, ghost};
    size_t end[3] = {ghost + tile->local_active[0],
                     ghost + tile->local_active[1],
                     ghost + tile->local_active[2]};
    begin[axis] = plane;
    end[axis] = plane + 1U;
    for (int d = 0; d < 3; ++d)
      if (d != axis && d != edge) ++end[d];
    for (size_t i = begin[0]; i < end[0]; ++i)
      for (size_t j = begin[1]; j < end[1]; ++j)
        for (size_t k = begin[2]; k < end[2]; ++k) {
          const size_t index = gamera_no_index3(
              s->edge_extent[edge], i, j, k);
          double *value = &s->edge_emf[edge][index];
          if (unpack) *value = buffer[position++];
          else buffer[position++] = *value;
        }
  }
}

int gamera_cs_mpi_sync_internal_edge_emf(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world) {
  int valid = local != NULL && tile != NULL && ghost >= 4;
  if (valid)
    for (int edge = 0; edge < 3; ++edge)
      if (local->edge_emf[edge] == NULL) valid = 0;
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;
  for (int axis = 0; axis < 3; ++axis) {
    if (tile->proc_dims[axis] == 1) continue;
    const size_t count = edge_plane_values(tile, axis);
    if (count == 0 || count > INT_MAX) return -1;
    double *send = malloc(count * sizeof(*send));
    double *received = malloc(count * sizeof(*received));
    valid = send != NULL && received != NULL;
    if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                      world) != MPI_SUCCESS || !all_valid) {
      free(send);
      free(received);
      return -1;
    }
    int low, high;
    if (tile_neighbor(tile, axis, -1, &low) != 0 ||
        tile_neighbor(tile, axis, 1, &high) != 0) return -1;
    transfer_edge_plane(local, tile, ghost, axis,
        ghost + tile->local_active[axis], send, 0);
    const int status = MPI_Sendrecv(send, (int)count, MPI_DOUBLE,
        high, 120 + axis, received, (int)count, MPI_DOUBLE,
        low, 120 + axis, world, MPI_STATUS_IGNORE);
    if (status == MPI_SUCCESS && low != MPI_PROC_NULL)
      transfer_edge_plane(local, tile, ghost, axis, ghost, received, 1);
    free(send);
    free(received);
    if (status != MPI_SUCCESS) return -1;
  }
  return 0;
}

struct gamera_cs_tiled_sparse_plan {
  gamera_no_sparse_plan sparse;
  MPI_Comm world;
  size_t stencil_count;
  size_t local_count;
  int channel_count;
  size_t *remote_slot;
  size_t *local_slot;
  size_t *local_cells;
  double *local_values;
  int compact_enabled, values_finite;
  int index_cache_enabled;
  int block_access_enabled;
  int parallel_interpolation;
  unsigned char *nonzero_count, *nonzero_indices;
  gamera_cs_profile perf;
};

static void tiled_halo_profile_finish(gamera_cs_tiled_sparse_plan *plan) {
  static const char *const names[] = {"local_pack", "local_consensus",
    "remote_pack", "neighbor", "remote_consensus_other", "finite_scan",
    "interpolate"};
  cs_profile_mark(&plan->perf, 6);
  if (++plan->perf.steps % 100 == 0) {
    int rank = -1;
    MPI_Comm_rank(plan->world, &rank);
    cs_profile_report(&plan->perf, plan->channel_count == 6 ?
        "CS_HALO_B_PROFILE" : "CS_HALO_CELL_PROFILE", rank, names, 7);
  }
}

typedef struct {
  const gamera_no_storage *storage;
  const gamera_cs_decomposition *tile;
  size_t ghost;
} tiled_sparse_pack_context;

static int pack_tiled_donor_cells(const size_t *local_cells,
                                  size_t cell_count, int channel_count,
                                  double *values, void *opaque) {
  const tiled_sparse_pack_context *context =
      (const tiled_sparse_pack_context *)opaque;
  if (context == NULL || context->storage == NULL ||
      context->tile == NULL || values == NULL ||
      (channel_count != GAMERA_CS_TILE_DONOR_CHANNELS &&
       channel_count != 6) ||
      (cell_count != 0 && local_cells == NULL)) return -1;
  const gamera_no_storage *s = context->storage;
  const gamera_cs_decomposition *tile = context->tile;
  size_t total = tile->local_active[0];
  if (tile->local_active[1] == 0 || tile->local_active[2] == 0 ||
      total > SIZE_MAX / tile->local_active[1]) return -1;
  total *= tile->local_active[1];
  if (total > SIZE_MAX / tile->local_active[2]) return -1;
  total *= tile->local_active[2];
  for (int axis = 0; axis < 3; ++axis)
    if (s->cell_extent[axis] != tile->local_active[axis] +
                                2U * context->ghost) return -1;
  if (s->conserved == NULL || s->old_conserved == NULL ||
      s->cell_magnetic == NULL || s->old_cell_magnetic == NULL) return -1;
  for (size_t item = 0; item < cell_count; ++item) {
    size_t linear = local_cells[item];
    if (linear >= total) return -1;
    const size_t k = linear % tile->local_active[2];
    linear /= tile->local_active[2];
    const size_t j = linear % tile->local_active[1];
    const size_t i = linear / tile->local_active[1];
    const size_t cell = gamera_no_index3(s->cell_extent,
        context->ghost + i, context->ghost + j, context->ghost + k);
    double *out = &values[item * (size_t)channel_count];
    if (channel_count == 6) {
      for (int axis = 0; axis < 3; ++axis) {
        out[axis] = s->cell_magnetic[cell].value[axis];
        out[3 + axis] = s->old_cell_magnetic[cell].value[axis];
      }
      continue;
    }
    for (int channel = 0; channel < GAMERA_NO_FLUX_COUNT; ++channel) {
      out[channel] = s->conserved[
          cell * GAMERA_NO_FLUX_COUNT + (size_t)channel];
      out[GAMERA_NO_FLUX_COUNT + channel] = s->old_conserved[
          cell * GAMERA_NO_FLUX_COUNT + (size_t)channel];
    }
    for (int axis = 0; axis < 3; ++axis) {
      out[2 * GAMERA_NO_FLUX_COUNT + axis] =
          s->cell_magnetic[cell].value[axis];
      out[2 * GAMERA_NO_FLUX_COUNT + 3 + axis] =
          s->old_cell_magnetic[cell].value[axis];
    }
  }
  return 0;
}

static int tiled_sparse_plan_create_channels(
    const gamera_cs_tiled_donor_stencil *stencils, size_t stencil_count,
    const gamera_cs_decomposition *tile, MPI_Comm world,
    int channel_count, gamera_cs_tiled_sparse_plan **out) {
  int rank = -1, ranks = 0;
  if (MPI_Comm_rank(world, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(world, &ranks) != MPI_SUCCESS) return -1;
  int valid = out != NULL && tile != NULL &&
      (stencil_count == 0 || stencils != NULL) &&
      stencil_count <= SIZE_MAX / 64U &&
      (channel_count == 6 ||
       channel_count == GAMERA_CS_TILE_DONOR_CHANNELS);
  size_t local_total = 0;
  if (valid) {
    gamera_cs_decomposition expected;
    valid = gamera_cs_decomposition_init(tile->global_active,
        tile->proc_dims, ranks, rank, &expected) == 0 &&
        expected.face == tile->face;
    for (int axis = 0; axis < 3 && valid; ++axis)
      valid = expected.coord[axis] == tile->coord[axis] &&
          expected.global_start[axis] == tile->global_start[axis] &&
          expected.local_active[axis] == tile->local_active[axis];
    if (valid && tile->local_active[0] <=
            SIZE_MAX / tile->local_active[1]) {
      local_total = tile->local_active[0] * tile->local_active[1];
      valid = local_total <= SIZE_MAX / tile->local_active[2];
      if (valid) local_total *= tile->local_active[2];
    } else valid = 0;
  }
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;
  *out = NULL;
  const size_t references = stencil_count * 64U;
  gamera_cs_tiled_sparse_plan *plan = calloc(1, sizeof(*plan));
  gamera_no_sparse_reference *ref = calloc(
      references != 0 ? references : 1U, sizeof(*ref));
  valid = plan != NULL && ref != NULL;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(ref);
    free(plan);
    return -1;
  }
  plan->sparse = (gamera_no_sparse_plan)GAMERA_NO_SPARSE_PLAN_INITIALIZER;
  plan->world = world;
  plan->stencil_count = stencil_count;
  plan->channel_count = channel_count;
  const char *compact = getenv("GAMERA_CS_HALO_COMPACT");
  plan->compact_enabled = compact != NULL && strcmp(compact, "1") == 0;
  const char *cache = getenv("GAMERA_CS_HALO_INDEX_CACHE");
  plan->index_cache_enabled = cache != NULL && strcmp(cache, "1") == 0;
  const char *block = getenv("GAMERA_CS_HALO_BLOCK_ACCESS");
  plan->block_access_enabled = block != NULL && strcmp(block, "1") == 0;
  const char *parallel = getenv("GAMERA_CS_HALO_PARALLEL");
  plan->parallel_interpolation = parallel != NULL && strcmp(parallel, "1") == 0;
  if (plan->index_cache_enabled) {
    plan->nonzero_count = calloc(stencil_count != 0 ? stencil_count : 1U, 1);
    plan->nonzero_indices = calloc(references != 0 ? references : 1U, 1);
  }
  plan->remote_slot = calloc(references != 0 ? references : 1U,
                             sizeof(*plan->remote_slot));
  plan->local_slot = calloc(references != 0 ? references : 1U,
                            sizeof(*plan->local_slot));
  plan->local_cells = calloc(references != 0 ? references : 1U,
                             sizeof(*plan->local_cells));
  valid = plan->remote_slot != NULL && plan->local_slot != NULL &&
          plan->local_cells != NULL && (!plan->index_cache_enabled ||
          (plan->nonzero_count != NULL && plan->nonzero_indices != NULL));
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(ref);
    gamera_cs_tiled_sparse_plan_destroy(plan);
    return -1;
  }
  size_t remote_count = 0;
  if (plan->index_cache_enabled)
    for (size_t item = 0; item < stencil_count; ++item)
      plan->nonzero_count[item] = cs_halo_nonzero_terms(
          stencils[item].weight, &plan->nonzero_indices[item * 64U]);
  for (size_t item = 0; item < stencil_count; ++item)
    for (size_t p = 0; p < 8; ++p)
      for (size_t q = 0; q < 8; ++q) {
        const size_t index = item * 64U + p * 8U + q;
        const size_t *cell = stencils[item].local_cell[p][q];
        const int donor_rank = stencils[item].rank[p][q];
        if (cell[0] >= tile->local_active[0] ||
            cell[1] >= tile->local_active[1] ||
            cell[2] >= tile->local_active[2] ||
            donor_rank < 0 || donor_rank >= ranks) valid = 0;
        if (valid) {
          const size_t donor_cell =
              (cell[0] * tile->local_active[1] + cell[1]) *
                  tile->local_active[2] + cell[2];
          if (donor_rank == rank) {
            plan->local_slot[index] = plan->local_count;
            plan->remote_slot[index] = SIZE_MAX;
            plan->local_cells[plan->local_count++] = donor_cell;
          } else {
            plan->local_slot[index] = SIZE_MAX;
            plan->remote_slot[index] = remote_count;
            ref[remote_count].donor_rank = donor_rank;
            ref[remote_count++].donor_cell = donor_cell;
          }
        }
      }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(ref);
    gamera_cs_tiled_sparse_plan_destroy(plan);
    return -1;
  }
  valid = plan->local_count <= SIZE_MAX / (size_t)channel_count;
  if (valid) {
    const size_t local_values = plan->local_count * (size_t)channel_count;
    plan->local_values = calloc(local_values != 0 ? local_values : 1U,
                                sizeof(*plan->local_values));
    valid = plan->local_values != NULL;
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(ref);
    gamera_cs_tiled_sparse_plan_destroy(plan);
    return -1;
  }
  const int status = gamera_no_sparse_plan_build(
      &plan->sparse, world, ref, remote_count, local_total,
      channel_count, 0, ranks);
  free(ref);
  if (status != 0) {
    gamera_cs_tiled_sparse_plan_destroy(plan);
    return -1;
  }
  if (getenv("GAMERA_CS_PROFILE_TILE") != NULL) {
    size_t nonzero = 0;
    for (size_t item = 0; item < stencil_count; ++item)
      for (size_t p = 0; p < 8; ++p)
        for (size_t q = 0; q < 8; ++q)
          nonzero += stencils[item].weight[0][p] *
                     stencils[item].weight[1][q] != 0.0;
    fprintf(stderr, "CS_HALO_PLAN rank=%d channels=%d ghosts=%zu refs=%zu nonzero=%zu recv_cells=%zu send_cells=%zu compact=%d\n",
        rank, channel_count, stencil_count, references, nonzero,
        plan->sparse.receive_cell_count, plan->sparse.send_cell_count,
        plan->compact_enabled);
  }
  *out = plan;
  return 0;
}

int gamera_cs_tiled_sparse_plan_create(
    const gamera_cs_tiled_donor_stencil *stencils, size_t stencil_count,
    const gamera_cs_decomposition *tile, MPI_Comm world,
    gamera_cs_tiled_sparse_plan **out) {
  return tiled_sparse_plan_create_channels(stencils, stencil_count,
      tile, world, GAMERA_CS_TILE_DONOR_CHANNELS, out);
}

int gamera_cs_tiled_sparse_plan_exchange(
    gamera_cs_tiled_sparse_plan *plan, const gamera_no_storage *local,
    const gamera_cs_decomposition *tile, size_t ghost) {
  if (plan == NULL) return -1;
  cs_profile_begin(&plan->perf);
  const tiled_sparse_pack_context context = {local, tile, ghost};
  const int local_failed = pack_tiled_donor_cells(
      plan->local_cells, plan->local_count,
      plan->channel_count, plan->local_values,
      (void *)&context) != 0;
  cs_profile_mark(&plan->perf, 0);
  int any_failed = 0;
  if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                    plan->world) != MPI_SUCCESS || any_failed) return -1;
  cs_profile_mark(&plan->perf, 1);
  const int status = gamera_no_sparse_plan_exchange(
      &plan->sparse, pack_tiled_donor_cells, (void *)&context);
  cs_profile_mark(&plan->perf, 4);
  if (plan->perf.enabled) {
    plan->perf.totals[2] += plan->sparse.last_pack_seconds;
    plan->perf.totals[3] += plan->sparse.last_collective_seconds;
    plan->perf.totals[4] -= plan->sparse.last_pack_seconds +
                            plan->sparse.last_collective_seconds;
  }
  plan->values_finite = 0;
  if (status == 0 && plan->compact_enabled)
    plan->values_finite = cs_halo_values_finite(plan->local_values,
        plan->local_count * (size_t)plan->channel_count) &&
        cs_halo_values_finite(plan->sparse.receive_values,
        plan->sparse.receive_cell_count * (size_t)plan->channel_count);
  cs_profile_mark(&plan->perf, 5);
  return status;
}

int gamera_cs_tiled_sparse_plan_value(
    const gamera_cs_tiled_sparse_plan *plan, size_t stencil_index,
    size_t p, size_t q, int channel, double *value) {
  if (plan == NULL || value == NULL ||
      stencil_index >= plan->stencil_count || p >= 8 || q >= 8 ||
      channel < 0 || channel >= plan->channel_count ||
      !plan->sparse.ready || plan->sparse.call_count == 0) return -1;
  const size_t reference = stencil_index * 64U + p * 8U + q;
  if (plan->local_slot[reference] != SIZE_MAX) {
    *value = plan->local_values[
        plan->local_slot[reference] * (size_t)plan->channel_count +
        (size_t)channel];
  } else {
    *value = gamera_no_sparse_plan_value(&plan->sparse,
        plan->remote_slot[reference], channel);
  }
  return 0;
}

/* Resolve one packed donor row, then reuse its address across channels.
 * The buffers are immutable until the next exchange. This changes neither
 * communication nor the order of interpolation terms or channel arithmetic.
 * Private callers know the row width (16 cell channels or 6 B channels). */
static const double *tiled_sparse_donor_values(
    const gamera_cs_tiled_sparse_plan *plan, size_t item, size_t p, size_t q) {
  if (plan == NULL || item >= plan->stencil_count || p >= 8 || q >= 8 ||
      !plan->sparse.ready || plan->sparse.call_count == 0) return NULL;
  const size_t ref = item * 64U + p * 8U + q;
  if (plan->local_slot[ref] != SIZE_MAX)
    return &plan->local_values[plan->local_slot[ref] *
                               (size_t)plan->channel_count];
  return &plan->sparse.receive_values[
      plan->sparse.reference_slot[plan->remote_slot[ref]] *
      (size_t)plan->channel_count];
}

void gamera_cs_tiled_sparse_plan_destroy(gamera_cs_tiled_sparse_plan *plan) {
  if (plan == NULL) return;
  gamera_no_sparse_plan_destroy(&plan->sparse);
  free(plan->remote_slot);
  free(plan->local_slot);
  free(plan->local_cells);
  free(plan->nonzero_count);
  free(plan->nonzero_indices);
  free(plan->local_values);
  free(plan);
}

struct gamera_cs_tiled_cell_halo_plan {
  gamera_cs_decomposition tile;
  size_t ghost;
  size_t count;
  size_t *target;
  gamera_cs_tiled_donor_stencil *stencil;
  gamera_cs_tiled_sparse_plan *sparse;
};

static int true_cube_face_ghost(const gamera_cs_decomposition *tile,
                                size_t ghost, size_t j, size_t k) {
  const size_t gj = tile->global_start[1] + j;
  const size_t gk = tile->global_start[2] + k;
  return gj < ghost || gj - ghost >= tile->global_active[1] ||
         gk < ghost || gk - ghost >= tile->global_active[2];
}

int gamera_cs_tiled_cell_halo_plan_create(
    const gamera_no_grid *grid, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world, gamera_cs_tiled_cell_halo_plan **out) {
  int valid = grid != NULL && tile != NULL && out != NULL &&
      ghost >= 4 && grid != NULL && grid->cell != NULL;
  if (valid)
    for (int axis = 0; axis < 3; ++axis)
      valid &= tile->local_active[axis] <= SIZE_MAX - 2U * ghost &&
          grid->cell_extent[axis] == tile->local_active[axis] + 2U * ghost;
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;
  *out = NULL;
  gamera_cs_tiled_cell_halo_plan *plan = calloc(1, sizeof(*plan));
  valid = plan != NULL;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(plan);
    return -1;
  }
  plan->tile = *tile;
  plan->ghost = ghost;
  const size_t nj = grid->cell_extent[1], nk = grid->cell_extent[2];
  for (size_t j = 0; j < nj; ++j)
    for (size_t k = 0; k < nk; ++k)
      if (true_cube_face_ghost(tile, ghost, j, k)) {
        if (plan->count > SIZE_MAX - tile->local_active[0]) valid = 0;
        else plan->count += tile->local_active[0];
      }
  valid &= plan->count <= SIZE_MAX / sizeof(*plan->target) &&
      plan->count <= SIZE_MAX / sizeof(*plan->stencil);
  if (valid) {
    plan->target = calloc(plan->count != 0 ? plan->count : 1U,
                          sizeof(*plan->target));
    plan->stencil = calloc(plan->count != 0 ? plan->count : 1U,
                           sizeof(*plan->stencil));
    valid = plan->target != NULL && plan->stencil != NULL;
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    gamera_cs_tiled_cell_halo_plan_destroy(plan);
    return -1;
  }
  size_t item = 0;
  for (size_t i = ghost; i < ghost + tile->local_active[0]; ++i)
    for (size_t j = 0; j < nj; ++j)
      for (size_t k = 0; k < nk; ++k) {
        if (!true_cube_face_ghost(tile, ghost, j, k)) continue;
        const size_t target = gamera_no_index3(
            grid->cell_extent, i, j, k);
        plan->target[item] = target;
        if (gamera_cs_tiled_donor_stencil_create(
                grid->cell[target].centroid, tile->global_active,
                tile->proc_dims, tile->global_start[0] + i - ghost,
                0, &plan->stencil[item]) != 0 ||
            plan->stencil[item].donor_face == tile->face)
          valid = 0;
        ++item;
      }
  valid &= item == plan->count;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    gamera_cs_tiled_cell_halo_plan_destroy(plan);
    return -1;
  }
  if (gamera_cs_tiled_sparse_plan_create(
          plan->stencil, plan->count, tile, world, &plan->sparse) != 0) {
    gamera_cs_tiled_cell_halo_plan_destroy(plan);
    return -1;
  }
  *out = plan;
  return 0;
}

static int tiled_cell_halo_fill_item(
    const gamera_cs_tiled_cell_halo_plan *plan,
    gamera_no_storage *local, size_t item) {
    double current[GAMERA_NO_FLUX_COUNT] = {0.0};
    double old[GAMERA_NO_FLUX_COUNT] = {0.0};
    gamera_no_vec3 current_b = {{0.0, 0.0, 0.0}};
    gamera_no_vec3 old_b = {{0.0, 0.0, 0.0}};
    const gamera_cs_tiled_donor_stencil *stencil = &plan->stencil[item];
    const int cached = plan->sparse->index_cache_enabled &&
                       plan->sparse->values_finite;
    const size_t terms = cached ? plan->sparse->nonzero_count[item] : 64U;
    for (size_t term = 0; term < terms; ++term) {
        const size_t slot = cached ?
            plan->sparse->nonzero_indices[item * 64U + term] : term;
        const size_t p = slot / 8U, q = slot % 8U;
        const double weight = stencil->weight[0][p] *
                              stencil->weight[1][q];
        if (cs_halo_skip_zero(weight, plan->sparse->values_finite)) continue;
        if (plan->sparse->block_access_enabled) {
          const double *values = tiled_sparse_donor_values(
              plan->sparse, item, p, q);
          if (values == NULL) return -1;
          for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
            current[m] += weight * values[m];
            old[m] += weight * values[GAMERA_NO_FLUX_COUNT + m];
          }
          for (int d = 0; d < 3; ++d) {
            current_b.value[d] += weight * values[2 * GAMERA_NO_FLUX_COUNT + d];
            old_b.value[d] += weight * values[2 * GAMERA_NO_FLUX_COUNT + 3 + d];
          }
          continue;
        }
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          double value = 0.0;
          if (gamera_cs_tiled_sparse_plan_value(
                  plan->sparse, item, p, q, m, &value) != 0) return -1;
          current[m] += weight * value;
          if (gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                  GAMERA_NO_FLUX_COUNT + m, &value) != 0) return -1;
          old[m] += weight * value;
        }
        for (int d = 0; d < 3; ++d) {
          double value = 0.0;
          if (gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                  2 * GAMERA_NO_FLUX_COUNT + d, &value) != 0) return -1;
          current_b.value[d] += weight * value;
          if (gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                  2 * GAMERA_NO_FLUX_COUNT + 3 + d, &value) != 0) return -1;
          old_b.value[d] += weight * value;
        }
      }
    if (!gamera_cs_halo_valid_state(current) ||
        !gamera_cs_halo_valid_state(old)) {
      const size_t p = stencil->nearest[0], q = stencil->nearest[1];
      for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
        if (gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                m, &current[m]) != 0 ||
            gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                GAMERA_NO_FLUX_COUNT + m, &old[m]) != 0) return -1;
      }
      for (int d = 0; d < 3; ++d)
        if (gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                2 * GAMERA_NO_FLUX_COUNT + d,
                &current_b.value[d]) != 0 ||
            gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                2 * GAMERA_NO_FLUX_COUNT + 3 + d,
                &old_b.value[d]) != 0) return -1;
    }
    const size_t target = plan->target[item];
    memcpy(&local->conserved[target * GAMERA_NO_FLUX_COUNT], current,
           sizeof(current));
    memcpy(&local->old_conserved[target * GAMERA_NO_FLUX_COUNT], old,
           sizeof(old));
    local->cell_magnetic[target] = current_b;
    local->old_cell_magnetic[target] = old_b;
  return 0;
}

int gamera_cs_tiled_cell_halo_fill(
    gamera_cs_tiled_cell_halo_plan *plan, gamera_no_storage *local) {
  if (plan == NULL || local == NULL) return -1;
  for (int axis = 0; axis < 3; ++axis)
    if (local->cell_extent[axis] !=
        plan->tile.local_active[axis] + 2U * plan->ghost) return -1;
  if (gamera_cs_tiled_sparse_plan_exchange(
          plan->sparse, local, &plan->tile, plan->ghost) != 0) return -1;
  if (plan->sparse->parallel_interpolation) {
    /* Creation enumerates each target cell once. Exchange packs all donors
     * before this loop, so workers read immutable buffers, not local outputs.
     * No MPI or profiler calls occur inside the parallel region. */
    int failed = 0;
#pragma omp parallel for schedule(static) reduction(| : failed)
    for (size_t item = 0; item < plan->count; ++item)
      failed |= tiled_cell_halo_fill_item(plan, local, item) != 0;
    if (failed) return -1;
  } else {
    for (size_t item = 0; item < plan->count; ++item)
      if (tiled_cell_halo_fill_item(plan, local, item) != 0) return -1;
  }
  tiled_halo_profile_finish(plan->sparse);
  return 0;
}

void gamera_cs_tiled_cell_halo_plan_destroy(
    gamera_cs_tiled_cell_halo_plan *plan) {
  if (plan == NULL) return;
  gamera_cs_tiled_sparse_plan_destroy(plan->sparse);
  free(plan->target);
  free(plan->stencil);
  free(plan);
}

struct gamera_cs_tiled_magnetic_halo_plan {
  gamera_cs_decomposition tile;
  const gamera_no_grid *grid;
  size_t ghost;
  size_t count;
  size_t *target;
  unsigned char *direction;
  gamera_cs_tiled_donor_stencil *stencil;
  gamera_cs_tiled_sparse_plan *sparse;
};

static int true_cube_face_magnetic_ghost(
    const gamera_cs_decomposition *tile, size_t ghost,
    int direction, size_t j, size_t k) {
  const size_t gj = tile->global_start[1] + j;
  const size_t gk = tile->global_start[2] + k;
  const size_t upper_j = tile->global_active[1] +
      (direction == 1 ? 1U : 0U);
  const size_t upper_k = tile->global_active[2] +
      (direction == 2 ? 1U : 0U);
  return gj < ghost || gj - ghost >= upper_j ||
         gk < ghost || gk - ghost >= upper_k;
}

static size_t donor_radial_cell(const gamera_cs_decomposition *tile,
                                size_t ghost, size_t i) {
  if (i < ghost) {
    const size_t distance = ghost - i;
    return tile->global_start[0] < distance ? 0 :
        tile->global_start[0] - distance;
  }
  const size_t index = tile->global_start[0] + i - ghost;
  return index < tile->global_active[0] ? index :
      tile->global_active[0] - 1U;
}

int gamera_cs_tiled_magnetic_halo_plan_create(
    const gamera_no_grid *grid, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world,
    gamera_cs_tiled_magnetic_halo_plan **out) {
  int valid = grid != NULL && tile != NULL && out != NULL && ghost >= 4;
  if (valid)
    for (int direction = 0; direction < 3; ++direction) {
      valid &= grid->face[direction].value != NULL;
      for (int axis = 0; axis < 3; ++axis)
        valid &= tile->local_active[axis] <= SIZE_MAX - 2U * ghost - 1U &&
            grid->face[direction].extent[axis] ==
            tile->local_active[axis] + 2U * ghost +
            (direction == axis ? 1U : 0U);
    }
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;
  *out = NULL;
  gamera_cs_tiled_magnetic_halo_plan *plan = calloc(1, sizeof(*plan));
  valid = plan != NULL;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(plan);
    return -1;
  }
  plan->tile = *tile;
  plan->grid = grid;
  plan->ghost = ghost;
  for (int direction = 0; direction < 3; ++direction) {
    const size_t *extent = grid->face[direction].extent;
    for (size_t j = 0; j < extent[1]; ++j)
      for (size_t k = 0; k < extent[2]; ++k)
        if (true_cube_face_magnetic_ghost(tile, ghost,
                                          direction, j, k)) {
          if (plan->count > SIZE_MAX - extent[0]) valid = 0;
          else plan->count += extent[0];
        }
  }
  valid &= plan->count <= SIZE_MAX / sizeof(*plan->target) &&
      plan->count <= SIZE_MAX / sizeof(*plan->direction) &&
      plan->count <= SIZE_MAX / sizeof(*plan->stencil);
  if (valid) {
    plan->target = calloc(plan->count != 0 ? plan->count : 1U,
                          sizeof(*plan->target));
    plan->direction = calloc(plan->count != 0 ? plan->count : 1U,
                             sizeof(*plan->direction));
    plan->stencil = calloc(plan->count != 0 ? plan->count : 1U,
                           sizeof(*plan->stencil));
    valid = plan->target != NULL && plan->direction != NULL &&
            plan->stencil != NULL;
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    gamera_cs_tiled_magnetic_halo_plan_destroy(plan);
    return -1;
  }
  size_t item = 0;
  for (int direction = 0; direction < 3; ++direction) {
    const size_t *extent = grid->face[direction].extent;
    for (size_t i = 0; i < extent[0]; ++i)
      for (size_t j = 0; j < extent[1]; ++j)
        for (size_t k = 0; k < extent[2]; ++k) {
          if (!true_cube_face_magnetic_ghost(tile, ghost,
                                              direction, j, k)) continue;
          const size_t target = gamera_no_index3(extent, i, j, k);
          plan->target[item] = target;
          plan->direction[item] = (unsigned char)direction;
          if (gamera_cs_tiled_donor_stencil_create(
                  grid->face[direction].value[target].centroid,
                  tile->global_active, tile->proc_dims,
                  donor_radial_cell(tile, ghost, i), 1,
                  &plan->stencil[item]) != 0 ||
              plan->stencil[item].donor_face == tile->face)
            valid = 0;
          ++item;
        }
  }
  valid &= item == plan->count;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    gamera_cs_tiled_magnetic_halo_plan_destroy(plan);
    return -1;
  }
  if (tiled_sparse_plan_create_channels(plan->stencil, plan->count,
          tile, world, 6, &plan->sparse) != 0) {
    gamera_cs_tiled_magnetic_halo_plan_destroy(plan);
    return -1;
  }
  *out = plan;
  return 0;
}

static int tiled_magnetic_halo_fill_item(
    const gamera_cs_tiled_magnetic_halo_plan *plan,
    gamera_no_storage *local, size_t item) {
    gamera_no_vec3 current = {{0.0, 0.0, 0.0}};
    gamera_no_vec3 old = current;
    const gamera_cs_tiled_donor_stencil *stencil = &plan->stencil[item];
    const int cached = plan->sparse->index_cache_enabled &&
                       plan->sparse->values_finite;
    const size_t terms = cached ? plan->sparse->nonzero_count[item] : 64U;
    for (size_t term = 0; term < terms; ++term) {
        const size_t slot = cached ?
            plan->sparse->nonzero_indices[item * 64U + term] : term;
        const size_t p = slot / 8U, q = slot % 8U;
        const double weight = stencil->weight[0][p] *
                              stencil->weight[1][q];
        if (cs_halo_skip_zero(weight, plan->sparse->values_finite)) continue;
        if (plan->sparse->block_access_enabled) {
          const double *values = tiled_sparse_donor_values(
              plan->sparse, item, p, q);
          if (values == NULL) return -1;
          for (int axis = 0; axis < 3; ++axis) {
            current.value[axis] += weight * values[axis];
            old.value[axis] += weight * values[3 + axis];
          }
          continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
          double value = 0.0;
          if (gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                  axis, &value) != 0) return -1;
          current.value[axis] += weight * value;
          if (gamera_cs_tiled_sparse_plan_value(plan->sparse, item, p, q,
                  3 + axis, &value) != 0) return -1;
          old.value[axis] += weight * value;
        }
      }
    const int direction = (int)plan->direction[item];
    const size_t target = plan->target[item];
    const gamera_no_vec3 area =
        plan->grid->face[direction].value[target].area_vector;
    double current_flux = 0.0, old_flux = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      current_flux += current.value[axis] * area.value[axis];
      old_flux += old.value[axis] * area.value[axis];
    }
    local->face_flux[direction][target] = current_flux;
    local->old_face_flux[direction][target] = old_flux;
  return 0;
}

int gamera_cs_tiled_magnetic_halo_fill(
    gamera_cs_tiled_magnetic_halo_plan *plan, gamera_no_storage *local) {
  if (plan == NULL || local == NULL) return -1;
  if (gamera_cs_tiled_sparse_plan_exchange(plan->sparse, local,
          &plan->tile, plan->ghost) != 0) return -1;
  if (plan->sparse->parallel_interpolation) {
    /* Each (direction,target) pair is unique, even across face directions.
     * Per-target term and component accumulation order stays unchanged. */
    int failed = 0;
#pragma omp parallel for schedule(static) reduction(| : failed)
    for (size_t item = 0; item < plan->count; ++item)
      failed |= tiled_magnetic_halo_fill_item(plan, local, item) != 0;
    if (failed) return -1;
  } else {
    for (size_t item = 0; item < plan->count; ++item)
      if (tiled_magnetic_halo_fill_item(plan, local, item) != 0) return -1;
  }
  tiled_halo_profile_finish(plan->sparse);
  return 0;
}

void gamera_cs_tiled_magnetic_halo_plan_destroy(
    gamera_cs_tiled_magnetic_halo_plan *plan) {
  if (plan == NULL) return;
  gamera_cs_tiled_sparse_plan_destroy(plan->sparse);
  free(plan->target);
  free(plan->direction);
  free(plan->stencil);
  free(plan);
}

static size_t cube_edge_face_index(const gamera_no_storage *local,
                                   const gamera_cs_decomposition *tile,
                                   size_t ghost, int edge,
                                   size_t radial, size_t along) {
  const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
  size_t index[3] = {ghost + radial, ghost + along, ghost + along};
  index[axis] = edge % 2 == 0 ? ghost :
      ghost + tile->local_active[axis];
  const size_t extent[3] = {
      tile->local_active[0] + 2U * ghost + (axis == 0 ? 1U : 0U),
      tile->local_active[1] + 2U * ghost + (axis == 1 ? 1U : 0U),
      tile->local_active[2] + 2U * ghost + (axis == 2 ? 1U : 0U)};
  return gamera_no_index3(local != NULL ? local->face_extent[axis] : extent,
                          index[0], index[1], index[2]);
}

int gamera_cs_mpi_sync_cube_magnetic_faces(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world) {
  gamera_cs_tiled_cube_face_sync_plan *plan = NULL;
  if (gamera_cs_tiled_cube_face_sync_plan_create(
          tile, ghost, world, &plan) != 0) return -1;
  const int status = gamera_cs_tiled_cube_face_sync_magnetic(plan, local);
  gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
  return status;
}

int gamera_cs_mpi_sync_cube_mhd_faces(
    gamera_no_storage *local, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world) {
  gamera_cs_tiled_cube_face_sync_plan *plan = NULL;
  if (gamera_cs_tiled_cube_face_sync_plan_create(
          tile, ghost, world, &plan) != 0) return -1;
  const int status = gamera_cs_tiled_cube_face_sync_mhd(plan, local);
  gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
  return status;
}

typedef struct {
  int axis;
  size_t face;
  double orientation;
} tiled_cube_face_entry;

struct gamera_cs_tiled_cube_face_sync_plan {
  gamera_cs_decomposition tile;
  MPI_Comm world;
  int ranks;
  size_t ghost;
  size_t max_along;
  size_t count;
  size_t serve_total;
  tiled_cube_face_entry *entry;
  size_t *reference_slot;
  size_t *serve_cells;
  int *request_counts, *request_displacements;
  int *serve_counts, *serve_displacements;
  double *send_values, *receive_values;
  MPI_Request *requests;
};

typedef struct {
  const gamera_cs_tiled_cube_face_sync_plan *plan;
  const gamera_no_storage *local;
} tiled_cube_face_pack_context;

static int pack_tiled_cube_faces(const size_t *requested,
                                 size_t count, int channels,
                                 double *values, void *opaque) {
  const tiled_cube_face_pack_context *context = opaque;
  if (context == NULL || context->plan == NULL ||
      context->local == NULL || values == NULL ||
      (count != 0 && requested == NULL) ||
      (channels != 2 && channels != GAMERA_NO_FLUX_COUNT + 3))
    return -1;
  const gamera_cs_tiled_cube_face_sync_plan *plan = context->plan;
  const size_t radial = plan->tile.local_active[0];
  const size_t per_edge = radial * plan->max_along;
  for (size_t item = 0; item < count; ++item) {
    if (requested[item] >= 4U * per_edge) return -1;
    const int edge = (int)(requested[item] / per_edge);
    const size_t remainder = requested[item] % per_edge;
    const size_t i = remainder / plan->max_along;
    const size_t along = remainder % plan->max_along;
    const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
    const int along_axis = axis == 1 ? 2 : 1;
    if (along >= plan->tile.local_active[along_axis]) return -1;
    const size_t face = cube_edge_face_index(context->local,
        &plan->tile, plan->ghost, edge, i, along);
    double *out = &values[item * (size_t)channels];
    if (channels == 2) {
      out[0] = context->local->face_flux[axis][face];
      out[1] = context->local->old_face_flux[axis][face];
    } else {
      for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
        out[m] = context->local->fluid_face_flux[axis][
            face * GAMERA_NO_FLUX_COUNT + (size_t)m];
      for (int d = 0; d < 3; ++d)
        out[GAMERA_NO_FLUX_COUNT + d] =
            context->local->maxwell_face_flux[axis][face].value[d];
    }
  }
  return 0;
}

int gamera_cs_tiled_cube_face_sync_plan_create(
    const gamera_cs_decomposition *tile, size_t ghost, MPI_Comm world,
    gamera_cs_tiled_cube_face_sync_plan **out) {
  int rank = -1, ranks = 0;
  if (MPI_Comm_rank(world, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(world, &ranks) != MPI_SUCCESS) return -1;
  int valid = out != NULL && tile != NULL && ghost >= 4;
  if (valid) {
    gamera_cs_decomposition expected;
    valid = gamera_cs_decomposition_init(tile->global_active,
        tile->proc_dims, ranks, rank, &expected) == 0 &&
        expected.face == tile->face &&
        tile->global_active[1] == tile->global_active[2];
    for (int axis = 0; axis < 3 && valid; ++axis)
      valid &= expected.coord[axis] == tile->coord[axis] &&
          expected.local_active[axis] == tile->local_active[axis] &&
          expected.global_start[axis] == tile->global_start[axis];
  }
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;
  *out = NULL;
  gamera_cs_tiled_cube_face_sync_plan *plan = calloc(1, sizeof(*plan));
  valid = plan != NULL;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(plan);
    return -1;
  }
  plan->tile = *tile;
  plan->world = world;
  plan->ranks = ranks;
  plan->ghost = ghost;
  plan->max_along = tile->local_active[1] > tile->local_active[2] ?
      tile->local_active[1] : tile->local_active[2];
  valid = tile->local_active[0] <= SIZE_MAX / plan->max_along &&
      tile->local_active[0] * plan->max_along <= SIZE_MAX / 4U;
  const size_t local_cells = valid ?
      4U * tile->local_active[0] * plan->max_along : 0;
  for (int edge = 0; valid && edge < GAMERA_CS_EDGE_COUNT; ++edge) {
    const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
    const int at_boundary = edge % 2 == 0 ?
        tile->coord[axis] == 0 :
        tile->coord[axis] == tile->proc_dims[axis] - 1;
    if (!at_boundary) continue;
    const int along_axis = axis == 1 ? 2 : 1;
    const size_t increment = tile->local_active[0] *
        tile->local_active[along_axis];
    if (plan->count > SIZE_MAX - increment) valid = 0;
    else plan->count += increment;
  }
  valid &= plan->count <= SIZE_MAX / sizeof(*plan->entry) &&
      plan->count <= SIZE_MAX / sizeof(gamera_no_sparse_reference);
  gamera_no_sparse_reference *references = NULL;
  if (valid) {
    plan->entry = calloc(plan->count != 0 ? plan->count : 1U,
                          sizeof(*plan->entry));
    references = calloc(plan->count != 0 ? plan->count : 1U,
                         sizeof(*references));
    valid = plan->entry != NULL && references != NULL;
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(references);
    gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
    return -1;
  }
  size_t item = 0;
  for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
    const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
    const int at_boundary = edge % 2 == 0 ?
        tile->coord[axis] == 0 :
        tile->coord[axis] == tile->proc_dims[axis] - 1;
    if (!at_boundary) continue;
    gamera_cs_neighbor neighbor;
    if (gamera_cs_edge_neighbor(tile->face, edge, &neighbor) != 0) {
      valid = 0;
      break;
    }
    const int target_axis = neighbor.edge < GAMERA_CS_B_LOWER ? 1 : 2;
    const int src_along_axis = axis == 1 ? 2 : 1;
    const int target_along_axis = target_axis == 1 ? 2 : 1;
    for (size_t i = 0; i < tile->local_active[0]; ++i)
      for (size_t along = 0; along <
           tile->local_active[src_along_axis]; ++along) {
        const size_t global_r = tile->global_start[0] + i;
        const size_t global_along =
            tile->global_start[src_along_axis] + along;
        const size_t target_along = neighbor.reversed ?
            tile->global_active[src_along_axis] - 1U - global_along :
            global_along;
        size_t global_cell[3] = {global_r, 0, 0};
        global_cell[target_axis] = neighbor.edge % 2 == 0 ?
            0 : tile->global_active[target_axis] - 1U;
        global_cell[target_along_axis] = target_along;
        size_t local_cell[3];
        if (gamera_cs_decomposition_cell_owner(tile->global_active,
                tile->proc_dims, neighbor.face, global_cell,
                &references[item].donor_rank, local_cell) != 0) {
          valid = 0;
          local_cell[0] = local_cell[1] = local_cell[2] = 0;
        }
        references[item].donor_cell =
            (size_t)neighbor.edge *
                tile->local_active[0] * plan->max_along +
            local_cell[0] * plan->max_along +
            local_cell[target_along_axis];
        plan->entry[item].axis = axis;
        plan->entry[item].face = cube_edge_face_index(NULL, tile,
            ghost, edge, i, along);
        plan->entry[item].orientation =
            edge % 2 == neighbor.edge % 2 ? -1.0 : 1.0;
        ++item;
      }
  }
  valid &= item == plan->count;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(references);
    gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
    return -1;
  }
  const size_t channels_max = (size_t)GAMERA_NO_FLUX_COUNT + 3U;
  valid = plan->count <= (size_t)INT_MAX / channels_max &&
      (size_t)ranks <= SIZE_MAX / sizeof(int) &&
      (size_t)ranks <= SIZE_MAX / (2U * sizeof(MPI_Request));
  if (valid) {
    plan->reference_slot = calloc(plan->count != 0 ? plan->count : 1U,
                                  sizeof(*plan->reference_slot));
    plan->request_counts = calloc((size_t)ranks,
                                  sizeof(*plan->request_counts));
    plan->request_displacements = calloc((size_t)ranks,
                                  sizeof(*plan->request_displacements));
    plan->serve_counts = calloc((size_t)ranks,
                                sizeof(*plan->serve_counts));
    plan->serve_displacements = calloc((size_t)ranks,
                                       sizeof(*plan->serve_displacements));
    plan->requests = calloc(2U * (size_t)ranks,
                            sizeof(*plan->requests));
    valid = plan->reference_slot != NULL &&
        plan->request_counts != NULL &&
        plan->request_displacements != NULL &&
        plan->serve_counts != NULL &&
        plan->serve_displacements != NULL && plan->requests != NULL;
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(references);
    gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
    return -1;
  }
  for (size_t n = 0; n < plan->count; ++n) {
    const int peer = references[n].donor_rank;
    if (peer < 0 || peer >= ranks ||
        references[n].donor_cell >= local_cells) valid = 0;
    else ++plan->request_counts[peer];
  }
  int next = 0;
  for (int peer = 0; peer < ranks; ++peer) {
    plan->request_displacements[peer] = next;
    next += plan->request_counts[peer];
  }
  valid &= next == (int)plan->count;
  uint64_t *out_cells = NULL, *incoming_cells = NULL;
  int *positions = NULL;
  if (valid) {
    out_cells = calloc(plan->count != 0 ? plan->count : 1U,
                       sizeof(*out_cells));
    positions = calloc((size_t)ranks, sizeof(*positions));
    valid = out_cells != NULL && positions != NULL;
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(positions);
    free(out_cells);
    free(references);
    gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
    return -1;
  }
  for (int peer = 0; peer < ranks; ++peer)
    positions[peer] = plan->request_displacements[peer];
  for (size_t n = 0; n < plan->count; ++n) {
    const int peer = references[n].donor_rank;
    const size_t slot = (size_t)positions[peer]++;
    plan->reference_slot[n] = slot;
    out_cells[slot] = (uint64_t)references[n].donor_cell;
  }
  free(positions);
  if (MPI_Alltoall(plan->request_counts, 1, MPI_INT,
                   plan->serve_counts, 1, MPI_INT, world) != MPI_SUCCESS) {
    free(out_cells);
    free(references);
    gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
    return -1;
  }
  size_t serve_total = 0;
  for (int peer = 0; peer < ranks; ++peer) {
    valid &= plan->serve_counts[peer] >= 0 &&
        plan->serve_counts[peer] <= INT_MAX / (int)channels_max &&
        serve_total <= (size_t)INT_MAX - (size_t)plan->serve_counts[peer];
    if (!valid) break;
    plan->serve_displacements[peer] = (int)serve_total;
    serve_total += (size_t)plan->serve_counts[peer];
  }
  plan->serve_total = serve_total;
  valid &= serve_total <= SIZE_MAX / sizeof(*plan->serve_cells) &&
      serve_total <= SIZE_MAX / channels_max / sizeof(double) &&
      plan->count <= SIZE_MAX / channels_max / sizeof(double);
  if (valid) {
    incoming_cells = calloc(serve_total != 0 ? serve_total : 1U,
                            sizeof(*incoming_cells));
    plan->serve_cells = calloc(serve_total != 0 ? serve_total : 1U,
                               sizeof(*plan->serve_cells));
    plan->send_values = calloc(serve_total != 0 ?
        serve_total * channels_max : 1U, sizeof(*plan->send_values));
    plan->receive_values = calloc(plan->count != 0 ?
        plan->count * channels_max : 1U, sizeof(*plan->receive_values));
    valid = incoming_cells != NULL && plan->serve_cells != NULL &&
        plan->send_values != NULL && plan->receive_values != NULL;
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(incoming_cells);
    free(out_cells);
    free(references);
    gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
    return -1;
  }
  if (MPI_Alltoallv(out_cells, plan->request_counts,
                    plan->request_displacements, MPI_UINT64_T,
                    incoming_cells, plan->serve_counts,
                    plan->serve_displacements, MPI_UINT64_T,
                    world) != MPI_SUCCESS) valid = 0;
  for (size_t n = 0; valid && n < serve_total; ++n) {
    if (incoming_cells[n] >= local_cells) valid = 0;
    else plan->serve_cells[n] = (size_t)incoming_cells[n];
  }
  free(incoming_cells);
  free(out_cells);
  free(references);
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    gamera_cs_tiled_cube_face_sync_plan_destroy(plan);
    return -1;
  }
  *out = plan;
  return 0;
}

static int sync_cube_faces_sparse(
    gamera_cs_tiled_cube_face_sync_plan *plan,
    gamera_no_storage *local, int magnetic) {
  if (plan == NULL || local == NULL) return -1;
  const tiled_cube_face_pack_context context = {plan, local};
  const int channels = magnetic ? 2 : GAMERA_NO_FLUX_COUNT + 3;
  int valid = pack_tiled_cube_faces(plan->serve_cells,
      plan->serve_total, channels, plan->send_values,
      (void *)&context) == 0;
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    plan->world) != MPI_SUCCESS || !all_valid) return -1;
  int requests = 0;
  const int tag = magnetic ? 2601 : 2602;
  for (int peer = 0; peer < plan->ranks; ++peer) {
    if (plan->request_counts[peer] == 0) continue;
    if (MPI_Irecv(plan->receive_values +
            (size_t)plan->request_displacements[peer] * channels,
            plan->request_counts[peer] * channels, MPI_DOUBLE,
            peer, tag, plan->world,
            &plan->requests[requests++]) != MPI_SUCCESS) return -1;
  }
  for (int peer = 0; peer < plan->ranks; ++peer) {
    if (plan->serve_counts[peer] == 0) continue;
    if (MPI_Isend(plan->send_values +
            (size_t)plan->serve_displacements[peer] * channels,
            plan->serve_counts[peer] * channels, MPI_DOUBLE,
            peer, tag, plan->world,
            &plan->requests[requests++]) != MPI_SUCCESS) return -1;
  }
  if (MPI_Waitall(requests, plan->requests,
                  MPI_STATUSES_IGNORE) != MPI_SUCCESS) return -1;
  for (size_t item = 0; item < plan->count; ++item) {
    const tiled_cube_face_entry *entry = &plan->entry[item];
    const int axis = entry->axis;
    const size_t face = entry->face;
    const double orientation = entry->orientation;
    const double *remote = &plan->receive_values[
        plan->reference_slot[item] * (size_t)channels];
    if (magnetic) {
      local->face_flux[axis][face] = 0.5 *
          (local->face_flux[axis][face] + orientation *
           remote[0]);
      local->old_face_flux[axis][face] = 0.5 *
          (local->old_face_flux[axis][face] + orientation *
           remote[1]);
    } else {
      for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
        double *value = &local->fluid_face_flux[axis][
            face * GAMERA_NO_FLUX_COUNT + (size_t)m];
        *value = 0.5 * (*value + orientation *
            remote[m]);
      }
      for (int d = 0; d < 3; ++d) {
        double *value = &local->maxwell_face_flux[axis][face].value[d];
        *value = 0.5 * (*value + orientation *
            remote[GAMERA_NO_FLUX_COUNT + d]);
      }
    }
  }
  return 0;
}

int gamera_cs_tiled_cube_face_sync_magnetic(
    gamera_cs_tiled_cube_face_sync_plan *plan, gamera_no_storage *local) {
  return sync_cube_faces_sparse(plan, local, 1);
}

int gamera_cs_tiled_cube_face_sync_mhd(
    gamera_cs_tiled_cube_face_sync_plan *plan, gamera_no_storage *local) {
  return sync_cube_faces_sparse(plan, local, 0);
}

void gamera_cs_tiled_cube_face_sync_plan_destroy(
    gamera_cs_tiled_cube_face_sync_plan *plan) {
  if (plan == NULL) return;
  free(plan->reference_slot);
  free(plan->serve_cells);
  free(plan->request_counts);
  free(plan->request_displacements);
  free(plan->serve_counts);
  free(plan->serve_displacements);
  free(plan->send_values);
  free(plan->receive_values);
  free(plan->requests);
  free(plan->entry);
  free(plan);
}

/* Keep the endpoint identity, orientation and arithmetic averaging of
 * cubedsphere_interface.c. Only the ownership/routing is tile-specific. */
typedef struct {
  int64_t endpoint[8];
  int sign;
} tiled_edge_wire;

typedef struct {
  tiled_edge_wire wire;
  int direction, owner;
  size_t index;
} tiled_edge_local;

typedef struct {
  int64_t endpoint[8];
  size_t received_index;
} tiled_edge_sorted;

struct gamera_cs_tiled_edge_sync_plan {
  MPI_Comm world;
  int ranks;
  size_t count, receive_total;
  tiled_edge_local *local;
  size_t *send_item;
  tiled_edge_wire *received;
  tiled_edge_sorted *sorted;
  int *send_counts, *send_displacements;
  int *receive_counts, *receive_displacements;
  double *sent_values, *received_values, *common_values, *return_values;
  MPI_Request *requests;
};

static int tiled_edge_key_compare(const int64_t left[8],
                                  const int64_t right[8]) {
  for (int d = 0; d < 8; ++d) {
    if (left[d] < right[d]) return -1;
    if (left[d] > right[d]) return 1;
  }
  return 0;
}

static int tiled_edge_sorted_compare(const void *left, const void *right) {
  const tiled_edge_sorted *a = left, *b = right;
  return tiled_edge_key_compare(a->endpoint, b->endpoint);
}

static int tiled_edge_wire_create(const gamera_cs_decomposition *tile,
                                  size_t ghost, int direction, size_t i,
                                  size_t j, size_t k,
                                  tiled_edge_wire *wire) {
  size_t start_vertex[3] = {
      tile->global_start[0] + i - ghost,
      tile->global_start[1] + j - ghost,
      tile->global_start[2] + k - ghost};
  size_t end_vertex[3] = {
      start_vertex[0], start_vertex[1], start_vertex[2]};
  ++end_vertex[direction];
  int64_t start[4], finish[4];
  if (gamera_cs_vertex_topology_key(tile->face, tile->global_active,
          start_vertex, start) != 0 ||
      gamera_cs_vertex_topology_key(tile->face, tile->global_active,
          end_vertex, finish) != 0) return -1;
  int order = 0;
  for (int d = 0; d < 4 && order == 0; ++d)
    order = start[d] < finish[d] ? -1 :
            start[d] > finish[d] ? 1 : 0;
  if (order == 0) return -1;
  wire->sign = order < 0 ? 1 : -1;
  for (int d = 0; d < 4; ++d) {
    wire->endpoint[d] = order < 0 ? start[d] : finish[d];
    wire->endpoint[d + 4] = order < 0 ? finish[d] : start[d];
  }
  return 0;
}

static int tiled_edge_owner(const tiled_edge_wire *wire, int ranks) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (int d = 0; d < 8; ++d) {
    hash ^= (uint64_t)wire->endpoint[d];
    hash *= UINT64_C(1099511628211);
  }
  return (int)(hash % (uint64_t)ranks);
}

static int collect_tiled_edges(const gamera_no_grid *grid,
                               const gamera_cs_decomposition *tile,
                               size_t ghost, tiled_edge_local *items,
                               size_t capacity, size_t *count) {
  size_t n = 0;
  for (int d = 0; d < 3; ++d) {
    const size_t upper_i = ghost + tile->local_active[0] + (d != 0);
    const size_t upper_j = ghost + tile->local_active[1] + (d != 1);
    const size_t upper_k = ghost + tile->local_active[2] + (d != 2);
    for (size_t i = ghost; i < upper_i; ++i)
      for (size_t j = ghost; j < upper_j; ++j)
        for (size_t k = ghost; k < upper_k; ++k) {
          const size_t global_j = tile->global_start[1] + j - ghost;
          const size_t global_k = tile->global_start[2] + k - ghost;
          const int on_a = global_j == 0 ||
              global_j == tile->global_active[1];
          const int on_b = global_k == 0 ||
              global_k == tile->global_active[2];
          const int shared = d == 0 ? (on_a || on_b) :
                             d == 1 ? on_b : on_a;
          if (!shared) continue;
          /* Same-face tile-boundary copies are propagated by the existing
           * internal edge sync after the physical cube-edge reconciliation.
           * Take only the lower tile's copy in each transverse dimension. */
          if ((d != 0 && i == ghost && tile->coord[0] != 0) ||
              (d != 1 && j == ghost && tile->coord[1] != 0) ||
              (d != 2 && k == ghost && tile->coord[2] != 0)) continue;
          if (items != NULL) {
            if (n >= capacity || tiled_edge_wire_create(tile, ghost,
                    d, i, j, k,
                    &items[n].wire) != 0) return -1;
            items[n].direction = d;
            items[n].index = gamera_no_index3(grid->edge[d].extent,
                                               i, j, k);
          }
          if (n == SIZE_MAX) return -1;
          ++n;
        }
  }
  *count = n;
  return 0;
}

void gamera_cs_tiled_edge_sync_plan_destroy(
    gamera_cs_tiled_edge_sync_plan *plan) {
  if (plan == NULL) return;
  free(plan->local);
  free(plan->send_item);
  free(plan->received);
  free(plan->sorted);
  free(plan->send_counts);
  free(plan->send_displacements);
  free(plan->receive_counts);
  free(plan->receive_displacements);
  free(plan->sent_values);
  free(plan->received_values);
  free(plan->common_values);
  free(plan->return_values);
  free(plan->requests);
  free(plan);
}

int gamera_cs_tiled_edge_sync_plan_create(
    const gamera_no_grid *grid, const gamera_cs_decomposition *tile,
    size_t ghost, MPI_Comm world, gamera_cs_tiled_edge_sync_plan **out) {
  int rank = -1, ranks = 0;
  if (MPI_Comm_rank(world, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(world, &ranks) != MPI_SUCCESS) return -1;
  int valid = grid != NULL && tile != NULL && out != NULL && ghost >= 4;
  if (valid) {
    gamera_cs_decomposition expected;
    valid = grid->vertex != NULL &&
        tile->global_active[1] == tile->global_active[2] &&
        gamera_cs_decomposition_init(tile->global_active,
            tile->proc_dims, ranks, rank, &expected) == 0 &&
        expected.face == tile->face;
    for (int d = 0; d < 3 && valid; ++d)
      valid &= expected.coord[d] == tile->coord[d] &&
          expected.local_active[d] == tile->local_active[d] &&
          expected.global_start[d] == tile->global_start[d] &&
          grid->vertex_extent[d] ==
              tile->local_active[d] + 2U * ghost + 1U;
  }
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) return -1;
  *out = NULL;
  gamera_cs_tiled_edge_sync_plan *plan = calloc(1, sizeof(*plan));
  valid = plan != NULL;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(plan);
    return -1;
  }
  plan->world = world;
  plan->ranks = ranks;
  valid = collect_tiled_edges(grid, tile, ghost, NULL, 0,
                              &plan->count) == 0 &&
      plan->count <= (size_t)INT_MAX / sizeof(tiled_edge_wire) &&
      (size_t)ranks <= SIZE_MAX / (4U * sizeof(MPI_Request));
  if (valid) {
    plan->local = calloc(plan->count != 0 ? plan->count : 1U,
                          sizeof(*plan->local));
    plan->send_item = calloc(plan->count != 0 ? plan->count : 1U,
                              sizeof(*plan->send_item));
    plan->send_counts = calloc((size_t)ranks, sizeof(*plan->send_counts));
    plan->send_displacements = calloc((size_t)ranks,
                                       sizeof(*plan->send_displacements));
    plan->receive_counts = calloc((size_t)ranks,
                                   sizeof(*plan->receive_counts));
    plan->receive_displacements = calloc((size_t)ranks,
                                          sizeof(*plan->receive_displacements));
    plan->requests = calloc(4U * (size_t)ranks,
                            sizeof(*plan->requests));
    valid = plan->local != NULL && plan->send_item != NULL &&
        plan->send_counts != NULL && plan->send_displacements != NULL &&
        plan->receive_counts != NULL &&
        plan->receive_displacements != NULL && plan->requests != NULL;
  }
  size_t filled = 0;
  if (valid)
    valid = collect_tiled_edges(grid, tile, ghost, plan->local,
        plan->count, &filled) == 0 && filled == plan->count;
  if (valid) {
    for (size_t n = 0; n < plan->count; ++n) {
      const int owner = tiled_edge_owner(&plan->local[n].wire, ranks);
      plan->local[n].owner = owner;
      ++plan->send_counts[owner];
    }
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    gamera_cs_tiled_edge_sync_plan_destroy(plan);
    return -1;
  }
  int next = 0;
  for (int peer = 0; peer < ranks; ++peer) {
    plan->send_displacements[peer] = next;
    next += plan->send_counts[peer];
  }
  tiled_edge_wire *send_wire = calloc(plan->count != 0 ?
      plan->count : 1U, sizeof(*send_wire));
  int *positions = calloc((size_t)ranks, sizeof(*positions));
  valid = send_wire != NULL && positions != NULL;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(send_wire);
    free(positions);
    gamera_cs_tiled_edge_sync_plan_destroy(plan);
    return -1;
  }
  for (int peer = 0; peer < ranks; ++peer)
    positions[peer] = plan->send_displacements[peer];
  for (size_t n = 0; n < plan->count; ++n) {
    const size_t slot = (size_t)positions[plan->local[n].owner]++;
    send_wire[slot] = plan->local[n].wire;
    plan->send_item[slot] = n;
  }
  free(positions);
  if (MPI_Alltoall(plan->send_counts, 1, MPI_INT,
                   plan->receive_counts, 1, MPI_INT, world) != MPI_SUCCESS) {
    free(send_wire);
    gamera_cs_tiled_edge_sync_plan_destroy(plan);
    return -1;
  }
  size_t total = 0;
  for (int peer = 0; peer < ranks; ++peer) {
    valid &= plan->receive_counts[peer] >= 0 &&
        total <= (size_t)INT_MAX - (size_t)plan->receive_counts[peer];
    if (!valid) break;
    plan->receive_displacements[peer] = (int)total;
    total += (size_t)plan->receive_counts[peer];
  }
  plan->receive_total = total;
  valid &= total <= SIZE_MAX / sizeof(*plan->received) &&
      total <= SIZE_MAX / sizeof(*plan->sorted);
  if (valid) {
    plan->received = calloc(total != 0 ? total : 1U,
                             sizeof(*plan->received));
    plan->sorted = calloc(total != 0 ? total : 1U,
                           sizeof(*plan->sorted));
    plan->sent_values = calloc(plan->count != 0 ? plan->count : 1U,
                                sizeof(*plan->sent_values));
    plan->return_values = calloc(plan->count != 0 ? plan->count : 1U,
                                  sizeof(*plan->return_values));
    plan->received_values = calloc(total != 0 ? total : 1U,
                                    sizeof(*plan->received_values));
    plan->common_values = calloc(total != 0 ? total : 1U,
                                  sizeof(*plan->common_values));
    valid = plan->received != NULL && plan->sorted != NULL &&
        plan->sent_values != NULL && plan->return_values != NULL &&
        plan->received_values != NULL && plan->common_values != NULL;
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    free(send_wire);
    gamera_cs_tiled_edge_sync_plan_destroy(plan);
    return -1;
  }
  MPI_Datatype wire_type = MPI_DATATYPE_NULL;
  valid = MPI_Type_contiguous((int)sizeof(tiled_edge_wire), MPI_BYTE,
              &wire_type) == MPI_SUCCESS &&
          MPI_Type_commit(&wire_type) == MPI_SUCCESS;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    if (wire_type != MPI_DATATYPE_NULL) MPI_Type_free(&wire_type);
    free(send_wire);
    gamera_cs_tiled_edge_sync_plan_destroy(plan);
    return -1;
  }
  const int transfer = MPI_Alltoallv(send_wire, plan->send_counts,
      plan->send_displacements, wire_type, plan->received,
      plan->receive_counts, plan->receive_displacements,
      wire_type, world);
  MPI_Type_free(&wire_type);
  free(send_wire);
  valid = transfer == MPI_SUCCESS;
  for (size_t n = 0; valid && n < total; ++n) {
    if (plan->received[n].sign != 1 &&
        plan->received[n].sign != -1) valid = 0;
    memcpy(plan->sorted[n].endpoint, plan->received[n].endpoint,
           sizeof(plan->sorted[n].endpoint));
    plan->sorted[n].received_index = n;
  }
  if (valid) {
    qsort(plan->sorted, total, sizeof(*plan->sorted),
          tiled_edge_sorted_compare);
    for (size_t first = 0; first < total;) {
      size_t last = first + 1U;
      while (last < total && tiled_edge_key_compare(
              plan->sorted[first].endpoint,
              plan->sorted[last].endpoint) == 0) ++last;
      if (last - first < 2U || last - first > 3U) valid = 0;
      first = last;
    }
  }
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    world) != MPI_SUCCESS || !all_valid) {
    gamera_cs_tiled_edge_sync_plan_destroy(plan);
    return -1;
  }
  *out = plan;
  return 0;
}

int gamera_cs_tiled_edge_sync(
    gamera_cs_tiled_edge_sync_plan *plan, gamera_no_storage *local) {
  if (plan == NULL || local == NULL) return -1;
  int valid = 1;
  for (int d = 0; d < 3; ++d)
    valid &= local->edge_emf[d] != NULL;
  int all_valid = 0;
  if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN,
                    plan->world) != MPI_SUCCESS || !all_valid) return -1;
  for (size_t slot = 0; slot < plan->count; ++slot) {
    const tiled_edge_local *entry = &plan->local[plan->send_item[slot]];
    plan->sent_values[slot] =
        local->edge_emf[entry->direction][entry->index];
  }
  int requests = 0;
  for (int peer = 0; peer < plan->ranks; ++peer)
    if (plan->receive_counts[peer] != 0 &&
        MPI_Irecv(plan->received_values + plan->receive_displacements[peer],
            plan->receive_counts[peer], MPI_DOUBLE, peer, 2603,
            plan->world, &plan->requests[requests++]) != MPI_SUCCESS)
      return -1;
  for (int peer = 0; peer < plan->ranks; ++peer)
    if (plan->send_counts[peer] != 0 &&
        MPI_Isend(plan->sent_values + plan->send_displacements[peer],
            plan->send_counts[peer], MPI_DOUBLE, peer, 2603,
            plan->world, &plan->requests[requests++]) != MPI_SUCCESS)
      return -1;
  if (MPI_Waitall(requests, plan->requests,
                  MPI_STATUSES_IGNORE) != MPI_SUCCESS) return -1;
  for (size_t first = 0; first < plan->receive_total;) {
    size_t last = first + 1U;
    while (last < plan->receive_total && tiled_edge_key_compare(
            plan->sorted[first].endpoint,
            plan->sorted[last].endpoint) == 0) ++last;
    double common = 0.0;
    for (size_t t = first; t < last; ++t) {
      const size_t n = plan->sorted[t].received_index;
      common += (double)plan->received[n].sign *
          plan->received_values[n];
    }
    common /= (double)(last - first);
    for (size_t t = first; t < last; ++t) {
      const size_t n = plan->sorted[t].received_index;
      plan->common_values[n] = (double)plan->received[n].sign * common;
    }
    first = last;
  }
  requests = 0;
  for (int peer = 0; peer < plan->ranks; ++peer)
    if (plan->send_counts[peer] != 0 &&
        MPI_Irecv(plan->return_values + plan->send_displacements[peer],
            plan->send_counts[peer], MPI_DOUBLE, peer, 2604,
            plan->world, &plan->requests[requests++]) != MPI_SUCCESS)
      return -1;
  for (int peer = 0; peer < plan->ranks; ++peer)
    if (plan->receive_counts[peer] != 0 &&
        MPI_Isend(plan->common_values + plan->receive_displacements[peer],
            plan->receive_counts[peer], MPI_DOUBLE, peer, 2604,
            plan->world, &plan->requests[requests++]) != MPI_SUCCESS)
      return -1;
  if (MPI_Waitall(requests, plan->requests,
                  MPI_STATUSES_IGNORE) != MPI_SUCCESS) return -1;
  for (size_t slot = 0; slot < plan->count; ++slot) {
    const tiled_edge_local *entry = &plan->local[plan->send_item[slot]];
    local->edge_emf[entry->direction][entry->index] =
        plan->return_values[slot];
  }
  return 0;
}
