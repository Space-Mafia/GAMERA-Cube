#include "cubedsphere_decomposition.h"

#include <limits.h>
#include <stdint.h>

static int patch_size(const int dims[3], int *size) {
  if (dims == NULL || size == NULL || dims[0] < 1 || dims[1] < 1 ||
      dims[2] < 1 || dims[0] > INT_MAX / dims[1] ||
      dims[0] * dims[1] > INT_MAX / dims[2]) return -1;
  *size = dims[0] * dims[1] * dims[2];
  return 0;
}

int gamera_cs_decomposition_rank(int face, const int coord[3],
                                 const int proc_dims[3], int *world_rank) {
  int count = 0;
  if (world_rank == NULL || coord == NULL ||
      face < 0 || face >= GAMERA_CS_FACE_COUNT ||
      patch_size(proc_dims, &count) != 0 ||
      count > INT_MAX / GAMERA_CS_FACE_COUNT) return -1;
  for (int axis = 0; axis < 3; ++axis)
    if (coord[axis] < 0 || coord[axis] >= proc_dims[axis]) return -1;
  *world_rank = face * count +
      (coord[0] * proc_dims[1] + coord[1]) * proc_dims[2] + coord[2];
  return 0;
}

int gamera_cs_decomposition_cell_owner(
    const size_t global_active[3], const int proc_dims[3], int face,
    const size_t global_index[3], int *world_rank,
    size_t local_index[3]) {
  if (global_active == NULL || proc_dims == NULL || global_index == NULL ||
      local_index == NULL || world_rank == NULL ||
      global_active[1] != global_active[2]) return -1;
  int coord[3];
  for (int axis = 0; axis < 3; ++axis) {
    if (proc_dims[axis] < 1 ||
        global_active[axis] % (size_t)proc_dims[axis] != 0 ||
        global_index[axis] >= global_active[axis]) return -1;
    const size_t width = global_active[axis] / (size_t)proc_dims[axis];
    if (width < 8U) return -1;
    coord[axis] = (int)(global_index[axis] / width);
    local_index[axis] = global_index[axis] % width;
  }
  return gamera_cs_decomposition_rank(face, coord, proc_dims, world_rank);
}

int gamera_cs_decomposition_init(const size_t global_active[3],
                                 const int proc_dims[3], int world_size,
                                 int world_rank,
                                 gamera_cs_decomposition *result) {
  int count = 0;
  if (result == NULL || global_active == NULL ||
      patch_size(proc_dims, &count) != 0 ||
      count > INT_MAX / GAMERA_CS_FACE_COUNT ||
      world_size != GAMERA_CS_FACE_COUNT * count ||
      world_rank < 0 || world_rank >= world_size ||
      global_active[1] != global_active[2]) return -1;
  for (int axis = 0; axis < 3; ++axis) {
    if (global_active[axis] > (size_t)INT_MAX ||
        global_active[axis] % (size_t)proc_dims[axis] != 0 ||
        global_active[axis] / (size_t)proc_dims[axis] < 8U)
      return -1;
    result->proc_dims[axis] = proc_dims[axis];
    result->global_active[axis] = global_active[axis];
    result->local_active[axis] =
        global_active[axis] / (size_t)proc_dims[axis];
  }
  result->face = world_rank / count;
  int patch_rank = world_rank % count;
  result->coord[2] = patch_rank % proc_dims[2];
  patch_rank /= proc_dims[2];
  result->coord[1] = patch_rank % proc_dims[1];
  result->coord[0] = patch_rank / proc_dims[1];
  for (int axis = 0; axis < 3; ++axis)
    result->global_start[axis] =
        (size_t)result->coord[axis] * result->local_active[axis];
  return 0;
}
