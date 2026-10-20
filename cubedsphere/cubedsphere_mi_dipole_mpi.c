#include "cubedsphere_mi_dipole_mpi.h"

#include "cubedsphere_mesh.h"
#include "nonorthogonal_background.h"
#include "nonorthogonal_grid.h"
#include "nonorthogonal_storage.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct gamera_cs_mi_dipole_mpi {
  gamera_cs_decomposition tile;
  MPI_Comm communicator;
  int rank, ranks;
  size_t ghost, shell_first;
  int send_count;
  int *recv_count, *displacement;
  double *send, *received, *potential;
  size_t potential_count;
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT];
  gamera_no_storage storage[GAMERA_CS_FACE_COUNT];
  gamera_no_background_data owned[GAMERA_CS_FACE_COUNT];
  gamera_no_background_field background[GAMERA_CS_FACE_COUNT];
  gamera_cs_mi_dipole *mi;
};

static int collective_error(int local, MPI_Comm comm) {
  int global = 0;
  return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm) == MPI_SUCCESS
      ? global : 1;
}

static size_t shell_overlap(const gamera_cs_decomposition *tile,
                            size_t first, size_t *start) {
  const size_t begin = tile->global_start[0] > first
      ? tile->global_start[0] : first;
  const size_t end_tile = tile->global_start[0] + tile->local_active[0];
  const size_t end_shell = first + 3U;
  const size_t end = end_tile < end_shell ? end_tile : end_shell;
  *start = begin;
  return begin < end ? end - begin : 0U;
}

static int shell_send_count(const gamera_cs_decomposition *tile,
                            size_t first, int *count) {
  size_t start = 0;
  const size_t radial = shell_overlap(tile, first, &start);
  (void)start;
  if (tile->local_active[1] > SIZE_MAX / tile->local_active[2] ||
      radial > SIZE_MAX / tile->local_active[1] /
                   tile->local_active[2] / 3U)
    return -1;
  const size_t n = radial * tile->local_active[1] *
                   tile->local_active[2] * 3U;
  if (n > INT_MAX) return -1;
  *count = (int)n;
  return 0;
}

void gamera_cs_mi_dipole_mpi_destroy(gamera_cs_mi_dipole_mpi *state) {
  if (state == NULL) return;
  gamera_cs_mi_dipole_destroy(state->mi);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_background_destroy(&state->owned[face]);
    gamera_no_storage_destroy(&state->storage[face]);
    gamera_no_grid_destroy(&state->grid[face]);
  }
  free(state->potential);
  free(state->received);
  free(state->send);
  free(state->recv_count);
  free(state->displacement);
  free(state);
}

static int create_root_shell(gamera_cs_mi_dipole_mpi *s,
                             const gamera_no_radial_map *map,
                             const gamera_cs_earth_startup *startup,
                             const gamera_cs_mi_dipole_config *config) {
  const size_t shell_active[3] = {
      3U, s->tile.global_active[1], s->tile.global_active[2]};
  const size_t start[3] = {s->shell_first, 0U, 0U};
  const size_t extent[3] = {shell_active[0] + 2U * s->ghost,
      shell_active[1] + 2U * s->ghost,
      shell_active[2] + 2U * s->ghost};
  const size_t vertex_extent[3] = {
      extent[0] + 1U, extent[1] + 1U, extent[2] + 1U};
  if (vertex_extent[0] > SIZE_MAX / vertex_extent[1] ||
      vertex_extent[0] * vertex_extent[1] >
          SIZE_MAX / vertex_extent[2] / sizeof(gamera_no_vec3)) return -1;
  const size_t vertices_count = vertex_extent[0] *
      vertex_extent[1] * vertex_extent[2];
  gamera_no_vec3 *vertices = calloc(vertices_count, sizeof(*vertices));
  if (vertices == NULL) return -1;
  int failed = 0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT && !failed; ++face) {
    if (gamera_cs_generate_vertices_radial_map_tile(
            face, s->tile.global_active, start, shell_active,
            s->ghost, map, vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &s->grid[face]) != 0 ||
        gamera_cs_calibrate_face_frames(&s->grid[face]) != 0 ||
        gamera_no_storage_create(extent, &s->storage[face]) != 0 ||
        gamera_no_background_create(&s->grid[face],
            gamera_cs_earth_startup_background, (void *)startup,
            &s->owned[face]) != 0) failed = 1;
    s->background[face] = s->owned[face].field;
  }
  free(vertices);
  if (failed) return -1;
  gamera_cs_mi_dipole_config shell_config = *config;
  shell_config.current_shell = 1U;
  return gamera_cs_mi_dipole_create(s->grid, s->storage,
      s->background, startup, shell_active, s->ghost, &shell_config, &s->mi);
}

int gamera_cs_mi_dipole_mpi_create(
    const gamera_cs_decomposition *tile, size_t ghost,
    const gamera_no_radial_map *radial_map,
    const gamera_cs_earth_startup *startup,
    const gamera_cs_mi_dipole_config *config, MPI_Comm communicator,
    gamera_cs_mi_dipole_mpi **result) {
  if (result == NULL) return -1;
  *result = NULL;
  int rank = -1, ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS) return -1;
  int bad = tile == NULL || radial_map == NULL || startup == NULL ||
      config == NULL || ghost < 4U || config->current_shell < 1U ||
      config->current_shell + 1U >= tile->global_active[0] ||
      tile->global_active[1] != tile->global_active[2] ||
      config->colatitude_points == 0U ||
      config->azimuth_points > SIZE_MAX / config->colatitude_points / 2U ||
      config->azimuth_points > (size_t)INT_MAX /
                                  config->colatitude_points / 2U;
  if (collective_error(bad, communicator)) return -1;
  gamera_cs_mi_dipole_mpi *s = calloc(1, sizeof(*s));
  bad = s == NULL;
  if (collective_error(bad, communicator)) { free(s); return -1; }
  s->tile = *tile;
  s->rank = rank;
  s->ranks = ranks;
  s->communicator = communicator;
  s->ghost = ghost;
  s->shell_first = config->current_shell - 1U;
  s->potential_count = 2U * config->azimuth_points *
      config->colatitude_points;
  bad = shell_send_count(tile, s->shell_first, &s->send_count) != 0;
  if (collective_error(bad, communicator)) goto failed;
  s->send = calloc((size_t)(s->send_count > 0 ? s->send_count : 1),
                   sizeof(*s->send));
  s->potential = calloc(s->potential_count, sizeof(*s->potential));
  bad = s->send == NULL || s->potential == NULL;
  if (rank == 0 && !bad) {
    s->recv_count = calloc((size_t)ranks, sizeof(int));
    s->displacement = calloc((size_t)ranks, sizeof(int));
    bad = s->recv_count == NULL || s->displacement == NULL;
    int total = 0;
    for (int r = 0; r < ranks && !bad; ++r) {
      gamera_cs_decomposition other;
      int count = 0;
      if (gamera_cs_decomposition_init(tile->global_active,
              tile->proc_dims, ranks, r, &other) != 0 ||
          shell_send_count(&other, s->shell_first, &count) != 0 ||
          count > INT_MAX - total) bad = 1;
      else {
        s->recv_count[r] = count;
        s->displacement[r] = total;
        total += count;
      }
    }
    if (!bad) {
      s->received = calloc((size_t)(total > 0 ? total : 1),
                            sizeof(*s->received));
      bad = s->received == NULL;
    }
  }
  if (collective_error(bad, communicator)) goto failed;
  bad = rank == 0
      ? create_root_shell(s, radial_map, startup, config) != 0
      : gamera_cs_mi_dipole_create_potential_view(startup, config,
                                                  &s->mi) != 0;
  if (collective_error(bad, communicator)) goto failed;
  *result = s;
  return 0;
failed:
  gamera_cs_mi_dipole_mpi_destroy(s);
  return -1;
}

static void pack_local(gamera_cs_mi_dipole_mpi *s,
                       const gamera_no_storage *local) {
  size_t start = 0;
  const size_t layers = shell_overlap(&s->tile, s->shell_first, &start);
  size_t q = 0;
  const size_t *extent = local->cell_extent;
  for (size_t gi = start; gi < start + layers; ++gi)
    for (size_t j = 0; j < s->tile.local_active[1]; ++j)
      for (size_t k = 0; k < s->tile.local_active[2]; ++k) {
        const size_t cell = gamera_no_index3(extent,
            s->ghost + gi - s->tile.global_start[0], s->ghost + j,
            s->ghost + k);
        for (int d = 0; d < 3; ++d)
          s->send[q++] = local->cell_magnetic[cell].value[d];
      }
}

static int unpack_root(gamera_cs_mi_dipole_mpi *s) {
  for (int rank = 0; rank < s->ranks; ++rank) {
    gamera_cs_decomposition other;
    if (gamera_cs_decomposition_init(s->tile.global_active,
            s->tile.proc_dims, s->ranks, rank, &other) != 0) return -1;
    size_t start = 0;
    const size_t layers = shell_overlap(&other, s->shell_first, &start);
    size_t q = (size_t)s->displacement[rank];
    gamera_no_storage *dst = &s->storage[other.face];
    for (size_t gi = start; gi < start + layers; ++gi)
      for (size_t j = 0; j < other.local_active[1]; ++j)
        for (size_t k = 0; k < other.local_active[2]; ++k) {
          const size_t cell = gamera_no_index3(dst->cell_extent,
              s->ghost + gi - s->shell_first,
              s->ghost + other.global_start[1] + j,
              s->ghost + other.global_start[2] + k);
          for (int d = 0; d < 3; ++d)
            dst->cell_magnetic[cell].value[d] = s->received[q++];
        }
    if (q != (size_t)s->displacement[rank] +
             (size_t)s->recv_count[rank]) return -1;
  }
  return 0;
}

int gamera_cs_mi_dipole_mpi_maybe_update(
    gamera_cs_mi_dipole_mpi *s, const gamera_no_storage *local,
    double time_code) {
  if (s == NULL || local == NULL) return -1;
  int due = 0;
  if (s->rank == 0)
    due = gamera_cs_mi_dipole_update_due(s->mi, time_code);
  if (MPI_Bcast(&due, 1, MPI_INT, 0, s->communicator) != MPI_SUCCESS ||
      due < 0) return -1;
  if (!due) return 0;
  int bad = local->cell_magnetic == NULL ||
      local->cell_extent[0] < s->tile.local_active[0] + 2U * s->ghost ||
      local->cell_extent[1] < s->tile.local_active[1] + 2U * s->ghost ||
      local->cell_extent[2] < s->tile.local_active[2] + 2U * s->ghost;
  if (collective_error(bad, s->communicator)) return -1;
  pack_local(s, local);
  if (MPI_Gatherv(s->send, s->send_count, MPI_DOUBLE, s->received,
          s->recv_count, s->displacement, MPI_DOUBLE, 0,
          s->communicator) != MPI_SUCCESS) return -1;
  bad = 0;
  double timing[2] = {0.0, 0.0};
  if (s->rank == 0) {
    bad = unpack_root(s) != 0 ||
        gamera_cs_mi_dipole_maybe_update(s->mi, time_code) != 1 ||
        gamera_cs_mi_dipole_export_potential(s->mi, s->potential,
            s->potential_count, &timing[0], &timing[1]) != 0;
  }
  if (collective_error(bad, s->communicator)) return -1;
  if (MPI_Bcast(timing, 2, MPI_DOUBLE, 0, s->communicator) != MPI_SUCCESS ||
      MPI_Bcast(s->potential, (int)s->potential_count, MPI_DOUBLE, 0,
          s->communicator) != MPI_SUCCESS) return -1;
  bad = s->rank != 0 && gamera_cs_mi_dipole_import_potential(
      s->mi, s->potential, s->potential_count,
      timing[0], timing[1]) != 0;
  return collective_error(bad, s->communicator) ? -1 : 1;
}

const gamera_cs_mi_dipole *gamera_cs_mi_dipole_mpi_state(
    const gamera_cs_mi_dipole_mpi *state) {
  return state != NULL ? state->mi : NULL;
}
