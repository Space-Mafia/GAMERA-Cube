#include "cubedsphere_mpi_tile.h"
#include "cubedsphere_mpi_step.h"
#include "nonorthogonal_state.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_edge_key(const gamera_no_grid *grid, int direction,
                         size_t i, size_t j, size_t k,
                         int64_t key[6], int *sign) {
  size_t end[3] = {i, j, k};
  ++end[direction];
  const gamera_no_vec3 a = grid->vertex[gamera_no_index3(
      grid->vertex_extent, i, j, k)];
  const gamera_no_vec3 b = grid->vertex[gamera_no_index3(
      grid->vertex_extent, end[0], end[1], end[2])];
  int64_t first[3], last[3];
  for (int d = 0; d < 3; ++d) {
    first[d] = llround(a.value[d] * 1e10);
    last[d] = llround(b.value[d] * 1e10);
  }
  int order = 0;
  for (int d = 0; d < 3 && order == 0; ++d)
    order = first[d] < last[d] ? -1 :
            first[d] > last[d] ? 1 : 0;
  if (order == 0) return -1;
  *sign = order < 0 ? 1 : -1;
  for (int d = 0; d < 3; ++d) {
    key[d] = order < 0 ? first[d] : last[d];
    key[d + 3] = order < 0 ? last[d] : first[d];
  }
  return 0;
}

static double sample(int face, int gi, int gj, int gk, int channel) {
  return 100000000.0 * face + 100000.0 * gi +
      1000.0 * gj + 10.0 * gk + channel;
}

static double physical_sample(int face, int gi, int gj, int gk,
                              int channel) {
  const double position = 0.1 * face + 0.001 * gi +
                          0.0001 * gj + 0.00001 * gk;
  if (channel == GAMERA_NO_FLUX_DENSITY) return 2.0 + position;
  if (channel == GAMERA_NO_FLUX_ENERGY) return 10.0 + position;
  if (channel < GAMERA_NO_FLUX_COUNT)
    return 0.01 * (channel + 1) + 0.001 * position;
  return 0.2 + 0.01 * channel + 0.001 * position;
}

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
  int rank = -1, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (argc != 4 && argc != 7) return 1;
  const int dims[3] = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
  size_t global[3] = {16, 16, 16};
  if (argc == 7)
    for (int axis = 0; axis < 3; ++axis) {
      char *end = NULL;
      const unsigned long parsed = strtoul(argv[4 + axis], &end, 10);
      if (argv[4 + axis][0] == '-' || end == argv[4 + axis] ||
          *end != '\0' || parsed == 0) return 1;
      global[axis] = (size_t)parsed;
    }
  const size_t ghost = 4;
  gamera_cs_decomposition tile;
  if (gamera_cs_decomposition_init(global, dims, ranks, rank, &tile) != 0)
    return 1;
  const size_t extent[3] = {
      tile.local_active[0] + 2 * ghost,
      tile.local_active[1] + 2 * ghost,
      tile.local_active[2] + 2 * ghost};
  gamera_no_storage storage = {0};
  if (gamera_no_storage_create(extent, &storage) != 0) return 1;
  const size_t count = gamera_no_element_count3(extent);
  for (size_t cell = 0; cell < count; ++cell) {
    for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
      storage.conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] = -777.0;
      storage.old_conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] = -777.0;
    }
    for (int d = 0; d < 3; ++d) {
      storage.cell_magnetic[cell].value[d] = -777.0;
      storage.old_cell_magnetic[cell].value[d] = -777.0;
    }
  }
  for (size_t i = ghost; i < ghost + tile.local_active[0]; ++i)
    for (size_t j = ghost; j < ghost + tile.local_active[1]; ++j)
      for (size_t k = ghost; k < ghost + tile.local_active[2]; ++k) {
        const int gi = (int)(tile.global_start[0] + i - ghost);
        const int gj = (int)(tile.global_start[1] + j - ghost);
        const int gk = (int)(tile.global_start[2] + k - ghost);
        const size_t cell = gamera_no_index3(extent, i, j, k);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          storage.conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              sample(tile.face, gi, gj, gk, m);
          storage.old_conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              sample(tile.face, gi, gj, gk, m) + 0.5;
        }
        for (int d = 0; d < 3; ++d) {
          storage.cell_magnetic[cell].value[d] =
              sample(tile.face, gi, gj, gk, GAMERA_NO_FLUX_COUNT + d);
          storage.old_cell_magnetic[cell].value[d] =
              sample(tile.face, gi, gj, gk, GAMERA_NO_FLUX_COUNT + d) + 0.5;
        }
      }
  int failed = gamera_cs_mpi_exchange_internal_cell_halos(
      &storage, &tile, ghost, MPI_COMM_WORLD) != 0;
  for (size_t i = 0; !failed && i < extent[0]; ++i)
    for (size_t j = 0; !failed && j < extent[1]; ++j)
      for (size_t k = 0; !failed && k < extent[2]; ++k) {
        const int gi = (int)tile.global_start[0] + (int)i - (int)ghost;
        const int gj = (int)tile.global_start[1] + (int)j - (int)ghost;
        const int gk = (int)tile.global_start[2] + (int)k - (int)ghost;
        if (gi < 0 || gj < 0 || gk < 0 ||
            gi >= (int)global[0] || gj >= (int)global[1] ||
            gk >= (int)global[2]) continue;
        const size_t cell = gamera_no_index3(extent, i, j, k);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          if (storage.conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] !=
                  sample(tile.face, gi, gj, gk, m) ||
              storage.old_conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] !=
                  sample(tile.face, gi, gj, gk, m) + 0.5) failed = 1;
        }
        for (int d = 0; d < 3; ++d)
          if (storage.cell_magnetic[cell].value[d] !=
                  sample(tile.face, gi, gj, gk, GAMERA_NO_FLUX_COUNT + d) ||
              storage.old_cell_magnetic[cell].value[d] !=
              sample(tile.face, gi, gj, gk, GAMERA_NO_FLUX_COUNT + d) + 0.5)
            failed = 1;
      }
  /* Reuse the accepted Yin-Yang sparse MPI exchange for cube-face donors.
   * The first point crosses a face seam; the second straddles angular tiles. */
  const double quarter_pi = 0.25 * acos(-1.0);
  gamera_no_vec3 point;
  gamera_cs_tiled_donor_stencil stencils[2];
  if (gamera_cs_direction(tile.face, quarter_pi + 0.01, 0.1,
                          &point) != 0 ||
      gamera_cs_tiled_donor_stencil_create(
          point, global, dims, tile.global_start[0] +
          tile.local_active[0] / 2U, 0, &stencils[0]) != 0 ||
      stencils[0].donor_face == tile.face ||
      gamera_cs_direction(tile.face, 0.0, 0.0, &point) != 0 ||
      gamera_cs_tiled_donor_stencil_create(
          point, global, dims, tile.global_start[0] +
          tile.local_active[0] / 2U, 0, &stencils[1]) != 0)
    failed = 1;
  int stencil_failed = 0;
  if (MPI_Allreduce(&failed, &stencil_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  if (stencil_failed && failed)
    fprintf(stderr, "rank %d: donor stencil or preceding halo check failed\n",
            rank);
  if (stencil_failed) failed = 1;
  gamera_cs_tiled_sparse_plan *sparse = NULL;
  if (!failed && gamera_cs_tiled_sparse_plan_create(
          stencils, 2, &tile, MPI_COMM_WORLD, &sparse) != 0) {
    fprintf(stderr, "rank %d: sparse donor plan create failed\n", rank);
    failed = 1;
  }
  if (!failed && gamera_cs_tiled_sparse_plan_exchange(
          sparse, &storage, &tile, ghost) != 0) {
    fprintf(stderr, "rank %d: sparse donor exchange failed\n", rank);
    failed = 1;
  }
  for (size_t item = 0; !failed && item < 2; ++item)
    for (size_t p = 0; !failed && p < 8; ++p)
      for (size_t q = 0; !failed && q < 8; ++q) {
        const int gi = (int)(tile.global_start[0] +
            tile.local_active[0] / 2U);
        const int gj = (int)(stencils[item].start[0] + p);
        const int gk = (int)(stencils[item].start[1] + q);
        for (int channel = 0; channel < GAMERA_CS_TILE_DONOR_CHANNELS;
             ++channel) {
          double value = 0.0;
          const int old = (channel >= GAMERA_NO_FLUX_COUNT &&
              channel < 2 * GAMERA_NO_FLUX_COUNT) ||
              channel >= 2 * GAMERA_NO_FLUX_COUNT + 3;
          const int base = channel < GAMERA_NO_FLUX_COUNT ? channel :
              channel < 2 * GAMERA_NO_FLUX_COUNT ?
              channel - GAMERA_NO_FLUX_COUNT :
              GAMERA_NO_FLUX_COUNT +
              (channel - 2 * GAMERA_NO_FLUX_COUNT) % 3;
          const double expected = sample(stencils[item].donor_face,
              gi, gj, gk, base) + (old ? 0.5 : 0.0);
          if (gamera_cs_tiled_sparse_plan_value(sparse, item, p, q,
                  channel, &value) != 0 || value != expected) {
            fprintf(stderr, "rank %d: sparse donor value item=%zu p=%zu "
                    "q=%zu channel=%d got=%.17g expected=%.17g\n",
                    rank, item, p, q, channel, value, expected);
            failed = 1;
          }
        }
      }
  gamera_cs_tiled_sparse_plan_destroy(sparse);
  for (int axis = 0; axis < 3; ++axis) {
    const size_t faces = gamera_no_element_count3(storage.face_extent[axis]);
    for (size_t face = 0; face < faces; ++face) {
      storage.face_flux[axis][face] = 1000.0 * rank + axis;
      storage.old_face_flux[axis][face] = 1000.0 * rank + axis + 0.5;
      for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
        storage.fluid_face_flux[axis][
            face * GAMERA_NO_FLUX_COUNT + (size_t)m] =
            1000.0 * rank + axis + 0.01 * m;
      for (int d = 0; d < 3; ++d)
        storage.maxwell_face_flux[axis][face].value[d] =
            1000.0 * rank + axis + 0.1 * d;
    }
    const size_t edges = gamera_no_element_count3(storage.edge_extent[axis]);
    for (size_t edge = 0; edge < edges; ++edge)
      storage.edge_emf[axis][edge] = 1000.0 * rank + axis;
  }
  if (gamera_cs_mpi_sync_internal_magnetic_faces(
          &storage, &tile, ghost, MPI_COMM_WORLD) != 0 ||
      gamera_cs_mpi_exchange_internal_magnetic_face_halos(
          &storage, &tile, ghost, MPI_COMM_WORLD) != 0 ||
      gamera_cs_mpi_sync_internal_mhd_faces(
          &storage, &tile, ghost, MPI_COMM_WORLD) != 0 ||
      gamera_cs_mpi_sync_internal_edge_emf(
          &storage, &tile, ghost, MPI_COMM_WORLD) != 0) failed = 1;
  for (int axis = 0; axis < 3; ++axis) {
    for (int direction = 0; direction < 3; ++direction)
      for (int delta = -1; delta <= 1; delta += 2) {
        const int next = tile.coord[axis] + delta;
        if (next < 0 || next >= tile.proc_dims[axis]) continue;
        int coord[3] = {tile.coord[0], tile.coord[1], tile.coord[2]};
        coord[axis] = next;
        int neighbor = -1;
        if (gamera_cs_decomposition_rank(tile.face, coord,
                                         tile.proc_dims, &neighbor) != 0)
          failed = 1;
        size_t index[3] = {ghost + 1U, ghost + 1U, ghost + 1U};
        index[axis] = delta < 0 ? ghost - 1U :
            ghost + tile.local_active[axis] +
                (direction == axis ? 1U : 0U);
        const size_t face = gamera_no_index3(
            storage.face_extent[direction], index[0], index[1], index[2]);
        if (storage.face_flux[direction][face] !=
                1000.0 * neighbor + direction ||
            storage.old_face_flux[direction][face] !=
                1000.0 * neighbor + direction + 0.5)
          failed = 1;
      }
    if (tile.coord[axis] == 0) continue;
    int coord[3] = {tile.coord[0], tile.coord[1], tile.coord[2]};
    --coord[axis];
    int low_rank = -1;
    if (gamera_cs_decomposition_rank(tile.face, coord,
                                     tile.proc_dims, &low_rank) != 0)
      failed = 1;
    size_t index[3] = {ghost + 1U, ghost + 1U, ghost + 1U};
    index[axis] = ghost;
    const size_t face = gamera_no_index3(storage.face_extent[axis],
                                        index[0], index[1], index[2]);
    if (storage.face_flux[axis][face] != 1000.0 * low_rank + axis ||
        storage.old_face_flux[axis][face] !=
            1000.0 * low_rank + axis + 0.5)
      failed = 1;
    for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
      if (storage.fluid_face_flux[axis][
              face * GAMERA_NO_FLUX_COUNT + (size_t)m] !=
          1000.0 * low_rank + axis + 0.01 * m) failed = 1;
    for (int d = 0; d < 3; ++d)
      if (storage.maxwell_face_flux[axis][face].value[d] !=
          1000.0 * low_rank + axis + 0.1 * d) failed = 1;
    for (int edge_axis = 0; edge_axis < 3; ++edge_axis) {
      if (edge_axis == axis) continue;
      const size_t edge = gamera_no_index3(
          storage.edge_extent[edge_axis], index[0], index[1], index[2]);
      if (storage.edge_emf[edge_axis][edge] !=
          1000.0 * low_rank + edge_axis) failed = 1;
    }
  }
  if (tile.coord[0] > 0 && tile.coord[1] > 0) {
    int coord[3] = {tile.coord[0] - 1, tile.coord[1] - 1,
                    tile.coord[2]};
    int diagonal = -1;
    if (gamera_cs_decomposition_rank(tile.face, coord,
                                     tile.proc_dims, &diagonal) != 0)
      failed = 1;
    for (int direction = 0; direction < 3; ++direction) {
      const size_t index[3] = {ghost - 1U, ghost - 1U, ghost + 1U};
      const size_t face = gamera_no_index3(
          storage.face_extent[direction], index[0], index[1], index[2]);
      if (storage.face_flux[direction][face] !=
              1000.0 * diagonal + direction ||
          storage.old_face_flux[direction][face] !=
              1000.0 * diagonal + direction + 0.5)
        failed = 1;
    }
  }
  /* Actual cube-face ghost fill on a decomposed, resolution-derived grid. */
  gamera_no_radial_map map;
  gamera_no_grid grid = {0};
  const size_t vertex_extent[3] = {extent[0] + 1U,
      extent[1] + 1U, extent[2] + 1U};
  gamera_no_vec3 *vertices = calloc(
      gamera_no_element_count3(vertex_extent), sizeof(*vertices));
  if (vertices == NULL || gamera_no_radial_map_init(&map,
          GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
          2.5, 200.0, 3.0) != 0 ||
      gamera_cs_generate_vertices_radial_map_tile(tile.face, global,
          tile.global_start, tile.local_active, ghost, &map,
          vertices) != 0 ||
      gamera_no_grid_create(extent, vertices, &grid) != 0) failed = 1;
  free(vertices);
  int setup_failed = 0;
  if (MPI_Allreduce(&failed, &setup_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  gamera_cs_tiled_cell_halo_plan *halo_plan = NULL;
  if (!setup_failed && gamera_cs_tiled_cell_halo_plan_create(
          &grid, &tile, ghost, MPI_COMM_WORLD, &halo_plan) != 0) {
    fprintf(stderr, "rank %d: tiled cell halo plan create failed\n", rank);
    failed = 1;
  }
  for (size_t i = ghost; i < ghost + tile.local_active[0]; ++i)
    for (size_t j = ghost; j < ghost + tile.local_active[1]; ++j)
      for (size_t k = ghost; k < ghost + tile.local_active[2]; ++k) {
        const int gi = (int)(tile.global_start[0] + i - ghost);
        const int gj = (int)(tile.global_start[1] + j - ghost);
        const int gk = (int)(tile.global_start[2] + k - ghost);
        const size_t cell = gamera_no_index3(extent, i, j, k);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          storage.conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              physical_sample(tile.face, gi, gj, gk, m);
          storage.old_conserved[cell * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              physical_sample(tile.face, gi, gj, gk, m) + 0.5;
        }
        for (int d = 0; d < 3; ++d) {
          storage.cell_magnetic[cell].value[d] =
              physical_sample(tile.face, gi, gj, gk,
                              GAMERA_NO_FLUX_COUNT + d);
          storage.old_cell_magnetic[cell].value[d] =
              storage.cell_magnetic[cell].value[d] + 0.5;
        }
      }
  int any_failed = 0;
  if (MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  if (!any_failed && gamera_cs_tiled_cell_halo_fill(
          halo_plan, &storage) != 0) {
    fprintf(stderr, "rank %d: tiled cell halo fill failed\n", rank);
    failed = 1;
  }
  for (size_t i = ghost; !failed && i < ghost + tile.local_active[0]; ++i)
    for (size_t j = 0; !failed && j < extent[1]; ++j)
      for (size_t k = 0; !failed && k < extent[2]; ++k) {
        const size_t gj_plus = tile.global_start[1] + j;
        const size_t gk_plus = tile.global_start[2] + k;
        if (gj_plus >= ghost && gj_plus - ghost < global[1] &&
            gk_plus >= ghost && gk_plus - ghost < global[2]) continue;
        const size_t cell = gamera_no_index3(extent, i, j, k);
        gamera_cs_tiled_donor_stencil stencil;
        if (gamera_cs_tiled_donor_stencil_create(
                grid.cell[cell].centroid, global, dims,
                tile.global_start[0] + i - ghost,
                0, &stencil) != 0) { failed = 1; break; }
        const int gi = (int)(tile.global_start[0] + i - ghost);
        for (int channel = 0; channel < GAMERA_CS_TILE_DONOR_CHANNELS;
             ++channel) {
          double expected = 0.0;
          const int old = (channel >= GAMERA_NO_FLUX_COUNT &&
              channel < 2 * GAMERA_NO_FLUX_COUNT) ||
              channel >= 2 * GAMERA_NO_FLUX_COUNT + 3;
          const int base = channel < GAMERA_NO_FLUX_COUNT ? channel :
              channel < 2 * GAMERA_NO_FLUX_COUNT ?
              channel - GAMERA_NO_FLUX_COUNT :
              GAMERA_NO_FLUX_COUNT +
              (channel - 2 * GAMERA_NO_FLUX_COUNT) % 3;
          for (size_t p = 0; p < 8; ++p)
            for (size_t q = 0; q < 8; ++q)
              expected += stencil.weight[0][p] * stencil.weight[1][q] *
                  (physical_sample(stencil.donor_face, gi,
                      (int)(stencil.start[0] + p),
                      (int)(stencil.start[1] + q), base) +
                   (old ? 0.5 : 0.0));
          const double got = channel < GAMERA_NO_FLUX_COUNT ?
              storage.conserved[cell * GAMERA_NO_FLUX_COUNT +
                                (size_t)channel] :
              channel < 2 * GAMERA_NO_FLUX_COUNT ?
              storage.old_conserved[cell * GAMERA_NO_FLUX_COUNT +
                  (size_t)(channel - GAMERA_NO_FLUX_COUNT)] :
              channel < 2 * GAMERA_NO_FLUX_COUNT + 3 ?
              storage.cell_magnetic[cell].value[
                  channel - 2 * GAMERA_NO_FLUX_COUNT] :
              storage.old_cell_magnetic[cell].value[
                  channel - 2 * GAMERA_NO_FLUX_COUNT - 3];
          if (fabs(got - expected) >
              3e-10 * fmax(1.0, fabs(expected))) {
            fprintf(stderr, "rank %d: cell halo mismatch channel=%d "
                    "got=%.17g expected=%.17g\n",
                    rank, channel, got, expected);
            failed = 1;
          }
        }
      }
  gamera_cs_tiled_magnetic_halo_plan *magnetic_plan = NULL;
  if (gamera_cs_tiled_magnetic_halo_plan_create(&grid, &tile, ghost,
          MPI_COMM_WORLD, &magnetic_plan) != 0) {
    fprintf(stderr, "rank %d: tiled magnetic halo plan create failed\n",
            rank);
    failed = 1;
  }
  if (!failed && gamera_cs_tiled_magnetic_halo_fill(
          magnetic_plan, &storage) != 0) {
    fprintf(stderr, "rank %d: tiled magnetic halo fill failed\n", rank);
    failed = 1;
  }
  for (int direction = 0; !failed && direction < 3; ++direction) {
    const size_t *fe = grid.face[direction].extent;
    for (size_t i = 0; !failed && i < fe[0]; ++i)
      for (size_t j = 0; !failed && j < fe[1]; ++j)
        for (size_t k = 0; !failed && k < fe[2]; ++k) {
          const size_t gj_plus = tile.global_start[1] + j;
          const size_t gk_plus = tile.global_start[2] + k;
          const size_t upper_j = global[1] + (direction == 1 ? 1U : 0U);
          const size_t upper_k = global[2] + (direction == 2 ? 1U : 0U);
          if (gj_plus >= ghost && gj_plus - ghost < upper_j &&
              gk_plus >= ghost && gk_plus - ghost < upper_k) continue;
          const size_t face = gamera_no_index3(fe, i, j, k);
          const size_t radial = i < ghost ?
              (tile.global_start[0] < ghost - i ? 0 :
               tile.global_start[0] - (ghost - i)) :
              fmin((double)(global[0] - 1U),
                   (double)(tile.global_start[0] + i - ghost));
          gamera_cs_tiled_donor_stencil stencil;
          if (gamera_cs_tiled_donor_stencil_create(
                  grid.face[direction].value[face].centroid,
                  global, dims, radial, 1, &stencil) != 0) {
            failed = 1;
            break;
          }
          double expected[2] = {0.0, 0.0};
          for (int axis = 0; axis < 3; ++axis) {
            double b[2] = {0.0, 0.0};
            for (size_t p = 0; p < 8; ++p)
              for (size_t q = 0; q < 8; ++q) {
                const double weight = stencil.weight[0][p] *
                                      stencil.weight[1][q];
                const double value = physical_sample(stencil.donor_face,
                    (int)radial, (int)(stencil.start[0] + p),
                    (int)(stencil.start[1] + q),
                    GAMERA_NO_FLUX_COUNT + axis);
                b[0] += weight * value;
                b[1] += weight * (value + 0.5);
              }
            const double area = grid.face[direction].value[face].
                area_vector.value[axis];
            expected[0] += b[0] * area;
            expected[1] += b[1] * area;
          }
          if (fabs(storage.face_flux[direction][face] - expected[0]) >
                  3e-10 * fmax(1.0, fabs(expected[0])) ||
              fabs(storage.old_face_flux[direction][face] - expected[1]) >
                  3e-10 * fmax(1.0, fabs(expected[1]))) {
            fprintf(stderr, "rank %d: magnetic halo mismatch dir=%d "
                    "i=%zu j=%zu k=%zu\n", rank, direction, i, j, k);
            failed = 1;
          }
        }
  }
  gamera_cs_tiled_magnetic_halo_plan_destroy(magnetic_plan);
  for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
    const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
    const int at_boundary = edge % 2 == 0 ?
        tile.coord[axis] == 0 :
        tile.coord[axis] == dims[axis] - 1;
    if (!at_boundary) continue;
    const int along_axis = axis == 1 ? 2 : 1;
    for (size_t i = 0; i < tile.local_active[0]; ++i)
      for (size_t along = 0; along <
           tile.local_active[along_axis]; ++along) {
        size_t coord[3] = {ghost + i, ghost + along, ghost + along};
        coord[axis] = edge % 2 == 0 ? ghost :
            ghost + tile.local_active[axis];
        const size_t face = gamera_no_index3(storage.face_extent[axis],
            coord[0], coord[1], coord[2]);
        const double value = 1000.0 * rank + 100.0 * i + along;
        storage.face_flux[axis][face] = value;
        storage.old_face_flux[axis][face] = value + 0.5;
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
          storage.fluid_face_flux[axis][
              face * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              value + 0.01 * m;
        for (int d = 0; d < 3; ++d)
          storage.maxwell_face_flux[axis][face].value[d] =
              value + 0.1 * d;
      }
  }
  if (gamera_cs_mpi_sync_cube_magnetic_faces(
          &storage, &tile, ghost, MPI_COMM_WORLD) != 0 ||
      gamera_cs_mpi_sync_cube_mhd_faces(
          &storage, &tile, ghost, MPI_COMM_WORLD) != 0) {
    fprintf(stderr, "rank %d: cube-face flux sync failed\n", rank);
    failed = 1;
  }
  for (int edge = 0; !failed && edge < GAMERA_CS_EDGE_COUNT; ++edge) {
    const int axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
    const int at_boundary = edge % 2 == 0 ?
        tile.coord[axis] == 0 :
        tile.coord[axis] == dims[axis] - 1;
    if (!at_boundary) continue;
    gamera_cs_neighbor neighbor;
    if (gamera_cs_edge_neighbor(tile.face, edge, &neighbor) != 0) {
      failed = 1;
      break;
    }
    const int target_axis = neighbor.edge < GAMERA_CS_B_LOWER ? 1 : 2;
    const int along_axis = axis == 1 ? 2 : 1;
    const int target_along_axis = target_axis == 1 ? 2 : 1;
    const double orientation = edge % 2 == neighbor.edge % 2 ?
        -1.0 : 1.0;
    for (size_t i = 0; i < tile.local_active[0]; ++i)
      for (size_t along = 0; along <
           tile.local_active[along_axis]; ++along) {
        const size_t global_along = tile.global_start[along_axis] + along;
        const size_t mapped_along = neighbor.reversed ?
            global[along_axis] - 1U - global_along : global_along;
        size_t global_cell[3] = {tile.global_start[0] + i, 0, 0};
        global_cell[target_axis] = neighbor.edge % 2 == 0 ?
            0 : global[target_axis] - 1U;
        global_cell[target_along_axis] = mapped_along;
        int partner = -1;
        size_t partner_local[3];
        if (gamera_cs_decomposition_cell_owner(global, dims,
                neighbor.face, global_cell, &partner,
                partner_local) != 0) { failed = 1; break; }
        size_t coord[3] = {ghost + i, ghost + along, ghost + along};
        coord[axis] = edge % 2 == 0 ? ghost :
            ghost + tile.local_active[axis];
        const size_t face = gamera_no_index3(storage.face_extent[axis],
            coord[0], coord[1], coord[2]);
        const double local_value = 1000.0 * rank + 100.0 * i + along;
        const double partner_value = 1000.0 * partner + 100.0 * i +
            partner_local[target_along_axis];
        const double common = 0.5 *
            (local_value + orientation * partner_value);
        if (storage.face_flux[axis][face] != common ||
            storage.old_face_flux[axis][face] !=
                common + 0.5 * (1.0 + orientation) * 0.5) {
          if (!failed) fprintf(stderr, "rank %d edge %d i %zu along %zu "
              "partner %d orientation %.0f magnetic got %.17g "
              "expected %.17g\n", rank, edge, i, along, partner,
              orientation, storage.face_flux[axis][face], common);
          failed = 1;
        }
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m)
          if (fabs(storage.fluid_face_flux[axis][
                  face * GAMERA_NO_FLUX_COUNT + (size_t)m] -
                  (common + 0.5 * (1.0 + orientation) * 0.01 * m)) >
              2e-14 * fmax(1.0, fabs(common))) {
            if (!failed) fprintf(stderr, "rank %d edge %d i %zu along %zu "
                "mhd fluid channel %d got %.17g expected %.17g\n",
                rank, edge, i, along, m,
                storage.fluid_face_flux[axis][
                    face * GAMERA_NO_FLUX_COUNT + (size_t)m],
                common + 0.5 * (1.0 + orientation) * 0.01 * m);
            failed = 1;
          }
        for (int d = 0; d < 3; ++d)
          if (fabs(storage.maxwell_face_flux[axis][face].value[d] -
                  (common + 0.5 * (1.0 + orientation) * 0.1 * d)) >
              2e-14 * fmax(1.0, fabs(common))) {
            if (!failed) fprintf(stderr, "rank %d edge %d i %zu along %zu "
                "mhd maxwell channel %d got %.17g expected %.17g\n",
                rank, edge, i, along, d,
                storage.maxwell_face_flux[axis][face].value[d],
                common + 0.5 * (1.0 + orientation) * 0.1 * d);
            failed = 1;
          }
      }
  }
  /* A uniform Cartesian E has one exact edge line integral on every face.
   * The distributed CT seam rule must leave that value unchanged, including
   * three-face corners and radial/angular tile boundaries. */
  for (int d = 0; d < 3; ++d)
    for (size_t i = 0; i < grid.edge[d].extent[0]; ++i)
      for (size_t j = 0; j < grid.edge[d].extent[1]; ++j)
        for (size_t k = 0; k < grid.edge[d].extent[2]; ++k) {
          size_t end[3] = {i, j, k};
          ++end[d];
          const gamera_no_vec3 a = grid.vertex[gamera_no_index3(
              grid.vertex_extent, i, j, k)];
          const gamera_no_vec3 b = grid.vertex[gamera_no_index3(
              grid.vertex_extent, end[0], end[1], end[2])];
          const size_t edge = gamera_no_index3(
              storage.edge_extent[d], i, j, k);
          storage.edge_emf[d][edge] =
              0.1 * (b.value[0] - a.value[0]) -
              0.2 * (b.value[1] - a.value[1]) +
              0.3 * (b.value[2] - a.value[2]);
        }
  int edge_setup_failed = 0;
  if (MPI_Allreduce(&failed, &edge_setup_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  gamera_cs_tiled_edge_sync_plan *edge_plan = NULL;
  if (!edge_setup_failed && gamera_cs_tiled_edge_sync_plan_create(
          &grid, &tile, ghost, MPI_COMM_WORLD, &edge_plan) != 0) {
    fprintf(stderr, "rank %d: tiled CT edge plan create failed\n", rank);
    failed = 1;
  }
  if (edge_plan != NULL &&
      gamera_cs_tiled_edge_sync(edge_plan, &storage) != 0) {
    fprintf(stderr, "rank %d: tiled CT edge sync failed\n", rank);
    failed = 1;
  }
  if (edge_plan != NULL)
    for (int d = 0; !failed && d < 3; ++d)
      for (size_t i = ghost; !failed &&
           i < ghost + tile.local_active[0] + (d != 0); ++i)
        for (size_t j = ghost; !failed &&
             j < ghost + tile.local_active[1] + (d != 1); ++j)
          for (size_t k = ghost; !failed &&
               k < ghost + tile.local_active[2] + (d != 2); ++k) {
            size_t end[3] = {i, j, k};
            ++end[d];
            const gamera_no_vec3 a = grid.vertex[gamera_no_index3(
                grid.vertex_extent, i, j, k)];
            const gamera_no_vec3 b = grid.vertex[gamera_no_index3(
                grid.vertex_extent, end[0], end[1], end[2])];
            const double expected =
                0.1 * (b.value[0] - a.value[0]) -
                0.2 * (b.value[1] - a.value[1]) +
                0.3 * (b.value[2] - a.value[2]);
            const size_t edge = gamera_no_index3(
                storage.edge_extent[d], i, j, k);
            if (fabs(storage.edge_emf[d][edge] - expected) >
                1e-9 * (1.0 + fabs(expected))) {
              fprintf(stderr, "rank %d CT edge mismatch d=%d "
                  "i=%zu j=%zu k=%zu got %.17g expected %.17g\n",
                  rank, d, i, j, k, storage.edge_emf[d][edge], expected);
              failed = 1;
            }
          }
  /* Independently check the arithmetic mean for one two-face seam edge
   * and one three-face cube corner after seeding rank-dependent EMFs. */
  int64_t sample_key[2][6] = {{0}};
  if (rank == 0) {
    int sign;
    if (test_edge_key(&grid, 0, ghost, ghost,
            ghost + global[2] / 4U, sample_key[0], &sign) != 0 ||
        test_edge_key(&grid, 0, ghost, ghost, ghost,
            sample_key[1], &sign) != 0) failed = 1;
  }
  if (MPI_Bcast(sample_key, 12, MPI_INT64_T, 0,
                MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  for (int d = 0; d < 3; ++d)
    for (size_t n = 0; n < gamera_no_element_count3(
         storage.edge_extent[d]); ++n)
      storage.edge_emf[d][n] = (double)rank + 0.01 * d;
  double local_sum[2] = {0.0, 0.0};
  int local_count[2] = {0, 0}, sample_sign[2] = {0, 0};
  size_t sample_index[2] = {0, 0};
  if (tile.coord[0] == 0)
    for (size_t j = ghost; j <= ghost + tile.local_active[1]; ++j)
      for (size_t k = ghost; k <= ghost + tile.local_active[2]; ++k) {
        if ((j == ghost && tile.coord[1] != 0) ||
            (k == ghost && tile.coord[2] != 0)) continue;
        int64_t key[6];
        int sign;
        if (test_edge_key(&grid, 0, ghost, j, k,
                          key, &sign) != 0) failed = 1;
        for (int sample = 0; sample < 2; ++sample)
          if (memcmp(key, sample_key[sample], sizeof(key)) == 0) {
            ++local_count[sample];
            local_sum[sample] += (double)sign * (double)rank;
            sample_sign[sample] = sign;
            sample_index[sample] = gamera_no_index3(
                storage.edge_extent[0], ghost, j, k);
          }
      }
  double total_sum[2] = {0.0, 0.0};
  int total_count[2] = {0, 0};
  if (MPI_Allreduce(local_sum, total_sum, 2, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(local_count, total_count, 2, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  if (total_count[0] != 2 || total_count[1] != 3 ||
      local_count[0] > 1 || local_count[1] > 1) failed = 1;
  int sampled_failed = 0;
  if (MPI_Allreduce(&failed, &sampled_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  if (!sampled_failed && gamera_cs_tiled_edge_sync(
          edge_plan, &storage) != 0) failed = 1;
  for (int sample = 0; !failed && sample < 2; ++sample)
    if (local_count[sample] == 1) {
      const double expected = sample_sign[sample] *
          total_sum[sample] / total_count[sample];
      const double got = storage.edge_emf[0][sample_index[sample]];
      if (fabs(got - expected) >
          1e-13 * (1.0 + fabs(expected))) {
        fprintf(stderr, "rank %d CT average sample %d got %.17g "
            "expected %.17g\n", rank, sample, got, expected);
        failed = 1;
      }
    }
  /* First actual tiled MHD update: the same uniform-state primitive and
   * Cartesian B as the accepted six-face GAMERA regression, with no Earth
   * boundary/MI callback. All numerical kernels remain unchanged. */
  const gamera_no_primitive primitive = {
      1.2, {{0.23, -0.17, 0.09}}, 0.74};
  const gamera_no_vec3 field = {{0.38, -0.21, 0.16}};
  double conserved[GAMERA_NO_FLUX_COUNT];
  if (gamera_no_primitive_to_conserved(&primitive, 5.0 / 3.0,
          1e-12, 1e-12, conserved) != 0) failed = 1;
  if (!failed) {
    for (size_t cell = 0; cell < gamera_no_element_count3(extent);
         ++cell) {
      memcpy(&storage.conserved[cell * GAMERA_NO_FLUX_COUNT],
             conserved, sizeof(conserved));
      memcpy(&storage.old_conserved[cell * GAMERA_NO_FLUX_COUNT],
             conserved, sizeof(conserved));
      storage.cell_magnetic[cell] = field;
      storage.old_cell_magnetic[cell] = field;
    }
    for (int d = 0; d < 3; ++d)
      for (size_t face = 0; face < gamera_no_element_count3(
           grid.face[d].extent); ++face) {
        double flux = 0.0;
        for (int c = 0; c < 3; ++c)
          flux += field.value[c] *
              grid.face[d].value[face].area_vector.value[c];
        storage.face_flux[d][face] = flux;
        storage.old_face_flux[d][face] = flux;
      }
  }
  gamera_cs_tiled_magnetic_halo_plan *mhd_magnetic_plan = NULL;
  gamera_cs_tiled_cube_face_sync_plan *mhd_face_plan = NULL;
  int mhd_setup_failed = 0;
  if (MPI_Allreduce(&failed, &mhd_setup_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  if (!mhd_setup_failed &&
      (gamera_cs_tiled_magnetic_halo_plan_create(&grid, &tile, ghost,
           MPI_COMM_WORLD, &mhd_magnetic_plan) != 0 ||
       gamera_cs_tiled_cube_face_sync_plan_create(&tile, ghost,
           MPI_COMM_WORLD, &mhd_face_plan) != 0)) {
    fprintf(stderr, "rank %d: tiled MHD plan setup failed\n", rank);
    failed = 1;
  }
  const gamera_no_advance_options options = {
      .stress = {5.0 / 3.0, 1e-12, 1e-12, 1.0, 0.0, 0.0, 100.0,
                 true, false, false, false},
      .emf = {1e-12, 1.0, 0.0, 100.0, 0.8, 0.01, false, false},
      .update = {5.0 / 3.0, 1e-12, 1e-12, 100.0, true, false, false},
      .source_time = 0.0};
  int global_mhd_setup_failed = 0;
  if (MPI_Allreduce(&failed, &global_mhd_setup_failed, 1, MPI_INT,
                    MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  if (!global_mhd_setup_failed)
    for (int step = 0; step < 3; ++step)
      if (gamera_cs_mpi_advance_tiled(&grid, &storage, &tile,
              ghost, 0.5, 1e-5, &options, NULL, halo_plan,
              mhd_magnetic_plan, mhd_face_plan, edge_plan,
              NULL, 0.0, 0.0, NULL, MPI_COMM_WORLD) != 0) {
        fprintf(stderr, "rank %d: tiled MHD step %d failed\n", rank,
                step);
        failed = 1;
        break;
      }
  double local_state_error = 0.0, local_divergence = 0.0;
  for (size_t i = ghost; !failed &&
       i < ghost + tile.local_active[0]; ++i)
    for (size_t j = ghost; j < ghost + tile.local_active[1]; ++j)
      for (size_t k = ghost; k < ghost + tile.local_active[2]; ++k) {
        const size_t cell = gamera_no_index3(extent, i, j, k);
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          const double error = fabs(storage.conserved[
              cell * GAMERA_NO_FLUX_COUNT + (size_t)m] - conserved[m]);
          if (error > local_state_error) local_state_error = error;
        }
        const double divergence = fabs(gamera_no_cell_net_flux(
            &storage, i, j, k)) / grid.cell[cell].volume;
        if (divergence > local_divergence)
          local_divergence = divergence;
      }
  double global_state_error = 0.0, global_divergence = 0.0;
  if (MPI_Allreduce(&local_state_error, &global_state_error, 1,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&local_divergence, &global_divergence, 1,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS)
    return 1;
  if (!isfinite(global_state_error) || !isfinite(global_divergence) ||
      global_state_error > 5e-6 || global_divergence > 1e-10) {
    if (rank == 0)
      fprintf(stderr, "tiled MHD uniform error=%.6e divB=%.6e\n",
              global_state_error, global_divergence);
    failed = 1;
  }
  gamera_cs_tiled_cube_face_sync_plan_destroy(mhd_face_plan);
  gamera_cs_tiled_magnetic_halo_plan_destroy(mhd_magnetic_plan);
  gamera_cs_tiled_edge_sync_plan_destroy(edge_plan);
  gamera_cs_tiled_cell_halo_plan_destroy(halo_plan);
  gamera_no_grid_destroy(&grid);
  gamera_no_storage_destroy(&storage);
  int global_failed = 0;
  if (MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  if (rank == 0 && global_failed == 0)
    printf("cubed-sphere internal tile halos PASS grid=%zu,%zu,%zu "
           "dims=%d,%d,%d ranks=%d\n", global[0], global[1], global[2],
           dims[0], dims[1], dims[2], ranks);
  MPI_Finalize();
  return global_failed;
}
