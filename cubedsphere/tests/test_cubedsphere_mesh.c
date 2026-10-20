#include "cubedsphere_mesh.h"
#include "cubedsphere_halo.h"
#include "cubedsphere_interface.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;

static void check(int condition, const char *description) {
  if (!condition) {
    fprintf(stderr, "FAIL %s\n", description);
    ++failures;
  }
}

static double distance(gamera_no_vec3 a, gamera_no_vec3 b) {
  double square = 0.0;
  for (int d = 0; d < 3; ++d) {
    square += (a.value[d] - b.value[d]) * (a.value[d] - b.value[d]);
  }
  return sqrt(square);
}

static double dot(gamera_no_vec3 a, gamera_no_vec3 b) {
  double result = 0.0;
  for (int d = 0; d < 3; ++d) {
    result += a.value[d] * b.value[d];
  }
  return result;
}

static gamera_no_vec3 seam_vertex(const gamera_no_grid *grid, int edge,
                                   size_t radial, size_t along,
                                   size_t n, size_t ghost) {
  const size_t j = edge == GAMERA_CS_A_LOWER ? ghost :
                   edge == GAMERA_CS_A_UPPER ? ghost + n : ghost + along;
  const size_t k = edge == GAMERA_CS_B_LOWER ? ghost :
                   edge == GAMERA_CS_B_UPPER ? ghost + n : ghost + along;
  return grid->vertex[gamera_no_index3(grid->vertex_extent, radial, j, k)];
}

static void check_topology_keys(size_t radial_cells, size_t angular_cells) {
  const size_t global_active[3] = {
      radial_cells, angular_cells, angular_cells};
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
      gamera_cs_neighbor neighbor;
      check(gamera_cs_edge_neighbor(face, edge, &neighbor) == 0,
            "topology-key neighbor exists");
      if (gamera_cs_edge_neighbor(face, edge, &neighbor) != 0) continue;
      for (size_t radial = 0; radial <= radial_cells;
           radial += radial == 0 ? radial_cells : radial_cells + 1)
        for (size_t along = 0; along <= angular_cells; ++along) {
          const size_t donor_along = neighbor.reversed ?
              angular_cells - along : along;
          const size_t source_vertex[3] = {
              radial,
              edge == GAMERA_CS_A_LOWER ? 0 :
              edge == GAMERA_CS_A_UPPER ? angular_cells : along,
              edge == GAMERA_CS_B_LOWER ? 0 :
              edge == GAMERA_CS_B_UPPER ? angular_cells : along};
          const size_t donor_vertex[3] = {
              radial,
              neighbor.edge == GAMERA_CS_A_LOWER ? 0 :
              neighbor.edge == GAMERA_CS_A_UPPER ? angular_cells : donor_along,
              neighbor.edge == GAMERA_CS_B_LOWER ? 0 :
              neighbor.edge == GAMERA_CS_B_UPPER ? angular_cells : donor_along};
          int64_t source_key[4] = {0}, donor_key[4] = {0};
          const int keys_valid =
              gamera_cs_vertex_topology_key(face, global_active,
                    source_vertex, source_key) == 0 &&
              gamera_cs_vertex_topology_key(neighbor.face, global_active,
                    donor_vertex, donor_key) == 0;
          check(keys_valid, "topology keys constructed");
          if (!keys_valid) continue;
          int equal = 1;
          for (int d = 0; d < 4; ++d)
            if (source_key[d] != donor_key[d]) equal = 0;
          check(equal, "shared seam vertex has exact topology key");
        }
    }
}

int main(int argc, char **argv) {
  check_topology_keys(128, 32);
  check_topology_keys(144, 32);
  check_topology_keys(256, 256);
  const size_t ghost = 4;
  const size_t tangential = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 10U;
  if (tangential < 9U || tangential > 48U) return EXIT_FAILURE;
  const size_t active[3] = {8, tangential, tangential};
  const size_t extent[3] = {active[0] + 2 * ghost,
                            active[1] + 2 * ghost,
                            active[2] + 2 * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1, extent[1] + 1,
                                   extent[2] + 1};
  const size_t vertex_count = gamera_no_element_count3(vertex_extent);
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT] = {{0}};
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 normal, a, b;
    check(gamera_cs_face_frame(face, &normal, &a, &b) == 0,
          "right-handed face frame exists");
    gamera_no_vec3 cross = {{a.value[1] * b.value[2] - a.value[2] * b.value[1],
                             a.value[2] * b.value[0] - a.value[0] * b.value[2],
                             a.value[0] * b.value[1] - a.value[1] * b.value[0]}};
    check(distance(cross, normal) < 1.0e-15, "face orientation");
    gamera_no_vec3 *vertices = calloc(vertex_count, sizeof(*vertices));
    check(vertices != NULL, "vertex allocation");
    if (vertices == NULL) {
      break;
    }
    check(gamera_cs_generate_vertices(face, active, ghost, 2.0, 3.0,
                                      vertices) == 0,
          "cubed-sphere vertex generation");
    check(gamera_no_grid_create(extent, vertices, &grid[face]) == 0,
          "GAMERA nonorthogonal geometry on cubed sphere");
    free(vertices);
    if (grid[face].cell == NULL) {
      continue;
    }
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t cell = gamera_no_index3(extent, i, j, k);
          check(isfinite(grid[face].cell[cell].volume) &&
                    grid[face].cell[cell].volume > 0.0,
                "positive finite active cell volume");
        }
      }
    }
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
      gamera_cs_neighbor neighbor;
      check(gamera_cs_edge_neighbor(face, edge, &neighbor) == 0,
            "edge neighbor exists");
      if (gamera_cs_edge_neighbor(face, edge, &neighbor) != 0) {
        continue;
      }
      gamera_cs_neighbor back;
      check(gamera_cs_edge_neighbor(neighbor.face, neighbor.edge, &back) == 0 &&
                back.face == face && back.edge == edge &&
                back.reversed == neighbor.reversed,
            "reciprocal edge topology");
      for (size_t radial = ghost; radial <= ghost + active[0]; ++radial) {
        for (size_t along = 0; along <= active[1]; ++along) {
          const size_t donor_along =
              neighbor.reversed ? active[1] - along : along;
          const gamera_no_vec3 source = seam_vertex(
              &grid[face], edge, radial, along, active[1], ghost);
          const gamera_no_vec3 donor = seam_vertex(
              &grid[neighbor.face], neighbor.edge, radial, donor_along,
              active[1], ghost);
          check(distance(source, donor) < 3.0e-15,
                "matching seam vertices without overset");
        }
      }
    }
  }
  /* A uniform Cartesian B must give exactly opposite physical fluxes on
   * adjacent oriented faces, including axis-swapped and reversed seams. */
  const gamera_no_vec3 magnetic = {{0.3, -0.2, 0.4}};
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
      gamera_cs_neighbor neighbor;
      if (gamera_cs_edge_neighbor(face, edge, &neighbor) != 0) continue;
      const int src_axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
      const int dst_axis = neighbor.edge < GAMERA_CS_B_LOWER ? 1 : 2;
      const size_t src_boundary = edge % 2 == 0 ? ghost : ghost + active[src_axis];
      const size_t dst_boundary = neighbor.edge % 2 == 0 ? ghost : ghost + active[dst_axis];
      const int src_sign = edge % 2 == 0 ? -1 : 1;
      const int dst_sign = neighbor.edge % 2 == 0 ? -1 : 1;
      for (size_t radial = ghost; radial < ghost + active[0]; ++radial) {
        for (size_t along = 0; along < active[1]; ++along) {
          const size_t donor_along = neighbor.reversed ? active[1] - 1 - along : along;
          size_t src[3] = {radial, ghost + along, ghost + along};
          size_t dst[3] = {radial, ghost + donor_along, ghost + donor_along};
          src[src_axis] = src_boundary;
          dst[dst_axis] = dst_boundary;
          const gamera_no_face_geometry *sf =
              &grid[face].face[src_axis].value[gamera_no_index3(
                  grid[face].face[src_axis].extent, src[0], src[1], src[2])];
          const gamera_no_face_geometry *df =
              &grid[neighbor.face].face[dst_axis].value[gamera_no_index3(
                  grid[neighbor.face].face[dst_axis].extent,
                  dst[0], dst[1], dst[2])];
          check(distance(sf->centroid, df->centroid) < 1.0e-12,
                "shared face centroid");
          const double outward_src = (double)src_sign * dot(magnetic, sf->area_vector);
          const double outward_dst = (double)dst_sign * dot(magnetic, df->area_vector);
          check(fabs(outward_src + outward_dst) < 1.0e-12,
                "shared magnetic flux has opposite outward signs");
        }
      }
    }
  }
  gamera_no_storage storage[GAMERA_CS_FACE_COUNT] = {{0}};
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    check(gamera_no_storage_create(extent, &storage[face]) == 0,
          "halo storage allocation");
    if (storage[face].conserved == NULL) continue;
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = ghost; j < ghost + active[1]; ++j) {
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t c = gamera_no_index3(extent, i, j, k);
          const gamera_no_vec3 p = grid[face].cell[c].centroid;
          const double rho = 1.0 + 0.01 *
              (p.value[0] + 2.0 * p.value[1] - p.value[2]);
          storage[face].conserved[c * GAMERA_NO_FLUX_COUNT] = rho;
          storage[face].old_conserved[c * GAMERA_NO_FLUX_COUNT] = rho + 0.1;
          storage[face].conserved[c * GAMERA_NO_FLUX_COUNT +
                                  GAMERA_NO_FLUX_ENERGY] = 3.0;
          storage[face].old_conserved[c * GAMERA_NO_FLUX_COUNT +
                                      GAMERA_NO_FLUX_ENERGY] = 3.1;
          storage[face].cell_magnetic[c] = magnetic;
          storage[face].old_cell_magnetic[c] = magnetic;
        }
      }
    }
  }
  check(gamera_cs_fill_tangential_cell_halos(grid, storage, active, ghost) == 0,
        "eight-by-eight face-owner ghost interpolation");
  double worst_halo = 0.0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (size_t i = ghost; i < ghost + active[0]; ++i) {
      for (size_t j = 0; j < extent[1]; ++j) {
        for (size_t k = 0; k < extent[2]; ++k) {
          if (j >= ghost && j < ghost + active[1] &&
              k >= ghost && k < ghost + active[2]) continue;
          const size_t c = gamera_no_index3(extent, i, j, k);
          const gamera_no_vec3 p = grid[face].cell[c].centroid;
          const double expected = 1.0 + 0.01 *
              (p.value[0] + 2.0 * p.value[1] - p.value[2]);
          const double actual = storage[face].conserved[
              c * GAMERA_NO_FLUX_COUNT];
          const double error = fabs(actual - expected);
          if (error > worst_halo) worst_halo = error;
          check(isfinite(actual) && actual > 0.0,
                "finite positive corner and seam halos");
          check(fabs(storage[face].old_conserved[
              c * GAMERA_NO_FLUX_COUNT] - actual - 0.1) < 1e-12,
                "both time levels use same donor interpolation");
          check(distance(storage[face].cell_magnetic[c], magnetic) < 1e-12,
                "Cartesian magnetic halo unchanged for uniform B");
        }
      }
    }
  }
  printf("max smooth cross-face ghost interpolation error %.6e\n",
         worst_halo);
  check(worst_halo < 5e-3, "smooth ghost interpolation accuracy");
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int d = 0; d < 3; ++d) {
      const size_t face_count = gamera_no_element_count3(
          grid[face].face[d].extent);
      for (size_t f = 0; f < face_count; ++f) {
        storage[face].face_flux[d][f] =
            dot(magnetic, grid[face].face[d].value[f].area_vector) +
            1e-4 * (double)face;
        for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
          storage[face].fluid_face_flux[d][
              f * GAMERA_NO_FLUX_COUNT + (size_t)m] =
              (double)(face + 1) * (double)(m + 1);
        }
        for (int m = 0; m < 3; ++m) {
          storage[face].maxwell_face_flux[d][f].value[m] =
              (double)(face + 1) * (double)(m + 1);
        }
      }
      const size_t edge_count = gamera_no_element_count3(
          grid[face].edge[d].extent);
      for (size_t e = 0; e < edge_count; ++e) {
        storage[face].edge_emf[d][e] = (double)(face + 1);
      }
    }
  }
  check(gamera_cs_sync_magnetic_face_flux(storage, active, ghost) == 0,
        "one physical magnetic face flux across each seam");
  check(gamera_cs_sync_mhd_face_fluxes(storage, active, ghost) == 0,
        "conservative shared MHD interface flux");
  check(gamera_cs_sync_edge_emf(grid, storage, active, ghost) == 0,
        "signed shared edge EMF including three-face corners");
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    for (int edge = 0; edge < GAMERA_CS_EDGE_COUNT; ++edge) {
      gamera_cs_neighbor neighbor;
      if (gamera_cs_edge_neighbor(face, edge, &neighbor) != 0 ||
          face >= neighbor.face) continue;
      const int src_axis = edge < GAMERA_CS_B_LOWER ? 1 : 2;
      const int dst_axis = neighbor.edge < GAMERA_CS_B_LOWER ? 1 : 2;
      const size_t src_boundary = edge % 2 == 0 ? ghost : ghost + active[src_axis];
      const size_t dst_boundary = neighbor.edge % 2 == 0 ? ghost : ghost + active[dst_axis];
      const double face_sign = edge % 2 == neighbor.edge % 2 ? -1.0 : 1.0;
      const int src_tangent_axis = src_axis == 1 ? 2 : 1;
      const int dst_tangent_axis = dst_axis == 1 ? 2 : 1;
      for (size_t i = ghost; i < ghost + active[0]; ++i) {
        for (size_t along = 0; along < active[1]; ++along) {
          const size_t dst_along = neighbor.reversed ? active[1] - 1U - along : along;
          size_t s[3] = {i, ghost + along, ghost + along};
          size_t t[3] = {i, ghost + dst_along, ghost + dst_along};
          s[src_axis] = src_boundary;
          t[dst_axis] = dst_boundary;
          const size_t sf = gamera_no_index3(storage[face].face_extent[src_axis],
                                             s[0], s[1], s[2]);
          const size_t tf = gamera_no_index3(
              storage[neighbor.face].face_extent[dst_axis],
              t[0], t[1], t[2]);
          check(fabs(storage[face].face_flux[src_axis][sf] - face_sign *
                     storage[neighbor.face].face_flux[dst_axis][tf]) < 1e-14,
                "seam magnetic flux single-valued");
          for (int m = 0; m < GAMERA_NO_FLUX_COUNT; ++m) {
            check(fabs(storage[face].fluid_face_flux[src_axis][
                sf * GAMERA_NO_FLUX_COUNT + (size_t)m] - face_sign *
                storage[neighbor.face].fluid_face_flux[dst_axis][
                tf * GAMERA_NO_FLUX_COUNT + (size_t)m]) < 1e-14,
                "seam Reynolds flux single-valued");
          }
          /* Tangential seam edge at the lower radial vertex. */
          const size_t se = gamera_no_index3(
              storage[face].edge_extent[src_tangent_axis], s[0], s[1], s[2]);
          const size_t te = gamera_no_index3(
              storage[neighbor.face].edge_extent[dst_tangent_axis],
              t[0], t[1], t[2]);
          const double tangent_sign = neighbor.reversed ? -1.0 : 1.0;
          check(fabs(storage[face].edge_emf[src_tangent_axis][se] -
                     tangent_sign * storage[neighbor.face]
                         .edge_emf[dst_tangent_axis][te]) < 1e-14,
                "seam tangential EMF single-valued");
        }
      }
      for (size_t i = ghost; i < ghost + active[0]; ++i) {
        for (size_t along = 0; along <= active[1]; ++along) {
          const size_t dst_along = neighbor.reversed ? active[1] - along : along;
          size_t s[3] = {i, ghost + along, ghost + along};
          size_t t[3] = {i, ghost + dst_along, ghost + dst_along};
          s[src_axis] = src_boundary;
          t[dst_axis] = dst_boundary;
          const size_t se = gamera_no_index3(storage[face].edge_extent[0],
                                              s[0], s[1], s[2]);
          const size_t te = gamera_no_index3(
              storage[neighbor.face].edge_extent[0], t[0], t[1], t[2]);
          check(fabs(storage[face].edge_emf[0][se] -
                     storage[neighbor.face].edge_emf[0][te]) < 1e-14,
                "seam radial EMF single-valued at two/three-face joins");
        }
      }
    }
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_storage_destroy(&storage[face]);
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_grid_destroy(&grid[face]);
  }
  if (failures != 0) {
    fprintf(stderr, "%d cubed-sphere mesh checks failed\n", failures);
    return EXIT_FAILURE;
  }
  puts("PASS six-face geometry, reciprocal seams, shared magnetic flux");
  return EXIT_SUCCESS;
}
