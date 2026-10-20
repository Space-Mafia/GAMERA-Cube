#include "cubedsphere_earth_boundary.h"
#include "cubedsphere_mi_dipole.h"

#include "nonorthogonal_step.h"
#include "nonorthogonal_state.h"

#include <math.h>

static int valid(const gamera_cs_earth_boundary *b) {
  return b != NULL && b->startup != NULL && b->wind != NULL &&
         b->background != NULL && isfinite(b->density_floor) &&
         b->density_floor > 0.0 && isfinite(b->pressure_floor) &&
         b->pressure_floor > 0.0;
}

static int sample_wind(const gamera_cs_earth_boundary *b,
                       gamera_no_vec3 point, double time,
                       gamera_solar_wind_state *wind, double *weight) {
  if (!valid(b) || wind == NULL || weight == NULL ||
      gamera_solar_wind_sample_at(b->wind, point.value, time,
                                  wind, NULL) != 0 ||
      gamera_solar_wind_weight(b->wind, point.value, time,
                               weight) != 0) return -1;
  return 0;
}

static int fill_cell_state(const gamera_no_grid *grid,
                           gamera_no_storage *storage,
                           const size_t active[3], size_t ghost,
                           const gamera_cs_earth_boundary *b,
                           double time, int old,
                           int use_inner, int use_outer) {
  const size_t outer = ghost + active[0];
  const size_t *extent = grid->cell_extent;
  double *conserved = old ? storage->old_conserved : storage->conserved;
  for (size_t i = 0; i < extent[0]; ++i) {
    if (i >= ghost && i < outer) continue;
    const int inner = i < ghost;
    if ((inner && !use_inner) || (!inner && !use_outer)) continue;
    const size_t source_i = inner ? 2U * ghost - 1U - i : outer - 1U;
    for (size_t j = 0; j < extent[1]; ++j) {
      for (size_t k = 0; k < extent[2]; ++k) {
        const size_t target = gamera_no_index3(extent, i, j, k);
        const size_t source = gamera_no_index3(extent, source_i, j, k);
        gamera_no_primitive primitive;
        if (gamera_no_conserved_to_primitive(
                &conserved[source * GAMERA_NO_FLUX_COUNT],
                b->startup->gamma, b->density_floor, b->pressure_floor,
                &primitive) != 0) return -1;
        const gamera_no_vec3 point = grid->cell[target].centroid;
        const double radius = sqrt(point.value[0] * point.value[0] +
                                   point.value[1] * point.value[1] +
                                   point.value[2] * point.value[2]);
        if (!(radius > 0.0)) return -1;
        if (inner) {
          /* Problem 21 reflect_inner_cell: rho/p zero radial gradient;
           * reverse only the normal velocity for a stationary hard wall. */
          if (gamera_cs_mi_dipole_ready(b->mi)) {
            gamera_no_vec3 ghost_velocity;
            if (gamera_cs_mi_dipole_ghost_velocity(
                    b->mi, point, primitive.velocity,
                    &ghost_velocity) != 0) return -1;
            if (b->strict_inner_gas_wall)
              for (int c = 0; c < GAMERA_NO_DIM; ++c)
                ghost_velocity.value[c] =
                    0.5 * (ghost_velocity.value[c] +
                           primitive.velocity.value[c]);
            primitive.velocity = ghost_velocity;
          } else if (b->strict_inner_gas_wall) {
            primitive.velocity = (gamera_no_vec3){{0.0, 0.0, 0.0}};
          } else {
            double vn = 0.0;
            for (int c = 0; c < GAMERA_NO_DIM; ++c)
              vn += primitive.velocity.value[c] * point.value[c] / radius;
            for (int c = 0; c < GAMERA_NO_DIM; ++c)
              primitive.velocity.value[c] -= 2.0 * vn *
                                             point.value[c] / radius;
          }
        } else {
          /* Problem 21 set_time_dependent_outer_cell: blend the advected
           * wind with subsonic-corrected outflow on the tail side. */
          gamera_solar_wind_state wind;
          double weight;
          if (sample_wind(b, point, time, &wind, &weight) != 0) return -1;
          double vn = 0.0;
          for (int c = 0; c < GAMERA_NO_DIM; ++c)
            vn += primitive.velocity.value[c] * point.value[c] / radius;
          const double cs = sqrt(b->startup->gamma * primitive.pressure /
                                 primitive.density);
          if (vn < cs)
            for (int c = 0; c < GAMERA_NO_DIM; ++c)
              primitive.velocity.value[c] += (cs - vn) *
                                             point.value[c] / radius;
          primitive.density = weight * wind.density +
                              (1.0 - weight) * primitive.density;
          primitive.pressure = weight * wind.pressure +
                               (1.0 - weight) * primitive.pressure;
          for (int c = 0; c < GAMERA_NO_DIM; ++c)
            primitive.velocity.value[c] =
                weight * wind.velocity[c] +
                (1.0 - weight) * primitive.velocity.value[c];
        }
        if (gamera_no_primitive_to_conserved(
                &primitive, b->startup->gamma, b->density_floor,
                b->pressure_floor,
                &conserved[target * GAMERA_NO_FLUX_COUNT]) != 0)
          return -1;
      }
    }
  }
  return 0;
}

static int fill_face_state(const gamera_no_grid *grid,
                           gamera_no_storage *storage,
                           const size_t active[3], size_t ghost,
                           const gamera_cs_earth_boundary *b,
                           const gamera_no_background_field *background,
                           double time, int old,
                           int use_inner, int use_outer) {
  const size_t outer = ghost + active[0];
  for (int direction = 0; direction < GAMERA_NO_DIM; ++direction) {
    double *flux = old ? storage->old_face_flux[direction]
                       : storage->face_flux[direction];
    for (size_t i = 0; i < grid->face[direction].extent[0]; ++i) {
      const int inner = i < ghost;
      const int beyond_outer = i >= outer + (direction == GAMERA_NO_I);
      if (!inner && !beyond_outer) continue;
      if ((inner && !use_inner) || (beyond_outer && !use_outer)) continue;
      for (size_t j = 0; j < grid->face[direction].extent[1]; ++j) {
        for (size_t k = 0; k < grid->face[direction].extent[2]; ++k) {
          const size_t index = gamera_no_index3(
              grid->face[direction].extent, i, j, k);
          const gamera_no_face_geometry *geometry =
              &grid->face[direction].value[index];
          if (inner) {
            const size_t source = gamera_no_index3(
                grid->face[direction].extent, ghost, j, k);
            const double source_area =
                grid->face[direction].value[source].area;
            if (!(source_area > 0.0)) return -1;
            flux[index] = flux[source] * geometry->area / source_area;
          } else {
            gamera_solar_wind_state wind;
            double weight;
            if (sample_wind(b, geometry->centroid, time,
                            &wind, &weight) != 0) return -1;
            double wind_flux = -background->face_flux[direction][index];
            for (int c = 0; c < GAMERA_NO_DIM; ++c)
              wind_flux += wind.magnetic[c] *
                           geometry->area_vector.value[c];
            const size_t source_i =
                direction == GAMERA_NO_I ? outer : outer - 1U;
            const size_t source = gamera_no_index3(
                grid->face[direction].extent, source_i, j, k);
            flux[index] = weight * wind_flux +
                          (1.0 - weight) * flux[source];
          }
        }
      }
    }
  }
  const double *flux[3] = {
      old ? storage->old_face_flux[0] : storage->face_flux[0],
      old ? storage->old_face_flux[1] : storage->face_flux[1],
      old ? storage->old_face_flux[2] : storage->face_flux[2]};
  gamera_no_vec3 *cell_field = old ? storage->old_cell_magnetic
                                   : storage->cell_magnetic;
  if (gamera_no_recover_magnetic_field(grid, flux, cell_field) != 0)
    return -1;
  /* Match the strict Yin-Yang magnetic wall's dB1/dr = 0 ghost storage. */
  if (use_inner)
    for (size_t i = 0; i < ghost; ++i)
      for (size_t j = 0; j < grid->cell_extent[1]; ++j)
        for (size_t k = 0; k < grid->cell_extent[2]; ++k) {
          const size_t target = gamera_no_index3(
              grid->cell_extent, i, j, k);
          const size_t source = gamera_no_index3(
              grid->cell_extent, ghost, j, k);
          cell_field[target] = cell_field[source];
        }
  return 0;
}

static int fill_radial_ghosts_filtered(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double current_time, double previous_time, void *context,
    int receiver) {
  const gamera_cs_earth_boundary *b = context;
  if (grid == NULL || storage == NULL || active == NULL || ghost < 4 ||
      !valid(b) || !isfinite(current_time) || !isfinite(previous_time))
    return -1;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    if (receiver >= 0 && face != receiver) continue;
    if (fill_cell_state(&grid[face], &storage[face], active, ghost,
                        b, current_time, 0, 1, 1) != 0 ||
        fill_cell_state(&grid[face], &storage[face], active, ghost,
                        b, previous_time, 1, 1, 1) != 0 ||
        fill_face_state(&grid[face], &storage[face], active, ghost,
                        b, &b->background[face], current_time, 0,
                        1, 1) != 0 ||
        fill_face_state(&grid[face], &storage[face], active, ghost,
                        b, &b->background[face], previous_time, 1,
                        1, 1) != 0)
      return -1;
  }
  return 0;
}

int gamera_cs_earth_fill_radial_ghosts(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double current_time, double previous_time, void *context) {
  return fill_radial_ghosts_filtered(grid, storage, active, ghost,
                                     current_time, previous_time,
                                     context, -1);
}

int gamera_cs_earth_fill_radial_ghosts_face(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double current_time, double previous_time, void *context, int face) {
  if (face < 0 || face >= GAMERA_CS_FACE_COUNT) return -1;
  return fill_radial_ghosts_filtered(grid, storage, active, ghost,
                                     current_time, previous_time,
                                     context, face);
}

int gamera_cs_earth_fill_radial_ghosts_tile(
    const gamera_no_grid *grid, gamera_no_storage *local,
    const gamera_cs_decomposition *tile, size_t ghost,
    const gamera_no_background_field *background,
    double current_time, double previous_time, void *context) {
  const gamera_cs_earth_boundary *b = context;
  if (grid == NULL || local == NULL || tile == NULL || ghost < 4 ||
      !valid(b) || !isfinite(current_time) ||
      !isfinite(previous_time)) return -1;
  const int inner = tile->coord[0] == 0;
  const int outer = tile->coord[0] == tile->proc_dims[0] - 1;
  /* Interior radial tiles still need both cell-B caches recovered from
   * synchronized face fluxes. The filtered fills below leave physical
   * ghost states and all face fluxes untouched on such tiles. */
  if (outer && background == NULL) return -1;
  const size_t *active = tile->local_active;
  return fill_cell_state(grid, local, active, ghost, b,
             current_time, 0, inner, outer) == 0 &&
         fill_cell_state(grid, local, active, ghost, b,
             previous_time, 1, inner, outer) == 0 &&
         fill_face_state(grid, local, active, ghost, b,
             background, current_time, 0, inner, outer) == 0 &&
         fill_face_state(grid, local, active, ghost, b,
             background, previous_time, 1, inner, outer) == 0 ? 0 : -1;
}

int gamera_cs_earth_inner_fluid_flux(
    gamera_no_storage *storage, const gamera_no_grid *grid,
    const size_t lower[3], const size_t upper[3], void *context) {
  (void)grid;
  (void)context;
  return gamera_no_trap_inner_outward_mass_energy_flux(
      storage, lower, upper, NULL);
}

static gamera_no_vec3 edge_midpoint(const gamera_no_grid *grid,
                                     int direction, size_t i,
                                     size_t j, size_t k) {
  size_t upper[3] = {i, j, k};
  ++upper[direction];
  const gamera_no_vec3 a = grid->vertex[gamera_no_index3(
      grid->vertex_extent, i, j, k)];
  const gamera_no_vec3 z = grid->vertex[gamera_no_index3(
      grid->vertex_extent, upper[0], upper[1], upper[2])];
  gamera_no_vec3 result;
  for (int c = 0; c < GAMERA_NO_DIM; ++c)
    result.value[c] = 0.5 * (a.value[c] + z.value[c]);
  return result;
}

static int face_flux_density(const gamera_no_grid *grid, const double *flux,
                             int direction, size_t i, size_t j, size_t k,
                             double *result) {
  if (grid == NULL || flux == NULL || result == NULL ||
      i >= grid->face[direction].extent[0] ||
      j >= grid->face[direction].extent[1] ||
      k >= grid->face[direction].extent[2]) return -1;
  const size_t index = gamera_no_index3(
      grid->face[direction].extent, i, j, k);
  const double area = grid->face[direction].value[index].area;
  if (!(area > 0.0)) return -1;
  *result = flux[index] / area;
  return isfinite(*result) ? 0 : -1;
}

/* Problem 21 add_wind_diffusive_emf, using this step's explicit dt/CFL. */
static int add_wind_diffusive_emf(
    gamera_no_storage *storage, const gamera_no_grid *grid,
    const size_t lower[3], const size_t upper[3],
    const gamera_cs_earth_boundary *b) {
  const size_t i = upper[GAMERA_NO_I];
  if (i == 0U || !(b->step_dt > 0.0) || !(b->cfl > 0.0)) return -1;
  for (size_t j = lower[GAMERA_NO_J]; j <= upper[GAMERA_NO_J]; ++j) {
    for (size_t k = lower[GAMERA_NO_K]; k <= upper[GAMERA_NO_K]; ++k) {
      if (j == 0U || k == 0U ||
          i >= grid->edge[GAMERA_NO_J].extent[0] ||
          j >= grid->edge[GAMERA_NO_J].extent[1] ||
          k >= grid->edge[GAMERA_NO_J].extent[2] ||
          i >= grid->edge[GAMERA_NO_K].extent[0] ||
          j >= grid->edge[GAMERA_NO_K].extent[1] ||
          k >= grid->edge[GAMERA_NO_K].extent[2]) continue;
      const size_t edge_j = gamera_no_index3(
          grid->edge[GAMERA_NO_J].extent, i, j, k);
      const size_t edge_k = gamera_no_index3(
          grid->edge[GAMERA_NO_K].extent, i, j, k);
      if (grid->edge[GAMERA_NO_J].valid[edge_j] == 0U ||
          grid->edge[GAMERA_NO_K].valid[edge_k] == 0U) continue;
      double bi_k_plus, bi_k_minus, bk_i_plus, bk_i_minus;
      double bj_i_plus, bj_i_minus, bi_j_plus, bi_j_minus;
      if (face_flux_density(grid,
              storage->predicted_face_flux[GAMERA_NO_I], GAMERA_NO_I,
              i, j, k, &bi_k_plus) != 0 ||
          face_flux_density(grid,
              storage->predicted_face_flux[GAMERA_NO_I], GAMERA_NO_I,
              i, j, k - 1U, &bi_k_minus) != 0 ||
          face_flux_density(grid,
              storage->predicted_face_flux[GAMERA_NO_K], GAMERA_NO_K,
              i, j, k, &bk_i_plus) != 0 ||
          face_flux_density(grid,
              storage->predicted_face_flux[GAMERA_NO_K], GAMERA_NO_K,
              i - 1U, j, k, &bk_i_minus) != 0 ||
          face_flux_density(grid,
              storage->predicted_face_flux[GAMERA_NO_J], GAMERA_NO_J,
              i, j, k, &bj_i_plus) != 0 ||
          face_flux_density(grid,
              storage->predicted_face_flux[GAMERA_NO_J], GAMERA_NO_J,
              i - 1U, j, k, &bj_i_minus) != 0 ||
          face_flux_density(grid,
              storage->predicted_face_flux[GAMERA_NO_I], GAMERA_NO_I,
              i, j, k, &bi_j_plus) != 0 ||
          face_flux_density(grid,
              storage->predicted_face_flux[GAMERA_NO_I], GAMERA_NO_I,
              i, j - 1U, k, &bi_j_minus) != 0) return -1;
      const double current_j =
          (bi_k_plus - bi_k_minus) - (bk_i_plus - bk_i_minus);
      const double current_k =
          (bj_i_plus - bj_i_minus) - (bi_j_plus - bi_j_minus);
      const double dl = sqrt(grid->edge[GAMERA_NO_J].value[edge_j].length *
                             grid->edge[GAMERA_NO_K].value[edge_k].length);
      const gamera_no_vec3 point_j =
          edge_midpoint(grid, GAMERA_NO_J, i, j, k);
      const gamera_no_vec3 point_k =
          edge_midpoint(grid, GAMERA_NO_K, i, j, k);
      gamera_solar_wind_state wind_j, wind_k;
      double weight_j, weight_k;
      if (!(dl > 0.0) ||
          sample_wind(b, point_j, b->step_time + 0.5 * b->step_dt,
                      &wind_j, &weight_j) != 0 ||
          sample_wind(b, point_k, b->step_time + 0.5 * b->step_dt,
                      &wind_k, &weight_k) != 0) return -1;
      const double coefficient_square =
          b->wind->by_coefficient * b->wind->by_coefficient +
          b->wind->bz_coefficient * b->wind->bz_coefficient;
      const double front_speed =
          fmax(fabs(wind_j.velocity[GAMERA_WIND_X]),
               fabs(wind_k.velocity[GAMERA_WIND_X])) /
          sqrt(1.0 + coefficient_square);
      const double diffusion_speed =
          fmin(front_speed, b->cfl * dl / b->step_dt);
      storage->edge_emf[GAMERA_NO_J][edge_j] +=
          weight_j * diffusion_speed * current_j * dl;
      storage->edge_emf[GAMERA_NO_K][edge_k] +=
          weight_k * diffusion_speed * current_k * dl;
    }
  }
  return 0;
}

static int earth_emf_filtered(
    gamera_no_storage *storage, const gamera_no_grid *grid,
    const size_t lower[3], const size_t upper[3],
    const gamera_cs_earth_boundary *b,
    int use_inner, int use_outer) {
  if (storage == NULL || grid == NULL || lower == NULL || upper == NULL ||
      !valid(b) || !isfinite(b->step_time) || !(b->step_dt > 0.0))
    return -1;
  const size_t outer = upper[GAMERA_NO_I];
  const size_t first = outer > 3U ? outer - 3U : 0U;
  if (use_outer)
    for (int direction = GAMERA_NO_I; direction <= GAMERA_NO_K;
         ++direction) {
    for (size_t i = first;
         i <= outer && i < grid->edge[direction].extent[0]; ++i)
      for (size_t j = 0; j < grid->edge[direction].extent[1]; ++j)
        for (size_t k = 0; k < grid->edge[direction].extent[2]; ++k) {
          const size_t index = gamera_no_index3(
              grid->edge[direction].extent, i, j, k);
          if (grid->edge[direction].valid[index] == 0U) continue;
          const gamera_no_vec3 point =
              edge_midpoint(grid, direction, i, j, k);
          gamera_solar_wind_state wind;
          double weight;
          if (sample_wind(b, point, b->step_time + 0.5 * b->step_dt,
                          &wind, &weight) != 0) return -1;
          if (i < outer) weight /= 1.0 + (double)(outer - i);
          const double electric[3] = {
              -(wind.velocity[1] * wind.magnetic[2] -
                wind.velocity[2] * wind.magnetic[1]),
              -(wind.velocity[2] * wind.magnetic[0] -
                wind.velocity[0] * wind.magnetic[2]),
              -(wind.velocity[0] * wind.magnetic[1] -
                wind.velocity[1] * wind.magnetic[0])};
          const gamera_no_edge_geometry *geometry =
              &grid->edge[direction].value[index];
          double line_integral = 0.0;
          for (int c = 0; c < GAMERA_NO_DIM; ++c)
            line_integral += electric[c] *
                             geometry->normal.value[c] * geometry->length;
          storage->edge_emf[direction][index] =
              (1.0 - weight) * storage->edge_emf[direction][index] +
              weight * line_integral;
        }
  }
  if (use_outer &&
      add_wind_diffusive_emf(storage, grid, lower, upper, b) != 0)
    return -1;
  /* E_t = 0 on the stationary, perfectly conducting inner shell. */
  if (use_inner)
    for (int direction = GAMERA_NO_J; direction <= GAMERA_NO_K;
         ++direction)
    for (size_t j = 0; j < grid->edge[direction].extent[1]; ++j)
      for (size_t k = 0; k < grid->edge[direction].extent[2]; ++k) {
        const size_t index = gamera_no_index3(
            grid->edge[direction].extent, lower[GAMERA_NO_I], j, k);
        if (grid->edge[direction].valid[index] != 0U)
          storage->edge_emf[direction][index] = 0.0;
      }
  if (use_inner && gamera_cs_mi_dipole_ready(b->mi) &&
      gamera_cs_mi_dipole_apply_inner_emf(
          b->mi, storage, grid, lower[GAMERA_NO_I]) != 0) return -1;
  return 0;
}

int gamera_cs_earth_no_mi_emf(
    gamera_no_storage *storage, const gamera_no_grid *grid,
    const size_t lower[3], const size_t upper[3], void *context) {
  return earth_emf_filtered(storage, grid, lower, upper,
                            context, 1, 1);
}

int gamera_cs_earth_tile_emf(
    gamera_no_storage *storage, const gamera_no_grid *grid,
    const size_t lower[3], const size_t upper[3], void *context) {
  const gamera_cs_earth_tile_emf_context *tile_context = context;
  if (tile_context == NULL || tile_context->tile == NULL ||
      tile_context->earth == NULL) return -1;
  const gamera_cs_decomposition *tile = tile_context->tile;
  return earth_emf_filtered(storage, grid, lower, upper,
      tile_context->earth, tile->coord[0] == 0,
      tile->coord[0] == tile->proc_dims[0] - 1);
}
