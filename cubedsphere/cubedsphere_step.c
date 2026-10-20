#include "cubedsphere_step.h"

#include "cubedsphere_halo.h"
#include "cubedsphere_interface.h"

#include <math.h>

static int same_value(double a, double b) {
  const double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
  return isfinite(a) && isfinite(b) && fabs(a - b) <= 8e-15 * scale;
}

int gamera_cs_advance_six_with_background_planned(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context) {
  if (grid == NULL || storage == NULL || active == NULL || options == NULL ||
      ghost < 4 || !isfinite(dt) || dt <= 0.0 ||
      (radial_boundary != NULL &&
       (!isfinite(current_time) || !isfinite(previous_time))) ||
      !isfinite(predictor_ratio) || active[1] != active[2] ||
      options->cell_source != NULL || options->indexed_cell_source != NULL ||
      options->update.use_background != options->stress.use_background ||
      options->update.use_background != options->emf.use_background ||
      (options->update.use_background && background == NULL) ||
      !isfinite(options->source_time) ||
      !same_value(options->stress.gamma, options->update.gamma) ||
      !same_value(options->stress.density_floor,
                  options->update.density_floor) ||
      !same_value(options->stress.pressure_floor,
                  options->update.pressure_floor) ||
      !same_value(options->emf.density_floor,
                  options->update.density_floor) ||
      options->stress.use_mhd != options->update.use_mhd ||
      options->stress.use_boris != options->update.use_boris ||
      options->emf.use_boris != options->update.use_boris ||
      (options->update.use_boris &&
       (!same_value(options->stress.light_speed,
                    options->update.light_speed) ||
        !same_value(options->emf.light_speed,
                    options->update.light_speed)))) return -1;
  const size_t lower[3] = {ghost, ghost, ghost};
  const size_t upper[3] = {ghost + active[0], ghost + active[1],
                           ghost + active[2]};
  if (gamera_cs_sync_magnetic_face_flux(storage, active, ghost) != 0 ||
      (halo_plan != NULL ?
       gamera_cs_fill_tangential_cell_halos_planned(
           halo_plan, grid, storage) :
       gamera_cs_fill_tangential_cell_halos(
           grid, storage, active, ghost)) != 0 ||
      gamera_cs_fill_radial_outflow_cells(storage, active, ghost) != 0 ||
      (halo_plan != NULL ?
       gamera_cs_fill_magnetic_face_halos_planned(
           halo_plan, grid, storage) :
       gamera_cs_fill_magnetic_face_halos(
           grid, storage, active, ghost)) != 0)
    return -1;
  if (radial_boundary != NULL &&
      radial_boundary(grid, storage, active, ghost, current_time,
                      previous_time, boundary_context) != 0) return -1;

  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    if (gamera_no_predict_storage(&storage[face], &grid[face],
            predictor_ratio, options->update.gamma,
            options->update.density_floor,
            options->update.pressure_floor) != 0) return -1;
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    const gamera_no_background_field *face_background =
        options->update.use_background ? &background[face] : NULL;
    const double *predicted_flux[3] = {
        storage[face].predicted_face_flux[0],
        storage[face].predicted_face_flux[1],
        storage[face].predicted_face_flux[2]};
    for (int direction = 0; direction < 3; ++direction) {
      if (gamera_no_sweep_face_fluxes(
              &grid[face], &storage[face],
              storage[face].predicted_conserved,
              storage[face].predicted_cell_magnetic,
              predicted_flux, direction, lower, upper,
              &options->stress, face_background) != 0) return -1;
    }
  }
  if (gamera_cs_sync_mhd_face_fluxes(storage, active, ghost) != 0) return -1;
  /* Reuse the non-orthogonal physical-boundary hook at the same point as
   * gamera_no_advance: after fluid sweeps, before stress-rate accumulation.
   * Cube-face flux reconciliation precedes this radial-boundary correction. */
  if (options->fluid_flux_sync != NULL) {
    for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
      if (options->fluid_flux_sync(&storage[face], &grid[face], lower, upper,
                                   options->fluid_flux_context) != 0)
        return -1;
    }
  }
  if (options->update.use_mhd) {
    gamera_no_emf_options emf = options->emf;
    emf.dt = dt;
    for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
      const gamera_no_background_field *face_background =
          options->update.use_background ? &background[face] : NULL;
      const double *predicted_flux[3] = {
          storage[face].predicted_face_flux[0],
          storage[face].predicted_face_flux[1],
          storage[face].predicted_face_flux[2]};
      for (int direction = 0; direction < 3; ++direction) {
        if (gamera_no_sweep_edge_emf(
                &grid[face], &storage[face],
                storage[face].predicted_conserved,
                predicted_flux, direction, lower, upper,
                &emf, face_background) != 0) return -1;
      }
    }
    /* Apply each patch's physical radial EMF condition before sharing one
     * signed EMF at every inter-patch edge and cube corner. */
    if (options->edge_emf_sync != NULL) {
      for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
        if (options->edge_emf_sync(&storage[face], &grid[face], lower, upper,
                                   options->edge_emf_context) != 0)
          return -1;
      }
    }
    if (gamera_cs_sync_edge_emf(grid, storage, active, ghost) != 0) return -1;
  }
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    const gamera_no_background_field *face_background =
        options->update.use_background ? &background[face] : NULL;
    const gamera_no_vec3 *face_force =
        face_background == NULL ? NULL : face_background->cell_force;
    const gamera_no_vec3 *face_magnetic =
        face_background == NULL ? NULL : face_background->cell_magnetic;
    if (gamera_no_calculate_stress_rates(&grid[face], &storage[face],
            lower, upper, options->update.use_mhd, face_force) != 0 ||
        gamera_no_apply_active_update(&storage[face], &grid[face],
            lower, upper, dt, &options->update, face_magnetic) != 0) return -1;
  }
  if (options->update.use_mhd) {
    if (gamera_cs_sync_magnetic_face_flux(storage, active, ghost) != 0)
      return -1;
    for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
      const double *flux[3] = {storage[face].face_flux[0],
                               storage[face].face_flux[1],
                               storage[face].face_flux[2]};
      if (gamera_no_recover_magnetic_field(&grid[face], flux,
               storage[face].cell_magnetic) != 0) return -1;
    }
  }
  return 0;
}

int gamera_cs_advance_six_with_background(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context) {
  return gamera_cs_advance_six_with_background_planned(
      grid, storage, active, ghost, predictor_ratio, dt, options,
      background, NULL, radial_boundary, current_time, previous_time,
      boundary_context);
}

int gamera_cs_advance_six_with_boundary(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context) {
  return gamera_cs_advance_six_with_background(
      grid, storage, active, ghost, predictor_ratio, dt, options, NULL,
      radial_boundary, current_time, previous_time, boundary_context);
}

int gamera_cs_advance_six(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options) {
  return gamera_cs_advance_six_with_boundary(
      grid, storage, active, ghost, predictor_ratio, dt, options,
      NULL, 0.0, 0.0, NULL);
}
