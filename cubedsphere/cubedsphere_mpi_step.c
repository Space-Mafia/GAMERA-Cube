#include "cubedsphere_mpi_step.h"

#include "cubedsphere_halo.h"
#include "cubedsphere_earth_boundary.h"
#include "nonorthogonal_step.h"
#include "cubedsphere_profile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { PERF_INIT_SYNC, PERF_HALOS_BOUNDARY, PERF_PREDICT,
       PERF_MHD_SWEEP, PERF_FLUX_SYNC, PERF_EMF_SWEEP, PERF_EMF_SYNC,
       PERF_UPDATE, PERF_FINAL_SYNC, PERF_STAGE_COUNT };

static int perf_enabled(void) {
  static int enabled = -1;
  if (enabled < 0)
    enabled = getenv("GAMERA_CS_PROFILE_STEP") != NULL;
  return enabled;
}

static void perf_report(int rank, const double part[PERF_STAGE_COUNT]) {
  static double totals[PERF_STAGE_COUNT] = {0};
  static long steps = 0;
  for (int stage = 0; stage < PERF_STAGE_COUNT; ++stage)
    totals[stage] += part[stage];
  if (++steps % 1000 != 0) return;
  fprintf(stderr,
      "CS_STEP_PROFILE rank=%d steps=%ld init_sync=%.6f "
      "halos_boundary=%.6f predict=%.6f mhd_sweep=%.6f "
      "flux_sync=%.6f emf_sweep=%.6f emf_sync=%.6f "
      "update=%.6f final_sync=%.6f\n",
      rank, steps, totals[PERF_INIT_SYNC], totals[PERF_HALOS_BOUNDARY],
      totals[PERF_PREDICT], totals[PERF_MHD_SWEEP],
      totals[PERF_FLUX_SYNC], totals[PERF_EMF_SWEEP],
      totals[PERF_EMF_SYNC], totals[PERF_UPDATE],
      totals[PERF_FINAL_SYNC]);
}

static int all_ok(int local, MPI_Comm communicator) {
  int global = 0;
  return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && global;
}

static int same_value(double a, double b) {
  const double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
  return isfinite(a) && isfinite(b) && fabs(a - b) <= 8e-15 * scale;
}

int gamera_cs_mpi_advance_tiled(
    const gamera_no_grid *grid, gamera_no_storage *local,
    const gamera_cs_decomposition *tile, size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field *background,
    gamera_cs_tiled_cell_halo_plan *cell_halo,
    gamera_cs_tiled_magnetic_halo_plan *magnetic_halo,
    gamera_cs_tiled_cube_face_sync_plan *face_plan,
    gamera_cs_tiled_edge_sync_plan *edge_plan,
    gamera_cs_tiled_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator) {
  static gamera_cs_profile tile_perf = {0};
  static const char *const tile_names[] = {"validation", "internal_cell", "initial_B_sync",
    "internal_B_halo", "cube_cell_halo", "cube_B_halo", "corner_cell", "corner_B",
    "radial_boundary", "predict", "mhd_sweep", "flux_sync", "emf_sweep",
    "emf_sync", "update", "final_B_sync"};
  cs_profile_begin(&tile_perf);
  int rank = -1, ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS) return -1;
  int valid = grid != NULL && local != NULL && tile != NULL &&
      options != NULL && cell_halo != NULL && magnetic_halo != NULL &&
      face_plan != NULL && ghost >= 4 && isfinite(dt) && dt > 0.0 &&
      isfinite(predictor_ratio) && isfinite(options->source_time) &&
      options->cell_source == NULL &&
      options->indexed_cell_source == NULL &&
      options->update.use_background == options->stress.use_background &&
      options->update.use_background == options->emf.use_background &&
      (!options->update.use_background || background != NULL) &&
      (!options->update.use_mhd || edge_plan != NULL) &&
      same_value(options->stress.gamma, options->update.gamma) &&
      same_value(options->stress.density_floor,
                 options->update.density_floor) &&
      same_value(options->stress.pressure_floor,
                 options->update.pressure_floor) &&
      same_value(options->emf.density_floor,
                 options->update.density_floor) &&
      options->stress.use_mhd == options->update.use_mhd &&
      options->stress.use_boris == options->update.use_boris &&
      options->emf.use_boris == options->update.use_boris &&
      (!options->update.use_boris ||
       (same_value(options->stress.light_speed,
                   options->update.light_speed) &&
        same_value(options->emf.light_speed,
                   options->update.light_speed))) &&
      (radial_boundary == NULL ||
       (isfinite(current_time) && isfinite(previous_time))) &&
      (tile == NULL || (ranks == GAMERA_CS_FACE_COUNT *
          tile->proc_dims[0] * tile->proc_dims[1] * tile->proc_dims[2]));
  /* A full-face Earth callback would apply both physical EMF boundaries
   * to an internal radial MPI tile. Require the ownership-aware adapter. */
  if (valid && tile->proc_dims[0] > 1 &&
      options->edge_emf_sync != NULL &&
      options->edge_emf_sync != gamera_cs_earth_tile_emf) valid = 0;
  if (valid)
    for (int axis = 0; axis < 3; ++axis)
      valid &= grid->cell_extent[axis] ==
          tile->local_active[axis] + 2U * ghost &&
          local->cell_extent[axis] == grid->cell_extent[axis];
  if (!all_ok(valid, communicator)) return -1;
  const size_t lower[3] = {ghost, ghost, ghost};
  const size_t upper[3] = {ghost + tile->local_active[0],
      ghost + tile->local_active[1], ghost + tile->local_active[2]};
  const gamera_no_background_field *field =
      options->update.use_background ? background : NULL;
  cs_profile_mark(&tile_perf, 0);

  if (cs_profile_status(&tile_perf, 1, gamera_cs_mpi_exchange_internal_cell_halos(
          local, tile, ghost, communicator)) != 0 ||
      cs_profile_status(&tile_perf, 2, gamera_cs_mpi_sync_internal_magnetic_faces(
          local, tile, ghost, communicator)) != 0 ||
      cs_profile_status(&tile_perf, 2, gamera_cs_tiled_cube_face_sync_magnetic(face_plan, local)) != 0 ||
      cs_profile_status(&tile_perf, 3, gamera_cs_mpi_exchange_internal_magnetic_face_halos(
          local, tile, ghost, communicator)) != 0 ||
      cs_profile_status(&tile_perf, 4, gamera_cs_tiled_cell_halo_fill(cell_halo, local)) != 0 ||
      cs_profile_status(&tile_perf, 5, gamera_cs_tiled_magnetic_halo_fill(magnetic_halo, local)) != 0 ||
      /* The cube-face fill refreshes angular ghosts only at active radial
       * planes. Copy those refreshed values through internal radial planes
       * as well, including the corner stencils used by CT edge EMFs. */
      cs_profile_status(&tile_perf, 6, gamera_cs_mpi_exchange_internal_cell_halos(
          local, tile, ghost, communicator)) != 0 ||
      cs_profile_status(&tile_perf, 7, gamera_cs_mpi_exchange_internal_magnetic_face_halos(
          local, tile, ghost, communicator)) != 0)
    return -1;
  int ok = radial_boundary == NULL ||
      radial_boundary(grid, local, tile, ghost, field,
          current_time, previous_time, boundary_context) == 0;
  const char *failure_stage = "radial_boundary";
  cs_profile_mark(&tile_perf, 8);
  if (ok) {
    failure_stage = "predict";
    ok = gamera_no_predict_storage(local, grid, predictor_ratio,
        options->update.gamma, options->update.density_floor,
        options->update.pressure_floor) == 0;
  }
  cs_profile_mark(&tile_perf, 9);
  if (ok) {
    failure_stage = "face_sweep";
    const double *predicted_flux[3] = {
        local->predicted_face_flux[0], local->predicted_face_flux[1],
        local->predicted_face_flux[2]};
    for (int direction = 0; direction < 3 && ok; ++direction) {
      ok = gamera_no_sweep_face_fluxes(grid, local,
          local->predicted_conserved, local->predicted_cell_magnetic,
          predicted_flux, direction, lower, upper, &options->stress,
          field) == 0;
      if (!ok) fprintf(stderr, "TILE_FAILURE rank=%d stage=face_sweep direction=%d\n",rank,direction);
    }
  }
  if (!ok) fprintf(stderr, "TILE_FAILURE rank=%d stage=%s time=%.17g dt=%.17g ratio=%.17g\n",rank,failure_stage,current_time,dt,predictor_ratio);
  cs_profile_mark(&tile_perf, 10);
  if (!all_ok(ok, communicator) ||
      gamera_cs_mpi_sync_internal_mhd_faces(
          local, tile, ghost, communicator) != 0 ||
      gamera_cs_tiled_cube_face_sync_mhd(face_plan, local) != 0)
    return -1;
  if (options->fluid_flux_sync != NULL && tile->coord[0] == 0)
    ok = options->fluid_flux_sync(local, grid, lower, upper,
                                  options->fluid_flux_context) == 0;
  if (!ok) fprintf(stderr, "TILE_FAILURE rank=%d stage=fluid_wall\n",rank);
  cs_profile_mark(&tile_perf, 11);
  if (ok && options->update.use_mhd) {
    gamera_no_emf_options emf = options->emf;
    emf.dt = dt;
    const double *predicted_flux[3] = {
        local->predicted_face_flux[0], local->predicted_face_flux[1],
        local->predicted_face_flux[2]};
    for (int direction = 0; direction < 3 && ok; ++direction) {
      ok = gamera_no_sweep_edge_emf(grid, local,
          local->predicted_conserved, predicted_flux, direction,
          lower, upper, &emf, field) == 0;
      if (!ok) fprintf(stderr, "TILE_FAILURE rank=%d stage=edge_sweep direction=%d time=%.17g dt=%.17g\n",rank,direction,current_time,dt);
    }
    if (ok && options->edge_emf_sync != NULL) {
      ok = options->edge_emf_sync(local, grid, lower, upper,
                                  options->edge_emf_context) == 0;
      if (!ok) fprintf(stderr, "TILE_FAILURE rank=%d stage=wall_emf\n",rank);
    }
  }
  cs_profile_mark(&tile_perf, 12);
  if (!all_ok(ok, communicator)) return -1;
  if (options->update.use_mhd) {
    if (gamera_cs_tiled_edge_sync(edge_plan, local) != 0) return -1;
    if (gamera_cs_mpi_sync_internal_edge_emf(
            local, tile, ghost, communicator) != 0) return -1;
  }
  cs_profile_mark(&tile_perf, 13);
  ok = gamera_no_calculate_stress_rates(grid, local, lower, upper,
           options->update.use_mhd,
           field == NULL ? NULL : field->cell_force) == 0 &&
       gamera_no_apply_active_update(local, grid, lower, upper, dt,
           &options->update,
           field == NULL ? NULL : field->cell_magnetic) == 0;
  if (!ok) fprintf(stderr, "TILE_FAILURE rank=%d stage=stress_or_active_update time=%.17g dt=%.17g\n",rank,current_time,dt);
  cs_profile_mark(&tile_perf, 14);
  if (!all_ok(ok, communicator)) return -1;
  if (options->update.use_mhd) {
    if (gamera_cs_mpi_sync_internal_magnetic_faces(
            local, tile, ghost, communicator) != 0 ||
        gamera_cs_tiled_cube_face_sync_magnetic(face_plan, local) != 0)
      return -1;
    const double *flux[3] = {local->face_flux[0], local->face_flux[1],
                             local->face_flux[2]};
    ok = gamera_no_recover_magnetic_field(grid, flux,
                                           local->cell_magnetic) == 0;
    if (!ok) fprintf(stderr, "TILE_FAILURE rank=%d stage=final_recover_B\n",rank);
    if (!all_ok(ok, communicator)) return -1;
  }
  cs_profile_mark(&tile_perf, 15);
  if (++tile_perf.steps % 100 == 0)
    cs_profile_report(&tile_perf, "CS_TILE_PROFILE", rank, tile_names, 16);
  return 0;
}

static int advance_six_impl(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    const gamera_cs_edge_sync_plan *edge_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator, int pre_synced, int sparse_final) {
  int rank = -1, ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS ||
      rank < 0 || ranks != GAMERA_CS_FACE_COUNT) return -1;
  const int valid = grid != NULL && storage != NULL && active != NULL &&
      options != NULL && ghost >= 4 && isfinite(dt) && dt > 0.0 &&
      isfinite(predictor_ratio) &&
      (radial_boundary == NULL ||
       (isfinite(current_time) && isfinite(previous_time))) &&
      (active != NULL && active[1] == active[2]) &&
      (options != NULL && options->cell_source == NULL &&
       options->indexed_cell_source == NULL &&
       isfinite(options->source_time) &&
       options->update.use_background == options->stress.use_background &&
       options->update.use_background == options->emf.use_background &&
       (!options->update.use_background || background != NULL) &&
       same_value(options->stress.gamma, options->update.gamma) &&
       same_value(options->stress.density_floor,
                  options->update.density_floor) &&
       same_value(options->stress.pressure_floor,
                  options->update.pressure_floor) &&
       same_value(options->emf.density_floor,
                  options->update.density_floor) &&
       options->stress.use_mhd == options->update.use_mhd &&
       options->stress.use_boris == options->update.use_boris &&
       options->emf.use_boris == options->update.use_boris &&
       (!options->update.use_boris ||
        (same_value(options->stress.light_speed,
                    options->update.light_speed) &&
         same_value(options->emf.light_speed,
                    options->update.light_speed))));
  if (!all_ok(valid, communicator)) return -1;
  const size_t lower[3] = {ghost, ghost, ghost};
  const size_t upper[3] = {ghost + active[0], ghost + active[1],
                           ghost + active[2]};
  gamera_no_storage *local = &storage[rank];
  const gamera_no_grid *local_grid = &grid[rank];
  const gamera_no_background_field *local_background =
      options->update.use_background ? &background[rank] : NULL;
  const int profiled = perf_enabled();
  double perf_part[PERF_STAGE_COUNT] = {0};
  double perf_last = profiled ? MPI_Wtime() : 0.0;
#define PERF_MARK(stage) do { \
    if (profiled) { \
      const double now = MPI_Wtime(); \
      perf_part[stage] = now - perf_last; \
      perf_last = now; \
    } \
  } while (0)

  /* The active-cell collective supplies both time levels and Cartesian B
   * to the unchanged high-order donor interpolation. All collectives are
   * reached in the same order even if an individual face operation fails. */
  if (!pre_synced &&
      (gamera_cs_mpi_exchange_active_cells(
           storage, active, ghost, communicator) != 0 ||
       gamera_cs_mpi_sync_magnetic_face_flux(
           storage, active, ghost, communicator) != 0)) return -1;
  PERF_MARK(PERF_INIT_SYNC);
  int ok = (halo_plan != NULL ?
      gamera_cs_fill_tangential_cell_halos_planned_face(
          halo_plan, grid, storage, rank) :
      gamera_cs_fill_tangential_cell_halos(
          grid, storage, active, ghost)) == 0 &&
      gamera_cs_fill_radial_outflow_cells_face(
          storage, active, ghost, rank) == 0 &&
      (halo_plan != NULL ?
       gamera_cs_fill_magnetic_face_halos_planned_face(
           halo_plan, grid, storage, rank) :
       gamera_cs_fill_magnetic_face_halos(
           grid, storage, active, ghost)) == 0;
  if (ok && radial_boundary == gamera_cs_earth_fill_radial_ghosts)
    ok = gamera_cs_earth_fill_radial_ghosts_face(
        grid, storage, active, ghost, current_time, previous_time,
        boundary_context, rank) == 0;
  else if (ok && radial_boundary != NULL)
    ok = radial_boundary(grid, storage, active, ghost, current_time,
                         previous_time, boundary_context) == 0;
  PERF_MARK(PERF_HALOS_BOUNDARY);
  if (ok)
    ok = gamera_no_predict_storage(local, local_grid, predictor_ratio,
              options->update.gamma, options->update.density_floor,
              options->update.pressure_floor) == 0;
  PERF_MARK(PERF_PREDICT);
  if (ok) {
    const double *predicted_flux[3] = {
        local->predicted_face_flux[0], local->predicted_face_flux[1],
        local->predicted_face_flux[2]};
    for (int direction = 0; direction < 3 && ok; ++direction)
      ok = gamera_no_sweep_face_fluxes(
          local_grid, local, local->predicted_conserved,
          local->predicted_cell_magnetic, predicted_flux, direction,
          lower, upper, &options->stress, local_background) == 0;
  }
  PERF_MARK(PERF_MHD_SWEEP);
  if (!all_ok(ok, communicator) ||
      gamera_cs_mpi_sync_mhd_face_fluxes(
          storage, active, ghost, communicator) != 0) return -1;
  PERF_MARK(PERF_FLUX_SYNC);
  if (options->fluid_flux_sync != NULL)
    ok = options->fluid_flux_sync(local, local_grid, lower, upper,
                                  options->fluid_flux_context) == 0;
  if (ok && options->update.use_mhd) {
    gamera_no_emf_options emf = options->emf;
    emf.dt = dt;
    const double *predicted_flux[3] = {
        local->predicted_face_flux[0], local->predicted_face_flux[1],
        local->predicted_face_flux[2]};
    for (int direction = 0; direction < 3 && ok; ++direction)
      ok = gamera_no_sweep_edge_emf(
          local_grid, local, local->predicted_conserved, predicted_flux,
          direction, lower, upper, &emf, local_background) == 0;
    if (ok && options->edge_emf_sync != NULL)
      ok = options->edge_emf_sync(local, local_grid, lower, upper,
                                  options->edge_emf_context) == 0;
  }
  PERF_MARK(PERF_EMF_SWEEP);
  if (!all_ok(ok, communicator)) return -1;
  if (options->update.use_mhd &&
      (edge_plan != NULL ?
       gamera_cs_mpi_sync_edge_emf_planned(
           grid, storage, active, ghost, edge_plan, communicator) :
       gamera_cs_mpi_sync_edge_emf(
           grid, storage, active, ghost, communicator)) != 0) return -1;
  PERF_MARK(PERF_EMF_SYNC);

  const gamera_no_vec3 *force = local_background == NULL ? NULL :
      local_background->cell_force;
  const gamera_no_vec3 *magnetic = local_background == NULL ? NULL :
      local_background->cell_magnetic;
  ok = gamera_no_calculate_stress_rates(
           local_grid, local, lower, upper, options->update.use_mhd,
           force) == 0 &&
       gamera_no_apply_active_update(
           local, local_grid, lower, upper, dt, &options->update,
           magnetic) == 0;
  PERF_MARK(PERF_UPDATE);
  if (!all_ok(ok, communicator)) return -1;
  if (options->update.use_mhd) {
    if (gamera_cs_mpi_sync_magnetic_face_flux(
            storage, active, ghost, communicator) != 0) return -1;
    const double *flux[3] = {local->face_flux[0], local->face_flux[1],
                             local->face_flux[2]};
    ok = gamera_no_recover_magnetic_field(local_grid, flux,
                                           local->cell_magnetic) == 0;
    if (!all_ok(ok, communicator)) return -1;
  }
  const int final_status = sparse_final ?
      gamera_cs_mpi_exchange_active_cells_sparse(
          storage, active, ghost, halo_plan, communicator) :
      gamera_cs_mpi_exchange_active_cells(
          storage, active, ghost, communicator);
  PERF_MARK(PERF_FINAL_SYNC);
  if (profiled && final_status == 0) perf_report(rank, perf_part);
#undef PERF_MARK
  return final_status;
}

int gamera_cs_mpi_advance_six_with_background_planned(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator) {
  return advance_six_impl(grid, storage, active, ghost, predictor_ratio, dt,
                          options, background, halo_plan, NULL,
                          radial_boundary, current_time, previous_time,
                          boundary_context, communicator, 0, 0);
}

int gamera_cs_mpi_advance_six_with_background_planned_cached_edges(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    const gamera_cs_edge_sync_plan *edge_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator) {
  return advance_six_impl(grid, storage, active, ghost, predictor_ratio, dt,
                          options, background, halo_plan, edge_plan,
                          radial_boundary, current_time, previous_time,
                          boundary_context, communicator, 0, 0);
}

int gamera_cs_mpi_advance_six_with_background_planned_cached_edges_pre_synced(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    const gamera_cs_edge_sync_plan *edge_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator) {
  return advance_six_impl(grid, storage, active, ghost, predictor_ratio, dt,
                          options, background, halo_plan, edge_plan,
                          radial_boundary, current_time, previous_time,
                          boundary_context, communicator, 1, 0);
}

int gamera_cs_mpi_advance_six_with_background_sparse_pre_synced(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double predictor_ratio, double dt,
    const gamera_no_advance_options *options,
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_halo_plan *halo_plan,
    const gamera_cs_edge_sync_plan *edge_plan,
    gamera_cs_radial_boundary_function radial_boundary,
    double current_time, double previous_time, void *boundary_context,
    MPI_Comm communicator) {
  if (halo_plan == NULL) return -1;
  return advance_six_impl(grid, storage, active, ghost, predictor_ratio, dt,
                          options, background, halo_plan, edge_plan,
                          radial_boundary, current_time, previous_time,
                          boundary_context, communicator, 1, 1);
}
