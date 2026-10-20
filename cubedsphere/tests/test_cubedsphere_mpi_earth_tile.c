#include "cubedsphere_decomposition.h"
#include "cubedsphere_earth_boundary.h"
#include "cubedsphere_earth_startup.h"
#include "cubedsphere_mesh.h"
#include "cubedsphere_mpi_step.h"
#include "cubedsphere_profile.h"
#include "cubedsphere_mi_dipole_mpi.h"
#include "nonorthogonal_background.h"
#include "nonorthogonal_initialization.h"
#include "nonorthogonal_radial_map.h"
#include "nonorthogonal_state.h"
#include "nonorthogonal_step.h"
#include "solar_wind.h"
#ifdef GAMERA_CS_DRIVE_HDF5
#include "cubedsphere_analysis_hdf5.h"
#endif

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A physical Earth update on truly local tiles. All MHD, background,
 * wind and wall routines are the same ones used by the accepted six-face
 * Earth driver; this test adds no alternate numerical scheme. */
static int positive_size(const char *text, size_t *value) {
  char *end = NULL;
  const unsigned long long parsed = strtoull(text, &end, 10);
  if (end == text || *end != '\0' || parsed == 0 ||
      parsed > (unsigned long long)SIZE_MAX) return -1;
  *value = (size_t)parsed;
  return 0;
}

static int positive_seconds(const char *name, double fallback,
                            double *seconds) {
  const char *text = getenv(name);
  if (text == NULL) {
    *seconds = fallback;
    return 0;
  }
  char *end = NULL;
  const double value = strtod(text, &end);
  if (end == text || *end != '\0' || !isfinite(value) || value <= 0.0)
    return -1;
  *seconds = value;
  return 0;
}

/* Stream one active tile at a time into the existing six-face CSV format.
 * Root never allocates a full radial grid; temporary chunks are removed only
 * after their rows have been copied into this run's completed frame. */
#ifndef GAMERA_CS_DRIVE_HDF5
static int write_tiled_frame(const char *prefix, long frame, long step,
                             double seconds, const gamera_no_grid *grid,
                             const gamera_no_storage *storage,
                             const gamera_no_background_data *owned,
                             const gamera_cs_decomposition *tile,
                             size_t ghost, double gamma,
                             const gamera_cs_mi_dipole_mpi *mi,
                             FILE *index, MPI_Comm comm) {
  int rank = -1, ranks = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  char output_path[4096], chunk_path[4096];
  const int output_length = snprintf(output_path, sizeof(output_path),
                                     "%s_%06ld.csv", prefix, frame);
  int failed = output_length < 0 ||
      (size_t)output_length >= sizeof(output_path);
  if (!failed) {
    const int chunk_length = snprintf(chunk_path, sizeof(chunk_path),
                                      "%s.rank%03d", output_path, rank);
    failed = chunk_length < 0 ||
        (size_t)chunk_length >= sizeof(chunk_path);
  }
  FILE *chunk = failed ? NULL : fopen(chunk_path, "w");
  if (chunk == NULL) failed = 1;
  const size_t lower[3] = {ghost, ghost, ghost};
  const size_t upper[3] = {
      ghost + tile->local_active[0], ghost + tile->local_active[1],
      ghost + tile->local_active[2]};
  if (!failed) {
    for (size_t i = lower[0]; i < upper[0]; ++i)
      for (size_t j = lower[1]; j < upper[1]; ++j)
        for (size_t k = lower[2]; k < upper[2]; ++k) {
          const size_t cell = gamera_no_index3(grid->cell_extent, i, j, k);
          gamera_no_primitive state;
          if (gamera_no_conserved_to_primitive(
                  &storage->conserved[cell * GAMERA_NO_FLUX_COUNT],
                  gamma, 1e-12, 1e-12, &state) != 0) {
            failed = 1;
            break;
          }
          const gamera_no_vec3 point = grid->cell[cell].centroid;
          const gamera_no_vec3 b0 = owned->cell_magnetic[cell];
          const gamera_no_vec3 b1 = storage->cell_magnetic[cell];
          if (fprintf(chunk,
                  "%d,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,"
                  "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                  tile->face, tile->global_start[0] + i - ghost,
                  tile->global_start[1] + j - ghost,
                  tile->global_start[2] + k - ghost,
                  point.value[0], point.value[1], point.value[2],
                  state.density, state.pressure,
                  state.velocity.value[0], state.velocity.value[1],
                  state.velocity.value[2],
                  b0.value[0] + b1.value[0],
                  b0.value[1] + b1.value[1],
                  b0.value[2] + b1.value[2]) < 0)
            failed = 1;
        }
    if (fclose(chunk) != 0) failed = 1;
  }
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
  if (any_failed) return -1;
  if (rank == 0) {
    FILE *output = fopen(output_path, "w");
    if (output == NULL ||
        fputs("face,i,j,k,x,y,z,rho,p,vx,vy,vz,bx,by,bz\n", output) < 0)
      failed = 1;
    for (int source = 0; source < ranks && !failed; ++source) {
      const int chunk_length = snprintf(chunk_path, sizeof(chunk_path),
                                        "%s.rank%03d", output_path, source);
      if (chunk_length < 0 ||
          (size_t)chunk_length >= sizeof(chunk_path)) {
        failed = 1;
        break;
      }
      FILE *input = fopen(chunk_path, "r");
      if (input == NULL) {
        failed = 1;
        break;
      }
      char buffer[65536];
      size_t count;
      while ((count = fread(buffer, 1, sizeof(buffer), input)) != 0)
        if (fwrite(buffer, 1, count, output) != count) {
          failed = 1;
          break;
        }
      if (ferror(input)) failed = 1;
      if (fclose(input) != 0) failed = 1;
      if (!failed && remove(chunk_path) != 0) failed = 1;
    }
    if (output != NULL && fclose(output) != 0) failed = 1;
    if (!failed && mi != NULL &&
        (gamera_cs_mi_dipole_write_csv(
             gamera_cs_mi_dipole_mpi_state(mi), output_path) != 0 ||
         gamera_cs_mi_dipole_write_shell_current_csv(
             gamera_cs_mi_dipole_mpi_state(mi), output_path) != 0))
      failed = 1;
    if (!failed && (index == NULL ||
        fprintf(index, "%ld,%ld,%.17g,%s\n", frame, step, seconds,
                output_path) < 0 || fflush(index) != 0)) failed = 1;
  }
  MPI_Bcast(&failed, 1, MPI_INT, 0, comm);
  return failed ? -1 : 0;
}
#endif

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = -1, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int failed = argc != 7 && argc != 8 && argc != 9;
  const int drive = getenv("GAMERA_CS_TILE_DRIVE") != NULL;
  const char *constant_bz_text = getenv("GAMERA_CS_EARTH_CONSTANT_BZ_NT");
  double constant_bz_nt = 0.0;
  if (constant_bz_text != NULL) {
    char *end = NULL;
    constant_bz_nt = strtod(constant_bz_text, &end);
    if (end == constant_bz_text || *end != '\0' || !isfinite(constant_bz_nt))
      failed = 1;
  }
  double first_switch_seconds = 3600.0;
  double second_switch_seconds = 10800.0;
  double stop_seconds = 18000.0;
  double output_seconds = 600.0;
  if (drive && (argc < 8 ||
      positive_seconds("GAMERA_CS_EARTH_FIRST_SWITCH_SECONDS", 3600.0,
                       &first_switch_seconds) != 0 ||
      positive_seconds("GAMERA_CS_EARTH_SECOND_SWITCH_SECONDS", 10800.0,
                       &second_switch_seconds) != 0 ||
      positive_seconds("GAMERA_CS_EARTH_STOP_SECONDS", 18000.0,
                       &stop_seconds) != 0 ||
      positive_seconds("GAMERA_CS_EARTH_OUTPUT_SECONDS", 600.0,
                       &output_seconds) != 0 ||
      (constant_bz_text == NULL &&
       !(first_switch_seconds < second_switch_seconds &&
         second_switch_seconds < stop_seconds)))) failed = 1;
  size_t global_active[3] = {0};
  int proc_dims[3] = {0};
  size_t steps = 0;
  if (!failed) {
    for (int d = 0; d < 3; ++d) {
      size_t parsed = 0;
      if (positive_size(argv[d + 1], &parsed) != 0 ||
          parsed > (size_t)INT_MAX) failed = 1;
      proc_dims[d] = (int)parsed;
      if (positive_size(argv[d + 4], &global_active[d]) != 0)
        failed = 1;
    }
    steps = 2;
    if (argc == 9) {
      if (strcmp(argv[8], "0") == 0) steps = 0;
      else if (positive_size(argv[8], &steps) != 0) failed = 1;
    }
  }
  gamera_cs_decomposition tile;
  if (!failed && gamera_cs_decomposition_init(
          global_active, proc_dims, ranks, rank, &tile) != 0)
    failed = 1;
  const size_t ghost = 4;
  gamera_no_radial_map map;
  if (!failed && gamera_no_radial_map_init(
          &map, GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
          2.5, 200.0, 3.0) != 0) failed = 1;
  gamera_no_grid grid = {0};
  gamera_no_storage storage = {0};
  gamera_no_background_data owned = {0};
  gamera_cs_tiled_cell_halo_plan *cell_halo = NULL;
  gamera_cs_tiled_magnetic_halo_plan *magnetic_halo = NULL;
  gamera_cs_tiled_cube_face_sync_plan *face_plan = NULL;
  gamera_cs_tiled_edge_sync_plan *edge_plan = NULL;
  gamera_cs_mi_dipole_mpi *distributed_mi = NULL;
  gamera_solar_wind_series wind;
  gamera_solar_wind_init(&wind);
  gamera_cs_earth_startup startup = {
      .inner_radius = 2.5, .density = 5.0, .speed = 4.0,
      .mach = 10.0, .gamma = 5.0 / 3.0,
      .ambient_density = 1.0, .ambient_pressure = 0.001,
      .front_x = -30.0, .front_width = 1.5,
      .cut_radius = 16.0, .cut_length = 8.0,
      .imf = {{0.0, 0.0, 0.0}},
      .dipole = {{{0.0, 0.0, -6761.730688424097}}}};
  /* Initialize the upstream part of the accepted front with the SAME IMF
   * as the constant driver. Interior dipole/front initialization is retained. */
  if (drive && constant_bz_text != NULL) {
    const double bnorm = 100000.0 * sqrt(4.0e-7 * acos(-1.0) * 1.67262192369e-21);
    startup.imf.value[2] = constant_bz_nt * 1e-9 / bnorm;
  }
  gamera_no_background_field background[GAMERA_CS_FACE_COUNT] = {{0}};
  if (!failed) {
    const size_t extent[3] = {
        tile.local_active[0] + 2 * ghost,
        tile.local_active[1] + 2 * ghost,
        tile.local_active[2] + 2 * ghost};
    const size_t vertices_count = (extent[0] + 1) *
        (extent[1] + 1) * (extent[2] + 1);
    gamera_no_vec3 *vertices = calloc(vertices_count, sizeof(*vertices));
    if (vertices == NULL ||
        gamera_cs_generate_vertices_radial_map_tile(
            tile.face, global_active, tile.global_start,
            tile.local_active, ghost, &map, vertices) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid) != 0 ||
        gamera_cs_calibrate_face_frames(&grid) != 0 ||
        gamera_no_storage_create(extent, &storage) != 0 ||
        gamera_no_background_create(&grid,
            gamera_cs_earth_startup_background, &startup, &owned) != 0)
      failed = 1;
    free(vertices);
  }
  if (!failed) {
    background[tile.face] = owned.field;
    if (gamera_no_initialize_primitives(&grid, &storage,
            gamera_cs_earth_startup_primitive, &startup, false,
            startup.gamma, 1e-6, 1e-8) != 0 ||
        gamera_no_initialize_flux_from_vector_potential(&grid, &storage,
            gamera_cs_earth_startup_potential, &startup) != 0 ||
        gamera_no_save_current_as_old(&storage) != 0 ||
        gamera_cs_earth_background_adjust(&grid, &owned, &startup) != 0 ||
        gamera_cs_earth_total_to_residual(&grid, &storage, &owned) != 0)
      failed = 1;
  }
  if (!failed) {
    background[tile.face] = owned.field;
    const double wind_time = 0.0;
    const gamera_solar_wind_state wind_state = {
        .density = startup.density,
        .pressure = startup.density * startup.speed * startup.speed /
                    (startup.gamma * startup.mach * startup.mach),
        .velocity = {startup.speed, 0.0, 0.0},
        .magnetic = {0.0, 0.0, 0.0}};
    if (drive) {
      /* Exactly the accepted six-face Earth drive and normalization. */
      const double rho_norm = 1.67262192369e-21;
      const double magnetic_norm = 100000.0 *
          sqrt(4.0e-7 * acos(-1.0) * rho_norm);
      const double bz5_code = 5.0e-9 / magnetic_norm;
      const double times[3] = {0.0, first_switch_seconds / 63.71,
                               second_switch_seconds / 63.71};
      gamera_solar_wind_state states[3] = {
          wind_state, wind_state, wind_state};
      states[1].magnetic[GAMERA_WIND_Z] = bz5_code;
      states[2].magnetic[GAMERA_WIND_Z] = -bz5_code;
      wind.linear_interpolation = 0;
      if (constant_bz_text != NULL) {
        states[0].magnetic[GAMERA_WIND_Z] = constant_bz_nt * 1e-9 / magnetic_norm;
        if (gamera_solar_wind_set(&wind, 1, times, states) != 0) failed = 1;
      } else if (gamera_solar_wind_set(&wind, 3, times, states) != 0) failed = 1;
      if (rank == 0 && constant_bz_text != NULL)
        printf("EARTH_CONSTANT_IMF Bz_nT=%.17g stop_s=%.17g output_s=%.17g\n",
               constant_bz_nt, stop_seconds, output_seconds);
    } else if (gamera_solar_wind_set(&wind, 1, &wind_time,
                                     &wind_state) != 0) failed = 1;
    wind.reference[0] = startup.front_x;
  }
  if (!failed &&
      (gamera_cs_tiled_cell_halo_plan_create(&grid, &tile, ghost,
           MPI_COMM_WORLD, &cell_halo) != 0 ||
       gamera_cs_tiled_magnetic_halo_plan_create(&grid, &tile, ghost,
           MPI_COMM_WORLD, &magnetic_halo) != 0 ||
       gamera_cs_tiled_cube_face_sync_plan_create(&tile, ghost,
           MPI_COMM_WORLD, &face_plan) != 0 ||
       gamera_cs_tiled_edge_sync_plan_create(&grid, &tile, ghost,
           MPI_COMM_WORLD, &edge_plan) != 0)) failed = 1;
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  const int use_mi = drive || getenv("GAMERA_CS_TILE_TEST_MI") != NULL;
  if (!any_failed && use_mi) {
    const gamera_cs_mi_dipole_config mi_config = {
        .azimuth_points = 192U, .colatitude_points = 22U,
        .current_shell = 1U, .ionosphere_radius_re = 1.0,
        .pedersen_siemens = 5.0, .coupling_interval_s = 10.0,
        .length_norm_m = 6371000.0, .speed_norm_m_s = 100000.0,
        .mass_density_norm_kg_m3 = 1.67262192369e-21,
        .maximum_iterations = 4000, .relative_tolerance = 1e-8};
    if (gamera_cs_mi_dipole_mpi_create(&tile, ghost, &map,
            &startup, &mi_config, MPI_COMM_WORLD,
            &distributed_mi) != 0) {
      fprintf(stderr, "rank %d: tiled MI create failed\n", rank);
      failed = 1;
    }
  }
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  FILE *drive_index = NULL;
#ifdef GAMERA_CS_DRIVE_HDF5
  gamera_cs_analysis_hdf5 analysis = {0};
  analysis.face_comm = MPI_COMM_NULL;
  if (!any_failed && drive &&
      gamera_cs_analysis_hdf5_init(&analysis, argv[7], &tile,
                                   MPI_COMM_WORLD) != 0) failed = 1;
#else
  if (!any_failed && drive && rank == 0) {
    char index_path[4096];
    const int length = snprintf(index_path, sizeof(index_path),
                                "%s.index.csv", argv[7]);
    if (length < 0 || (size_t)length >= sizeof(index_path) ||
        (drive_index = fopen(index_path, "w")) == NULL ||
        fputs("frame,step,simulation_seconds,mhd_csv\n",
              drive_index) < 0)
      failed = 1;
  }
#endif
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (!any_failed) {
    gamera_cs_earth_boundary boundary = {
        .startup = &startup, .wind = &wind, .background = background,
        .mi = gamera_cs_mi_dipole_mpi_state(distributed_mi),
        .density_floor = 1e-6, .pressure_floor = 1e-8,
        .cfl = 0.3, .strict_inner_gas_wall = 1};
    gamera_cs_earth_tile_emf_context emf_context = {
        .tile = &tile, .earth = &boundary};
    gamera_no_advance_options options = {
        .stress = {startup.gamma, 1e-6, 1e-8, 1.0,
                   0.25, 0.25, 10.0, true, true, false, true},
        .emf = {1e-6, 1.0, 0.5, 10.0, 0.3, 0.0, false, true},
        .update = {startup.gamma, 1e-6, 1e-8, 10.0,
                   true, false, true},
        .fluid_flux_sync = gamera_cs_earth_inner_fluid_flux,
        .edge_emf_sync = gamera_cs_earth_tile_emf,
        .edge_emf_context = &emf_context,
        .source_time = 0.0};
    /* The accepted Earth driver enables Boris after constructing the
     * aggregate options; match that finalized configuration exactly. */
    options.stress.use_boris = true;
    options.emf.use_boris = true;
    options.update.use_boris = true;
    const gamera_no_timestep_options timestep = {
        startup.gamma, 1e-6, 1e-8, 0.3, 10.0,
        true, true, true};
    const size_t lower[3] = {ghost, ghost, ghost};
    const size_t upper[3] = {
        ghost + tile.local_active[0],
        ghost + tile.local_active[1],
        ghost + tile.local_active[2]};
    double time = 0.0, previous_dt = 0.0;
    double next_output_seconds = output_seconds;
    long drive_frame = 0;
    int completed_drive = 0;
    gamera_cs_profile driver_perf = {0};
    static const char *const driver_names[] = {"mi", "hdf5", "dt_scan", "mhd", "pressure_control", "bookkeeping"};
    for (size_t step = 0; step < steps; ++step) {
      cs_profile_begin(&driver_perf);
      if (use_mi && gamera_cs_mi_dipole_mpi_maybe_update(
              distributed_mi, &storage, time) < 0) {
        fprintf(stderr, "rank %d: tiled MI update failed step=%zu\n",
                rank, step);
        failed = 1;
        break;
      }
      cs_profile_mark(&driver_perf, 0);
      if (drive && step == 0) {
#ifdef GAMERA_CS_DRIVE_HDF5
        if (gamera_cs_analysis_hdf5_write(&analysis, drive_frame, 0,
                0.0, 0.0, &grid, &storage, &owned, ghost, startup.gamma,
                gamera_cs_mi_dipole_mpi_state(distributed_mi),
                MPI_COMM_WORLD) != 0) {
#else
        if (write_tiled_frame(argv[7], drive_frame, 0, 0.0,
                &grid, &storage, &owned, &tile, ghost, startup.gamma,
                distributed_mi, drive_index, MPI_COMM_WORLD) != 0) {
#endif
          failed = 1;
          break;
        }
        if (rank == 0) printf("earth-drive frame=%ld step=0 time=0 s\n",
                              drive_frame);
        ++drive_frame;
      }
      cs_profile_mark(&driver_perf, 1);
      double local_dt = INFINITY, global_dt = INFINITY;
      if (gamera_no_local_timestep(&grid, storage.conserved,
              storage.cell_magnetic, lower, upper, &timestep,
              owned.cell_magnetic, &local_dt) != 0) failed = 1;
      MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);
      if (!(global_dt > 0.0) || !isfinite(global_dt)) failed = 1;
      MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      if (any_failed) break;
      const double dt = step == 0 ?
          (steps == 1 ? fmin(1e-4, 0.1 * global_dt) :
                        0.5 * global_dt) :
          fmin(0.5 * global_dt, 1.25 * previous_dt);
      boundary.step_time = time;
      boundary.step_dt = dt;
      cs_profile_mark(&driver_perf, 2);
      if (gamera_cs_mpi_advance_tiled(&grid, &storage, &tile,
              ghost, step == 0 ? 0.0 : 0.5 * dt / previous_dt,
              dt, &options, &owned.field, cell_halo, magnetic_halo,
              face_plan, edge_plan,
              gamera_cs_earth_fill_radial_ghosts_tile,
              time, step == 0 ? time : time - previous_dt,
              &boundary, MPI_COMM_WORLD) != 0) {
        fprintf(stderr, "rank %d: tiled MHD update failed step=%zu\n",
                rank, step);
        failed = 1;
        break;
      }
      /* Controlled parity experiment: invoke the exact accepted Yin-Yang
       * Earth production post-step pressure control, without changing any
       * GAMERA reconstruction, flux, CT, or cube-interface formula. */
      cs_profile_mark(&driver_perf, 3);
      if (getenv("GAMERA_CS_TILE_PRODUCTION_PRESSURE_CONTROL") != NULL) {
        size_t controlled_low_density = 0U, controlled_high_sound = 0U;
        double maximum_sound = 0.0, maximum_pressure = 0.0;
        if (gamera_no_apply_pressure_control(
                &storage, &grid, lower, upper, dt, startup.gamma,
                1e-6, 1e-8, 10.0, 1.0e-3, 1.0e-2 / 100000.0,
                &controlled_low_density, &controlled_high_sound,
                &maximum_sound, &maximum_pressure) != 0) {
          fprintf(stderr, "rank %d: production pressure control failed "
                  "step=%zu\n", rank, step);
          failed = 1;
          break;
        }
        if (controlled_low_density != 0U || controlled_high_sound != 0U)
          printf("production pressure control rank=%d step=%zu "
                 "low=%zu high=%zu pre_cs=%.9g pre_p=%.9g\n",
                 rank, step, controlled_low_density, controlled_high_sound,
                 maximum_sound, maximum_pressure);
      }
      cs_profile_mark(&driver_perf, 4);
      time += dt;
      previous_dt = dt;
      if (drive && time * 63.71 >= next_output_seconds) {
#ifdef GAMERA_CS_DRIVE_HDF5
        if (gamera_cs_analysis_hdf5_write(&analysis, drive_frame,
                (long)step + 1, time, time * 63.71, &grid, &storage,
                &owned, ghost, startup.gamma,
                gamera_cs_mi_dipole_mpi_state(distributed_mi),
                MPI_COMM_WORLD) != 0) {
#else
        if (write_tiled_frame(argv[7], drive_frame, (long)step + 1,
                time * 63.71, &grid, &storage, &owned, &tile, ghost,
                startup.gamma, distributed_mi, drive_index,
                MPI_COMM_WORLD) != 0) {
#endif
          failed = 1;
          break;
        }
        if (rank == 0)
          printf("earth-drive frame=%ld step=%zu time=%.9g s\n",
                 drive_frame, step + 1, time * 63.71);
        ++drive_frame;
        do { next_output_seconds += output_seconds; }
        while (next_output_seconds <= time * 63.71);
      }
      cs_profile_mark(&driver_perf, 1);
      if (drive && rank == 0 && (step + 1) % 1000 == 0)
        printf("earth-local step=%zu time=%.9g dt=%.9g\n",
               step + 1, time, dt);
      cs_profile_mark(&driver_perf, 5);
      ++driver_perf.steps;
      if (drive && time * 63.71 >= stop_seconds) {
        completed_drive = 1;
        break;
      }
    }
    cs_profile_report(&driver_perf, "CS_TILE_DRIVER_PROFILE", rank, driver_names, 6);
    if (drive && !completed_drive) {
      if (rank == 0) fprintf(stderr,
          "tiled Earth drive stopped before %.9g s at %.9g s\n",
          stop_seconds, time * 63.71);
      failed = 1;
    }
    double minimum_density = INFINITY, minimum_pressure = INFINITY;
    double maximum_divergence = 0.0;
    for (size_t i = lower[0]; i < upper[0]; ++i)
      for (size_t j = lower[1]; j < upper[1]; ++j)
        for (size_t k = lower[2]; k < upper[2]; ++k) {
          const size_t cell = gamera_no_index3(grid.cell_extent, i, j, k);
          gamera_no_primitive primitive;
          if (gamera_no_conserved_to_primitive(
                  &storage.conserved[cell * GAMERA_NO_FLUX_COUNT],
                  startup.gamma, 1e-12, 1e-12, &primitive) != 0) {
            failed = 1;
            continue;
          }
          minimum_density = fmin(minimum_density, primitive.density);
          minimum_pressure = fmin(minimum_pressure, primitive.pressure);
          double net_flux = 0.0;
          for (int d = 0; d < 3; ++d) {
            size_t upper_face[3] = {i, j, k};
            ++upper_face[d];
            const size_t low = gamera_no_index3(
                grid.face[d].extent, i, j, k);
            const size_t high = gamera_no_index3(
                grid.face[d].extent, upper_face[0],
                upper_face[1], upper_face[2]);
            net_flux += storage.face_flux[d][high] +
                        owned.face_flux[d][high] -
                        storage.face_flux[d][low] -
                        owned.face_flux[d][low];
          }
          maximum_divergence = fmax(maximum_divergence,
                                    fabs(net_flux) / grid.cell[cell].volume);
        }
    double global_minimum_density, global_minimum_pressure;
    double global_maximum_divergence;
    MPI_Allreduce(&minimum_density, &global_minimum_density, 1,
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&minimum_pressure, &global_minimum_pressure, 1,
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maximum_divergence, &global_maximum_divergence, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!(global_minimum_density > 0.0) ||
        !(global_minimum_pressure > 0.0) ||
        !isfinite(global_maximum_divergence) ||
        global_maximum_divergence > 1e-7) {
      if (rank == 0)
        fprintf(stderr, "tiled Earth validation failed rho=%.9g "
                "p=%.9g divB=%.9g\n", global_minimum_density,
                global_minimum_pressure, global_maximum_divergence);
      failed = 1;
    }
    if (!failed && !drive && argc >= 8) {
      char path[4096];
      const int length = snprintf(path, sizeof(path), "%s.rank%03d.csv",
                                  argv[7], rank);
      FILE *output = length > 0 && (size_t)length < sizeof(path) ?
          fopen(path, "w") : NULL;
      if (output == NULL) failed = 1;
      if (!failed) {
        fputs("face,i,j,k,x,y,z,rho,p,vx,vy,vz,bx,by,bz\n", output);
        for (size_t i = lower[0]; i < upper[0]; ++i)
          for (size_t j = lower[1]; j < upper[1]; ++j)
            for (size_t k = lower[2]; k < upper[2]; ++k) {
              const size_t cell = gamera_no_index3(grid.cell_extent,
                                                   i, j, k);
              gamera_no_primitive state;
              if (gamera_no_conserved_to_primitive(
                      &storage.conserved[cell * GAMERA_NO_FLUX_COUNT],
                      startup.gamma, 1e-12, 1e-12, &state) != 0)
                failed = 1;
              const gamera_no_vec3 point = grid.cell[cell].centroid;
              const gamera_no_vec3 b0 = owned.cell_magnetic[cell];
              const gamera_no_vec3 b1 = storage.cell_magnetic[cell];
              if (fprintf(output,
                    "%d,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,"
                    "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    tile.face, tile.global_start[0] + i - ghost,
                    tile.global_start[1] + j - ghost,
                    tile.global_start[2] + k - ghost,
                    point.value[0], point.value[1], point.value[2],
                    state.density, state.pressure,
                    state.velocity.value[0], state.velocity.value[1],
                    state.velocity.value[2],
                    b0.value[0] + b1.value[0],
                    b0.value[1] + b1.value[1],
                    b0.value[2] + b1.value[2]) < 0) failed = 1;
            }
        if (fclose(output) != 0) failed = 1;
      }
    }
    if (!failed && !drive && use_mi && rank == 0 && argc >= 8 &&
        (gamera_cs_mi_dipole_write_csv(
             gamera_cs_mi_dipole_mpi_state(distributed_mi), argv[7]) != 0 ||
         gamera_cs_mi_dipole_write_shell_current_csv(
             gamera_cs_mi_dipole_mpi_state(distributed_mi), argv[7]) != 0))
      failed = 1;
    if (rank == 0 && !failed)
      printf("tiled Earth MHD PASS grid=%zu,%zu,%zu dims=%d,%d,%d "
             "steps=%zu time=%.9g divB=%.3e\n",
             global_active[0], global_active[1], global_active[2],
             proc_dims[0], proc_dims[1], proc_dims[2], steps,
             time, global_maximum_divergence);
  }
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (drive_index != NULL && fclose(drive_index) != 0) failed = 1;
#ifdef GAMERA_CS_DRIVE_HDF5
  gamera_cs_analysis_hdf5_close(&analysis);
#endif
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  gamera_cs_tiled_edge_sync_plan_destroy(edge_plan);
  gamera_cs_mi_dipole_mpi_destroy(distributed_mi);
  gamera_cs_tiled_cube_face_sync_plan_destroy(face_plan);
  gamera_cs_tiled_magnetic_halo_plan_destroy(magnetic_halo);
  gamera_cs_tiled_cell_halo_plan_destroy(cell_halo);
  gamera_solar_wind_destroy(&wind);
  gamera_no_background_destroy(&owned);
  gamera_no_storage_destroy(&storage);
  gamera_no_grid_destroy(&grid);
  MPI_Finalize();
  return any_failed ? 1 : 0;
}
