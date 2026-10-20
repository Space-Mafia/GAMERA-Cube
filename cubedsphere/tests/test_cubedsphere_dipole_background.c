#include "cubedsphere_step.h"
#include "cubedsphere_earth_startup.h"
#include "cubedsphere_earth_boundary.h"
#include "cubedsphere_mi_dipole.h"
#include "nonorthogonal_background.h"
#include "nonorthogonal_radial_map.h"
#include "nonorthogonal_state.h"
#include "nonorthogonal_step.h"
#ifdef GAMERA_CS_MPI_DRIVER
#include "cubedsphere_mpi_step.h"
#endif

#include <math.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Grid limits follow the Up7 stencil and allocation arithmetic, not a list
 * of named resolutions.  A decomposed driver applies the same rule per tile. */
static int cells_from_env(const char *name, size_t fallback,
                          size_t minimum, size_t *cells) {
  const char *value = getenv(name);
  if (cells == NULL) return -1;
  *cells = fallback;
  if (value == NULL) return 0;
  if (*value == '\0' || *value == '-' || *value == '+') return -1;
  char *end = NULL;
  errno = 0;
  const uintmax_t parsed = strtoumax(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0' || parsed < minimum ||
      parsed > SIZE_MAX - 9U) return -1;
  *cells = (size_t)parsed;
  return 0;
}

static int checked_vertex_count(const size_t active[3], size_t ghost,
                                size_t *count) {
  size_t result = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (ghost > (SIZE_MAX - 1U - active[axis]) / 2U) return -1;
    const size_t extent = active[axis] + 2U * ghost + 1U;
    if (result > SIZE_MAX / extent) return -1;
    result *= extent;
  }
  if (result > SIZE_MAX / sizeof(gamera_no_vec3)) return -1;
  *count = result;
  return 0;
}

#ifdef GAMERA_CS_MPI_DRIVER
#define ROOT_PRINTF(...) do { if (mpi_rank == 0) printf(__VA_ARGS__); } while (0)
#else
#define ROOT_PRINTF(...) printf(__VA_ARGS__)
#endif

/* Re-evaluate the MI diagnostic on a saved physical snapshot without
 * advancing the MHD solver. This keeps seam-current experiments controlled. */
static int load_mhd_snapshot(const char *path,
                             const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
                             gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
                             const gamera_no_background_data owned[GAMERA_CS_FACE_COUNT],
                             const size_t active[3], size_t ghost,
                             double gamma) {
  FILE *file = fopen(path, "r");
  if (file == NULL) return -1;
  char line[1024];
  if (fgets(line, sizeof(line), file) == NULL ||
      strcmp(line, "face,i,j,k,x,y,z,rho,p,vx,vy,vz,bx,by,bz\n") != 0) {
    fclose(file);
    return -1;
  }
  size_t count = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
    int face;
    size_t i, j, k;
    double xyz[3], b[3], rho, pressure, v[3];
    if (sscanf(line, "%d,%zu,%zu,%zu,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
               &face, &i, &j, &k, &xyz[0], &xyz[1], &xyz[2],
               &rho, &pressure, &v[0], &v[1], &v[2],
               &b[0], &b[1], &b[2]) != 15 ||
        face < 0 || face >= GAMERA_CS_FACE_COUNT ||
        i >= active[0] || j >= active[1] || k >= active[2]) {
      fclose(file);
      return -1;
    }
    const size_t cell = gamera_no_index3(grid[face].cell_extent,
                                         i + ghost, j + ghost, k + ghost);
    for (int axis = 0; axis < 3; ++axis)
      if (fabs(grid[face].cell[cell].centroid.value[axis] - xyz[axis]) > 1e-8) {
        fclose(file);
        return -1;
      }
    const gamera_no_primitive state = {
        rho, {{v[0], v[1], v[2]}}, pressure};
    double conserved[GAMERA_NO_FLUX_COUNT];
    if (gamera_no_primitive_to_conserved(&state, gamma, 1e-12, 1e-12,
                                         conserved) != 0) {
      fclose(file);
      return -1;
    }
    memcpy(&storage[face].conserved[cell * GAMERA_NO_FLUX_COUNT],
           conserved, sizeof(conserved));
    memcpy(&storage[face].old_conserved[cell * GAMERA_NO_FLUX_COUNT],
           conserved, sizeof(conserved));
    for (int axis = 0; axis < 3; ++axis) {
      const double residual = b[axis] - owned[face].cell_magnetic[cell].value[axis];
      storage[face].cell_magnetic[cell].value[axis] = residual;
      storage[face].old_cell_magnetic[cell].value[axis] = residual;
    }
    ++count;
  }
  const int close_result = fclose(file);
  return close_result == 0 && count == GAMERA_CS_FACE_COUNT * active[0] *
      active[1] * active[2] ? 0 : -1;
}

static int write_earth_snapshot(
    const char *path, const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    const gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const gamera_no_background_data owned[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, double gamma,
    const gamera_cs_mi_dipole *mi) {
  FILE *output = fopen(path, "w");
  if (output == NULL) return -1;
  fputs("face,i,j,k,x,y,z,rho,p,vx,vy,vz,bx,by,bz\n", output);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
    for (size_t i = ghost; i < ghost + active[0]; ++i)
      for (size_t j = ghost; j < ghost + active[1]; ++j)
        for (size_t k = ghost; k < ghost + active[2]; ++k) {
          const size_t cell = gamera_no_index3(grid[face].cell_extent,
                                               i, j, k);
          gamera_no_primitive state;
          if (gamera_no_conserved_to_primitive(
                  &storage[face].conserved[cell * GAMERA_NO_FLUX_COUNT],
                  gamma, 1e-12, 1e-12, &state) != 0) {
            fclose(output);
            return -1;
          }
          const gamera_no_vec3 point = grid[face].cell[cell].centroid;
          const gamera_no_vec3 b1 = storage[face].cell_magnetic[cell];
          const gamera_no_vec3 b0 = owned[face].cell_magnetic[cell];
          fprintf(output,
                  "%d,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,"
                  "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                  face, i - ghost, j - ghost, k - ghost,
                  point.value[0], point.value[1], point.value[2],
                  state.density, state.pressure,
                  state.velocity.value[0], state.velocity.value[1],
                  state.velocity.value[2],
                  b0.value[0] + b1.value[0],
                  b0.value[1] + b1.value[1],
                  b0.value[2] + b1.value[2]);
        }
  if (fclose(output) != 0) return -1;
  return mi == NULL ? 0 :
      (gamera_cs_mi_dipole_write_csv(mi, path) == 0 &&
       gamera_cs_mi_dipole_write_shell_current_csv(mi, path) == 0 ? 0 : -1);
}

static int positive_seconds_from_env(const char *name, double fallback,
                                     double *result) {
  const char *value = getenv(name);
  if (value == NULL) {
    *result = fallback;
    return 0;
  }
  char *end = NULL;
  const double parsed = strtod(value, &end);
  if (end == value || *end != '\0' || !isfinite(parsed) || parsed <= 0.0)
    return -1;
  *result = parsed;
  return 0;
}

static int write_drive_frame(
    const char *prefix, long frame, long step, double seconds,
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    const gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const gamera_no_background_data owned[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost, double gamma,
    const gamera_cs_mi_dipole *mi, FILE *index) {
  char path[4096];
  const int length = snprintf(path, sizeof(path), "%s_%06ld.csv",
                              prefix, frame);
  if (length < 0 || (size_t)length >= sizeof(path) || index == NULL ||
      write_earth_snapshot(path, grid, storage, owned, active, ghost,
                           gamma, mi) != 0 ||
      fprintf(index, "%ld,%ld,%.17g,%s\n", frame, step, seconds, path) < 0 ||
      fflush(index) != 0) return -1;
  return 0;
}

static int all_ranks_succeeded(int local_ok) {
#ifdef GAMERA_CS_MPI_DRIVER
  int global_ok = 0;
  return MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && global_ok;
#else
  return local_ok;
#endif
}

int main(int argc, char **argv) {
#ifdef GAMERA_CS_MPI_DRIVER
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
  int mpi_rank = -1, mpi_ranks = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks) != MPI_SUCCESS ||
      mpi_ranks != GAMERA_CS_FACE_COUNT) {
    if (mpi_rank == 0) fprintf(stderr, "requires exactly six MPI ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
#endif
  setvbuf(stdout, NULL, _IOLBF, 0);
  const int replay = argc >= 2 &&
                     strcmp(argv[1], "earth-mi-replay") == 0;
  const int drive = argc >= 2 &&
                    strcmp(argv[1], "earth-mi-drive") == 0;
  const int earth_mi_local = argc >= 2 &&
                             (strcmp(argv[1], "earth-mi-local") == 0 ||
                              replay || drive);
  const int earth_local = earth_mi_local ||
      (argc >= 2 && strcmp(argv[1], "earth-local") == 0);
  char *end = NULL;
  const long requested_steps = replay ? 0L :
      argc >= 3 ? strtol(argv[2], &end, 10) : 1L;
  if ((replay && argc != 4) || (!replay && argc > 4) ||
      (argc >= 2 && !earth_local) ||
      (drive && argc != 4) ||
      (!replay && argc >= 3 && (end == argv[2] || *end != '\0' ||
                     requested_steps < 1 || requested_steps > 1000000)) ||
      (argc == 4 && !earth_local)) {
    fprintf(stderr, "usage: %s [earth-local|earth-mi-local [steps [output.csv]]]"
                    " | earth-mi-replay snapshot.csv output-prefix"
                    " | earth-mi-drive max-steps output-prefix\n", argv[0]);
    return 1;
  }
  double first_switch_seconds = 3600.0, second_switch_seconds = 10800.0;
  double stop_seconds = 21600.0, output_seconds = 600.0;
  if (drive &&
      (positive_seconds_from_env("GAMERA_CS_EARTH_FIRST_SWITCH_SECONDS",
                                 3600.0, &first_switch_seconds) != 0 ||
       positive_seconds_from_env("GAMERA_CS_EARTH_SECOND_SWITCH_SECONDS",
                                 10800.0, &second_switch_seconds) != 0 ||
       positive_seconds_from_env("GAMERA_CS_EARTH_STOP_SECONDS",
                                 21600.0, &stop_seconds) != 0 ||
       positive_seconds_from_env("GAMERA_CS_EARTH_OUTPUT_SECONDS",
                                 600.0, &output_seconds) != 0 ||
       !(first_switch_seconds < second_switch_seconds &&
         second_switch_seconds < stop_seconds))) return 1;
#ifdef GAMERA_CS_MPI_DRIVER
  const int writer_rank = mpi_rank == 0;
#else
  const int writer_rank = 1;
#endif
  size_t radial_cells = 72U;
  size_t angular_cells = 16U;
  if (earth_local &&
      (cells_from_env("GAMERA_CS_EARTH_RADIAL_CELLS", radial_cells, 8U,
                      &radial_cells) != 0 ||
       cells_from_env("GAMERA_CS_EARTH_ANGULAR_CELLS", angular_cells, 8U,
                      &angular_cells) != 0)) {
    fprintf(stderr, "Earth cell counts must be integers >= 8\n");
    return 1;
  }
  const size_t ghost = 4;
  const size_t active[3] = {earth_local ? radial_cells : 3U,
                            earth_local ? angular_cells : 9U,
                            earth_local ? angular_cells : 9U};
  size_t vertex_count = 0;
  if (checked_vertex_count(active, ghost, &vertex_count) != 0) return 1;
  const size_t extent[3] = {active[0] + 2 * ghost,
                            active[1] + 2 * ghost,
                            active[2] + 2 * ghost};
  const size_t vertex_extent[3] = {extent[0] + 1, extent[1] + 1,
                                   extent[2] + 1};
  const size_t lower[3] = {ghost, ghost, ghost};
  const size_t upper[3] = {ghost + active[0], ghost + active[1],
                           ghost + active[2]};
  const gamera_no_dipole dipole = {{{0.0, 0.0, 10.0}}};
  gamera_cs_earth_startup startup = {
      .inner_radius = 2.5,
      .density = 5.0, .speed = 4.0, .mach = 10.0,
      .gamma = 5.0 / 3.0,
      .ambient_density = 1.0, .ambient_pressure = 0.001,
      .front_x = -30.0, .front_width = 1.5,
      .cut_radius = 16.0, .cut_length = 8.0,
      .imf = {{0.0, 0.0, 0.0}},
      /* 0.31 G equatorial surface dipole in the accepted Earth units:
       * x_Norm=1 RE, u_Norm=100 km/s, rho_Norm=1 cm^-3 proton mass. */
      .dipole = {{{0.0, 0.0, -6761.730688424097}}}};
  const char *front_text = getenv("GAMERA_CS_EARTH_FRONT_X");
  if (earth_local && front_text != NULL) {
    char *front_end = NULL;
    const double front = strtod(front_text, &front_end);
    if (front_end == front_text || *front_end != '\0' ||
        !isfinite(front) || front < -100.0 || front > -10.0) return 1;
    startup.front_x = front;
  }
  const gamera_no_primitive initial = {1.0, {{0.0, 0.0, 0.0}}, 1.0};
  const double gamma = 5.0 / 3.0;
  double initial_conserved[GAMERA_NO_FLUX_COUNT];
  gamera_no_grid grid[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_storage storage[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_background_data owned[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_background_field background[GAMERA_CS_FACE_COUNT] = {{0}};
  gamera_no_radial_map radial_map;
  size_t inner_uniform_cells = 0;
  const char *inner_uniform_text =
      getenv("GAMERA_CS_EARTH_INNER_UNIFORM_CELLS");
  if (earth_local && inner_uniform_text != NULL &&
      (cells_from_env("GAMERA_CS_EARTH_INNER_UNIFORM_CELLS", 0U, 1U,
                      &inner_uniform_cells) != 0 ||
       inner_uniform_cells >= radial_cells)) return 1;
  if (earth_local &&
      (inner_uniform_text == NULL
           ? gamera_no_radial_map_init(
                 &radial_map, GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
                 2.5, 200.0, 3.0)
           : gamera_no_radial_map_init_earth_v4_resolution(
                 &radial_map, 2.5, 200.0,
                 radial_cells, inner_uniform_cells)) != 0) return 1;
  if (gamera_no_primitive_to_conserved(&initial, gamma, 1e-12, 1e-12,
                                       initial_conserved) != 0) return 1;

  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_vec3 *vertices = calloc(vertex_count, sizeof(*vertices));
    if (vertices == NULL ||
        (earth_local
             ? gamera_cs_generate_vertices_radial_map(
                   face, active, ghost, &radial_map, vertices)
             : gamera_cs_generate_vertices(face, active, ghost, 3.0, 4.0,
                                           vertices)) != 0 ||
        gamera_no_grid_create(extent, vertices, &grid[face]) != 0 ||
        gamera_cs_calibrate_face_frames(&grid[face]) != 0 ||
        gamera_no_storage_create(extent, &storage[face]) != 0 ||
        gamera_no_background_create(
            &grid[face], earth_local ? gamera_cs_earth_startup_background
                                     : gamera_no_dipole_field,
            earth_local ? (void *)&startup : (void *)&dipole,
            &owned[face]) != 0) {
      fprintf(stderr, "dipole setup failed on face %d\n", face);
      free(vertices);
      return 1;
    }
    free(vertices);
    background[face] = owned[face].field;
    if (earth_local) {
      if (gamera_no_initialize_primitives(
              &grid[face], &storage[face],
              gamera_cs_earth_startup_primitive, (void *)&startup,
              false, gamma, 1e-6, 1e-8) != 0 ||
          gamera_no_initialize_flux_from_vector_potential(
              &grid[face], &storage[face],
              gamera_cs_earth_startup_potential, (void *)&startup) != 0 ||
          gamera_no_save_current_as_old(&storage[face]) != 0 ||
          gamera_cs_earth_background_adjust(
              &grid[face], &owned[face], &startup) != 0 ||
          gamera_cs_earth_total_to_residual(
              &grid[face], &storage[face], &owned[face]) != 0) {
        fprintf(stderr, "Earth startup initialization failed on face %d\n",
                face);
        return 1;
      }
    } else {
      const size_t cells = gamera_no_element_count3(extent);
      for (size_t cell = 0; cell < cells; ++cell) {
        memcpy(&storage[face].conserved[cell * GAMERA_NO_FLUX_COUNT],
               initial_conserved, sizeof(initial_conserved));
        memcpy(&storage[face].old_conserved[cell * GAMERA_NO_FLUX_COUNT],
               initial_conserved, sizeof(initial_conserved));
      }
    }
  }

  if (earth_local) {
    double maximum_residual = 0.0;
    double maximum_residual_fraction = 0.0;
    for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
      for (size_t i = lower[0]; i < lower[0] + 8U; ++i)
        for (size_t j = lower[1]; j < upper[1]; ++j)
          for (size_t k = lower[2]; k < upper[2]; ++k) {
            const size_t cell = gamera_no_index3(extent, i, j, k);
            double b0_squared = 0.0, b1_squared = 0.0;
            for (int c = 0; c < GAMERA_NO_DIM; ++c) {
              const double b0 = owned[face].cell_magnetic[cell].value[c];
              const double b1 = storage[face].cell_magnetic[cell].value[c];
              b0_squared += b0 * b0;
              b1_squared += b1 * b1;
            }
            maximum_residual = fmax(maximum_residual, sqrt(b1_squared));
            maximum_residual_fraction = fmax(
                maximum_residual_fraction,
                sqrt(b1_squared / fmax(b0_squared, 1e-30)));
          }
    ROOT_PRINTF("Earth startup inner-eight residual |B1|max=%.9g, "
           "max |B1|/|B0|=%.9g\n",
           maximum_residual, maximum_residual_fraction);
  }

  const gamera_no_timestep_options normal = {
      gamma, 1e-6, 1e-8, 0.3, 10.0, true, false, true};
  gamera_no_timestep_options boris = normal;
  boris.use_boris = true;
  double smallest_normal = INFINITY, smallest_boris = INFINITY;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    double dt_normal, dt_boris;
    if (gamera_no_local_timestep(&grid[face], storage[face].conserved,
                                storage[face].cell_magnetic, lower, upper,
                                &normal, owned[face].cell_magnetic,
                                &dt_normal) != 0 ||
        gamera_no_local_timestep(&grid[face], storage[face].conserved,
                                storage[face].cell_magnetic, lower, upper,
                                &boris, owned[face].cell_magnetic,
                                &dt_boris) != 0) return 1;
    smallest_normal = fmin(smallest_normal, dt_normal);
    smallest_boris = fmin(smallest_boris, dt_boris);
  }
  if (!(smallest_normal > 0.0) ||
      !(smallest_boris > smallest_normal) ||
      !isfinite(smallest_boris)) {
    fprintf(stderr, "background/Boris CFL failed: %.9g %.9g\n",
            smallest_normal, smallest_boris);
    return 1;
  }
  double dt_fraction = 0.5;
  const char *fraction_text = getenv("GAMERA_CS_EARTH_DT_FRACTION");
  if (earth_local && fraction_text != NULL) {
    char *fraction_end = NULL;
    dt_fraction = strtod(fraction_text, &fraction_end);
    if (fraction_end == fraction_text || *fraction_end != '\0' ||
        !isfinite(dt_fraction) || dt_fraction <= 0.0 ||
        dt_fraction > 1.0) return 1;
  }

  gamera_no_advance_options options = {
      .stress = {gamma, 1e-6, 1e-8, 1.0, 0.25, 0.25, 10.0,
                 true, true, false, true},
      .emf = {1e-6, 1.0, 0.5, 10.0, 0.3, 0.0, false, true},
      .update = {gamma, 1e-6, 1e-8, 10.0, true, false, true}};
  const int use_boris = !(earth_local &&
                          getenv("GAMERA_CS_EARTH_NO_BORIS") != NULL);
  options.stress.use_boris = use_boris;
  options.emf.use_boris = use_boris;
  options.update.use_boris = use_boris;
  gamera_solar_wind_series wind;
  gamera_solar_wind_init(&wind);
  gamera_cs_earth_boundary boundary = {
      .startup = &startup, .wind = &wind, .background = background,
      .density_floor = 1e-6, .pressure_floor = 1e-8, .cfl = 0.3};
  boundary.strict_inner_gas_wall =
      getenv("GAMERA_CS_EARTH_STRICT_INNER_GAS_WALL") != NULL;
  if (earth_local) {
    const double wind_time = 0.0;
    const gamera_solar_wind_state wind_state = {
        .density = startup.density,
        .pressure = startup.density * startup.speed * startup.speed /
                    (startup.gamma * startup.mach * startup.mach),
        .velocity = {startup.speed, 0.0, 0.0},
        .magnetic = {startup.imf.value[0], startup.imf.value[1],
                     startup.imf.value[2]}};
    if (drive) {
      /* Physical units are the accepted Earth normalization: 1 RE,
       * 100 km/s, 1 proton cm^-3; one code time is 63.71 s. */
      const double rho_norm = 1.67262192369e-21;
      const double magnetic_norm = 100000.0 *
          sqrt(4.0e-7 * acos(-1.0) * rho_norm);
      const double bz5_code = 5.0e-9 / magnetic_norm;
      const double drive_times[3] = {
          0.0, first_switch_seconds / 63.71,
          second_switch_seconds / 63.71};
      gamera_solar_wind_state drive_states[3] = {
          wind_state, wind_state, wind_state};
      drive_states[1].magnetic[GAMERA_WIND_Z] = bz5_code;
      drive_states[2].magnetic[GAMERA_WIND_Z] = -bz5_code;
      wind.linear_interpolation = 0;
      if (gamera_solar_wind_set(&wind, 3, drive_times,
                                drive_states) != 0) return 1;
      ROOT_PRINTF("Earth drive at upstream reference x=%.6g RE: "
                  "Bz=0 until %.6g s, +5 nT until %.6g s, "
                  "then -5 nT until %.6g s; 1 nT=%.9g code B\n",
                  startup.front_x, first_switch_seconds,
                  second_switch_seconds, stop_seconds,
                  bz5_code / 5.0);
    } else if (gamera_solar_wind_set(&wind, 1, &wind_time,
                                      &wind_state) != 0) return 1;
    wind.reference[0] = startup.front_x;
    if (drive) {
      const double reference[3] = {startup.front_x, 0.0, 0.0};
      const double probe_time[3] = {
          0.0, first_switch_seconds / 63.71,
          second_switch_seconds / 63.71};
      for (int phase = 0; phase < 3; ++phase) {
        gamera_solar_wind_state sampled;
        if (gamera_solar_wind_sample_at(
                &wind, reference, probe_time[phase], &sampled,
                NULL) != 0 ||
            sampled.magnetic[GAMERA_WIND_Z] !=
                wind.state[phase].magnetic[GAMERA_WIND_Z]) return 1;
      }
    }
    options.fluid_flux_sync = gamera_cs_earth_inner_fluid_flux;
    options.edge_emf_sync = gamera_cs_earth_no_mi_emf;
    options.edge_emf_context = &boundary;
  }
  gamera_cs_mi_dipole *mi = NULL;
  if (earth_mi_local) {
    const gamera_cs_mi_dipole_config mi_config = {
        .azimuth_points = 192U, .colatitude_points = 22U,
        .current_shell = 1U, .ionosphere_radius_re = 1.0,
        .pedersen_siemens = 5.0, .coupling_interval_s = 10.0,
        .length_norm_m = 6371000.0, .speed_norm_m_s = 100000.0,
        .mass_density_norm_kg_m3 = 1.67262192369e-21,
        .maximum_iterations = 4000, .relative_tolerance = 1e-8};
    if (gamera_cs_mi_dipole_create(grid, storage, background, &startup,
                                   active, ghost, &mi_config, &mi) != 0)
      return 1;
    boundary.mi = mi;
    ROOT_PRINTF("Earth aligned-dipole MI: 192x22 per hemisphere, "
           "SigmaP=5 S, SigmaH=0, FAC shell=2, cadence=10 s\n");
  }
  if (replay) {
    if (load_mhd_snapshot(argv[2], grid, storage, owned, active, ghost,
                          gamma) != 0 ||
        gamera_cs_mi_dipole_maybe_update(mi, 0.0) != 1 ||
        gamera_cs_mi_dipole_write_csv(mi, argv[3]) != 0 ||
        gamera_cs_mi_dipole_write_shell_current_csv(mi, argv[3]) != 0) {
      fprintf(stderr, "MI snapshot replay failed\n");
      return 1;
    }
    gamera_cs_mi_dipole_destroy(mi);
    for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
      gamera_no_background_destroy(&owned[face]);
      gamera_no_storage_destroy(&storage[face]);
      gamera_no_grid_destroy(&grid[face]);
    }
    return 0;
  }
  gamera_cs_halo_plan *halo_plan = NULL;
  if (earth_local && getenv("GAMERA_CS_EARTH_DISABLE_HALO_PLAN") == NULL &&
      gamera_cs_halo_plan_create(grid, active, ghost, &halo_plan) != 0)
    return 1;
#ifdef GAMERA_CS_MPI_DRIVER
  gamera_cs_edge_sync_plan *edge_plan = NULL;
  if (earth_local &&
      gamera_cs_edge_sync_plan_create(
          grid, active, ghost, &edge_plan) != 0) return 1;
#endif
  FILE *drive_index = NULL;
  if (drive && writer_rank) {
    char index_path[4096];
    const int length = snprintf(index_path, sizeof(index_path),
                                "%s.index.csv", argv[3]);
    if (length < 0 || (size_t)length >= sizeof(index_path)) return 1;
    drive_index = fopen(index_path, "w");
    if (drive_index != NULL) {
      fputs("frame,step,simulation_seconds,mhd_csv\n", drive_index);
      fflush(drive_index);
    }
  }
  int index_ok = !drive || !writer_rank || drive_index != NULL;
#ifdef GAMERA_CS_MPI_DRIVER
  int all_index_ok = 0;
  if (MPI_Allreduce(&index_ok, &all_index_ok, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
  index_ok = all_index_ok;
#endif
  if (!index_ok) return 1;
  double time = 0.0, previous_dt = 0.0;
  double next_output_seconds = output_seconds;
  double last_frame_seconds = 0.0;
  int completed_drive = 0;
  long completed_steps = 0;
  long drive_frame = 0;
  double dt = requested_steps == 1 ?
      fmin(1e-4, 0.1 * (use_boris ? smallest_boris : smallest_normal)) :
      dt_fraction * (use_boris ? smallest_boris : smallest_normal);
#ifdef GAMERA_CS_MPI_DRIVER
  const int profile_step = getenv("GAMERA_CS_PROFILE_STEP") != NULL;
  const int sparse_active = drive && earth_local && halo_plan != NULL &&
      getenv("GAMERA_CS_SPARSE_ACTIVE") != NULL;
  int full_active_state = 1;
  double profile_dt_scan = 0.0, profile_mi = 0.0;
#endif
  for (long step = 0; step < requested_steps; ++step) {
#ifdef GAMERA_CS_MPI_DRIVER
    const double dt_scan_start = profile_step ? MPI_Wtime() : 0.0;
#endif
    if (step > 0) {
      double limit = INFINITY;
      for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
#ifdef GAMERA_CS_MPI_DRIVER
        if (sparse_active && face != mpi_rank) continue;
#endif
        double local;
        if (gamera_no_local_timestep(&grid[face], storage[face].conserved,
                                    storage[face].cell_magnetic, lower, upper,
                                    use_boris ? &boris : &normal,
                                    owned[face].cell_magnetic,
                                    &local) != 0) return 1;
        limit = fmin(limit, local);
      }
#ifdef GAMERA_CS_MPI_DRIVER
      if (sparse_active) {
        double global_limit = INFINITY;
        if (MPI_Allreduce(&limit, &global_limit, 1, MPI_DOUBLE, MPI_MIN,
                          MPI_COMM_WORLD) != MPI_SUCCESS) return 1;
        limit = global_limit;
      }
#endif
      dt = fmin(dt_fraction * limit, 1.25 * previous_dt);
    }
#ifdef GAMERA_CS_MPI_DRIVER
    if (profile_step) profile_dt_scan += MPI_Wtime() - dt_scan_start;
    const double mi_start = profile_step ? MPI_Wtime() : 0.0;
#endif
    if (earth_mi_local) {
#ifdef GAMERA_CS_MPI_DRIVER
      if (sparse_active && !full_active_state) {
        const int due = gamera_cs_mi_dipole_update_due(mi, time);
        if (due < 0) return 1;
        if (due && gamera_cs_mpi_exchange_active_cells(
                storage, active, ghost, MPI_COMM_WORLD) != 0) return 1;
        if (due) full_active_state = 1;
      }
#endif
      const int updated = gamera_cs_mi_dipole_maybe_update(mi, time);
      if (updated < 0) {
        fprintf(stderr, "six-face dipole MI update failed at step %ld "
                "t=%.9g\n", step + 1, time);
        return 1;
      }
      if (updated > 0) {
        double fac_min, fac_max, potential_min, potential_max;
        gamera_mi_solver_stats solve[2];
        if (gamera_cs_mi_dipole_stats(mi, &fac_min, &fac_max,
                &potential_min, &potential_max, solve) != 0) return 1;
        if (step == 0 &&
            (fmax(fabs(fac_min), fabs(fac_max)) > 1e-12 ||
             fmax(fabs(potential_min), fabs(potential_max)) > 1e-2)) {
          fprintf(stderr, "aligned-dipole MI zero-FAC baseline failed\n");
          return 1;
        }
        ROOT_PRINTF("earth-mi update time=%.9g s FAC=[%.9g,%.9g] A/m2 "
               "potential=[%.9g,%.9g] kV iterations=(%d,%d)\n",
               time * 63.71, fac_min, fac_max,
               potential_min * 1e-3, potential_max * 1e-3,
               solve[0].iterations, solve[1].iterations);
      }
    }
#ifdef GAMERA_CS_MPI_DRIVER
    if (profile_step) profile_mi += MPI_Wtime() - mi_start;
#endif
    if (drive && step == 0) {
      const int frame_ok = !writer_rank ||
          write_drive_frame(argv[3], drive_frame, 0, 0.0,
                            grid, storage, owned, active, ghost,
                            gamma, mi, drive_index) == 0;
      if (!all_ranks_succeeded(frame_ok)) return 1;
      ROOT_PRINTF("earth-drive frame=%ld step=0 time=0 s\n", drive_frame);
      ++drive_frame;
    }
    boundary.step_dt = dt;
    boundary.step_time = time;
    const double predictor_ratio =
        step == 0 ? 0.0 : 0.5 * dt / previous_dt;
#ifdef GAMERA_CS_MPI_DRIVER
    /* Both exchange paths leave the exact halo donors current. The sparse
     * path gathers all faces only when global MI/output diagnostics need it. */
    const int step_status =
        (sparse_active && step > 0 ?
         gamera_cs_mpi_advance_six_with_background_sparse_pre_synced :
         step == 0 ?
         gamera_cs_mpi_advance_six_with_background_planned_cached_edges :
         gamera_cs_mpi_advance_six_with_background_planned_cached_edges_pre_synced)(
            grid, storage, active, ghost, predictor_ratio, dt,
            &options, background, halo_plan, edge_plan,
            earth_local ? gamera_cs_earth_fill_radial_ghosts : NULL,
            time, step == 0 ? time : time - previous_dt,
            earth_local ? &boundary : NULL, MPI_COMM_WORLD);
    if (step_status != 0) {
#else
    if (gamera_cs_advance_six_with_background_planned(
            grid, storage, active, ghost, predictor_ratio, dt,
            &options, background, halo_plan,
            earth_local ? gamera_cs_earth_fill_radial_ghosts : NULL,
            time, step == 0 ? time : time - previous_dt,
            earth_local ? &boundary : NULL) != 0) {
#endif
      fprintf(stderr, "six-face dipole/Boris MHD step %ld failed at t=%.9g\n",
              step + 1, time);
      return 1;
    }
    time += dt;
    previous_dt = dt;
    completed_steps = step + 1;
#ifdef GAMERA_CS_MPI_DRIVER
    if (sparse_active) full_active_state = step == 0;
#endif
#ifdef GAMERA_CS_MPI_DRIVER
    if (profile_step && completed_steps % 1000 == 0)
      fprintf(stderr,
          "CS_DRIVER_PROFILE rank=%d steps=%ld dt_scan=%.6f mi=%.6f\n",
          mpi_rank, completed_steps, profile_dt_scan, profile_mi);
#endif
    if (drive && time * 63.71 >= next_output_seconds) {
#ifdef GAMERA_CS_MPI_DRIVER
      if (sparse_active && !full_active_state) {
        if (gamera_cs_mpi_exchange_active_cells(
                storage, active, ghost, MPI_COMM_WORLD) != 0) return 1;
        full_active_state = 1;
      }
#endif
      const int frame_ok = !writer_rank ||
          write_drive_frame(argv[3], drive_frame, step + 1, time * 63.71,
                            grid, storage, owned, active, ghost,
                            gamma, mi, drive_index) == 0;
      if (!all_ranks_succeeded(frame_ok)) return 1;
      ROOT_PRINTF("earth-drive frame=%ld step=%ld time=%.9g s\n",
                  drive_frame, step + 1, time * 63.71);
      last_frame_seconds = time * 63.71;
      ++drive_frame;
      do { next_output_seconds += output_seconds; }
      while (next_output_seconds <= time * 63.71);
    }
    if (earth_local && (step < 20 ||
                        (step + 1) % (drive ? 1000 : 10) == 0)) {
#ifdef GAMERA_CS_MPI_DRIVER
      if (sparse_active && !full_active_state) {
        if (gamera_cs_mpi_exchange_active_cells(
                storage, active, ghost, MPI_COMM_WORLD) != 0) return 1;
        full_active_state = 1;
      }
#endif
      double p_min = INFINITY, p_max = 0.0, v_max = 0.0, b1_max = 0.0;
      int p_max_face = -1;
      int b1_max_face = -1;
      size_t p_max_i = 0, p_max_j = 0, p_max_k = 0;
      size_t b1_max_i = 0, b1_max_j = 0, b1_max_k = 0;
      for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face)
        for (size_t i = lower[0]; i < lower[0] + 8U; ++i)
          for (size_t j = lower[1]; j < upper[1]; ++j)
            for (size_t k = lower[2]; k < upper[2]; ++k) {
              const size_t cell = gamera_no_index3(extent, i, j, k);
              gamera_no_primitive state;
              if (gamera_no_conserved_to_primitive(
                      &storage[face].conserved[cell * GAMERA_NO_FLUX_COUNT],
                      gamma, 1e-6, 1e-8, &state) != 0) return 1;
              p_min = fmin(p_min, state.pressure);
              if (state.pressure > p_max) {
                p_max = state.pressure;
                p_max_face = face;
                p_max_i = i - ghost;
                p_max_j = j - ghost;
                p_max_k = k - ghost;
              }
              double v2 = 0.0, b12 = 0.0;
              for (int c = 0; c < GAMERA_NO_DIM; ++c) {
                v2 += state.velocity.value[c] * state.velocity.value[c];
                const double b1 = storage[face].cell_magnetic[cell].value[c];
                b12 += b1 * b1;
              }
              v_max = fmax(v_max, sqrt(v2));
              if (sqrt(b12) > b1_max) {
                b1_max = sqrt(b12);
                b1_max_face = face;
                b1_max_i = i - ghost;
                b1_max_j = j - ghost;
                b1_max_k = k - ghost;
              }
            }
      ROOT_PRINTF("inner shell step=%ld p=[%.9g,%.9g] vmax=%.9g "
             "B1max=%.9g pmax_location=(%d,%zu,%zu,%zu) "
             "B1max_location=(%d,%zu,%zu,%zu)\n",
             step + 1, p_min, p_max, v_max, b1_max, p_max_face,
             p_max_i, p_max_j, p_max_k, b1_max_face,
             b1_max_i, b1_max_j, b1_max_k);
    }
    if (requested_steps > 1 &&
        ((step + 1) % (drive ? 1000 : 10) == 0 ||
         step + 1 == requested_steps))
      ROOT_PRINTF("earth-local step=%ld time=%.9g dt=%.9g\n",
             step + 1, time, dt);
    if (drive && time * 63.71 >= stop_seconds) {
      completed_drive = 1;
      break;
    }
  }
  if (drive && time * 63.71 > last_frame_seconds + 1e-9) {
#ifdef GAMERA_CS_MPI_DRIVER
    if (sparse_active && !full_active_state &&
        gamera_cs_mpi_exchange_active_cells(
            storage, active, ghost, MPI_COMM_WORLD) != 0) return 1;
#endif
    const int frame_ok = !writer_rank ||
        write_drive_frame(argv[3], drive_frame, completed_steps,
                          time * 63.71,
                          grid, storage, owned, active, ghost,
                          gamma, mi, drive_index) == 0;
    if (!all_ranks_succeeded(frame_ok)) return 1;
    ROOT_PRINTF("earth-drive final frame=%ld time=%.9g s\n",
                drive_frame, time * 63.71);
  }
  double minimum_density = INFINITY, minimum_pressure = INFINITY;
  double maximum_total_divergence = 0.0;
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
#ifdef GAMERA_CS_MPI_DRIVER
    /* Non-owned active states are replicated for exact donor interpolation,
     * but their internal face fluxes are not. Diagnose only owned CT faces. */
    if (face != mpi_rank) continue;
#endif
    for (size_t i = lower[0]; i < upper[0]; ++i) {
      for (size_t j = lower[1]; j < upper[1]; ++j) {
        for (size_t k = lower[2]; k < upper[2]; ++k) {
          const size_t cell = gamera_no_index3(extent, i, j, k);
          gamera_no_primitive state;
          if (gamera_no_conserved_to_primitive(
                  &storage[face].conserved[cell * GAMERA_NO_FLUX_COUNT],
                  gamma, 1e-12, 1e-12, &state) != 0) return 1;
          minimum_density = fmin(minimum_density, state.density);
          minimum_pressure = fmin(minimum_pressure, state.pressure);
          double total_net_flux = 0.0;
          for (int direction = 0; direction < GAMERA_NO_DIM; ++direction) {
            size_t high[3] = {i, j, k};
            ++high[direction];
            const size_t low_face = gamera_no_index3(
                grid[face].face[direction].extent, i, j, k);
            const size_t high_face = gamera_no_index3(
                grid[face].face[direction].extent,
                high[0], high[1], high[2]);
            total_net_flux += storage[face].face_flux[direction][high_face] +
                              owned[face].face_flux[direction][high_face] -
                              storage[face].face_flux[direction][low_face] -
                              owned[face].face_flux[direction][low_face];
          }
          maximum_total_divergence = fmax(
              maximum_total_divergence,
              fabs(total_net_flux) / grid[face].cell[cell].volume);
        }
      }
    }
  }
#ifdef GAMERA_CS_MPI_DRIVER
  double global_minimum_density, global_minimum_pressure;
  double global_maximum_total_divergence;
  if (MPI_Allreduce(&minimum_density, &global_minimum_density, 1,
                    MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&minimum_pressure, &global_minimum_pressure, 1,
                    MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&maximum_total_divergence,
                    &global_maximum_total_divergence, 1,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS)
    return 1;
  minimum_density = global_minimum_density;
  minimum_pressure = global_minimum_pressure;
  maximum_total_divergence = global_maximum_total_divergence;
#endif
  gamera_solar_wind_destroy(&wind);
  ROOT_PRINTF("six-face dipole/Boris gate (%zux%zux%zu/face): initial CFL %.6g -> %.6g, t %.6g, min rho %.6g, "
         "min p %.6g, total divB %.6g\n",
         active[0], active[1], active[2],
         smallest_normal, smallest_boris, time, minimum_density,
         minimum_pressure, maximum_total_divergence);
  if (argc == 4 && !drive
#ifdef GAMERA_CS_MPI_DRIVER
      && mpi_rank == 0
#endif
      ) {
    if (write_earth_snapshot(argv[3], grid, storage, owned, active,
                             ghost, gamma, earth_mi_local ? mi : NULL) != 0)
      return 1;
  }
  if (drive && writer_rank && fclose(drive_index) != 0) return 1;
  gamera_cs_halo_plan_destroy(halo_plan);
#ifdef GAMERA_CS_MPI_DRIVER
  gamera_cs_edge_sync_plan_destroy(edge_plan);
#endif
  gamera_cs_mi_dipole_destroy(mi);
  for (int face = 0; face < GAMERA_CS_FACE_COUNT; ++face) {
    gamera_no_background_destroy(&owned[face]);
    gamera_no_storage_destroy(&storage[face]);
    gamera_no_grid_destroy(&grid[face]);
  }
#ifdef GAMERA_CS_MPI_DRIVER
  MPI_Finalize();
#endif
  return (!drive || completed_drive) &&
         minimum_density > 0.0 && minimum_pressure > 0.0 &&
                 maximum_total_divergence < 1e-9 ? 0 : 1;
}
