#ifndef GAMERA_CUBEDSPHERE_MI_DIPOLE_H
#define GAMERA_CUBEDSPHERE_MI_DIPOLE_H

#include "cubedsphere_earth_startup.h"
#include "cubedsphere_mesh.h"
#include "nonorthogonal_background.h"
#include "nonorthogonal_mi.h"
#include "nonorthogonal_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gamera_cs_mi_dipole gamera_cs_mi_dipole;

typedef struct gamera_cs_mi_dipole_config {
  size_t azimuth_points;
  size_t colatitude_points;
  size_t current_shell; /* zero-based active radial cell */
  double ionosphere_radius_re;
  double pedersen_siemens;
  double coupling_interval_s;
  double length_norm_m;
  double speed_norm_m_s;
  double mass_density_norm_kg_m3;
  int maximum_iterations;
  double relative_tolerance;
} gamera_cs_mi_dipole_config;

/* Read-only analysis data from the already solved MI state. The two arrays
 * use [colatitude, azimuth] order; no solver arrays are copied or changed. */
typedef struct {
  size_t azimuth_points, colatitude_points;
  size_t shell_j_cells, shell_k_cells;
  double maximum_colatitude_rad, epoch_s, sample_radius_re;
  double pedersen_siemens;
  const double *fac_parallel_a_m2[2];
  const double *potential_v[2];
} gamera_cs_mi_analysis_view;
int gamera_cs_mi_dipole_analysis_view(
    const gamera_cs_mi_dipole *state, gamera_cs_mi_analysis_view *view);
/* Same native sample-shell J_parallel as the existing CSV diagnostic,
 * exposed as read-only arrays for HDF5. Arrays are [face,j,k]. */
int gamera_cs_mi_dipole_export_shell_current(
    const gamera_cs_mi_dipole *state, double *jparallel_a_m2,
    double *mlat_deg, double *phi_deg, size_t count);

/* Serial six-face adapter for the accepted aligned-dipole Yin-Yang MI
 * equations.  The FAC curl and Pedersen solver are the existing
 * non-orthogonal kernels; only six-face angular sampling differs. */
int gamera_cs_mi_dipole_create(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const gamera_no_background_field background[GAMERA_CS_FACE_COUNT],
    const gamera_cs_earth_startup *startup, const size_t active[3],
    size_t ghost, const gamera_cs_mi_dipole_config *config,
    gamera_cs_mi_dipole **result);
void gamera_cs_mi_dipole_destroy(gamera_cs_mi_dipole *state);

/* Lightweight rank-local view of the existing solved polar potential.
 * No MHD shell/grid or new MI equation is allocated on non-root ranks. */
int gamera_cs_mi_dipole_create_potential_view(
    const gamera_cs_earth_startup *startup,
    const gamera_cs_mi_dipole_config *config,
    gamera_cs_mi_dipole **result);
int gamera_cs_mi_dipole_export_potential(
    const gamera_cs_mi_dipole *state, double *values, size_t count,
    double *epoch_s, double *next_update_s);
int gamera_cs_mi_dipole_import_potential(
    gamera_cs_mi_dipole *state, const double *values, size_t count,
    double epoch_s, double next_update_s);

/* Update at natural MHD-step times once the physical cadence has passed.
 * The held potential remains valid between updates; no timestep clipping. */
int gamera_cs_mi_dipole_maybe_update(gamera_cs_mi_dipole *state,
                                     double time_code);
/* Read-only copy of the existing update cadence condition. */
int gamera_cs_mi_dipole_update_due(const gamera_cs_mi_dipole *state,
                                   double time_code);
int gamera_cs_mi_dipole_ready(const gamera_cs_mi_dipole *state);
int gamera_cs_mi_dipole_potential(const gamera_cs_mi_dipole *state,
                                  gamera_no_vec3 point, double *potential_v);
int gamera_cs_mi_dipole_apply_inner_emf(
    const gamera_cs_mi_dipole *state, gamera_no_storage *storage,
    const gamera_no_grid *grid, size_t wall_i);
int gamera_cs_mi_dipole_ghost_velocity(
    const gamera_cs_mi_dipole *state, gamera_no_vec3 point,
    gamera_no_vec3 source_velocity, gamera_no_vec3 *ghost_velocity);
int gamera_cs_mi_dipole_write_csv(const gamera_cs_mi_dipole *state,
                                  const char *prefix);
/* Native sample-shell J_parallel before dipole mapping; for seam diagnostics. */
int gamera_cs_mi_dipole_write_shell_current_csv(
    const gamera_cs_mi_dipole *state, const char *prefix);
int gamera_cs_mi_dipole_stats(const gamera_cs_mi_dipole *state,
                              double *fac_min, double *fac_max,
                              double *potential_min_v,
                              double *potential_max_v,
                              gamera_mi_solver_stats solve[2]);

#ifdef __cplusplus
}
#endif

#endif
