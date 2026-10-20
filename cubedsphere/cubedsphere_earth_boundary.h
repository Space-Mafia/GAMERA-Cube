#ifndef GAMERA_CUBEDSPHERE_EARTH_BOUNDARY_H
#define GAMERA_CUBEDSPHERE_EARTH_BOUNDARY_H

#include "cubedsphere_earth_startup.h"
#include "cubedsphere_decomposition.h"
#include "cubedsphere_step.h"
#include "solar_wind.h"

struct gamera_cs_mi_dipole;

/* Adapter for the accepted Yin-Yang Earth no-MI radial boundary. The wind
 * sampler and MHD/CT kernels remain the existing non-orthogonal modules. */
typedef struct {
  const gamera_cs_earth_startup *startup;
  const gamera_solar_wind_series *wind;
  const gamera_no_background_field *background; /* six faces */
  const struct gamera_cs_mi_dipole *mi; /* NULL retains the no-MI wall */
  double density_floor;
  double pressure_floor;
  double step_time;
  double step_dt;
  double cfl;
  int strict_inner_gas_wall; /* Accepted Yin-Yang direct wall-state variant. */
} gamera_cs_earth_boundary;

int gamera_cs_earth_fill_radial_ghosts(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double current_time, double previous_time, void *context);
int gamera_cs_earth_fill_radial_ghosts_face(
    const gamera_no_grid grid[GAMERA_CS_FACE_COUNT],
    gamera_no_storage storage[GAMERA_CS_FACE_COUNT],
    const size_t active[3], size_t ghost,
    double current_time, double previous_time, void *context, int face);

/* Same accepted radial states and magnetic wall, applied only by actual
 * global inner/outer radial tiles. Internal radial MPI halos are untouched. */
int gamera_cs_earth_fill_radial_ghosts_tile(
    const gamera_no_grid *grid, gamera_no_storage *local,
    const gamera_cs_decomposition *tile, size_t ghost,
    const gamera_no_background_field *background,
    double current_time, double previous_time, void *context);

int gamera_cs_earth_inner_fluid_flux(
    gamera_no_storage *storage, const gamera_no_grid *grid,
    const size_t lower[3], const size_t upper[3], void *context);

int gamera_cs_earth_no_mi_emf(
    gamera_no_storage *storage, const gamera_no_grid *grid,
    const size_t lower[3], const size_t upper[3], void *context);

typedef struct {
  const gamera_cs_decomposition *tile;
  gamera_cs_earth_boundary *earth;
} gamera_cs_earth_tile_emf_context;

/* Same accepted wind-diffusive outer EMF and inner E_t/MI EMF, selected by
 * global radial tile ownership. Safe as a GAMERA edge_emf_sync callback. */
int gamera_cs_earth_tile_emf(
    gamera_no_storage *storage, const gamera_no_grid *grid,
    const size_t lower[3], const size_t upper[3], void *context);

#endif
