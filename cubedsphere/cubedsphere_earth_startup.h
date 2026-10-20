#ifndef GAMERA_CUBEDSPHERE_EARTH_STARTUP_H
#define GAMERA_CUBEDSPHERE_EARTH_STARTUP_H

#include "nonorthogonal_background.h"
#include "nonorthogonal_initialization.h"

/* Geometry-independent Problem 21 startup data. The cube-sphere adapter
 * uses the accepted Yin-Yang Earth's upstream-front and cut-dipole formulas;
 * the GAMERA MHD/CT solver and initialization kernels are unchanged. */
typedef struct {
  double inner_radius;
  double density;
  double speed;
  double mach;
  double gamma;
  double ambient_density;
  double ambient_pressure;
  double front_x;
  double front_width;
  double cut_radius;
  double cut_length;
  gamera_no_vec3 imf;
  gamera_no_dipole dipole;
} gamera_cs_earth_startup;

int gamera_cs_earth_startup_primitive(gamera_no_vec3 point, void *context,
                                      gamera_no_primitive *primitive);
int gamera_cs_earth_startup_potential(gamera_no_vec3 point, void *context,
                                      gamera_no_vec3 *potential);
int gamera_cs_earth_startup_background(gamera_no_vec3 point, void *context,
                                       gamera_no_vec3 *field);
int gamera_cs_earth_background_adjust(const gamera_no_grid *grid,
                                      gamera_no_background_data *background,
                                      const gamera_cs_earth_startup *startup);
int gamera_cs_earth_total_to_residual(
    const gamera_no_grid *grid, gamera_no_storage *storage,
    const gamera_no_background_data *background);

#endif
