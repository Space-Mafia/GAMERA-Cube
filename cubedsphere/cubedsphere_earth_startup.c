#include "cubedsphere_earth_startup.h"

#include "nonorthogonal_step.h"

#include <math.h>

static double radius(gamera_no_vec3 point) {
  return sqrt(point.value[0] * point.value[0] +
              point.value[1] * point.value[1] +
              point.value[2] * point.value[2]);
}

static double cubic_ramp_down(double coordinate, double start,
                              double length) {
  const double scaled = (coordinate - start) / length;
  if (scaled <= 0.0) return 1.0;
  if (scaled >= 1.0) return 0.0;
  return 1.0 - 3.0 * scaled * scaled + 2.0 * scaled * scaled * scaled;
}

static double wind_fraction(double x, const gamera_cs_earth_startup *p) {
  return 0.5 * (1.0 - tanh((x - p->front_x) / p->front_width));
}

/* Same antiderivative/gauge as Problem 21's startup_front_antiderivative. */
static double front_antiderivative(double x,
                                   const gamera_cs_earth_startup *p) {
  const double q = (x - p->front_x) / p->front_width;
  const double absolute_q = fabs(q);
  const double log_cosh =
      absolute_q + log1p(exp(-2.0 * absolute_q)) - log(2.0);
  return 0.5 * (x - p->front_width * log_cosh - p->front_x -
                p->front_width * log(2.0));
}

static int valid(const gamera_cs_earth_startup *p) {
  return p != NULL && isfinite(p->inner_radius) && p->inner_radius > 0.0 &&
         isfinite(p->density) && p->density > 0.0 &&
         isfinite(p->speed) && p->speed > 0.0 &&
         isfinite(p->mach) && p->mach > 0.0 &&
         isfinite(p->gamma) && p->gamma > 1.0 &&
         isfinite(p->ambient_density) && p->ambient_density > 0.0 &&
         isfinite(p->ambient_pressure) && p->ambient_pressure > 0.0 &&
         isfinite(p->front_x) && isfinite(p->front_width) &&
         p->front_width > 0.0 && isfinite(p->cut_radius) &&
         p->cut_radius > p->inner_radius && isfinite(p->cut_length) &&
         p->cut_length > 0.0;
}

int gamera_cs_earth_startup_primitive(gamera_no_vec3 point, void *context,
                                      gamera_no_primitive *primitive) {
  const gamera_cs_earth_startup *p = context;
  if (!valid(p) || primitive == NULL) return -1;
  const double fraction = wind_fraction(point.value[0], p);
  const double inflow_pressure =
      p->density * p->speed * p->speed / (p->gamma * p->mach * p->mach);
  primitive->density = p->ambient_density +
                       fraction * (p->density - p->ambient_density);
  primitive->pressure = p->ambient_pressure +
                        fraction * (inflow_pressure - p->ambient_pressure);
  double wall_fraction = (radius(point) - p->inner_radius) / 2.5;
  wall_fraction = fmax(0.0, fmin(1.0, wall_fraction));
  const double wall_ramp = wall_fraction * wall_fraction *
                           (3.0 - 2.0 * wall_fraction);
  primitive->velocity =
      (gamera_no_vec3){{fraction * p->speed * wall_ramp, 0.0, 0.0}};
  return 0;
}

int gamera_cs_earth_startup_potential(gamera_no_vec3 point, void *context,
                                      gamera_no_vec3 *potential) {
  const gamera_cs_earth_startup *p = context;
  if (!valid(p) || potential == NULL) return -1;
  const double r = radius(point);
  if (!(r > 0.0)) return -1;
  const double weight = cubic_ramp_down(r, p->cut_radius, p->cut_length);
  const double factor = weight / (r * r * r);
  const double *m = p->dipole.moment.value;
  const double *x = point.value;
  *potential = (gamera_no_vec3){{
      factor * (m[1] * x[2] - m[2] * x[1]),
      factor * (m[2] * x[0] - m[0] * x[2]),
      factor * (m[0] * x[1] - m[1] * x[0])}};
  potential->value[1] -= 0.5 * p->imf.value[0] * x[2];
  potential->value[2] += 0.5 * p->imf.value[0] * x[1];
  const double front = front_antiderivative(x[0], p);
  potential->value[1] += p->imf.value[2] * front;
  potential->value[2] -= p->imf.value[1] * front;
  return 0;
}

int gamera_cs_earth_startup_background(gamera_no_vec3 point, void *context,
                                       gamera_no_vec3 *field) {
  const gamera_cs_earth_startup *p = context;
  if (!valid(p) || field == NULL ||
      gamera_no_dipole_field(point, (void *)&p->dipole, field) != 0)
    return -1;
  const double weight = cubic_ramp_down(radius(point), p->cut_radius,
                                        p->cut_length);
  for (int c = 0; c < GAMERA_NO_DIM; ++c) field->value[c] *= weight;
  return 0;
}

int gamera_cs_earth_background_adjust(const gamera_no_grid *grid,
                                      gamera_no_background_data *background,
                                      const gamera_cs_earth_startup *startup) {
  if (grid == NULL || background == NULL || background->cell_force == NULL ||
      !valid(startup)) return -1;
  const size_t count = gamera_no_element_count3(grid->cell_extent);
  for (size_t cell = 0; cell < count; ++cell) {
    if (radius(grid->cell[cell].centroid) <= 0.75 * startup->cut_radius)
      background->cell_force[cell] = (gamera_no_vec3){{0.0, 0.0, 0.0}};
  }
  return 0;
}

int gamera_cs_earth_total_to_residual(
    const gamera_no_grid *grid, gamera_no_storage *storage,
    const gamera_no_background_data *background) {
  if (grid == NULL || storage == NULL || background == NULL) return -1;
  for (int direction = 0; direction < GAMERA_NO_DIM; ++direction) {
    if (storage->face_flux[direction] == NULL ||
        storage->old_face_flux[direction] == NULL ||
        background->face_flux[direction] == NULL) return -1;
    const size_t count =
        gamera_no_element_count3(grid->face[direction].extent);
    for (size_t face = 0; face < count; ++face) {
      storage->face_flux[direction][face] -=
          background->face_flux[direction][face];
      storage->old_face_flux[direction][face] -=
          background->face_flux[direction][face];
    }
  }
  const double *current[3] = {storage->face_flux[0],
                              storage->face_flux[1],
                              storage->face_flux[2]};
  const double *old[3] = {storage->old_face_flux[0],
                          storage->old_face_flux[1],
                          storage->old_face_flux[2]};
  return gamera_no_recover_magnetic_field(grid, current,
                                         storage->cell_magnetic) == 0 &&
                 gamera_no_recover_magnetic_field(
                     grid, old, storage->old_cell_magnetic) == 0
             ? 0 : -1;
}
