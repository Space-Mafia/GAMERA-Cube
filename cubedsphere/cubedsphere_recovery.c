#include "nonorthogonal_operators.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

/* The original GAMERA moment recovery is retained, then calibrated against
 * its response to the three exactly uniform Cartesian fields. On warped
 * bilinear faces, scalar-area centroids are not flux-weighted centroids, so
 * the uncalibrated moment formula can misrecover a uniform B by O(h^2).
 * This correction is linked only into the cubed-sphere target. */
int gamera_cs_uncalibrated_flux_to_cell_field(
    const gamera_no_cell_geometry *geometry,
    const double face_flux[GAMERA_NO_DIM][2],
    gamera_no_vec3 *field, double *net_flux);

static int solve3(const double matrix[3][3], const double rhs[3],
                  gamera_no_vec3 *solution) {
  double augmented[3][4];
  double scale = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      augmented[i][j] = matrix[i][j];
      scale = fmax(scale, fabs(matrix[i][j]));
    }
    augmented[i][3] = rhs[i];
  }
  if (!isfinite(scale) || scale <= DBL_MIN) {
    return -1;
  }
  for (int column = 0; column < 3; ++column) {
    int pivot = column;
    for (int row = column + 1; row < 3; ++row) {
      if (fabs(augmented[row][column]) >
          fabs(augmented[pivot][column])) {
        pivot = row;
      }
    }
    if (!isfinite(augmented[pivot][column]) ||
        fabs(augmented[pivot][column]) <= 128.0 * DBL_EPSILON * scale) {
      return -1;
    }
    if (pivot != column) {
      for (int j = column; j < 4; ++j) {
        const double temporary = augmented[column][j];
        augmented[column][j] = augmented[pivot][j];
        augmented[pivot][j] = temporary;
      }
    }
    const double inverse = 1.0 / augmented[column][column];
    for (int j = column; j < 4; ++j) {
      augmented[column][j] *= inverse;
    }
    for (int row = 0; row < 3; ++row) {
      if (row == column) continue;
      const double factor = augmented[row][column];
      for (int j = column; j < 4; ++j) {
        augmented[row][j] -= factor * augmented[column][j];
      }
    }
  }
  for (int i = 0; i < 3; ++i) {
    if (!isfinite(augmented[i][3])) return -1;
    solution->value[i] = augmented[i][3];
  }
  return 0;
}

int gamera_no_flux_to_cell_field(
    const gamera_no_cell_geometry *geometry,
    const double face_flux[GAMERA_NO_DIM][2],
    gamera_no_vec3 *field, double *net_flux) {
  if (geometry == NULL || face_flux == NULL || field == NULL) return -1;
  gamera_no_vec3 raw;
  if (gamera_cs_uncalibrated_flux_to_cell_field(
          geometry, face_flux, &raw, net_flux) != 0) {
    return -1;
  }
  double response[3][3];
  for (int basis = 0; basis < 3; ++basis) {
    double basis_flux[GAMERA_NO_DIM][2];
    for (int d = 0; d < GAMERA_NO_DIM; ++d) {
      for (int side = 0; side < 2; ++side) {
        basis_flux[d][side] =
            geometry->face[d][side].area_vector.value[basis];
      }
    }
    gamera_no_vec3 column;
    if (gamera_cs_uncalibrated_flux_to_cell_field(
            geometry, basis_flux, &column, NULL) != 0) {
      return -1;
    }
    for (int d = 0; d < 3; ++d) {
      response[d][basis] = column.value[d];
    }
  }
  return solve3(response, raw.value, field);
}
