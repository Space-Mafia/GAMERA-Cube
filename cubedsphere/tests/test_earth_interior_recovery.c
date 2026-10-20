#include "cubedsphere_earth_boundary.h"
#include "cubedsphere_mesh.h"
#include "nonorthogonal_operators.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d\n", __LINE__); return 1; } } while (0)

/* An interior radial tile has no physical BC to fill, but BOTH cell-B
 * caches must be recovered after synchronization of the face fluxes. */
int main(void) {
  const size_t active[3] = {24,16,16}, global[3] = {72,16,16};
  const size_t start[3] = {24,0,0}, extent[3] = {32,24,24};
  const size_t ve[3] = {33,25,25};
  gamera_no_vec3 *vertices = calloc(gamera_no_element_count3(ve), sizeof(*vertices));
  gamera_no_radial_map map;
  gamera_no_grid grid = {0};
  gamera_no_storage s = {0};
  CHECK(vertices);
  CHECK(gamera_no_radial_map_init(&map, GAMERA_NO_RADIAL_MAP_EARTH_PRODUCTION_V4,
                                  2.5,200,3) == 0);
  CHECK(gamera_cs_generate_vertices_radial_map_tile(0,global,start,active,4,&map,vertices) == 0);
  CHECK(gamera_no_grid_create(extent,vertices,&grid) == 0);
  CHECK(gamera_cs_calibrate_face_frames(&grid) == 0);
  CHECK(gamera_no_storage_create(extent,&s) == 0);
  const size_t nc = gamera_no_element_count3(extent);
  gamera_no_vec3 *expected = calloc(nc,sizeof(*expected));
  CHECK(expected);
  double *saved[3], *old_saved[3];
  for (int d=0;d<3;++d) {
    size_t n = gamera_no_element_count3(grid.face[d].extent);
    saved[d]=malloc(n*sizeof(double)); old_saved[d]=malloc(n*sizeof(double));
    CHECK(saved[d] && old_saved[d]);
    for (size_t q=0;q<n;++q) {
      s.face_flux[d][q] = grid.face[d].value[q].area_vector.value[2];
      s.old_face_flux[d][q] = 2*s.face_flux[d][q];
    }
    memcpy(saved[d],s.face_flux[d],n*sizeof(double));
    memcpy(old_saved[d],s.old_face_flux[d],n*sizeof(double));
  }
  const double *flux[3]={s.face_flux[0],s.face_flux[1],s.face_flux[2]};
  CHECK(gamera_no_recover_magnetic_field(&grid,flux,expected) == 0);
  for(size_t q=0;q<nc;++q) for(int c=0;c<3;++c)
    s.cell_magnetic[q].value[c]=s.old_cell_magnetic[q].value[c]=12345;
  gamera_cs_decomposition tile={0};
  tile.coord[0]=1; tile.proc_dims[0]=3;
  memcpy(tile.local_active,active,sizeof(active));
  gamera_cs_earth_startup startup={0};
  gamera_solar_wind_series wind;
  gamera_solar_wind_init(&wind);
  gamera_no_background_field bg[6]={{0}};
  gamera_cs_earth_boundary b={.startup=&startup,.wind=&wind,.background=bg,
                            .density_floor=1e-6,.pressure_floor=1e-8};
  CHECK(gamera_cs_earth_fill_radial_ghosts_tile(&grid,&s,&tile,4,NULL,0,0,&b)==0);
  CHECK(memcmp(expected,s.cell_magnetic,nc*sizeof(*expected))==0);
  for(int d=0;d<3;++d) flux[d]=s.old_face_flux[d];
  CHECK(gamera_no_recover_magnetic_field(&grid,flux,expected)==0);
  CHECK(memcmp(expected,s.old_cell_magnetic,nc*sizeof(*expected))==0);
  for(int d=0;d<3;++d) {
    size_t n=gamera_no_element_count3(grid.face[d].extent);
    CHECK(memcmp(saved[d],s.face_flux[d],n*sizeof(double))==0);
    CHECK(memcmp(old_saved[d],s.old_face_flux[d],n*sizeof(double))==0);
    free(saved[d]); free(old_saved[d]);
  }
  /* Zero-initialized conserved states must remain untouched. */
  for(size_t q=0;q<nc*GAMERA_NO_FLUX_COUNT;++q)
    CHECK(s.conserved[q]==0 && s.old_conserved[q]==0);
  free(expected); free(vertices);
  gamera_no_storage_destroy(&s); gamera_no_grid_destroy(&grid);
  puts("EARTH_INTERIOR_RECOVERY_PASS");
  return 0;
}
