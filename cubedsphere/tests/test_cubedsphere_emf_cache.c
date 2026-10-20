#define _POSIX_C_SOURCE 200809L
#include "nonorthogonal_sweep.h"
#include "nonorthogonal_flux.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Compare actual double EMFs, including untouched entries on error. This is
 * stronger than comparing the float32 analysis output of a full simulation.
 * Repeated state changes also detect incorrectly persistent/stale caches. */
static int check_mesh(const size_t extent[3]) {
  size_t ve[3] = {extent[0]+1, extent[1]+1, extent[2]+1};
  gamera_no_vec3 *vertices = malloc(gamera_no_element_count3(ve)*sizeof(*vertices));
  if (!vertices) return 1;
  for (size_t i=0; i<ve[0]; ++i)
    for (size_t j=0; j<ve[1]; ++j)
      for (size_t k=0; k<ve[2]; ++k) {
        double x=(double)i, y=(double)j, z=(double)k;
        vertices[gamera_no_index3(ve,i,j,k)] = (gamera_no_vec3){{
          x+0.12*y+0.003*x*x, y+0.07*z+0.002*y*y,
          z+0.05*x+0.001*z*z}};
      }
  gamera_no_grid grid={0};
  gamera_no_storage storage={0};
  if (gamera_no_grid_create(extent,vertices,&grid) ||
      gamera_no_storage_create(extent,&storage)) return 1;
  free(vertices);
  const size_t cells=gamera_no_element_count3(extent);
  double *volumes=malloc(cells*sizeof(*volumes));
  if (!volumes) return 1;
  for (size_t c=0;c<cells;++c) volumes[c]=grid.cell[c].volume;
  gamera_no_background_field background={0};
  gamera_no_vec3 *background_edges[3]={0};
  const double *flux[3];
  for (int d=0;d<3;++d) {
    size_t nf=gamera_no_element_count3(grid.face[d].extent);
    size_t ne=gamera_no_element_count3(grid.edge[d].extent);
    for (size_t f=0;f<nf;++f)
      storage.predicted_face_flux[d][f]=0.1*sin(0.017*(double)(f+3*d));
    flux[d]=storage.predicted_face_flux[d];
    background_edges[d]=malloc(ne*sizeof(gamera_no_vec3));
    if (!background_edges[d]) return 1;
    for (size_t e=0;e<ne;++e)
      background_edges[d][e]=(gamera_no_vec3){{0.3,-0.2,0.1}};
    background.edge_magnetic[d]=background_edges[d];
  }
  int failures=0;
  for (int sample=0;sample<10;++sample) {
    for (size_t c=0;c<cells;++c) {
      grid.cell[c].volume=sample==6 ? 0.0 : volumes[c];
      double rho=sample==2 ? 1.e-14 : 0.8+0.2*sin((double)c*0.13+sample);
      double *u=&storage.predicted_conserved[c*GAMERA_NO_FLUX_COUNT];
      u[GAMERA_NO_FLUX_DENSITY]=rho;
      for (int a=0;a<3;++a)
        u[GAMERA_NO_FLUX_MOMENTUM_X+a]=sample==3 ? -0.0 :
            rho*0.4*cos((double)c*0.07+sample+a);
    }
    size_t badcell=gamera_no_index3(extent,4,4,4);
    if (sample==7) grid.cell[badcell].volume=NAN;
    if (sample==8) storage.predicted_conserved[
        badcell*GAMERA_NO_FLUX_COUNT+GAMERA_NO_FLUX_MOMENTUM_X]=INFINITY;
    gamera_no_emf_options options={1.e-12,1.0,0.6,2.3,0.8,0.02,
                                   sample%2!=0,sample%3!=0};
    for (int d=0;d<3;++d) {
      size_t lower[3]={4,4,4};
      size_t upper[3]={extent[0]-4,extent[1]-4,extent[2]-4};
      /* Non-transverse direction needs no ghosts: exercise the full range. */
      if (sample==4) {lower[d]=0; upper[d]=extent[d];}
      if (sample==5) upper[d]=lower[d]; /* empty sweep */
      size_t ne=gamera_no_element_count3(grid.edge[d].extent);
      size_t badedge=gamera_no_index3(grid.edge[d].extent,4,4,4);
      unsigned char valid=grid.edge[d].valid[badedge];
      if (sample==9) grid.edge[d].valid[badedge]=0;
      double *reference=malloc(ne*sizeof(*reference));
      if (!reference) return 1;
      int status[2];
      for (int mode=0;mode<2;++mode) {
        if (setenv("GAMERA_NO_EMF_FACE_CACHE",mode?"1":"0",1)) return 1;
        for (size_t e=0;e<ne;++e) storage.edge_emf[d][e]=-123.25;
        status[mode]=gamera_no_sweep_edge_emf(&grid,&storage,
            storage.predicted_conserved,flux,d,lower,upper,&options,&background);
        if (!mode) memcpy(reference,storage.edge_emf[d],ne*sizeof(*reference));
      }
      if (status[0]!=status[1] ||
          memcmp(reference,storage.edge_emf[d],ne*sizeof(*reference)) ||
          (sample<5 && status[0]!=0) || (sample>=5 && status[0]==0)) {
        fprintf(stderr,"EMF_CACHE_PARITY_FAIL sample=%d d=%d status=%d,%d\n",
                sample,d,status[0],status[1]);
        ++failures;
      }
      free(reference);
      grid.edge[d].valid[badedge]=valid;
    }
  }
  for (int d=0;d<3;++d) free(background_edges[d]);
  free(volumes);
  gamera_no_storage_destroy(&storage);
  gamera_no_grid_destroy(&grid);
  return failures;
}

int main(void) {
  const size_t extents[][3]={{9,9,9},{12,13,14},{16,10,11}};
  int failures=0;
  for (size_t n=0;n<sizeof(extents)/sizeof(extents[0]);++n)
    failures+=check_mesh(extents[n]);
  if (failures) return 1;
  puts("EMF_CACHE_DOUBLE_BIT_PARITY_PASS cases=90 directions=3 finite/error/stale-state");
  return 0;
}
