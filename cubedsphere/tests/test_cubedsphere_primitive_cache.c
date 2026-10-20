#define _POSIX_C_SOURCE 200809L
#include "nonorthogonal_sweep.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Actual double fluid/Maxwell flux and emergency diagnostics, not float32
 * analysis output. Rebuild the input on every sample to detect stale reuse. */
static int check_mesh(const size_t extent[3]) {
  size_t ve[3]={extent[0]+1,extent[1]+1,extent[2]+1};
  gamera_no_vec3 *vertices=malloc(gamera_no_element_count3(ve)*sizeof(*vertices));
  if (!vertices) return 1;
  for (size_t i=0;i<ve[0];++i)
    for (size_t j=0;j<ve[1];++j)
      for (size_t k=0;k<ve[2];++k) {
        double x=(double)i,y=(double)j,z=(double)k;
        vertices[gamera_no_index3(ve,i,j,k)]=(gamera_no_vec3){{
            x+0.12*y+0.003*x*x,y+0.07*z+0.002*y*y,z+0.05*x+0.001*z*z}};
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
  gamera_no_vec3 *bgfaces[3]={0};
  const double *flux[3];
  for (int d=0;d<3;++d) {
    size_t nf=gamera_no_element_count3(grid.face[d].extent);
    bgfaces[d]=malloc(nf*sizeof(*bgfaces[d]));
    if (!bgfaces[d]) return 1;
    for (size_t f=0;f<nf;++f) {
      storage.predicted_face_flux[d][f]=0.1*sin(0.017*(double)(f+3*d));
      storage.face_flux[d][f]=0.01;
      bgfaces[d][f]=(gamera_no_vec3){{0.3,-0.2,0.1}};
    }
    flux[d]=storage.predicted_face_flux[d];
    background.face_flux[d]=storage.face_flux[d];
    background.face_magnetic[d]=bgfaces[d];
  }
  int failures=0;
  for (int sample=0;sample<28;++sample) {
    gamera_no_sweep_options options={5.0/3.0,1.e-12,1.e-12,1.0,0.5,0.5,2.3,
        (sample&1)!=0,(sample&2)!=0,(sample&4)!=0,(sample&8)!=0};
    if (sample>=16) options.use_mhd=options.use_hogs=options.use_boris=true;
    for (size_t c=0;c<cells;++c) {
      grid.cell[c].volume=sample==25 ? 0.0 : volumes[c];
      double rho=sample==16 ? 1.e-14 : 0.8+0.2*sin((double)c*0.13+sample);
      double speed2=0.0;
      double *u=&storage.predicted_conserved[c*GAMERA_NO_FLUX_COUNT];
      u[GAMERA_NO_FLUX_DENSITY]=rho;
      for (int a=0;a<3;++a) {
        double v=sample==18 ? -0.0 : sample==19 ? 12.0 :
            0.4*cos((double)c*0.07+sample+a);
        u[GAMERA_NO_FLUX_MOMENTUM_X+a]=rho*v;
        speed2+=v*v;
        storage.predicted_cell_magnetic[c].value[a]=0.12*sin(0.1*(double)c+a);
      }
      u[GAMERA_NO_FLUX_ENERGY]=sample==17 ? -0.1 :
          0.9/(options.gamma-1.0)+0.5*rho*speed2;
    }
    size_t bad=gamera_no_index3(extent,4,4,4)*GAMERA_NO_FLUX_COUNT;
    if (sample==20) storage.predicted_conserved[bad+GAMERA_NO_FLUX_DENSITY]=0.0;
    if (sample==21) storage.predicted_conserved[bad+GAMERA_NO_FLUX_DENSITY]=-1.0;
    if (sample==22) storage.predicted_conserved[bad+GAMERA_NO_FLUX_DENSITY]=NAN;
    if (sample==23) storage.predicted_conserved[bad+GAMERA_NO_FLUX_MOMENTUM_X]=INFINITY;
    if (sample==24) storage.predicted_conserved[bad+GAMERA_NO_FLUX_ENERGY]=NAN;
    if (sample==27) options.gamma=1.0;
    for (int d=0;d<3;++d) {
      size_t lower[3]={4,4,4};
      size_t upper[3]={extent[0]-4,extent[1]-4,extent[2]-4};
      if (sample==26) upper[d]=lower[d];
      if (sample==15) {
        for (int a=0;a<3;++a) if(a!=d) {lower[a]=0;upper[a]=extent[a];}
      }
      size_t nf=gamera_no_element_count3(grid.face[d].extent);
      double *fluid=malloc(nf*GAMERA_NO_FLUX_COUNT*sizeof(*fluid));
      gamera_no_vec3 *maxwell=malloc(nf*sizeof(*maxwell));
      if (!fluid || !maxwell) return 1;
      int status[2]; size_t counts[2]; double maxspeed[2];
      for (int mode=0;mode<2;++mode) {
        if (setenv("GAMERA_NO_PRIMITIVE_CACHE",mode?"1":"0",1)) return 1;
        for (size_t f=0;f<nf;++f) {
          for (int v=0;v<GAMERA_NO_FLUX_COUNT;++v)
            storage.fluid_face_flux[d][f*GAMERA_NO_FLUX_COUNT+v]=-123.25;
          storage.maxwell_face_flux[d][f]=(gamera_no_vec3){{-1.25,-2.25,-3.25}};
        }
        storage.emergency_diffusion_face_count=7;
        storage.emergency_diffusion_max_interface_speed=0.3;
        status[mode]=gamera_no_sweep_face_fluxes(&grid,&storage,
            storage.predicted_conserved,storage.predicted_cell_magnetic,
            flux,d,lower,upper,&options,&background);
        counts[mode]=storage.emergency_diffusion_face_count;
        maxspeed[mode]=storage.emergency_diffusion_max_interface_speed;
        if (!mode) {
          memcpy(fluid,storage.fluid_face_flux[d],nf*GAMERA_NO_FLUX_COUNT*sizeof(*fluid));
          memcpy(maxwell,storage.maxwell_face_flux[d],nf*sizeof(*maxwell));
        }
      }
      if (status[0]!=status[1] || counts[0]!=counts[1] ||
          memcmp(&maxspeed[0],&maxspeed[1],sizeof(double)) ||
          memcmp(fluid,storage.fluid_face_flux[d],nf*GAMERA_NO_FLUX_COUNT*sizeof(*fluid)) ||
          memcmp(maxwell,storage.maxwell_face_flux[d],nf*sizeof(*maxwell)) ||
          (sample<20 && status[0]!=0) || (sample>=20 && status[0]==0) ||
          (sample==19 && counts[0]<=7)) {
        fprintf(stderr,"PRIMITIVE_CACHE_PARITY_FAIL sample=%d d=%d status=%d,%d\n",
                sample,d,status[0],status[1]);
        ++failures;
      }
      free(fluid); free(maxwell);
    }
  }
  for (int d=0;d<3;++d) free(bgfaces[d]);
  free(volumes);
  gamera_no_storage_destroy(&storage);
  gamera_no_grid_destroy(&grid);
  return failures;
}
int main(void) {
  const size_t extents[][3]={{9,9,9},{12,13,14},{16,10,11}};
  int failures=0;
  for(size_t n=0;n<sizeof(extents)/sizeof(extents[0]);++n) failures+=check_mesh(extents[n]);
  if(failures) return 1;
  puts("PRIMITIVE_CACHE_DOUBLE_BIT_PARITY_PASS cases=252 finite/error/HD/MHD/HOGS/Boris/background");
  return 0;
}
