#include <hdf5.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int shape(hid_t file,const char *name,int ndims,const hsize_t *expected) {
  hid_t set=H5Dopen2(file,name,H5P_DEFAULT);
  if(set<0) return -1;
  hid_t space=H5Dget_space(set);
  hsize_t actual[4]={0};
  int failed=space<0||H5Sget_simple_extent_ndims(space)!=ndims ||
             H5Sget_simple_extent_dims(space,actual,NULL)<0;
  for(int d=0;d<ndims && !failed;++d) failed|=actual[d]!=expected[d];
  if(space>=0) H5Sclose(space);
  H5Dclose(set);
  return failed?-1:0;
}

static int scalar(hid_t file,const char *name,double *value) {
  hid_t set=H5Dopen2(file,name,H5P_DEFAULT);
  int failed=set<0||H5Dread(set,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
                             H5P_DEFAULT,value)<0||!isfinite(*value);
  if(set>=0) H5Dclose(set);
  return failed?-1:0;
}

static int finite_double_dataset(hid_t file,const char *name,int ndims,
                                 const hsize_t *expected,int positive) {
  if(shape(file,name,ndims,expected)) return -1;
  size_t count=1;
  for(int d=0;d<ndims;++d) count*=expected[d];
  double *values=malloc(count*sizeof(*values));
  if(!values) return -1;
  hid_t set=H5Dopen2(file,name,H5P_DEFAULT);
  int failed=set<0||H5Dread(set,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
                             H5P_DEFAULT,values)<0;
  if(set>=0) H5Dclose(set);
  if(!failed) for(size_t q=0;q<count;++q)
    if(!isfinite(values[q])||(positive&&values[q]<=0.0)) {
      failed=1;break;
    }
  free(values);
  return failed?-1:0;
}

int main(int argc,char **argv) {
  if(argc!=12||strcmp(argv[2],"--radial")||strcmp(argv[4],"--j-cells")||
     strcmp(argv[6],"--k-cells")||strcmp(argv[8],"--stop-seconds")||
     strcmp(argv[10],"--output-seconds")) return 2;
  const char *directory=argv[1];
  const size_t ni=(size_t)strtoull(argv[3],NULL,10),
               nj=(size_t)strtoull(argv[5],NULL,10),
               nk=(size_t)strtoull(argv[7],NULL,10);
  const double stop=strtod(argv[9],NULL),output=strtod(argv[11],NULL);
  if(!ni||!nj||!nk||!(stop>0)||!(output>0)) return 2;
  const hsize_t state_shape[4]={8,ni,nj,nk},
                grid_shape[4]={3,ni,nj,nk},
                vertex_shape[4]={3,ni+1,nj+1,nk+1};
  const size_t nstate=8*ni*nj*nk;
  float *values=malloc(nstate*sizeof(*values));
  if(!values) return 2;
  char path[4096];
  for(int face=0;face<6;++face) {
    snprintf(path,sizeof(path),"%s/analysis_grid_p%d.h5",directory,face);
    hid_t file=H5Fopen(path,H5F_ACC_RDONLY,H5P_DEFAULT);
    if(file<0||shape(file,"/coordinates",4,grid_shape)||
       shape(file,"/vertices",4,vertex_shape)) {
      fprintf(stderr,"invalid grid %s\n",path); free(values); return 1;
    }
    H5Fclose(file);
  }
  int frames=0;
  double last_time=-1;
  for(int sequence=0;sequence<1000000;++sequence) {
    snprintf(path,sizeof(path),"%s/analysis_p0_%06d.h5",directory,sequence);
    if(access(path,F_OK)!=0) break;
    double frame_time=-1;
    for(int face=0;face<6;++face) {
      snprintf(path,sizeof(path),"%s/analysis_p%d_%06d.h5",directory,face,sequence);
      hid_t file=H5Fopen(path,H5F_ACC_RDONLY,H5P_DEFAULT);
      double time=0;
      if(file<0||shape(file,"/state",4,state_shape)||
         scalar(file,"/time_seconds",&time)||
         (face>0&&fabs(time-frame_time)>1e-7)) {
        fprintf(stderr,"invalid analysis %s\n",path);free(values);return 1;
      }
      hid_t set=H5Dopen2(file,"/state",H5P_DEFAULT);
      if(set<0||H5Dread(set,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,
                        H5P_DEFAULT,values)<0) {
        fprintf(stderr,"cannot read %s\n",path);free(values);return 1;
      }
      H5Dclose(set);H5Fclose(file);
      for(size_t q=0;q<nstate;++q) if(!isfinite(values[q])||
          (q<2*ni*nj*nk&&values[q]<=0)) {
        fprintf(stderr,"nonfinite/nonpositive state %s element=%zu\n",path,q);
        free(values);return 1;
      }
      frame_time=time;
    }
    snprintf(path,sizeof(path),"%s/mi_ionosphere_%06d.h5",directory,sequence);
    hid_t mi=H5Fopen(path,H5F_ACC_RDONLY,H5P_DEFAULT);
    const hsize_t polar[2]={22,192};
    const hsize_t shell[3]={6,nj,nk};
    if(mi<0||H5Lexists(mi,"/north/fac_parallel_A_m2",H5P_DEFAULT)<=0||
       H5Lexists(mi,"/south/potential_V",H5P_DEFAULT)<=0||
       finite_double_dataset(mi,"/north/fac_parallel_A_m2",2,polar,0)||
       finite_double_dataset(mi,"/south/fac_parallel_A_m2",2,polar,0)||
       finite_double_dataset(mi,"/north/potential_V",2,polar,0)||
       finite_double_dataset(mi,"/south/potential_V",2,polar,0)||
       finite_double_dataset(mi,"/north/Sigma_P_S",2,polar,1)||
       finite_double_dataset(mi,"/south/Sigma_P_S",2,polar,1)||
       finite_double_dataset(mi,"/shell_current/jparallel_A_m2",3,shell,0)||
       finite_double_dataset(mi,"/shell_current/mlat_deg",3,shell,0)||
       finite_double_dataset(mi,"/shell_current/phi_deg",3,shell,0)) {
      fprintf(stderr,"invalid MI %s\n",path);free(values);return 1;
    }
    H5Fclose(mi);
    if(frame_time<=last_time) {
      fprintf(stderr,"non-monotonic output time frame=%d\n",sequence);
      free(values);return 1;
    }
    last_time=frame_time;
    ++frames;
  }
  free(values);
  const int expected=(int)ceil(stop/output-1e-10)+1;
  if(frames<expected||last_time+1e-6<stop) {
    fprintf(stderr,"incomplete HDF5 frames=%d expected=%d final=%.9g target=%.9g\n",
            frames,expected,last_time,stop);
    return 1;
  }
  printf("EARTH_HDF5_OUTPUT_PASS frames=%d faces=6 final=%.9g s\n",
         frames,last_time);
  return 0;
}
