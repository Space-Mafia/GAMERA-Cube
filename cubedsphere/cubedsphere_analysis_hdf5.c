#include "cubedsphere_analysis_hdf5.h"

#include "nonorthogonal_state.h"

#include <hdf5.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { FIELD_COUNT = 8, NAME_BYTES = 16 };

static int consensus(const gamera_cs_analysis_hdf5 *o, int failed) {
  int any = 1;
  return MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX,
                       o->face_comm) == MPI_SUCCESS ? any : 1;
}

static int global_consensus(MPI_Comm comm, int failed) {
  int any = 1;
  return MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX,
                       comm) == MPI_SUCCESS ? any : 1;
}

static hid_t parallel_file(const gamera_cs_analysis_hdf5 *o,
                           const char *path) {
  hid_t access = H5Pcreate(H5P_FILE_ACCESS);
  if (consensus(o, access < 0)) {
    if (access >= 0) H5Pclose(access);
    return -1;
  }
  int failed = H5Pset_fapl_mpio(access, o->face_comm, MPI_INFO_NULL) < 0;
  if (consensus(o, failed)) {
    H5Pclose(access);
    return -1;
  }
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, access);
  H5Pclose(access);
  return file;
}

static int root_dataset(const gamera_cs_analysis_hdf5 *o, hid_t file,
                        const char *name, hid_t disk_type, hid_t memory_type,
                        int ndims, const hsize_t *shape, const void *data) {
  hid_t space = H5Screate_simple(ndims, shape, NULL);
  hid_t set = space < 0 ? -1 : H5Dcreate2(file, name, disk_type, space,
                                          H5P_DEFAULT, H5P_DEFAULT,
                                          H5P_DEFAULT);
  int failed = consensus(o, space < 0 || set < 0);
  if (!failed && o->face_rank == 0 &&
      H5Dwrite(set, memory_type, H5S_ALL, H5S_ALL,
               H5P_DEFAULT, data) < 0) failed = 1;
  if (set >= 0) H5Dclose(set);
  if (space >= 0) H5Sclose(space);
  return consensus(o, failed) ? -1 : 0;
}

static int slab(const gamera_cs_analysis_hdf5 *o, hid_t file,
                const char *name, hid_t disk_type, hid_t memory_type,
                const hsize_t global[4], const hsize_t local[4],
                const hsize_t offset[4], const void *data) {
  hid_t file_space = H5Screate_simple(4, global, NULL);
  hid_t memory_space = H5Screate_simple(4, local, NULL);
  hid_t set = file_space < 0 ? -1 :
      H5Dcreate2(file, name, disk_type, file_space, H5P_DEFAULT,
                 H5P_DEFAULT, H5P_DEFAULT);
  hid_t transfer = H5Pcreate(H5P_DATASET_XFER);
  int failed = consensus(o, file_space < 0 || memory_space < 0 ||
                            set < 0 || transfer < 0);
  if (!failed && (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset,
                                       NULL, local, NULL) < 0 ||
                  H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0))
    failed = 1;
  failed = consensus(o, failed);
  if (!failed && H5Dwrite(set, memory_type, memory_space, file_space,
                          transfer, data) < 0) failed = 1;
  if (transfer >= 0) H5Pclose(transfer);
  if (set >= 0) H5Dclose(set);
  if (memory_space >= 0) H5Sclose(memory_space);
  if (file_space >= 0) H5Sclose(file_space);
  return consensus(o, failed) ? -1 : 0;
}

static int metadata(const gamera_cs_analysis_hdf5 *o, hid_t file,
                    long sequence, long step, double time_code,
                    double time_seconds) {
  const hsize_t one[1] = {1}, three[1] = {3}, ten[1] = {10};
  const int schema = 1, face = o->tile.face, backend = 3, problem = 1;
  const int seq = (int)sequence;
  const int64_t step64 = (int64_t)step;
  const int cells[3] = {(int)o->tile.global_active[0],
                        (int)o->tile.global_active[1],
                        (int)o->tile.global_active[2]};
  const double logical_lower[3] = {2.5, -1.0, -1.0};
  const double logical_upper[3] = {200.0, 1.0, 1.0};
  const double mu0 = 4.0e-7 * acos(-1.0);
  const double x = 6371000.0, u = 100000.0, rho = 1.67262192369e-21;
  const double b = u * sqrt(mu0 * rho);
  const double norm[10] = {mu0, x, u, x/u, rho, rho*u*u, b,
                           b*x*x*x/mu0, u/x, 1.0/(mu0*u)};
  if (sequence > INT32_MAX || step < 0 ||
      root_dataset(o,file,"/analysis_schema",H5T_STD_I32LE,H5T_NATIVE_INT,1,one,&schema) ||
      root_dataset(o,file,"/patch_id",H5T_STD_I32LE,H5T_NATIVE_INT,1,one,&face) ||
      root_dataset(o,file,"/sequence",H5T_STD_I32LE,H5T_NATIVE_INT,1,one,&seq) ||
      root_dataset(o,file,"/step",H5T_STD_I64LE,H5T_NATIVE_INT64,1,one,&step64) ||
      root_dataset(o,file,"/coordinate_backend",H5T_STD_I32LE,H5T_NATIVE_INT,1,one,&backend) ||
      root_dataset(o,file,"/problem_id",H5T_STD_I32LE,H5T_NATIVE_INT,1,one,&problem) ||
      root_dataset(o,file,"/global_cells",H5T_STD_I32LE,H5T_NATIVE_INT,1,three,cells) ||
      root_dataset(o,file,"/logical_lower",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,1,three,logical_lower) ||
      root_dataset(o,file,"/logical_upper",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,1,three,logical_upper) ||
      root_dataset(o,file,"/time_code",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,1,one,&time_code) ||
      root_dataset(o,file,"/time_seconds",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,1,one,&time_seconds) ||
      root_dataset(o,file,"/normalization",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,1,ten,norm)) return -1;
  static const char names[FIELD_COUNT][NAME_BYTES] = {
      "rho","p","vx","vy","vz","Bx","By","Bz"};
  hid_t type = H5Tcopy(H5T_C_S1);
  int failed = type < 0 || H5Tset_size(type, NAME_BYTES) < 0 ||
               H5Tset_strpad(type, H5T_STR_NULLTERM) < 0;
  if (consensus(o, failed)) {
    if (type >= 0) H5Tclose(type);
    return -1;
  }
  const hsize_t eight[1] = {FIELD_COUNT};
  failed = root_dataset(o,file,"/field_names",type,type,1,eight,names);
  H5Tclose(type);
  return failed;
}

int gamera_cs_analysis_hdf5_init(gamera_cs_analysis_hdf5 *o,
                                  const char *legacy_prefix,
                                  const gamera_cs_decomposition *tile,
                                  MPI_Comm world) {
  if (!o || !legacy_prefix || !tile) return -1;
  memset(o, 0, sizeof(*o));
  o->face_comm = MPI_COMM_NULL;
  o->tile = *tile;
  MPI_Comm_rank(world, &o->world_rank);
  const char *slash = strrchr(legacy_prefix, '/');
  size_t length = slash ? (size_t)(slash - legacy_prefix) : 1U;
  if (length >= sizeof(o->directory)) return -1;
  if (slash) memcpy(o->directory, legacy_prefix, length);
  else o->directory[0] = '.';
  o->directory[length] = '\0';
  if (MPI_Comm_split(world, tile->face, o->world_rank,
                     &o->face_comm) != MPI_SUCCESS) return -1;
  MPI_Comm_rank(o->face_comm, &o->face_rank);
  return 0;
}

void gamera_cs_analysis_hdf5_close(gamera_cs_analysis_hdf5 *o) {
  if (o && o->face_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&o->face_comm);
    o->face_comm = MPI_COMM_NULL;
  }
}

static int grid_file(gamera_cs_analysis_hdf5 *o, const gamera_no_grid *grid,
                     size_t ghost, double time_code, double time_seconds) {
  const size_t ni=o->tile.local_active[0], nj=o->tile.local_active[1],
               nk=o->tile.local_active[2];
  const int vi=(int)ni+(o->tile.coord[0]+1==o->tile.proc_dims[0]);
  const int vj=(int)nj+(o->tile.coord[1]+1==o->tile.proc_dims[1]);
  const int vk=(int)nk+(o->tile.coord[2]+1==o->tile.proc_dims[2]);
  const size_t nc=ni*nj*nk, nv=(size_t)vi*(size_t)vj*(size_t)vk;
  int failed = nc==0 || nv==0 || nc>SIZE_MAX/(3*sizeof(double)) ||
               nv>SIZE_MAX/(3*sizeof(double));
  double *coords=failed?NULL:malloc(3*nc*sizeof(double));
  double *vertices=failed?NULL:malloc(3*nv*sizeof(double));
  failed |= coords==NULL || vertices==NULL;
  if (consensus(o,failed)) { free(coords); free(vertices); return -1; }
  for(size_t i=0;i<ni;++i) for(size_t j=0;j<nj;++j) for(size_t k=0;k<nk;++k) {
    size_t q=(i*nj+j)*nk+k;
    size_t cell=gamera_no_index3(grid->cell_extent,i+ghost,j+ghost,k+ghost);
    for(int c=0;c<3;++c) coords[(size_t)c*nc+q]=grid->cell[cell].centroid.value[c];
  }
  for(int i=0;i<vi;++i) for(int j=0;j<vj;++j) for(int k=0;k<vk;++k) {
    size_t q=((size_t)i*(size_t)vj+(size_t)j)*(size_t)vk+(size_t)k;
    size_t v=gamera_no_index3(grid->vertex_extent,(size_t)i+ghost,
                               (size_t)j+ghost,(size_t)k+ghost);
    for(int c=0;c<3;++c) vertices[(size_t)c*nv+q]=grid->vertex[v].value[c];
  }
  char path[4096];
  int length=snprintf(path,sizeof(path),"%s/analysis_grid_p%d.h5",
                      o->directory,o->tile.face);
  if (consensus(o,length<0||(size_t)length>=sizeof(path))) {
    free(coords); free(vertices); return -1;
  }
  hid_t file=parallel_file(o,path);
  failed=consensus(o,file<0);
  const hsize_t global[4]={3,o->tile.global_active[0],o->tile.global_active[1],o->tile.global_active[2]};
  const hsize_t local[4]={3,ni,nj,nk};
  const hsize_t offset[4]={0,o->tile.global_start[0],o->tile.global_start[1],o->tile.global_start[2]};
  const hsize_t vg[4]={3,o->tile.global_active[0]+1,o->tile.global_active[1]+1,o->tile.global_active[2]+1};
  const hsize_t vl[4]={3,(hsize_t)vi,(hsize_t)vj,(hsize_t)vk};
  if (!failed && (metadata(o,file,-1,0,time_code,time_seconds) ||
      slab(o,file,"/coordinates",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,global,local,offset,coords) ||
      slab(o,file,"/vertices",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,vg,vl,offset,vertices))) failed=1;
  if (file>=0 && H5Fclose(file)<0) failed=1;
  free(coords); free(vertices);
  if (consensus(o,failed)) return -1;
  o->grid_written=1;
  return 0;
}

static int serial_dataset(hid_t parent,const char *name,int ndims,
                          const hsize_t *shape,const double *data) {
  hid_t space=H5Screate_simple(ndims,shape,NULL);
  hid_t set=space<0?-1:H5Dcreate2(parent,name,H5T_IEEE_F64LE,space,
                                    H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  int failed=space<0||set<0||H5Dwrite(set,H5T_NATIVE_DOUBLE,H5S_ALL,
                                     H5S_ALL,H5P_DEFAULT,data)<0;
  if(set>=0) H5Dclose(set);
  if(space>=0) H5Sclose(space);
  return failed?-1:0;
}

static int scalar_attribute(hid_t parent,const char *name,hid_t disk_type,
                            hid_t memory_type,const void *data) {
  hid_t space=H5Screate(H5S_SCALAR);
  hid_t attr=space<0?-1:H5Acreate2(parent,name,disk_type,space,
                                    H5P_DEFAULT,H5P_DEFAULT);
  int failed=space<0||attr<0||H5Awrite(attr,memory_type,data)<0;
  if(attr>=0) H5Aclose(attr);
  if(space>=0) H5Sclose(space);
  return failed?-1:0;
}

static int mi_file(const gamera_cs_analysis_hdf5 *o,long sequence,
                   double time_code,double time_seconds,
                   const gamera_cs_mi_dipole *mi) {
  gamera_cs_mi_analysis_view view;
  if(gamera_cs_mi_dipole_analysis_view(mi,&view)) return -1;
  char path[4096];
  int length=snprintf(path,sizeof(path),"%s/mi_ionosphere_%06ld.h5",
                      o->directory,sequence);
  if(length<0||(size_t)length>=sizeof(path)) return -1;
  hid_t file=H5Fcreate(path,H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
  if(file<0) return -1;
  const size_t np=view.azimuth_points,nt=view.colatitude_points;
  const size_t ns=6U*view.shell_j_cells*view.shell_k_cells;
  double *longitude=malloc(np*sizeof(double));
  double *colatitude=malloc(nt*sizeof(double));
  double *conductance=malloc(np*nt*sizeof(double));
  double *shell_j=malloc(ns*sizeof(double));
  double *shell_mlat=malloc(ns*sizeof(double));
  double *shell_phi=malloc(ns*sizeof(double));
  int failed=longitude==NULL||colatitude==NULL||conductance==NULL||
             shell_j==NULL||shell_mlat==NULL||shell_phi==NULL;
  if(!failed) {
    for(size_t p=0;p<np;++p) longitude[p]=2.0*acos(-1.0)*(double)p/(double)np;
    for(size_t t=0;t<nt;++t) colatitude[t]=view.maximum_colatitude_rad*(double)t/(double)(nt-1);
    for(size_t q=0;q<np*nt;++q) conductance[q]=view.pedersen_siemens;
    const hsize_t az[1]={np},co[1]={nt},polar[2]={nt,np},one[1]={1};
    const hsize_t shell_shape[3]={6,view.shell_j_cells,view.shell_k_cells};
    const int schema=1,zero=0,yes=1;
    failed|=scalar_attribute(file,"schema_version",H5T_STD_I32LE,H5T_NATIVE_INT,&schema);
    failed|=scalar_attribute(file,"polar_values_are_nodes",H5T_STD_I32LE,H5T_NATIVE_INT,&yes);
    failed|=scalar_attribute(file,"hemisphere_longitude_axes_identical",H5T_STD_I32LE,H5T_NATIVE_INT,&yes);
    failed|=scalar_attribute(file,"electron_precipitation_enabled",H5T_STD_I32LE,H5T_NATIVE_INT,&zero);
    failed|=scalar_attribute(file,"time_code",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,&time_code);
    failed|=scalar_attribute(file,"time_seconds",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,&time_seconds);
    failed|=scalar_attribute(file,"sample_radius_RE",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,&view.sample_radius_re);
    failed|=scalar_attribute(file,"pedersen_siemens",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,&view.pedersen_siemens);
    failed|=serial_dataset(file,"longitude_rad",1,az,longitude);
    failed|=serial_dataset(file,"colatitude_rad",1,co,colatitude);
    failed|=serial_dataset(file,"time_code",1,one,&time_code);
    failed|=serial_dataset(file,"time_seconds",1,one,&time_seconds);
    failed|=serial_dataset(file,"potential_epoch_seconds",1,one,&view.epoch_s);
    failed|=serial_dataset(file,"sample_radius_RE",1,one,&view.sample_radius_re);
    for(int h=0;h<2;++h) {
      hid_t group=H5Gcreate2(file,h==0?"north":"south",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      if(group<0) { failed=1; continue; }
      failed|=serial_dataset(group,"fac_parallel_A_m2",2,polar,view.fac_parallel_a_m2[h]);
      failed|=serial_dataset(group,"potential_V",2,polar,view.potential_v[h]);
      failed|=serial_dataset(group,"Sigma_P_S",2,polar,conductance);
      if(H5Gclose(group)<0) failed=1;
    }
    if(gamera_cs_mi_dipole_export_shell_current(mi,shell_j,shell_mlat,
                                                 shell_phi,ns)) failed=1;
    if(!failed) {
      hid_t group=H5Gcreate2(file,"shell_current",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      if(group<0) failed=1;
      else {
        failed|=serial_dataset(group,"jparallel_A_m2",3,shell_shape,shell_j);
        failed|=serial_dataset(group,"mlat_deg",3,shell_shape,shell_mlat);
        failed|=serial_dataset(group,"phi_deg",3,shell_shape,shell_phi);
        if(H5Gclose(group)<0) failed=1;
      }
    }
  }
  free(longitude);free(colatitude);free(conductance);
  free(shell_j);free(shell_mlat);free(shell_phi);
  if(H5Fclose(file)<0) failed=1;
  return failed?-1:0;
}

int gamera_cs_analysis_hdf5_write(gamera_cs_analysis_hdf5 *o,
                                   long sequence,long step,double time_code,
                                   double time_seconds,const gamera_no_grid *grid,
                                   const gamera_no_storage *storage,
                                   const gamera_no_background_data *background,
                                   size_t ghost,double gamma,
                                   const gamera_cs_mi_dipole *mi,
                                   MPI_Comm world) {
  if(!o||!grid||!storage||!background||sequence<0||step<0) return -1;
  int grid_failed=!o->grid_written &&
      grid_file(o,grid,ghost,time_code,time_seconds);
  if(global_consensus(world,grid_failed)) return -1;
  const size_t ni=o->tile.local_active[0],nj=o->tile.local_active[1],
               nk=o->tile.local_active[2],count=ni*nj*nk;
  int failed=count==0||count>SIZE_MAX/(FIELD_COUNT*sizeof(float));
  float *state=failed?NULL:malloc(FIELD_COUNT*count*sizeof(float));
  failed|=state==NULL;
  if(global_consensus(world,consensus(o,failed))) {free(state);return -1;}
  for(size_t i=0;i<ni;++i) for(size_t j=0;j<nj;++j) for(size_t k=0;k<nk;++k) {
    size_t q=(i*nj+j)*nk+k;
    size_t cell=gamera_no_index3(grid->cell_extent,i+ghost,j+ghost,k+ghost);
    gamera_no_primitive p;
    if(gamera_no_conserved_to_primitive(&storage->conserved[cell*GAMERA_NO_FLUX_COUNT],
                                       gamma,1e-12,1e-12,&p)) {failed=1;continue;}
    const double fields[FIELD_COUNT]={p.density,p.pressure,
      p.velocity.value[0],p.velocity.value[1],p.velocity.value[2],
      storage->cell_magnetic[cell].value[0]+background->cell_magnetic[cell].value[0],
      storage->cell_magnetic[cell].value[1]+background->cell_magnetic[cell].value[1],
      storage->cell_magnetic[cell].value[2]+background->cell_magnetic[cell].value[2]};
    for(int f=0;f<FIELD_COUNT;++f) {
      state[(size_t)f*count+q]=(float)fields[f];
      failed|=!isfinite(fields[f])||!isfinite(state[(size_t)f*count+q]);
    }
  }
  if(global_consensus(world,consensus(o,failed))) {free(state);return -1;}
  char path[4096];
  int length=snprintf(path,sizeof(path),"%s/analysis_p%d_%06ld.h5",
                      o->directory,o->tile.face,sequence);
  if(global_consensus(world,consensus(o,length<0||(size_t)length>=sizeof(path)))) {
    free(state);return -1;
  }
  hid_t file=parallel_file(o,path);
  failed=consensus(o,file<0);
  const hsize_t global[4]={FIELD_COUNT,o->tile.global_active[0],
                            o->tile.global_active[1],o->tile.global_active[2]};
  const hsize_t local[4]={FIELD_COUNT,ni,nj,nk};
  const hsize_t offset[4]={0,o->tile.global_start[0],
                            o->tile.global_start[1],o->tile.global_start[2]};
  if(!failed && (metadata(o,file,sequence,step,time_code,time_seconds)||
                slab(o,file,"/state",H5T_IEEE_F32LE,H5T_NATIVE_FLOAT,
                     global,local,offset,state))) failed=1;
  if(file>=0 && H5Fclose(file)<0) failed=1;
  free(state);
  if(global_consensus(world,consensus(o,failed))) return -1;
  int mi_failed=0;
  if(o->world_rank==0 && mi) mi_failed=mi_file(o,sequence,time_code,time_seconds,mi);
  return global_consensus(world,mi_failed)?-1:0;
}
