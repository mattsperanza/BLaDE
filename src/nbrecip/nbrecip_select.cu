#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#include "nbrecip/nbrecip.h"
#include "main/defines.h"
#include "system/system.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "enhanced/enhanced.h"

#include "main/real3.h"

// getforce_ewald_spread_kernel<<<>>>(N,p->charge_d,m->atomBlock_d,(real3*)s->position_d,s->orthBox,m->lambda_d,((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
template <bool flagBox,int order,typename box_type>
__global__ void getforce_ewald_spread_kernel_select(
    int atomCount,real *charge,int *atomBlock,
    real3* position,box_type kbox,real *lambda,
    int* selections, bool spread_selected,
    int3 gridDimPME,real *chargeGridPME)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real q;
  int b;
  real l=1;
  real3 xi; // position
  real u; // fractional coordinate remainder
  int3 u0; // index of grid point
  real Meven,Modd; // even and odd order B splines
  real3 density;
  // real3 dDensity;
  real3 dIndex;
  int j,jx,jy,jz,jxs,jys,jzs;
  int3 index;
  int iThread=rectify_modulus(threadIdx.x,32); // threadIdx.x&31; // within warp
  int iAtom=i/8;
  int threadOfAtom=rectify_modulus(i,8); // iThread%order;

  u0=make_int3(0,0,0);

  if (iAtom<atomCount) {
    q=charge[iAtom];
    // Only modification in this kernel
    if(spread_selected && !selections[iAtom]){
      q=0;
    } else if (!spread_selected && selections[iAtom]){
      q=0;
    }

    // Scaling
    b=atomBlock[iAtom];
    if (b) {
      l=lambda[b];
    }
    q*=l;

    xi=position[iAtom];

    // Get grid position
    if (flagBox) {
      u=gridDimPME.x*(xi.x*boxxx(kbox)+xi.y*boxxy(kbox)+xi.z*boxxz(kbox));
    } else {
      u=xi.x*gridDimPME.x*boxxx(kbox);
    }
    u0.x=(int)floor(u);
    u-=u0.x;
    if (flagBox) { // x-direction could be up to 1.25*Lx+diff out of box
      u0.x=nearby_modulus(u0.x,gridDimPME.x);
    }
    u0.x=nearby_modulus(u0.x,gridDimPME.x);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.x=Meven;
  // dDensity.x=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  if (iAtom<atomCount) {
    if (flagBox) {
      u=gridDimPME.y*(xi.y*boxyy(kbox)+xi.z*boxyz(kbox));
    } else {
      u=xi.y*gridDimPME.y*boxyy(kbox);
    }
    u0.y=(int)floor(u);
    u-=u0.y;
    // y direction is only 0.5*Ly+diff out of triclinic box
    u0.y=nearby_modulus(u0.y,gridDimPME.y);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.y=Meven;
  // dDensity.y=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  if (iAtom<atomCount) {
    if (flagBox) {
      u=gridDimPME.z*(xi.z*boxzz(kbox));
    } else {
      u=xi.z*gridDimPME.z*boxzz(kbox);
    }
    u0.z=(int)floor(u);
    u-=u0.z;
    // z direction is only diff out of triclinic box, same as orthogonal box
    u0.z=nearby_modulus(u0.z,gridDimPME.z);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.z=Meven;
  // dDensity.z=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  jxs=(threadIdx.x&4)==4;
  jys=(threadIdx.x&2)==2;
  jzs=(threadIdx.x&1);
  for (jx=0; jx<order/2; jx++) {
    dIndex.x=__shfl_sync(0xFFFFFFFF,density.x,2*jx+jxs,8);
    index.x=over_modulus(u0.x+2*jx+jxs,gridDimPME.x);
    for (jy=0; jy<order/2; jy++) {
      dIndex.y=__shfl_sync(0xFFFFFFFF,density.y,2*jy+jys,8);
      index.y=index.x*gridDimPME.y+over_modulus(u0.y+2*jy+jys,gridDimPME.y);
      for (jz=0; jz<order/2; jz++) {
        dIndex.z=__shfl_sync(0xFFFFFFFF,density.z,2*jz+jzs,8);
        index.z=index.y*gridDimPME.z+over_modulus(u0.z+2*jz+jzs,gridDimPME.z);
        if (iAtom<atomCount) {
          atomicAdd(&chargeGridPME[index.z],q*dIndex.x*dIndex.y*dIndex.z);
        }
      }
    }
  }
}

// getforce_ewald_convolution_kernel<<<blockCount,blockSize,0,p->nbrecipStream>>>(((int3*)gridDimPME)[0],p->fourierGridPME_d,p->bGridPME_d,system->run->betaEwald,s->orthoBox)
template <bool flagBox,typename box_type>
__global__ void getforce_ewald_convolution_kernel_select(int3 gridDimPME,myCufftComplex *fourierGridPME,real *bGridPME,real betaEwald,box_type kbox)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  int k=blockIdx.z*blockDim.z+threadIdx.z;
  int ijk=((i*gridDimPME.y)+j)*(gridDimPME.z/2+1)+k;
  real Vinv=boxxx(kbox)*boxyy(kbox)*boxzz(kbox);
  real kcomp;
  real k2; // squared reciprocal space vector
  real factor;

  if (i<gridDimPME.x && j<gridDimPME.y && k<(gridDimPME.z/2+1)) {
    i=((2*i<=gridDimPME.x)?i:i-gridDimPME.x);
    j=((2*j<=gridDimPME.y)?j:j-gridDimPME.y);
    // only lower half of k is used...
    if (flagBox) {
      kcomp=i*boxxx(kbox);
      k2=kcomp*kcomp;
      kcomp=i*boxxy(kbox)+j*boxyy(kbox);
      k2+=kcomp*kcomp;
      kcomp=i*boxxz(kbox)+j*boxyz(kbox)+k*boxzz(kbox);
      k2+=kcomp*kcomp;
    } else {
      kcomp=i*boxxx(kbox);
      k2=kcomp*kcomp;
      kcomp=j*boxyy(kbox);
      k2+=kcomp*kcomp;
      kcomp=k*boxzz(kbox);
      k2+=kcomp*kcomp;
    }
    factor=bGridPME[ijk];
    factor*=(((real)0.5)*kELECTRIC/((real)M_PI));
    factor*=Vinv*exp(-((real)M_PI)*((real)M_PI)*k2/(betaEwald*betaEwald));
    factor/=k2;
    factor=(ijk==0?0:factor);
    fourierGridPME[ijk].x*=factor;
    fourierGridPME[ijk].y*=factor;
  }
}

// getforce_ewald_gather_kernel<<<>>>(N,p->charge_d,prefactor,m->atomBlock_d,((int3*)p->gridDimPME)[0],p->potentialGridPME_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <bool flagBox,int order,typename box_type>
__global__ void getforce_ewald_gather_kernel_select(
  int atomCount,
  real *charge,
  int *atomBlock,
  int3 gridDimPME,
#ifdef USE_TEXTURE
  cudaTextureObject_t potentialGridPME,
#else
  real *potentialGridPME,
#endif
#ifdef USE_TEXTURE
  cudaTextureObject_t potentialGridPME_sele,
#else
  real *potentialGridPME_sele,
#endif
#ifdef USE_TEXTURE
  cudaTextureObject_t potentialGridPME_unsele,
#else
  real *potentialGridPME_unsele,
#endif
  real3 *position,
  int* selections,
  real3_f *force,
  real3_f *force_ss,
  real3_f *force_su,
  real3_f *force_uu,
  box_type kbox,
  real *lambda,
  real_f *lambdaForce_ss,
  real_f *lambdaForce_su,
  real_f *lambdaForce_uu,
  real_e* U_ss,
  real_e* U_su,
  real_e* U_uu,
  real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real q=0; // Avoid unitialized values for iAtom>=atomCount
  int b;
  real l=1;
  real3 xi; // position
  real3 fi;
  real3 fi_s;
  real3 fi_ss;
  real3 fi_su;
  real3 fi_uu;
  real u; // fractional coordinate remainder
  int3 u0; // index of grid point
  real Meven,Modd; // even and odd order B splines
  real3 density;
  real3 dDensity;
  real3 dIndex, dDIndex;
  int j,jx,jy,jz,jxs,jys,jzs;
  int3 index;
  int iThread=rectify_modulus(threadIdx.x,32); // threadIdx.x&31; // within warp
  int iAtom=i/8;
  int threadOfAtom=rectify_modulus(i,8); // iThread%order;
  real lEnergy=0;
  real lU_s=0;
  real lU_ss=0;
  real lU_su=0;
  real lU_uu=0;
  int selected=0;
  extern __shared__ real sEnergy[];

  u0=make_int3(0,0,0);

  if (iAtom<atomCount) {
    q=charge[iAtom];

    // Scaling
    b=atomBlock[iAtom];
    if (b) {
      l=lambda[b];
    }
    // q*=l; // do this scaling later in the kernel
    selected=selections[iAtom];

    xi=position[iAtom];

    // Get grid position
    if (flagBox) {
      u=gridDimPME.x*(xi.x*boxxx(kbox)+xi.y*boxxy(kbox)+xi.z*boxxz(kbox));
    } else {
      u=xi.x*gridDimPME.x*boxxx(kbox);
    }
    u0.x=(int)floor(u);
    u-=u0.x;
    if (flagBox) { // x-direction could be up to 1.25*Lx+diff out of box
      u0.x=nearby_modulus(u0.x,gridDimPME.x);
    }
    u0.x=nearby_modulus(u0.x,gridDimPME.x);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.x=Meven;
  dDensity.x=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  if (iAtom<atomCount) {
    if (flagBox) {
      u=gridDimPME.y*(xi.y*boxyy(kbox)+xi.z*boxyz(kbox));
    } else {
      u=xi.y*gridDimPME.y*boxyy(kbox);
    }
    u0.y=(int)floor(u);
    u-=u0.y;
    // y direction is only 0.5*Ly+diff out of triclinic box
    u0.y=nearby_modulus(u0.y,gridDimPME.y);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.y=Meven;
  dDensity.y=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  if (iAtom<atomCount) {
    if (flagBox) {
      u=gridDimPME.z*(xi.z*boxzz(kbox));
    } else {
      u=xi.z*gridDimPME.z*boxzz(kbox);
    }
    u0.z=(int)floor(u);
    u-=u0.z;
    // z direction is only diff out of triclinic box, same as orthogonal box
    u0.z=nearby_modulus(u0.z,gridDimPME.z);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.z=Meven;
  dDensity.z=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  // Everything is the same as the spread kernel up to this point
  jxs=(threadIdx.x&4)==4;
  jys=(threadIdx.x&2)==2;
  jzs=(threadIdx.x&1);
  fi_s.x=0;
  fi_s.y=0;
  fi_s.z=0;
  fi_ss.x=0;
  fi_ss.y=0;
  fi_ss.z=0;
  fi_su.x=0;
  fi_su.y=0;
  fi_su.z=0;
  fi_uu.x=0;
  fi_uu.y=0;
  fi_uu.z=0;
  fi.x=0;
  fi.y=0;
  fi.z=0;
  for (jx=0; jx<order/2; jx++) {
    dIndex.x=__shfl_sync(0xFFFFFFFF,density.x,2*jx+jxs,8);
    dDIndex.x=__shfl_sync(0xFFFFFFFF,dDensity.x,2*jx+jxs,8);
    index.x=over_modulus(u0.x+2*jx+jxs,gridDimPME.x);
    for (jy=0; jy<order/2; jy++) {
      dIndex.y=__shfl_sync(0xFFFFFFFF,density.y,2*jy+jys,8);
      dDIndex.y=__shfl_sync(0xFFFFFFFF,dDensity.y,2*jy+jys,8);
      index.y=index.x*gridDimPME.y+over_modulus(u0.y+2*jy+jys,gridDimPME.y);
      for (jz=0; jz<order/2; jz++) {
        dIndex.z=__shfl_sync(0xFFFFFFFF,density.z,2*jz+jzs,8);
        dDIndex.z=__shfl_sync(0xFFFFFFFF,dDensity.z,2*jz+jzs,8);
        index.z=index.y*gridDimPME.z+over_modulus(u0.z+2*jz+jzs,gridDimPME.z);
        if (iAtom<atomCount) {
          // Potential from selected atoms
#ifdef USE_TEXTURE
          real P_sele=tex1Dfetch<real>(potentialGridPME_sele,index.z);
#else
          real P_sele=potentialGridPME_sele[index.z];
#endif
          lU_s += P_sele*dIndex.x*dIndex.y*dIndex.z;
          fi_s.x+=P_sele*dDIndex.x*dIndex.y*dIndex.z;
          fi_s.y+=P_sele*dIndex.x*dDIndex.y*dIndex.z;
          fi_s.z+=P_sele*dIndex.x*dIndex.y*dDIndex.z;
          if(selected){
            lU_ss += P_sele*dIndex.x*dIndex.y*dIndex.z;
            fi_ss.x+=P_sele*dDIndex.x*dIndex.y*dIndex.z;
            fi_ss.y+=P_sele*dIndex.x*dDIndex.y*dIndex.z;
            fi_ss.z+=P_sele*dIndex.x*dIndex.y*dDIndex.z;
          } else {
            lU_su += P_sele*dIndex.x*dIndex.y*dIndex.z;
            fi_su.x+=P_sele*dDIndex.x*dIndex.y*dIndex.z;
            fi_su.y+=P_sele*dIndex.x*dDIndex.y*dIndex.z;
            fi_su.z+=P_sele*dIndex.x*dIndex.y*dDIndex.z;
          }
          // potential from unselected atoms
#ifdef USE_TEXTURE
          real P_unsele=tex1Dfetch<real>(potentialGridPME_unsele,index.z);
#else
          real P_unsele=potentialGridPME_unsele[index.z];
#endif
          if(selected){
            lU_su += P_unsele*dIndex.x*dIndex.y*dIndex.z;
            fi_su.x+=P_unsele*dDIndex.x*dIndex.y*dIndex.z;
            fi_su.y+=P_unsele*dIndex.x*dDIndex.y*dIndex.z;
            fi_su.z+=P_unsele*dIndex.x*dIndex.y*dDIndex.z;
          } else {
            lU_uu += P_unsele*dIndex.x*dIndex.y*dIndex.z;
            fi_uu.x+=P_unsele*dDIndex.x*dIndex.y*dIndex.z;
            fi_uu.y+=P_unsele*dIndex.x*dDIndex.y*dIndex.z;
            fi_uu.z+=P_unsele*dIndex.x*dIndex.y*dDIndex.z;
          }
          // Potential from all atoms
#ifdef USE_TEXTURE
          real P=tex1Dfetch<real>(potentialGridPME,index.z);
#else
          real P=potentialGridPME[index.z];
#endif
          fi.x+=P*dDIndex.x*dIndex.y*dIndex.z;
          fi.y+=P*dIndex.x*dDIndex.y*dIndex.z;
          fi.z+=P*dIndex.x*dIndex.y*dDIndex.z;
          if (b || energy) {
            lEnergy+=P*dIndex.x*dIndex.y*dIndex.z;
          }
          //if (abs(P - (P_sele + P_unsele)) > 1e-3){
          //  printf("Wrong Energy! P: %f, P calc: %f\n", P, P_sele + P_unsele);
          //}
        }
      }
    }
  }

  // Reductions sele
  fi_s.x+=__shfl_down_sync(0xFFFFFFFF,fi_s.x,1);
  fi_s.y+=__shfl_down_sync(0xFFFFFFFF,fi_s.y,1);
  fi_s.z+=__shfl_down_sync(0xFFFFFFFF,fi_s.z,1);
  lU_s+=  __shfl_down_sync(0xFFFFFFFF,lU_s,1);
  fi_s.x+=__shfl_down_sync(0xFFFFFFFF,fi_s.x,2);
  fi_s.y+=__shfl_down_sync(0xFFFFFFFF,fi_s.y,2);
  fi_s.z+=__shfl_down_sync(0xFFFFFFFF,fi_s.z,2);
  lU_s+=  __shfl_down_sync(0xFFFFFFFF,lU_s,2);
  fi_s.x+=__shfl_down_sync(0xFFFFFFFF,fi_s.x,4);
  fi_s.y+=__shfl_down_sync(0xFFFFFFFF,fi_s.y,4);
  fi_s.z+=__shfl_down_sync(0xFFFFFFFF,fi_s.z,4);
  lU_s+=  __shfl_down_sync(0xFFFFFFFF,lU_s,4);
  // Reductions ss
  fi_ss.x+=__shfl_down_sync(0xFFFFFFFF,fi_ss.x,1);
  fi_ss.y+=__shfl_down_sync(0xFFFFFFFF,fi_ss.y,1);
  fi_ss.z+=__shfl_down_sync(0xFFFFFFFF,fi_ss.z,1);
  lU_ss+=  __shfl_down_sync(0xFFFFFFFF,lU_ss,1);
  fi_ss.x+=__shfl_down_sync(0xFFFFFFFF,fi_ss.x,2);
  fi_ss.y+=__shfl_down_sync(0xFFFFFFFF,fi_ss.y,2);
  fi_ss.z+=__shfl_down_sync(0xFFFFFFFF,fi_ss.z,2);
  lU_ss+=  __shfl_down_sync(0xFFFFFFFF,lU_ss,2);
  fi_ss.x+=__shfl_down_sync(0xFFFFFFFF,fi_ss.x,4);
  fi_ss.y+=__shfl_down_sync(0xFFFFFFFF,fi_ss.y,4);
  fi_ss.z+=__shfl_down_sync(0xFFFFFFFF,fi_ss.z,4);
  lU_ss+=  __shfl_down_sync(0xFFFFFFFF,lU_ss,4);
  // Reductions su
  fi_su.x+=__shfl_down_sync(0xFFFFFFFF,fi_su.x,1);
  fi_su.y+=__shfl_down_sync(0xFFFFFFFF,fi_su.y,1);
  fi_su.z+=__shfl_down_sync(0xFFFFFFFF,fi_su.z,1);
  lU_su+=  __shfl_down_sync(0xFFFFFFFF,lU_su,1);
  fi_su.x+=__shfl_down_sync(0xFFFFFFFF,fi_su.x,2);
  fi_su.y+=__shfl_down_sync(0xFFFFFFFF,fi_su.y,2);
  fi_su.z+=__shfl_down_sync(0xFFFFFFFF,fi_su.z,2);
  lU_su+=  __shfl_down_sync(0xFFFFFFFF,lU_su,2);
  fi_su.x+=__shfl_down_sync(0xFFFFFFFF,fi_su.x,4);
  fi_su.y+=__shfl_down_sync(0xFFFFFFFF,fi_su.y,4);
  fi_su.z+=__shfl_down_sync(0xFFFFFFFF,fi_su.z,4);
  lU_su+=  __shfl_down_sync(0xFFFFFFFF,lU_su,4);
  // Reductions uu
  fi_uu.x+=__shfl_down_sync(0xFFFFFFFF,fi_uu.x,1);
  fi_uu.y+=__shfl_down_sync(0xFFFFFFFF,fi_uu.y,1);
  fi_uu.z+=__shfl_down_sync(0xFFFFFFFF,fi_uu.z,1);
  lU_uu+=  __shfl_down_sync(0xFFFFFFFF,lU_uu,1);
  fi_uu.x+=__shfl_down_sync(0xFFFFFFFF,fi_uu.x,2);
  fi_uu.y+=__shfl_down_sync(0xFFFFFFFF,fi_uu.y,2);
  fi_uu.z+=__shfl_down_sync(0xFFFFFFFF,fi_uu.z,2);
  lU_uu+=  __shfl_down_sync(0xFFFFFFFF,lU_uu,2);
  fi_uu.x+=__shfl_down_sync(0xFFFFFFFF,fi_uu.x,4);
  fi_uu.y+=__shfl_down_sync(0xFFFFFFFF,fi_uu.y,4);
  fi_uu.z+=__shfl_down_sync(0xFFFFFFFF,fi_uu.z,4);
  lU_uu+=  __shfl_down_sync(0xFFFFFFFF,lU_uu,4);
  // Reductions total
  fi.x+=__shfl_down_sync(0xFFFFFFFF,fi.x,1);
  fi.y+=__shfl_down_sync(0xFFFFFFFF,fi.y,1);
  fi.z+=__shfl_down_sync(0xFFFFFFFF,fi.z,1);
  lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,1);
  fi.x+=__shfl_down_sync(0xFFFFFFFF,fi.x,2);
  fi.y+=__shfl_down_sync(0xFFFFFFFF,fi.y,2);
  fi.z+=__shfl_down_sync(0xFFFFFFFF,fi.z,2);
  lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,2);
  fi.x+=__shfl_down_sync(0xFFFFFFFF,fi.x,4);
  fi.y+=__shfl_down_sync(0xFFFFFFFF,fi.y,4);
  fi.z+=__shfl_down_sync(0xFFFFFFFF,fi.z,4);
  lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,4);
  // Reduction only correct for threadOfAtom==0

  // Lambda force
  if (iAtom<atomCount) {
    if (b && threadOfAtom==0) {
      //atomicAdd(&lambdaForce[b],2*q*lEnergy);
      atomicAdd(&lambdaForce_ss[b],2*q*(lU_ss+lU_su)); // lEnergy = lU_ss + (lEnergy-lU_ss)
      // Alchemical atoms have only ss interactions
    }
  }

  // Spatial force
  if (iAtom<atomCount) {
    if (flagBox) {
      // ss
      fi_s.z*=2*l*q*gridDimPME.z;
      fi_s.y*=2*l*q*gridDimPME.y;
      fi_s.x*=2*l*q*gridDimPME.x;
      fi_s.z=fi_s.x*boxxz(kbox)+fi_s.y*boxyz(kbox)+fi_s.z*boxzz(kbox);
      fi_s.y=fi_s.x*boxxy(kbox)+fi_s.y*boxyy(kbox);
      fi_s.x=fi_s.x*boxxx(kbox);
      // ss
      fi_ss.z*=2*l*q*gridDimPME.z;
      fi_ss.y*=2*l*q*gridDimPME.y;
      fi_ss.x*=2*l*q*gridDimPME.x;
      fi_ss.z=fi_ss.x*boxxz(kbox)+fi_ss.y*boxyz(kbox)+fi_ss.z*boxzz(kbox);
      fi_ss.y=fi_ss.x*boxxy(kbox)+fi_ss.y*boxyy(kbox);
      fi_ss.x=fi_ss.x*boxxx(kbox);
      // su
      fi_su.z*=2*l*q*gridDimPME.z;
      fi_su.y*=2*l*q*gridDimPME.y;
      fi_su.x*=2*l*q*gridDimPME.x;
      fi_su.z=fi_su.x*boxxz(kbox)+fi_su.y*boxyz(kbox)+fi_su.z*boxzz(kbox);
      fi_su.y=fi_su.x*boxxy(kbox)+fi_su.y*boxyy(kbox);
      fi_su.x=fi_su.x*boxxx(kbox);
      // uu
      fi_uu.z*=2*l*q*gridDimPME.z;
      fi_uu.y*=2*l*q*gridDimPME.y;
      fi_uu.x*=2*l*q*gridDimPME.x;
      fi_uu.z=fi_uu.x*boxxz(kbox)+fi_uu.y*boxyz(kbox)+fi_uu.z*boxzz(kbox);
      fi_uu.y=fi_uu.x*boxxy(kbox)+fi_uu.y*boxyy(kbox);
      fi_uu.x=fi_uu.x*boxxx(kbox);
      // total
      fi.z*=2*l*q*gridDimPME.z;
      fi.y*=2*l*q*gridDimPME.y;
      fi.x*=2*l*q*gridDimPME.x;
      fi.z=fi.x*boxxz(kbox)+fi.y*boxyz(kbox)+fi.z*boxzz(kbox);
      fi.y=fi.x*boxxy(kbox)+fi.y*boxyy(kbox);
      fi.x=fi.x*boxxx(kbox);
    } else {
      // sele
      fi_s.x*=2*l*q*gridDimPME.x*boxxx(kbox);
      fi_s.y*=2*l*q*gridDimPME.y*boxyy(kbox);
      fi_s.z*=2*l*q*gridDimPME.z*boxzz(kbox);
      // ss
      fi_ss.x*=2*l*q*gridDimPME.x*boxxx(kbox);
      fi_ss.y*=2*l*q*gridDimPME.y*boxyy(kbox);
      fi_ss.z*=2*l*q*gridDimPME.z*boxzz(kbox);
      // su
      fi_su.x*=2*l*q*gridDimPME.x*boxxx(kbox);
      fi_su.y*=2*l*q*gridDimPME.y*boxyy(kbox);
      fi_su.z*=2*l*q*gridDimPME.z*boxzz(kbox);
      // uu
      fi_uu.x*=2*l*q*gridDimPME.x*boxxx(kbox);
      fi_uu.y*=2*l*q*gridDimPME.y*boxyy(kbox);
      fi_uu.z*=2*l*q*gridDimPME.z*boxzz(kbox);
      // total
      fi.x*=2*l*q*gridDimPME.x*boxxx(kbox);
      fi.y*=2*l*q*gridDimPME.y*boxyy(kbox);
      fi.z*=2*l*q*gridDimPME.z*boxzz(kbox);
    }
    if (threadOfAtom==0) {
      /* 
      if(selected){
        at_real3_inc(&force_ss[iAtom], fi_s);
        at_real3_inc(&force_su[iAtom], real3_sub(fi, fi_s));
      } else {
        at_real3_inc(&force_su[iAtom], fi_s);
      }
      */
      // dU_ss = dU_sele for selected
      real dx = real3_mag<real, real3>(real3_sub(fi_s, fi_ss));
      if(selected && abs(dx) > 1e-5){
        printf("dx: %f\n", dx);
      }
      // dU_su = dU_total - dU_sele for selected
      real3 tmp = real3_sub(fi, fi_s);
      dx = real3_mag<real, real3>(real3_sub(tmp, fi_su));
      if(selected && abs(dx) > 1e-5){
        printf("dx: %f\n", dx);
      }
      // dU_su = dU_sele for unselected
      dx = real3_mag<real, real3>(real3_sub(fi_s, fi_su));
      if(!selected && abs(dx) > 1e-5){
        printf("dx: %f\n", dx);
      }
      // dU_uu = dU_tot - dU_sele for unselected
      tmp = real3_sub(fi, fi_s);
      dx = real3_mag<real, real3>(real3_sub(tmp, fi_uu));
      if(!selected && abs(dx) > 1e-5){
        printf("dx: %f\n", dx);
      }
      //at_real3_inc(&force[iAtom], fi);
      if(selected){
        at_real3_inc(&force_ss[iAtom], fi_ss);
        at_real3_inc(&force_su[iAtom], real3_sub(fi, fi_ss));
      } else {
        at_real3_inc(&force_su[iAtom], fi_ss);
        at_real3_inc(&force_uu[iAtom], real3_sub(fi, fi_ss));
      }
      //at_real3_inc(&force_ss[iAtom], fi_ss);
      //at_real3_inc(&force_su[iAtom], fi_su);
      //at_real3_inc(&force_uu[iAtom], fi_uu);
      /*
      if (selected) {
        at_real3_inc(&force_ss[iAtom], fi_ss);
        at_real3_inc(&force_su[iAtom], fi_su);
        at_real3_inc(&force_uu[iAtom], fi_uu);
      } else {
        at_real3_inc(&force_su[iAtom], fi_su);
        at_real3_inc(&force_uu[iAtom], fi_uu);
      }
        */
    }
  }

  // Energy, if requested
  if (energy) {
    if (threadOfAtom==0) {
      lEnergy*=l*q;
      lU_s*=l*q;
      /*
      if(selected){
        lU_ss = lU_s;
        lU_su = lEnergy - lU_s;
      } else {
        lU_su = lU_s;
        lU_uu = lEnergy - lU_s;
      }
      */
      lU_ss*=l*q;
      lU_su*=l*q;
      lU_uu*=l*q;
      if(selected){
        real lU_ss_tmp = lU_s;
        if(abs(lU_ss_tmp - lU_ss) > 1e-5){
          printf("Wrong ss!lU_ss_true: %f, wrong: %f\n", lU_ss_tmp, lU_ss);
        } 
        real lU_su_tmp = lEnergy - lU_s;
        if(abs(lU_su_tmp - lU_su) > 1e-5){
          printf("Wrong su! lU_su_true: %f, wrong: %f\n", lU_su_tmp, lU_su);
        } 
      } else {
        real lU_su_tmp = lU_s;
        if(abs(lU_su_tmp - lU_su) > 1e-5){
          printf("Wrong su! lU_su_true: %f, wrong: %f\n", lU_su_tmp, lU_su);
        } 
        real lU_uu_tmp = lEnergy - lU_s;
        if(abs(lU_uu_tmp - lU_uu) > 1e-5){
          printf("Wrong su! lU_su_true: %f, wrong: %f\n", lU_uu_tmp, lU_uu);
        } 
      }
    } else {
      lEnergy=0;
      lU_ss=0;
      lU_su=0;
      lU_uu=0;
    }
    // if (!isfinite(lEnergy)) { // code to look for causes of nan in lEnergy
    //   printf("Error: lEnergy is not finite for atom %d\n",iAtom);
    // }
    // note, had to use reduction without shared memory here for a while to avoid errors. That seems to have passed.
    //real_sum_reduce(lEnergy,sEnergy,energy);
    real_sum_reduce(lU_ss,sEnergy,U_ss);
    real_sum_reduce(lU_su,sEnergy,U_su);
    real_sum_reduce(lU_uu,sEnergy,U_uu);
    // real_sum_reduce(lEnergy,energy);
  }
}

template <bool flagBox,int order,typename box_type>
void getforce_ewaldTT_select(System *system,box_type kbox,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  Run *r=system->run;
  int N=p->atomCount;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->usePME==false) return;
  if (r->calcTermFlag[eenbrecip]==false) return;

  if (calcEnergy) {
    shMem=BLNB*sizeof(real)/32;
    pEnergy=s->energy_d+eenbrecip;
  }

  int* selections = NULL;
  if(system->enhanced && system->enhanced->special_elec){
    selections=system->enhanced->atom_selection_primary_d;
  }

  // Potential due to selected atoms
  cudaMemsetAsync(p->chargeGridPME_d,0,p->gridDimPME[0]*p->gridDimPME[1]*p->gridDimPME[2]*sizeof(myCufftReal),r->nbrecipStream);
  int spreadGatherBlocks=(N + BLNB/8 - 1)/(BLNB/8);
  getforce_ewald_spread_kernel_select<flagBox,order><<<spreadGatherBlocks,BLNB,0,r->nbrecipStream>>>(
    N,p->charge_d,m->atomBlock_d,
    (real3*)s->position_fd,kbox,
    s->lambda_fd,selections, true, // only spread selected
    ((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
  myCufftExecR2C(p->planFFTPME,p->chargeGridPME_d,p->fourierGridPME_d);
  dim3 blockCount((p->gridDimPME[0]+8-1)/8,(p->gridDimPME[1]+8-1)/8,(p->gridDimPME[2]/2+1+8-1)/8);
  dim3 blockSize(8,8,8);
  getforce_ewald_convolution_kernel_select<flagBox><<<blockCount,blockSize,0,r->nbrecipStream>>>(((int3*)p->gridDimPME)[0],p->fourierGridPME_d,p->bGridPME_d,system->run->betaEwald,kbox);
  myCufftExecC2R(p->planIFFTPME,p->fourierGridPME_d,p->potentialGridPME_sele_d);
  // Potential due to unselected atoms
  cudaMemsetAsync(p->chargeGridPME_d,0,p->gridDimPME[0]*p->gridDimPME[1]*p->gridDimPME[2]*sizeof(myCufftReal),r->nbrecipStream);
  getforce_ewald_spread_kernel_select<flagBox,order><<<spreadGatherBlocks,BLNB,0,r->nbrecipStream>>>(
    N,p->charge_d,m->atomBlock_d,
    (real3*)s->position_fd,kbox,
    s->lambda_fd,selections, false, // only spread unselected
    ((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
  myCufftExecR2C(p->planFFTPME,p->chargeGridPME_d,p->fourierGridPME_d);
  getforce_ewald_convolution_kernel_select<flagBox><<<blockCount,blockSize,0,r->nbrecipStream>>>(((int3*)p->gridDimPME)[0],p->fourierGridPME_d,p->bGridPME_d,system->run->betaEwald,kbox);
  myCufftExecC2R(p->planIFFTPME,p->fourierGridPME_d,p->potentialGridPME_unsele_d);

  // Compute forces from U_sele & U_unsele
  getforce_ewald_gather_kernel_select<flagBox,order><<<spreadGatherBlocks,BLNB,shMem,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,((int3*)p->gridDimPME)[0],
#ifdef USE_TEXTURE
    p->potentialGridPME_tex,
#else
    p->potentialGridPME_d,
#endif
#ifdef USE_TEXTURE
    p->potentialGridPME_sele_tex,
#else
    p->potentialGridPME_sele_d,
#endif
#ifdef USE_TEXTURE
    p->potentialGridPME_unsele_tex,
#else
    p->potentialGridPME_unsele_d,
#endif
    (real3*)s->position_fd,
    selections,
    (real3_f*)s->force_d,
    (real3_f*)s->dU_ss_spaceBuffer3_d,(real3_f*)s->dU_su_spaceBuffer3_d,(real3_f*)s->dU_uu_spaceBuffer3_d,
    kbox,s->lambda_fd,
    s->dU_ss_lambdaForce_d,s->dU_su_lambdaForce_d,s->dU_uu_lambdaForce_d,
    s->U_ss_d, s->U_su_d,s->U_uu_d,
    pEnergy);
}

template <bool flagBox,typename box_type>
void getforce_ewaldT_select(System *system,box_type kbox,bool calcEnergy)
{
  if (system->run->orderEwald==4) {
    getforce_ewaldTT_select<flagBox,4>(system,kbox,calcEnergy);
  } else if (system->run->orderEwald==6) {
    getforce_ewaldTT_select<flagBox,6>(system,kbox,calcEnergy);
  } else if (system->run->orderEwald==8) {
    getforce_ewaldTT_select<flagBox,8>(system,kbox,calcEnergy);
  }
}

void getforce_ewald_select(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_ewaldT_select<true>(system,system->state->kTricBox_f,calcEnergy);
  } else {
    getforce_ewaldT_select<false>(system,system->state->kOrthBox_f,calcEnergy);
  }
}
