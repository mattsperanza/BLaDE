//
// Created by matthew-speranza on 1/16/25.
//
#include <cassert>
#include <cuda_runtime.h>

#include "enhanced/enhanced.h"
#include "enhanced/osrw/osrw.h"
#include "system/system.h"
#include "system/state.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "main/defines.h"
#include "main/real3.h"


#ifdef DOUBLE
#define fasterfc erfc
#else
// Directly from CHARMM source code, charmm/source/domdec_gpu/gpu_utils.h
// #warning "From CHARMM, not fully compatible"
static __forceinline__ __device__ float __internal_fmad(float a, float b, float c)
{
#if __CUDA_ARCH__ >= 200
  return __fmaf_rn (a, b, c);
#else // __CUDA_ARCH__ >= 200
  return a * b + c;
#endif // __CUDA_ARCH__ >= 200
}

// Following inline functions are copied from PMEMD CUDA implementation.
// Credit goes to:
/*             Scott Le Grand (NVIDIA)             */
/*               Duncan Poole (NVIDIA)             */
/*                Ross Walker (SDSC)               */
//
// Faster ERFC approximation courtesy of Norbert Juffa. NVIDIA Corporation
static __forceinline__ __device__ float fasterfc(float a)
{
  /* approximate log(erfc(a)) with rel. error < 7e-9 */
  float t, x = a;
  t =                       (float)-1.6488499458192755E-006;
  t = __internal_fmad(t, x, (float)2.9524665006554534E-005);
  t = __internal_fmad(t, x, (float)-2.3341951153749626E-004);
  t = __internal_fmad(t, x, (float)1.0424943374047289E-003);
  t = __internal_fmad(t, x, (float)-2.5501426008983853E-003);
  t = __internal_fmad(t, x, (float)3.1979939710877236E-004);
  t = __internal_fmad(t, x, (float)2.7605379075746249E-002);
  t = __internal_fmad(t, x, (float)-1.4827402067461906E-001);
  t = __internal_fmad(t, x, (float)-9.1844764013203406E-001);
  t = __internal_fmad(t, x, (float)-1.6279070384382459E+000);
  t = t * x;
  return exp2f(t);
}
#endif

#define WARPSPERBLOCK 2
__host__ __device__ inline
bool check_proximity(DomdecBlockVolume a,real3 b,real c2)
{
  real bufferA,bufferB,buffer2;

  bufferB=b.x-a.max.x; // Distance one way
  bufferA=a.min.x-b.x; // Distance the other way
  bufferA=(bufferA>bufferB?bufferA:bufferB);
  bufferA=(bufferA<0?0:bufferA);
  buffer2=bufferA*bufferA;

  bufferB=b.y-a.max.y; // Distance one way
  bufferA=a.min.y-b.y; // Distance the other way
  bufferA=(bufferA>bufferB?bufferA:bufferB);
  bufferA=(bufferA<0?0:bufferA);
  buffer2+=bufferA*bufferA;

  bufferB=b.z-a.max.z; // Distance one way
  bufferA=a.min.z-b.z; // Distance the other way
  bufferA=(bufferA>bufferB?bufferA:bufferB);
  bufferA=(bufferA<0?0:bufferA);
  buffer2+=bufferA*bufferA;

  return buffer2<=c2;
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
__global__ void getforce_nbdirect_oss_kernel(
  int startBlock,
  int endBlock,
  int maxPartnersPerBlock,
  const int* __restrict__ blockBounds,
  const int* __restrict__ blockPartnerCount,
  const struct DomdecBlockPartners* __restrict__ blockPartners,
  const struct DomdecBlockVolume* __restrict__ blockVolume,
  const struct NbondPotential* __restrict__ nbonds,
  int vdwParameterCount,
#ifdef USE_TEXTURE
  cudaTextureObject_t vdwParameters,
#else
  const struct VdwPotential* __restrict__ vdwParameters,
#endif
  const int* __restrict__ blockExcls,
  struct Cutoffs cutoffs, real scrVdw, real scrElec,
  const real3* __restrict__ position,
  real3_f* __restrict__ force,
  box_type box,
  const real* __restrict__ lambda,
  real_f* __restrict__ lambdaForce,
  real* __restrict__ dGdF)
{
// NYI - maybe energy should be a double
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iWarp=i>>5; // i/32
  int iThread=i-32*iWarp;
  int iBlock=(iWarp>>WARPSPERBLOCK)+startBlock;
  int jBlock;
  int iCount,jCount;
  __shared__ struct DomdecBlockVolume iBlockVolume;
  __shared__ int jnext;
  int ii,jj;
  int j,jmax;
  int jtmp;
  real r,rinv;
  char4 shift;
  real3 dr;
  NbondPotential inp,jnp;
  real jtmpnp_q;
  int jtmpnp_typeIdx;
  real fij_ost;
  real d2U_drij_dli, d2U_drij_dlj, d2U_dli_dlj, d2U_dli2, d2U_dlj2;
  real3 xi,xj,xjtmp;
  // OST Forces (added into normal force array)
  real3 fi_ost,fj_ost,fjtmp_ost;
  real fli_ost,flj_ost,fljtmp_ost;
  int bi,bj,bjtmp,bi_any,bj_any;
  real li,lj,ljtmp,lixljtmp;
  real rEff; // Soft core stuff
  int exclAddress, exclMask;
  real dGdFi, dGdFj, dGdFjtmp;

  if (iBlock<endBlock && threadIdx.x==0) iBlockVolume=blockVolume[iBlock];
  if (iBlock<endBlock && threadIdx.x==0) jnext=0;
  __syncthreads();
  if (iBlock<endBlock) {
    ii=blockBounds[iBlock];
    iCount=blockBounds[iBlock+1]-ii;
    ii+=(iThread);
    if ((iThread)<iCount) {
      // inp=nbonds[ii];
      // xi=position[ii];
      inp=nbonds[32*iBlock+iThread];
      xi=position[32*iBlock+iThread];
      bi=inp.siteBlock;
      li=1;
      dGdFi=0;
      if (bi) {
        li=lambda[0xFFFF & bi];
        dGdFi=dGdF[0xFFFF & bi];
      }
    }
    // Check if bi has alchemical atoms via warp reduction
    bi_any = __any_sync(0xFFFFFFFF, bi);
    // iBlockVolume=blockVolume[iBlock];

    fi_ost=real3_reset<real3>();
    fli_ost=0;

    // used i/32 instead of iBlock to shift to beginning of array
    jmax=blockPartnerCount[iWarp>>WARPSPERBLOCK];
    if (iThread==0) j=atomicInc((unsigned int*)(&jnext),0xFFFFFFFF);
    j=__shfl_sync(0xFFFFFFFF,j,0);
    // for (j=rectify_modulus(iWarp,1<<WARPSPERBLOCK); j<jmax; j+=(1<<WARPSPERBLOCK))
    for (; j<jmax;) {
      jBlock=blockPartners[maxPartnersPerBlock*(iWarp>>WARPSPERBLOCK)+j].jBlock;
      shift=blockPartners[maxPartnersPerBlock*(iWarp>>WARPSPERBLOCK)+j].shift;
      // boxShift.x*=box.x;
      // boxShift.y*=box.y;
      // boxShift.z*=box.z;
      exclAddress=blockPartners[maxPartnersPerBlock*(iWarp>>WARPSPERBLOCK)+j].exclAddress;
      if (exclAddress==-1) {
        exclMask=0xFFFFFFFF;
      } else {
        exclMask=blockExcls[32*exclAddress+(iThread)];
      }
      if (iBlock==jBlock && shift.x==0 && shift.y==0 && shift.z==0) {
        exclMask>>=(iThread+1);
        exclMask<<=(iThread+1);
      }
      jj=blockBounds[jBlock];
      jCount=blockBounds[jBlock+1]-jj;
      jj+=(iThread);
      if ((iThread)<jCount) {
        // jnp=nbonds[jj];
        // xj=position[jj];
        jnp=nbonds[32*jBlock+iThread];
        xj=position[32*jBlock+iThread];
        // // real3_inc(&xj,boxShift);
        // xj.x+=boxShift.x;
        // xj.y+=boxShift.y;
        // xj.z+=boxShift.z;
        if (flagBox) {
          xj.x+=shift.z*boxzx(box)+shift.y*boxyx(box)+shift.x*boxxx(box);
          xj.y+=shift.z*boxzy(box)+shift.y*boxyy(box);
          xj.z+=shift.z*boxzz(box);
        } else {
          xj.x+=shift.x*boxxx(box);
          xj.y+=shift.y*boxyy(box);
          xj.z+=shift.z*boxzz(box);
        }
        bj=jnp.siteBlock;
        lj=1;
        dGdFj=0;
        if (bj) {
          lj=lambda[0xFFFF & bj];
          dGdFj=dGdF[0xFFFF & bj];
        }
      }
      // Sum warp bj's to check # alchemical atoms via warp reduction
      bj_any = __any_sync(0xFFFFFFFF, bj);
      // If bi && bj are both 0, then we can skip this block-block interaction
      bool jFlag=check_proximity(iBlockVolume,xj,cutoffs.rCut*cutoffs.rCut);

      fj_ost=real3_reset<real3>();
      flj_ost=0;

      // If bi && bj are both 0, then we can skip this block-block interaction
      if (bi_any != 0 || bj_any != 0) {
        for (jtmp=0; jtmp<jCount; jtmp++) {
          if (__shfl_sync(0xFFFFFFFF,jFlag,jtmp)) {
            jtmpnp_q=__shfl_sync(0xFFFFFFFF,jnp.q,jtmp);
            jtmpnp_typeIdx=__shfl_sync(0xFFFFFFFF,jnp.typeIdx,jtmp);
            xjtmp.x=__shfl_sync(0xFFFFFFFF,xj.x,jtmp);
            xjtmp.y=__shfl_sync(0xFFFFFFFF,xj.y,jtmp);
            xjtmp.z=__shfl_sync(0xFFFFFFFF,xj.z,jtmp);
            bjtmp=__shfl_sync(0xFFFFFFFF,bj,jtmp);
            ljtmp=__shfl_sync(0xFFFFFFFF,lj,jtmp);
            dGdFjtmp=__shfl_sync(0xFFFFFFFF,dGdFj,jtmp);

            fjtmp_ost=real3_reset<real3>();
            fljtmp_ost=0;
            if (iThread<iCount && ((1<<jtmp)&exclMask)) {
#ifdef USE_TEXTURE
              struct VdwPotential vdwp;
              ((real2*)(&vdwp))[0]=tex1Dfetch<real2>(vdwParameters,inp.typeIdx*vdwParameterCount+jtmpnp_typeIdx);
#else
              struct VdwPotential vdwp=vdwParameters[inp.typeIdx*vdwParameterCount+jtmpnp_typeIdx];
#endif

              // Geometry
              dr=real3_sub(xi,xjtmp);
              r=real3_mag<real>(dr);

              // Skip env-env interactions 
              if (r<cutoffs.rCut && (bi || bjtmp)) {
                rEff=r;
                // Lambda Scaling
                real dlixlj_dli, dlixlj_dlj, d2lixlj_dli_dlj;
                if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) { // same site (m == n)
                  if (bi==bjtmp) { // intra-sub (i == j, alc-alc or env-env)
                    lixljtmp = li;
                    dlixlj_dli = bi ? 1 : 0;
                    dlixlj_dlj = 0; // prevent over-count
                    dGdFjtmp = 0;
                    d2lixlj_dli_dlj = 0;
                  } else { // intra-site (i != j, alc-alc or env-env)
                    lixljtmp = 0; // zero contribution from env-env anyway
                    dlixlj_dli = 0;
                    dlixlj_dlj = 0;
                    d2lixlj_dli_dlj = 0;
                  }
                } else { // inter-site (m != n)
                  lixljtmp = li*ljtmp;
                  dlixlj_dli = ljtmp;
                  dlixlj_dlj = li;
                  d2lixlj_dli_dlj = bi && bjtmp ? 1 : 0;
                }
              
                // Electrostatics
                real t[cr_count] = {0};  // Softcore c.r. terms
                t[drijp_drij] = 1;
                if (useSoftCore && (bi || bjtmp)) {
                  set_soft(r, scrElec, 
                    lixljtmp, dlixlj_dli, dlixlj_dlj, d2lixlj_dli_dlj,
                    t, &rEff);
                }
                real rinv=1/rEff;
                if(usePME){
                  real br=cutoffs.betaEwald*rEff;
                  real br2 = br*br;
                  real erfcrinv=fasterfc(br)*rinv;
                  real qxq = kELECTRIC*inp.q*jtmpnp_q;
                  real rinv3 = rinv*rinv*rinv;
                  real sqrt_1_PI = (real)0.564189583547756;
                  real exp_mbr2 = expf(-br2);
                  real U_dir = qxq*erfcrinv;
                  real dU_drijp_tmp = -qxq*rinv*(erfcrinv+((real)1.128379167095513)*cutoffs.betaEwald*exp_mbr2);
                  real dU_drijp = lixljtmp * dU_drijp_tmp;
                  real d2U_drijp2 = lixljtmp*2*qxq*rinv3*(2*sqrt_1_PI*br*exp_mbr2*(br2 + 1) + erfcrinv/rinv);
                  real d2U_drijp_dli = dlixlj_dli*dU_drijp_tmp;
                  real d2U_drijp_dlj = dlixlj_dlj*dU_drijp_tmp;
                  real d2Up_dli_dlj = d2lixlj_dli_dlj*U_dir;
                  // Accumulate forces w/ chain rule terms
                  d2U_drij_dli = dU_drijp*t[d2rijp_drij_dli] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_drij];
                  d2U_drij_dlj = dU_drijp*t[d2rijp_drij_dlj] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_drij];
                  d2U_dli_dlj = d2Up_dli_dlj + d2U_drijp_dlj*t[drijp_dli] + dU_drijp*t[d2rijp_dli_dlj] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dlj];
                  d2U_dli2 = d2U_drijp_dli*t[drijp_dli] + dU_drijp*t[d2rijp_dli2] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dli];
                  d2U_dlj2 = d2U_drijp_dlj*t[drijp_dlj] + dU_drijp*t[d2rijp_dlj2] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_dlj];
                }

                // Van der Waals
                if (abs(scrVdw - scrElec) > 1e-4){ // update softcore cr terms to reflect scrVdw
                  for(int i = 0; i < cr_count; i++){ t[i] = 0; }
                  t[drijp_drij] = 1;
                  if (useSoftCore && (bi || bjtmp)) {
                     set_soft(r, scrVdw, 
                      lixljtmp, dlixlj_dli, dlixlj_dlj, d2lixlj_dli_dlj,
                      t, &rEff);
                  }
                  rinv=1/rEff;
                }
                real rinv3=rinv*rinv*rinv;
                real rinv6=rinv3*rinv3;
                real rCut3=cutoffs.rCut*cutoffs.rCut*cutoffs.rCut;
                real rSwitch3=cutoffs.rSwitch*cutoffs.rSwitch*cutoffs.rSwitch;
                if (rEff<cutoffs.rSwitch) {
                  real dv6=usevdWSwitch?0:1/(rCut3*rSwitch3);
                  real U_dir = vdwp.c12*(rinv6*rinv6-dv6*dv6)-vdwp.c6*(rinv6-dv6);
                  real dU_drijp_tmp = rinv*rinv6*(-12.0*vdwp.c12*rinv6+6*vdwp.c6);
                  real dU_drijp = lixljtmp*dU_drijp_tmp;
                  real d2U_drijp2 = lixljtmp*6*rinv6*rinv*rinv*(26.0*vdwp.c12*rinv6 - 7.0*vdwp.c6);
                  real d2U_drijp_dli = dlixlj_dli*dU_drijp_tmp;
                  real d2U_drijp_dlj = dlixlj_dlj*dU_drijp_tmp;
                  real d2Up_dli_dlj = d2lixlj_dli_dlj*U_dir;
                  // Accumulate with chain rule terms
                  d2U_drij_dli += dU_drijp*t[d2rijp_drij_dli] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_drij];
                  d2U_drij_dlj += dU_drijp*t[d2rijp_drij_dlj] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_drij];
                  d2U_dli_dlj += d2Up_dli_dlj + d2U_drijp_dlj*t[drijp_dli] + dU_drijp*t[d2rijp_dli_dlj] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dlj];
                  d2U_dli2 += d2U_drijp_dli*t[drijp_dli] + dU_drijp*t[d2rijp_dli2] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dli];
                  d2U_dlj2 += d2U_drijp_dlj*t[drijp_dlj] + dU_drijp*t[d2rijp_dlj2] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_dlj];
                }
                else {
                  if ( !usevdWSwitch ) { // Force Switch
                    real k6=rCut3/(rCut3-rSwitch3);
                    real k12=rCut3*rCut3/(rCut3*rCut3-rSwitch3*rSwitch3);
                    real rCutinv3=1/rCut3;
                    real fij_tmp=(((real)6.0)*vdwp.c6*k6*(rinv3-rCutinv3)*rinv3-((real)12.0)*vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*rinv6)*rinv;
                    real eij_tmp=vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*(rinv6-rCutinv3*rCutinv3)-vdwp.c6*k6*(rinv3-rCutinv3)*(rinv3-rCutinv3);
                    // not soft-cored:
                    d2U_drij_dli += dlixlj_dli*fij_tmp;
                    d2U_drij_dlj += dlixlj_dlj*fij_tmp;
                    d2U_dli_dlj += d2lixlj_dli_dlj*eij_tmp;
                  }
                  else { // Potential Switch
                    real c2ofnb=cutoffs.rCut*cutoffs.rCut;
                    real c2onnb=cutoffs.rSwitch*cutoffs.rSwitch;
                    real rul3=(c2ofnb-c2onnb)*(c2ofnb-c2onnb)*(c2ofnb-c2onnb);
                    real rul12 = ((real)12.0)/rul3;
                    real rijl = c2onnb - rEff * rEff;
                    real riju = c2ofnb - rEff * rEff;
                    real fsw = riju*riju*(riju-((real)3.0)*rijl)/rul3;
                    real dfsw = rijl*riju*rul12;
                    real fij_tmp=fsw*(((real)6.0)*vdwp.c6-((real)12.0)*vdwp.c12*rinv6)*rinv6*rinv\
                      +dfsw*(vdwp.c12*rinv6-vdwp.c6)*rinv6;
                    real eij_tmp=fsw*(vdwp.c12*rinv6-vdwp.c6)*rinv6;
                    // not soft-cored:
                    d2U_drij_dli += dlixlj_dli*fij_tmp;
                    d2U_drij_dlj += dlixlj_dlj*fij_tmp;
                    d2U_dli_dlj += d2lixlj_dli_dlj*eij_tmp;
                  }
                }

                // Interaction
                fij_ost = dGdFi*d2U_drij_dli + dGdFjtmp*d2U_drij_dlj;
                fli_ost += dGdFi*d2U_dli2 + dGdFjtmp*d2U_dli_dlj;
                if(bjtmp && (bi&0xFFFF0000)!=(bjtmp&0xFFFF0000)){
                  fljtmp_ost += dGdFi*d2U_dli_dlj + dGdFjtmp*d2U_dlj2;
                }

                // Accumulate OST spatial forces
                real3_scaleinc(&fi_ost, fij_ost/r,dr);
                fjtmp_ost=real3_scale<real3>(-fij_ost/r,dr);
              }
            }
            __syncwarp();
            // OST Forces
            fjtmp_ost.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.x,1);
            fjtmp_ost.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.x,2);
            fjtmp_ost.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.x,4);
            fjtmp_ost.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.x,8);
            fjtmp_ost.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.x,16);
            if (iThread==jtmp) fj_ost.x=fjtmp_ost.x;
            fjtmp_ost.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.y,1);
            fjtmp_ost.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.y,2);
            fjtmp_ost.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.y,4);
            fjtmp_ost.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.y,8);
            fjtmp_ost.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.y,16);
            if (iThread==jtmp) fj_ost.y=fjtmp_ost.y;
            fjtmp_ost.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.z,1);
            fjtmp_ost.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.z,2);
            fjtmp_ost.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.z,4);
            fjtmp_ost.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.z,8);
            fjtmp_ost.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp_ost.z,16);
            if (iThread==jtmp) fj_ost.z=fjtmp_ost.z;
            if (bjtmp) {
              fljtmp_ost+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ost,1);
              fljtmp_ost+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ost,2);
              fljtmp_ost+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ost,4);
              fljtmp_ost+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ost,8);
              fljtmp_ost+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ost,16);
              if (iThread==jtmp) flj_ost=fljtmp_ost;
            }
          }
        }
      }

      __syncwarp();
      if ((iThread)<jCount) {
        if (bj) {
          atomicAdd(&lambdaForce[0xFFFF & bj],flj_ost);
        }
        at_real3_inc(&force[32*jBlock+iThread],fj_ost);
      }
      if (iThread==0) j=atomicInc((unsigned int*)(&jnext),0xFFFFFFFF);
      j=__shfl_sync(0xFFFFFFFF,j,0);
    }
    __syncwarp();
    if ((iThread)<iCount) {
      if (bi) {
        atomicAdd(&lambdaForce[0xFFFF & bi],fli_ost);
      }
      at_real3_inc(&force[32*iBlock+iThread],fi_ost);
    }
  }
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
void getforce_nbdirect_ossTTTT(System *system,box_type box)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  Domdec *d=system->domdec;
  int id=d->id;
  int startBlock=d->blockCount[id];
  int endBlock=d->blockCount[id+1];
  int N=endBlock-startBlock;
  int shMem=0;

  if (r->calcTermFlag[eenbdirect]==false) return;

  if (r->scrVdw >= r->cutoffs.rSwitch || r->scrElec >= r->cutoffs.rSwitch){
    printf("Derivatives of soft-core switch function not implemented! Please raise switch radius or decrease SOFT.\n");
    exit(1);
  }

  // need to clear memory from regular nbdirect kernel
  cudaMemsetAsync(system->domdec->localForce_d,0,32*system->domdec->maxBlocks*sizeof(real3_f), r->nbdirectStream);
  getforce_nbdirect_oss_kernel<flagBox,useSoftCore,usevdWSwitch,usePME><<<((32<<WARPSPERBLOCK)*N+(32<<WARPSPERBLOCK)-1)/(32<<WARPSPERBLOCK),(32<<WARPSPERBLOCK),shMem,r->nbdirectStream>>>(
    startBlock,endBlock,d->maxPartnersPerBlock,d->blockBounds_d,d->blockPartnerCount_d,d->blockPartners_d,d->blockVolume_d,d->localNbonds_d,p->vdwParameterCount,
#ifdef USE_TEXTURE
    p->vdwParameters_tex,
#else
    p->vdwParameters_d,
#endif
    system->domdec->blockExcls_d,
    system->run->cutoffs,r->scrVdw, r->scrElec,
    d->localPosition_d,
    d->localForce_d,
    box,s->lambda_fd,
    s->lambdaForce_d,
    system->enhanced->osrw->dGdF_d);

  system->domdec->unpack_forces(system);
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,typename box_type>
void getforce_nbdirect_ossTTT(System *system,box_type box)
{
  if (system->run->usePME) {
    getforce_nbdirect_ossTTTT<flagBox,useSoftCore,usevdWSwitch,true>(system,box);
  } else {
    getforce_nbdirect_ossTTTT<flagBox,useSoftCore,usevdWSwitch,false>(system,box);
  }
}

template <bool flagBox,bool useSoftCore,typename box_type>
void getforce_nbdirect_ossTT(System *system,box_type box)
{
  if (!system->run->vfSwitch) {
    getforce_nbdirect_ossTTT<flagBox,useSoftCore,true>(system,box);
  } else {
    getforce_nbdirect_ossTTT<flagBox,useSoftCore,false>(system,box);
  }
}

template <bool flagBox,typename box_type>
void getforce_nbdirect_ossT(System *system,box_type box)
{
  if (system->msld->useSoftCore) {
    getforce_nbdirect_ossTT<flagBox,true>(system,box);
  } else {
    getforce_nbdirect_ossTT<flagBox,false>(system,box);
  }
}

void getforce_nbdirect_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_nbdirect_ossT<true>(system,system->state->tricBox_f);
  } else {
    getforce_nbdirect_ossT<false>(system,system->state->orthBox_f);
  }
}

template <typename real_type>
__global__ void getforce_nbdirect_reduce_oss_kernel(int atomCount,int idCount,real_type *force)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j;
  real_type f=0;

  if (i<atomCount) {
    for (j=1; j<idCount; j++) {
      f+=force[j*atomCount+i];
    }
    atomicAdd(&force[i],f);
  }
}

void getforce_nbdirect_oss_reduce(System *system) {
  State *s=system->state;
  Run *r=system->run;
  int N=3*s->atomCount+2*s->lambdaCount;

  // TODO: This doesn't accound for multi-gpu yet?
  getforce_nbdirect_reduce_oss_kernel<<<(N+BLNB-1)/BLNB,BLNB,0,r->nbdirectStream>>>(N,system->idCount,s->forceBuffer_d);
}