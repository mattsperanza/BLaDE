//
// Created by matthew-speranza on 1/16/25.
//
#include <cassert>
#include <cuda_runtime.h>

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
  struct Cutoffs cutoffs,
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
  real fij;
  real3 xi,xj,xjtmp;
  real3 fi,fj,fjtmp;
  real fli,flj,fljtmp;
  int bi,bj,bjtmp,bi_any,bj_any;
  real li,lj,ljtmp,lixljtmp;
  real rEff,dredr,dredll; // Soft core stuff
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
      dGdF=0;
      if (bi) {
        li=lambda[0xFFFF & bi];
        dGdFi=dGdF[0xFFFF & bi];
        if (isnan(dGdFi)) {
          printf("dGdF is nan bi=%d, li=%f, 0xFFFF & bi: %d\n", bi, li, 0xFFFF&bi);
        }
      }
    }
    // Check if bi has alchemical atoms via warp reduction
    bi_any = __any_sync(0xFFFFFFFF, bi);
    // iBlockVolume=blockVolume[iBlock];

    fi=real3_reset<real3>();
    fli=0;

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
          if (isnan(dGdFj)) {
            printf("dGdF is nan bj=%d, lj=%f, 0xFFFF & bj: %d\n", bj, lj, 0xFFFF&bj);
          }
        }
      }
      // Sum warp bj's to check # alchemical atoms via warp reduction
      bj_any = __any_sync(0xFFFFFFFF, bj);
      // If bi && bj are both 0, then we can skip this block-block interaction
      if (bi_any == 0 && bj_any == 0) {
        // TODO: Figure out how to make this work correctly
        //continue;
      }
      bool jFlag=check_proximity(iBlockVolume,xj,cutoffs.rCut*cutoffs.rCut);

      fj=real3_reset<real3>();
      flj=0;

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

          fjtmp=real3_reset<real3>();
          fljtmp=0;
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

            if (r<cutoffs.rCut && (bi || bjtmp)) {
              // Scaling
              real dlixlj_dli, dlixlj_dlj, dlixlj_dli_dlj;
              // for l_{m,i} and l_{n,j}
              if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) { // same site (m == n)
                if (bi==bjtmp) { // intra-sub (i == j, alc-alc or env-env)
                  lixljtmp = li;
                  dlixlj_dli = bi ? 1 : 0;
                  dlixlj_dlj = 0; // prevent over-count
                  dGdFjtmp = 0;
                  dlixlj_dli_dlj = 0;
                } else { // intra-site (i != j, alc-alc or env-env)
                  lixljtmp = 0; // zero contribution from env-env anyway
                  dlixlj_dli = 0;
                  dlixlj_dlj = 0;
                  dlixlj_dli_dlj = 0;
                }
              } else { // inter-site (m != n)
                lixljtmp = li*ljtmp;
                dlixlj_dli = bi ? ljtmp : 0;
                dlixlj_dlj = bjtmp ? li : 0;
                dlixlj_dli_dlj = bi && bjtmp ? 1 : 0;
              }
              if (isnan(dGdFi)) {
                printf("dGdFi is nan bi=%d, li=%f, 0xFFFF & bi: %d\n", bi, li, 0xFFFF&bi);
                dGdFi = 0;
              }
              if (isinf(dGdFi)) {
                printf("dGdFi is inf bi=%d, li=%f, 0xFFFF & bi: %d\n", bi, li, 0xFFFF&bi);
                dGdFi = 0;
              }
              if (isnan(dGdFjtmp)) {
                printf("dGdFjtmp is nan bjtmp=%d, ljtmp=%f, 0xFFFF & bjtmp: %d\n", bjtmp, ljtmp, 0xFFFF&bjtmp);
                dGdFjtmp = 0;
              }
              if (isinf(dGdFjtmp)) {
                printf("dGdFjtmp is inf bjtmp=%d, ljtmp=%f, 0xFFFF & bjtmp: %d\n", bjtmp, ljtmp, 0xFFFF&bjtmp);
                dGdFjtmp = 0;
              }

              // Softcore OST
              rEff=r;
              // rSoft derivatives
              real drijp_drij=1;
              real drijp_dlami=0;
              real drijp_dlamj=0;
              real d2rijp_drij_dlamj=0;
              real d2rijp_drij_dlami=0;
              real d2rijp_dlami_dlamj=0;
              real d2rijp_dlami2=0;
              real d2rijp_dlamj2=0;
              if (useSoftCore) {
                real rSoft=SOFTCORERADIUS*(1-lixljtmp);
                if (bi || bjtmp) { // if either is a site
                  if (r<rSoft) {
                    // Original soft
                    real rdivrs=r/rSoft;
                    rEff=1-((real)0.5)*rdivrs;
                    rEff=rEff*rdivrs*rdivrs*rdivrs+((real)0.5); // Soft-core: rEff = rL * (.5 + (r/rL)^3 - .5*(r/rL)^4)
                    rEff*=rSoft;
                    real drlam_dlami = -SOFTCORERADIUS*dlixlj_dli;
                    real drlam_dlamj = -SOFTCORERADIUS*dlixlj_dlj;
                    real d2rlam_dli_dlj = -SOFTCORERADIUS*dlixlj_dli_dlj;
                    real r_rsoft = r / rSoft;
                    real r_rsoft3 = r_rsoft * r_rsoft * r_rsoft;
                    real r_rsoft4 = r_rsoft * r_rsoft3;
                    real drijp_drlam = ((real)-.5)*r_rsoft4 + r_rsoft3 + (((real)2.0)*r_rsoft4 - ((real)3.0)*r_rsoft3)+((real).5);
                    real r2 = r*r;
                    real r3 = r2*r;
                    real rSoft3 = rSoft*rSoft*rSoft;
                    real six = (real) 6.0;
                    real d2rijp_drlam_drij = r2*(six*r_rsoft-six)/rSoft3;
                    real d2rijp_drlam2 = r3*(-six*r_rsoft+six)/(rSoft3*rSoft);
                    // First partials
                    drijp_drij = (((real)-2.0)*r_rsoft3+((real)3.0)*r_rsoft*r_rsoft);
                    drijp_dlami = drijp_drlam*drlam_dlami;
                    drijp_dlamj = drijp_drlam*drlam_dlamj;
                    // Second mixed partials
                    d2rijp_drij_dlami = d2rijp_drlam_drij*drlam_dlami;
                    d2rijp_drij_dlamj = d2rijp_drlam_drij*drlam_dlamj;
                    // This one depends on what lixljtmp looks like - set lixlj = li
                    d2rijp_dlami_dlamj = d2rijp_drlam2*drlam_dlami*drlam_dlamj + drijp_drlam*d2rlam_dli_dlj;
                    d2rijp_dlami2 = d2rijp_drlam2 * drlam_dlami*drlam_dlami;
                    d2rijp_dlamj2 = d2rijp_drlam2 * drlam_dlamj*drlam_dlamj;
                  }
                }
              }
              rinv=1/rEff;

              // Terms which require calculation for soft-coring - both vdw and elec can be accumulated here
              real dU_drijp = 0;
              real d2U_drijp2 = 0;
              real d2U_drijp_dlami = 0;
              real d2U_drijp_dlamj = 0;
              // The p in this term represents that it is a function of rijp not rij
              real d2Up_dlami_dlamj = 0;
              fij=0;
              // Electrostatics (define the above variables)
              if (usePME) { // precision matters for test, exp vs expf
                real br=cutoffs.betaEwald*rEff;
                real erfcrinv=fasterfc(br)*rinv;
                real U_dir = kELECTRIC*inp.q*jtmpnp_q*erfcrinv;
                // First Derivatives - these should eventually match current implementation
                real dU_drijp_tmp = -kELECTRIC*inp.q*jtmpnp_q*rinv
                  *(erfcrinv+((real)1.128379167095513)*cutoffs.betaEwald*expf(-br*br));
                dU_drijp += lixljtmp * dU_drijp_tmp;
                // Second Derivatives
                // Distribute expf(-br*br) term to avoid exp(br*br)
                real two = (real) 2.0;
                d2U_drijp2 += lixljtmp*two*kELECTRIC*inp.q*jtmpnp_q*rinv*rinv*rinv*(1/M_PI)*
                  (two*sqrt(M_PI)*br*br*br*expf(-br*br) + two*sqrt(M_PI)*br*expf(-br*br) + M_PI*fasterfc(br));
                d2U_drijp_dlami += dlixlj_dli * dU_drijp_tmp;
                d2U_drijp_dlamj += dlixlj_dlj * dU_drijp_tmp;
                d2Up_dlami_dlamj += dlixlj_dli_dlj * U_dir;
                // Accumulate later...
              }
              else {
                // TODO: Implement this path
                printf("Coulomb Cutoff OSS path not implemented!");
                assert(false);
                real roff2=cutoffs.rCut*cutoffs.rCut;
                real ron2=cutoffs.rSwitch*cutoffs.rSwitch;
                real ginv=1/((roff2-ron2)*(roff2-ron2)*(roff2-ron2));
                real Aconst=roff2*roff2*(roff2-3*ron2)*ginv;
                real Bconst=6*roff2*ron2*ginv;
                real Cconst=-(ron2+roff2)*ginv;
                real Dconst=2*ginv/5;
                real dvc=8*(ron2*roff2*(cutoffs.rCut-cutoffs.rSwitch)-(roff2*roff2*cutoffs.rCut-ron2*ron2*cutoffs.rSwitch)/5)*ginv;
                real r2=rEff*rEff;
                real r3=r2*rEff;
                real r5=r3*r2;
                real fij_tmp=(rEff<=cutoffs.rSwitch)?
                  -kELECTRIC*inp.q*jtmpnp_q*rinv*rinv:
                  -kELECTRIC*inp.q*jtmpnp_q*rinv*(Aconst*rinv+Bconst*rEff+3*Cconst*r3+5*Dconst*r5);
                real eij_tmp=(rEff<=cutoffs.rSwitch)?
                  kELECTRIC*inp.q*jtmpnp_q*(rinv+dvc):
                  kELECTRIC*inp.q*jtmpnp_q*(Aconst*(rinv-1/cutoffs.rCut)+Bconst*(cutoffs.rCut-rEff)+Cconst*(roff2*cutoffs.rCut-r3)+Dconst*(roff2*roff2*cutoffs.rCut-r5));

                if (rEff<cutoffs.rSwitch) {
                  // OSS on soft-cored coulomb:
                  // Accumulate later...
                } else {
                  // OSS on non-soft-cored tapered coulomb:
                }
              }

              // Van der Waals
              real rinv3=rinv*rinv*rinv;
              real rinv6=rinv3*rinv3;
              real rCut3=cutoffs.rCut*cutoffs.rCut*cutoffs.rCut;
              real rSwitch3=cutoffs.rSwitch*cutoffs.rSwitch*cutoffs.rSwitch;
              if (rEff<cutoffs.rSwitch) {
                real dv6=usevdWSwitch?0:1/(rCut3*rSwitch3);
                real U_dir = vdwp.c12*(rinv6*rinv6-dv6*dv6)-vdwp.c6*(rinv6-dv6);
                real dU_drijp_tmp = rinv*rinv6*(((real)-12.0)*vdwp.c12*rinv6+6*vdwp.c6);
                dU_drijp += lixljtmp * dU_drijp_tmp;
                d2U_drijp2 += lixljtmp*((real)6.0)*rinv6*rinv*rinv*(((real)26.0)*vdwp.c12*rinv6-((real)7.0)*vdwp.c6);
                d2U_drijp_dlami += dlixlj_dli * dU_drijp_tmp;
                d2U_drijp_dlamj += dlixlj_dlj * dU_drijp_tmp;
                d2Up_dlami_dlamj += dlixlj_dli_dlj * U_dir;
                // Accumulate later...
              }
              else {
                // Soft-coring c.r. terms make these work out in same var theoretically, I don't try very hard to make that work
                if ( !usevdWSwitch ) { // Force Switch
                  real k6=rCut3/(rCut3-rSwitch3);
                  real k12=rCut3*rCut3/(rCut3*rCut3-rSwitch3*rSwitch3);
                  real rCutinv3=1/rCut3;
                  real fij_tmp=(((real)6.0)*vdwp.c6*k6*(rinv3-rCutinv3)*rinv3-((real)12.0)*vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*rinv6)*rinv;
                  real eij_tmp=vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*(rinv6-rCutinv3*rCutinv3)-vdwp.c6*k6*(rinv3-rCutinv3)*(rinv3-rCutinv3);

                  // OSS calculation (not soft-cored):
                  real d2U_drij_dlami = dlixlj_dli * fij_tmp;
                  real d2U_drij_dlamj = dlixlj_dlj * fij_tmp;
                  real d2U_dlami_dlamj = dlixlj_dli_dlj * eij_tmp;
                  // Accumulate ost forces into force variables directly since they aren't soft-cored
                  fij += dGdFi * d2U_drij_dlami + dGdFjtmp * d2U_drij_dlamj;
                  fli += dGdFjtmp * d2U_dlami_dlamj;
                  fljtmp += dGdFi * d2U_dlami_dlamj;
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

                  // OSS calculation (not soft-cored):
                  real d2U_drij_dlami = dlixlj_dli * fij_tmp;
                  real d2U_drij_dlamj = dlixlj_dlj * fij_tmp;
                  real d2U_dlami_dlamj = dlixlj_dli_dlj * eij_tmp;
                  // Accumulate ost forces into force variables directly since they aren't soft-cored and don't need chain rule terms
                  fij += dGdFi * d2U_drij_dlami + dGdFjtmp * d2U_drij_dlamj;
                  fli += dGdFjtmp * d2U_dlami_dlamj;
                  fljtmp += dGdFi * d2U_dlami_dlamj;
                }
              }

              // Calculate chain rule derivatives due to soft-coring
              real fij_ost = 0;
              real d2U_drij_dlami = dU_drijp*d2rijp_drij_dlami + (d2U_drijp_dlami + d2U_drijp2*drijp_dlami) * drijp_drij;
              real d2U_drij_dlamj = dU_drijp*d2rijp_drij_dlamj + (d2U_drijp_dlamj + d2U_drijp2*drijp_dlamj) * drijp_drij;
              // Interaction feels i, j, or both histograms
              fij_ost = dGdFi*d2U_drij_dlami + dGdFjtmp*d2U_drij_dlamj;

              // OST lambda force:
              real fli_ost = 0;
              real fljtmp_ost = 0;
              // Only exists for alc-alc interactions
              real d2U_dlami_dlamj = d2Up_dlami_dlamj +
                d2U_drijp_dlamj*drijp_dlami + dU_drijp*d2rijp_dlami_dlamj + (d2U_drijp_dlami + d2U_drijp2*drijp_dlami) * drijp_dlamj;
              // Missing first term from previous - these are definitely not zero with soft-coring - exist for every alc-___ interaction
              real d2U_dlami2 = d2U_drijp_dlami*drijp_dlami + dU_drijp * d2rijp_dlami2 + (d2U_drijp_dlami + d2U_drijp2 * drijp_dlami)*drijp_dlami;
              real d2U_dlamj2 = d2U_drijp_dlamj*drijp_dlamj + dU_drijp * d2rijp_dlamj2 + (d2U_drijp_dlamj + d2U_drijp2 * drijp_dlamj)*drijp_dlamj;
              fli_ost = dGdFi*d2U_dlami2 + dGdFjtmp*d2U_dlami_dlamj;
              fljtmp_ost = dGdFi*d2U_dlami_dlamj + dGdFjtmp*d2U_dlamj2;

              // Accumulate ost forces
              if (isnan(fij) || isnan(fli) || isnan(fljtmp)) {
                printf("bi: %d, bjtmp: %d, li: %f, lj: %f, dGdFi: %f, dGdFj, %f, fij: %f, fli: %f, flj: %f, fij_ost: %f, fli_ost: %f, fljtmp_ost: %f\n",
                  bi, bjtmp, li, ljtmp, dGdFi, dGdFjtmp, fij, fli, fljtmp, fij_ost, fli_ost, fljtmp_ost);
              }
              fij += fij_ost;
              fli += fli_ost;
              fljtmp += fljtmp_ost;
              real3_scaleinc(&fi, fij/r,dr);
              fjtmp=real3_scale<real3>(-fij/r,dr);
            }
          }
          __syncwarp();
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,1);
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,2);
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,4);
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,8);
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,16);
          if (iThread==jtmp) fj.x=fjtmp.x;
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,1);
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,2);
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,4);
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,8);
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,16);
          if (iThread==jtmp) fj.y=fjtmp.y;
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,1);
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,2);
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,4);
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,8);
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,16);
          if (iThread==jtmp) fj.z=fjtmp.z;
          if (bjtmp) {
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,1);
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,2);
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,4);
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,8);
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,16);
            if (iThread==jtmp) flj=fljtmp;
          }
        }
      }
      __syncwarp();
      if ((iThread)<jCount) {
        if (bj) {
          atomicAdd(&lambdaForce[0xFFFF & bj],flj);
        }
        at_real3_inc(&force[32*jBlock+iThread],fj);
      }
      if (iThread==0) j=atomicInc((unsigned int*)(&jnext),0xFFFFFFFF);
      j=__shfl_sync(0xFFFFFFFF,j,0);
    }
    __syncwarp();
    if ((iThread)<iCount) {
      if (bi) {
        atomicAdd(&lambdaForce[0xFFFF & bi],fli);
      }
      at_real3_inc(&force[32*iBlock+iThread],fi);
    }
  }
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
void getforce_nbdirect_ossTTTT(System *system,box_type box)
{
  system->domdec->pack_positions(system);
  system->domdec->recull_blocks(system);

  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  Domdec *d=system->domdec;
  int id=d->id;
  int startBlock=d->blockCount[id];
  int endBlock=d->blockCount[id+1];
  int N=endBlock-startBlock;
  int shMem=0;

  // TODO: Kill program if rSoft > rSwitch
  if (r->calcTermFlag[eenbdirect]==false) return;
  getforce_nbdirect_oss_kernel<flagBox,useSoftCore,usevdWSwitch,usePME><<<((32<<WARPSPERBLOCK)*N+(32<<WARPSPERBLOCK)-1)/(32<<WARPSPERBLOCK),(32<<WARPSPERBLOCK),shMem,r->ossDirect>>>(startBlock,endBlock,d->maxPartnersPerBlock,d->blockBounds_d,d->blockPartnerCount_d,d->blockPartners_d,d->blockVolume_d,d->localNbonds_d,p->vdwParameterCount,
#ifdef USE_TEXTURE
    p->vdwParameters_tex,
#else
    p->vdwParameters_d,
#endif
    system->domdec->blockExcls_d,system->run->cutoffs,d->localPosition_d,d->localForce_d,box,s->lambda_fd,s->lambdaForce_d,
    system->msld->dGdF_d);

  // Note this should be done once per force calculation and is potentially done here instead of in reg nbdirect
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

  getforce_nbdirect_reduce_oss_kernel<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->forceBuffer_d);
}