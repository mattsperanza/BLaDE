#include <cassert>
#include <cuda_runtime.h>
#include <math.h>

#include "enhanced/enhanced.h"
#include "enhanced/osrw/osrw.h"
#include "main/defines.h"
#include "system/system.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"

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

template <bool flagBox, bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
__global__ void getforce_exclusion_pair_kernel_oss(
  int pairCount,NbExPotential *pairs,Cutoffs cutoffs,
    real3 *position,real3_f *force, box_type box,
    real *lambda,real_f *lambdaForce, 
    real *dGdF) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  real r;
  real3 dr;
  NbExPotential pp;
  real3 xi,xj;
  real lEnergy = 0;
  extern __shared__ real sEnergy[];
  int b[2];

  if (i<pairCount) {
    // Geometry
    pp=pairs[i];
    ii=pp.idx[0];
    jj=pp.idx[1];
    xi=position[ii];
    xj=position[jj];
    dr=real3_subpbc<flagBox>(xi,xj,box);
    r=real3_mag<real>(dr);
    // Scaling
    b[0]=0xFFFF & pp.siteBlock[0];
    b[1]=0xFFFF & pp.siteBlock[1];
    if (b[0]){ // only perform if alchem atom present
      // Consistant notation with nbdirect_oss
      int bi = b[0];
      int bjtmp = b[1];
      real li = lambda[b[0]];
      real ljtmp = lambda[b[1]];
      real dGdFi = dGdF[b[0]];
      real dGdFjtmp = dGdF[b[1]];
      // Force storage
      real fij_ost=0;
      real fli_ost=0;
      real fljtmp_ost=0;
      real fij=0;
      real fli=0;
      real flj=0;

      // NbExclusion correction - never soft-cored so we can just use fij
      real rinv=1/r;
      real br=cutoffs.betaEwald*r;
      real kqq=kELECTRIC*pp.qxq;
      real fij_tmp=kqq*(erff(br)*rinv-((real)1.128379167095513)*cutoffs.betaEwald*expf(-br*br))*rinv;
      real uij=-kqq*erff(br)*rinv;
      // Vanilla force & energy
      lEnergy = li*ljtmp*uij;
      fij = li*ljtmp*fij_tmp; // no soft, no alpha scaling since we didn't for recip
      fli = ljtmp*uij;
      flj = bjtmp ? li*uij : 0;

      // OST Derivatives - lixlij is always just li*lj for ewald?
      real dU_drij_dli = ljtmp*fij_tmp;
      real dU_drij_dlj = bjtmp ? li*fij_tmp : 0;
      real dU_dlami_dlamj = bi && bjtmp ? uij : 0;

      fij_ost = dGdFi * dU_drij_dli + dGdFjtmp * dU_drij_dlj;
      fli_ost = dGdFjtmp * dU_dlami_dlamj;
      fljtmp_ost = dGdFi * dU_dlami_dlamj; // this doesn't over-count due to product rule

      // Lambda and spatial forces
      atomicAdd(&lambdaForce[b[0]], fli_ost);
      if (b[1]) {
        atomicAdd(&lambdaForce[b[1]], fljtmp_ost);
      }
      at_real3_scaleinc(&force[ii], fij_ost/r,dr);
      at_real3_scaleinc(&force[jj],-fij_ost/r,dr);
    }
  }
}

template <bool flagBox, bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
__global__ void getforce_14pair_kernel_oss(
    int pairCount,Nb14Potential *pairs,Cutoffs cutoffs,real scrVdw, real scrElec,
    real3 *position,real3_f *force, box_type box,
    real *lambda,real_f *lambdaForce, 
    real *dGdF, real a)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  real r;
  real3 dr;
  Nb14Potential pp;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj;
  int b[2];
  real rEff,dredr,dredll; // Soft core stuff

  if (i<pairCount) {
    // Geometry
    pp=pairs[i];
    ii=pp.idx[0];
    jj=pp.idx[1];
    xi=position[ii];
    xj=position[jj];
    dr=real3_subpbc<flagBox>(xi,xj,box);
    r=real3_mag<real>(dr);

    b[0]=0xFFFF & pp.siteBlock[0];
    b[1]=0xFFFF & pp.siteBlock[1];
    if (b[0]){ // only perform if alchem atom present
      // Consistant notation with nbdirect_oss
      int bi = b[0];
      int bjtmp = b[1];
      real li = lambda[b[0]];
      real ljtmp = lambda[b[1]];
      real dGdFi = dGdF[b[0]];
      real dGdFjtmp = dGdF[b[1]];
      real lixljtmp, dlixlj_dli, dlixlj_dlj, d2lixlj_dli_dlj;
      if ((pp.siteBlock[0]&0xFFFF0000)==(pp.siteBlock[1]&0xFFFF0000)) { // same site (m == n)
        printf("Unexpected scaling case occurred! Didn't expect this to run! Contact devs!\n");
      }
      lixljtmp = li*ljtmp;
      dlixlj_dli = ljtmp;
      dlixlj_dlj = li;
      d2lixlj_dli_dlj = bi && bjtmp ? 1 : 0;

      // Force storage
      real fij_ost=0;
      real fli_ost=0;
      real fljtmp_ost=0;
      real eij=0;
      real fij=0;
      real fli=0;
      real flj=0;

      if (r<cutoffs.rCut){
        rEff=r;
        // Total interaction derivatives
        real d2U_drij_dli = 0;
        real d2U_drij_dlj = 0;
        real d2U_dli_dlj = 0;
        real d2U_dli2 = 0;
        real d2U_dlj2 = 0;

        // Electrostatics - Ewald (define the above variables for soft-core OST interactions)
        real t[cr_count] = {0};  // Softcore c.r. terms
        t[drijp_drij] = 1;
        if (useSoftCore && (bi || bjtmp)) {
          set_soft(r, scrElec, 
            lixljtmp, dlixlj_dli, dlixlj_dlj, d2lixlj_dli_dlj,
            t, &rEff);
        }
        real rinv=1/rEff;
        if (usePME) {
          // TODO: Potentially combine these two
          // lixlj*e14/r = lixlj*e14fac/rEff + (-li*lj*erfc(rEff)/rEff + li*lj*erf(r)/r) (first term here)
          // Ewald Correction - Soft Ewald (end-states are what matter):
          real br=cutoffs.betaEwald*rEff;
          real erfrinv=erf(br)*rinv;
          real kqq = kELECTRIC*pp.qxq;
          real two = (real) 2.0;
          real U_dir = -kqq*erfrinv;
          real dU_drijp_tmp = -kqq*rinv*((2/sqrt(M_PI))*cutoffs.betaEwald*expf(-br*br)-erfrinv);
          real dU_drijp = li*ljtmp * dU_drijp_tmp;
          real d2U_drijp2 = -li*ljtmp*two*kqq*rinv*rinv*rinv*(1/M_PI)*
            (-two*sqrt(M_PI)*br*br*br*expf(-br*br) - two*sqrt(M_PI)*br*expf(-br*br) + M_PI*erf(br));
          real d2U_drijp_dli = dlixlj_dli * dU_drijp_tmp;
          real d2U_drijp_dlj = dlixlj_dlj * dU_drijp_tmp;
          real d2Up_dli_dlj = d2lixlj_dli_dlj * U_dir;
          // Accumulate
          d2U_drij_dli += dU_drijp*t[d2rijp_drij_dli] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_drij];
          d2U_drij_dlj += dU_drijp*t[d2rijp_drij_dlj] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_drij];
          d2U_dli_dlj += d2Up_dli_dlj + d2U_drijp_dlj*t[drijp_dli] + dU_drijp*t[d2rijp_dli_dlj] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dlj];
          d2U_dli2 += d2U_drijp_dli*t[drijp_dli] + dU_drijp*t[d2rijp_dli2] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dli];
          d2U_dlj2 += d2U_drijp_dlj*t[drijp_dlj] + dU_drijp*t[d2rijp_dlj2] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_dlj];

          // Coulomb - soft-cored - lixlj * e14/rEff
          rinv = 1/rEff; // Corrected above change to go back to soft-coring
          U_dir = kELECTRIC*pp.qxq*pp.e14fac*rinv;
          dU_drijp_tmp = -U_dir*rinv;
          dU_drijp = lixljtmp * dU_drijp_tmp;
          d2U_drijp2 = -2*lixljtmp*dU_drijp_tmp*rinv;
          d2U_drijp_dli = dlixlj_dli * dU_drijp_tmp;
          d2U_drijp_dlj = dlixlj_dlj * dU_drijp_tmp;
          d2Up_dli_dlj = d2lixlj_dli_dlj * U_dir;
          // Accumulate
          d2U_drij_dli += dU_drijp*t[d2rijp_drij_dli] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_drij];
          d2U_drij_dlj += dU_drijp*t[d2rijp_drij_dlj] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_drij];
          d2U_dli_dlj += d2Up_dli_dlj + d2U_drijp_dlj*t[drijp_dli] + dU_drijp*t[d2rijp_dli_dlj] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dlj];
          d2U_dli2 += d2U_drijp_dli*t[drijp_dli] + dU_drijp*t[d2rijp_dli2] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dli];
          d2U_dlj2 += d2U_drijp_dlj*t[drijp_dlj] + dU_drijp*t[d2rijp_dlj2] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_dlj];
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
        if (rEff<cutoffs.rSwitch) { // Soft-cored
          // OSS calculation:
          real dv6=usevdWSwitch?0:1/(rCut3*rSwitch3);
          real U_dir = pp.c12*(rinv6*rinv6-dv6*dv6)-pp.c6*(rinv6-dv6);
          real dU_drijp_tmp = rinv*rinv6*(-12*pp.c12*rinv6+6*pp.c6);
          real dU_drijp = lixljtmp * dU_drijp_tmp;
          real d2U_drijp2 = lixljtmp*6*rinv6*rinv*rinv*(26*pp.c12*rinv6-7*pp.c6);
          real d2U_drijp_dli = dlixlj_dli * dU_drijp_tmp;
          real d2U_drijp_dlj = dlixlj_dlj * dU_drijp_tmp;
          real d2Up_dli_dlj = d2lixlj_dli_dlj * U_dir;
          // Accumulate
          d2U_drij_dli += dU_drijp*t[d2rijp_drij_dli] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_drij];
          d2U_drij_dlj += dU_drijp*t[d2rijp_drij_dlj] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_drij];
          d2U_dli_dlj += d2Up_dli_dlj + d2U_drijp_dlj*t[drijp_dli] + dU_drijp*t[d2rijp_dli_dlj] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dlj];
          d2U_dli2 += d2U_drijp_dli*t[drijp_dli] + dU_drijp*t[d2rijp_dli2] + (d2U_drijp_dli + d2U_drijp2*t[drijp_dli])*t[drijp_dli];
          d2U_dlj2 += d2U_drijp_dlj*t[drijp_dlj] + dU_drijp*t[d2rijp_dlj2] + (d2U_drijp_dlj + d2U_drijp2*t[drijp_dlj])*t[drijp_dlj];
        }
        else { // Not soft-cored
          if ( !usevdWSwitch ) { // Force Switch
            real k6=rCut3/(rCut3-rSwitch3);
            real k12=rCut3*rCut3/(rCut3*rCut3-rSwitch3*rSwitch3);
            real rCutinv3=1/rCut3;
            real fij_tmp=(6*pp.c6*k6*(rinv3-rCutinv3)*rinv3-12*pp.c12*k12*(rinv6-rCutinv3*rCutinv3)*rinv6)*rinv;
            real eij_tmp=pp.c12*k12*(rinv6-rCutinv3*rCutinv3)*(rinv6-rCutinv3*rCutinv3)-pp.c6*k6*(rinv3-rCutinv3)*(rinv3-rCutinv3);
            // OSS calculation (not soft-cored):
            d2U_drij_dli += dlixlj_dli*fij_tmp;
            d2U_drij_dlj += dlixlj_dlj*fij_tmp;
            d2U_dli_dlj += d2lixlj_dli_dlj*eij_tmp;
          } else { // Potential Switch
            real c2ofnb=cutoffs.rCut*cutoffs.rCut;
            real c2onnb=cutoffs.rSwitch*cutoffs.rSwitch;
            real rul3=(c2ofnb-c2onnb)*(c2ofnb-c2onnb)*(c2ofnb-c2onnb);
            real rul12 = 12/rul3;
            real rijl = c2onnb - rEff * rEff;
            real riju = c2ofnb - rEff * rEff;
            real fsw = riju*riju*(riju-3*rijl)/rul3;
            real dfsw = rijl*riju*rul12;
            real fij_tmp=fsw*(6*pp.c6-12*pp.c12*rinv6)*rinv6*rinv\
              +dfsw*(pp.c12*rinv6-pp.c6)*rinv6;
            real eij_tmp=fsw*(pp.c12*rinv6-pp.c6)*rinv6;
            // OSS calculation (not soft-cored):
            d2U_drij_dli += dlixlj_dli*fij_tmp;
            d2U_drij_dlj += dlixlj_dlj*fij_tmp;
            d2U_dli_dlj += d2lixlj_dli_dlj*eij_tmp;
          }
        }

        // OSS Forces
        fij_ost += dGdFi*d2U_drij_dli + dGdFjtmp*d2U_drij_dlj;
        fli_ost += dGdFi*d2U_dli2 + dGdFjtmp*d2U_dli_dlj;
        if(bjtmp){
          fljtmp_ost += dGdFi*d2U_dli_dlj + dGdFjtmp*d2U_dlj2;
        }

        // OSS & Vanilla forces
        atomicAdd(&lambdaForce[b[0]], fli_ost);
        if (b[1]) {
          atomicAdd(&lambdaForce[b[1]], fljtmp_ost);
        }
        at_real3_scaleinc(&force[ii], fij_ost/r,dr);
        at_real3_scaleinc(&force[jj],-fij_ost/r,dr);
      }
    }
  }
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
void getforce_nb14TTTT_oss(System *system,box_type box)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N=p->nb14Count;
  int shMem=BLBO*sizeof(real)/32;

  if (r->calcTermFlag[eenb14]==false) return;

  if (N==0) return;

  getforce_14pair_kernel_oss<flagBox,useSoftCore,usevdWSwitch,usePME> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
      N,p->nb14s_d,r->cutoffs,r->scrVdw,r->scrElec,
      (real3*)s->position_fd,
      (real3_f*)s->force_d,box,
      s->lambda_fd,s->lambdaForce_d, 
      system->enhanced->osrw->dGdF_d, 
      1.0);
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,typename box_type>
void getforce_nb14TTT_oss(System *system,box_type box)
{
  if (system->run->usePME) {
    getforce_nb14TTTT_oss<flagBox,useSoftCore,usevdWSwitch,true>(system,box);
  } else {
    getforce_nb14TTTT_oss<flagBox,useSoftCore,usevdWSwitch,false>(system,box);
  }
}

template <bool flagBox,bool useSoftCore,typename box_type>
void getforce_nb14TT_oss(System *system,box_type box)
{
  if (!system->run->vfSwitch) {
    getforce_nb14TTT_oss<flagBox,useSoftCore,true>(system,box);
  } else {
    getforce_nb14TTT_oss<flagBox,useSoftCore,false>(system,box);
  }
}

template <bool flagBox,typename box_type>
void getforce_nb14T_oss(System *system,box_type box)
{
  if (system->msld->useSoftCore14) {
    getforce_nb14TT_oss<flagBox,true>(system,box);
  } else {
    getforce_nb14TT_oss<flagBox,false>(system,box);
  }
}

void getforce_nb14_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_nb14T_oss<true>(system,system->state->tricBox_f);
  } else {
    getforce_nb14T_oss<false>(system,system->state->orthBox_f);
  }
}



template <bool flagBox,typename box_type>
void getforce_nbexT_oss(System *system,box_type box)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N=p->nbexCount;
  int shMem=BLBO*sizeof(real)/32;

  if (N==0) return;

  if (r->usePME==false) return;
  if (r->calcTermFlag[eenbrecipexcl]==false) return;

  // Ewald already soft-cored
  getforce_exclusion_pair_kernel_oss<flagBox,false,false,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->nbexs_d,system->run->cutoffs,
    (real3*)s->position_fd, (real3_f*)s->force_d,box,
    s->lambda_fd,s->lambdaForce_d, 
    system->enhanced->osrw->dGdF_d);
}

void getforce_nbex_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_nbexT_oss<true>(system,system->state->tricBox_f);
  } else {
    getforce_nbexT_oss<false>(system,system->state->orthBox_f);
  }
}