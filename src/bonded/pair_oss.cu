#include <cassert>
#include <cuda_runtime.h>
#include <math.h>

#include "bonded.h"
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
    real3 *position,real3_f *force,box_type box,
    real *lambda,real_f *lambdaForce,
    real *dGdF)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  real r;
  real3 dr;
  NbExPotential pp;
  real3 xi,xj;
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
      real fij=0;
      real fli=0;
      real fljtmp=0;

      // NbExclusion correction - never soft-cored so we can just use fij
      real rinv=1/r;
      real br=cutoffs.betaEwald*r;
      real kqq=kELECTRIC*pp.qxq;
      fij=kqq*(erff(br)*rinv-((real)1.128379167095513)*cutoffs.betaEwald*expf(-br*br))*rinv;
      real uij=-kqq*erff(br)*rinv;

      // OST Derivatives - lixlij is always just li*lj for ewald?
      real dU_drij_dli = ljtmp*fij;
      real dU_drij_dlj = bjtmp ? li*fij : 0;
      real dU_dlami_dlamj = bi && bjtmp ? uij : 0;

      // Forces redef
      fij = dGdFi * dU_drij_dli + dGdFjtmp * dU_drij_dlj;
      fli = dGdFjtmp * dU_dlami_dlamj;
      fljtmp = dGdFi * dU_dlami_dlamj; // this doesn't over-count due to product rule

      // Lambda and spatial forces
      atomicAdd(&lambdaForce[b[0]], fli);
      if (b[1]) {
        atomicAdd(&lambdaForce[b[1]], fljtmp);
      }
      at_real3_scaleinc(&force[ii], fij/r,dr);
      at_real3_scaleinc(&force[jj],-fij/r,dr);
    }
  }
}


template <bool flagBox, bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
__global__ void getforce_14pair_kernel_oss(
    int pairCount,Nb14Potential *pairs,Cutoffs cutoffs,
    real3 *position,real3_f *force,box_type box,
    real *lambda,real_f *lambdaForce,
    real *dGdF)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  real r;
  real3 dr;
  Nb14Potential pp;
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
      real lixljtmp, dlixlj_dli, dlixlj_dlj, dlixlj_dli_dlj;
      if ((pp.siteBlock[0]&0xFFFF0000)==(pp.siteBlock[1]&0xFFFF0000)) { // same site (m == n)
        printf("Unexpected scaling case occurred! Didn't expect this to run! Contact devs!\n");
      }
      lixljtmp = li*ljtmp;
      dlixlj_dli = bi ? ljtmp : 0;
      dlixlj_dlj = bjtmp ? li : 0;
      dlixlj_dli_dlj = bi && bjtmp ? 1 : 0;

      // Force storage
      real fij=0;
      real fli=0;
      real fljtmp=0;

      if (r<cutoffs.rCut){
        rEff=r;
        // Softcore OST
        real drijp_drij=1;
        real drijp_dlami=0;
        real drijp_dlamj=0;
        real d2rijp_drij_dlamj=0;
        real d2rijp_drij_dlami=0;
        real d2rijp_dlami_dlamj=0;
        real d2rijp_dlami2=0;
        real d2rijp_dlamj2=0;
        real rSoft=SOFTCORERADIUS*(1-lixljtmp);
        if (useSoftCore) {
          if (bi || bjtmp) { // if either is a site
            if (r<rSoft) {
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
        real rinv=1/rEff;

        // Terms which require calculation for soft-coring - both vdw and elec can be accumulated here
        real dU_drijp = 0;
        real d2U_drijp2 = 0;
        real d2U_drijp_dlami = 0;
        real d2U_drijp_dlamj = 0;
        real d2Up_dlami_dlamj = 0;
        // Electrostatics (define the above variables for soft-core OST interactions)
        if (usePME) {
          // lixlj*e14/r = lixlj*e14fac/rEff + (-li*lj*erfc(rEff)/rEff + li*lj*erf(r)/r) (first term here)
          // Ewald Correction - Soft Ewald (end-states are what matter):
          real br=cutoffs.betaEwald*rEff;
          real erfrinv=erf(br)*rinv;
          real kqq = kELECTRIC*pp.qxq;
          real two = (real) 2.0;
          real U_dir = -kqq*erfrinv;
          real dU_drijp_tmp = -kqq*rinv*((2/sqrt(M_PI))*cutoffs.betaEwald*expf(-br*br)-erfrinv);
          dU_drijp += li*ljtmp * dU_drijp_tmp;
          d2U_drijp2 += -li*ljtmp*two*kqq*rinv*rinv*rinv*(1/M_PI)*
            (-two*sqrt(M_PI)*br*br*br*expf(-br*br) - two*sqrt(M_PI)*br*expf(-br*br) + M_PI*erf(br));
          d2U_drijp_dlami += dlixlj_dli * dU_drijp_tmp;
          d2U_drijp_dlamj += dlixlj_dlj * dU_drijp_tmp;
          d2Up_dlami_dlamj += dlixlj_dli_dlj * U_dir;
          // Accumulate later... or do I need to use special soft-core variables

          // Coulomb - soft-cored - lixlj * e14/rEff
          rinv = 1/rEff; // Corrected above change to go back to soft-coring
          U_dir = kELECTRIC*pp.qxq*pp.e14fac*rinv;
          dU_drijp_tmp = -U_dir*rinv;
          dU_drijp += lixljtmp * dU_drijp_tmp;
          d2U_drijp2 += -2*lixljtmp*dU_drijp_tmp*rinv;
          d2U_drijp_dlami += dlixlj_dli * dU_drijp_tmp;
          d2U_drijp_dlamj += dlixlj_dlj * dU_drijp_tmp;
          d2Up_dlami_dlamj += dlixlj_dli_dlj * U_dir;
          // Accumulate later...
        }
        else {
          //TODO: Implement this path
          printf("Coulomb cutoff oss path not implemented!");
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
            -kELECTRIC*pp.qxq*rinv*rinv:
            -kELECTRIC*pp.qxq*rinv*(Aconst*rinv+Bconst*rEff+3*Cconst*r3+5*Dconst*r5);
          real eij_tmp=(rEff<=cutoffs.rSwitch)?
            kELECTRIC*pp.qxq*(rinv+dvc):
            kELECTRIC*pp.qxq*(Aconst*(rinv-1/cutoffs.rCut)+Bconst*(cutoffs.rCut-rEff)+Cconst*(roff2*cutoffs.rCut-r3)+Dconst*(roff2*roff2*cutoffs.rCut-r5));
        }


        // Van der Waals
        real rinv3=rinv*rinv*rinv;
        real rinv6=rinv3*rinv3;
        real rCut3=cutoffs.rCut*cutoffs.rCut*cutoffs.rCut;
        real rSwitch3=cutoffs.rSwitch*cutoffs.rSwitch*cutoffs.rSwitch;
        if (rEff<cutoffs.rSwitch) { // Soft-cored
          // OSS calculation:
          real dv6=usevdWSwitch?0:1/(rCut3*rSwitch3);
          real U_dir = pp.c12*(rinv6*rinv6-dv6*dv6)-pp.c6*(rinv6-dv6);
          real dU_drijp_tmp = rinv*rinv6*(-12*pp.c12*rinv6+6*pp.c6);
          dU_drijp += lixljtmp * dU_drijp_tmp;
          d2U_drijp2 += lixljtmp*6*rinv6*rinv*rinv*(26*pp.c12*rinv6-7*pp.c6);
          d2U_drijp_dlami += dlixlj_dli * dU_drijp_tmp;
          d2U_drijp_dlamj += dlixlj_dlj * dU_drijp_tmp;
          d2Up_dlami_dlamj += dlixlj_dli_dlj * U_dir;
          // Accumulate later...
        }
        else { // Not soft-cored
          if ( !usevdWSwitch ) { // Force Switch
            real k6=rCut3/(rCut3-rSwitch3);
            real k12=rCut3*rCut3/(rCut3*rCut3-rSwitch3*rSwitch3);
            real rCutinv3=1/rCut3;
            real fij_tmp=(6*pp.c6*k6*(rinv3-rCutinv3)*rinv3-12*pp.c12*k12*(rinv6-rCutinv3*rCutinv3)*rinv6)*rinv;
            real eij_tmp=pp.c12*k12*(rinv6-rCutinv3*rCutinv3)*(rinv6-rCutinv3*rCutinv3)-pp.c6*k6*(rinv3-rCutinv3)*(rinv3-rCutinv3);
            // OSS calculation (not soft-cored):
            real d2U_drij_dlami = dlixlj_dli * fij_tmp;
            real d2U_drij_dlamj = dlixlj_dlj * fij_tmp;
            real d2U_dli_dlj = dlixlj_dli_dlj * eij_tmp;
            // No soft-core = direct accumulation
            fij += dGdFi * d2U_drij_dlami + dGdFjtmp * d2U_drij_dlamj;
            fli += dGdFjtmp * d2U_dli_dlj;
            fljtmp += dGdFi * d2U_dli_dlj;
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
            real d2U_drij_dlami = dlixlj_dli * fij_tmp;
            real d2U_drij_dlamj = dlixlj_dlj * fij_tmp;
            real d2U_dli_dlj = dlixlj_dli_dlj * eij_tmp;
            // No soft-core = direct accumulation
            fij += dGdFi * d2U_drij_dlami + dGdFjtmp * d2U_drij_dlamj;
            fli += dGdFjtmp * d2U_dli_dlj;
            fljtmp += dGdFi * d2U_dli_dlj;
          }
        }

        // Calculate chain rule derivatives due to soft-coring
        real fij_ost = 0;
        real d2U_drij_dlami = dU_drijp*d2rijp_drij_dlami + (d2U_drijp_dlami + d2U_drijp2*drijp_dlami) * drijp_drij;
        real d2U_drij_dlamj = dU_drijp*d2rijp_drij_dlamj + (d2U_drijp_dlamj + d2U_drijp2*drijp_dlamj) * drijp_drij;
        // Interaction feels i, j, or both histograms
        fij_ost = dGdFi * d2U_drij_dlami + dGdFjtmp * d2U_drij_dlamj;

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
        fij += fij_ost;
        fli += fli_ost;
        fljtmp += fljtmp_ost;
        atomicAdd(&lambdaForce[b[0]], fli);
        if (b[1]) {
          atomicAdd(&lambdaForce[b[1]], fljtmp);
        }
        at_real3_scaleinc(&force[ii], fij/r,dr);
        at_real3_scaleinc(&force[jj],-fij/r,dr);
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
  int shMem=0;

  if (r->calcTermFlag[eenb14]==false) return;

  if (N==0) return;

  getforce_14pair_kernel_oss<flagBox,useSoftCore,usevdWSwitch,usePME> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->ossBonded>>>(
      N,p->nb14s_d,system->run->cutoffs,(real3*)s->position_fd,
      (real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,
      system->msld->dGdF_d);
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
  int shMem=0;

  if (N==0) return;

  if (r->usePME==false) return;
  if (r->calcTermFlag[eenbrecipexcl]==false) return;

  // Never use soft cores for nbex, they're already soft.
  getforce_exclusion_pair_kernel_oss<flagBox,false,false,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->ossBonded>>>(
      N,p->nbexs_d,system->run->cutoffs,(real3*)s->position_fd,
      (real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,
      system->msld->dGdF_d);
}

void getforce_nbex_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_nbexT_oss<true>(system,system->state->tricBox_f);
  } else {
    getforce_nbexT_oss<false>(system,system->state->orthBox_f);
  }
}
