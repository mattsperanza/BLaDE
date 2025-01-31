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

// Not good ways to check type
__device__ bool isNbEx(NbExPotential pp) {
  return true;
}

__device__ bool isNbEx(Nb14Potential pp) {
  return false;
}

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
      // Derivatives of the lambda function
      real dlixlj_dli = ljtmp; // this is simplified
      dlixlj_dli = bi ? dlixlj_dli : 0;
      real dlixlj_dlj = li;
      dlixlj_dlj = bjtmp ? dlixlj_dlj : 0;
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

      // TODO: Check if product of lambdas is correct
      // OST Derivatives - lixlij is always just li*lj for ewald?
      real dU_drij_dli = dlixlj_dli*fij;
      real dU_drij_dlj = dlixlj_dlj*fij;
      real dU_dlami_dlamj = uij;

      // Forces redef
      fij = dGdFi * dU_drij_dli + dGdFjtmp * dU_drij_dlj;
      fli = dGdFjtmp * dU_dlami_dlamj;
      fljtmp = dGdFi * dU_dlami_dlamj; // this doesn't over-count due to product rule

      // Lambda and spatial forces
      atomicAdd(&lambdaForce[b[0]], fli);
      atomicAdd(&lambdaForce[b[1]], fljtmp);
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

    // Everything below is basically (see lambda scaling for diff)
    // a cut and paste from nbdirect_oss
    // so I don't have to debug in 2 places
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
      // Two types of lambda scaling:
      // 1) Normal one for Coulomb and VdW
      // Lambda Scaling option one: lixljtmp -> this is used in nbdirect
      real lixljtmp = 0.0;
      if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) { // TODO: Make sure this is correct for Nb14 same as regular Nb
        if (bi==bjtmp) {
          lixljtmp=li;
        } else {
          lixljtmp=0;
        }
      } else {
        lixljtmp=li*ljtmp;
      }
      real dlixlj_dli = bi != bjtmp ? ljtmp : .5;
      dlixlj_dli = bi ? dlixlj_dli : 0;
      real dlixlj_dlj = bi != bjtmp ? li : .5;
      dlixlj_dlj = bjtmp ? dlixlj_dlj : 0;
      real dlixlj_dli_dlj = bi && bjtmp && bi != bjtmp ? 1 : 0;
      // 2) PME correction with li*lj -> these are simpler so they are implemented in-line

      // Force storage
      real fij=0;
      real fli=0;
      real fljtmp=0;

      if (r<cutoffs.rCut){
        rEff=r;
        // Softcore OST
        real drlam_dlami=0;
        real drlam_dlamj=0;
        real drijp_drij=1;
        real drijp_dlami=0;
        real drijp_dlamj=0;
        real drijp_drlam=0;
        real d2rijp_drij_dlamj=0;
        real d2rijp_drij_dlami=0;
        real d2rijp_dlami_dlamj=0;
        real d2rijp_drlam_drij=0;
        real d2rijp_drlam2=0;
        real rSoft=SOFTCORERADIUS*(1-lixljtmp);
        if (useSoftCore) {
          dredr=1; // d(rEff) / d(r)
          dredll=0; // d(rEff) / d(lixljtmp)
          if (bi || bjtmp) { // if either is a site
            if (r<rSoft) {
              // Original soft
              real rdivrs=r/rSoft;
              rEff=1-((real)0.5)*rdivrs;
              rEff=rEff*rdivrs*rdivrs*rdivrs+((real)0.5); // Soft-core: rEff = rL * (.5 + (r/rL)^3 - .5*(r/rL)^4)
              dredr=3-2*rdivrs;
              dredr*=rdivrs*rdivrs;
              dredll=rEff-dredr*rdivrs;
              dredll*=-SOFTCORERADIUS; // missing lambda factor corrected later
              rEff*=rSoft;
              // Terms with li or lj in it need to be conditioned
              drlam_dlami = bi ? -SOFTCORERADIUS*dlixlj_dli : 0;
              drlam_dlamj = bjtmp ? -SOFTCORERADIUS*dlixlj_dlj : 0;
              drijp_drlam = -.5*pow(r/rSoft, 4) + pow(r/rSoft,3) +
                rSoft*(2*pow(r,4)/pow(rSoft,5) - 3*pow(r,3)/pow(rSoft,4))+.5;
              d2rijp_drlam_drij = r*r*(6*r/rSoft-6)/pow(rSoft,3);
              d2rijp_drlam2 = r*r*r*(-6*r/rSoft+6)/pow(rSoft, 3);
              // First partials
              drijp_drij = rSoft*(-2.0*pow(r,3)/pow(rSoft,4)+3*pow(r,2)/pow(rSoft,3));
              drijp_dlami = drijp_drlam*drlam_dlami;
              drijp_dlamj = drijp_drlam*drlam_dlamj;
              // Second mixed partials
              d2rijp_drij_dlami = d2rijp_drlam_drij * drlam_dlami;
              d2rijp_drij_dlamj = d2rijp_drlam_drij * drlam_dlamj;
              d2rijp_dlami_dlamj = d2rijp_drlam2 * drlam_dlami * drlam_dlamj; // symmetric
            }
          }
        }

        real rinv=1/rEff;
        // Terms which require calculation for soft-coring - both vdw and elec can be accumulated here
        real dU_drijp, d2U_drijp2, d2U_drijp_dlami, d2U_drijp_dlamj, d2Up_dlami_dlamj;
        // Electrostatics (define the above variables for soft-core OST interactions)
        if (usePME) {
          // lixlj*e14/r = lixlj*e14fac/rEff + (li*lj(erfc(r) - 1)/r + li*lj*erf(r)/r)
          // Ewald Correction - never soft-cored
          rinv = 1/r; // Corrected later
          real br=cutoffs.betaEwald*r;
          real erfcrinv=(fasterfc(br) - 1)*rinv; // missing e14fac compared to pair.cu
          real U_dir = kELECTRIC*pp.qxq*erfcrinv;
          real dU_drij_tmp = -kELECTRIC*pp.qxq*rinv*(erfcrinv+((real)1.128379167095513)*cutoffs.betaEwald*expf(-br*br));
          real d2U_drij_dlami = bi ? ljtmp * dU_drij_tmp : 0;
          real d2U_drij_dlamj = bjtmp ? li * dU_drij_tmp : 0;
          real d2U_dli_dlj = bi && bjtmp ? U_dir : 0; // this doesn't have second condition since li*lj always
          fij += dGdFi * d2U_drij_dlami + dGdFjtmp * d2U_drij_dlamj;
          fli += dGdFjtmp * d2U_dli_dlj;
          fljtmp += dGdFi * d2U_dli_dlj; // implicit product rule mult by 2 if li == lj
          // Coulomb - soft-cored
          rinv = 1/rEff; // Corrected above change to go back to soft-coring
          // lixlj * e14/rEff
          U_dir = kELECTRIC*pp.qxq*pp.e14fac*rinv;
          real dU_drijp_tmp = -U_dir*rinv;
          dU_drijp = lixljtmp*dU_drijp_tmp;
          d2U_drijp2 = 2*U_dir*rinv*rinv;
          d2U_drijp_dlami = dlixlj_dli * dU_drijp_tmp;
          d2U_drijp_dlamj = dlixlj_dlj * dU_drijp_tmp;
          d2Up_dlami_dlamj = dlixlj_dli_dlj * U_dir;
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
        real fij_ost, fli_ost, fljtmp_ost;
        real d2U_drij_dlami = dU_drijp*d2rijp_drij_dlami + (d2U_drijp_dlami + d2U_drijp2*drijp_dlami) * drijp_drij;
        real d2U_drij_dlamj = dU_drijp*d2rijp_drij_dlamj + (d2U_drijp_dlamj + d2U_drijp2*drijp_dlamj) * drijp_drij;
        // Interaction feels i, j, or both histograms
        fij_ost = dGdFi * d2U_drij_dlami + dGdFjtmp * d2U_drij_dlamj;
        // First term is with holding rij' constant
        real d2U_dlami_dlamj_tot = d2Up_dlami_dlamj + (d2U_drijp_dlamj*drijp_dlami + dU_drijp * d2rijp_dlami_dlamj)
          + (d2U_drijp_dlami + d2U_drijp2*drijp_dlami) * drijp_dlamj;
        fli_ost = dGdFjtmp * d2U_dlami_dlamj_tot; // lam_i feels lam_j histogram
        fljtmp_ost = dGdFi * d2U_dlami_dlamj_tot; // lam_j feels lam_i histogram
        // If d2U_dli_dlj != 0 for i=j, above doesn't over-count due to power rule
        // Accumulate ost forces
        fij += fij_ost;
        fli += fli_ost;
        fljtmp += fljtmp_ost;
      }
      atomicAdd(&lambdaForce[b[0]], fli);
      if (b[1]) {
        atomicAdd(&lambdaForce[b[1]], fljtmp);
      }
      at_real3_scaleinc(&force[ii], fij/r,dr);
      at_real3_scaleinc(&force[jj],-fij/r,dr);
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

  getforce_14pair_kernel_oss<flagBox,useSoftCore,usevdWSwitch,usePME> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
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
  getforce_exclusion_pair_kernel_oss<flagBox,false,false,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
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
