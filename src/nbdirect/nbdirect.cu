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
// __global__ void getforce_nbdirect_kernel(int startBlock,int endBlock,int maxPartnersPerBlock,int *blockBounds,int *blockPartnerCount,struct DomdecBlockPartners *blockPartners,struct DomdecBlockVolume *blockVolume,struct NbondPotential *nbonds,int vdwParameterCount,struct VdwPotential *vdwParameters,int *blockExcls,struct Cutoffs cutoffs,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
__global__ void getforce_nbdirect_kernel(
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
  real_e* __restrict__ energy,
  real_f* __restrict__ dGdF
  )
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
  real fij,eij,ost_test_fij,ost_test_eij; // These should be the same
  real lEnergy=0;
  // extern __shared__ real sEnergy[];
  real3 xi,xj,xjtmp;
  real3 fi,fj,fjtmp;
  real fli,flj,fljtmp, ost_test_fli, ost_test_fljtmp;
  int bi,bj,bjtmp;
  real li,lj,ljtmp,lixljtmp;
  real rEff,dredr,dredll; // Soft core stuff
  int exclAddress, exclMask;
  bool OST_flag = true;
  real fij_ost, fli_ost, fljtmp_ost, eij_ost;
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
      if (bi) {
        li=lambda[0xFFFF & bi];
        dGdFi=dGdF[0xFFFF & bi];
      }
    }
    // iBlockVolume=blockVolume[iBlock];

    fi=real3_reset<real3>();
    fli=0;
    fli_ost=0;
    ost_test_fli = 0;

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
        if (bj) {
          lj=lambda[0xFFFF & bj];
          dGdFj=dGdF[0xFFFF & bj];
        }
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
          fljtmp_ost=0;
          ost_test_fljtmp = 0;
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

            if (r<cutoffs.rCut) {
              // Scaling
              if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) { // same site
                if (bi==bjtmp) { // intra-site/inta-block - only scales by one lambda
                  lixljtmp=li;
                } else {
                  lixljtmp=0; // cancels intra-site/inter-block
                }
              } else { // inter-site/alchemical-environment/environment-environment
                lixljtmp=li*ljtmp;
              }

              rEff=r;
              // Softcore OST
              // rSoft derivatives
              real drlam_dlami=0;
              real drlam_dlamj=0;
              // Softcore primary first derivatives
              real drijp_drij=1;
              real drijp_dlami=0;
              real drijp_dlamj=0;
              // First derivative intermediates
              real drijp_drlam=0;
              // Primary second derivatives
              real d2rijp_drij2=0;
              // Primary cross derivatives
              real d2rijp_drij_dlamj=0;
              real d2rijp_drij_dlami=0;
              real d2rijp_dlami_dlamj=0;
              // Second derivative intermediates
              real d2rijp_drlam_drij=0;
              real d2rijp_drlam2=0;
              if (useSoftCore) {
                dredr=1; // d(rEff) / d(r)
                dredll=0; // d(rEff) / d(lixljtmp)
                if (bi || bjtmp) { // if either is a site
                  // real rSoft=(2.0*ANGSTROM*sqrt(4.0))*(1.0-lixljtmp);
                  real rSoft=SOFTCORERADIUS*(1-lixljtmp); // rL
                  if (r<rSoft) {
                    // Original soft
                    real rdivrs=r/rSoft;
                    rEff=1-((real)0.5)*rdivrs;
                    rEff=rEff*rdivrs*rdivrs*rdivrs+((real)0.5); // Soft-core: rEff = rL * (.5 + (r/rL)^3 - .5*(r/rL)^4)
                    dredr=3-2*rdivrs;
                    dredr*=rdivrs*rdivrs;
                    dredll=rEff-dredr*rdivrs;
                    // dredll*=-(2.0*ANGSTROM*sqrt(4.0));
                    dredll*=-SOFTCORERADIUS; // missing lambda factor corrected later
                    rEff*=rSoft;
                    if (OST_flag) {
                      // Intermediates
                      // Terms with li or lj in it need to be conditioned
                      drlam_dlami = bi ? -SOFTCORERADIUS*lixljtmp/li : 0; // Correct
                      drlam_dlamj = bjtmp ? -SOFTCORERADIUS*lixljtmp/ljtmp : 0; // Correct
                      drijp_drlam = -.5*pow(r/rSoft, 4) + pow(r/rSoft,3) +
                        rSoft*(2*pow(r,4)/pow(rSoft,5) - 3*pow(r,3)/pow(rSoft,4))+.5; // Correct
                      d2rijp_drlam_drij = r*r*(6*r/rSoft-6)/pow(rSoft,3);
                      d2rijp_drlam2 = r*r*r*(-6*r/rSoft+6)/pow(rSoft, 3);
                      // First partials
                      drijp_drij = rSoft*(-2.0*pow(r,3)/pow(rSoft,4)+3*pow(r,2)/pow(rSoft,3)); // Correct
                      drijp_dlami = drijp_drlam*drlam_dlami;
                      drijp_dlamj = drijp_drlam*drlam_dlamj;
                      // Second partials
                      d2rijp_drij2 = -r*(5*r/rSoft-6)/pow(rSoft,2);
                      // Second mixed partials
                      d2rijp_drij_dlami = d2rijp_drlam_drij * drlam_dlami;
                      d2rijp_drij_dlamj = d2rijp_drlam_drij * drlam_dlamj;
                      d2rijp_dlami_dlamj = d2rijp_drlam2 * drlam_dlami * drlam_dlamj; // symmetric
                    }
                  }
                }
              }
              rinv=1/rEff;

              // interaction
              // Electrostatics
              /*fij=-kELECTRIC*inp.q*jtmpnp_q*rinv*rinv;
              if (bi || bjtmp || energy) {
                eij=kELECTRIC*inp.q*jtmpnp_q*rinv;
              }*/
              if (usePME) {
                real br=cutoffs.betaEwald*rEff;
                // real erfcrinv=erfcf(br)*rinv;
                real erfcrinv=fasterfc(br)*rinv;
                // fij=-kELECTRIC*inp.q*jtmpnp_q*(erfcrinv+(2/sqrt(M_PI))*cutoffs.betaEwald*expf(-br*br))*rinv;
                // fij=-kELECTRIC*inp.q*jtmpnp_q*(erfcrinv+1.128379167095513f*cutoffs.betaEwald*expf(-br*br))*rinv;
                // fij=-kELECTRIC*inp.q*jtmpnp_q*(erfcrinv+((real)(2/sqrt(M_PI)))*cutoffs.betaEwald*expf(-br*br))*rinv;
                fij=-kELECTRIC*inp.q*jtmpnp_q*(erfcrinv+((real)1.128379167095513)*cutoffs.betaEwald*exp(-br*br))*rinv;
                if (bi || bjtmp || energy) {
                  eij=kELECTRIC*inp.q*jtmpnp_q*erfcrinv;
                }
              }
              else {
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
                fij=(rEff<=cutoffs.rSwitch)?
                  -kELECTRIC*inp.q*jtmpnp_q*rinv*rinv:
                  -kELECTRIC*inp.q*jtmpnp_q*rinv*(Aconst*rinv+Bconst*rEff+3*Cconst*r3+5*Dconst*r5);
                if (bi || bjtmp || energy) {
                  eij=(rEff<=cutoffs.rSwitch)?
                    kELECTRIC*inp.q*jtmpnp_q*(rinv+dvc):
                    kELECTRIC*inp.q*jtmpnp_q*(Aconst*(rinv-1/cutoffs.rCut)+Bconst*(cutoffs.rCut-rEff)+Cconst*(roff2*cutoffs.rCut-r3)+Dconst*(roff2*roff2*cutoffs.rCut-r5));
                }
              }

              // OST direct ewald forces
              if (OST_flag) {
                // Intermediates
                real eBeta = cutoffs.betaEwald;
                real v_ewald = kELECTRIC*inp.q*jtmpnp_q*fasterfc(eBeta*rEff)*rinv;
                real derf = -2*eBeta*exp(-eBeta*eBeta*rEff*rEff)/sqrt(M_PI)-rinv*fasterfc(eBeta*rEff);
                real dv_ewald_drijp = kELECTRIC*inp.q*jtmpnp_q*rinv*derf;
                real d2v_ewald_drijp2 = kELECTRIC*inp.q*jtmpnp_q*
                  (exp(-eBeta*eBeta*rEff*rEff)*(-4*eBeta*eBeta*eBeta/sqrt(M_PI) - 4*eBeta*rinv*rinv/sqrt(M_PI))
                  + rinv*rinv*rinv*2*fasterfc(eBeta*rEff));
                real U_ewaldp = lixljtmp*v_ewald;
                real dU_ewald_drijp = lixljtmp*dv_ewald_drijp;
                // Dividing here handles the intra-sub scaling
                real dU_ewaldp_dlami = bi ? lixljtmp/li*v_ewald : 0; // derivative at constant rijp
                real dU_ewaldp_dlamj = bjtmp ? lixljtmp/ljtmp*v_ewald : 0; // derivative at constant rijp
                real d2U_ewald_drijp2 = lixljtmp*d2v_ewald_drijp2;
                real d2U_ewald_drijp_dlami = bi ? lixljtmp/li*dv_ewald_drijp + d2U_ewald_drijp2*drijp_drij*drijp_dlami : 0;
                real d2U_ewald_drijp_dlamj = bjtmp ? lixljtmp/li*dv_ewald_drijp + d2U_ewald_drijp2*drijp_drij*drijp_dlamj : 0;
                real d2U_ewaldp_dlamj_dlami = bi && bjtmp && bi != bjtmp ? v_ewald : 0; // symmetric, intra-sub only scaled by lambda 1 time

                // True Potential
                ost_test_eij = U_ewaldp;
                // Forces
                real dU_ewald_drij = dU_ewald_drijp*drijp_drij;
                ost_test_fij = dU_ewald_drij;
                real dU_ewald_dlami =  dU_ewaldp_dlami +  dU_ewald_drijp*drijp_dlami;
                ost_test_fli += dU_ewald_dlami;
                real dU_ewald_dlamj = dU_ewaldp_dlamj + dU_ewald_drijp*drijp_dlamj;
                ost_test_fljtmp += bi == bjtmp ? 0.0 : dU_ewald_dlamj;
                // OST forces - every atom feels both of these scaled by lami and lamj's histogram force
                real d2U_ewald_drij_dlami = d2U_ewald_drijp_dlami*drijp_drij + d2U_ewald_drijp2*drijp_drij*drijp_dlami +
                  dU_ewald_drijp*d2rijp_drij_dlami;
                real d2U_ewald_drij_dlamj = d2U_ewald_drijp_dlamj*drijp_drij + d2U_ewald_drijp2*drijp_drij*drijp_dlamj +
                  dU_ewald_drijp*d2rijp_drij_dlamj;
                real d2U_ewald_dlami_dlamj = d2U_ewaldp_dlamj_dlami + d2U_ewald_drijp_dlamj*drijp_dlami + dU_ewald_drijp*d2rijp_dlami_dlamj
                + d2U_ewald_drijp_dlami*drijp_dlamj;
                real d2U_ewald_dlamj_dlami = d2U_ewaldp_dlamj_dlami + d2U_ewald_drijp_dlami*drijp_dlamj + dU_ewald_drijp*d2rijp_dlami_dlamj
                + d2U_ewald_drijp_dlamj*drijp_dlami;
                // Add forces
                fij_ost = dGdFi * d2U_ewald_drij_dlami + dGdFjtmp * d2U_ewald_drij_dlamj;
                fli_ost += dGdFi * d2U_ewald_dlami_dlamj;
                fljtmp_ost += dGdFjtmp * d2U_ewald_dlami_dlamj;
                if (abs(d2U_ewald_dlami_dlamj - d2U_ewald_dlamj_dlami) > 1e-10) {
                  printf("bi = %d, bj = %d, Ewald lambda hessian: ij = %f    ji = %f\n", bi, bjtmp, d2U_ewald_dlami_dlamj, d2U_ewald_dlamj_dlami);
                }
              }

              // Van der Waals
              real rinv3=rinv*rinv*rinv;
              real rinv6=rinv3*rinv3;
              /*fij+=-(12*(vdwp.c12*rinv6)-6*(vdwp.c6))*rinv6*rinv;
              if (bi || bjtmp || energy) {
                eij+=(vdwp.c12*rinv6-vdwp.c6)*rinv6;
              }*/
              // See charmm/source/domdec/enbxfast.F90, functions calc_vdw_constants, vdw_attraction, vdw_repulsion
              real rCut = cutoffs.rCut;
              real rSwitch = cutoffs.rSwitch;
              real rCut3=cutoffs.rCut*cutoffs.rCut*cutoffs.rCut;
              real rSwitch3=cutoffs.rSwitch*cutoffs.rSwitch*cutoffs.rSwitch;

              if (rEff<cutoffs.rSwitch) { // Normal soft-core vdw interaction
                fij+=(6*vdwp.c6-12*vdwp.c12*rinv6)*rinv6*rinv;
                if (bi || bjtmp || energy) {
                  real dv6=usevdWSwitch?0:1/(rCut3*rSwitch3); // force switch causes potential shift
                  dv6 = 0.0;
                  eij+=vdwp.c12*(rinv6*rinv6-dv6*dv6)-vdwp.c6*(rinv6-dv6);
                }
              }
              else { // Tapered
                if ( !usevdWSwitch ) { // Force switch
                  real k6=rCut3/(rCut3-rSwitch3);
                  real k12=rCut3*rCut3/(rCut3*rCut3-rSwitch3*rSwitch3);
                  real rCutinv3=1/rCut3;
                  fij+=(6*vdwp.c6*k6*(rinv3-rCutinv3)*rinv3-12*vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*rinv6)*rinv;
                  if (bi || bjtmp || energy) {
                    // e=A*rCut^6*(rCut^6/(rCut^3-rSwitch^3))*(1/rEff^6-1/rCut^6)^2-
                    //   B*(rCut^3/(rCut^3-rSwitch^3))*(1/rEff^3-1/rCut^3)^2
                    eij+=vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*(rinv6-rCutinv3*rCutinv3)-vdwp.c6*k6*(rinv3-rCutinv3)*(rinv3-rCutinv3);
                  }
                }
                else { // Potential Switch
                  real c2ofnb=cutoffs.rCut*cutoffs.rCut;
                  real c2onnb=cutoffs.rSwitch*cutoffs.rSwitch;
                  real rul3=(c2ofnb-c2onnb)*(c2ofnb-c2onnb)*(c2ofnb-c2onnb);
                  real rul12 = 12/rul3;
                  real rijl = c2onnb - rEff * rEff;
                  real riju = c2ofnb - rEff * rEff;
                  // v_taper = (rCut^2 - rEff^2)^2*(rCut^2-rEff^2-3*(rSwitch^2-rEff^2)/(rCut^2-rSwitch^2)^3
                  real fsw = riju*riju*(riju-3*rijl)/rul3;
                  // dv_taper_drij' = 12*(rSwitch^2-rEff^2)*(rCut^2-rEff^2)/(rCut^2-rSwitch^2)^3
                  real dfsw = rijl*riju*rul12;
                  fij+=fsw*(6*vdwp.c6-12*vdwp.c12*rinv6)*rinv6*rinv\
                    +dfsw*(vdwp.c12*rinv6-vdwp.c6)*rinv6;
                  if (bi || bjtmp || energy) {
                    eij+=fsw*(vdwp.c12*rinv6-vdwp.c6)*rinv6;
                  }
                }
              }
              fij*=lixljtmp;

              // OST vdw forces
              if (OST_flag){
                // Intermediates
                // Taper
                bool taper = rEff<cutoffs.rSwitch;
                real v_taper, dv_taper_drijp, d2v_taper_drijp2;
                if (usevdWSwitch) { // Potential Switch
                  v_taper = taper ? 1.0 :
                    pow(rCut*rCut - rEff*rEff, 2)
                    *(rCut*rCut + 2*rEff*rEff - 3*rSwitch*rSwitch)/pow(rCut*rCut - rSwitch*rSwitch, 3); // Correct
                  dv_taper_drijp = taper ? 0.0 :
                    12*(rCut*rCut - rEff*rEff) * (rSwitch*rSwitch - rEff*rEff)
                  / pow(rCut*rCut - rSwitch*rSwitch, 3); // Correct - needed to take out extra factor of rEff?
                  d2v_taper_drijp2 = taper ? 0.0 :
                    4 * (8*(rEff*rEff)*(rEff*rEff - rCut*rCut)
                    + pow(rEff*rEff - rCut*rCut, 2)
                    + (3*rEff*rEff - rCut*rCut)*(2*rEff*rEff + rCut*rCut - 3*rSwitch*rSwitch));
                }
                // VdW 12-6
                real v_12_6 = rinv6*(vdwp.c12*rinv6 - vdwp.c6); // A = c12, B = c6
                real dv_12_6_drijp = 6*(-2*vdwp.c12*rinv6 + vdwp.c6)*rinv6*rinv;
                real d2v_12_6_drijp2 = 6*(26*vdwp.c12*rinv6 - 7*vdwp.c6)*rinv6*rinv*rinv;
                // VdW potential
                real U_vdw = lixljtmp*v_12_6*v_taper;
                real dU_vdw_drijp = lixljtmp*(dv_taper_drijp*v_12_6 + v_taper*dv_12_6_drijp);
                real dU_vdwp_dlami = bi ? lixljtmp/li*v_12_6*v_taper : 0;
                real dU_vdwp_dlamj = bjtmp ? lixljtmp/ljtmp*v_12_6*v_taper : 0;
                real d2U_vdw_drij2p = lixljtmp*(d2v_12_6_drijp2*v_taper + 2*dv_12_6_drijp*dv_taper_drijp + v_12_6*d2v_taper_drijp2);
                real d2U_vdw_drijp_dlami = bi ? lixljtmp/li*(dv_12_6_drijp*v_taper + v_12_6*dv_taper_drijp) : 0;
                real d2U_vdw_drijp_dlamj = bjtmp ? lixljtmp/ljtmp*(dv_12_6_drijp*v_taper + v_12_6*dv_taper_drijp) : 0;
                real d2U_vdwp_dlami_dlamj = bi && bjtmp && bi!=bjtmp? v_12_6*v_taper : 0; // symmetric

                // True Potential
                ost_test_eij += U_vdw;
                // First derivatives - Regular forces - includes all interaction components
                real dU_vdw_drij = dU_vdw_drijp*drijp_drij;
                ost_test_fij += dU_vdw_drij;
                real dU_vdw_dlami = dU_vdwp_dlami + dU_vdw_drijp*drijp_dlami;
                ost_test_fli += dU_vdw_dlami;
                real dU_vdw_dlamj = dU_vdwp_dlamj + dU_vdw_drijp*drijp_dlamj;
                ost_test_fljtmp += bi == bjtmp ? 0.0 : dU_vdw_dlamj;
                // Second derivatives - OST forces
                real d2U_vdw_drij_dlami = d2U_vdw_drijp_dlami*drijp_drij + dU_vdw_drijp*d2rijp_drij_dlami +
                  drijp_dlami*(d2U_vdw_drij2p*drijp_drij + dU_vdw_drijp*d2rijp_drij2);
                real d2U_vdw_drij_dlamj = d2U_vdw_drijp_dlamj*drijp_drij + dU_vdw_drijp*d2rijp_drij_dlamj +
                  drijp_dlamj*(d2U_vdw_drij2p*drijp_drij + dU_vdw_drijp*d2rijp_drij2);
                real d2U_vdw_dlami_dlamj = d2U_vdwp_dlami_dlamj + d2U_vdw_drijp_dlamj*drijp_dlami +
                  dU_vdw_drijp*d2rijp_dlami_dlamj
                + drijp_dlamj * (d2U_vdw_drijp_dlami + d2U_vdw_drij2p*drijp_dlami);
                real d2U_vdw_dlamj_dlami = d2U_vdwp_dlami_dlamj + d2U_vdw_drijp_dlami*drijp_dlamj +
                  dU_vdw_drijp*d2rijp_dlami_dlamj
                + drijp_dlami * (d2U_vdw_drijp_dlamj + d2U_vdw_drij2p*drijp_dlamj);
                // Add forces
                fij_ost += dGdFi * d2U_vdw_drij_dlami + dGdFjtmp * d2U_vdw_drij_dlamj;
                fli_ost += dGdFi * d2U_vdw_dlami_dlamj;
                fljtmp_ost += dGdFjtmp * d2U_vdw_dlami_dlamj;
                if (abs(d2U_vdw_dlami_dlamj - d2U_vdw_dlamj_dlami) > 1e-10) {
                  printf("bi = %d, bj = %d, VdW lambda hessian: ij = %f    ji = %f\n", bi, bjtmp, d2U_vdw_dlami_dlamj, d2U_vdw_dlamj_dlami);
                }
              }

              // Lambda force
              if (bi || bjtmp) { // Alchemical interaction
                if (useSoftCore) {
                  fljtmp= eij + fij*dredll; // dU/dL + dU/drij' * drij'/dl (neither has lambda scaling)
                } else {
                  fljtmp=eij; // dU/dL (no lambda scaling)
                }
                // Scaling
                if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) { // Same site
                  if (bi==bjtmp) { // Same sub
                    fli+=fljtmp; // This interaction is only scaled by a single lambda
                  }
                  fljtmp=0; // No intra-site interactions
                } else { // Different site
                  fli+=ljtmp*fljtmp; // force on lam.i drops lam.i factor
                  fljtmp*=li; // dU/dLj drops lam.j factor
                }
                // OST forces after scaling since they already include it
                fli += fli_ost;
                fljtmp += fljtmp_ost;
              }

              // Spatial force
              if (useSoftCore) {
                rinv=1/r; // chain rule projection onto x,y,z-axis is based on rij not rEff
                fij*=dredr; // previous terms have ignored chain rule term
              }
              fij += fij_ost; // after all scaling since ost forces has all chain rule terms up to rij
              real3_scaleinc(&fi, fij*rinv,dr); // drij/dx = sqrt(x^2+y^2+z^2)^-1/2 * dx
              fjtmp=real3_scale<real3>(-fij*rinv,dr); // Newton's laws, equal opposite

              // Check if energy and forces are correct
              if (bi != 0 && abs(ost_test_fli - fli) > 1e-10) {
                printf("li = %f, lj = %f, lixlj = %f, bi %d, bj %d, taper? %d, LiForce: mine = %f, theirs = %f\n", li, ljtmp, lixljtmp, bi, bjtmp, rEff < cutoffs.rSwitch, ost_test_fli, fli);
              }
              if (bjtmp != 0 && abs(ost_test_fljtmp - fljtmp) > 1e-10) {
                printf("li = %f, lj = %f, lixlj = %f, bi %d, bj %d, taper? %d, LjForce: mine = %f, theirs = %f\n", li, ljtmp, lixljtmp, bi, bjtmp, rEff < cutoffs.rSwitch, ost_test_fljtmp, fljtmp);
              }
              if (abs(ost_test_eij - lixljtmp*eij) > 1e-10) {
                printf("li = %f, lj = %f, lixlj = %f, bi %d, bj %d, taper? %d, Energy: mine = %f, theirs = %f\n", li, ljtmp, lixljtmp, bi, bjtmp, rEff < cutoffs.rSwitch, ost_test_eij, eij);
              }
              if (abs(ost_test_fij - fij) > 1e-10) {
                printf("li = %f, lj = %f, lixlj = %f, bi %d, bj %d, taper? %d, Force: mine = %f, theirs = %f\n", li, ljtmp, lixljtmp, bi, bjtmp, rEff < cutoffs.rSwitch, ost_test_fij, fij);
              }

              // Energy, if requested
              if (energy) {
                // if (!(lixljtmp*eij>-5000000)) printf("lixljtmp*eij=%f lixljtmp=%f eij=%f\n",lixljtmp*eij,lixljtmp,eij);
                lEnergy+=lixljtmp*eij;
              }
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

  // Energy, if requested
  if (energy) {
    // Use of shared memory here causes error when getforce_nbrecip_gather is executed concurrently. Whatever CUDA...
    // #warning "Using reduction without shared memory"
    // real_sum_reduce(lEnergy,sEnergy,energy);
    real_sum_reduce(lEnergy,energy);
  }
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
void getforce_nbdirectTTTT(System *system,box_type box,bool calcEnergy)
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
  real_e *pEnergy=NULL;
  real_f *dGdF = system->msld->dGdF_d;

  if (r->calcTermFlag[eenbdirect]==false) return;

  if (calcEnergy) {
    // shMem=(1<<WARPSPERBLOCK)*sizeof(real);
    shMem=0;
    pEnergy=s->energy_d+eenbdirect;
  }
  getforce_nbdirect_kernel<flagBox,useSoftCore,usevdWSwitch,usePME><<<((32<<WARPSPERBLOCK)*N+(32<<WARPSPERBLOCK)-1)/(32<<WARPSPERBLOCK),(32<<WARPSPERBLOCK),shMem,r->nbdirectStream>>>(startBlock,endBlock,d->maxPartnersPerBlock,d->blockBounds_d,d->blockPartnerCount_d,d->blockPartners_d,d->blockVolume_d,d->localNbonds_d,p->vdwParameterCount,
#ifdef USE_TEXTURE
    p->vdwParameters_tex,
#else
    p->vdwParameters_d,
#endif
    system->domdec->blockExcls_d,system->run->cutoffs,d->localPosition_d,d->localForce_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy, dGdF);

  system->domdec->unpack_forces(system);
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,typename box_type>
void getforce_nbdirectTTT(System *system,box_type box,bool calcEnergy)
{
  if (system->run->usePME) {
    getforce_nbdirectTTTT<flagBox,useSoftCore,usevdWSwitch,true>(system,box,calcEnergy);
  } else {
    getforce_nbdirectTTTT<flagBox,useSoftCore,usevdWSwitch,false>(system,box,calcEnergy);
  }
}

template <bool flagBox,bool useSoftCore,typename box_type>
void getforce_nbdirectTT(System *system,box_type box,bool calcEnergy)
{
  if (!system->run->vfSwitch) {
    getforce_nbdirectTTT<flagBox,useSoftCore,true>(system,box,calcEnergy);
  } else {
    getforce_nbdirectTTT<flagBox,useSoftCore,false>(system,box,calcEnergy);
  }
}

template <bool flagBox,typename box_type>
void getforce_nbdirectT(System *system,box_type box,bool calcEnergy)
{
  if (system->msld->useSoftCore) {
    getforce_nbdirectTT<flagBox,true>(system,box,calcEnergy);
  } else {
    getforce_nbdirectTT<flagBox,false>(system,box,calcEnergy);
  }
}

void getforce_nbdirect(System *system,bool calcEnergy)
{
  calcEnergy = true;
  if (system->state->typeBox) {
    getforce_nbdirectT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_nbdirectT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

template <typename real_type>
__global__ void getforce_nbdirect_reduce_kernel(int atomCount,int idCount,real_type *force)
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

void getforce_nbdirect_reduce(System *system,bool calcEnergy)
{
  State *s=system->state;
  Run *r=system->run;
  int N=3*s->atomCount+2*s->lambdaCount;

  getforce_nbdirect_reduce_kernel<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->forceBuffer_d);

  if (calcEnergy) {
    N=eeend;
    getforce_nbdirect_reduce_kernel<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->energy_d);
  }
}
