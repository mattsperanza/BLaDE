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

template <bool flagBox,bool calcAlch,bool useSoftCore,bool usevdWSwitch,bool usePME,bool calcEnergy,typename box_type>
// __global__ void getforce_nbdirect_kernel(int startBlock,int endBlock,int maxPartnersPerBlock,int *blockBounds,int *blockPartnerCount,struct DomdecBlockPartners *blockPartners,struct DomdecBlockVolume *blockVolume,struct NbondPotential *nbonds,int vdwParameterCount,struct VdwPotential *vdwParameters,int *blockExcls,struct Cutoffs cutoffs,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
__global__ void getforce_nbdirect_kernel_special(
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
  real3_f* __restrict__ force_ss,
  real3_f* __restrict__ force_su,
  real3_f* __restrict__ force_uu,
  box_type box,
  const real* __restrict__ lambda,
  real_f* __restrict__ lambdaForce,
  real_f* __restrict__ lambdaForce_ss,
  real_f* __restrict__ lambdaForce_su,
  real_f* __restrict__ lambdaForce_uu,
  real_e* __restrict__ U_ss,
  real_e* __restrict__ U_su,
  real_e* __restrict__ U_uu,
  real_e* __restrict__ energy)
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
  int jtmpnp_selection;
  real fij,eij;
  real lEnergy=0;
  real lEnergy_ss=0;
  real lEnergy_su=0;
  real lEnergy_uu=0;
  // extern __shared__ real sEnergy[];
  real3 xi,xj,xjtmp;
  real3 fi,fj,fjtmp;
  real3 fi_ss, fj_ss,fjtmp_ss;
  real3 fi_su, fj_su,fjtmp_su;
  real3 fi_uu, fj_uu,fjtmp_uu;
  real fli,flj,fljtmp;
  real fli_ss, flj_ss, fljtmp_ss;
  real fli_su, flj_su, fljtmp_su;
  real fli_uu, flj_uu, fljtmp_uu;
  int bi,bj,bjtmp;
  real li,lj,ljtmp,lixljtmp;
  real rEff,dredr,dredll; // Soft core stuff
  int exclAddress, exclMask;

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
      if (calcAlch) {
      bi=inp.siteBlock;
      li=1;
      if (bi) li=lambda[0xFFFF & bi];
      }
    }
    // iBlockVolume=blockVolume[iBlock];

    fi=real3_reset<real3>();
    fi_ss=real3_reset<real3>();
    fi_su=real3_reset<real3>();
    fi_uu=real3_reset<real3>();
    if (calcAlch) {
      fli=0;
      fli_ss=0;
      fli_su=0;
      fli_uu=0;
    }

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
        if (calcAlch) {
        bj=jnp.siteBlock;
        lj=1;
        if (bj) lj=lambda[0xFFFF & bj];
        }
      }
      bool jFlag=check_proximity(iBlockVolume,xj,cutoffs.rCut*cutoffs.rCut);

      fj=real3_reset<real3>();
      if (calcAlch) {
        flj=0;
        flj_ss=0;
        flj_su=0;
        flj_uu=0;
      }

      for (jtmp=0; jtmp<jCount; jtmp++) {
        if (__shfl_sync(0xFFFFFFFF,jFlag,jtmp)) {
          jtmpnp_q=__shfl_sync(0xFFFFFFFF,jnp.q,jtmp);
          jtmpnp_typeIdx=__shfl_sync(0xFFFFFFFF,jnp.typeIdx,jtmp);
          jtmpnp_selection=__shfl_sync(0xFFFFFFFF,jnp.selection,jtmp);
          xjtmp.x=__shfl_sync(0xFFFFFFFF,xj.x,jtmp);
          xjtmp.y=__shfl_sync(0xFFFFFFFF,xj.y,jtmp);
          xjtmp.z=__shfl_sync(0xFFFFFFFF,xj.z,jtmp);
          if (calcAlch) {
          bjtmp=__shfl_sync(0xFFFFFFFF,bj,jtmp);
          ljtmp=__shfl_sync(0xFFFFFFFF,lj,jtmp);
          }

          fjtmp=real3_reset<real3>();
          if (calcAlch){
            fljtmp=0;
            fljtmp_ss=0;
            fljtmp_su=0;
          }
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
              if (calcAlch) {
              if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) {
                if (bi==bjtmp) {
                  lixljtmp=li;
                } else {
                  lixljtmp=0;
                }
              } else {
                lixljtmp=li*ljtmp;
              }
              }

              rEff=r;
              if (calcAlch && useSoftCore) {
                dredr=1; // d(rEff) / d(r)
                dredll=0; // d(rEff) / d(lixljtmp)
                if (bi || bjtmp) {
                  // real rSoft=(2.0*ANGSTROM*sqrt(4.0))*(1.0-lixljtmp);
                  real rSoft=SOFTCORERADIUS*(1-lixljtmp);
                  if (r<rSoft) {
                    real rdivrs=r/rSoft;
                    rEff=1-((real)0.5)*rdivrs;
                    rEff=rEff*rdivrs*rdivrs*rdivrs+((real)0.5);
                    dredr=3-2*rdivrs;
                    dredr*=rdivrs*rdivrs;
                    dredll=rEff-dredr*rdivrs;
                    // dredll*=-(2.0*ANGSTROM*sqrt(4.0));
                    dredll*=-SOFTCORERADIUS;
                    rEff*=rSoft;
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
                fij=-kELECTRIC*inp.q*jtmpnp_q*(erfcrinv+((real)1.128379167095513)*cutoffs.betaEwald*expf(-br*br))*rinv;
                if (calcEnergy || (calcAlch && (bi || bjtmp))) {
                  eij=kELECTRIC*inp.q*jtmpnp_q*erfcrinv;
                }
              } else {
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
                if (calcEnergy || (calcAlch && (bi || bjtmp))) {
                  eij=(rEff<=cutoffs.rSwitch)?
                    kELECTRIC*inp.q*jtmpnp_q*(rinv+dvc):
                    kELECTRIC*inp.q*jtmpnp_q*(Aconst*(rinv-1/cutoffs.rCut)+Bconst*(cutoffs.rCut-rEff)+Cconst*(roff2*cutoffs.rCut-r3)+Dconst*(roff2*roff2*cutoffs.rCut-r5));
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
              real rCut3=cutoffs.rCut*cutoffs.rCut*cutoffs.rCut;
              real rSwitch3=cutoffs.rSwitch*cutoffs.rSwitch*cutoffs.rSwitch;

              if (rEff<cutoffs.rSwitch) {
                fij+=(6*vdwp.c6-12*vdwp.c12*rinv6)*rinv6*rinv;
                if (calcEnergy || (calcAlch && (bi || bjtmp))) {
                  real dv6=(usevdWSwitch?0:1/(rCut3*rSwitch3));
                  eij+=vdwp.c12*(rinv6*rinv6-dv6*dv6)-vdwp.c6*(rinv6-dv6);
                }
              } else {
                if ( !usevdWSwitch ) {
                  real k6=rCut3/(rCut3-rSwitch3);
                  real k12=rCut3*rCut3/(rCut3*rCut3-rSwitch3*rSwitch3);
                  real rCutinv3=1/rCut3;
                  fij+=(6*vdwp.c6*k6*(rinv3-rCutinv3)*rinv3-12*vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*rinv6)*rinv;
                  if (calcEnergy || (calcAlch && (bi || bjtmp))) {
                    eij+=vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*(rinv6-rCutinv3*rCutinv3)-vdwp.c6*k6*(rinv3-rCutinv3)*(rinv3-rCutinv3);
                  }
                } else {
                  real c2ofnb=cutoffs.rCut*cutoffs.rCut;
                  real c2onnb=cutoffs.rSwitch*cutoffs.rSwitch;
                  real rul3=(c2ofnb-c2onnb)*(c2ofnb-c2onnb)*(c2ofnb-c2onnb);
                  real rul12 = 12/rul3;
                  real rijl = c2onnb - rEff * rEff;
                  real riju = c2ofnb - rEff * rEff;
                  real fsw = riju*riju*(riju-3*rijl)/rul3;
                  real dfsw = rijl*riju*rul12;
                  fij+=fsw*(6*vdwp.c6-12*vdwp.c12*rinv6)*rinv6*rinv\
                    +dfsw*(vdwp.c12*rinv6-vdwp.c6)*rinv6;
                  if (calcEnergy || (calcAlch && (bi || bjtmp))) {
                    eij+=fsw*(vdwp.c12*rinv6-vdwp.c6)*rinv6;
                  }
                }
              }
              if (calcAlch) fij*=lixljtmp;

              // Lambda force
              real fli_tmp = 0;
              if (calcAlch && (bi || bjtmp)) {
                if (useSoftCore) {
                  fljtmp=eij+fij*dredll;
                } else {
                  fljtmp=eij;
                }
                if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) {
                  if (bi==bjtmp) {
                    fli+=fljtmp;
                    fli_tmp+=fljtmp;
                  }
                  fljtmp=0;
                } else {
                  fli+=ljtmp*fljtmp;
                  fli_tmp+=ljtmp*fljtmp;
                  fljtmp*=li;
                }
              }

              // Spatial force
              if (calcAlch && useSoftCore) {
                rinv=1/r;
                fij*=dredr;
              }
              real3_scaleinc(&fi, fij*rinv,dr);
              fjtmp=real3_scale<real3>(-fij*rinv,dr);

              // current interaction stored in eij & fij
              // Calculate if this is a ss (>2), su (1), or uu (0) interaction
              int t = inp.selection + jtmpnp_selection;
              int jidx = 32*jBlock+jtmp;
              if(t>=2){ 
                real3_scaleinc(&fi_ss, fij*rinv, dr);
                at_real3_scaleinc(&force_ss[jidx], -fij*rinv, dr);
                if(calcAlch) {
                  if (bi) fli_ss += fli_tmp;
                  //if(bjtmp) atomicAdd(&lambdaForce_ss[0xFFFF & bjtmp], fljtmp); // significant noise
                  if(bjtmp) fljtmp_ss += fljtmp;
                  lEnergy_ss += lixljtmp*eij;
                } else if (calcEnergy) {
                  lEnergy_ss += eij;
                }
              } else if (t==1) {
                real3_scaleinc(&fi_su, fij*rinv, dr);
                at_real3_scaleinc(&force_su[jidx], -fij*rinv, dr);
                if(calcAlch) {
                  if (bi) fli_su += fli_tmp;
                  //if (bjtmp) atomicAdd(&lambdaForce_su[0xFFFF & bjtmp], fljtmp); // significant noise
                  if(bjtmp) fljtmp_ss += fljtmp;
                  lEnergy_su += lixljtmp*eij;
                } else if (calcEnergy) {
                  lEnergy_su += eij;
                }
              } 

              // Energy, if requested
              if (calcEnergy) {
                // if (!(lixljtmp*eij>-5000000)) printf("lixljtmp*eij=%f lixljtmp=%f eij=%f\n",lixljtmp*eij,lixljtmp,eij);
                if (calcAlch) {
                lEnergy+=lixljtmp*eij;
                } else {
                lEnergy+=eij;
                }
              }
            }
          }
          __syncwarp();
          ////// x 
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,1);
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,2);
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,4);
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,8);
          fjtmp.x+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.x,16);
          if (iThread==jtmp) fj.x=fjtmp.x;
          ////// y
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,1);
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,2);
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,4);
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,8);
          fjtmp.y+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.y,16);
          if (iThread==jtmp) fj.y=fjtmp.y;
          ////// z
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,1);
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,2);
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,4);
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,8);
          fjtmp.z+=__shfl_xor_sync(0xFFFFFFFF,fjtmp.z,16);
          if (iThread==jtmp) fj.z=fjtmp.z;
          // Alchem
          if (calcAlch && bjtmp) {
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,1);
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,2);
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,4);
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,8);
            fljtmp+=__shfl_xor_sync(0xFFFFFFFF,fljtmp,16);
            if (iThread==jtmp) flj=fljtmp;
            // accumulate like this to avoid significant noise
            // ss
            fljtmp_ss+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ss,1);
            fljtmp_ss+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ss,2);
            fljtmp_ss+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ss,4);
            fljtmp_ss+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ss,8);
            fljtmp_ss+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_ss,16);
            if (iThread==jtmp) flj_ss=fljtmp_ss;
            // su
            fljtmp_su+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_su,1);
            fljtmp_su+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_su,2);
            fljtmp_su+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_su,4);
            fljtmp_su+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_su,8);
            fljtmp_su+=__shfl_xor_sync(0xFFFFFFFF,fljtmp_su,16);
            if (iThread==jtmp) flj_su=fljtmp_su;
          }
        }
      }
      __syncwarp();
      if ((iThread)<jCount) {
        if (calcAlch && bj) {
          atomicAdd(&lambdaForce[0xFFFF & bj],flj);
          atomicAdd(&lambdaForce_ss[0xFFFF & bj],flj_ss);
          atomicAdd(&lambdaForce_su[0xFFFF & bj],flj_su);
          //atomicAdd(&lambdaForce_uu[0xFFFF & bj],flj_uu);
        }
        at_real3_inc(&force[32*jBlock+iThread],fj);
      }
      if (iThread==0) j=atomicInc((unsigned int*)(&jnext),0xFFFFFFFF);
      j=__shfl_sync(0xFFFFFFFF,j,0);
    }
    __syncwarp();
    if ((iThread)<iCount) {
      if (calcAlch && bi) {
        atomicAdd(&lambdaForce[0xFFFF & bi],fli);
        atomicAdd(&lambdaForce_ss[0xFFFF & bi],fli_ss);
        atomicAdd(&lambdaForce_su[0xFFFF & bi],fli_su);
        atomicAdd(&lambdaForce_uu[0xFFFF & bi],fli_uu);
      }
      at_real3_inc(&force[32*iBlock+iThread],fi);
      at_real3_inc(&force_ss[32*iBlock+iThread],fi_ss);
      at_real3_inc(&force_su[32*iBlock+iThread],fi_su);
      at_real3_inc(&force_uu[32*iBlock+iThread],fi_uu);
    }
  }

  // Energy, if requested
  if (calcEnergy) {
    // Use of shared memory here causes error when getforce_nbrecip_gather is executed concurrently. Whatever CUDA...
    // #warning "Using reduction without shared memory"
    // real_sum_reduce(lEnergy,sEnergy,energy);
    real_sum_reduce(lEnergy,energy);
    real_sum_reduce(lEnergy_ss,U_ss);
    real_sum_reduce(lEnergy_su,U_su);
    //real_sum_reduce(lEnergy_uu,U_uu);
  }
}

template <bool flagBox,bool calcAlch,bool useSoftCore,bool usevdWSwitch,bool usePME,bool calcEnergy,typename box_type>
void getforce_nbdirectTTTTT_special(System *system,box_type box)
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

  if (r->calcTermFlag[eenbdirect]==false) return;

  if (calcEnergy) {
    // shMem=(1<<WARPSPERBLOCK)*sizeof(real);
    shMem=0;
    pEnergy=s->energy_d+eenbdirect;
  }
  getforce_nbdirect_kernel_special<flagBox,calcAlch,useSoftCore,usevdWSwitch,usePME,calcEnergy><<<((32<<WARPSPERBLOCK)*N+(32<<WARPSPERBLOCK)-1)/(32<<WARPSPERBLOCK),(32<<WARPSPERBLOCK),shMem,r->nbdirectStream>>>(
    startBlock,endBlock,d->maxPartnersPerBlock,d->blockBounds_d,
    d->blockPartnerCount_d,d->blockPartners_d,d->blockVolume_d,
    d->localNbonds_d,p->vdwParameterCount,
#ifdef USE_TEXTURE
    p->vdwParameters_tex,
#else
    p->vdwParameters_d,
#endif
    system->domdec->blockExcls_d,system->run->cutoffs,
    d->localPosition_d,d->localForce_d,
    d->localdU_ss_d, d->localdU_su_d, d->localdU_uu_d,
    box,
    s->lambda_fd,s->lambdaForce_d,
    s->dU_ss_lambdaForce_d, s->dU_su_lambdaForce_d, s->dU_uu_lambdaForce_d,
    s->U_ss_d, s->U_su_d, s->U_uu_d, pEnergy);

  system->domdec->unpack_forces_special(system);
}

template <bool flagBox,bool calcAlch,bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
void getforce_nbdirectTTTT_special(System *system,box_type box,bool calcEnergy)
{
  if (calcEnergy) {
    getforce_nbdirectTTTTT_special<flagBox,calcAlch,useSoftCore,usevdWSwitch,usePME,true>(system,box);
  } else {
    getforce_nbdirectTTTTT_special<flagBox,calcAlch,useSoftCore,usevdWSwitch,usePME,false>(system,box);
  }
}

template <bool flagBox,bool calcAlch,bool useSoftCore,bool usevdWSwitch,typename box_type>
void getforce_nbdirectTTT_special(System *system,box_type box,bool calcEnergy)
{
  if (system->run->usePME) {
    getforce_nbdirectTTTT_special<flagBox,calcAlch,useSoftCore,usevdWSwitch,true>(system,box,calcEnergy);
  } else {
    getforce_nbdirectTTTT_special<flagBox,calcAlch,useSoftCore,usevdWSwitch,false>(system,box,calcEnergy);
  }
}

template <bool flagBox,bool calcAlch,bool useSoftCore,typename box_type>
void getforce_nbdirectTT_special(System *system,box_type box,bool calcEnergy)
{
  if (!system->run->vfSwitch) {
    getforce_nbdirectTTT_special<flagBox,calcAlch,useSoftCore,true>(system,box,calcEnergy);
  } else {
    getforce_nbdirectTTT_special<flagBox,calcAlch,useSoftCore,false>(system,box,calcEnergy);
  }
}

template <bool flagBox,typename box_type>
void getforce_nbdirectT_special(System *system,box_type box,bool calcEnergy)
{
  if (system->msld->blockCount==1) {
    getforce_nbdirectTT_special<flagBox,false,false>(system,box,calcEnergy);
  } else if (system->msld->useSoftCore) {
    getforce_nbdirectTT_special<flagBox,true,true>(system,box,calcEnergy);
  } else {
    getforce_nbdirectTT_special<flagBox,true,false>(system,box,calcEnergy);
  }
}

void getforce_nbdirect_special(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_nbdirectT_special<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_nbdirectT_special<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

template <typename real_type>
__global__ void getforce_nbdirect_reduce_kernel_special(int atomCount,int idCount,real_type *force)
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

void getforce_nbdirect_reduce_special(System *system,bool calcEnergy)
{
  State *s=system->state;
  Run *r=system->run;
  int N=3*s->atomCount+2*s->lambdaCount;

  getforce_nbdirect_reduce_kernel_special<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->forceBuffer_d);
  getforce_nbdirect_reduce_kernel_special<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->dU_ss_buffer_d);
  getforce_nbdirect_reduce_kernel_special<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->dU_su_buffer_d);

  if (calcEnergy) {
    N=eeend;
    getforce_nbdirect_reduce_kernel_special<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->energy_d);
    N=1;
    getforce_nbdirect_reduce_kernel_special<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->U_ss_d);
    getforce_nbdirect_reduce_kernel_special<<<(N+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(N,system->idCount,s->U_su_d);
  }
}
