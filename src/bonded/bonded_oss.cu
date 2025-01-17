#include <cuda_runtime.h>
#include <math.h>

#include "main/defines.h"
#include "system/system.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"

#include "main/real3.h"

template <bool flagBox,bool soft,typename box_type>
__global__ void getforce_bond_kernel_oss(
    int bond12Count,int bondCount,struct BondPotential *bonds,
    real3 *position,real3_f *force,box_type box,real *lambda,
    real_f *lambdaForce,real softAlpha,real softExp,
    real* dGdF
    )
{
// NYI - maybe energy should be a double
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  real r;
  real3 dr;
  BondPotential bp;
  real fbond;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj;
  int b[2];
  real l[2]={1,1};
  real chain[2]={0,0};
  
  if (i<bondCount) {
    // Geometry
    bp=bonds[i];
    ii=bp.idx[0];
    jj=bp.idx[1];
    xi=position[ii];
    xj=position[jj];
// NOTE #warning "Unprotected division"
    dr=real3_subpbc<flagBox>(xi,xj,box);
// NOTE #warning "Unprotected sqrt"
    r=real3_mag<real>(dr);
    
    // Scaling
    b[0]=0xFFFF & bp.siteBlock[0];
    b[1]=0xFFFF & bp.siteBlock[1];
    if (b[0]) {
      l[0]=lambda[b[0]];
      chain[0] = dGdF[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
        chain[1] = dGdF[b[1]];
      }
    }

    // Skip if both sites are non-alchemical
    // TODO: Implement soft-bond ost derivatives
    if (soft && b[0]) {
      // interaction
      fbond=bp.kb*(r-bp.b0);
      real lambdaExpM1=pow(l[0]*l[1],softExp-1);
      real lambdaExp=lambdaExpM1*l[0]*l[1];
      real softFactor=1/(1+(1-lambdaExp)*softAlpha*(r-bp.b0)*(r-bp.b0));
      real dsfdle=softFactor*softFactor*softAlpha*(r-bp.b0)*(r-bp.b0);
      real dsfdr=-2*softFactor*softFactor*(1-lambdaExp)*softAlpha*(r-bp.b0);
      if (b[0]) {
        lEnergy=((real)0.5)*bp.kb*(r-bp.b0)*(r-bp.b0);
      }
      fbond=lambdaExp*(fbond*softFactor+lEnergy*dsfdr);
      real flambda=lEnergy*softExp*lambdaExpM1*(softFactor+lambdaExp*dsfdle);
      lEnergy*=lambdaExpM1*softFactor; // last factor of l[0]*l[1] shows up later

      // Lambda force
      if (b[0]) {
        //atomicAdd(&lambdaForce[b[0]],l[1]*flambda);
        if (b[1]) {
          //atomicAdd(&lambdaForce[b[1]],l[0]*flambda);
        }
      }

      // Spatial force
// NOTE #warning "division in kernel"
      //at_real3_scaleinc(&force[ii], fbond/r,dr);
      //at_real3_scaleinc(&force[jj],-fbond/r,dr);
    }
    else if (b[0]) {
      // interaction
      lEnergy=((real)0.5)*bp.kb*(r-bp.b0)*(r-bp.b0);
      fbond=bp.kb*(r-bp.b0);
      // Remove lambda scaling for OST derivatives
      //fbond*=l[0]*l[1];

      // Lambda force
      real fij = 0;
      real fli,flj;
      // Environment lambda block always first and is 0 if environment
      //atomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      real d2U_drij_dli = fbond * l[1];
      fij += chain[1] * d2U_drij_dli;
      if (b[1]) {
        //atomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
        real d2U_drij_dlj = fbond * l[0];
        fij += chain[0] * d2U_drij_dli;
        real d2U_dli_dlj = fbond;
        fli = chain[0] * d2U_dli_dlj;
        atomicAdd(&lambdaForce[b[0]], fli);
        flj = chain[1] * d2U_dli_dlj;
        atomicAdd(&lambdaForce[b[1]], flj);
      }

      // Spatial force
      at_real3_scaleinc(&force[ii], fij/r,dr);
      at_real3_scaleinc(&force[jj],-fij/r,dr);
    }
  }
}

template <bool flagBox,typename box_type>
void getforce_bondT_oss(System *system,box_type box)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softAlpha=1/(system->msld->softBondRadius*system->msld->softBondRadius);
  real softExp=system->msld->softBondExponent;
  int N12,N;
  struct BondPotential *bonds;
  int shMem=0;

  if (r->calcTermFlag[eebond]==false && r->calcTermFlag[eeurey]==false) return;

  N12=(r->calcTermFlag[eebond]?p->bond12Count:0);
  N=N12+(r->calcTermFlag[eeurey]?p->bond13Count:0);
  bonds=p->bonds_d+(p->bond12Count-N12);
  if (N>0) getforce_bond_kernel_oss<flagBox,false><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N12,N,bonds,(real3*)s->position_fd,
    (real3_f*)s->force_d,box,s->lambda_fd,
    s->lambdaForce_d,0,1,
    system->msld->dGdF_d);

  N=p->softBondCount;
  N12=(r->calcTermFlag[eebond]?p->softBond12Count:0);
  N=N12+(r->calcTermFlag[eeurey]?p->softBond13Count:0);
  bonds=p->softBonds_d+(p->softBond12Count-N12);
  if (N>0) getforce_bond_kernel_oss<flagBox,true><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N12,N,bonds,(real3*)s->position_fd,
    (real3_f*)s->force_d,box,s->lambda_fd,
    s->lambdaForce_d,softAlpha,softExp,
    system->msld->dGdF_d);
}

void getforce_bond_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_bondT_oss<true>(system,system->state->tricBox_f);
  } else {
    getforce_bondT_oss<false>(system,system->state->orthBox_f);
  }
}



// getforce_angle_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->angles_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <bool flagBox,bool soft,typename box_type>
__global__ void getforce_angle_kernel_oss(int angleCount,struct AnglePotential *angles,real3 *position,
  real3_f *force,box_type box,real *lambda,
  real_f *lambdaForce,real softExp,
  real *dGdF)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk;
  AnglePotential ap;
  real3 drij,drkj;
  real t;
  real dotp, mcrop;
  real3 crop;
  real3 fi,fj,fk;
  real fangle;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi, xj, xk;
  int b[2];
  real l[2]={1,1};

  if (i<angleCount) {
    // Geometry
    ap=angles[i];
    ii=ap.idx[0];
    jj=ap.idx[1];
    kk=ap.idx[2];
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];
    
    drij=real3_subpbc<flagBox>(xi,xj,box);
    drkj=real3_subpbc<flagBox>(xk,xj,box);
    dotp=real3_dot<real>(drij,drkj);
    crop=real3_cross(drij,drkj); // c = a x b
    mcrop=real3_mag<real>(crop);
    t=atan2f(mcrop,dotp);

    // Scaling
    b[0]=0xFFFF & ap.siteBlock[0];
    b[1]=0xFFFF & ap.siteBlock[1];
    if (b[0]) {
      l[0]=lambda[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
      }
    }

    // Interaction
    fangle=ap.kangle*(t-ap.angle0);
    if (b[0]) {
      lEnergy=((real)0.5)*ap.kangle*(t-ap.angle0)*(t-ap.angle0);
    }
    if (soft) {
      fangle*=pow(l[0]*l[1],softExp);
    } else {
      fangle*=l[0]*l[1];
    }

    // Lambda force
    if (soft) {
      lEnergy*=softExp*pow(l[0]*l[1],softExp-1);
    }
    if (b[0]) {
      //atomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      if (b[1]) {
        //atomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
      }
    }
    if (soft) {
      lEnergy/=softExp;
    }

    // Spatial force
    fi=real3_cross(drij,crop);
// NOTE #warning "division on kernel, was using realRecip before."
    real3_scaleself(&fi, fangle/(mcrop*real3_mag2<real>(drij)));
    //at_real3_inc(&force[ii], fi);
    fk=real3_cross(drkj,crop);
    real3_scaleself(&fk,-fangle/(mcrop*real3_mag2<real>(drkj)));
    //at_real3_inc(&force[kk], fk);
    fj=real3_add(fi,fk);
    real3_scaleself(&fj,-1);
    //at_real3_inc(&force[jj], fj);
  }

  // Energy, if requested
  lEnergy*=l[0]*l[1];
}

template <bool flagBox,typename box_type>
void getforce_angleT_oss(System *system,box_type box)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softExp=system->msld->softNotBondExponent;
  int N;
  int shMem=0;

  if (r->calcTermFlag[eeangle]==false) return;

  N=p->angleCount;
  if (N>0) getforce_angle_kernel_oss<flagBox,false><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->angles_d,(real3*)s->position_fd,
    (real3_f*)s->force_d,box,s->lambda_fd,
    s->lambdaForce_d,1,
    system->msld->dGdF_d);
  N=p->softAngleCount;
  if (N>0) getforce_angle_kernel_oss<flagBox,true><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->softAngles_d,(real3*)s->position_fd,
    (real3_f*)s->force_d,box,s->lambda_fd,
    s->lambdaForce_d,softExp,
    system->msld->dGdF_d);
}

void getforce_angle_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_angleT_oss<true>(system,system->state->tricBox_f);
  } else {
    getforce_angleT_oss<false>(system,system->state->orthBox_f);
  }
}



__device__ void function_torsion_oss(DihePotential dp,real phi,real *fphi,real *lE)
{
  real dphi;

  dphi=dp.ndih*phi-dp.dih0;
  fphi[0]=-dp.kdih*dp.ndih*sinf(dphi);
  lE[0]=dp.kdih*(cosf(dphi)+1);
}

__device__ void function_torsion_oss(ImprPotential ip,real phi,real *fphi,real *lE)
{
  real dphi;

  if (ip.nimp>0) {
    dphi=ip.nimp*phi-ip.imp0;
    fphi[0]=-ip.kimp*ip.nimp*sinf(dphi);
    lE[0]=ip.kimp*(cosf(dphi)+1);
  } else {
    dphi=phi-ip.imp0;
    dphi-=(2*((real)M_PI))*floor((dphi+((real)M_PI))/(2*((real)M_PI)));
    fphi[0]=ip.kimp*dphi;
    lE[0]=((real)0.5)*ip.kimp*dphi*dphi;
  }
}

// getforce_dihe_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->dihes_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <bool flagBox,class TorsionPotential,bool soft,typename box_type>
__global__ void getforce_torsion_kernel_oss(int torsionCount,TorsionPotential *torsions,real3 *position,
  real3_f *force,box_type box,real *lambda,
  real_f *lambdaForce,real softExp,
  real *dGdF)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk,ll;
  TorsionPotential tp;
  real rjk;
  real3 drij,drjk,drkl;
  real3 mvec,nvec;
  real phi,sign,ipr;
  real cosp,sinp;
  real3 dsinp;
  real minv2,ninv2,rjkinv2;
  real p,q;
  real3 fi,fj,fk,fl;
  real ftorsion;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj,xk,xl;
  int b[2];
  real l[2]={1,1};

  if (i<torsionCount) {
    // Geometry
    tp=torsions[i];
    ii=tp.idx[0];
    jj=tp.idx[1];
    kk=tp.idx[2];
    ll=tp.idx[3];
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];
    xl=position[ll];

    drij=real3_subpbc<flagBox>(xi,xj,box);
    drjk=real3_subpbc<flagBox>(xj,xk,box);
    drkl=real3_subpbc<flagBox>(xk,xl,box);
    mvec=real3_cross(drij,drjk);
    nvec=real3_cross(drjk,drkl);
    dsinp=real3_cross(mvec,nvec);
    sinp=real3_mag<real>(dsinp);
    cosp=real3_dot<real>(mvec,nvec);
    phi=atan2f(sinp,cosp);
    ipr=real3_dot<real>(drij,nvec);
    sign=(ipr > 0.0) ? -1.0 : 1.0; // Opposite of gromacs because m and n are opposite
    phi=sign*phi;

    // Scaling
    b[0]=0xFFFF & tp.siteBlock[0];
    b[1]=0xFFFF & tp.siteBlock[1];
    if (b[0]) {
      l[0]=lambda[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
      }
    }

    // Interaction
    function_torsion_oss(tp,phi,&ftorsion,&lEnergy);
    if (soft) {
      ftorsion*=pow(l[0]*l[1],softExp);
    } else {
      ftorsion*=l[0]*l[1];
    }

    // Lambda force
    if (soft) {
      lEnergy*=softExp*pow(l[0]*l[1],softExp-1);
    }
    if (b[0]) {
      //atomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      if (b[1]) {
        //atomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
      }
    }
    if (soft) {
      lEnergy/=softExp;
    }

    // Spatial force
// NOTE #warning "Division and sqrt in kernel"
    minv2=1/(real3_mag2<real>(mvec));
    ninv2=1/(real3_mag2<real>(nvec));
    rjk=sqrt(real3_mag2<real>(drjk));
    rjkinv2=1/(rjk*rjk);
    fi=real3_scale<real3>(-ftorsion*rjk*minv2,mvec);
    //at_real3_inc(&force[ii], fi);

    fk=real3_scale<real3>(-ftorsion*rjk*ninv2,nvec);
    p=real3_dot<real>(drij,drjk)*rjkinv2;
    q=real3_dot<real>(drkl,drjk)*rjkinv2;
    fj=real3_scale<real3>(-p,fi);
    real3_scaleinc(&fj,-q,fk);
    fl=real3_scale<real3>(-1,fk);
    //at_real3_inc(&force[ll], fl);

    real3_dec(&fk,fj);
    //at_real3_inc(&force[kk], fk);

    real3_dec(&fj,fi);
    //at_real3_inc(&force[jj], fj);
  }

  // Energy, if requested
  lEnergy*=l[0]*l[1];
}

template <bool flagBox,typename box_type>
void getforce_diheT_oss(System *system,box_type box)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softExp=system->msld->softNotBondExponent;
  int N;
  int shMem=0;

  if (r->calcTermFlag[eedihe]==false) return;

  N=p->diheCount;
  if (N>0) getforce_torsion_kernel_oss<flagBox,DihePotential,false> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->dihes_d,(real3*)s->position_fd,
    (real3_f*) s->force_d, box,
    s->lambda_fd,s->lambdaForce_d, 1,
    system->msld->dGdF_d);

  N=p->softDiheCount;
  if (N>0) getforce_torsion_kernel_oss<flagBox,DihePotential,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->softDihes_d,(real3*)s->position_fd,
    (real3_f*) s->force_d, box,
    s->lambda_fd, s->lambdaForce_d, softExp,
    system->msld->dGdF_d);
}

void getforce_dihe_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_diheT_oss<true>(system,system->state->tricBox_f);
  } else {
    getforce_diheT_oss<false>(system,system->state->orthBox_f);
  }
}

template <bool flagBox,typename box_type>
void getforce_imprT_oss(System *system,box_type box)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softExp=system->msld->softNotBondExponent;
  int N;
  int shMem=0;

  if (r->calcTermFlag[eeimpr]==false) return;

  N=p->imprCount;
  if (N>0) getforce_torsion_kernel_oss <flagBox,ImprPotential,false> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->imprs_d,(real3*)s->position_fd,
    (real3_f*) s->force_d, box,s->lambda_fd,
    s->lambdaForce_d, 1,
    system->msld->dGdF_d);
  N=p->softImprCount;
  if (N>0) getforce_torsion_kernel_oss <flagBox,ImprPotential,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->softImprs_d,(real3*)s->position_fd,
    (real3_f*) s->force_d,
    box,s->lambda_fd,
    s->lambdaForce_d,
    softExp,
    system->msld->dGdF_d);
}

void getforce_impr_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_imprT_oss<true>(system,system->state->tricBox_f);
  } else {
    getforce_imprT_oss<false>(system,system->state->orthBox_f);
  }
}



// getforce_cmap_kernel<<<(2*N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->cmaps_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <bool flagBox,bool soft,typename box_type>
__global__ void getforce_cmap_kernel_oss(
  int cmapCount,struct CmapPotential *cmaps,real3 *position,
  real3_f* force,
  box_type box,
  real *lambda,
  real *lambdaForce,
  real softExp,
  real *dGdF
  )
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk,ll;
  CmapPotential cp;
  real rjk;
  real3 drij,drjk,drkl;
  real3 mvec,nvec;
  real phi,sign,ipr;
  real rPhi[2]; // remainders, one for each angle
  real cosp,sinp;
  real3 dsinp;
  real minv2,ninv2,rjkinv2;
  real p,q;
  real3 fi,fj,fk,fl;
  int lastBit;
  int binPhi[2];
  int cmapBin;
  real invSpace;
  real fcmapPhi[2]; // one for each angle
  real fcmapPhiColumn[2]; // one for each angle
  real fcmap;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj,xk,xl;
  int b[3];
  real l[3]={1,1,1};
  real chain[3]={0,0,0};

  lastBit=threadIdx.x&1;
  if (i<2*cmapCount) { // Threads work in pairs
    // Geometry
    cp=cmaps[i>>1];
    ii=cp.idx[0+4*lastBit];
    jj=cp.idx[1+4*lastBit];
    kk=cp.idx[2+4*lastBit];
    ll=cp.idx[3+4*lastBit];
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];
    xl=position[ll];

    drij=real3_subpbc<flagBox>(xi,xj,box);
    drjk=real3_subpbc<flagBox>(xj,xk,box);
    drkl=real3_subpbc<flagBox>(xk,xl,box);
    mvec=real3_cross(drij,drjk);
    nvec=real3_cross(drjk,drkl);
    dsinp=real3_cross(mvec,nvec);
    sinp=real3_mag<real>(dsinp);
    cosp=real3_dot<real>(mvec,nvec);
    phi=atan2f(sinp,cosp);
    ipr=real3_dot<real>(drij,nvec);
    sign=(ipr > 0.0) ? -1.0 : 1.0; // Opposite of gromacs because m and n are opposite
    phi=sign*phi;

    // Scaling
    b[0]=0xFFFF & cp.siteBlock[0]; // b array is sorted so environment will be first always
    b[1]=0xFFFF & cp.siteBlock[1];
    b[2]=0xFFFF & cp.siteBlock[2];
    if (b[0]) {
      l[0]=lambda[b[0]];
      chain[0] = dGdF[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
        chain[1] = dGdF[b[1]];
        if (b[2]) {
          l[2]=lambda[b[2]];
          chain[2] = dGdF[b[2]];
        }
      }
    }

    // Interaction
      // get phi and psi (both called phi), phi[0] is on even threads, phi[1] on odd
    rPhi[lastBit]=phi;
  }
    rPhi[1-lastBit]=__shfl_xor_sync(0xFFFFFFFF,phi,1);
  if (i<2*cmapCount) { // Avoid hang
      // Get the remainders within each box
    invSpace=cp.ngrid*(1/(2*((real) M_PI)));
    rPhi[0]*=invSpace;
    binPhi[0]=((int) floor(rPhi[0]));
    rPhi[0]-=binPhi[0];
    binPhi[0]+=cp.ngrid/2;
    binPhi[0]+=(binPhi[0]>=cp.ngrid?-cp.ngrid:0);
    binPhi[0]+=(binPhi[0]<0?cp.ngrid:0);
    rPhi[1]*=invSpace;
    binPhi[1]=((int) floor(rPhi[1]));
    rPhi[1]-=binPhi[1];
    binPhi[1]+=cp.ngrid/2;
    binPhi[1]+=(binPhi[1]>=cp.ngrid?-cp.ngrid:0);
    binPhi[1]+=(binPhi[1]<0?cp.ngrid:0);
    cmapBin=cp.ngrid*binPhi[0]+binPhi[1];
      // compute forces (and energy)
    fcmapPhiColumn[0]=3*cp.kcmapPtr[cmapBin][3][2+lastBit];
    fcmapPhiColumn[1]=  cp.kcmapPtr[cmapBin][3][2+lastBit];

    fcmapPhiColumn[0]*=rPhi[0];
    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[0]+=2*cp.kcmapPtr[cmapBin][2][2+lastBit];
    fcmapPhiColumn[1]+=  cp.kcmapPtr[cmapBin][2][2+lastBit];

    fcmapPhiColumn[0]*=rPhi[0];
    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[0]+=cp.kcmapPtr[cmapBin][1][2+lastBit];
    fcmapPhiColumn[1]+=cp.kcmapPtr[cmapBin][1][2+lastBit];

    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[1]+=cp.kcmapPtr[cmapBin][0][2+lastBit];

    if (b[0]) {
      lEnergy=rPhi[1]*rPhi[1]*fcmapPhiColumn[1];
    }
    fcmapPhi[0]=rPhi[1]*rPhi[1]*fcmapPhiColumn[0];
    fcmapPhi[1]=(2+lastBit)*rPhi[1]*fcmapPhiColumn[1];
    fcmapPhi[1]*=(lastBit?rPhi[1]:1);

    fcmapPhiColumn[0]=3*cp.kcmapPtr[cmapBin][3][lastBit];
    fcmapPhiColumn[1]=  cp.kcmapPtr[cmapBin][3][lastBit];

    fcmapPhiColumn[0]*=rPhi[0];
    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[0]+=2*cp.kcmapPtr[cmapBin][2][lastBit];
    fcmapPhiColumn[1]+=  cp.kcmapPtr[cmapBin][2][lastBit];

    fcmapPhiColumn[0]*=rPhi[0];
    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[0]+=cp.kcmapPtr[cmapBin][1][lastBit];
    fcmapPhiColumn[1]+=cp.kcmapPtr[cmapBin][1][lastBit];

    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[1]+=cp.kcmapPtr[cmapBin][0][lastBit];

    if (b[0]) {
      lEnergy+=fcmapPhiColumn[1];
      lEnergy*=(lastBit?rPhi[1]:1);
      // Put all energy on first thread of pair
      // NOHANG lEnergy+=__shfl_xor_sync(0xFFFFFFFF,lEnergy,1);
      // NOHANG lEnergy=(lastBit?0:lEnergy);
    }
    fcmapPhi[0]+=fcmapPhiColumn[0];
    fcmapPhi[0]*=(lastBit?rPhi[1]:1);
    fcmapPhi[1]+=lastBit*fcmapPhiColumn[1];

      // Put partner's force in fcmap for exchange
    fcmap=fcmapPhi[1-lastBit];
  }
    fcmap=__shfl_xor_sync(0xFFFFFFFF,fcmap,1);
  if (i<2*cmapCount) { // Avoid hang
      // Add own force
    fcmap+=fcmapPhi[lastBit];
    fcmap*=invSpace;
    if (soft) {
      fcmap*=pow(l[0]*l[1]*l[2],softExp);
    } else {
      fcmap*=l[0]*l[1]*l[2];
    }

    // Lambda force
    if (soft) {
      lEnergy*=softExp*pow(l[0]*l[1]*l[2],softExp-1);
    }
    if (b[0]) {
      //atomicAdd(&lambdaForce[b[0]],l[1]*l[2]*lEnergy);
      if (b[1]) {
        //atomicAdd(&lambdaForce[b[1]],l[0]*l[2]*lEnergy);
        if (b[2]) {
          //atomicAdd(&lambdaForce[b[2]],l[0]*l[1]*lEnergy);
        }
      }
    }
    if (soft) {
      lEnergy/=softExp;
    }

    // Spatial force
    minv2=1/(real3_mag2<real>(mvec));
    ninv2=1/(real3_mag2<real>(nvec));
    rjk=sqrt(real3_mag2<real>(drjk));
    rjkinv2=1/(rjk*rjk);
    fi=real3_scale<real3>(-fcmap*rjk*minv2,mvec);
    //at_real3_inc(&force[ii], fi);

    fk=real3_scale<real3>(-fcmap*rjk*ninv2,nvec);
    p=real3_dot<real>(drij,drjk)*rjkinv2;
    q=real3_dot<real>(drkl,drjk)*rjkinv2;
    fj=real3_scale<real3>(-p,fi);
    real3_scaleinc(&fj,-q,fk);
    fl=real3_scale<real3>(-1,fk);
    //at_real3_inc(&force[ll], fl);

    real3_dec(&fk,fj);
    //at_real3_inc(&force[kk], fk);

    real3_dec(&fj,fi);
    //at_real3_inc(&force[jj], fj);

    // TODO: Handle soft exp case
    // OST derivatives
  }

  // Energy, if requested
  lEnergy*=l[0]*l[1]*l[2];
}

template <bool flagBox,typename box_type>
void getforce_cmapT_oss(System *system,box_type box)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softExp=system->msld->softNotBondExponent;
  int N;
  int shMem=0;

  if (r->calcTermFlag[eecmap]==false) return;

  N=p->cmapCount;
  if (N>0) getforce_cmap_kernel_oss<flagBox,false><<<(2*N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->cmaps_d,
    (real3*)s->position_fd, (real3_f*) s->force_d, box,
    s->lambda_fd,s->lambdaForce_d, 1,
    system->msld->dGdF_d);

  N=p->softCmapCount;
  if (N>0) getforce_cmap_kernel_oss<flagBox,true><<<(2*N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    N,p->softCmaps_d,
    (real3*)s->position_fd, (real3_f*) s->force_d, box,
    s->lambda_fd,s->lambdaForce_d, softExp,
    system->msld->dGdF_d);
}

void getforce_cmap_oss(System *system)
{
  if (system->state->typeBox) {
    getforce_cmapT_oss<true>(system,system->state->tricBox_f);
  } else {
    getforce_cmapT_oss<false>(system,system->state->orthBox_f);
  }
}
