#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>

#include "msld/msld.h"

#include <cfloat>
#include <math.h>
#include <math.h>
#include <random>

#include "system/system.h"
#include "io/io.h"
#include "main/gpu_check.h"
#include "system/selections.h"
#include "system/structure.h"
#include "system/state.h"
#include "system/potential.h"
#include "run/run.h"

#include "main/real3.h"



// Class constructors
Msld::Msld() {
  blockCount=1;
  atomBlock=NULL;
  lambdaSite=NULL;
  lambdaBias=NULL;
  theta=NULL;
  thetaVelocity=NULL;
  thetaMass=NULL;
  lambdaCharge=NULL;

  atomBlock_d=NULL;
  lambdaSite_d=NULL;
  lambdaBias_d=NULL;
  lambdaCharge_d=NULL;

  blocksPerSite=NULL;
  blocksPerSite_d=NULL;
  siteBound=NULL;
  siteBound_d=NULL;

  atomsByBlock=NULL;

  rest=NULL;
  restScaling=1.0;

  gamma=1.0/PICOSECOND; // ps^-1
  fnex=5.5;

  for (int i=0; i<6; i++) {
    scaleTerms[i]=true;
  }

  variableBias_tmp.clear();
  variableBias=NULL;
  variableBias_d=NULL;

  thetaCollBiasCount=0;
  kThetaCollBias=NULL;
  kThetaCollBias_d=NULL;
  nThetaCollBias=NULL;
  nThetaCollBias_d=NULL;
  thetaIndeBiasCount=0;
  kThetaIndeBias=NULL;
  kThetaIndeBias_d=NULL;

  histogram_1D_d=NULL;

  softBonds.clear();
  atomRestraints.clear();

  atomRestraintCount=0;
  atomRestraintBounds=NULL;
  atomRestraintBounds_d=NULL;
  atomRestraintIdx=NULL;
  atomRestraintIdx_d=NULL;

  useSoftCore=false;
  useSoftCore14=false;
  msldEwaldType=2; // 1-3, not set up to read arguments currently (1=on, 2=ex, 3=nn)

  kRestraint=59.2*KCAL_MOL/(ANGSTROM*ANGSTROM);
  kChargeRestraint=0;
  softBondRadius=1.0*ANGSTROM;
  softBondExponent=2.0;
  softNotBondExponent=1.0;

  fix=false; // ffix
}

Msld::~Msld() {
  if (atomBlock) free(atomBlock);
  if (lambdaSite) free(lambdaSite);
  if (lambdaBias) free(lambdaBias);
  if (theta) free(theta);
  if (thetaVelocity) free(thetaVelocity);
  if (thetaMass) free(thetaMass);
  if (lambdaCharge) free(lambdaCharge);
  if (variableBias) free(variableBias);
  if (kThetaCollBias) free(kThetaCollBias);
  if (nThetaCollBias) free(nThetaCollBias);
  if (kThetaIndeBias) free(kThetaIndeBias);

  if (atomBlock_d) cudaFree(atomBlock_d);
  if (lambdaSite_d) cudaFree(lambdaSite_d);
  if (lambdaBias_d) cudaFree(lambdaBias_d);
  if (lambdaCharge_d) cudaFree(lambdaCharge_d);
  if (variableBias_d) cudaFree(variableBias_d);
  if (kThetaCollBias_d) cudaFree(kThetaCollBias_d);
  if (nThetaCollBias_d) cudaFree(nThetaCollBias_d);
  if (kThetaIndeBias_d) cudaFree(kThetaIndeBias_d);

  if (blocksPerSite) free(blocksPerSite);
  if (blocksPerSite_d) cudaFree(blocksPerSite_d);
  if (siteBound) free(siteBound);
  if (siteBound_d) cudaFree(siteBound_d);

  if (atomsByBlock) delete [] atomsByBlock;

  if (rest) free(rest);

  if (atomRestraintBounds) free(atomRestraintBounds);
  if (atomRestraintBounds_d) cudaFree(atomRestraintBounds_d);
  if (atomRestraintIdx) free(atomRestraintIdx);
  if (atomRestraintIdx_d) cudaFree(atomRestraintIdx_d);
}



// Class parsing
void parse_msld(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  int i,j;

  if (system->structure==NULL) {
    fatal(__FILE__,__LINE__,"selections cannot be defined until structure has been defined\n");
  }

  if (system->msld==NULL) {
    system->msld=new Msld;
  }

  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    if (system->msld) {
      delete(system->msld);
      system->msld=NULL;
    }
  } else if (strcmp(token,"nblocks")==0) {
    if (system->msld) {
      delete(system->msld);
    }
    system->msld=new(Msld);
    system->msld->blockCount=io_nexti(line)+1;
    system->msld->atomBlock=(int*)calloc(system->structure->atomList.size(),sizeof(int));
    system->msld->lambdaSite=(int*)calloc(system->msld->blockCount,sizeof(int));
    system->msld->lambdaBias=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->theta=(real_x*)calloc(system->msld->blockCount,sizeof(real_x));
    system->msld->thetaVelocity=(real_v*)calloc(system->msld->blockCount,sizeof(real_v));
    system->msld->thetaMass=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->thetaMass[0]=1;
    system->msld->lambdaCharge=(real*)calloc(system->msld->blockCount,sizeof(real));

    cudaMalloc(&(system->msld->atomBlock_d),system->structure->atomCount*sizeof(int));
    cudaMalloc(&(system->msld->lambdaSite_d),system->msld->blockCount*sizeof(int));
    cudaMalloc(&(system->msld->lambdaBias_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->lambdaCharge_d),system->msld->blockCount*sizeof(real));

    // NYI - this would be a lot easier to read if these were split in to parsing functions.
    fprintf(stdout,"NYI - Initialize all blocks in first site %s:%d\n",__FILE__,__LINE__);
  } else if (strcmp(token,"call")==0) {
    i=io_nexti(line);
    if (i<0 || i>=system->msld->blockCount) {
      fatal(__FILE__,__LINE__,"Error, tried to edit block %d of %d which does not exist.\n",i,system->msld->blockCount-1);
    }
    std::string name=io_nexts(line);
    if (system->selections->selectionMap.count(name)==0) {
      fatal(__FILE__,__LINE__,"Selection %s not found\n",name.c_str());
    }
    for (j=0; j<system->structure->atomList.size(); j++) {
      if (system->selections->selectionMap[name].boolSelection[j]==1) {
        system->msld->atomBlock[j]=i;
      }
    }
  } else if (strcmp(token,"rest")==0) {
    std::string name=io_nexts(line);
    if (system->selections->selectionMap.count(name)==0) {
      fatal(__FILE__,__LINE__,"Selection %s not found\n",name.c_str());
    }
    if (system->msld->rest) free(system->msld->rest);
    system->msld->rest=(int*)calloc(system->structure->atomList.size(),sizeof(int));
    for (i=0; i<system->structure->atomList.size(); i++) {
      system->msld->rest[i]=system->selections->selectionMap[name].boolSelection[i];
    }
// LDIN 3   0.4   0.0   20.0   5.0
// CHARMM: LDIN BLOCK L0 LVEL LMASS LBIAS
// BLOCK SITE THETA THETAV THETAM LBIAS
  } else if (strcmp(token,"initialize")==0) {
    i=io_nexti(line);
    if (i<0 || i>=system->msld->blockCount) {
      fatal(__FILE__,__LINE__,"Error, tried to edit block %d of %d which does not exist.\n",i,system->msld->blockCount-1);
    }
    system->msld->lambdaSite[i]=io_nexti(line);
    system->msld->theta[i]=io_nextf(line);
    system->msld->thetaVelocity[i]=io_nextf(line);
    system->msld->thetaMass[i]=io_nextf(line);
    system->msld->lambdaBias[i]=io_nextf(line);
    system->msld->lambdaCharge[i]=io_nextf(line);
  } else if (strcmp(token,"gamma")==0) {
    system->msld->gamma=io_nextf(line)/PICOSECOND; // units: ps^-1
  } else if (strcmp(token,"fnex")==0) {
    system->msld->fnex=io_nextf(line);
  } else if (strcmp(token,"bias")==0) {
    // NYI - add option to reset variable biases
    struct VariableBias vb;
    vb.i=io_nexti(line);
    if (vb.i<0 || vb.i>=system->msld->blockCount) {
      fatal(__FILE__,__LINE__,"Error, tried to edit block %d of %d which does not exist.\n",i,system->msld->blockCount-1);
    }
    vb.j=io_nexti(line);
    if (vb.j<0 || vb.j>=system->msld->blockCount) {
      fatal(__FILE__,__LINE__,"Error, tried to edit block %d of %d which does not exist.\n",i,system->msld->blockCount-1);
    }
    vb.type=io_nexti(line);
    // if (vb.type!=6 && vb.type!=8 && vb.type!=10)
    if (vb.type<=0 || vb.type>12) {
      fatal(__FILE__,__LINE__,"Type of variable bias (%d) is not a recognized type\n",vb.type);
    }
    vb.l0=io_nextf(line);
    vb.k=io_nextf(line);
    vb.n=io_nexti(line);
    system->msld->variableBias_tmp.push_back(vb);
  } else if (strcmp(token,"thetaebias")==0) {
    std::string name;
    name=io_nexts(line);
    if (name=="collective") {
      name=io_nexts(line);
      if (name=="nsites") {
        system->msld->thetaCollBiasCount=io_nexti(line)+1;
        system->msld->kThetaCollBias=(real*)calloc(system->msld->thetaCollBiasCount,sizeof(real));
        system->msld->nThetaCollBias=(real*)calloc(system->msld->thetaCollBiasCount,sizeof(real));
      } else if (name=="set") {
        if (system->msld->kThetaCollBias==NULL) {
          fatal(__FILE__,__LINE__,"Call thetabias collective nsites [int] before thetabias set or thetabias auto\n");
        }
        int i0=io_nexti(line);
        int i1=i0+1;
        if (i0==0) {
          i0=1;
          i1=system->msld->thetaCollBiasCount;
        }
        real k,N;
        k=io_nextf(line);
        N=io_nextf(line);
        for (i=i0; i<i1; i++) {
          system->msld->kThetaCollBias[i]=k;
          system->msld->nThetaCollBias[i]=N;
        }
      } else {
        fatal(__FILE__,__LINE__,"Unrecognized token %s in thetabias\n",name.c_str());
      }
    } else if (name=="independent") {
      name=io_nexts(line);
      if (name=="nsites") {
        system->msld->thetaIndeBiasCount=io_nexti(line)+1;
        system->msld->kThetaIndeBias=(real*)calloc(system->msld->thetaIndeBiasCount,sizeof(real));
      } else if (name=="set" || name=="auto") {
        if (system->msld->kThetaIndeBias==NULL) {
          fatal(__FILE__,__LINE__,"Call thetabias indepdendent nsites [int] before thetabias set or thetabias auto\n");
        }
        int i0=io_nexti(line);
        int i1=i0+1;
        if (i0==0) {
          i0=1;
          i1=system->msld->thetaIndeBiasCount;
        }
        real k,Ns,T;
        if (name=="auto") {
          Ns=io_nextf(line);
          T=io_nextf(line);
        }
        if (name=="set") {
          k=io_nextf(line);
        }
        for (i=i0; i<i1; i++) {
          if (name=="auto") {
            k=1;
            for (j=0; j<50; j++) {
              // k=0.5*log(k*Ns*Ns*M_PI/2);
              k=0.5*log(0.25*k*Ns*Ns*M_PI/2);
              if (!(k>0)) k=0;
            }
            k=kB*T;
          }
          system->msld->kThetaIndeBias[i]=k;
        }
      } else {
        fatal(__FILE__,__LINE__,"Unrecognized token %s in thetabias\n",name.c_str());
      }
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized token %s for thetabias type. Use collective or independent\n",name.c_str());
    }
  } else if (strcmp(token,"removescaling")==0) {
    std::string name;
    while ((name=io_nexts(line))!="") {
      if (name=="bond") {
        system->msld->scaleTerms[0]=false;
      } else if (name=="urey") {
        system->msld->scaleTerms[1]=false;
      } else if (name=="angle") {
        system->msld->scaleTerms[2]=false;
      } else if (name=="dihe") {
        system->msld->scaleTerms[3]=false;
      } else if (name=="impr") {
        system->msld->scaleTerms[4]=false;
      } else if (name=="cmap") {
        system->msld->scaleTerms[5]=false;
      } else {
        fatal(__FILE__,__LINE__,"Unrecognized token %s. Valid options are bond, urey, angle, dihe, impr, or cmap.\n",name.c_str());
      }
    }
  } else if (strcmp(token,"softcore")==0) {
    system->msld->useSoftCore=io_nextb(line);
  } else if (strcmp(token,"softcore14")==0) {
    system->msld->useSoftCore14=io_nextb(line);
  } else if (strcmp(token,"ewaldtype")==0) {
    system->msld->msldEwaldType=io_nexti(line);
    if (system->msld->msldEwaldType<=0 || system->msld->msldEwaldType>3) {
      fatal(__FILE__,__LINE__,"Invalid choice of %d for msld ewaldtype. Must choose 1, 2, or 3 (default 2).\n",system->msld->msldEwaldType);
    }
  } else if (strcmp(token,"parameter")==0) {
    std::string parameterToken=io_nexts(line);
    if (parameterToken=="krestraint") {
      system->msld->kRestraint=io_nextf(line)*KCAL_MOL/(ANGSTROM*ANGSTROM);
    } else if (parameterToken=="kchargerestraint") {
      system->msld->kChargeRestraint=io_nextf(line)*KCAL_MOL;
    } else if (parameterToken=="softbondradius") {
      system->msld->softBondRadius=io_nextf(line)*ANGSTROM;
    } else if (parameterToken=="softbondexponent") {
      system->msld->softBondExponent=io_nextf(line);
    } else if (parameterToken=="softnotbondexponent") {
      system->msld->softNotBondExponent=io_nextf(line);
    } else if (parameterToken=="restscaling") {
      system->msld->restScaling=io_nextf(line);
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized parameter name %s for msld parameter\n",parameterToken.c_str());
    }
  } else if (strcmp(token,"restrain")==0) {
    std::string name=io_nexts(line);
    int i;
    if (name=="reset") {
      system->msld->atomRestraints.clear();
    } else if (system->selections->selectionMap.count(name)==1) {
      std::vector<int> atoms;
      atoms.clear();
      for (i=0; i<system->selections->selectionMap[name].boolCount; i++) {
        if (system->selections->selectionMap[name].boolSelection[i]) {
          atoms.push_back(i);
        }
      }
      if (atoms.size()>0) {
        // NYI - checking for one atom per substituent
        system->msld->atomRestraints.push_back(atoms);
      }
    } else {
      fatal(__FILE__,__LINE__,"Selection %s not found for msld atom restraints\n");
    }
// NYI - restorescaling option to complement remove scaling
  } else if (strcmp(token,"softbond")==0) {
// NYI - check selection length
    std::string name=io_nexts(line);
    int i;
    int j;
    Int2 i2;
    if (system->selections->selectionMap.count(name)==1) {
      j=0;
      for (i=0; i<system->selections->selectionMap[name].boolCount; i++) {
        if (system->selections->selectionMap[name].boolSelection[i]) {
          if (j<2) {
            i2.i[j]=i;
          }
          j++;
        }
      }
      if (j==2) {
        system->msld->softBonds.push_back(i2);
      } else {
        fatal(__FILE__,__LINE__,"Found soft bond selection with %d atoms when expected 2 atoms\n",j);
      }
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized token %s used for softbond selection name. Use selection print to see available tokens.\n",name.c_str());
    }
    // NYI - load in soft bond parameters later
  } else if (strcmp(token,"atomrestraint")==0) {
    std::string name=io_nexts(line);
    int i;
    std::vector<int> ar; // NYI - error checking, requires site assignment - ,br,sr; // atom, block, and site
    ar.clear();
    if (system->selections->selectionMap.count(name)==1) {
      for (i=0; i<system->selections->selectionMap[name].boolCount; i++) {
        if (system->selections->selectionMap[name].boolSelection[i]) {
          ar.push_back(i);
        }
      }
      system->msld->atomRestraints.push_back(ar);
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized token %s used for atomrestraint selection name. Use selection print to see available tokens.\n",name.c_str());
    }
  } else if (strcmp(token,"fix")==0) { // ffix
    system->msld->fix=io_nextb(line); // ffix
  } else if (strcmp(token, "slow_fix") == 0){
    system->msld->slow_fix=io_nextb(line);
// NYI - charge restraints, put Q in initialize
  } else if (strcmp(token,"print")==0) {
    system->selections->dump();
  } else if (strcmp(token, "GaMD_total") == 0){
    system->msld->GaMD_total=io_nextb(line);
  } else if (strcmp(token, "GaMD_torsion") == 0){
    system->msld->GaMD_torsion=io_nextb(line);
  } else if (strcmp(token, "GaMD_alchem") == 0){
    system->msld->GaMD_alchem=io_nextb(line);
  } else if (strcmp(token, "GaMD_orth") == 0){
    system->msld->GaMD_orth=io_nextb(line);
  } else if (strcmp(token, "GaMD_init_steps") == 0){ // These next two need to be set in order
    system->msld->init_steps=io_nexti(line);
  } else if (strcmp(token, "GaMD_equil_steps") == 0){
    system->msld->equil_steps=system->msld->init_steps+io_nexti(line);
  } else if (strcmp(token, "GaMD_low_threshold") == 0){
    system->msld->GaMD_low_threshold=io_nextb(line);
  } else if (strcmp(token, "alpha") == 0){
    system->msld->alpha=io_nextf(line);
  } else if (strcmp(token, "oss") == 0) {
    system->msld->oss=io_nextb(line);
  } else if (strcmp(token, "abf") == 0){
    system->msld->abf=io_nextb(line);
  } else if (strcmp(token, "L_1D_bins") == 0){
    system->msld->L_1D_bins=io_nexti(line);
  } else if (strcmp(token, "L_std") == 0){
    system->msld->L_std=io_nextf(line);
  } else if (strcmp(token, "edge_KDE_std") == 0){
    system->msld->edge_KDE_std=io_nextf(line);
  } else if (strcmp(token, "warmup_samples") == 0){
    system->msld->warmup_samples=io_nexti(line);
  } else if (strcmp(token, "update_steps") == 0){
    system->msld->update_steps=io_nexti(line);
  } else if (strcmp(token, "update_fe") == 0) {
    system->msld->update_fe_surface=io_nextb(line);
  } else if (strcmp(token, "tracking_only") == 0){
    system->msld->tracking_only=io_nextb(line);
    if(system->msld->tracking_only){
      system->msld->abf = true;
      system->msld->update_fe_surface = true;
    }
  } else if (strcmp(token, "sample_freq") == 0){
    system->msld->sample_freq=io_nexti(line);
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized selection token: %s\n",token);
  }
}

int merge_site_block(int site,int block)
{
  if (site>=(1<<16) || block>=(1<<16)) {
    fatal(__FILE__,__LINE__,"Site or block cap of 2^16 exceeded. Site=%d,Block=%d\n",site,block);
  }
  return ((site<<16)|block);
}

bool Msld::check_soft(int *idx,int Nat)
{
  int i,j,k;
  bool found[2];

  for (i=0; i<softBonds.size(); i++) {
    for (j=0; j<2; j++) {
      found[j]=false;
      for (k=0; k<Nat; k++) {
        if (softBonds[i].i[j]==idx[k]) {
          found[j]=true;
        }
      }
    }
    if (found[0] && found[1]) return true;
  }
  return false;
}

bool Msld::check_restrained(int atom)
{
  int i,j;
  for (i=0; i<atomRestraints.size(); i++) {
    for (j=0; j<atomRestraints[i].size(); j++) {
      if (atomRestraints[i][j]==atom) return true;
    }
  }
  return false;
}

#define NscMAX 3
bool Msld::bonded_scaling(int *idx,int *siteBlock,int type,int Nat,int Nsc)
{
  bool scale,soft;
  int i,j,k;
  int ab;
  int block[NscMAX+2]={0}; // First term is blockCount, last term is for error checking
  block[0]=blockCount;

  if (Nsc>NscMAX) {
    fatal(__FILE__,__LINE__,"Nsc=%d greater than NscMAX=%d\n",Nsc,NscMAX);
  }

  // Sort into a descending list with no duplicates.
  scale=scaleTerms[type];
  soft=check_soft(idx,Nat);
  for (i=1; i<Nsc+2; i++) {
    for (j=0; j<Nat; j++) {
      ab=atomBlock[idx[j]];
      if ((!soft) && (!scale)) {
        for (k=0; k<Nat; k++) {
          if (atomBlock[idx[j]]==atomBlock[idx[k]] && !check_restrained(idx[k])) {
            ab=0;
          }
        }
      }
      if (ab>block[i] && ab<block[i-1]) {
        block[i]=ab;
      }
    }
  }
  // Check for errors
  for (i=1; i<Nsc+1; i++) {
    for (j=i+1; j<Nsc+1; j++) {
      if (block[i]>0 && block[j]>0 && block[i]!=block[j] && lambdaSite[block[i]]==lambdaSite[block[j]]) {
        fatal(__FILE__,__LINE__,"Illegal MSLD scaling between two atoms in the same site (%d) but different blocks (%d and %d)\n",lambdaSite[block[i]],block[i],block[j]);
      }
    }
  }
  if (block[Nsc+1] != 0) {
    fatal(__FILE__,__LINE__,"Only %d lambda scalings allowed in a group of %d bonded atoms\n",Nsc,Nat);
  }

  for (i=0; i<Nsc; i++) {
    siteBlock[i]=merge_site_block(lambdaSite[block[i+1]],block[i+1]);
  }
  return soft;
}

void Msld::nonbonded_scaling(int *idx,int *siteBlock,int Nat)
{
  int i,j;
  int ab;
  int Nsc=Nat;
  int block[NscMAX+2]={0}; // First term is blockCount, last term is for error checking
  block[0]=blockCount;

  if (Nsc>NscMAX) {
    fatal(__FILE__,__LINE__,"Nsc=%d greater than NscMAX=%d\n",Nsc,NscMAX);
  }

  // Sort into a descending list with no duplicates.
  for (i=1; i<Nsc+2; i++) {
    for (j=0; j<Nat; j++) {
      ab=atomBlock[idx[j]];
      if (ab>block[i] && ab<block[i-1]) {
        block[i]=ab;
      }
    }
  }

  for (i=0; i<Nsc; i++) {
    siteBlock[i]=merge_site_block(lambdaSite[block[i+1]],block[i+1]);
  }
}


bool Msld::bond_scaling(int idx[2],int siteBlock[2])
{
  return bonded_scaling(idx,siteBlock,0,2,2);
}

bool Msld::ureyb_scaling(int idx[3],int siteBlock[2])
{
  return bonded_scaling(idx,siteBlock,1,3,2);
}

bool Msld::angle_scaling(int idx[3],int siteBlock[2])
{
  return bonded_scaling(idx,siteBlock,2,3,2);
}

bool Msld::dihe_scaling(int idx[4],int siteBlock[2])
{
  return bonded_scaling(idx,siteBlock,3,4,2);
}

bool Msld::impr_scaling(int idx[4],int siteBlock[2])
{
  return bonded_scaling(idx,siteBlock,4,4,2);
}

bool Msld::cmap_scaling(int idx[8],int siteBlock[3])
{
  return bonded_scaling(idx,siteBlock,5,8,3);
}

void Msld::nb14_scaling(int idx[2],int siteBlock[2])
{
  nonbonded_scaling(idx,siteBlock,2);
  if ((siteBlock[0]!=siteBlock[1]) && ((siteBlock[0]&0xFFFF0000)==(siteBlock[1]&0xFFFF0000))) {
    fatal(__FILE__,__LINE__,"Illegal 14 interaction between atom %d and %d\n",idx[0],idx[1]);
  }
}

bool Msld::nbex_scaling(int idx[2],int siteBlock[2])
{
  // nonbonded_scaling(idx,siteBlock,2);
  bool include=true;
  int i;
  int ab;
  int block[2];

  for (i=0; i<2; i++) {
    block[i]=atomBlock[idx[i]];
  }

  // Sort into a descending list (allow duplicates).
  if (block[0]<block[1]) {
    ab=block[0];
    block[0]=block[1];
    block[1]=ab;
  }

  if (msldEwaldType==1) {
    if (block[0]==block[1]) {
      block[1]=0;
    }
    if (block[0]!=block[1] && lambdaSite[block[0]]==lambdaSite[block[1]]) {
      include=false;
    }
  } else if (msldEwaldType==2) {
    if (block[0]!=block[1] && lambdaSite[block[0]]==lambdaSite[block[1]]) {
      include=false;
    }
  } else if (msldEwaldType==3) {
    // Do nothing, scale by both atom's lambdas regardless of site
  } else {
    fatal(__FILE__,__LINE__,"Illegal msldEwaldType parameter of %d, only 1, 2, or 3 is allowed\n",msldEwaldType);
  }

  for (i=0; i<2; i++) {
    siteBlock[i]=merge_site_block(lambdaSite[block[i]],block[i]);
  }

  return include;
}

void Msld::nbond_scaling(int idx[1],int siteBlock[1])
{
  nonbonded_scaling(idx,siteBlock,1);
}

bool Msld::interacting(int i,int j)
{
  return atomBlock[i]==atomBlock[j] || lambdaSite[atomBlock[i]]!=lambdaSite[atomBlock[j]];
}

// Initialize MSLD for a simulation
void Msld::initialize(System *system)
{
  int i,j;

  // Send the biases over
  cudaMemcpy(atomBlock_d,atomBlock,system->structure->atomCount*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(lambdaBias_d,lambdaBias,blockCount*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(lambdaCharge_d,lambdaCharge,blockCount*sizeof(real),cudaMemcpyHostToDevice);
  variableBiasCount=variableBias_tmp.size();
  variableBias=(struct VariableBias*)calloc(variableBiasCount,sizeof(struct VariableBias));
  cudaMalloc(&variableBias_d,variableBiasCount*sizeof(struct VariableBias));
  for (i=0; i<variableBiasCount; i++) {
    variableBias[i]=variableBias_tmp[i];
  }
  cudaMemcpy(variableBias_d,variableBias,variableBiasCount*sizeof(struct VariableBias),cudaMemcpyHostToDevice);

  if (thetaCollBiasCount>0) {
    cudaMalloc(&kThetaCollBias_d,thetaCollBiasCount*sizeof(real));
    cudaMemcpy(kThetaCollBias_d,kThetaCollBias,thetaCollBiasCount*sizeof(real),cudaMemcpyHostToDevice);
    cudaMalloc(&nThetaCollBias_d,thetaCollBiasCount*sizeof(real));
    cudaMemcpy(nThetaCollBias_d,nThetaCollBias,thetaCollBiasCount*sizeof(real),cudaMemcpyHostToDevice);
  }
  if (thetaIndeBiasCount>0) {
    cudaMalloc(&kThetaIndeBias_d,thetaIndeBiasCount*sizeof(real));
    cudaMemcpy(kThetaIndeBias_d,kThetaIndeBias,thetaIndeBiasCount*sizeof(real),cudaMemcpyHostToDevice);
  }

  // Get blocksPerSite
  siteCount=1;
  for (i=0; i<blockCount; i++) {
    siteCount=((siteCount>lambdaSite[i])?siteCount:(lambdaSite[i]+1));
    if (i!=0 && lambdaSite[i]!=lambdaSite[i-1] && lambdaSite[i]!=lambdaSite[i-1]+1) {
      fatal(__FILE__,__LINE__,"Blocks must be ordered by consecutive sites. Block %d (site %d) is out of order with block %d (site %d)\n",i,lambdaSite[i],i-1,lambdaSite[i-1]);
    }
  }
  blocksPerSite=(int*)calloc(siteCount,sizeof(int));
  siteBound=(int*)calloc(siteCount+1,sizeof(int));
  cudaMalloc(&blocksPerSite_d,siteCount*sizeof(int));
  cudaMalloc(&siteBound_d,(siteCount+1)*sizeof(int));
  for (i=0; i<blockCount; i++) {
    blocksPerSite[lambdaSite[i]]++;
  }
  if (blocksPerSite[0]!=1) fatal(__FILE__,__LINE__,"Only one block allowed in site 0\n");
  siteBound[0]=0;
  for (i=0; i<siteCount; i++) {
    if (i && blocksPerSite[i]<2) fatal(__FILE__,__LINE__,"At least two blocks are required in each site. %d found at site %d\n",blocksPerSite[i],i);
    siteBound[i+1]=siteBound[i]+blocksPerSite[i];
  }
  cudaMemcpy(blocksPerSite_d,blocksPerSite,siteCount*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(siteBound_d,siteBound,(siteCount+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(lambdaSite_d,lambdaSite,blockCount*sizeof(int),cudaMemcpyHostToDevice);

  // Slow Growth
  if(slow_fix){
    real_x delta[blockCount];
    memcpy(delta, theta, blockCount*sizeof(real_x)); // when fix is set, theta should have the desired lambda states
    memset(theta, 0, blockCount*sizeof(real_x)); // reset lambda to wt 
    real_x rate = 1.0 / (system->run->nsteps);
    int prev = 0;
    for(int i = 0; i < siteCount; i++){ // WT is 1 for the first block in each site
      // delta = (desired state - WT state) * rate
      delta[prev] -= 1;
      theta[prev] = 1;
      prev += blocksPerSite[i];
    }
    for(int i = 0; i < blockCount; i++){
      delta[i] *= rate;
    }
    cudaMalloc(&lambda_delta_d, blockCount*sizeof(real_x));
    cudaMemcpy(lambda_delta_d, delta, blockCount*sizeof(real_x), cudaMemcpyDefault);
  }

  // OSS variables we always want allocated
  dUdL_msld = (real*)malloc(blockCount*sizeof(real));
  cudaMalloc(&dUdL_alf_d, blockCount*sizeof(real));
  cudaMemset(dUdL_alf_d, 0, blockCount*sizeof(real));
  cudaMalloc(&dUdL_bonded_d, blockCount*sizeof(real));
  cudaMemset(dUdL_bonded_d, 0, blockCount*sizeof(real));
  cudaMalloc(&dUdL_msld_d, blockCount*sizeof(real));
  cudaMemset(dUdL_msld_d, 0, blockCount*sizeof(real));
  cudaMalloc(&dUdL_abf_d, blockCount*sizeof(real));
  cudaMemset(dUdL_abf_d, 0, blockCount*sizeof(real));
  cudaMalloc(&hist_potential_d, blockCount*sizeof(real)); 
  cudaMemset(hist_potential_d, 0, blockCount*sizeof(real));
  cudaMalloc(&dGdF_d, blockCount*sizeof(real)); // indexed same as lambda array
  cudaMemset(dGdF_d, 0, blockCount*sizeof(real));
  hist_potential = (real*)malloc(blockCount*sizeof(real));
  // Of total # of steps, use 1/4th to develop bias as default
  if (update_steps == -1){
    update_steps = .25*system->run->nsteps;
  }
  if ((abf || oss) && !histogram_1D_d){ // abf-like tracking features get used when OSS is on 
    init_oss(system);
  }

  // GaMD
  int nAtoms = system->structure->atomCount;
  int nL = system->msld->blockCount;
  int rootFactor=(system->id==0?system->idCount:1);
  int DOF = (nAtoms*3 + nL);
  cudaMalloc(&GaMD_torsion_force_d, DOF*sizeof(real));
  cudaMemset(GaMD_torsion_force_d, 0, DOF*sizeof(real));
  cudaMalloc(&GaMD_alchem_force_d, DOF*sizeof(real));
  cudaMemset(GaMD_alchem_force_d, 0, DOF*sizeof(real));
  cudaMalloc(&alchem_energy_d, sizeof(real));
  alchem_energy = (real*) malloc(sizeof(real));
  *alchem_energy = 0;
  memset(GaMD_bias_added, 0, GaMD_modes*sizeof(real));
  if(GaMD_total || GaMD_torsion || GaMD_alchem || GaMD_orth){
    memset(total_p_stats, 0, num_GaMD_stats*sizeof(double));
    total_p_stats[0] = 1e9; // Min starts at large value
    total_p_stats[1] = -1e9; // Max starts at small value
    total_p_stats[4] = 6; // Max std of boost
    memset(torsion_p_stats, 0, num_GaMD_stats*sizeof(double));
    torsion_p_stats[0] = 1e9; 
    torsion_p_stats[1] = -1e9; 
    torsion_p_stats[4] = 10;
    memset(alchem_p_stats, 0, num_GaMD_stats*sizeof(double));
    alchem_p_stats[0] = 1e9; 
    alchem_p_stats[1] = -1e9; 
    alchem_p_stats[4] = 40;
    for (int i = 0; i < blockCount; i++){
      int id = i*num_GaMD_stats;
      orth_p_stats[id] = 1e9;
      orth_p_stats[id+1] = -1e9;
      orth_p_stats[id+4] = 1;
    }
  }

  // Atom restraints
  atomRestraintCount=atomRestraints.size();
  if (atomRestraintCount>0) {
    atomRestraintBounds=(int*)calloc(atomRestraintCount+1,sizeof(int));
    cudaMalloc(&atomRestraintBounds_d,(atomRestraintCount+1)*sizeof(int));
    atomRestraintBounds[0]=0;
    for (i=0; i<atomRestraintCount; i++) {
      atomRestraintBounds[i+1]=atomRestraintBounds[i]+atomRestraints[i].size();
    }
    atomRestraintIdx=(int*)calloc(atomRestraintBounds[atomRestraintCount],sizeof(int));
    cudaMalloc(&atomRestraintIdx_d,atomRestraintBounds[atomRestraintCount]*sizeof(int));
    for (i=0; i<atomRestraintCount; i++) {
      for (j=0; j<atomRestraints[i].size(); j++) {
        atomRestraintIdx[atomRestraintBounds[i]+j]=atomRestraints[i][j];
      }
    }
    cudaMemcpy(atomRestraintBounds_d,atomRestraintBounds,(atomRestraintCount+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(atomRestraintIdx_d,atomRestraintIdx,atomRestraintBounds[atomRestraintCount]*sizeof(int),cudaMemcpyHostToDevice);
  }

  atomsByBlock=new std::set<int>[blockCount];
  for (i=0; i<system->structure->atomCount; i++) {
    atomsByBlock[atomBlock[i]].insert(i);
  }
}

void Msld::gamd_reset(System* system){
  GaMD_samples = 0;
  for (int i = 0; i < num_GaMD_stats-3; i++){ // Don't reset sigmaV0, E, or k 
    total_p_stats[i] = 0;
    torsion_p_stats[i] = 0;
    alchem_p_stats[i] = 0;
    for (int j = 0; j < blockCount; j++){
      int id = i + j*num_GaMD_stats;
      orth_p_stats[id] = 0;
    }
  }
  total_p_stats[0] = 1e9; 
  total_p_stats[1] = -1e9; 
  torsion_p_stats[0] = 1e9; 
  torsion_p_stats[1] = -1e9; 
  alchem_p_stats[0] = 1e9; 
  alchem_p_stats[1] = -1e9; 
  for (int i = 0; i < blockCount; i++){
      int id = i*num_GaMD_stats;
      orth_p_stats[id] = 1e9;
      orth_p_stats[id+1] = -1e9;
  }
}

void update_E_k(bool conservative, int num_samples, double* stats){
  // [Vmin, Vmax, Vavg, M2, Vstd_max, E, k]
  real Vmin = stats[0];
  real Vmax = stats[1];
  real Vavg = stats[2];
  real Vstd = sqrt(stats[3]/num_samples);
  real Vstd_max = stats[4];
  if (conservative){ // Weaker bias
    stats[5] = Vmax;
    real k0p = (Vstd_max / Vstd) * (Vmax - Vmin) / (Vmax - Vavg);
    stats[6] = min(1.0, k0p); 
  } else { // Stronger bias
    real k0pp = (1.0 - Vstd_max/Vstd) * (Vmax - Vmin) / (Vavg - Vmin);
    if (k0pp > 0 && k0pp <= 1.0){
      stats[5] = Vmin + (Vmax-Vmin)/k0pp;
      stats[6] = k0pp; 
    } else { // Back to conservative
      stats[5] = Vmax;
      real k0p = (Vstd_max / Vstd) * (Vmax - Vmin) / (Vmax - Vavg);
      stats[6] = min(1.0, k0p);
    }
  }
  stats[6] /= Vmax - Vmin;
}

void update_stats(real V_sample, double* V_stats, int n_samples, bool conservative, bool calc_E_k){
  // Max/Min
  if (V_sample > V_stats[1]){
    V_stats[1] = V_sample;
  }
  if (V_sample < V_stats[0]){
    V_stats[0] = V_sample;
  }
  // Avg/Std
  real Vdiff = V_sample - V_stats[2];
  real Vavg = V_stats[2]*n_samples;
  Vavg += V_sample;
  Vavg /= n_samples + 1;
  V_stats[2] = Vavg;
  V_stats[3] += Vdiff*(V_sample-Vavg);
  // Harmonic Params
  if (calc_E_k){
    update_E_k(conservative, n_samples, V_stats);
  }
}

void Msld::gamd_update(System* system, bool calc_E_k) {
  // [Vmin, Vmax, Vavg, M2, Vstd_max, E, k]
  if (GaMD_total){
    real V = system->state->energy[eepotential];
    if (GaMD_torsion){ V -= system->state->energy[eedihe] + system->state->energy[eeimpr]; }
    if (GaMD_alchem){ V -= *system->msld->alchem_energy; }
    update_stats(V, total_p_stats, GaMD_samples, GaMD_low_threshold, calc_E_k);
  }
  if (GaMD_torsion){
    real V = system->state->energy[eedihe] + system->state->energy[eeimpr];
    update_stats(V, torsion_p_stats, GaMD_samples, GaMD_low_threshold, calc_E_k);
  }
  GaMD_samples++;

  // Logging
  if (system->run->step % 1000 == 0){
    printf("Potential Energy: %f\n", system->state->energy[eepotential]);
    if (GaMD_total){
      real V = system->state->energy[eepotential];
      if (GaMD_torsion){ V -= system->state->energy[eedihe] + system->state->energy[eeimpr]; }
      if (GaMD_alchem){ V -= *system->msld->alchem_energy; }
      printf("Potential (w/o extra): %f +/- %f, Boost: %f, total_p_stats: [ ", 
        V, sqrt(total_p_stats[3] / GaMD_samples), GaMD_bias_added[0]);
      for(int i = 0; i < num_GaMD_stats; i++){
        printf("%f, ", total_p_stats[i]);
      }
      printf(" ]\n");
    }
    if (GaMD_torsion){
      real V = system->state->energy[eedihe] + system->state->energy[eeimpr];
      printf("Torsion Potential: %f +/- %f, Boost: %f, torsion_p_stats: [ ", 
        V, sqrt(torsion_p_stats[3] / GaMD_samples), GaMD_bias_added[1]);
      for(int i = 0; i < num_GaMD_stats; i++){
        printf("%f, ", torsion_p_stats[i]);
      }
      printf(" ]\n");
    }
  }
}

void __global__ gamd_kernel(
  int len, int blockCount,
  real dBoost_total, real dBoost_torsion, real dBoost_alchem, 
  real* force, 
  real* torsion_force, bool remove_torsion, 
  real* alchem_force, bool remove_alchem,
  real* alf_force
) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<len) { 
    real remaining_force = force[i];
    // Remove separate biases
    remaining_force -= remove_torsion ? torsion_force[i] : 0;
    remaining_force -= remove_alchem ? alchem_force[i] : 0;
    force[i] += remaining_force*dBoost_total; 
    force[i] += torsion_force[i]*dBoost_torsion;
    force[i] += alchem_force[i]*dBoost_alchem;
  }
}

// Already waited on energy & gradient
void Msld::getforce_gamd(System* system) {
  cudaStream_t stream = system->run->gamdBias;
  memset(GaMD_bias_added, 0, GaMD_modes*sizeof(real));
  real dBoost_total = 0;
  real dBoost_torsion = 0;
  real dBoost_alchem = 0;
  // stats: [Vmin, Vmax, Vavg, M2, Vstd_max, E, k]
  if(GaMD_total){
    // Calculate total potential without other variables we are biasing with 
    real V = system->state->energy[eepotential];
    if (GaMD_torsion){ V -= system->state->energy[eedihe] + system->state->energy[eeimpr]; }
    if (GaMD_alchem){ V -= *system->msld->alchem_energy; }
    real dV = V - total_p_stats[5];
    real boost = 0;
    if (dV < 0){
      boost = .5*total_p_stats[6]*dV*dV;
      GaMD_bias_added[0] = boost;
      dBoost_total = total_p_stats[6]*dV;
    }
  }
  if(GaMD_torsion){
    real V = system->state->energy[eedihe] + system->state->energy[eeimpr];
    real dV = V - torsion_p_stats[5];
    real boost = 0;
    if (dV < 0){
      boost = .5*torsion_p_stats[6]*dV*dV;
      GaMD_bias_added[1] = boost;
      dBoost_torsion = torsion_p_stats[6]*dV;
    }
  }

  int nAtoms = system->state->atomCount;
  int nL = system->msld->blockCount;
  int DOF = nAtoms*3 + nL;
  gamd_kernel<<<(DOF+BLMS-1)/BLMS,BLMS,0,stream>>>(
    DOF, blockCount, dBoost_total, dBoost_torsion, dBoost_alchem,
    (real*) system->state->forceBuffer_d, 
    GaMD_torsion_force_d, GaMD_torsion,
    GaMD_alchem_force_d, GaMD_alchem,
    dUdL_alf_d
  );

  // Reset force arrays with rest of forces
}

/** 
 * Histogram defined to have elements for the first and last elements at max and min respectively. Width of the
 * first and last bins are half length. num_bins-1 whole bins fit in the [min, max) range. Uniform histogram.
 * Element is relative to min.
 * 
 * 
 * Ex. Lambda bins: [0, .005), [.005, .015), ..., [.985, .995), [.995, 1.0) -> range [0,1), 101 bins
 *  Lambda Centers: [   0   ], [   .010   ], ..., [   .990   ], [   1.0   ]
 *    Lambda Index: [   0   ], [     1    ], ..., [    99    ], [   100   ]
*/
static __forceinline__ __device__ int get_histogram_index(real val, int num_bins, real max, real min){
  real tmp = val - min;
  real range = max - min;
  real resolution = range / (num_bins-1);
  return round(tmp/resolution);
}

static int histogram_index(real val, int num_bins, real max, real min) {
  real tmp = val - min;
  real range = max - min;
  real resolution = range / (num_bins-1);
  return round(tmp/resolution);
}

void Msld::init_oss(System* system){
  int nL = blockCount-1; // 0th lambda is environment
  // 1D Storage
  cudaMalloc(&histogram_1D_d, nL*L_1D_bins*sizeof(real));
  cudaMemset(histogram_1D_d, 0, nL*L_1D_bins*sizeof(real));
  cudaMalloc(&average_dUdL_d, nL*L_1D_bins*sizeof(real));
  cudaMemset(average_dUdL_d, 0, nL*L_1D_bins*sizeof(real));
  cudaMalloc(&average_dUdL2_d, nL*L_1D_bins*sizeof(real));
  cudaMemset(average_dUdL2_d, 0, nL*L_1D_bins*sizeof(real));
  cudaMalloc(&variance_dUdL_d, nL*L_1D_bins*sizeof(real));
  cudaMemset(variance_dUdL_d, 0, nL*L_1D_bins*sizeof(real));
  cudaMalloc(&weights_d, nL*L_1D_bins*sizeof(real));
  cudaMemset(weights_d, 0, nL*L_1D_bins*sizeof(real));
  cudaMalloc(&offsets_d, nL*L_1D_bins*sizeof(real));
  cudaMemset(offsets_d, 0, nL*L_1D_bins*sizeof(real));
  cudaMalloc(&weighted_dUdL_d, nL*L_1D_bins*sizeof(real));
  cudaMemset(weighted_dUdL_d, 0, nL*L_1D_bins*sizeof(real));
  cudaMalloc(&ensemble_dUdL_d, nL*L_1D_bins*sizeof(real));
  cudaMemset(ensemble_dUdL_d, 0, nL*L_1D_bins*sizeof(real));
  cudaMalloc(&abf_TI_d, nL*sizeof(real));
  cudaMemset(abf_TI_d, 0, nL*sizeof(real));
  cudaMalloc(&dABF_dl_d, nL*sizeof(real));
  cudaMemset(dABF_dl_d, 0, nL*sizeof(real));

  // Path Storage for all paths between i->j
  int numPaths = 0; 
  for(int i = 1; i < siteCount; i++){ // don't include environment
    int subs = blocksPerSite[i];
    numPaths += subs*(subs-1);
  }
  path_count = numPaths;
  // O(L_1D_bins*blockCount^2)
  cudaMalloc(&path_weights_d, numPaths*L_1D_bins*sizeof(real));
  cudaMalloc(&path_unweights_d, numPaths*L_1D_bins*sizeof(real));
  cudaMalloc(&path_weight_offsets_d, numPaths*L_1D_bins*sizeof(real));
  cudaMalloc(&path_samples_d, numPaths*sizeof(real));
  cudaMalloc(&path_sample_offsets_d, numPaths*sizeof(real));
  cudaMalloc(&path_unsamples_d, numPaths*sizeof(real));
  real weight[numPaths*L_1D_bins];
  real offsets[numPaths*L_1D_bins];
  real prior = 0;
  for(int i = 0; i < numPaths*L_1D_bins; i++){
    weight[i] = prior;
    offsets[i] = -1e8;
  }
  real init_tot_path_weights[numPaths];
  real init_sample_off[numPaths];
  for(int i = 0; i < numPaths; i++){
    init_tot_path_weights[i] = prior*L_1D_bins;
    init_sample_off[i] = -1e8;
  }
  cudaMemcpy(path_weights_d, weight, numPaths*L_1D_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(path_unweights_d, weight, numPaths*L_1D_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(path_weight_offsets_d, offsets, numPaths*L_1D_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(path_samples_d, init_tot_path_weights, numPaths*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(path_unsamples_d, init_tot_path_weights, numPaths*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(path_sample_offsets_d, init_sample_off, numPaths*sizeof(real), cudaMemcpyDefault);
  cudaMalloc(&path_weighted_dUdL_d, numPaths*L_1D_bins*sizeof(real));
  cudaMemset(path_weighted_dUdL_d, 0, numPaths*L_1D_bins*sizeof(real));
  cudaMalloc(&path_weighted_dUdL2_d, numPaths*L_1D_bins*sizeof(real));
  cudaMemset(path_weighted_dUdL2_d, 0, numPaths*L_1D_bins*sizeof(real));
  cudaMalloc(&path_ensemble_dUdL_d, numPaths*L_1D_bins*sizeof(real));
  cudaMemset(path_ensemble_dUdL_d, 0, numPaths*L_1D_bins*sizeof(real));
  cudaMalloc(&path_dUdL_variance_d, numPaths*L_1D_bins*sizeof(real));
  cudaMemset(path_dUdL_variance_d, 0, numPaths*L_1D_bins*sizeof(real));

  cudaMalloc(&path_histogram_d, (numPaths/2)*L_2D_bins*dUdL_bins*sizeof(real));
  cudaMemset(path_histogram_d, 0, (numPaths/2)*L_2D_bins*dUdL_bins*sizeof(real));
}

void Msld::recv_meta(){
  cudaMemcpy(hist_potential, hist_potential_d, blockCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdL_msld, dUdL_msld_d, blockCount*sizeof(real), cudaMemcpyDefault);
}

void Msld::log_sampling(System* system, int step){
  State* s = system->state;
  Run* r = system->run;
  if(step % (sample_freq*100) != 0 || step == 0){
    return;
  }

  int len = (blockCount-1)*L_1D_bins;
  real counts[len], ensemble_dUdL[len], average_dUdL[len], var_dUdL[len];
  real path_weights[path_count*L_1D_bins], path_ensemble_dUdL[path_count*L_1D_bins];
  real path_var[path_count*L_1D_bins], path_dUdL_diff[path_count*L_1D_bins];
  real path_dUdL_diff_var[path_count*L_1D_bins];
  cudaMemcpy(counts, histogram_1D_d, len*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(ensemble_dUdL, ensemble_dUdL_d, len*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(average_dUdL, average_dUdL_d, len*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(var_dUdL, variance_dUdL_d, len*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(path_weights, path_weights_d, path_count*L_1D_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(path_ensemble_dUdL, path_ensemble_dUdL_d, path_count*L_1D_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(path_var, path_dUdL_variance_d, path_count*L_1D_bins*sizeof(real), cudaMemcpyDefault);

  system->state->recv_lambda();

  printf("Step: %ld\n", r->step);
  int prev_subs = 0; // num of alchemical blocks already logged
  int prev_paths = 0;
  for (int site = 1; site < siteCount; site++) { // Skip environment site
    real samples = 0;
    for (int j = 0; j < L_1D_bins; j++) {
      samples += counts[j];
    }
    real fractionPhysical = 0.0;
    for (int sub = prev_subs; sub < prev_subs+blocksPerSite[site]; sub++) {
      int start = sub*L_1D_bins;
      int physical = counts[start+L_1D_bins-1];
      fractionPhysical += physical;
      printf("Site %d, Sub %d, Lambda: %f, Count > .99: %d / %d \n", 
        site-1, sub, s->lambda[sub+1], physical, (int) samples); 
      printf("Counts: [ ");
      for (int i = 0; i < L_1D_bins; i++){
        printf("%d, ", (int)counts[start+i]);
      }
      printf("]\n");
      if(abf){
        printf("All Paths Counts:\n");
        int count = 0;
        for(int i = 0; i < blocksPerSite[site]; i++){
          if(i == sub-prev_subs){
            printf("  No self path! \n");
          } else {
            printf("  Sub %d -> %d Path: [", i, sub);
            for(int j = 0; j < L_1D_bins; j++){
              printf(" %6.2f, ", path_weights[(prev_paths+count)*L_1D_bins + j]);
            }
            printf(" ]\n");
            count++;
          }
        }
        printf("All Paths <dU/dL>:\n");
        count = 0;
        for(int i = 0; i < blocksPerSite[site]; i++){
          if(i == sub-prev_subs){
            printf("  No self path! \n");
          } else {
            printf("  Sub %d -> %d Path: [", i, sub);
            for(int j = 0; j < L_1D_bins; j++){
              printf(" %6.2f, ", path_ensemble_dUdL[(prev_paths+count)*L_1D_bins + j]);
            }
            printf(" ]\n");
            count++;
          }
        }
        printf("All Paths <Std>:\n");
        count = 0;
        for(int i = 0; i < blocksPerSite[site]; i++){
          if(i == sub-prev_subs){
            printf("  No self path! \n");
          } else {
            printf("  Sub %d -> %d Path: [", i, sub);
            for(int j = 0; j < L_1D_bins; j++){
              printf(" %6.2f, ", sqrt(path_var[(prev_paths+count)*L_1D_bins + j]));
            }
            printf(" ]\n");
            count++;
          }
        } 
        prev_paths += blocksPerSite[site]-1; // no i->i path
      }
      printf("\n");
    }
    if(abf){ // Free Energy
      int nsubs = blocksPerSite[site];
      int npaths = nsubs * (nsubs - 1); // all i->j paths, no self-paths
      real path_integral[nsubs][nsubs];
      for (int i = 0; i < nsubs; i++) {
        for (int j = 0; j < nsubs; j++) {
          path_integral[i][j] = 0.0;
        }
      }
      int path_idx = 0;
      for (int i = 0; i < nsubs; i++) {
        for (int j = 0; j < nsubs; j++) {
          if (i == j) continue;
          // Trapz sum
          real delta_TI = 0.0;
          for (int b = 0; b < L_1D_bins-1; b++) {
            real du = path_ensemble_dUdL[path_idx * L_1D_bins + b] + path_ensemble_dUdL[path_idx*L_1D_bins + b+1];
            du /= 2.0;
            real width = 1.0 / (L_1D_bins - 1);
            delta_TI += du * width;
          }
          path_integral[i][j] = delta_TI;
          path_idx++;
        }
      }
      // Print so that 0->j relative looks as we would expect
      printf("Free Energy Differences:\n");
      for (int i = 0; i < nsubs; i++) {
        printf("dG %d -> j [ ", i); // i->j means i is the reference state
        for (int j = 0; j < nsubs; j++) {
          if (j >= i){
            // dGij = int(<dU/dLj - dU/dLi>) = int(<dU/dLj>) - int(<dU/dLi>) 
            // Direction of sums don't matter 
            printf("%6.2f, ", path_integral[j][i] - path_integral[i][j]); // ij is reference substituents integral
          } else {
            printf("        ");
          }
        }
        printf("]\n");
      }
    }
    fractionPhysical /= samples;
    printf("Site fraction Physical: %f\n\n", fractionPhysical);
    prev_subs += blocksPerSite[site];
  }
}

__global__ void add_sample_1D_kernel(
  int nL, real* lambdas, 
  real* lambdaForce, 
  real* dUdL_msld, 
  real* dUdL_alf, 
  real* dUdL_bonded, 
  real* dUdL_abf, 
  int* lambdaSites, int* subsPerSite,
  int L_bins, real L_max, real L_min, 
  real* histogram_counts, real* average_dUdL, 
  real* average_dUdL2, real* variance, 
  real* hist_potential, real kT,
  real* weights, real* offsets, real* weighted_dUdL,
  real* ensemble_dUdL, 
  bool update, // stops path/oss sampling/fes updating
  int warmup_samples,
  real edge_std,
  real* path_samples,
  real* path_sample_offsets,
  real* path_weights, 
  real* path_unsamples,
  real* path_unweights,
  real* path_weight_offsets,
  real* path_weighted_dUdL,
  real* path_weighted_dUdL2,
  real* path_ensemble_dUdL,
  real* path_dUdL_variance,
  bool oss,
  real* path_histogram,
  int L_2D_bins,
  int dUdL_bins, real dUdL_max, real dUdL_min,
  real bias_mag, real temper_amount
) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<nL) { // nL is nBlock-1
    real lambda = lambdas[i+1];
    real dUdL =  dUdL_msld[i+1]; 
    int bin = get_histogram_index(lambda, L_bins, L_max, L_min); 
    int hist_bin = i*L_bins + bin;

    // Simple Average Stats
    histogram_counts[hist_bin] += 1;
 
    // Weighted Stats
    if (update){
      // Bias from PB Meta
      real bias = 0;
      for(int j = 0; j < nL; j++){
        bias += hist_potential[j+1];
      } 
      bias /= kT;
      if(bias > offsets[hist_bin]){
        real correction = exp(offsets[hist_bin] - bias);
        weights[hist_bin] *= correction;
        weighted_dUdL[hist_bin] *= correction;
        offsets[hist_bin] = bias;
      }
      weights[hist_bin] += exp(bias - offsets[hist_bin]);
      weighted_dUdL[hist_bin] += dUdL*exp(bias - offsets[hist_bin]);
      ensemble_dUdL[hist_bin] = weighted_dUdL[hist_bin] / weights[hist_bin];
    }

    // N^2 Path Updates 
    if(update){
      // Compute start of this blocks paths
      int site = lambdaSites[i+1];
      int start_path = 0; // number of previous paths
      int start_hist = 0; // number of previous histograms
      int start_block = 0; // number of previous blocks
      for(int j = 1; j < site; j++){ // not including env
        start_path += subsPerSite[j] * (subsPerSite[j]-1);
        start_hist += subsPerSite[j] * (subsPerSite[j]-1) / 2; 
        start_block += subsPerSite[j];
      }
      start_path += (i - start_block) * (subsPerSite[site]-1); // add up to this blocks path position within the site
      for(int j = 0; j < i-start_block; j++){
        start_hist += subsPerSite[site]-1-j;
      }
      // Sum of lambda squares at this site (for distance)
      real sum = 0;
      for(int j = start_block; j < start_block + subsPerSite[site]; j++){
        sum += lambdas[j+1]*lambdas[j+1];
      }
      // Compute bias from metadynamics 
      real bias = 0;
      for(int j = 0; j < nL; j++){
        bias += hist_potential[j+1]; // blockCount-1 length array
      }
      bias /= kT;
      // For every sub in my site
      int count = 0;
      int count_hist = 0;
      for(int j = start_block; j < start_block+subsPerSite[site]; j++){ // doesn't have path to itself
        if (i == j){ continue; }
        real partner_lambda = lambdas[j+1]; 
        real partner_dUdL = dUdL_msld[j+1];
        real dUdL_diff = (dUdL_msld[j+1]-dUdL_bonded[j+1]) - (dUdL_msld[i+1]-dUdL_bonded[i+1]);
        // Distance to point on (i,j)-edge of simplex
        real combined_lambda = (1 - partner_lambda + lambda) / 2.0; // project onto edge so (li=0, lj=1) is at 0
        real di = lambda - combined_lambda;
        real dj = partner_lambda - (1 - combined_lambda);
        real distance2 = sum - lambda*lambda - partner_lambda*partner_lambda; 
        distance2 += di*di + dj*dj;
        real weight = exp(-.5*distance2/(edge_std*edge_std)); // KDE of edge dU/dL samples
        // Add sample to projected bin in (lij, dU/dLi-dU/dLj) space
        if(i < j && oss){
          int X = get_histogram_index(combined_lambda, L_2D_bins, L_max, L_min);
          int Y = get_histogram_index(dUdL_diff, dUdL_bins, dUdL_max, dUdL_min);
          if(Y <= dUdL_bins && Y >= 0 && X <= L_2D_bins && X >= 0){
            real decay = exp(-bias / temper_amount);
            int index = (start_hist+count_hist)*L_2D_bins*dUdL_bins + X*dUdL_bins + Y;
            path_histogram[index] += bias_mag*decay*weight;
            //printf("path: %d, i: %d, j: %d, li: %f, lj: %f, bias: %f, comb: %f, dUdL_diff: %f, (%d, %d)\n", 
            //  start_hist + count_hist, i, j, lambda, partner_lambda, bias, combined_lambda, dUdL_diff, X, Y);
          } else {
            //printf("Out of bounds!!\n");
          }
          count_hist++;
        }
        // Add sample to projected bin
        bin = get_histogram_index(combined_lambda, L_bins, L_max, L_min); 
        hist_bin = (start_path+count)*L_bins + bin; // path# * length + index
        path_unsamples[start_path+count] += weight;
        path_unweights[hist_bin] += weight;
        // Add to weighted total
        if(bias > path_sample_offsets[start_path+count]){
          real correction = exp(path_sample_offsets[start_path+count] - bias);
          path_samples[start_path+count] *= correction;
          path_sample_offsets[start_path+count] = bias;
        }
        path_samples[start_path+count] += weight*exp(bias-path_sample_offsets[start_path+count]);

        // Add to weighted path
        if(bias > path_weight_offsets[hist_bin]){
          real correction = exp(path_weight_offsets[hist_bin] - bias);
          path_weights[hist_bin] *= correction;
          path_weighted_dUdL[hist_bin] *= correction;
          path_weighted_dUdL2[hist_bin] *= correction;
          path_weight_offsets[hist_bin] = bias;
        }
        weight *= exp(bias-path_weight_offsets[hist_bin]); 
        path_weights[hist_bin] += weight;
        path_weighted_dUdL[hist_bin] += weight*dUdL;
        path_weighted_dUdL2[hist_bin] += weight*dUdL*dUdL;
        if(path_weights[hist_bin] > 1e-5){ // prevent nan
          path_ensemble_dUdL[hist_bin] = path_weighted_dUdL[hist_bin] / path_weights[hist_bin];
          real ens_dUdL2 = path_weighted_dUdL2[hist_bin] / path_weights[hist_bin];
          // <dU/dL^2> - <dU/dL>^2
          path_dUdL_variance[hist_bin] = ens_dUdL2 - pow(path_ensemble_dUdL[hist_bin], 2.0);
        }
        if(path_unweights[hist_bin] < warmup_samples){ // based on number of samples only accounting for the distance
          path_ensemble_dUdL[hist_bin] *= path_unweights[hist_bin]/((real)warmup_samples);
          // Don't do this for dUdL_diff since its used for a different purpose
        }
        count++;
      }
    }
  }
}

void Msld::add_sample(System* system, int step) { // Step = 0 during NPT steps
  cudaStream_t stream = 0;
  Run* r = system->run;
  State* s = system->state;
  int shMem = 0;

  if (system->run) {
    stream=system->run->ossBias;
  }

  if(step % sample_freq != 0 || step == 0){ 
    return;
  }

  if(step % (sample_freq*10000) == 0){
    // This hasn't been modified to do paths yet
    //write_histogram_file(system, "hist_restart.txt", false);
  }

  // This continues after bias updates 
  real kT = system->state->leapParms1->kT; 
  add_sample_1D_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,0,stream>>>(
    blockCount-1, s->lambda_fd, 
    s->lambdaForce_d, 
    dUdL_msld_d, dUdL_alf_d, dUdL_bonded_d, dUdL_abf_d,
    lambdaSite_d, blocksPerSite_d,
    L_1D_bins, L_max, L_min, 
    histogram_1D_d, average_dUdL_d, average_dUdL2_d, variance_dUdL_d,
    hist_potential_d, s->leapParms1->kT, 
    weights_d, offsets_d, weighted_dUdL_d, ensemble_dUdL_d, 
    step < update_steps, // stop updating <dU/dL> that ABF uses
    warmup_samples,
    edge_KDE_std,
    path_samples_d,
    path_sample_offsets_d,
    path_weights_d,
    path_unsamples_d,
    path_unweights_d,
    path_weight_offsets_d,
    path_weighted_dUdL_d,
    path_weighted_dUdL2_d,
    path_ensemble_dUdL_d,
    path_dUdL_variance_d,
    oss,
    path_histogram_d,
    L_2D_bins,
    dUdL_bins,
    dUdL_max, dUdL_min,
    bias_mag,
    temper_amount
  );
}

// This is done on GPU just to avoid moving data back and forth
__global__ void getforce_abf_kernel(
  real kT, int nL,
  real* lambdas, 
  real* lambdaForce, real* dUdL_msld, real* dUdL_bonded,
  int num_bins, real L_max, real L_min,
  int* lambdaSite, int* subsPerSite,
  real* path_ensemble_dUdL,
  real_e *energy 
) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;
  if (i < nL) {
    // ABF along paths
    int site = lambdaSite[i+1];
    int start_path = 0; // number of previous paths
    int start = 0; // number of previous blocks
    for(int j = 1; j < site; j++){ // not including env
      start_path += subsPerSite[j] * (subsPerSite[j]-1);
      start += subsPerSite[j];
    }
    start_path += (i - start) * (subsPerSite[site]-1); // add up to this blocks position within the site
    real lambda = lambdas[i+1];
    int count = 0;
    for(int j = start; j < start+subsPerSite[site]; j++){
      if(j == i){continue;}
      real partner_lambda = lambdas[j+1]; 
      real combined_lambda = (1.0 - partner_lambda + lambda)/2.0; // .5 + .5*li - .5*lj
      real dproj_dli = 1.0/2.0;
      real dproj_dlj = -1.0/2.0;
      // |lj| / (1.0 - |li|)
      real fl = lambda < .99999 ? partner_lambda / (1.0-lambda) : 1.0 / (subsPerSite[site]-1.0);
      // fyi - product rules also work
      real dfl_dli = 0; // These are zero if using abs value function, these aren't used
      real dfl_dlj = 0; // Making this not zero for ABF requires something that isn't computed
      int bin = get_histogram_index(combined_lambda, num_bins, L_max, L_min); 
      int hist_bin = (start_path+count)*num_bins + bin;

      // Lerp implementation of ABF along this path
      real res = 1.0/(num_bins-1.0); // also the width
      real center = bin*res;
      real dUdL = path_ensemble_dUdL[hist_bin];
      real dist = combined_lambda-center;
      real partner_center, partner_dUdL, interp;
      real dUdL_abf = 0.0;
      int partner_id = 0;
      if(dist > 0){ // lambda in upper half of bin -> never true for bin=num_bins-1
        partner_center = center + res;
        partner_dUdL = path_ensemble_dUdL[hist_bin+1];
        partner_id = hist_bin+1;
        interp = dist / res;
        dUdL_abf = (1 - interp)*dUdL + interp*partner_dUdL;
      } else {
        partner_center = center - res;
        partner_dUdL = path_ensemble_dUdL[hist_bin-1];
        partner_id = hist_bin-1;
        interp = (combined_lambda - partner_center) / res;
        dUdL_abf = (1 - interp)*partner_dUdL + interp*dUdL;
      }
      // we add -'ve of <dU/dL>
      dUdL_abf = -dUdL_abf;
      if(i == 0 && false){
        printf("i: %d, li: %f, lj: %f, dUdL_abf: %f, lambdaForce[i+1]: %f, sum: %f\n",
          i, lambda, partner_lambda, dUdL_abf, lambdaForce[i+1], dUdL_abf + lambdaForce[i+1]);
      }
      atomicAdd(&lambdaForce[i+1], dUdL_abf);
    }
  }

  if (energy) {
    __syncthreads();
    real_sum_reduce(lEnergy, sEnergy, energy);
  }
}

__global__ void getforce_hist_kernel(
  int blockCount, int paths, 
  int siteCount, int* lambdaSites, int* subsPerSite, 
  real* histogram, real* lambdas, real* dUdL_msld, real* dUdL_bonded,
  int dUdL_bins, real dUdL_max, real dUdL_min, int dUdL_search,
  int L_bins, real L_max, real L_min, int L_search,
  real dUdL_std, real L_std,
  real* hist_potential, real* lambdaForce, real* dGdF
) {
  // 2D grid: blockIdx.y = lambda path index, blockIdx.x*blockDim.x+threadIdx.x = L search offset
  int iSearch = blockIdx.x * blockDim.x + threadIdx.x;
  int path = blockIdx.y;
  // Shared memory for energy reduction
  extern __shared__ real sEnergy[];
  real local_bias = 0.0;
  real local_dUdL_force = 0.0;
  real local_L_force = 0.0;
  if (path < paths) {
    // path is one of the i->j paths, i < j, combined_lambda = (1 - lj + li) / 2
    int nL = blockCount-1;
    int iL = 0; 
    int jL = 0;
    int site = 0;
    // Find which lambdas i & j this path corresponds to by guess & check in order
    int site_paths = 0;
    int prev_block = 0;
    real site_sum = 0;
    for(int j = 1; j < siteCount; j++){
      int count = 0;
      real sum = 0;
      for(int k = 0; k < subsPerSite[j]; k++){
        for(int l = k+1; l < subsPerSite[j]; l++){
          sum += lambdas[prev_block + k] + lambdas[prev_block + l];
          int p = site_paths + count;
          count++;
          if(path == p){
            iL = prev_block + k;
            jL = prev_block + l;
          }
        }
      }
      if ((iL != 0 || jL != 0) && site_sum < 1e-5){
        site_sum = sum;
      }
      site_paths += subsPerSite[j]*(subsPerSite[j]-1)/2;
      prev_block += subsPerSite[j];
    }
    real dUdL = (dUdL_msld[jL+1]-dUdL_bonded[jL+1]) - (dUdL_msld[iL+1]-dUdL_bonded[iL+1]);
    real lam = (1.0 - lambdas[jL+1] + lambdas[iL+1]) / 2.0;
    real interp = lambdas[jL+1] + lambdas[iL+1] / site_sum;
    // Get location in histogram (L, dUdL) => (X, Y) starting from lower left corner of hist
    int X = get_histogram_index(lam, L_bins, L_max, L_min);
    int Y = get_histogram_index(dUdL, dUdL_bins, dUdL_max, dUdL_min);
    int j = X - L_search + iSearch;
    if (j >= X-L_search && j <= X+L_search) {
      if(false && j == X)
        printf("path: %d, li: %d, lj: %d, x: %d, interp: %f, site_sum: %f, lam: %f, dUdL: %f, L_search: %d\n", path, iL, jL, j, interp, site_sum, lam, dUdL, L_search);
      int L_index = j;
      real mirrorFactor = 1;
      L_index = (L_index < 0) ? -L_index : L_index;
      L_index = (L_index > L_bins-1) ? L_index - 2*(L_index-(L_bins-1)) : L_index;
      mirrorFactor = L_index == 0 || L_index == L_bins-1 ? 2.0 : 1.0;
      if (L_index >= 0 && L_index < L_bins) {
        real L_resolution = (L_max-L_min)/(L_bins-1.0);
        real L_center = L_min + j*L_resolution; 
        real L_distance = (lam - L_center) / L_std;
        real L_gaussian = expf(-0.5*L_distance*L_distance);
        // Slice of dU/dL space
        for (int k = Y-dUdL_search; k <= Y+dUdL_search; k++) {
          int dUdL_index = k;
          if (dUdL_index < 0) continue;
          if (dUdL_index >= dUdL_bins) break;

          real dUdL_resolution = (dUdL_max-dUdL_min)/(dUdL_bins-1.0);
          real dUdL_center = dUdL_min + k*dUdL_resolution;
          real dUdL_distance = (dUdL - dUdL_center) / dUdL_std;
          real dUdL_gaussian = expf(-0.5*dUdL_distance*dUdL_distance);
          int index = path*dUdL_bins*L_bins + L_index*dUdL_bins + dUdL_index;
          real weight = mirrorFactor * histogram[index];

          real tmp_bias = weight * L_gaussian * dUdL_gaussian;
          local_bias += tmp_bias;
          // dUdL & L distances already include 1 div by respective std
          local_dUdL_force += -dUdL_distance/dUdL_std * tmp_bias;
          local_L_force += -L_distance/L_std * tmp_bias;
        }
        real dproj_dli = .5;
        real dproj_dlj = -.5;
        atomicAdd(&dGdF[iL+1], -interp*local_dUdL_force); // since we subtracted this lambda force
        atomicAdd(&dGdF[jL+1], interp*local_dUdL_force); 
        atomicAdd(&lambdaForce[iL+1], interp*local_L_force*dproj_dli);
        atomicAdd(&lambdaForce[jL+1], interp*local_L_force*dproj_dlj);
        atomicAdd(&hist_potential[iL+1], interp*local_bias);
      }
      if(iL == 7 && false)
        printf("bias: %f, dgdf: %f, dgdl: %f\n", local_bias, local_dUdL_force, local_L_force);
    }
  }
}

void Msld::getforce_oss(System *system, bool calcEnergy) {
  cudaStream_t stream = 0;
  Run *r = system->run;
  State *s = system->state;
  real_e *pEnergy = NULL;
  int shMem = 0;

  if (r->calcTermFlag[eebias] == false) return;
  if (calcEnergy) {
    shMem = BLMS*sizeof(real)/32;
    pEnergy=s->energy_d+eebias;
  }
  if (system->run) {
    stream=system->run->ossBias;
  }

  // Change which <dU/dL> we use for ABF forces
  real* dUdL = ensemble_dUdL_d;
  int bins = L_1D_bins;
  real kT = system->state->leapParms1->kT; 
  getforce_abf_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    s->leapParms1->kT, blockCount-1,
    s->lambda_fd, s->lambdaForce_d, dUdL_msld_d, dUdL_bonded_d, 
    L_1D_bins, L_max, L_min,
    lambdaSite_d, blocksPerSite_d,
    path_ensemble_dUdL_d, 
    pEnergy
  );

  // Set up grid and block dimensions
  L_search = (4.0*L_std/((abs(L_max) + abs(L_min))/L_2D_bins));
  dUdL_search = (4.0*dUdL_std/((abs(dUdL_max) + abs(dUdL_min))/dUdL_bins));
  dim3 blockDim(BLMS, 1, 1);
  dim3 gridDim;
  gridDim.x = (2*L_search + 1 + BLMS - 1) / BLMS; // Ceiling division for L search range
  gridDim.y = path_count / 2;                     // Number of lambda paths
  gridDim.z = 1;
  getforce_hist_kernel<<<gridDim, blockDim, shMem, stream>>>(
    blockCount, path_count/2, 
    siteCount, lambdaSite_d, blocksPerSite_d, 
    path_histogram_d, s->lambda_fd, dUdL_msld_d, dUdL_bonded_d,
    dUdL_bins, dUdL_max, dUdL_min, dUdL_search,
    L_2D_bins, L_max, L_min, L_search,
    dUdL_std, L_std,
    hist_potential_d, s->lambdaForce_d, dGdF_d
  );
}

__global__ void calc_lambda_from_theta_kernel(real_x *lambda,real_x *theta,int siteCount,int *siteBound,real fnex)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j,ji,jf;
  real_x lLambda;
  real_x norm=0;

  if (i<siteCount) {
    ji=siteBound[i];
    jf=siteBound[i+1];
    for (j=ji; j<jf; j++) {
      lLambda=exp(fnex*sin(theta[j]*ANGSTROM));
      lambda[j]=lLambda;
      norm+=lLambda;
    }
    norm=1/norm;
    for (j=ji; j<jf; j++) {
      lambda[j]*=norm;
    }
  }
}

__global__ void push_lambda_kernel(int nL, real_x *lambda, real_x *delta){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i < nL){
    lambda[i+1] += delta[i+1];
  }
}

void Msld::calc_lambda_from_theta(cudaStream_t stream,System *system)
{
  State *s=system->state;
  if (!fix) { // ffix
    calc_lambda_from_theta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,0,stream>>>(s->lambda_d,s->theta_d,siteCount,siteBound_d,fnex);
  } else {
    if(slow_fix){
      push_lambda_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,0,stream>>>(blockCount-1, s->lambda_d, lambda_delta_d);
      if (system->run->step == system->run->nsteps-2){ // not sure why this needs to be 2
        slow_fix = false;
      }
    }
    cudaMemcpy(s->theta_d,s->lambda_d,s->lambdaCount*sizeof(real_x),cudaMemcpyDeviceToDevice);
  }
}

void Msld::init_lambda_from_theta(cudaStream_t stream,System *system)
{
  State *s=system->state;
  if (!fix) { // ffix
    calc_lambda_from_theta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,0,stream>>>(s->lambda_d,s->theta_d,siteCount,siteBound_d,fnex);
  } else {
    cudaMemcpy(s->lambda_d,s->theta_d,s->lambdaCount*sizeof(real_x),cudaMemcpyDeviceToDevice);
  }
}

__global__ void calc_thetaForce_from_lambdaForce_kernel(real *lambda,real *theta,real_f *lambdaForce,real_f *thetaForce,int blockCount,int *lambdaSite,int *siteBound,real fnex)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j, ji, jf;
  real li, fi;

  if (i<blockCount) {
    li=lambda[i];
    fi=lambdaForce[i];
    ji=siteBound[lambdaSite[i]];
    jf=siteBound[lambdaSite[i]+1];
    for (j=ji; j<jf; j++) {
      fi+=-lambda[j]*lambdaForce[j];
    }
    fi*=li*fnex*cosf(ANGSTROM*theta[i])*ANGSTROM;
    atomicAdd(&thetaForce[i],fi);
  }
}

void Msld::calc_thetaForce_from_lambdaForce(cudaStream_t stream,System *system)
{
  State *s=system->state;
  if (!fix) { // ffix
    calc_thetaForce_from_lambdaForce_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,0,stream>>>(s->lambda_fd,s->theta_fd,s->lambdaForce_d,s->thetaForce_d,blockCount,lambdaSite_d,siteBound_d,fnex);
  }
}

__global__ void getforce_fixedBias_kernel(real *lambda,real *lambdaBias,real_f *lambdaForce, real* dU_alf, real_e *energy,int blockCount)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  if (i<blockCount) {
    atomicAdd(&lambdaForce[i],lambdaBias[i]);
    atomicAdd(&dU_alf[i],lambdaBias[i]);
    if (energy) {
      lEnergy=lambdaBias[i]*lambda[i];
    }
  }

  // Energy, if requested
  if (energy) {
    __syncthreads();
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void Msld::getforce_fixedBias(System *system,bool calcEnergy)
{
  cudaStream_t stream=0;
  Run *r=system->run;
  State *s=system->state;
  real_e *pEnergy=NULL;
  int shMem=0;

  if (r->calcTermFlag[eelambda]==false) return;

  if (calcEnergy) {
    shMem=BLMS*sizeof(real)/32;
    pEnergy=s->energy_d+eelambda;
  }
  if (system->run) {
    stream=system->run->biaspotStream;
  }

  getforce_fixedBias_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(s->lambda_fd,lambdaBias_d,s->lambdaForce_d, dUdL_alf_d, pEnergy,blockCount);
}

__global__ void getforce_variableBias_kernel(real *lambda,real_f *lambdaForce, real* dU_alf, real_e *energy,int variableBiasCount,struct VariableBias *variableBias)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct VariableBias vb;
  real li,lj;
  real fi,fj;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  if (i<variableBiasCount) {
    vb=variableBias[i];
    li=lambda[vb.i];
    lj=lambda[vb.j];
    if (vb.type==6) {
      lEnergy=vb.k*li*lj;
      fi=vb.k*lj;
      fj=vb.k*li;
    } else if (vb.type==8) {
      lEnergy=vb.k*li*lj/(li+vb.l0);
      fi=vb.k*vb.l0*lj/((li+vb.l0)*(li+vb.l0));
      fj=vb.k*li/(li+vb.l0);
    } else if (vb.type==10) {
      lEnergy=vb.k*lj*(1-expf(vb.l0*li));
      fi=vb.k*lj*(-vb.l0*expf(vb.l0*li));
      fj=vb.k*(1-expf(vb.l0*li));
    } else if (vb.type==1) {
      lEnergy=((li<vb.l0)?(vb.k*pow(li-vb.l0,vb.n)):0);
      fi=((li<vb.l0)?(vb.n*vb.k*pow(li-vb.l0,vb.n-1)):0);
      fj=0;
    } else if (vb.type==2) {
      lEnergy=((li>vb.l0)?(vb.k*pow(li-vb.l0,vb.n)):0);
      fi=((li>vb.l0)?(vb.n*vb.k*pow(li-vb.l0,vb.n-1)):0);
      fj=0;
    } else if (vb.type==3) {
      lEnergy=vb.k*pow(li-lj,vb.n);
      fi=vb.n*vb.k*pow(li-lj,vb.n-1);
      fj=-fi;
    } else if (vb.type==4) {
      real bicut=0.8;
      real bicut2in=1.5625; // 1/(bicut*bicut)
      lEnergy=((li>bicut)?(-vb.k):(-vb.k*(1-bicut2in*(li-bicut)*(li-bicut))));
      fi=((li>bicut)?0:(2*vb.k*bicut2in*(li-bicut)));
      fj=0;
    } else if (vb.type==5) {
      lEnergy=-vb.k*li;
      fi=-vb.k;
      fj=0;
    } else if (vb.type==7) {
      lEnergy=vb.k*li*(1-li)/(li+vb.l0);
      fi=vb.k*(vb.l0*(1+vb.l0)/((li+vb.l0)*(li+vb.l0))-1);
      fj=0;
    } else if (vb.type==9) {
      lEnergy=vb.k*lj*(1-pow((li+vb.l0)/vb.l0,vb.n));
      fi=-vb.k*lj*pow((li+vb.l0)/vb.l0,vb.n-1)/vb.l0;
      fj=vb.k*(1-pow((li+vb.l0)/vb.l0,vb.n));
    } else if (vb.type==11) {
      lEnergy=vb.k*pow(li,vb.l0)*pow(lj,vb.n);
      fi=vb.k*vb.l0*pow(li,vb.l0-1)*pow(lj,vb.n);
      fj=vb.k*vb.n*pow(li,vb.l0)*pow(lj,vb.n-1);
    } else if (vb.type==12) {
      lEnergy=vb.k*li*pow(lj,vb.n)/(li+vb.l0);
      fi=vb.k*vb.l0*pow(lj,vb.n)/((li+vb.l0)*(li+vb.l0));
      fj=vb.k*vb.n*li*pow(lj,vb.n-1)/(li+vb.l0);
/*
    } else if (vb.type==13 && vb.k) {
      lEnergy=vb.k*li*lj*(1-li-lj)/(vb.l0+1-li-lj);
      fi=-vb.k*vb.l0*li*lj/((vb.l0+1-li-lj)*(vb.l0+1-li-lj)); // partial force
      fj=fi+vb.k*li*(1-li-lj)/(vb.l0+1-li-lj); // full force
      fi=fi+vb.k*lj*(1-li-lj)/(vb.l0+1-li-lj); // full force
    } else if (vb.type==14) {
      lEnergy=vb.k*li*lj*(1-li-lj)/(vb.l0+li);
      fi=-vb.k*li*lj/(vb.l0+li); // partial force
      fj=fi+vb.k*li*(1-li-lj)/(vb.l0+li); // full force
      fi=fi+vb.k*vb.l0*lj*(1-li-lj)/((vb.l0+li)*(vb.l0+li)); // full force
*/
    } else {
      lEnergy=0;
      fi=0;
      fj=0;
    }
    atomicAdd(&lambdaForce[vb.i],fi);
    atomicAdd(&dU_alf[vb.i],fi);
    if (fj) { 
      atomicAdd(&lambdaForce[vb.j],fj); 
      atomicAdd(&dU_alf[vb.j],fj);
    }
  }

  // Energy, if requested
  if (energy) {
    __syncthreads();
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void Msld::getforce_variableBias(System *system,bool calcEnergy)
{
  cudaStream_t stream=0;
  Run *r=system->run;
  State *s=system->state;
  real_e *pEnergy=NULL;
  int shMem=0;

  if (r->calcTermFlag[eelambda]==false) return;

  if (calcEnergy) {
    shMem=BLMS*sizeof(real)/32;
    pEnergy=s->energy_d+eelambda;
  }
  if (system->run) {
    stream=system->run->biaspotStream;
  }

  if (variableBiasCount>0) {
    getforce_variableBias_kernel<<<(variableBiasCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(s->lambda_fd,s->lambdaForce_d, dUdL_alf_d, pEnergy,variableBiasCount,variableBias_d);
  }
}

__global__ void getforce_thetaCollBias_kernel(real *theta,real_f *thetaForce,real_e *energy,int siteCount,int *siteBound,real *k,real *n)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real nhigh,nlow;
  int j,ji,jf;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  if (i<siteCount) {
    ji=siteBound[i];
    jf=siteBound[i+1];
    nhigh=0;
    nlow=0;
    for (j=ji; j<jf; j++) {
      nhigh+=pow(((real)0.5)*sin(ANGSTROM*theta[j])+((real)0.5),n[i]);
      nlow+=pow(-((real)0.5)*sin(ANGSTROM*theta[j])+((real)0.5),n[i]);
    }
    lEnergy=((real)0.5)*k[i]*((nhigh-1)*(nhigh-1)+(nlow-(jf-ji-1))*(nlow-(jf-ji-1)));
    for (j=ji; j<jf; j++) {
      atomicAdd(&thetaForce[j], k[i]*n[i]*((nhigh-1)*pow(((real)0.5)*sin(ANGSTROM*theta[j])+((real)0.5),n[i]-1)-(nlow-(jf-ji-1))*pow(-((real)0.5)*sin(ANGSTROM*theta[j])+((real)0.5),n[i]-1))*cos(ANGSTROM*theta[j])*ANGSTROM);
    }
  }

  // Energy, if requested
  if (energy) {
    __syncthreads();
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}
__global__ void getforce_thetaIndeBias_kernel(real *theta,real_f *thetaForce,real_e *energy,int blockCount,int *lambdaSite,real *k)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;
  real ki,s,s3;

  if (i<blockCount) {
    ki=k[lambdaSite[i]];
    s=((real)0.5)-((real)0.5)*sin(ANGSTROM*theta[i]);
    s3=s*s*s;
    lEnergy=ki*(((real)1)-s*s3);
    atomicAdd(&thetaForce[i], ((real)2)*ki*s3*cos(ANGSTROM*theta[i])*ANGSTROM);
  }

  // Energy, if requested
  if (energy) {
    __syncthreads();
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void Msld::getforce_thetaBias(System *system,bool calcEnergy)
{
  cudaStream_t stream=0;
  Run *r=system->run;
  State *s=system->state;
  real_e *pEnergy=NULL;
  int shMem=0;

  if (r->calcTermFlag[eelambda]==false) return;



  if (calcEnergy) {
    shMem=BLMS*sizeof(real)/32;
    pEnergy=s->energy_d+eelambda;
  }
  if (system->run) {
    stream=system->run->biaspotStream;
  }

  if (thetaCollBiasCount!=0) {
    getforce_thetaCollBias_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(s->theta_fd,s->thetaForce_d,pEnergy,siteCount,siteBound_d,kThetaCollBias_d,nThetaCollBias_d);
  }

  if (thetaIndeBiasCount!=0) {
    getforce_thetaIndeBias_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(s->theta_fd,s->thetaForce_d,pEnergy,blockCount,lambdaSite_d,kThetaIndeBias_d);
  }
}

template <bool flagBox,typename box_type>
__global__ void getforce_atomRestraints_kernel(real3 *position,real3_f *force,box_type box,real_e *energy,int atomRestraintCount,int *atomRestraintBounds,int *atomRestraintIdx,real kRestraint)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int bounds[2];
  int j, idx;
  real3 x0, xCenter, dx;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  if (i<atomRestraintCount) {
    bounds[0]=atomRestraintBounds[i];
    bounds[1]=atomRestraintBounds[i+1];
    x0=position[atomRestraintIdx[bounds[0]]];
    xCenter=real3_reset<real3>();
    for (j=bounds[0]+1; j<bounds[1]; j++) {
      real3_inc(&xCenter,real3_subpbc<flagBox>(position[atomRestraintIdx[j]],x0,box));
    }
    real3_scaleself(&xCenter,((real)1.0)/(bounds[1]-bounds[0]));
    real3_inc(&xCenter,x0);

    for (j=bounds[0]; j<bounds[1]; j++) {
      idx=atomRestraintIdx[j];
      dx=real3_subpbc<flagBox>(position[idx],xCenter,box);
      at_real3_scaleinc(&force[idx],kRestraint,dx);
      if (energy) {
        lEnergy+=((real)0.5)*kRestraint*real3_mag2<real>(dx);
      }
    }
  }

  // Energy, if requested
  if (energy) {
    __syncthreads();
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_atomRestraintsT(System *system,box_type box,bool calcEnergy)
{
  cudaStream_t stream=0;
  Run *r=system->run;
  State *s=system->state;
  real_e *pEnergy=NULL;
  Msld *m=system->msld;
  int shMem=0;

  if (r->calcTermFlag[eebias]==false) return;

  if (calcEnergy) {
    shMem=BLMS*sizeof(real)/32;
    pEnergy=s->energy_d+eebias;
  }
  if (system->run) {
    stream=system->run->biaspotStream;
  }

  if (m->atomRestraintCount) {
    getforce_atomRestraints_kernel<flagBox><<<(m->atomRestraintCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>((real3*)s->position_fd,(real3_f*)s->force_d,box,pEnergy,m->atomRestraintCount,m->atomRestraintBounds_d,m->atomRestraintIdx_d,m->kRestraint);
  }
}

void Msld::getforce_atomRestraints(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_atomRestraintsT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_atomRestraintsT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

__global__ void getforce_chargeRestraints_kernel(real *lambda,real_f *lambdaForce,real_e *energy,int blockCount,real kChargeRestraint,real *lambdaCharge)
{
  int i=threadIdx.x;
  int j;
  real netCharge=0;
  extern __shared__ real sEnergy[];

  for (j=i; j<blockCount; j+=BLMS) {
    netCharge+=lambda[j]*lambdaCharge[j];
  }

  __syncthreads();
  netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,1);
  netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,2);
  netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,4);
  netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,8);
  netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,16);
  __syncthreads();
  if ((0x1F & threadIdx.x)==0) {
    sEnergy[threadIdx.x>>5]=netCharge;
  }
  __syncthreads();
  netCharge=0;
  if (threadIdx.x < (blockDim.x>>5)) {
    netCharge=sEnergy[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    if (blockDim.x>=64) netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,1);
    if (blockDim.x>=128) netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,2);
    if (blockDim.x>=256) netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,4);
    if (blockDim.x>=512) netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,8);
    if (blockDim.x>=1024) netCharge+=__shfl_down_sync(0xFFFFFFFF,netCharge,16);
  }
  if (threadIdx.x==0) {
    sEnergy[0]=netCharge;
    if (energy) {
      atomicAdd(energy,(real_e)(0.5*kChargeRestraint*netCharge*netCharge));
    }
  }
  __syncthreads();
  netCharge=sEnergy[0];

  for (j=i; j<blockCount; j+=BLMS) {
    if (j>0) {
      atomicAdd(&lambdaForce[j],kChargeRestraint*netCharge*lambdaCharge[j]);
    }
  }
}

void Msld::getforce_chargeRestraints(System *system,bool calcEnergy)
{
  cudaStream_t stream=0;
  Run *r=system->run;
  State *s=system->state;
  real_e *pEnergy=NULL;
  int shMem=0;

  if (r->calcTermFlag[eebias]==false) return;

  shMem=BLMS*sizeof(real)/32; // Always needs shared memory for reduction
  if (calcEnergy) {
    pEnergy=s->energy_d+eebias;
  }
  if (system->run) {
    stream=system->run->biaspotStream;
  }

  if (kChargeRestraint>0) {
    getforce_chargeRestraints_kernel<<<1,BLMS,shMem,stream>>>(s->lambda_fd,s->lambdaForce_d,pEnergy,blockCount,kChargeRestraint,lambdaCharge_d);
  }
}



void blade_init_msld(System *system,int nblocks)
{
  system+=omp_get_thread_num();
  if (system->msld) {
    delete(system->msld);
  }
  system->msld=new Msld();

  system->msld->blockCount=nblocks+1;
  system->msld->atomBlock=(int*)calloc(system->structure->atomList.size(),sizeof(int));
  system->msld->lambdaSite=(int*)calloc(system->msld->blockCount,sizeof(int));
  system->msld->lambdaBias=(real*)calloc(system->msld->blockCount,sizeof(real));
  system->msld->theta=(real_x*)calloc(system->msld->blockCount,sizeof(real_x));
  system->msld->thetaVelocity=(real_v*)calloc(system->msld->blockCount,sizeof(real_v));
  system->msld->thetaMass=(real*)calloc(system->msld->blockCount,sizeof(real));
  system->msld->thetaMass[0]=1;
  system->msld->lambdaCharge=(real*)calloc(system->msld->blockCount,sizeof(real));

  cudaMalloc(&(system->msld->atomBlock_d),system->structure->atomCount*sizeof(int));
  cudaMalloc(&(system->msld->lambdaSite_d),system->msld->blockCount*sizeof(int));
  cudaMalloc(&(system->msld->lambdaBias_d),system->msld->blockCount*sizeof(real));
  cudaMalloc(&(system->msld->lambdaCharge_d),system->msld->blockCount*sizeof(real));
}

void blade_dest_msld(System *system)
{
  system+=omp_get_thread_num();
  if (system->msld) {
    delete(system->msld);
  }
  system->msld=NULL;
}

void blade_add_msld_atomassignment(System *system,int atomIdx,int blockIdx)
{
  system+=omp_get_thread_num();
  system->msld->atomBlock[atomIdx-1]=blockIdx-1;
}

void blade_add_msld_initialconditions(System *system,int blockIdx,int siteIdx,double theta0,double thetaVelocity,double thetaMass,double fixBias,double blockCharge)
{
  blockIdx-=1;
  system+=omp_get_thread_num();
  system->msld->lambdaSite[blockIdx]=siteIdx-1;
  system->msld->theta[blockIdx]=theta0;
  system->msld->thetaVelocity[blockIdx]=thetaVelocity;
  system->msld->thetaMass[blockIdx]=thetaMass;
  system->msld->lambdaBias[blockIdx]=fixBias;
  system->msld->lambdaCharge[blockIdx]=blockCharge;
}

void blade_add_msld_termscaling(System *system,int scaleBond,int scaleUrey,int scaleAngle,int scaleDihe,int scaleImpr,int scaleCmap)
{
  system+=omp_get_thread_num();
  system->msld->scaleTerms[0]=scaleBond;
  system->msld->scaleTerms[1]=scaleUrey;
  system->msld->scaleTerms[2]=scaleAngle;
  system->msld->scaleTerms[3]=scaleDihe;
  system->msld->scaleTerms[4]=scaleImpr;
  system->msld->scaleTerms[5]=scaleCmap;
}

void blade_add_msld_flags(System *system,double gamma,double fnex,int useSoftCore,int useSoftCore14,int msldEwaldType,double kRestraint,double kChargeRestraint,double softBondRadius,double softBondExponent,double softNotBondExponent,int fix)
{
  system+=omp_get_thread_num();
  system->msld->gamma=gamma;
  system->msld->fnex=fnex;
  system->msld->useSoftCore=useSoftCore;
  system->msld->useSoftCore14=useSoftCore14;
  system->msld->msldEwaldType=msldEwaldType;
  system->msld->kRestraint=kRestraint;
  system->msld->kChargeRestraint=kChargeRestraint;
  system->msld->softBondRadius=softBondRadius;
  system->msld->softBondExponent=softBondExponent;
  system->msld->softNotBondExponent=softNotBondExponent;
  system->msld->fix=fix;
}

void blade_add_msld_bias(System *system,int i,int j,int type,double l0,double k,int n)
{
  system+=omp_get_thread_num();
  struct VariableBias vb;
  vb.i=i-1;
  vb.j=j-1;
  vb.type=type;
  // if (vb.type!=6 && vb.type!=8 && vb.type!=10)
  if (vb.type<=0 || vb.type>12) {
    fatal(__FILE__,__LINE__,"Type of variable bias (%d) is not a recognized type\n",vb.type);
  }
  vb.l0=l0;
  vb.k=k;
  vb.n=n;
  system->msld->variableBias_tmp.push_back(vb);
}

void blade_add_msld_thetacollbias(System *system,int sites,int i,double k,double n)
{
  system+=omp_get_thread_num();
  if (system->msld->thetaCollBiasCount==0) {
    system->msld->thetaCollBiasCount=sites;
    system->msld->kThetaCollBias=(real*)calloc(system->msld->thetaCollBiasCount,sizeof(real));
    system->msld->nThetaCollBias=(real*)calloc(system->msld->thetaCollBiasCount,sizeof(real));
  }
  system->msld->kThetaCollBias[i-1]=k;
  system->msld->nThetaCollBias[i-1]=n;
}

void blade_add_msld_thetaindebias(System *system,int sites,int i,double k)
{
  system+=omp_get_thread_num();
  if (system->msld->thetaIndeBiasCount==0) {
    system->msld->thetaIndeBiasCount=sites;
    system->msld->kThetaIndeBias=(real*)calloc(system->msld->thetaIndeBiasCount,sizeof(real));
  } 
  system->msld->kThetaIndeBias[i-1]=k;
}

void blade_add_msld_softbond(System *system,int i,int j)
{
  system+=omp_get_thread_num();
  Int2 i2;
  i2.i[0]=i-1;
  i2.i[1]=j-1;
  system->msld->softBonds.push_back(i2);
}

void blade_add_msld_atomrestraint(System *system)
{
  system+=omp_get_thread_num();
  std::vector<int> ar;
  ar.clear();
  system->msld->atomRestraints.push_back(ar);
}

void blade_add_msld_atomrestraint_element(System *system,int i)
{
  system+=omp_get_thread_num();
  system->msld->atomRestraints.back().push_back(i-1);
}
