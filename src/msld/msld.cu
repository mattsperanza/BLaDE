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
  } else if (strcmp(token, "alpha") == 0){
    system->msld->alpha=io_nextf(line);
  } else if (strcmp(token, "leus") == 0){
    system->msld->L_LEUS=io_nextb(line); 
  } else if (strcmp(token, "leus_func") == 0){
    std::string name = io_nexts(line);
    if(name=="linear"){
      system->msld->L_LEUS_function=leus_linear;
    } else if (name=="cubic"){
      system->msld->L_LEUS_function=leus_cubic;
    } else if (name=="quintic"){
      system->msld->L_LEUS_function=leus_quintic;
    } else if (name=="sin2"){
      system->msld->L_LEUS_function=leus_sin2;
    } else if (name=="x-sin"){
      system->msld->L_LEUS_function=leus_xsin;
      system->msld->xsin_n = (real)io_nexti(line);
    } else {
      printf("Function %s not supported for L-LEUS!", name);
      exit(1);
    }
  } else if (strcmp(token, "plateau_w") == 0){
    system->msld->plateau_w=io_nextf(line);
  } else if (strcmp(token, "transition_w") == 0){
    system->msld->transition_w=io_nextf(line);
  } else if (strcmp(token, "linear_shift") == 0){
    system->msld->linear_shift=io_nextb(line);
  } else if (strcmp(token, "LE") == 0){
    system->msld->LE=io_nextb(line);
  } else if (strcmp(token, "oss") == 0) {
    system->msld->oss=io_nextb(line);
  } else if (strcmp(token, "oss_k") == 0){
    system->msld->oss_k=io_nextf(line);
  } else if (strcmp(token, "oss_restartable") == 0){
    system->msld->oss_restartable = io_nextb(line);
  } else if (strcmp(token, "oss_log_freq") == 0){
    system->msld->oss_log_freq=io_nexti(line);
  } else if (strcmp(token, "oss_write_freq")==0){
    system->msld->oss_write_freq=io_nexti(line);
  } else if (strcmp(token, "bias_mag") == 0){
    system->msld->bias_mag=io_nextf(line);
  } else if (strcmp(token, "pb_meta") == 0){
    system->msld->pb_meta=io_nextb(line);
  } else if (strcmp(token, "temper") == 0){
    system->msld->temper=io_nextb(line);
  } else if (strcmp(token, "standard_tempering") == 0){
    system->msld->standard_tempering=io_nextb(line);
  } else if (strcmp(token, "temper_amount") == 0){
    system->msld->temper_amount=io_nextf(line);
  } else if (strcmp(token, "temper_offset") == 0){
    system->msld->temper_offset=io_nextf(line);
  } else if (strcmp(token, "T_std") == 0){
    system->msld->T_std=io_nextf(line);
  } else if (strcmp(token, "dUdT_std") == 0){
    system->msld->dUdT_std=io_nextf(line);
  } else if (strcmp(token, "bins_per_std") == 0){
    system->msld->bins_per_std=io_nexti(line);
  } else if (strcmp(token, "n_std_search") == 0){
    system->msld->n_std_search=io_nexti(line);
  } else if (strcmp(token, "dUdT_max") == 0){
    system->msld->dUdT_max=io_nextf(line);
  } else if (strcmp(token, "warmup_samples") == 0){
    system->msld->warmup_samples=io_nexti(line);
  } else if (strcmp(token, "update_steps") == 0){
    system->msld->update_steps=io_nexti(line);
  } else if (strcmp(token, "update_fe") == 0) {
    system->msld->update_fe_surface=io_nextb(line);
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

  // L_LEUS storage
  cudaMalloc(&dLdT_d, blockCount*sizeof(real)); 
  cudaMemset(dLdT_d, 1, blockCount*sizeof(real)); // makes OSS default to bias of dU/dL
  cudaMalloc(&d2LdT2_d, blockCount*sizeof(real));
  cudaMemset(d2LdT2_d, 0, blockCount*sizeof(real)); // makes OSS default to bias of dU/dL
  site_period = (real*) calloc(siteCount, sizeof(real));
  for(int i = 1; i < siteCount; i++){
    int Ns = system->msld->blocksPerSite[i];
    site_period[i] = Ns*(transition_w+plateau_w);
  }
  cudaMalloc(&site_period_d, siteCount*sizeof(real));
  cudaMemcpy(site_period_d, site_period, siteCount*sizeof(real), cudaMemcpyDefault);
  if(L_LEUS){
    memset(theta, 0, blockCount*sizeof(real_x));
  }

  // blockCount
  dUdT_msld = (real*)malloc(blockCount*sizeof(real));
  cudaMalloc(&dUdT_msld_d, blockCount*sizeof(real));
  cudaMemset(dUdT_msld_d, 0, blockCount*sizeof(real));
  dUdL_msld = (real*)malloc(blockCount*sizeof(real));
  cudaMalloc(&dUdL_msld_d, blockCount*sizeof(real));
  cudaMemset(dUdL_msld_d, 0, blockCount*sizeof(real));
  cudaMalloc(&dGdF_d, blockCount*sizeof(real)); 
  cudaMemset(dGdF_d, 0, blockCount*sizeof(real));
  cudaMalloc(&dGdT_d, blockCount*sizeof(real)); 
  cudaMemset(dGdT_d, 0, blockCount*sizeof(real));
  // site count
  bias_potential = (real*)malloc(siteCount*sizeof(real));
  cudaMalloc(&bias_potential_d, siteCount*sizeof(real)); 
  cudaMemset(bias_potential_d, 0, siteCount*sizeof(real)); 
  // Of total # of steps, use 1/4th to develop bias as default
  if (update_steps == -1){
    update_steps = .25*system->run->nsteps;
  }
  if (oss || LE){ // allocate histograms
    if(!system->msld->L_LEUS){
      printf("OSS cannot be run without L-LEUS theta dynamics scheme!\n");
      exit(1);
    }
    init_oss(system);
    // Restart and evaluate potential on histogram everywhere
    oss_restart_success = false; 
    if (oss_restartable) oss_restart_success = read_histogram_file(system, "hist_restart.txt");
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

/** 
 * Histogram defined to have elements for the first and last elements at max and min respectively. Width of the
 * first and last bins are half length. num_bins-1 whole bins fit in the [min, max) range. Uniform histogram.
 * Element is relative to min.
 * 
 * Safe because it won't index beyond what num_bins specifies.
 * 
 * Ex. Lambda bins: [0, .005), [.005, .015), ..., [.985, .995), [.995, 1.0) -> range [0,1), 101 bins
 *  Lambda Centers: [   0   ], [   .010   ], ..., [   .990   ], [   1.0   ]
 *    Lambda Index: [   0   ], [     1    ], ..., [    99    ], [   100   ] -> last is num_bins-1
*/
static __device__ int safe_histogram_index(real val, int num_bins, real max, real min){
  real tmp = val - min;
  real range = max - min;
  real resolution = range / (num_bins-1);
  int index = round(tmp/resolution);
  // If constraint fails don't access bad memory
  if(index < 0){
    index = 0;
  } else if(index >= num_bins){
    index = num_bins-1;
  }
  return index;
}

void Msld::init_oss(System* system){
  T_res = T_std / bins_per_std;
  dUdT_res = dUdT_std / bins_per_std;
  T_search = (int)(n_std_search*T_std/T_res); // 10 by default
  dUdT_search = (int)(n_std_search*dUdT_std/dUdT_res); // 10 by default
  if((2*T_search+1)*(2*dUdT_search+1) > 1024){
    printf("Please decrease n_std_search or bins_per_std to decrease threads required per histogram evaluation!");
    exit(1);
  }

  total_T_bins = 0;
  T_bins = (int*) calloc(siteCount, sizeof(int));
  dUdT_min = -dUdT_max;
  dUdT_bins = (int) ((abs(dUdT_max) + abs(dUdT_min)) / dUdT_res) + 1; // histogram has bins exactly on edges
  for(int i = 1; i < siteCount; i++){
    T_bins[i] = (int) (site_period[i] / T_res) + 1;
    total_T_bins += T_bins[i];
  }
  cudaMalloc(&T_bins_d, siteCount*sizeof(int));
  cudaMemcpy(T_bins_d, T_bins, siteCount*sizeof(int), cudaMemcpyDefault);

  cudaMalloc(&oss_histogram_d, total_T_bins*dUdT_bins*sizeof(real));
  cudaMemset(oss_histogram_d, 0, total_T_bins*dUdT_bins*sizeof(real));

  cudaMalloc(&oss_potential_d, total_T_bins*dUdT_bins*sizeof(real));
  cudaMemset(oss_potential_d, 0, total_T_bins*dUdT_bins*sizeof(real));

  cudaMalloc(&total_bias_d, sizeof(real));
  cudaMemset(total_bias_d, 0, sizeof(real));
  total_bias = (real*)malloc(sizeof(real));

  cudaMalloc(&oss_ensemble_dUdT_d, total_T_bins*sizeof(real));
  cudaMemset(oss_ensemble_dUdT_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&oss_eff_n_d, total_T_bins*sizeof(real));
  cudaMemset(oss_eff_n_d, 0, total_T_bins*sizeof(real));
  
  cudaMalloc(&oss_dUdT_var_d, total_T_bins*sizeof(real));
  cudaMemset(oss_dUdT_var_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&oss_theta_counts_d, total_T_bins*sizeof(real));
  cudaMemset(oss_theta_counts_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&oss_dUdT_max_d, total_T_bins*sizeof(int));
  cudaMemset(oss_dUdT_max_d, 0, total_T_bins*sizeof(int));

  cudaMalloc(&oss_max_pot_d, total_T_bins*sizeof(real));
  cudaMemset(oss_max_pot_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&oss_min_max_d, sizeof(real));
  cudaMemset(oss_min_max_d, 0, sizeof(real));

  int min[total_T_bins];
  for(int i = 0; i < total_T_bins; i++){ min[i] = dUdT_bins-1; }
  cudaMalloc(&oss_dUdT_min_d, total_T_bins*sizeof(int));
  cudaMemcpy(oss_dUdT_min_d, min, total_T_bins*sizeof(int), cudaMemcpyDefault);

  // Local Elevation
  cudaMalloc(&M_d, total_T_bins*sizeof(real));
  cudaMemset(M_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&theta_sweep_d, total_T_bins*sizeof(int));
  cudaMemset(theta_sweep_d, 0, total_T_bins*sizeof(int));

  real R[siteCount];
  for(int i = 0; i < siteCount; i++){ R[i] = 1.0; }
  cudaMalloc(&R_d, siteCount*sizeof(real));
  cudaMemcpy(R_d, R, siteCount*sizeof(real), cudaMemcpyDefault);

  cudaMalloc(&visited_bins_d, siteCount*sizeof(int));
  cudaMemset(visited_bins_d, 0, siteCount*sizeof(int));
}

void Msld::recv_meta(){
  cudaMemcpy(bias_potential, bias_potential_d, siteCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(total_bias, total_bias_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdL_msld, dUdL_msld_d, blockCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdT_msld, dUdT_msld_d, blockCount*sizeof(real), cudaMemcpyDefault);
}

void Msld::log_sampling(System* system, int step){
  State* s = system->state;
  Run* r = system->run;

  if(step % (oss_log_freq) != 0 || step == 0){ 
    return;
  }

  real ensemble_dUdT[total_T_bins], dUdT_var[total_T_bins], counts[total_T_bins], eff_counts[total_T_bins];
  real dGdF[blockCount];
  real min_max;
  cudaMemcpy(counts, oss_theta_counts_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(eff_counts, oss_eff_n_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(ensemble_dUdT, oss_ensemble_dUdT_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdT_var, oss_dUdT_var_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(bias_potential, bias_potential_d, siteCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dGdF, dGdF_d, blockCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&min_max, oss_min_max_d, sizeof(real), cudaMemcpyDefault);

  real M[total_T_bins]; 
  real R[siteCount];
  int theta_sweep[total_T_bins];
  int visited_bins[siteCount];
  cudaMemcpy(M, M_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(theta_sweep, theta_sweep_d, total_T_bins*sizeof(int), cudaMemcpyDefault);
  cudaMemcpy(visited_bins, visited_bins_d, siteCount*sizeof(int), cudaMemcpyDefault);
  cudaMemcpy(R, R_d, siteCount*sizeof(real), cudaMemcpyDefault);
  system->state->recv_state();
  system->state->recv_lambda();

  int prev_subs = 1; // num of alchemical blocks already logged
  int prev_bins = 0;
  int start = 0;
  for (int site = 1; site < siteCount; site++) { // Skip environment site
    printf("Step: %ld\n", r->step);
    printf("Site: %d, Theta: %f, Lambdas: [", site, system->state->theta[prev_subs]);
    real l_sum = 0;
    for(int i = prev_subs; i < prev_subs+blocksPerSite[site]; i++){
      printf(" %f,", system->state->lambda[i]);
    }
    printf("] \n");
    real U = max(bias_potential[site] - temper_offset, 0.0) / system->state->leapParms1->kT;
    real decay_local = exp(-U/temper_amount);
    U = max(min_max - temper_offset, 0.0) / system->state->leapParms1->kT;
    real decay_global = exp(-U/temper_amount);
    printf("Bias: %f, Local Tempering: %f, Global Tempering: %f, MinX_MaxY: %f\n",
      bias_potential[site], decay_local, decay_global, min_max);
    printf("dGdF: [");
    for(int i = prev_subs; i < prev_subs+blocksPerSite[site]; i++){
      printf(" %f,", dGdF[i]);
    }
    printf("] \n");
    real period = transition_w + plateau_w;
    printf("Sub Period: %f, Period: %f, T_bins: %d\n", period, site_period[site], T_bins[site]);
    real rel[blocksPerSite[site]];
    real dG = 0;
    real var[blocksPerSite[site]];
    real dG_var = 0;
    int count = 1;
    printf("dG i->i+1: [");
    for(int i = prev_bins; i < prev_bins+T_bins[site]-1; i++){
      dG += T_res*(ensemble_dUdT[i]+ensemble_dUdT[i+1])/2.0; 
      dG_var += T_res*T_res*dUdT_var[i];
      real T = T_res*(i-prev_bins+1); // dG at this point
      if(T >= count*period+.5*plateau_w){
        rel[count-1] = dG;
        var[count-1] = dG_var;
        printf(" %5.2f +- %5.2f (T=%4.2f),", dG, sqrt(dG_var), T);
        dG = 0;
        dG_var = 0;
        count++;
      }
    }
    rel[blocksPerSite[site]-1] = dG;
    var[blocksPerSite[site]-1] = dG_var;
    printf(" %5.2f +- %5.2f (T=%4.2f) ]\n", dG, sqrt(dG_var), (T_bins[site]-1.0)*T_res);
    printf("dG 0->i: [");
    dG = 0;
    dG_var = 0;
    for(int i = 0; i < blocksPerSite[site]; i++){
      dG += rel[i];
      dG_var += var[i];
      printf(" %5.2f +- %5.2f,", dG, sqrt(dG_var));
    }
    printf("]\n");
    printf("Samples: [");
    real samples = 0;
    count = 1;
    for(int i = 0; i < T_bins[site]-1; i++){
      samples += counts[i];
      real T = T_res*(i+1); 
      if(T > count*period){
        printf(" %f,", samples);
        samples = 0;
        count++;
      }
    } 
    printf(" %f ]\n", samples);
    printf("Effective Samples: [");
    samples = 0;
    count = 1;
    for(int i = 0; i < T_bins[site]-1; i++){
      samples += eff_counts[i];
      real T = T_res*(i+1); // dG at this point
      if(T > count*period){
        printf(" %f,", samples);
        samples = 0;
        count++;
      }
    } 
    printf(" %f ]\n\n", samples);
    if(LE){
      printf("Site R: %f, k_LE*f_red^R: %f\n", R[site], k_LE*pow(f_red, R[site]));
      printf("M: [");
      for(int i = prev_bins; i < prev_bins + T_bins[site]; i++){
        printf(" %f,", M[i]);
      }
      printf("] \n");
      printf("Visited Bins: %d, Sweep: [", visited_bins[site]);
      for(int i = prev_bins; i < prev_bins + T_bins[site]; i++){
        printf(" %d,", theta_sweep[i]);
      }
      printf("] \n");
    }
    prev_subs += blocksPerSite[site];
    prev_bins += T_bins[site];
  }
}

__global__ void calc_thetaForce_contributions(
  int blockCount,
  // Input
  real* dUdL_msld, real* dLdT,
  // Output
  real *dUdT_msld)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<blockCount) {
    real fi=dUdL_msld[i];
    fi *= dLdT[i];
    atomicAdd(&dUdT_msld[i],fi); // Each lambdas contribution to theta force
    // dU/dT = sum(dUdT_msld[i]) -> this is calculated in real dU/dL->dU/dT kernel & anywhere dU/dT is needed
  }
}

void Msld::oss_lambda_to_theta_force(System* system){
  calc_thetaForce_contributions<<<(blockCount+BLMS-1)/BLMS,BLMS,0,system->run->ossBias>>>(
      blockCount, 
      // Input
      dUdL_msld_d, dLdT_d,
      // Output
      dUdT_msld_d);
}

__global__ void add_sample_hist_kernel(
  real kT, int siteCount, int* blocksPerSite,
  int* T_bins, real* site_period,
  int dUdT_bins, real dUdT_max, real dUdT_min,
  real transition_w, real plateau_w, 
  bool standard_tempering,
  // Input
  bool pb_meta, bool temper,
  real* thetas, real* dUdT_msld, 
  real tempering, real temper_offset, real* bias_potential, real* oss_max_bias,
  real k_LE, real f_red,
  // Output - OSS
  real* histogram, real* theta_counts, 
  int* dUdT_sampled_max, int* dUdT_sampled_min,
  real* oss_min_max,
  // Output - LE
  real* R, int* visited_bins, 
  int* theta_sweep, real* M 
){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < siteCount && i != 0){ // none for environment
    int start_site = 0;
    int theta_start = 0;
    for(int j = 0; j < i; j++){
      start_site += blocksPerSite[j];
      theta_start += T_bins[j];
    }
    real T = thetas[start_site];
    real dUdT = 0;
    for(int j = 0; j < blocksPerSite[i]; j++){
      dUdT += dUdT_msld[start_site+j];
    }

    // Find min of max bias in each T
    real minimum = 1000;
    for(int j = theta_start; j < theta_start+T_bins[i]; j++){
      minimum = minimum > oss_max_bias[j] ? oss_max_bias[j] : minimum;
    }
    oss_min_max[0] = minimum; // only 1 element
    real bias = standard_tempering ? max(bias_potential[i]-temper_offset, 0.0) : max(minimum-temper_offset, 0.0);
    real decay = temper ? exp(-bias/(tempering*kT)) : 1.0;

    real pb_scale = 0;
    for(int j = 1; j < siteCount; j++){
      pb_scale += exp(-bias_potential[j]/kT);
    }
    pb_scale = pb_meta ? exp(-bias_potential[i]/kT)/pb_scale : 1.0;

    // Location of point on histogram
    int X = safe_histogram_index(T, T_bins[i], site_period[i], 0);
    int Y = safe_histogram_index(dUdT, dUdT_bins, dUdT_max, dUdT_min);
    //printf("i: %d, (T: %f, dUdT: %f), (X1: %d, Y1: %d), T_bins: %d\n", i, T, dUdT, X, Y, T_bins[i]);
    if(X >= 0 && X <= T_bins[i]-1 && Y >= 0 && Y <= dUdT_bins-1){
      theta_counts[theta_start + X] += 1.0;
      int hist_index = (theta_start+X)*dUdT_bins + Y;
      //printf("i: %d, Index: %d, theta_start: %d, X: %d, Y: %d\n", i, hist_index, theta_start, X, Y);
      histogram[hist_index] += decay*pb_scale;
      dUdT_sampled_max[theta_start+X] = Y > dUdT_sampled_max[theta_start+X] ? Y : dUdT_sampled_max[theta_start+X];
      dUdT_sampled_min[theta_start+X] = Y < dUdT_sampled_min[theta_start+X] ? Y : dUdT_sampled_min[theta_start+X];
    } 

    // Local Elevation - Order of these checks prevents weird double counting
    if(X >= 0 && X <= T_bins[i]-2){ // First and last bin are identical centers, don't include last, account for that in visited check
      if (visited_bins[i] >= 2*T_bins[i]-2) { // Double sweep complete, increment and clear mem
        R[i] += 1;
        for(int j = theta_start; j < theta_start + T_bins[i]; j++){
          theta_sweep[j] = 0;
        }
        visited_bins[i] = 0;
      }
      if (theta_sweep[theta_start + X] == 1 && visited_bins[i] >= T_bins[i]-1){ // New bin sample on second sweep 
        theta_sweep[theta_start + X] = 2;
        visited_bins[i] += 1;
      }
      if (theta_sweep[theta_start + X] == 0){ // New bin sample on first sweep
        theta_sweep[theta_start + X] = 1;
        visited_bins[i] += 1;
      }
      M[theta_start + X] += k_LE*pow(f_red, R[i]);
    }
  }
}

void Msld::add_sample(System* system, int step) { 
  cudaStream_t stream = 0;
  Run* r = system->run;
  State* s = system->state;
  int shMem = 0;

  if (system->run) {
    stream=system->run->ossBias;
  }

  // Step = 0 during NPT MC trials, don't add samples (maybe don't need to do this)
  if(step == 0 || !update_fe_surface){
    return;
  }

  update_fe_surface = step < update_steps; // last sample if false

  if(step % sample_freq == 0){
    real kT = system->state->leapParms1->kT; 
    add_sample_hist_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
      kT, siteCount, blocksPerSite_d,
      T_bins_d, site_period_d, 
      dUdT_bins, dUdT_max, dUdT_min,
      transition_w, plateau_w,
      standard_tempering,
      // Input
      pb_meta, temper,
      s->theta_fd, dUdT_msld_d, 
      temper_amount, temper_offset, bias_potential_d, oss_max_pot_d,
      k_LE, f_red,
      // Output
      oss_histogram_d, oss_theta_counts_d,
      oss_dUdT_max_d, oss_dUdT_min_d, oss_min_max_d, 
      R_d, visited_bins_d,
      theta_sweep_d, M_d);

    // updates potential with new sample (only where affected) & recalculates <dU/dT>
    update_ABF_from_hist(system, 2*T_search+1, 2*dUdT_search+1, true);
  }

  if(step % oss_write_freq == 0){ // write files every 1k samples (1k-10k steps or 1-20ps)
    //printf("Writing hist_potential.txt!\n");
    write_histogram_file(system, "hist_potential.txt", true);
    //printf("Writing hist_restart.txt!\n");
    write_histogram_file(system, "hist_restart.txt", false);
  }
}

__global__ void getforce_abf_kernel(
  int siteCount, int* blocksPerSite, 
  int* T_bins, real* site_period, 
  // Inputs
  bool linear_shift, real oss_k, 
  real* thetas, real* ensemble_dUdT, real* dUdT_msld, 
  // Outputs
  real_f* thetaForce, real* dGdF 
){
  int i=blockIdx.x*blockDim.x+threadIdx.x; // site
  if(i < siteCount && i != 0){
    // Count to first sub index
    int start = 0;
    int start_bin = 0;
    int prev_bins = 0;
    for(int j = 0; j < i; j++){ 
      start += blocksPerSite[j];
      prev_bins += T_bins[j];
    }
    real theta = thetas[start];
    int bin = safe_histogram_index(theta, T_bins[i], site_period[i], 0);
    // Lerp implementation of ABF along theta
    real res = site_period[i]/(T_bins[i]-1.0); // also the width
    real bin_center = bin*res;
    int hist_bin = prev_bins + bin;
    real dUdT = ensemble_dUdT[hist_bin];
    real dist = theta-bin_center;
    real partner_center, partner_dUdT, interp;
    real dUdT_abf = 0.0;
    int partner_id;
    if(dist > 0){ // lambda in upper half of bin -> never true for bin=num_bins-1
      partner_center = bin_center + res;
      partner_dUdT = ensemble_dUdT[hist_bin+1];
      partner_id = hist_bin+1;
      interp = dist / res;
      dUdT_abf = (1 - interp)*dUdT + interp*partner_dUdT;
    } else { // lambda in lower half of bin
      partner_center = bin_center - res;
      partner_dUdT = ensemble_dUdT[hist_bin-1];
      partner_id = hist_bin-1;
      interp = (theta - partner_center) / res;
      dUdT_abf = (1 - interp)*partner_dUdT + interp*dUdT;
    }
    // add -'ve
    dUdT_abf = -dUdT_abf;
    atomicAdd(&thetaForce[start], dUdT_abf);
    //printf("i: %d, T: %f, <dU/dT>: %f\n", i, theta, dUdT_abf);
    // This ABF is not part of the energy conservation
    if(linear_shift){
      real TI = 0;
      for(int j = prev_bins; j < prev_bins+T_bins[i]-1; j++){
        TI += res*(ensemble_dUdT[j] + ensemble_dUdT[j+1])/2.0;
      }
      real slope = TI/site_period[i];
      real bias = slope*theta;
      atomicAdd(&thetaForce[start], slope);
    }

    // U_bias = oss_k*F_\theta -> encourages sampling of configurations with lower dU/dT values
    for(int j = start; j < start+blocksPerSite[i]; j++){
      atomicAdd(&dGdF[j], oss_k);
    }
  }
}

__global__ void getforce_hist_kernel(
  int siteCount, int* blocksPerSite, 
  int* T_bins, real* site_period, 
  int dUdT_bins, real dUdT_max, real dUdT_min,
  real T_std, real dUdT_std,
  int T_search, int dUdT_search,
  real gauss_weight, 
  bool relative_indexing, // center on current (T, dUdT) or (0, 0)
  int vert_slices, int horz_slices,
  // Inputs
  bool energy, 
  real* thetas, real* dUdT_msld, real* histogram,
  // Outputs
  real* dGdT, real* dGdF, real* bias_potential,
  real* potential_grid 
) {
  // Potential eval at this grid point
  int iSite = blockIdx.x + 1;  // Skip site 0
  int iT = blockIdx.y;         
  int idUdT = blockIdx.z;      
  // Eval potential due to this relative grid point
  int iSearch_T = threadIdx.x;    // T search offset
  int iSearch_dUdT = threadIdx.y; // dUdT search offset
  if (iSite >= siteCount) return;
  if (energy && (iT >= vert_slices || idUdT >= horz_slices)) return;
  
  int start_site = 0;
  int theta_start = 0;
  for(int j = 0; j < iSite; j++){
    start_site += blocksPerSite[j];
    theta_start += T_bins[j];
  }
  // Determine central evaluation point (T, dUdT)
  real T = thetas[start_site];
  real dUdT = 0;
  for(int j = 0; j < blocksPerSite[iSite]; j++){
    dUdT += dUdT_msld[start_site+j];
  }
  if(!relative_indexing){
    // relative_indexing doesn't really matter in X direction since it will get wrapped
    T = 0;
    dUdT = 0;
  }
  real T_resolution = (site_period[iSite])/(T_bins[iSite]-1.0);
  real dUdT_resolution = (dUdT_max-dUdT_min)/(dUdT_bins-1.0);
  if (energy) { // Re-center to evaluate potential at other grid points
    int T_index = safe_histogram_index(T, T_bins[iSite], site_period[iSite], 0);
    T_index += -floor(vert_slices/2.0) + iT;
    T = T_index * T_resolution; 
    if (T < 0) { T += site_period[iSite]; } // T periodic
    if (T > site_period[iSite]) { T -= site_period[iSite]; } // would make this inclusive, but theres a bin exactly on site_period
    // Corresponding dUdT from idUdT
    int dUdT_index = safe_histogram_index(dUdT, dUdT_bins, dUdT_max, dUdT_min);
    dUdT_index += -floor(horz_slices/2.0) + idUdT;
    dUdT = dUdT_min + dUdT_index*dUdT_resolution;
  }
  
  // Point on grid close to (T, dUdT) where we evalutate potential
  int X = safe_histogram_index(T, T_bins[iSite], site_period[iSite], 0);
  int Y = safe_histogram_index(dUdT, dUdT_bins, dUdT_max, dUdT_min);
  if(energy && iSearch_T == 0 && iSearch_dUdT == 0){ // (0,0) block thread sets potential at block origin to zero 
    int output_index = (theta_start + X)*dUdT_bins + Y;
    potential_grid[output_index] = 0;
  }
  __syncthreads();
  int j = X - T_search + iSearch_T;
  int true_j = j; // used to correctly calculate distances
  if (j < 0) { j += T_bins[iSite]; }  // wrap j 
  if (j >= T_bins[iSite]) { j -= T_bins[iSite]; } 
  int k = Y - dUdT_search + iSearch_dUdT;
  if (j >= 0 && j < T_bins[iSite] && k >= 0 && k < dUdT_bins) {
    int T_index = j + theta_start; // shift into correct histogram for site
    int dUdL_index = k;
    
    // gaussians from (T_c, dUdT_c) evaluated at (T, dUdT)
    real T_center = true_j * T_resolution; 
    real T_distance = (T - T_center) / T_std;
    real T_gaussian = expf(-0.5*T_distance*T_distance);
    
    real dUdT_center = dUdT_min + k*dUdT_resolution;
    real dUdT_distance = (dUdT - dUdT_center) / dUdT_std;
    real dUdT_gaussian = expf(-0.5*dUdT_distance*dUdT_distance);
    
    int hist_index = T_index*dUdT_bins + dUdL_index;
    real weight = gauss_weight * histogram[hist_index];
    real tmp_bias = weight * T_gaussian * dUdT_gaussian;
    if (!energy) {
      real local_dUdT_force = -dUdT_distance/dUdT_std * tmp_bias;
      real local_T_force = -T_distance/T_std * tmp_bias;
      for(int l = 0; l < blocksPerSite[iSite]; l++){
        atomicAdd(&dGdF[start_site + l], local_dUdT_force);
      }
      atomicAdd(&dGdT[start_site], local_T_force);
      atomicAdd(&bias_potential[iSite], tmp_bias);
      //printf("tmp_bias: %f, (%d, %d, %d), (%d (%d), %d) = (%f, %f) -> (%d, %d) = (%f, %f)\n", 
      //  tmp_bias, iSite, iT, idUdT, j, true_j, k, T_center, dUdT_center, X, Y, T, dUdT);
    } else {
      int output_index = (theta_start + X)*dUdT_bins + Y;
      atomicAdd(&potential_grid[output_index], tmp_bias);
      //printf("tmp_bias: %f, id: %d, T: %f, iT: %d, (%d, %d, %d), (%d (%d), %d) = (%f, %f) -> (%d, %d) = (%f, %f)\n", 
      //  tmp_bias, output_index, thetas[start_site], T_index, iSite, iT, idUdT, j, true_j, k, T_center, dUdT_center, X, Y, T, dUdT);
    }
  }
}

__global__ void getforce_pb_meta_kernel(
  int siteCount, real kT, 
  int* blocksPerSite, int* siteBound,
  // Inputs
  bool pb_meta, 
  real* bias_potential, real* dGdT, real* dGdF,
  // Outputs
  real* thetaForce, real* total_bias, real_e* energy // dGdF is also an output
){
  int i=blockIdx.x*blockDim.x+threadIdx.x; // site
  real lEnergy = 0;
  extern __shared__ real sEnergy[];
  if(i < siteCount && i != 0){ // PB meta across site histograms
    int ji = siteBound[i];
    int jf = ji + blocksPerSite[i];
    if(pb_meta){
      real sum = 0;
      for(int j = 1; j < siteCount; j++){
        sum += exp(-bias_potential[j]/kT);
      }
      real pb_chain = exp(-bias_potential[i]/kT) / sum;
      lEnergy = -kT*log(sum)/(siteCount-1.0);
      atomicAdd(total_bias, lEnergy);
      atomicAdd(&thetaForce[ji], dGdT[ji]*pb_chain);
      for (int j = ji; j < jf; j++){
        dGdF[j] *= pb_chain;
      }
    } else { // simple sum of histograms
      lEnergy = bias_potential[i];
      atomicAdd(total_bias, lEnergy);
      atomicAdd(&thetaForce[ji], dGdT[ji]);
    }
  }

  // Energy, if requested
  if (energy) {
    __syncthreads();
    //real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

__global__ void getforce_LE_kernel(
  int siteCount, int* blocksPerSite, 
  int* T_bins, real* site_period,
  // Inputs
  real* thetas, real* M,
  // Outputs,
  real* thetaForce, real_e* energy){
    int i=blockIdx.x*blockDim.x+threadIdx.x; // site
  if(i < siteCount && i != 0){
    // Count to first sub index
    int start = 0;
    int start_bin = 0;
    int prev_bins = 0;
    for(int j = 0; j < i; j++){ 
      start += blocksPerSite[j];
      prev_bins += T_bins[j];
    }
    real theta = thetas[start];
    int center_bin = safe_histogram_index(theta, T_bins[i], site_period[i], 0); // last bin is zero
    // Local dot product of cubic b splines centered at grid points evaluated at T with M of b spline grid point
    real res = site_period[i]/(T_bins[i]-1.0); // also the width
    real bin_center = center_bin*res;
    real bias = 0;
    real dbdt = 0;
    for(int j = center_bin - 4; j <= center_bin + 4; j++){ // extra bins to catch last bin and first bin being same
      real bin_center = j*res; 
      real correction = 0; // apply correction so that first and last bin are calculated at same point
      if(j < 0){ correction += res; }
      if(j >= T_bins[i]){ correction -= res; }
      real dist = theta - (bin_center+correction); 
      dist /= res; // normalize so deltas are 1
      // Compute cubic b-spline
      real dabs_dt = dist > 0 ? 1 : -1; // derivative at zero is zero anyway, so def of this doesn't matter
      real t = abs(dist);
      real b_spline = 0;
      real db_spline = 0;
      if(t >= 1.0 && t <= 2.0){
        real tmp = 2.0-t;
        b_spline = tmp*tmp*tmp/4.0; 
        db_spline = 3.0*tmp*tmp/4.0*-1.0*dabs_dt;
      } else if (t < 1.0){
        b_spline = (4.0+3.0*t*t*t-6.0*t*t)/4.0;
        db_spline = (9.0*t*t-12.0*t)/4.0*dabs_dt;
      }
      // Wrap with period
      int bin = j;
      if(j < 0){ bin += T_bins[i]; }
      if(j >= T_bins[i]){ bin -= T_bins[i]; }
      int hist_bin = prev_bins + bin; 
      bias += M[hist_bin]*b_spline;
      dbdt += M[hist_bin]*db_spline;
      //printf("bin_center: %f, dist: %f, t: %f, M[%d]: %f, b_sp: %f, db_sp: %f\n", bin_center, dist, t, hist_bin, M[hist_bin], b_spline, db_spline);
    }
    //printf("T: %f, Bias: %f, dBdT: %f, start: %d, bins: %d, period: %f\n", theta, bias, dbdt, start, T_bins[i], site_period[i]);
    if(energy) atomicAdd(energy, bias);
    atomicAdd(&thetaForce[start], dbdt);
  }
}

void Msld::getforce_bias(System *system, bool calcEnergy) {
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

  if(oss_restart_success){ // evaluate potential everywhere to get ABF
    oss_restart_success=false;
    update_ABF_from_hist(system, total_T_bins, dUdT_bins, false);
  }

  if(!update_fe_surface) { linear_shift=false; } // never want non-equilibrium bias in when not updating

  // 1D Theta Flattening
  if(LE){
    getforce_LE_kernel<<<(siteCount+BLMS-1)/BLMS, BLMS,shMem,stream>>>(
      siteCount, blocksPerSite_d,
      T_bins_d, site_period_d,
      // Inputs
      s->theta_fd, M_d, 
      // Outputs
      s->thetaForce_d, pEnergy);
  } else { // If OSS is only option set
    // Compute <dU/dT> from array grid points via lerp & subtract from dU/dT
    getforce_abf_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
      siteCount, blocksPerSite_d, 
      T_bins_d, site_period_d, 
      // Inputs
      linear_shift, oss_k,
      s->theta_fd, oss_ensemble_dUdT_d, dUdT_msld_d, 
      // Outputs
      s->thetaForce_d, dGdF_d);
  }

  // 2D (T, dU/dT) flattening
  if(oss && bias_mag > 1e-5){ // don't do 2D if bias mag is off (ABF only in that situation)
    // Compute dGdT & dGdF from gauss sum on histogram
    int vert_slices = 1;
    int horz_slices = 1;
    dim3 blockDim(2*T_search+1, 2*dUdT_search+1, 1); // both search params should be 10
    dim3 gridDim(siteCount-1, 1, 1);
    getforce_hist_kernel<<<gridDim, blockDim, shMem, stream>>>(
      siteCount, blocksPerSite_d,
      T_bins_d, site_period_d, 
      dUdT_bins, dUdT_max, dUdT_min, 
      T_std, dUdT_std, T_search, dUdT_search, 
      bias_mag, 
      true,
      vert_slices, horz_slices,
      // Inputs
      false,
      s->theta_fd, dUdT_msld_d, oss_histogram_d,
      // Outputs
      dGdT_d, dGdF_d, bias_potential_d,
      oss_potential_d);
  
    getforce_pb_meta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
      siteCount, system->state->leapParms1->kT,
      blocksPerSite_d, siteBound_d, 
      // Inputs
      pb_meta,
      bias_potential_d, dGdT_d, dGdF_d,
      // Outputs
      s->thetaForce_d, total_bias_d, pEnergy);
  }
}

__global__ void hist_ensemble_dUdT_kernel(
    real kT, int nSite, real* thetas, int* siteBound,
    int* T_bins, real* site_periods,
    int dUdT_bins, real dUdT_max, real dUdT_min,
    int* dUdT_sampled_max, int* dUdT_sampled_min,
    int slice_count, real warmup_samples, real bias_mag,
    // Inputs
    real* potential_grid, real* histogram, 
    // Output
    real* oss_ensemble_dUdT, 
    real* oss_dUdT_var, real* oss_eff_n,
    real* oss_max_potential
) {
  int site = threadIdx.x+1;
  int tid = blockIdx.x;
  if (site < nSite && tid < slice_count) { // each site
    real dUdT_res = (dUdT_max - dUdT_min) / (dUdT_bins - 1.0);
    real offset = 0.0;
    real weighted_dUdT = 0;
    real weighted_dUdT2 = 0;
    real Z = 0;
    real Z2 = 0;
    real samples = 0;
    real max_bias = 0;
    int start_1D = 0;
    for(int j = 0; j < site; j++){
      start_1D += T_bins[j];
    }
    int X = safe_histogram_index(thetas[siteBound[site]], T_bins[site], site_periods[site], 0);
    int center_X = X;
    X += -floor(slice_count/2.0) + tid;
    if(X > T_bins[site]) { X -= T_bins[site]; }
    if(X < 0) { X += T_bins[site]; }
    int low = dUdT_sampled_min[start_1D + X];
    int high = dUdT_sampled_max[start_1D + X];
    if(high >= low && low >= 0 && high < dUdT_bins && X >= 0 && X < T_bins[site]){ // if no samples this isn't true
      for (int y = low; y <= high; y++) {
        real dUdT = dUdT_min + y*dUdT_res;
        int grid_index = (start_1D+X)*dUdT_bins + y;
        real current_bias = potential_grid[grid_index]/kT;
        max_bias = current_bias > max_bias ? current_bias : max_bias;
        if(histogram[grid_index] < 1e-5){ continue; } // Don't include non-sampled regions
        samples += histogram[grid_index];
        if(bias_mag > 1e-5){ // Boltzman weighting
          if (current_bias > offset) {
            real correction = exp(offset - current_bias); 
            weighted_dUdT *= correction;
            weighted_dUdT2 *= correction;
            Z *= correction;
            Z2 *= correction*correction;
            offset = current_bias;
          }
          weighted_dUdT += dUdT * exp(current_bias - offset);
          weighted_dUdT2 += dUdT*dUdT*exp(current_bias - offset);
          Z += exp(current_bias - offset);
          Z2 += exp(2*(current_bias - offset)); // Z^2
        } else { // Uniform weighting
          weighted_dUdT += dUdT * histogram[grid_index];
          weighted_dUdT2 += dUdT*dUdT * histogram[grid_index];
          Z += histogram[grid_index];
          Z2 += histogram[grid_index]*histogram[grid_index];
        }
      }
      oss_ensemble_dUdT[start_1D + X] = Z > 1e-5 ? weighted_dUdT / Z : 0.0;
      oss_dUdT_var[start_1D + X] = Z > 1e-5 ? weighted_dUdT2 / Z : 0.0;
      oss_eff_n[start_1D + X] = Z2 > 1e-5 ? Z*Z / Z2 : 0.0;
      if(samples < warmup_samples){ // tempering only changes this slightly
        oss_ensemble_dUdT[start_1D+X] *= samples/warmup_samples;
      }
      oss_max_potential[start_1D+X] = max_bias*kT; // kT will get multiplied back later
      if(abs(X - center_X) < 5 && false){
        printf("site: %d, X: %d, T: %f, <dU/dT>: %f, low: %d, high: %d, max_bias: %f\n", 
          site, X, thetas[siteBound[site]], oss_ensemble_dUdT[start_1D+X], low, high, max_bias);
      }
    } 
  }
}

// Compute potential and <dU/dT> from histogram for ABF for each site
void Msld::update_ABF_from_hist(System *system, int vert_slices, int horz_slices, bool relative_indexing){
  cudaStream_t stream = 0;
  Run *r = system->run;
  State *s = system->state;
  real_e *pEnergy = NULL;
  int shMem = 0;
  if (system->run) {
    stream=system->run->ossBias;
  }
  // Update potential grid with new sample
  dim3 blockDim(2*T_search+1, 2*dUdT_search+1, 1); // both search params should be 10
  dim3 gridDim(siteCount-1, vert_slices, horz_slices); // (iSite, iT, idUdT)
  getforce_hist_kernel<<<gridDim, blockDim, shMem, stream>>>(
    siteCount, blocksPerSite_d,
    T_bins_d, site_period_d, 
    dUdT_bins, dUdT_max, dUdT_min, 
    T_std, dUdT_std, T_search, dUdT_search, 
    bias_mag, relative_indexing, vert_slices, horz_slices,
    // Inputs
    true,
    s->theta_fd, dUdT_msld_d, oss_histogram_d,
    // Outputs
    s->thetaForce_d, dGdF_d, bias_potential_d,
    oss_potential_d);

  // Compute <dU/dT> in region where potential got updated
  dim3 blockDim1(siteCount-1, 1, 1); 
  dim3 gridDim1(vert_slices, 1, 1);
  // relative_indexing doesn't really matter in X direction since it will get wrapped
  hist_ensemble_dUdT_kernel<<<gridDim1, blockDim1, 0, stream>>>(
    s->leapParms1->kT, siteCount, s->theta_fd, siteBound_d,
    T_bins_d, site_period_d, 
    dUdT_bins, dUdT_max, dUdT_min, 
    oss_dUdT_max_d, oss_dUdT_min_d, 
    vert_slices,
    warmup_samples, bias_mag,
    // Inputs
    oss_potential_d, oss_histogram_d, 
    // Outputs
    oss_ensemble_dUdT_d, oss_dUdT_var_d, oss_eff_n_d,
    oss_max_pot_d);
}

// fills dF[0] = L(T), dF[1] = dLdT, dF[1] = d2LdT2
__device__ void leus_f(real_x theta, real transition_w, real plateau_w, real_x* dF, leus_func func_type, real xsin_n){
  // Transition regions have inclusive bounds, plateaus have exclusive bounds (if plateau = 0, never hit that region)
  bool forward = theta >= 0.0 && theta <= transition_w;
  bool backward = theta >= transition_w + plateau_w && theta <= 2*transition_w + plateau_w; 
  if (forward || backward){ // forward region 0->j
    theta = backward ? 2*transition_w + plateau_w - theta : theta;
    theta /= transition_w;
    real_x dTdT = backward ? -1.0 : 1.0; // d(2*W + w - T)/dT : d(T)/dT -> aren't functions of t, don't effect second derivative
    dTdT *= 1 / transition_w;
    if(func_type == leus_linear){
      dF[0] = theta;
      dF[1] = dTdT;
      dF[2] = 0;
    } else if (func_type == leus_cubic){ 
      dF[0] = 3.0*pow(theta, 2.0) - 2.0*pow(theta,3.0);
      dF[1] = (6.0*theta - 6.0*pow(theta,2.0))*dTdT;
      dF[2] = (6.0 - 12.0*theta)*dTdT*dTdT;
    } else if (func_type == leus_quintic){
      dF[0] = 6.0*pow(theta, 5.0) - 15.0*pow(theta, 4.0) + 10.0*pow(theta, 3.0);
      dF[1] = (30.0*pow(theta, 4.0) - 60.0*pow(theta, 3.0) + 30.0*pow(theta, 2.0))*dTdT;
      dF[2] = (120.0*pow(theta, 3.0) - 180.0*pow(theta, 2.0) + 60.0*theta)*dTdT*dTdT;
    } else if (func_type == leus_sin2){
      real_x sinT = sin((M_PI/2.0)*theta);
      real_x cosT = cos((M_PI/2.0)*theta);
      dF[0] = sinT*sinT;
      dF[1] = M_PI*sinT*cosT*dTdT;
      dF[2] = .5*pow(M_PI*dTdT, 2.0)*(cosT*cosT - sinT*sinT);
    } else if (func_type == leus_xsin){
      real arg = 2*M_PI*xsin_n;
      real_x sinT = sin(arg*theta);
      real_x cosT = cos(arg*theta);
      dF[0] = theta - (sinT/arg);
      dF[1] = dTdT - cosT*dTdT;
      dF[2] = sinT*arg*dTdT*dTdT;
    }
  } else if (theta > transition_w && theta < transition_w + plateau_w){ // j physical state
    dF[0] = 1;
    dF[1] = 0;
    dF[2] = 0;
  } else { // 0 sub physical state
    dF[0] = 0;
    dF[1] = 0;
    dF[2] = 0;
  }
}

__global__ void calc_lambda_from_theta_kernel(
  real_x *lambda,real_x *theta,
  int siteCount,int *siteBound,
  real fnex, 
  bool L_LEUS, leus_func func_type, real xsin_n,
  real plateau_w, real transition_w, 
  real* dLdT, real* d2LdT2){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  int j,ji,jf;
  real_x lLambda;
  real_x norm=0;

  if (i<siteCount) {
    ji=siteBound[i];
    jf=siteBound[i+1];
    if (L_LEUS){ // uses first theta of site, sets all other thetas to zero
      real subs = jf - ji;
      // Wrap theta to [0,Ns*(W+w)), step should never move theta so significantly to require a loop
      real period = subs*(transition_w+plateau_w); // 1 transition & plateau per sub
      if (theta[ji] < 0){
        theta[ji] += period;
      } 
      if (theta[ji] >= period){
        theta[ji] -= period;
      }
      real_x T = theta[ji];
      // Apply LEUS function to site theta
      real_x sum = 0;
      real_x dsum = 0;
      real_x d2sum = 0;
      for(j = 1; j < subs; j++){
        real_x dF[3];
        // Shift function into place
        real_x shift = -(j-1)*(transition_w + plateau_w) - plateau_w;
        leus_f(T + shift, transition_w, plateau_w, dF, func_type, xsin_n);
        lambda[ji+j] = dF[0];
        dLdT[ji+j] = dF[1];
        d2LdT2[ji+j] = dF[2];
        sum += dF[0];
        dsum += dF[1];
        d2sum += dF[2];
        //printf("j: %d, T: %f, L: %f, dLdT: %f, d2LdT2: %f\n", j, T, dF[0], dF[1], dF[2]);
        theta[ji+j] = 0; 
      }
      // f0(theta) = 1 - sum(fi(theta))
      lambda[ji] = abs(1.0 - sum); // numerically unstable if not forced positive
      dLdT[ji] = -dsum;
      d2LdT2[ji] = -d2sum;
      if(i == 0){theta[ji] = 0;} // environment
      /*
      if(i == 0){return;}
      printf("Lambdas: [ ");
      for(int j = ji; j < jf; j++){
        printf(" %f, ", lambda[j]);
      }
      printf("] \n");
      */
    } else { // soft-max like constraint
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
    calc_lambda_from_theta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,0,stream>>>(
      s->lambda_d,s->theta_d,siteCount,siteBound_d,fnex,
      L_LEUS, L_LEUS_function, xsin_n,
      plateau_w, transition_w,
      dLdT_d, d2LdT2_d);
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
    calc_lambda_from_theta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,0,stream>>>(
      s->lambda_d,s->theta_d,siteCount,siteBound_d,fnex,
      L_LEUS, L_LEUS_function, xsin_n, 
      plateau_w, transition_w,
      dLdT_d, d2LdT2_d);
  } else {
    cudaMemcpy(s->lambda_d,s->theta_d,s->lambdaCount*sizeof(real_x),cudaMemcpyDeviceToDevice);
  }
}

__global__ void calc_thetaForce_from_lambdaForce_kernel(
  real *lambda,real *theta,
  real_f *lambdaForce,real_f *thetaForce,
  int blockCount,int *lambdaSite,int *siteBound,
  real fnex, 
  bool L_LEUS, real* dLdT)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j, ji, jf;
  real li, fi;

  if (i<blockCount) {
    li=lambda[i];
    fi=lambdaForce[i];
    ji=siteBound[lambdaSite[i]];
    jf=siteBound[lambdaSite[i]+1];
    if(L_LEUS){ // single degree of freedom per site constraint
      // U = U_e + \sum_{N_sub_i} U_i(li)
      // dU/dTi = pU/pTi + sum_{N_sub_i}(pU/pLi * pLi/pT)
      //printf("i: %d, ti: %f, li: %f, dUdL: %f, dLdT[i]: %f, li*dLdT[i]: %f, dUdT: %f\n", 
      //  i, theta[i], li, fi, dLdT[i], fi*dLdT[i], fi*dLdT[i] + thetaForce[i]);
      fi *= dLdT[i];
      atomicAdd(&thetaForce[ji],fi); // adding into first theta of this site
      if(i != ji){ // accumulate forces added to other positions in thetaForce array
        atomicAdd(&thetaForce[ji], thetaForce[i]);
        thetaForce[i] = 0; // once it is added, clear theta force
      }
      __syncthreads();
      //if(i == ji){ printf("dUdT[%d]: %f\n", ji, thetaForce[ji]);}
    } else {
      for (j=ji; j<jf; j++) {
        fi+=-lambda[j]*lambdaForce[j];
      }
      fi*=li*fnex*cosf(ANGSTROM*theta[i])*ANGSTROM;
      atomicAdd(&thetaForce[i],fi);
    } 
  }
}

void Msld::calc_thetaForce_from_lambdaForce(cudaStream_t stream,System *system)
{
  State *s=system->state;
  if (!fix) { // ffix
    calc_thetaForce_from_lambdaForce_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,0,stream>>>(
      s->lambda_fd,s->theta_fd,
      s->lambdaForce_d,s->thetaForce_d,
      blockCount,lambdaSite_d,siteBound_d,
      fnex,
      L_LEUS, dLdT_d);
  }
}

__global__ void getforce_fixedBias_kernel(real *lambda,real *lambdaBias,real_f *lambdaForce, real_e *energy,int blockCount)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  if (i<blockCount) {
    atomicAdd(&lambdaForce[i],lambdaBias[i]);
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

  getforce_fixedBias_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(s->lambda_fd,lambdaBias_d,s->lambdaForce_d, pEnergy,blockCount);
}

__global__ void getforce_variableBias_kernel(real *lambda,real_f *lambdaForce, real_e *energy,int variableBiasCount,struct VariableBias *variableBias)
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
    if (fj) { 
      atomicAdd(&lambdaForce[vb.j],fj); 
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
    getforce_variableBias_kernel<<<(variableBiasCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(s->lambda_fd,s->lambdaForce_d,pEnergy,variableBiasCount,variableBias_d);
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
