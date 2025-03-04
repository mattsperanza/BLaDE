#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>

#include "msld/msld.h"

#include <cfloat>
#include <math.h>
#include <math.h>

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

  meta_histogram_d=NULL;
  meta_index_d=NULL;
  oss_histogram_d=NULL;
  oss_index_d=NULL;

  abf_index_d=NULL;
  abf_histogram_d=NULL;
  ensemble_dUdL_d=NULL;
  ensemble_dUdL2_d=NULL;
  weights_d=NULL;
  weighted_dUdL_d=NULL;
  weighted_dUdL2_d=NULL;
  offsets_d=NULL;
  average_dUdL_d=NULL;
  ensemble_var_d=NULL;


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
// NYI - charge restraints, put Q in initialize
  } else if (strcmp(token,"print")==0) {
    system->selections->dump();
  } else if (strcmp(token, "oss") == 0) {
    system->msld->oss=io_nextb(line);
  } else if (strcmp(token, "oss_abf") == 0){
    system->msld->oss_abf=io_nextb(line);
    if (system->msld->oss_abf) {
      system->msld->abf = true;
      if (!system->msld->oss) {
        printf("Need to set oss flag before setting oss_abf!\n");
        exit(1);
      }
    }
  } else if (strcmp(token, "bias_mag") == 0) {
    system->msld->gaussian_weight=io_nextf(line);
  } else if (strcmp(token, "L_std") == 0) {
    system->msld->L_std=io_nextf(line);
    system->msld->L_search = 4.0*(system->msld->L_std/system->msld->L_resolution); // ~4 L std in each direction
  } else if (strcmp(token, "mir_Lmin") == 0){
    system->msld->mirror_Lmin=io_nextb(line);
  } else if (strcmp(token, "mir_Lmax") == 0){
    system->msld->mirror_Lmax=io_nextb(line);
  } else if (strcmp(token, "dUdL_std") == 0){
    system->msld->dUdL_std=io_nextf(line);
    system->msld->dUdL_search = 4.0*(system->msld->dUdL_std/system->msld->dUdL_resolution); // ~4 dUdL std in each direction
  } else if (strcmp(token, "temper") == 0) {
    system->msld->temper=io_nextb(line);
  } else if (strcmp(token, "temper_amount") == 0) {
    system->msld->tempering=io_nextf(line);
  } else if (strcmp(token, "min_bias") == 0){
    system->msld->temper_min=io_nextf(line);
  } else if (strcmp(token, "umbrella_abf") == 0) {
    system->msld->abf=io_nextb(line);
    if (system->msld->abf) {
      system->msld->oss_abf = false;
    }
  } else if (strcmp(token, "tracking_only") == 0){
    system->msld->tracking_only=io_nextb(line);
    if (system->msld->tracking_only) {
      system->msld->abf = true;
    }
  } else if (strcmp(token, "meta") == 0){
    system->msld->meta=io_nextb(line);
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


  // Thermodynamic Integration/Metadynamics/OST variables
  cudaMalloc(&dU_msld_d, blockCount*sizeof(real)); // lambda force prior to biasing
  cudaMalloc(&step_force_d, (blockCount-1)*sizeof(real));
  cudaMalloc(&hist_potential_d, (blockCount-1)*sizeof(real)); // use blockCount so that it is indexed same as lambda array
  cudaMemset(hist_potential_d, 0, (blockCount-1)*sizeof(real));
  if ((abf || tracking_only || oss_abf) && !abf_histogram_d) {
    init_abf(system);
  }
  if (meta && !meta_histogram_d) {
    init_meta(system);
  }
  if (oss && !oss_histogram_d) {
    init_oss(system);
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

void Msld::init_meta(System* system){
  int nL = blockCount-1;
  int index[nL];
  int n_edges = L_meta_bins;
  for (int i = 0; i < nL; i++) {
    index[i] = i*n_edges;
  }
  cudaMalloc(&meta_index_d, nL*sizeof(int));
  cudaMemcpy(meta_index_d, index, nL*sizeof(int), cudaMemcpyDefault);
  cudaMalloc(&meta_histogram_d, nL*n_edges*sizeof(real));
  cudaMemset(meta_histogram_d, 0, nL*n_edges*sizeof(real));
}

void __global__ add_sample_meta_kernel(
  real kT, int nL, real* lambdas,
  real weight, real tempering,
  real* histogram, int* hist_indices, real* hist_potential,
  int L_bins, real L_max, real L_min, bool temper) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i < nL) {
    // Get location in histogram (L, dUdL) => (X, Y) starting from lower left corner of hist
    int X =  get_histogram_index(lambdas[i+1], L_bins, L_max, L_min);
    int index = hist_indices[i] + X;
    real sum = 0.0;
    for (int j = 0; j < nL; j++) {
      sum += exp(-hist_potential[j] / kT);
    }
    real factorDecouple = exp(-hist_potential[i] / kT) / sum;
    real factorTemper = temper ? exp(-hist_potential[i] / (tempering*kT)) : 1.0;
    if (X >= 0 && X < L_bins) {
      histogram[index] += weight * factorDecouple * factorTemper;
    }
  }
}

// Only gets called if meta=true
void Msld::add_sample_meta(System *system) {
  cudaStream_t stream = 0;
  Run *r = system->run;
  State *s = system->state;
  int shMem = 0;
  if (system->run) {
    stream=system->run->metaBias;
  }

  add_sample_meta_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    s->leapParms1->kT, blockCount-1, s->lambda_fd,
    gaussian_weight, tempering,
    meta_histogram_d, meta_index_d, hist_potential_d,
    L_oss_bins, L_max, L_min, temper);
}

void __global__ getforce_meta_kernel(
  int nL, real* lambdas, real* lambdaForce,
  real* hist_potential, real* step_force,
  int* hist_indices, real* histogram,
  int L_bins, real L_max, real L_min,
  real L_std, int L_search, real_e* energy,
  bool mirror_Lmin, bool mirror_Lmax) {
  extern __shared__ real sEnergy[];
  real lEnergy = 0.0;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<nL) {
    real bias = 0.0;
    real L_force = 0.0;
    int X = get_histogram_index(lambdas[i+1], L_bins, L_max, L_min);
    for (int j = X-L_search; j <= X+L_search; j++) {
      int L_index = j;
      real mirrorFactor = 1.0;
      if (mirror_Lmin) {
        L_index = (L_index < 0) ? -L_index : L_index;
        mirrorFactor = L_index == 0 ? 2.0 : 1.0;
      }
      if (mirror_Lmax) {
        L_index = (L_index > L_bins-1) ? L_index - 2*(L_index-(L_bins-1)) : L_index;
        mirrorFactor = L_index == L_bins-1 ? 2.0 : 1.0;
      }
      // Still check it is in bounds
      if (L_index < 0 && L_index >= L_bins) { continue; }
      real L_resolution = (L_max-L_min)/(L_bins-1.0);
      real L_center = L_min + j*L_resolution;
      real L_distance = (lambdas[i+1] - L_center) / L_std;
      real L_gaussian = exp(-.5*L_distance*L_distance);
      int index = hist_indices[i] + j;
      real weight = mirrorFactor * histogram[index];
      real tmp_bias = weight * L_gaussian;
      bias += tmp_bias;
      L_force += -L_distance/L_std * tmp_bias;
    }
    atomicAdd(&lambdaForce[i+1], L_force);
    atomicAdd(&step_force[i], L_force);
    if (energy) {
      atomicAdd(&hist_potential[i], bias);
      lEnergy = bias;
    }
  }

  if (energy) {
    __syncthreads();
    real_sum_reduce(lEnergy, sEnergy, energy);
  }
}

void Msld::get_force_meta(System *system, bool calcEnergy) {
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
    stream=system->run->metaBias;
  }

  getforce_meta_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    blockCount-1, s->lambda_fd, s->lambdaForce_d,
    hist_potential_d, step_force_d,
    meta_index_d, meta_histogram_d, L_meta_bins, L_max, L_min, L_std, L_search,
    pEnergy, mirror_Lmin, mirror_Lmax);
}


void Msld::init_abf(System* system){
  int nL = blockCount-1; // 0th lambda is environment
  int index[nL];
  int n_edges = L_abf_bins;
  for (int i = 0; i < nL; i++) {
    index[i] = i*n_edges;
  }
  cudaMalloc(&abf_index_d, nL*sizeof(int));
  cudaMemcpy(abf_index_d, index, nL*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&abf_histogram_d, nL*n_edges*sizeof(real));
  cudaMemset(abf_histogram_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&offsets_d, nL*n_edges*sizeof(real));
  cudaMemset(offsets_d, 0, nL*n_edges*sizeof(real)); // bias always > offset at beginning
  cudaMalloc(&ensemble_dUdL_d, nL*n_edges*sizeof(real));
  cudaMemset(ensemble_dUdL_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&ensemble_dUdL2_d, nL*n_edges*sizeof(real));
  cudaMemset(ensemble_dUdL2_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&ensemble_var_d, nL*n_edges*sizeof(real));
  cudaMemset(ensemble_var_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&weights_d, nL*n_edges*sizeof(real));
  cudaMemset(weights_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&partition_functions, nL*sizeof(real));
  cudaMemset(partition_functions, 0, nL*sizeof(real));
  cudaMalloc(&partition_offsets, nL*sizeof(real));
  cudaMemset(partition_offsets, 0, nL*sizeof(real));
  cudaMalloc(&weighted_dUdL_d, nL*n_edges*sizeof(real));
  cudaMemset(weighted_dUdL_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&weighted_dUdL2_d, nL*n_edges*sizeof(real));
  cudaMemset(weighted_dUdL2_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&average_dUdL_d, nL*n_edges*sizeof(real));
  cudaMemset(average_dUdL_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&average_dUdL2_d, nL*n_edges*sizeof(real));
  cudaMemset(average_dUdL2_d, 0, nL*n_edges*sizeof(real));
  cudaMalloc(&ave_var_d, nL*n_edges*sizeof(real));
  cudaMemset(ave_var_d, 0, nL*n_edges*sizeof(real));
}

__global__ void add_sample_abf_kernel(
  real kT, real nFull, int nL,
  real* lambdas, real* dU_msld, real* hist_potential,
  real* histogram_counts, int* hist_indices, 
  int num_bins, real L_max, real L_min, real* offsets,
  real* partition, real* partition_offset,
  real* weights, real* weighted_dU_dL, real* weighted_dUdL2,
  real* ensemble_dUdL, real* ensemble_dUdL2, real* variance,
  real* average_dUdL, real* average_dUdL2, real* ave_var) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<nL) {
    // PBMeta Bias
    real sum = 0;
    for (int j = 0; j < nL; j++) {
      sum += exp(-hist_potential[j]/kT);
    }
    real potEnergy = -kT*log(sum);
    real lambda = lambdas[i+1];
    int bin = get_histogram_index(lambda, num_bins, L_max, L_min); 
    int hist_bin = bin + hist_indices[i];
    real dUdL = dU_msld[i+1]; // lambdaForce before biasing forces
    // Uniform Estimations
    average_dUdL[hist_bin] = (average_dUdL[hist_bin] * histogram_counts[hist_bin] + dUdL) / (histogram_counts[hist_bin] + 1);
    average_dUdL2[hist_bin] = (average_dUdL2[hist_bin] * histogram_counts[hist_bin] + dUdL*dUdL) / (histogram_counts[hist_bin] + 1);
    ave_var[hist_bin] = average_dUdL2[hist_bin] - pow(average_dUdL[hist_bin], 2);
    histogram_counts[hist_bin] += 1;
    real beta = 1 / kT;
    real bias = beta * potEnergy;
    if (bias >= offsets[hist_bin]) { // largest boltzmann weight = 1
      real correction = exp(offsets[hist_bin]-bias); // exp(U-old)*exp(old-new) = exp(U-new)
      offsets[hist_bin] = bias;
      weights[hist_bin] *= correction;
      weighted_dU_dL[hist_bin] *= correction;
      weighted_dUdL2[hist_bin] *= correction;
    }
    weights[hist_bin] += exp(bias - offsets[hist_bin]);
    weighted_dU_dL[hist_bin] += dUdL * exp(bias - offsets[hist_bin]);
    weighted_dUdL2[hist_bin] += dUdL*dUdL * exp(bias - offsets[hist_bin]);
    if (bias >= partition_offset[i]) {
      real correction = exp(partition_offset[i]-bias);
      offsets[i] = bias;
      partition[i] *= correction;
    }
    partition[i] += exp(bias - partition_offset[i]);
    // The offset we defined cancels in this calculation
    // sum(Fl*exp(g(L, dU))) / sum(exp(g(L, dU))
    ensemble_dUdL[hist_bin] = weighted_dU_dL[hist_bin] / weights[hist_bin];
    ensemble_dUdL2[hist_bin] = weighted_dUdL2[hist_bin] / weights[hist_bin];
    if (histogram_counts[hist_bin] < nFull) {
      ensemble_dUdL[hist_bin] = histogram_counts[hist_bin] / nFull * ensemble_dUdL[hist_bin];
    }
    variance[hist_bin] = ensemble_dUdL2[hist_bin] - pow(ensemble_dUdL[hist_bin], 2);
  }
}

// This is only called if update histogram is called
void Msld::add_sample_abf(System* system){
  cudaStream_t stream = 0;
  Run *r = system->run;
  State *s = system->state;
  int shMem = 0;

  if (system->run) { // need to wait for energy
    stream=system->run->abfBias;
  }

  add_sample_abf_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    s->leapParms1->kT, nFull, blockCount-1,
    s->lambda_fd, dU_msld_d, hist_potential_d,
    abf_histogram_d, abf_index_d, L_abf_bins, L_max, L_min,
    offsets_d, partition_functions, partition_offsets,
    weights_d, weighted_dUdL_d, weighted_dUdL2_d,
    ensemble_dUdL_d, ensemble_dUdL2_d, ensemble_var_d,
    average_dUdL_d, average_dUdL2_d, ave_var_d);

  // Logging for testing
  // Copy to host and print out info
  if (system->run->step % 10000 == 0){
    int len = (blockCount-1)*L_abf_bins;
    real counts[len], ens_dUdL[len], ens_var_dUdL[len], ave_dUdL[len], ave_var_dUdL[len], weights[len], offsets[len];
    real Z[blockCount-1], Z_off[blockCount-1];
    int indices[blockCount-1];
    cudaMemcpy(counts, abf_histogram_d, len*sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(ens_dUdL, ensemble_dUdL_d, len*sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(ens_var_dUdL, ensemble_var_d, len*sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(ave_dUdL, average_dUdL_d, len*sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(ave_var_dUdL, ave_var_d, len*sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights, weights_d, len*sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(offsets, offsets_d, len*sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices, abf_index_d, (blockCount-1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Z, partition_functions, (blockCount-1)*sizeof(real), cudaMemcpyDefault);
    cudaMemcpy(Z_off, partition_offsets, (blockCount-1)*sizeof(real), cudaMemcpyDefault);
    int count = 0;
    system->state->recv_lambda();
    printf("Step: %ld\n", system->run->step);
    real fractionPhysical = 0.0;
    for (int i = 0; i < siteCount-1; i++) {
      int refID = count;
      int startRef = indices[count];
      real refUniformTI = 0;
      real refUmbrellaTI = 0;
      real refOSSTI = 0;
      for (int k = 0; k < blocksPerSite[i+1]; k++) {
        int start= indices[count];
        // dG WT->j = -kT ln(Zj(1)/ZWT(1)) - (TI_j - TI_WT)
        int bins = system->msld->L_abf_bins-1; // index to last bin
        real wWT = weights[startRef + bins];
        real offWT = offsets[startRef + bins];
        real ZWT = Z[refID];
        real offZWT = Z_off[refID];
        real wj = weights[indices[count] + bins];
        real offJ = offsets[indices[count] + bins];
        real Zj = Z[count];
        real offZj = Z_off[count];
        // Accounts for bias felt when sampled L>Lc assuming TI estimate is in equilibrium/constant (bad assumption?)
        real logProb = -system->state->leapParms1->kT*log((wj*exp(offJ - offWT) / wWT)*(ZWT*exp(offZWT - offZj)/Zj));
        // Common Info
        printf("Site %d, Sub %d\n", i, k);
        if (tracking_only) {
          printf("Tracking only mode!!\n");
        }
        printf("Lambda: %f \n", s->lambda[count+1]);
        printf("Count >= %3.2f: %f\n", 1.0 - .5/(system->msld->L_abf_bins-1.0), counts[start + system->msld->L_abf_bins-1]);
        fractionPhysical += counts[start + system->msld->L_abf_bins-1];
        real samples = 0;
        for (int j = 0; j < L_abf_bins; j++) {
          samples += counts[start + j];
        }
        printf("Total Samples: %f\n", samples);
        printf("Histogram: [ ");
        for (int j = 0; j < L_abf_bins; j++) {
          printf("%f, ", counts[start+j]);
        }
        printf("]\n");
        // Standard ABF (equal weights)
        printf("\nUniform Weighting:\n");
        printf("E[dU/dL]: [");
        for (int j = 0; j < L_abf_bins; j++) {
          printf("%f, ", ave_dUdL[start+j]);
        }
        printf("]\n");
        printf("Std[dU/dL]: [");
        for (int j = 0; j < L_abf_bins; j++) {
          printf("%f, ", sqrt(ave_var_dUdL[start+j]));
        }
        printf("]\n");
        real sum = 0;
        printf("dG 0->i: [");
        for (int j = 0; j < L_abf_bins-1; j++) {
          real factor = j == 0 || j == L_abf_bins-2 ? .5 : 1.0;
          real width = factor * (L_max - L_min) / (L_abf_bins-1.0);
          sum += width * (ave_dUdL[start + j] + ave_dUdL[start + j+1]) / 2.0;
          printf("%f, ", sum);
        }
        printf("]\n");
        if (count == refID) {
          refUniformTI = sum;
        }
        real TI_j = sum;
        real dG = logProb - (TI_j - refUniformTI);
        printf("Uniform dG 0->1 = %f - (%f - %f) = %f \n", logProb, TI_j, refUniformTI, dG);
        // PBMetaD ABF
        printf("\nABF Umbrella Re-weighting:\n");
        if (!oss_abf && !tracking_only) {
          printf("Using this as ABF potential!!!\n");
        }
        printf("<dU/dL>: [ ");
        for (int j = 0; j < L_abf_bins; j++) {
          printf("%f, ", ens_dUdL[start+j]);
        }
        printf("]\n");
        printf("<Std[dU/dL]>: [");
        for (int j = 0; j < L_abf_bins; j++) {
          printf("%f, ", sqrt(ens_var_dUdL[start+j]));
        }
        printf("]\n");
        printf("Umbrella dG 0->i: [");
        sum = 0;
        for (int j = 0; j < L_abf_bins-1; j++) {
          real factor = j == 0 || j == L_abf_bins-2 ? .5 : 1.0;
          real width = factor * (L_max - L_min) / (L_abf_bins-1.0);
          sum += width * (ens_dUdL[start + j] + ens_dUdL[start + j+1]) / 2.0;
          printf("%f, ", sum);
        }
        printf("]\n");
        if (count == refID) {
          refUmbrellaTI = sum;
        }
        TI_j = sum;
        dG = logProb - (TI_j - refUmbrellaTI);
        printf("dG 0->1 = %f - (%f - %f) = %f \n", logProb, TI_j, refUmbrellaTI, dG);

        if (oss) {
          printf("\nOSS Re-weighting:\n");
          if (oss_abf && !tracking_only) {
            printf("Using this as ABF potential!!\n");
          }
          if (temper) {
            real minFL[blockCount-1];
            cudaMemcpy(minFL, minL_maxdUdL_d, (blockCount-1)*sizeof(real), cudaMemcpyDefault);
            real temperFactor = exp(-max(0.0, minFL[count] - temper_min) / (tempering*system->state->leapParms1->kT));
            printf("OSS Tempering Percentage: %6.3f%%, minL(maxdUdL(potential)): %f\n", 100*temperFactor, minFL[count]);
          }
          real oss_dUdL[L_oss_bins*(blockCount-1)], oss_var[L_oss_bins*(blockCount-1)];
          start = count*L_oss_bins;
          cudaMemcpy(oss_dUdL, oss_ensemble_dUdL_d, (blockCount-1)*L_oss_bins*sizeof(real), cudaMemcpyDefault);
          cudaMemcpy(oss_var, oss_var_d, (blockCount-1)*L_oss_bins*sizeof(real), cudaMemcpyDefault);
          printf("<dU/dL>: [");
          for (int j = 0; j < L_oss_bins-1; j++) {
            printf("%f, ", oss_dUdL[start + j]);
          }
          printf("]\n");
          printf("<Std[dU/dL]>: [");
          for (int j = 0; j < L_oss_bins-1; j++) {
            printf("%f, ", sqrt(oss_var[start + j]));
          }
          printf("]\n");
          printf("OST dG 0->i: [");
          sum = 0;
          for (int j = 0; j < L_oss_bins-1; j++) {
            real factor = j == 0 || j == L_oss_bins-2 ? .5 : 1.0;
            real width = factor * (L_max - L_min) / (L_oss_bins-1.0);
            sum += width * (oss_dUdL[start + j] + oss_dUdL[start + j+1]) / 2.0;
            printf("%f, ", sum);
          }
          printf("]\n");
          if (count == refID) {
            refOSSTI = sum;
          }
          TI_j = sum;
          dG = logProb - (TI_j - refOSSTI);
          printf("dG 0->1 = %f - (%f - %f) = %f \n", logProb, TI_j, refOSSTI, dG);
        }
        printf("\n\n");
        count++;
      }
    }
    real totalSamples = 0;
    for (int i = 0; i < L_abf_bins; i++) {
      totalSamples += counts[i];
    }
    fractionPhysical /= totalSamples;
    printf("Fraction Physical: %f", fractionPhysical);
    printf("\n\n");
  }
}

// This is done on GPU just to avoid moving data back and forth
__global__ void getforce_abf_kernel(
  real* lambdas, real* lambdaForce, real* step_force,
  int num_bins, real L_max, real L_min,
  real* ensemble_dUdL,
  real_e *energy, int nL)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;
  real dUdL_abf = 0;

  if (i < nL) {
    real lambda = lambdas[i+1]; // always < 1.0
    int bin = get_histogram_index(lambda, num_bins, L_max, L_min);
    int histBin = bin + i*num_bins;
    // Lerp implementation of ABF 
    real res = 1.0/(num_bins-1); // first and last are half bins so -1 total bins in range [L_max, L_min)
    real center = bin*res;
    real dUdL = ensemble_dUdL[histBin];
    real dist = lambda-center;
    real partner_center, partner_dUdL, interp;
    if(dist > 0){ // lambda in upper half of bin -> never true for bin=num_bins-1
      partner_center = center + res;
      partner_dUdL = ensemble_dUdL[histBin+1];
      interp = dist / res;
      dUdL_abf = (1 - interp)*dUdL + interp*partner_dUdL;
    } else {
      partner_center = center - res;
      partner_dUdL = ensemble_dUdL[histBin-1];
      interp = (lambda - partner_center) / res;
      dUdL_abf = (1 - interp)*partner_dUdL + interp*dUdL;
    }
    dUdL_abf = -dUdL_abf; // we add -'ve of <dU/dL>
    atomicAdd(&lambdaForce[i+1], dUdL_abf);
    atomicAdd(&step_force[i], dUdL_abf);
    if (energy) {
      int start = i*num_bins;
      for (int j = 0; j < bin - 1; j++) { 
        real factor = j == 0 ? .5 : 1;
        lEnergy += factor*(ensemble_dUdL[start+j] + ensemble_dUdL[start+j+1])*(.5/(num_bins-1));
      }
      if(dist > 0){
        lEnergy += (ensemble_dUdL[histBin] + dUdL_abf)*(.5/(num_bins-1));
      } else {
        lEnergy += (ensemble_dUdL[histBin-1] + dUdL_abf)*(.5/(num_bins-1));
      }
      lEnergy = -lEnergy; // We add -<dU/dL> to the force
    }
  }

  if (energy) {
    __syncthreads();
    real_sum_reduce(lEnergy, sEnergy, energy);
  }
}

void Msld::getforce_abf(System *system, bool calcEnergy) {
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
    stream=system->run->abfBias;
  }

  // Change which <dU/dL> we use for ABF forces
  real* dUdL = oss_abf ? oss_ensemble_dUdL_d : ensemble_dUdL_d;
  int bins = oss_abf ? L_oss_bins : L_abf_bins;
  getforce_abf_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    s->lambda_fd, s->lambdaForce_d, step_force_d,
    bins, L_max, L_min, dUdL,
    pEnergy, blockCount-1);
}

void Msld::init_oss(System* system){
  int nL = blockCount-1; // 0th lambda is environment
    int index[nL];
    for (int i = 0; i < nL; i++) {
      index[i] = i*L_oss_bins*dUdL_bins;
    }
    cudaMalloc(&oss_index_d, nL*sizeof(int)); // index into lambda's histogram
    cudaMemcpy(oss_index_d, index, nL*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&oss_ensemble_dUdL_d, nL*L_oss_bins*sizeof(real));
    cudaMemset(oss_ensemble_dUdL_d, 0, nL*L_oss_bins*sizeof(real));
    cudaMalloc(&oss_var_d, nL*L_oss_bins*sizeof(real));
    cudaMemset(oss_var_d, 0, nL*L_oss_bins*sizeof(real));
    cudaMalloc(&oss_potential_d, nL*L_oss_bins*dUdL_bins*sizeof(real)); // ~4/8 Mb per histogram
    cudaMemset(oss_potential_d, 0, nL*L_oss_bins*dUdL_bins*sizeof(real));
    cudaMalloc(&oss_histogram_d, nL*L_oss_bins*dUdL_bins*sizeof(real)); // ~4/8 Mb per histogram
    cudaMemset(oss_histogram_d, 0, nL*L_oss_bins*dUdL_bins*sizeof(real));

    cudaMalloc(&minL_maxdUdL_d, (blockCount-1)*sizeof(real));
    cudaMemset(minL_maxdUdL_d, 0, (blockCount-1)*sizeof(real));
    cudaMalloc(&dGdF_d, blockCount*sizeof(real)); // use blockCount so that it is indexed same as lambda array
    cudaMemset(dGdF_d, 0, blockCount*sizeof(real));
    cudaMalloc(&dGdL_d, blockCount*sizeof(real));
    cudaMemset(dGdL_d, 0, blockCount*sizeof(real));
}

__global__ void add_sample_hist_kernel(
  real kT, int nL, real* lambdas, real* dU_msld, real* hist_potential,
  real weight, real tempering, real temper_min, real* minL_maxdUdL_d,
  real* histogram, int* hist_indices,
  int L_bins, real L_max, real L_min, 
  int dUdL_bins, real dUdL_max, real dUdL_min,
  bool temper) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i < nL) {
    // Get location in histogram (L, dUdL) => (X, Y) starting from lower left corner of hist
    int X =  get_histogram_index(lambdas[i+1], L_bins, L_max, L_min);
    int Y = get_histogram_index(dU_msld[i+1], dUdL_bins, dUdL_max, dUdL_min);
    int index = hist_indices[i] + X*dUdL_bins + Y;
    real sum = 0.0;
    for (int j = 0; j < nL; j++) {
      sum += exp(-hist_potential[j] / kT);
    }
    real factorDecouple = exp(-hist_potential[i] / kT) / sum;
    real potential = max(0.0, minL_maxdUdL_d[i] - temper_min);
    real factorTemper = temper ? exp(-potential / (tempering*kT)) : 1.0;
    if (X >= 0 && X < L_bins && Y >= 0 && Y < dUdL_bins) {
      histogram[index] += weight * factorDecouple * factorTemper;
    }
  }
};

// Only called if update fe is true
void Msld::add_sample_hist(System* system) {
  cudaStream_t stream = 0;
  Run* r = system->run;
  State* s = system->state;
  int shMem = 0;

  if (system->run) {
    stream=system->run->ossBias;
  }

  if (system->run->step % 100*system->msld->sample_freq == 0 && temper) {
    system->msld->get_tempering_hist(system); // Evaluates potential on grid everywhere
  }

  add_sample_hist_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    s->leapParms1->kT, blockCount-1, s->lambda_fd, dU_msld_d, hist_potential_d,
    gaussian_weight, tempering, temper_min, minL_maxdUdL_d, oss_histogram_d, oss_index_d,
    L_oss_bins, L_max, L_min, dUdL_bins, dUdL_max, dUdL_min,
    temper);
}

__global__ void getforce_hist_kernel(
  int nL, real* lambdas, real* dU_msld,
  real* hist_potential, int* hist_indices, real* histogram,
  int dUdL_bins, real dUdL_max, real dUdL_min, int dUdL_search,
  int L_bins, real L_max, real L_min, int L_search,
  real dUdL_std, real L_std,
  real* dGdF, real* dGdL,
  bool mirror_Lmin, bool mirror_Lmax
) {
  // 2D grid: blockIdx.y = lambda index, blockIdx.x*blockDim.x+threadIdx.x = L search offset
  int iSearch = blockIdx.x * blockDim.x + threadIdx.x;
  int iL = blockIdx.y;
  // Shared memory for energy reduction
  extern __shared__ real sEnergy[];
  real local_bias = 0.0;
  real local_dUdL_force = 0.0;
  real local_L_force = 0.0;
  if (iL < nL) {
    // Get location in histogram (L, dUdL) => (X, Y) starting from lower left corner of hist
    int X = get_histogram_index(lambdas[iL+1], L_bins, L_max, L_min);
    int Y = get_histogram_index(dU_msld[iL+1], dUdL_bins, dUdL_max, dUdL_min);
    int j = X - L_search + iSearch;
    if (j >= X-L_search && j <= X+L_search) {
      int L_index = j;
      real mirrorFactor = 1;
      // Optional mirror at L=0 & L_1 to put L_index in range
      if (mirror_Lmin) {
        L_index = (L_index < 0) ? -L_index : L_index;
        mirrorFactor = L_index == 0 ? 2.0 : 1.0;
      }
      if (mirror_Lmax) {
        L_index = (L_index > L_bins-1) ? L_index - 2*(L_index-(L_bins-1)) : L_index;
        mirrorFactor = (mirror_Lmin && L_index == 0) || L_index == L_bins-1 ? 2.0 : 1.0;
      }
      if (L_index >= 0 && L_index < L_bins) {
        real L_resolution = (L_max-L_min)/(L_bins-1.0);
        real L_center = L_min + j*L_resolution; // Important that this is j and below dUdL_center has k
        real L_distance = (lambdas[iL+1] - L_center) / L_std;
        real L_gaussian = expf(-0.5*L_distance*L_distance);
        for (int k = Y-dUdL_search; k <= Y+dUdL_search; k++) {
          int dUdL_index = k;
          if (dUdL_index < 0) continue;
          if (dUdL_index >= dUdL_bins) break;

          real dUdL_resolution = (dUdL_max-dUdL_min)/(dUdL_bins-1.0);
          real dUdL_center = dUdL_min + k*dUdL_resolution;
          real dUdL_distance = (dU_msld[iL+1] - dUdL_center) / dUdL_std;
          real dUdL_gaussian = expf(-0.5*dUdL_distance*dUdL_distance);
          int index = hist_indices[iL] + L_index*dUdL_bins + dUdL_index;
          real weight = mirrorFactor * histogram[index];

          real tmp_bias = weight * L_gaussian * dUdL_gaussian;
          local_bias += tmp_bias;
          local_dUdL_force += -dUdL_distance/dUdL_std * tmp_bias;
          local_L_force += -L_distance/L_std * tmp_bias;
        }
        atomicAdd(&dGdF[iL+1], local_dUdL_force);
        atomicAdd(&dGdL[iL+1], local_L_force);
        atomicAdd(&hist_potential[iL], local_bias);
      }
    }
  }
}

__global__ void getforce_pb_oss(int nL, real kT, real* hist_potential, real* dGdF, real* dGdL, real* lambdaForce, real_e* energy) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i < nL) {
    real denom = 0.0;
    for (int j = 0; j < nL; j++) {
      denom += exp(-hist_potential[j] / kT);
    }
    real pbFactor = exp(-hist_potential[i] / kT) / denom;
    atomicAdd(&lambdaForce[i+1], pbFactor * dGdL[i+1]);
    dGdF[i+1] *= pbFactor;
    if (energy && i==0) {
      atomicAdd(energy, -kT*log(denom));
    }
  }
}

void Msld::getforce_hist(System *system, bool calcEnergy) {
  cudaStream_t stream = 0;
  Run *r = system->run;
  State *s = system->state;
  real_e *pEnergy = NULL;

  if (r->calcTermFlag[eebias] == false) return;
  if (system->run) {
    stream = system->run->ossBias;
  }
  // Calculate shared memory size for energy reduction
  int shMem = calcEnergy ? BLMS * sizeof(real) : 0;
  // Set up energy pointer if needed
  if (calcEnergy) {
    pEnergy = s->energy_d + eebias;
  }

  // Force from individual histograms
  // Set up grid and block dimensions
  dim3 blockDim(BLMS, 1, 1);
  dim3 gridDim;
  gridDim.x = (2*L_search + 1 + BLMS - 1) / BLMS; // Ceiling division for L search range
  gridDim.y = blockCount - 1;                     // Number of lambdas
  gridDim.z = 1;
  getforce_hist_kernel<<<gridDim, blockDim, shMem, stream>>>(
    blockCount-1, s->lambda_fd, dU_msld_d,
    hist_potential_d,
    oss_index_d, oss_histogram_d,
    dUdL_bins, dUdL_max, dUdL_min, dUdL_search,
    L_oss_bins, L_max, L_min, L_search,
    dUdL_std, L_std,
    dGdF_d, dGdL_d,
    mirror_Lmin, mirror_Lmax);
  // Force from PBMetaD combined function -kT ln(sum(-gi(li, Fi)/kT)))
  getforce_pb_oss<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    blockCount-1, system->state->leapParms1->kT,
    hist_potential_d, dGdF_d, dGdL_d,
    s->lambdaForce_d, pEnergy);
}

__global__ void getpotential_hist_kernel(
  int nL, int* hist_indices, real* histogram,
  int dUdL_bins, real dUdL_max, real dUdL_min, int dUdL_search,
  int L_bins, real L_max, real L_min, int L_search,
  real dUdL_std, real L_std,
  real* potential_grid,
  bool mirror_Lmin, bool mirror_Lmax
) {
  // TODO: Make much faster this is really slow with large L_oss_bins
  // Calculate thread global coordinates
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_bins = L_bins * dUdL_bins;
  int iL = blockIdx.y; // Use blockIdx.y for iL dimension
  real X_res = (L_max - L_min) / (L_bins - 1.0);
  real Y_res = (dUdL_max - dUdL_min) / (dUdL_bins - 1.0);
  for (int bin_idx = tid; bin_idx < total_bins; bin_idx += blockDim.x * gridDim.x) {
    if (iL >= nL) continue;
    // Convert linear bin index to 2D X,Y coordinates
    int X = bin_idx / dUdL_bins;
    int Y = bin_idx % dUdL_bins;
    if (X >= L_bins || Y >= dUdL_bins) continue;
    real X_center = L_min + X * X_res;
    real Y_center = dUdL_min + Y * Y_res;
    real bias = 0.0;
    for (int j = X - L_search; j <= X + L_search; j++) {
      int L_index = j;
      real mirrorFactor = 1.0;
      if (mirror_Lmin) {
        L_index = (L_index < 0) ? -L_index : L_index;
        mirrorFactor = (L_index == 0) ? 2.0 : 1.0;
      }
      if (mirror_Lmax) {
        L_index = (L_index >= L_bins) ? L_bins - 1 - (L_index - (L_bins - 1)) : L_index;
        mirrorFactor = (mirror_Lmin && L_index == 0) || L_index == L_bins - 1 ? 2.0 : 1.0;
      }
      if (L_index < 0 || L_index >= L_bins) continue;

      real L_center = L_min + j * X_res;
      real L_distance = (X_center - L_center) / L_std;
      real L_gaussian = expf(-0.5 * L_distance * L_distance);
      for (int k = Y - dUdL_search; k <= Y + dUdL_search; k++) {
        if (k < 0 || k >= dUdL_bins) continue;
        real dUdL_center = dUdL_min + k * Y_res;
        real dUdL_distance = (Y_center - dUdL_center) / dUdL_std;
        real dUdL_gaussian = expf(-0.5 * dUdL_distance * dUdL_distance);
        int index = hist_indices[iL] + L_index * dUdL_bins + k;
        real weight = mirrorFactor * histogram[index];

        bias += weight * L_gaussian * dUdL_gaussian;
      }
    }
    int grid_index = hist_indices[iL] + X * dUdL_bins + Y;
    potential_grid[grid_index] = bias;
  }
}

void Msld::getpotential_hist(System *system){
  // Calculate grid dimensions based on bin counts and nL
  dim3 grid((L_oss_bins * dUdL_bins + BLMS - 1) / BLMS, blockCount-1);
  getpotential_hist_kernel<<<grid, BLMS>>>(
      blockCount-1, oss_index_d, oss_histogram_d,
      dUdL_bins, dUdL_max, dUdL_min, dUdL_search,
      L_oss_bins, L_max, L_min, L_search,
      dUdL_std, L_std, oss_potential_d,
      mirror_Lmin, mirror_Lmax);
}

__global__ void min_L_max_dUdL_kernel(
    int nL, int* hist_indices,
    int L_bins, int dUdL_bins,
    real dUdL_max, real dUdL_min,
    real* oss_ens_dUdL, real* oss_var,
    real* potential_grid, real* histogram,
    real* min_bias
) {
  // TODO: Claude 3.7 wrote, eventually write your own reductions
  extern __shared__ real shared_max_bias[];
  int iL = blockIdx.x;
  int tid = threadIdx.x;
  int stride = blockDim.x;
  if (iL < nL) {
    // Each thread initializes its assigned X positions
    for (int x = tid; x < L_bins; x += stride) {
      shared_max_bias[x] = -INFINITY;
    }
    __syncthreads();

    // For each X position, find the maximum bias across all Y values
    real dUdL_res = (dUdL_max - dUdL_min) / (dUdL_bins - 1.0);
    for (int x = tid; x < L_bins; x += stride) {
      real local_max = -INFINITY;
      real offset = 0.0;
      real weighted_dUdL = 0;
      real weighted_dUdL2 = 0;
      real Z = 0;
      for (int y = 0; y < dUdL_bins; y++) {
        int grid_index = hist_indices[iL] + x * dUdL_bins + y;
        real current_bias = potential_grid[grid_index];
        local_max = max(local_max, current_bias);
        if (current_bias <= 0.0) { continue; } // if we don't do this the bins with zero bias contribute
        if (histogram[grid_index] <= 0.0) { continue; } // No contribution from dUdL without samples
        real dUdL = dUdL_min + y*dUdL_res;
        if (current_bias > offset) {
          real correction = exp(offset - current_bias); // Zero on first execution since -INFINITY
          weighted_dUdL *= correction;
          weighted_dUdL2 *= correction;
          Z *= correction;
          offset = current_bias;
        }
        weighted_dUdL += dUdL * exp(current_bias - offset);
        weighted_dUdL2 += dUdL*dUdL * exp(current_bias - offset);
        Z += exp(current_bias - offset);
      }
      shared_max_bias[x] = local_max;
      oss_ens_dUdL[iL*L_bins + x] = Z > 1e-5 ? weighted_dUdL / Z : 0.0;
      oss_var[iL*L_bins + x] = Z > 1e-5 ? weighted_dUdL2 / Z - pow(oss_ens_dUdL[iL*L_bins + x], 2): 0.0;
    }
    __syncthreads();

    if (tid == 0) {
      real min_of_max_bias = INFINITY;
      for (int x = 0; x < L_bins; x++) {
        min_of_max_bias = min(min_of_max_bias, shared_max_bias[x]);
      }
      min_bias[iL] = min_of_max_bias;
    }
  }
}

void Msld::get_tempering_hist(System* system) {
  // Evaluate potential everywhere
  int sharedMemSize = L_oss_bins * sizeof(real);
  dim3 grid((L_oss_bins * dUdL_bins + BLMS - 1) / BLMS, blockCount-1);
  getpotential_hist_kernel<<<grid, BLMS, sharedMemSize, system->run->ossBias>>>(
      blockCount-1, oss_index_d, oss_histogram_d,
      dUdL_bins, dUdL_max, dUdL_min, dUdL_search,
      L_oss_bins, L_max, L_min, L_search,
      dUdL_std, L_std, oss_potential_d,
      mirror_Lmin, mirror_Lmax
  );

  // minL(maxdUdL(potential_grid))
  min_L_max_dUdL_kernel<<<blockCount-1, BLMS, sharedMemSize, system->run->ossBias>>>(
      blockCount-1, oss_index_d,
      L_oss_bins, dUdL_bins,
      dUdL_max, dUdL_min,
      oss_ensemble_dUdL_d, oss_var_d,
      oss_potential_d, oss_histogram_d,
      minL_maxdUdL_d);
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

void Msld::calc_lambda_from_theta(cudaStream_t stream,System *system)
{
  State *s=system->state;
  if (!fix) { // ffix
    calc_lambda_from_theta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,0,stream>>>(s->lambda_d,s->theta_d,siteCount,siteBound_d,fnex);
  } else {
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

__global__ void getforce_fixedBias_kernel(real *lambda,real *lambdaBias,real_f *lambdaForce,real_e *energy,int blockCount)
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

  getforce_fixedBias_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(s->lambda_fd,lambdaBias_d,s->lambdaForce_d,pEnergy,blockCount);
}

__global__ void getforce_variableBias_kernel(real *lambda,real_f *lambdaForce,real_e *energy,int variableBiasCount,struct VariableBias *variableBias)
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
    if (fj) atomicAdd(&lambdaForce[vb.j],fj);
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
