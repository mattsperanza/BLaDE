#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>

#include "msld/msld.h"

#include <bits/stl_algo.h>

#include "system/system.h"
#include "io/io.h"
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


  // Histogram details
  bin_edges=NULL;
  integral_components=NULL;

  histogram_counts=NULL;
  ensemble_dUdL=NULL;
  ensemble_dUdL2=NULL;
  offsets=NULL;
  weights=NULL;
  weighted_dUdL=NULL;
  weighted_dUdL2=NULL;
  ensemble_dUdL2=NULL;
  variance=NULL;

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
    if (vb.type<=0 || vb.type>10) {
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
void Msld::initialize(System *system) {
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

  // TODO: Check if TI is requested and add inp options
  // Histogram variables
  int nL = blockCount-1;
  // Histogram details
  total_bins=first_half_bins+second_half_bins;
  bin_edges=(real*) calloc((siteCount-1)*(total_bins+1), sizeof(real));
  // Lambdas required for correct uniform distribution
  assign_edges(siteCount-1, blocksPerSite, first_half_bins, second_half_bins, bin_edges);
  hist_index = (int*) malloc(nL*sizeof(int));
  cudaMalloc(&bin_edges_d, (total_bins+1)*sizeof(real));
  cudaMemcpy(bin_edges_d, bin_edges, (total_bins+1)*sizeof(real), cudaMemcpyHostToDevice);
  cudaMalloc(&offsets_d, nL*total_bins*sizeof(real));
  for(int i = 0; i < nL; i++) {
    // Edges indexed with this number + i*nL since they are one larger
    hist_index[i] = i * total_bins;
  }
  cudaMalloc(&hist_index_d, nL*sizeof(int));
  cudaMemcpy(hist_index_d, hist_index, nL*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&step_force_d, nL*sizeof(real));
  cudaMalloc(&step_potential_d, nL*sizeof(real));
  opes_barrier = 30;
  opes_gamma = 1.0/(kB*298.15) * opes_barrier;
  opes_eps = exp(-opes_barrier/(kB*298.15*(1-1/opes_gamma)));

  // Depth = n storage
  offsets = (real**) malloc(depth*sizeof(real*));
  offsets_d = (real**) malloc(depth*sizeof(real*));
  integral_components = (real**) malloc(depth*sizeof(real*));
  integral_components_d = (real**) malloc(depth*sizeof(real*));
  histogram_counts = (real**) malloc(depth*sizeof(real*));
  histogram_counts_d = (real**) malloc(depth*sizeof(real*));
  weights = (real**) malloc(depth*sizeof(real*));
  weights_d = (real**) malloc(depth*sizeof(real*));
  weighted_dUdL= (real**) malloc(depth*sizeof(real*));
  weighted_dUdL_d = (real**) malloc(depth*sizeof(real*));
  ensemble_dUdL = (real**) malloc(depth*sizeof(real*));
  ensemble_dUdL_d = (real**) malloc(depth*sizeof(real*));
  average_dUdL = (real**) malloc(depth*sizeof(real*));
  average_dUdL_d = (real**) malloc(depth*sizeof(real*));
  ensemble_dUdL2 = (real**) malloc(depth*sizeof(real*));
  ensemble_dUdL2_d = (real**) malloc(depth*sizeof(real*));
  variance = (real**) malloc(depth*sizeof(real*));
  variance_d = (real**) malloc(depth*sizeof(real*));
  weighted_dUdL2= (real**) malloc(depth*sizeof(real*));
  weighted_dUdL2_d = (real**) malloc(depth*sizeof(real*));
  partition_function = (real**) malloc(depth*sizeof(real*));
  partition_function_d = (real**) malloc(depth*sizeof(real*));
  partition_offset_d = (real**) malloc(depth*sizeof(real*));
  probability_distribution = (real**) malloc(depth*sizeof(real*));
  probability_distribution_d = (real**) malloc(depth*sizeof(real*));
  dPdL = (real**) malloc(depth*sizeof(real*));
  dPdL_d = (real**) malloc(depth*sizeof(real*));
  opes_potential = (real**) malloc(depth*sizeof(real));
  opes_potential_d = (real**) malloc(depth*sizeof(real));
  opes_force = (real**) malloc(depth*sizeof(real));
  opes_force_d = (real**) malloc(depth*sizeof(real*));
  weighted_dUbias_dL_d = (real**) malloc(depth*sizeof(real*));
  weighted_partition_function_d = (real**) malloc(depth*sizeof(real*));
  for(int i = 0; i < depth; i++) { // these are really long for GPU sake, otherwise would be separated into different classes
    offsets[i] = (real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&offsets_d[i], nL*total_bins*sizeof(real));
    integral_components[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&integral_components_d[i], nL*total_bins*sizeof(real));
    histogram_counts[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&histogram_counts_d[i],nL*total_bins*sizeof(real));
    average_dUdL[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&average_dUdL_d[i], nL*total_bins*sizeof(real));
    ensemble_dUdL[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&ensemble_dUdL_d[i], nL*total_bins*sizeof(real));
    weights[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&weights_d[i], nL*total_bins*sizeof(real));
    weighted_dUdL[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&weighted_dUdL_d[i], nL*total_bins*sizeof(real));
    weighted_dUdL2[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&weighted_dUdL2_d[i], nL*total_bins*sizeof(real));
    ensemble_dUdL2[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&ensemble_dUdL2_d[i], nL*total_bins*sizeof(real));
    variance[i]=(real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&variance_d[i], nL*total_bins*sizeof(real));
    partition_function[i] = (real*) malloc(nL*sizeof(real));
    cudaMalloc(&partition_function_d[i], nL*sizeof(real));
    probability_distribution[i] = (real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&probability_distribution_d[i], nL*total_bins*sizeof(real));
    cudaMalloc(&partition_offset_d[i], nL*sizeof(real));
    dPdL[i] = (real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&dPdL_d[i], nL*total_bins*sizeof(real));
    opes_potential[i] = (real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&opes_potential_d[i], nL*total_bins*sizeof(real));
    opes_force[i] = (real*) malloc(nL*total_bins*sizeof(real));
    cudaMalloc(&opes_force_d[i], nL*total_bins*sizeof(real));
    cudaMalloc(&weighted_dUbias_dL_d[i], nL*total_bins*sizeof(real));
    cudaMalloc(&weighted_partition_function_d[i], nL*sizeof(real));
  }
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


real beta_cdf_inverse(const real y, const int b){
  return 1 - pow(1.0 - y, 1.0 / b);
}

/*
 * Uniform multi-dim distribution of samples creates a marginal beta distribution. This takes the form of:
 *
 *  p(x) = (a+b)!/(a!b!) * x^(a-1) * (1-x)^(b-1)
 *
 * In this case, a = 1 and b = nLamdas - 1. We want each bin in the lower half to have the
 * same probability density. To do this we integrate the beta distribution from 0 to x to get
 * the CDF and take evenly spaced samples of that function's inverse. This can be solved analytically
 * to give:
 *
 * x = 1 - (1-y)^(1/b)
 *
 * The second half of the bins are assigned linearly from the first half to 1.
*/
void assign_site_edges(const int num_lambdas, const int first, const int second, real *edges){
  int total = first + second + 1; // One bin between halves
  for (int i = 0; i < first; i++){
    edges[i] = beta_cdf_inverse((real) i / (real) first / 2.0, num_lambdas-1);
  }
  real rest = 1.0 - edges[first-1];
  real gap = rest / (second+1);
  for (int i = first; i < total; i++){
    edges[i] = (i-first+1)*gap + edges[first-1];
  }
}

void Msld::assign_edges(const int num_sites, const int *blocksPerSite, const int first_half_bins, const int second_half_bins, real *bin_edges){
  int total = first_half_bins + second_half_bins + 1; // Edge for last bin
  for (int i = 0; i < num_sites; i++){
    int num_lambdas = blocksPerSite[i+1];
    assign_site_edges(num_lambdas, first_half_bins, second_half_bins, bin_edges + i * total);
    // Print out the edges for debugging
    printf("Histogram edges for site %d\n", i);
  }
}

// This is unlikely to be worth speeding up w/ binary search
__device__ inline int get_bin_index(int site, real lambda, int total_bins, const real *bin_edges){
  int start = site*(total_bins+1);
  for (int i = start; i < start+total_bins; i++){
    if (lambda < bin_edges[i+1]){ // i+1 always in range for edges
      return i;
    }
  }
  // Out of range?
  return total_bins-1;
}

// There will never be simultaneous access of any element
// This is done on the gpu just to avoid moving data back and forth
__global__ void add_sample_kernel(
  real kT, int nFull, int* lambdaSites,
  real* lambdas, real* lambdaForce, real* step_force, real* step_potential, real potEnergy,
  real* histogram_counts, int* hist_indices, int total_bins, real* bin_edges, real* offsets,
  real* weights, real* weighted_dU_dL, real* weighted_dUbias_dL, real* weighted_dUdL2,
  real* partition_function, real* weighted_partition, real* partition_offset,
  real* probability, real* dPdL, real *opes_potential, real *opes_force, real opes_gamma, real opes_eps,
  real* ensemble_dUdL, real* average_dUdL, real* ensemble_dUdL2, real* variance,
  real* integral_components,
  int blockCount
  ) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<blockCount) {
    // Get U_msld = a log weight in the unbiased potential
    potEnergy = 0;
    for (int j = 0; j < blockCount-1; j++) {
      potEnergy += step_potential[j];
    }
    real lambda = lambdas[i+1];
    int bin = get_bin_index(lambdaSites[i+1]-1, lambda, total_bins, bin_edges);
    int hist_bin = bin + hist_indices[i];
    real dUdL = lambdaForce[i+1] - step_force[i]; // Correction to get dU_msld/dL
    average_dUdL[hist_bin] = (average_dUdL[hist_bin] * histogram_counts[hist_bin] + dUdL) / (histogram_counts[hist_bin] + 1);
    histogram_counts[hist_bin] += 1;
    real beta = 1 / kT;
    real bias = beta * potEnergy;
    if (bias > offsets[hist_bin]) { // largest weight = 1
      real correction = exp(offsets[hist_bin]-bias); // exp(U-old)*exp(old-new) = exp(U-new)
      offsets[hist_bin] = bias;
      weights[hist_bin] *= correction;
      weighted_dU_dL[hist_bin] *= correction;
      weighted_dUbias_dL[hist_bin] *= correction;
      weighted_dUdL2[hist_bin] *= correction;
    }
    weights[hist_bin] += exp(bias - offsets[hist_bin]);
    weighted_dU_dL[hist_bin] += dUdL * exp(bias - offsets[hist_bin]);
    weighted_dUbias_dL[hist_bin] += step_force[i] * exp(bias - offsets[hist_bin]);
    weighted_dUdL2[hist_bin] += dUdL*dUdL * exp(bias - offsets[hist_bin]);
    // The offset we defined cancels in this calculation - (exp(offset)*exp(U-offset))/(exp(offset)*sum(exp(U-offset)))
    ensemble_dUdL[hist_bin] = weighted_dU_dL[hist_bin] / weights[hist_bin];
    ensemble_dUdL2[hist_bin] = weighted_dUdL2[hist_bin] / weights[hist_bin];
    if (histogram_counts[hist_bin] < nFull) {
      ensemble_dUdL[hist_bin] = histogram_counts[hist_bin] / nFull * ensemble_dUdL[hist_bin];
    }
    variance[hist_bin] = ensemble_dUdL2[hist_bin] - pow(ensemble_dUdL[hist_bin], 2);
    // Integral components
    real lower_dUdL = ensemble_dUdL[hist_bin];
    int id = bin >= total_bins ? hist_bin : hist_bin + 1;
    real upper_dUdL = ensemble_dUdL[id];
    real width = bin_edges[bin+1] - bin_edges[bin];
    integral_components[hist_bin] = (lower_dUdL + upper_dUdL)/2 * width;
    // Partition function
    if (bias > partition_offset[i]) {
      real correction = exp(partition_offset[i] - bias);
      partition_function[i] *= correction;
      weighted_partition[i] *= correction;
      partition_offset[i] = bias;
    }
    partition_function[i] += exp(bias - partition_offset[i]);
    // Probability distribution & opes
    weighted_partition[i] += step_force[i] * exp(bias - partition_offset[i]);
    // TODO: Finish this part so that dPdL gives correct value
    for (int j = i*total_bins; j < i*total_bins+total_bins; j++) {
      probability[j] = exp(offsets[j]-partition_offset[i]) * weights[j] / partition_function[i];
      opes_potential[j] = (1-1/opes_gamma)/kT * log(probability[j] + opes_eps);
      dPdL[j] = (weighted_dUbias_dL[j] * partition_function[i] -
        weights[j] * weighted_partition[i]) / pow(partition_function[i], 2);
      // OPES potential and force
      opes_force[j] = -(opes_potential[j] - opes_potential[j+1]) / width;
    }
  }
}

__global__ void combine_histogram_kernel(
  int accumulate_into, int sample_from,
  int every_bin, int bin_per_histogram,
  real* bin_edges,
  real* histogram_counts, real* histogram_counts_into,
  real* average_dUdL, real* average_dUdL_into,
  real* offsets, real* offsets_into,
  real* weights, real* weights_into,
  real* partition_function, real* partition_function_into,
  real* partition_offset, real* partition_offset_into,
  real* probability_distribution, real* probability_distribution_into,
  real* weighted_dUdL, real* weighted_dUdL_into,
  real* weighted_dUdL2, real* weighted_dUdL2_into,
  real* ensemble_dUdL, real* ensemble_dUdL_into,
  real* ensemble_dUdL2, real* ensemble_dUdL2_into,
  real* variance, real* variance_into,
  real* integral_components, real* integral_components_into
  ) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int nL = every_bin / bin_per_histogram;
  // Combine histograms (only if has been sampled)
  if (i<every_bin && histogram_counts_into[i] + histogram_counts[i] > 0) {
    // Averages
    average_dUdL[i] = average_dUdL[i] * histogram_counts[i] + average_dUdL_into[i] * histogram_counts_into[i];
    average_dUdL[i] /= histogram_counts[i] + histogram_counts_into[i];
    average_dUdL_into[i] = 0;
    histogram_counts[i] += histogram_counts_into[i];
    histogram_counts_into[i] = 0;
    // Weights
    if (offsets_into[i] > offsets[i]) { // apply correction to sample_from samples
      real correction = expf(offsets[i] - offsets_into[i]);
      weights[i] = correction * weights[i] + weights_into[i];
      weighted_dUdL[i] = correction * weighted_dUdL[i] + weighted_dUdL_into[i];
      weighted_dUdL2[i] = correction * weighted_dUdL2[i] + weighted_dUdL2_into[i];
      offsets[sample_from] = offsets[accumulate_into];
    } else { // apply correction to accumulate_into samples
      real correction = expf(offsets_into[i] - offsets[i]);
      weights[i] += correction * weights_into[i];
      weighted_dUdL[i] += correction * weighted_dUdL_into[i];
      weighted_dUdL2[i] += correction * weighted_dUdL2_into[i];
    }
    weights_into[i] = 0;
    weighted_dUdL_into[i] = 0;
    weighted_dUdL2_into[i] = 0;
    offsets_into[i] = 0;
    // Recalculate ensemble averages
    ensemble_dUdL[i] = weighted_dUdL[i] / weights[i];
    ensemble_dUdL_into[i] = 0;
    ensemble_dUdL2[i] = weighted_dUdL2[i] / weights[i];
    ensemble_dUdL2_into[i] = 0;
    // Recalculate variance
    variance[i] = ensemble_dUdL2[i] - pow(ensemble_dUdL[i], 2);
    variance_into[i] = 0;
  }
  // Correct partition functions (every histogram)
  if (i < nL && partition_offset_into[i] > partition_offset[i]) {
    real correction = expf(partition_offset[i] - partition_offset_into[i]);
    partition_function[i] = correction * partition_function[i] + partition_function_into[i];
    partition_offset[i] = partition_offset_into[i];
    partition_function_into[i] = 0;
    partition_offset_into[i] = 0;
  } else if (i < nL) {
    real correction = expf(partition_offset_into[i] - partition_offset[i]);
    partition_function[i] += correction * partition_function_into[i];
    partition_function_into[i] = 0;
    partition_offset_into[i] = 0;
  }
  __syncthreads(); // wait for all threads to finish
  if (i<every_bin) {
    // Recalculate probability distribution
    int lambdaSite = i / bin_per_histogram;
    probability_distribution[i] = exp(offsets[i]-partition_offset[lambdaSite]) * weights[i] / partition_function[lambdaSite];
    probability_distribution_into[i] = 0;
    // Recalculate integral components
    real lower_dUdL = ensemble_dUdL[i];
    int id = i >= bin_per_histogram ? i : i + 1;
    real upper_dUdL = ensemble_dUdL[id];
    real width = bin_edges[i%bin_per_histogram+1] - bin_edges[i%bin_per_histogram];
    integral_components[i] = (lower_dUdL + upper_dUdL)/2 * width;
    integral_components_into[i] = 0;
  }
}

//TODO: test everything with multiple sites - sofar only tested with T4L
void Msld::add_sample(System* system){
  cudaStream_t stream = 0;
  Run *r = system->run;
  State *s = system->state;
  int shMem = 0;

  if (system->run) { // need to wait for energy
    stream=system->run->updateStream;
  }

  int id = system->msld->accumulate_into;
  add_sample_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    system->state->leapParms1->kT, system->msld->nFull, lambdaSite_d,
    s->lambda_fd, s->lambdaForce_d, step_force_d, step_potential_d, s->energy[eepotential],
    histogram_counts_d[id], hist_index_d, total_bins, bin_edges_d, offsets_d[id],
    weights_d[id], weighted_dUdL_d[id], weighted_dUbias_dL_d[id], weighted_dUdL2_d[id],
    partition_function_d[id], weighted_partition_function_d[id], partition_offset_d[id],
    probability_distribution_d[id], dPdL_d[id],
    opes_potential_d[id], opes_force_d[id], opes_gamma, opes_eps,
    ensemble_dUdL_d[id], average_dUdL_d[id], ensemble_dUdL2_d[id], variance_d[id],
    integral_components_d[id],
    blockCount-1);

  if (system->msld->accumulate_length != -1
    && system->run->step % system->msld->accumulate_length < system->msld->sampleFrequency) {
    // Combine histogram "accumulate_into" into "sample_from" and empty "accumulate_into"
    int N = (blockCount-1)*total_bins;
    int from = system->msld->sample_from;
    int into = system->msld->accumulate_into;
    combine_histogram_kernel<<<(N+BLMS-1)/BLMS,BLMS,shMem, stream>>>(
      system->msld->accumulate_into, system->msld->sample_from,
      N, total_bins,
      bin_edges_d,
      histogram_counts_d[from], histogram_counts_d[into],
      average_dUdL_d[from], average_dUdL_d[into],
      offsets_d[from], offsets_d[into],
      weights_d[from], weights_d[into],
      partition_function_d[from], partition_function_d[into],
      partition_offset_d[from], partition_offset_d[into],
      probability_distribution_d[from], probability_distribution_d[into],
      weighted_dUdL_d[from], weighted_dUdL_d[into],
      weighted_dUdL2_d[from], weighted_dUdL2_d[into],
      ensemble_dUdL_d[from], ensemble_dUdL_d[into],
      ensemble_dUdL2_d[from], ensemble_dUdL2_d[into],
      variance_d[from], variance_d[into],
      integral_components_d[from], integral_components_d[into]
    );
  }

  // Copy to host and print out info
  cudaMemcpy(histogram_counts[0], histogram_counts_d[0], (blockCount-1)*total_bins*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(ensemble_dUdL[0], ensemble_dUdL_d[0], (blockCount-1)*total_bins*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(integral_components[0], integral_components_d[0], (blockCount-1)*total_bins*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(weights[0], weights_d[0], (blockCount-1)*total_bins*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(partition_function[0], partition_function_d[0], (blockCount-1)*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(probability_distribution[0], probability_distribution_d[0], (blockCount-1)*total_bins*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(dPdL[0], dPdL_d[0], (blockCount-1)*total_bins*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(opes_potential[0], opes_potential_d[0], (blockCount-1)*total_bins*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(opes_force[0], opes_force_d[0], (blockCount-1)*total_bins*sizeof(real), cudaMemcpyDeviceToHost);
  // Print out lambda values
  // Print out values as arrays
  if (system->run->step % 100000 == 0){
    int count = 0;
    printf("Step: %ld\n", system->run->step);
    for (int i = 0; i < siteCount-1; i++) {
      real relative = 0;
      real Z0 = 0;
      for (int k = 0; k < blocksPerSite[i+1]; k++) {
        printf("Site %d, Sub %d\n", i, k);
        printf("Lambda: %f \n", s->lambda[count+1]);
        printf("Felt dUdL: %f\n", s->lambdaForce[count+1]);
        printf("PDF: [ %f ", probability_distribution[0][count*total_bins]);
        real cdf = probability_distribution[0][count*total_bins];
        for (int j = 1; j < total_bins; j++) {
          printf("%f, ", probability_distribution[0][count*total_bins+j]);
          cdf += probability_distribution[0][count*total_bins+j];
          //printf("%f, ", cdf);
        }
        printf("]\n");
        if (abs(cdf - 1.0) > 1e-4 && abs(cdf) > 1e-4){
          printf("Prob Sum: %f\n", cdf);
          printf("exiting...");
          exit(1);
        }
        printf("Histogram: [ %f, ", histogram_counts[0][count*total_bins]);
        for (int j = 1; j < total_bins; j++) {
          printf("%f, ", histogram_counts[0][count*total_bins+j]);
        }
        printf("]\n");
        printf("Weights: [ %f, ", weights[0][count*total_bins]);
        for (int j = 1; j < total_bins; j++) {
          printf("%f, ", weights[0][count*total_bins+j]);
        }
        printf("]\n");
        printf("<dU/dL>: [ %f, ", ensemble_dUdL[0][count*total_bins]);
        for (int j = 1; j < total_bins; j++) {
          printf("%f, ", ensemble_dUdL[0][count*total_bins+j]);
        }
        printf("]\n");
        real sum= 0;
        real Z = 0;
        printf("dG 0->i: [ %f, ", sum); // This is our FES
        for (int j = 1; j < total_bins; j++) {
          sum += integral_components[0][count*total_bins+j];
          printf("%f, ", sum);
          Z += exp(-1/(kB*298.15)*sum);
        }
        printf("]\n");
        if (k == 0) {
          Z0 = Z; // c0 = 0
        }
        int c = -kB*298.15*log(Z0/Z);
        if (k == 0){
          relative = sum;
        }
        sum -= relative - c;
        printf("dG 0->1: %f\n\n", sum);
        count++;
      }
    }
    printf("\n\n");
  }
}

// This is done on GPU just to avoid moving data back and forth
__global__ void getforce_histogram_kernel(
  real* lambdas, int* lambdaSites, real* lambdaForce, real* step_force, real* step_potential,
  int* histIndices, int total_bins, real* bin_edges,
  real* ensemble_dUdL, real* integral_components,
  real* opes_potential, real* opes_force,
  real_e *energy,
  int blockCount)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  real dUdL_abf = 0;
  real dUdL_opes = 0;
  if (i < blockCount) {
    real lambda = lambdas[i+1];
    // This value is in range for bins, +1 is in range for edges, but + 2 may be out of range
    int bin = get_bin_index(lambdaSites[i+1]-1, lambda, total_bins, bin_edges);
    int histBin = bin + histIndices[i];
    // Lerp implementation of ABF - values are defined on the bin's lower edges to avoid edge cases
    real low_edge = bin_edges[bin];
    real high_edge = bin_edges[bin+1];
    real dUdL_low = ensemble_dUdL[histBin];
    int id = bin+1 >= total_bins ? histBin : histBin + 1; // last edge has same value as last bin
    real dUdL_high = ensemble_dUdL[id];
    real interp = (lambda - low_edge) / (high_edge - low_edge);
    dUdL_abf = (1-interp) * dUdL_low + interp * dUdL_high;
    dUdL_abf = -dUdL_abf;
    // OPES/Meta
    //dUdL_opes = opes_force[histBin];
    real dUdL = dUdL_abf + dUdL_opes;
    atomicAdd(&lambdaForce[i+1], dUdL);
    step_force[i] = dUdL;
    if (energy) {
      // TODO: Move this into sample kernel
      for (int j = 0; j < bin-1; j++) { // integrate up to the bin prior to this one
        lEnergy += integral_components[j+histIndices[i]];
      }
      lEnergy += (dUdL_low + dUdL_abf) / 2 * (lambda - low_edge);
      lEnergy = -lEnergy; // We add -<dU/dL> to the force
      //lEnergy += opes_potential[histBin];
      step_potential[i] = lEnergy;
    }
  }

  if (energy) { // not sure this is necessary
    __syncthreads();
    real_sum_reduce(lEnergy, sEnergy, energy);
  }
}

void Msld::getforce_histogram(System *system, bool calcEnergy) {
  cudaStream_t stream = 0;
  Run *r = system->run;
  State *s = system->state;
  real_e *pEnergy = NULL;
  int shMem = 0;

  if (r->calcTermFlag[eelambda] == false) return;
  if (calcEnergy) {
    shMem = BLMS*sizeof(real)/32;
    pEnergy=s->energy_d+eelambda;
  }
  if (system->run) {
    stream=system->run->biaspotStream;
  }

  int id = system->msld->sample_from;
  getforce_histogram_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
    s->lambda_fd, lambdaSite_d, s->lambdaForce_d, step_force_d, step_potential_d,
    hist_index_d, total_bins, bin_edges_d,
    ensemble_dUdL_d[id], integral_components_d[id],
    opes_potential_d[id], opes_force_d[id],
    pEnergy, blockCount-1);
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
