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

  dLdT_d=NULL;
  d2LdT2_d=NULL;
  site_period_d=NULL;
  site_period=NULL;

  dUdL_msld_d=NULL;
  dUdL_msld=NULL;
  dUdT_msld_d=NULL;
  dUdT_msld=NULL;
  theta_temp_d=NULL;
  theta_temp=NULL;

  theta_histogram_d=NULL;
  histogram_1D_d=NULL;
  potential_1D_d=NULL;
  max_pot_d=NULL;
  min_dUdT_index_d=NULL;
  max_dUdT_index_d=NULL;
  histogram_2D_d=NULL;
  potential_2D_d=NULL;
  T_bins_d=NULL;
  T_bins=NULL;

  meta_bias_d=NULL;
  meta_bias=NULL;
  meta_min_V_d=NULL;
  oss_bias_d=NULL;
  oss_bias=NULL;
  dGdF_d=NULL;
  oss_min_V_d=NULL;

  abf_weighted_dUdT_d=NULL;
  abf_weighted_dUdT2_d=NULL;
  abf_weights_d=NULL;
  abf_offsets_d=NULL;
  abf_ensemble_dUdT_d=NULL;
  abf_variance_dUdT_d=NULL;
  abf_bias_d=NULL;
  abf_bias=NULL;
  dUdT_abf_d=NULL;
  dUdT_abf=NULL;

  LE_bias_d=NULL;
  LE_bias=NULL;
  LE_bins_d=NULL;
  LE_bins=NULL;
  LE_R_d=NULL;
  LE_visited_bins_d=NULL;
  LE_theta_sweep_d=NULL;
  LE_M_d=NULL;

  W_d=NULL;

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
  } else if (strcmp(token, "theta_fix") == 0){
    system->msld->theta_fix_value=io_nextf(line);
  } else if (strcmp(token, "theta_slow_fix") == 0){
    system->msld->theta_slow_fix=true;
    system->msld->theta_start=io_nextf(line);
    system->msld->theta_end=io_nextf(line);
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
  } else if (strcmp(token, "restartable") == 0){
    system->msld->restartable = io_nextb(line);
  } else if (strcmp(token, "log_freq") == 0){
    system->msld->log_freq=io_nexti(line);
  } else if (strcmp(token, "write_freq")==0){
    system->msld->write_freq=io_nexti(line);
  } else if (strcmp(token, "sample_freq") == 0){
    system->msld->sample_freq=io_nexti(line);
  } else if (strcmp(token, "LE") == 0){
    system->msld->LE=io_nextb(line);
  } else if (strcmp(token, "oss") == 0) {
    system->msld->oss=io_nextb(line);
  } else if (strcmp(token, "abf") == 0){
    system->msld->abf=io_nextb(line);
  } else if (strcmp(token, "meta") == 0){
    system->msld->meta=io_nextb(line);
  } else if (strcmp(token, "LE_k") == 0){
    system->msld->LE_k=io_nextf(line);
  } else if (strcmp(token, "LE_f_red") == 0){
    system->msld->LE_f_red=io_nextf(line);
  } else if (strcmp(token, "abf_oss") == 0){
    system->msld->abf_oss=io_nextb(line);
  } else if (strcmp(token, "abf_umbrella") == 0){
    system->msld->abf_umbrella=io_nextb(line);
  } else if (strcmp(token, "abf_unweighted") == 0){
    system->msld->abf_unweighted=io_nextb(line);
  } else if (strcmp(token, "oss_k") == 0){
    system->msld->oss_k=io_nextf(line);
  } else if (strcmp(token, "oss_bias_mag") == 0){
    system->msld->oss_bias_mag=io_nextf(line);
  } else if (strcmp(token, "meta_bias_mag") == 0){
    system->msld->meta_bias_mag=io_nextf(line);
  } else if (strcmp(token, "bias_mult") == 0){
    system->msld->bias_mult=io_nextf(line);
  } else if (strcmp(token, "temper") == 0){
    system->msld->temper=io_nextb(line);
  } else if (strcmp(token, "transition_tempering") == 0){
    system->msld->transition_tempering=io_nextb(line);
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
  } else if (strcmp(token, "abf_warmup_samples") == 0){
    system->msld->abf_warmup_samples=io_nexti(line);
  } else if (strcmp(token, "update_steps") == 0){
    system->msld->update_steps=io_nexti(line);
  } else if (strcmp(token, "update_fe") == 0) {
    system->msld->update_fe_surface=io_nextb(line);
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

  // L_LEUS storage
  if(!dLdT_d){
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
  }
  // Theta Fix (w/ L_LEUS)
  if(theta_slow_fix){
    if(!L_LEUS){
      printf("Cannot run slow_fix on theta without L_LEUS interpolation scheme!\nExiting...");
      exit(1);
    }
    if(siteCount>2){
      printf("Cannot do theta fix with multi-site systems!\nExiting...");
      exit(1);
    }
    if(!W_d) cudaMalloc(&W_d, sizeof(real));
    cudaMemset(W_d, 0, sizeof(real));
  }
  // Theta Slow Growth (w/ L_LEUS)
  if(theta_slow_fix){
    if(!L_LEUS){
      printf("Cannot run slow_fix on theta without L_LEUS interpolation scheme!\nExiting...");
      exit(1);
    }
    if(siteCount>2){
      printf("Cannot do theta growth with multi-site systems!\nExiting...");
      exit(1);
    }
    if(abs(theta_start+1)<1e-5 || abs(theta_end+1)<1e-5){ // too close to -1 defaults
      printf("Must at least set endpoint for slow growth!\nExiting...");
      exit(1);
    }
    theta_delta = (theta_end - theta_start) / system->run->nsteps;
    theta_current = theta_start;
    if(!W_d) cudaMalloc(&W_d, sizeof(real));
    cudaMemset(W_d, 0, sizeof(real));
    // Reset for next call (boolean theta_slow_fix gets reset once transform is complete)
    theta_start=-1;
    theta_end=-1;
  }

  // Of total # of steps, use 1/4th to develop bias as default
  if (update_steps == -1){
    update_steps = .25*system->run->nsteps;
  }
  // Allocate memory for enhanced sampling algorithms
  if ((oss || LE || abf || meta) && !theta_temp_d){ // don't reallocate if done already
    if(!system->msld->L_LEUS){
      printf("Enhanced sampling cannot be run without L-LEUS theta dynamics scheme!\n");
      exit(1);
    }
    init_enhanced_sampling(system);
    restart_success = false; 
    if (restartable) restart_success = read_histogram_file(system, "hist_restart.txt");
    if (!restart_success && restartable) {
      printf("Restart failed!\nExiting...");
      exit(1);
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

void Msld::init_enhanced_sampling(System* system){
  // General Memory
  theta_temp = (real*)malloc(siteCount*sizeof(real));
  cudaMalloc(&theta_temp_d, siteCount*sizeof(real)); 
  cudaMemset(theta_temp_d, 0, siteCount*sizeof(real)); 

  // Potential
  oss_bias = (real*)malloc(siteCount*sizeof(real));
  cudaMalloc(&oss_bias_d, siteCount*sizeof(real)); 
  cudaMemset(oss_bias_d, 0, siteCount*sizeof(real)); 
  abf_bias = (real*)malloc(siteCount*sizeof(real));
  cudaMalloc(&abf_bias_d, siteCount*sizeof(real)); 
  cudaMemset(abf_bias_d, 0, siteCount*sizeof(real)); 
  LE_bias = (real*)malloc(siteCount*sizeof(real));
  cudaMalloc(&LE_bias_d, siteCount*sizeof(real)); 
  cudaMemset(LE_bias_d, 0, siteCount*sizeof(real)); 
  meta_bias = (real*)malloc(siteCount*sizeof(real));
  cudaMalloc(&meta_bias_d, siteCount*sizeof(real)); 
  cudaMemset(meta_bias_d, 0, siteCount*sizeof(real)); 
  // Forces
  dUdT_msld = (real*)malloc(blockCount*sizeof(real));
  cudaMalloc(&dUdT_msld_d, blockCount*sizeof(real));
  cudaMemset(dUdT_msld_d, 0, blockCount*sizeof(real));
  dUdL_msld = (real*)malloc(blockCount*sizeof(real));
  cudaMalloc(&dUdL_msld_d, blockCount*sizeof(real));
  cudaMemset(dUdL_msld_d, 0, blockCount*sizeof(real));
  dUdT_abf = (real*)malloc(sizeof(real));
  cudaMalloc(&dUdT_abf_d, siteCount*sizeof(real)); 
  cudaMemset(dUdT_abf_d, 0, siteCount*sizeof(real));
  cudaMalloc(&dGdF_d, blockCount*sizeof(real)); 
  cudaMemset(dGdF_d, 0, blockCount*sizeof(real));

  // OSS
  T_res = T_std / bins_per_std;
  dUdT_res = dUdT_std / bins_per_std;
  T_search = (int)(n_std_search*T_std/T_res); // 10 by default
  dUdT_search = (int)(n_std_search*dUdT_std/dUdT_res); // 10 by default
  if((2*T_search+1)*(2*dUdT_search+1) > 1024){
    printf("Please decrease n_std_search or bins_per_std to decrease threads required per histogram evaluation!");
    exit(1);
  }

  dUdT_min = -dUdT_max;
  dUdT_bins = (int) ((abs(dUdT_max) + abs(dUdT_min)) / dUdT_res) + 1; // histogram has bins exactly on edges
  total_T_bins = 0;
  T_bins = (int*) calloc(siteCount, sizeof(int));
  LE_total_bins = 0;
  LE_bins = (int*) calloc(siteCount, sizeof(int));
  for(int i = 1; i < siteCount; i++){
    T_bins[i] = (int) (site_period[i] / T_res); // still consider last bin as separate
    total_T_bins += T_bins[i]; // account for periodicity, first and last bins are identical 
    LE_bins[i] = (int) (site_period[i] / LE_T_res); 
    LE_total_bins += LE_bins[i];
  }
  cudaMalloc(&T_bins_d, siteCount*sizeof(int));
  cudaMemcpy(T_bins_d, T_bins, siteCount*sizeof(int), cudaMemcpyDefault);
  cudaMalloc(&LE_bins_d, siteCount*sizeof(int));
  cudaMemcpy(LE_bins_d, LE_bins, siteCount*sizeof(int), cudaMemcpyDefault);

  cudaMalloc(&theta_histogram_d, total_T_bins*sizeof(real)); 
  cudaMemset(theta_histogram_d, 0, total_T_bins*sizeof(real)); 

  cudaMalloc(&histogram_2D_d, total_T_bins*dUdT_bins*sizeof(real));
  cudaMemset(histogram_2D_d, 0, total_T_bins*dUdT_bins*sizeof(real));

  cudaMalloc(&potential_2D_d, total_T_bins*dUdT_bins*sizeof(real));
  cudaMemset(potential_2D_d, 0, total_T_bins*dUdT_bins*sizeof(real));

  cudaMalloc(&max_pot_d, total_T_bins*sizeof(real));
  cudaMemset(max_pot_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&oss_min_V_d, sizeof(real));
  cudaMemset(oss_min_V_d, 0, sizeof(real));

  // ABF
  cudaMalloc(&abf_ensemble_dUdT_d, total_T_bins*sizeof(real));
  cudaMemset(abf_ensemble_dUdT_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&abf_variance_dUdT_d, total_T_bins*sizeof(real));
  cudaMemset(abf_variance_dUdT_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&abf_offsets_d, total_T_bins*sizeof(real));
  cudaMemset(abf_offsets_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&abf_weights_d, total_T_bins*sizeof(real));
  cudaMemset(abf_weights_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&abf_weighted_dUdT_d, total_T_bins*sizeof(real));
  cudaMemset(abf_weighted_dUdT_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&abf_weighted_dUdT2_d, total_T_bins*sizeof(real));
  cudaMemset(abf_weighted_dUdT2_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&abf_ensemble_dUdT_d, total_T_bins*sizeof(real));
  cudaMemset(abf_ensemble_dUdT_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&abf_bias_d, siteCount*sizeof(real));
  cudaMemset(abf_bias_d, 0, siteCount*sizeof(real));

  // Speed up <dU/dT> calculations for abf_oss
  cudaMalloc(&max_dUdT_index_d, total_T_bins*sizeof(int));
  cudaMemset(max_dUdT_index_d, 0, total_T_bins*sizeof(int));

  int min[total_T_bins];
  for(int i = 0; i < total_T_bins; i++){ min[i] = dUdT_bins-1; }
  cudaMalloc(&min_dUdT_index_d, total_T_bins*sizeof(int));
  cudaMemcpy(min_dUdT_index_d, min, total_T_bins*sizeof(int), cudaMemcpyDefault);

  // Meta
  cudaMalloc(&histogram_1D_d, total_T_bins*sizeof(real));
  cudaMemset(histogram_1D_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&potential_1D_d, total_T_bins*sizeof(real));
  cudaMemset(potential_1D_d, 0, total_T_bins*sizeof(real));

  cudaMalloc(&meta_bias_d, siteCount*sizeof(real));
  cudaMemset(meta_bias_d, 0, siteCount*sizeof(real));

  cudaMalloc(&meta_min_V_d, sizeof(real));
  cudaMemset(meta_min_V_d, 0, sizeof(real));

  // Local Elevation
  cudaMalloc(&LE_M_d, LE_total_bins*sizeof(real));
  cudaMemset(LE_M_d, 0, LE_total_bins*sizeof(real));

  cudaMalloc(&LE_theta_sweep_d, LE_total_bins*sizeof(int));
  cudaMemset(LE_theta_sweep_d, 0, LE_total_bins*sizeof(int));

  real R[siteCount];
  for(int i = 0; i < siteCount; i++){ R[i] = 1.0; }
  cudaMalloc(&LE_R_d, siteCount*sizeof(real));
  cudaMemcpy(LE_R_d, R, siteCount*sizeof(real), cudaMemcpyDefault);

  cudaMalloc(&LE_visited_bins_d, siteCount*sizeof(int));
  cudaMemset(LE_visited_bins_d, 0, siteCount*sizeof(int));
}

void Msld::destroy_enhanced_sampling(System* system){
  // TODO
}

void Msld::recv_meta(){
  cudaMemcpy(oss_bias, oss_bias_d, siteCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(meta_bias, meta_bias_d, siteCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdL_msld, dUdL_msld_d, blockCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdT_msld, dUdT_msld_d, blockCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdT_abf, dUdT_abf_d, siteCount*sizeof(real), cudaMemcpyDefault);
}

void Msld::copy_reset_memory(System* system){
  Run* r = system->run;
  Msld* m = system->msld;
  // Copy MSLD theta and lambda forces, clear memory from last step
  cudaMemcpyAsync(m->dUdL_msld_d, system->state->lambdaForce_d, m->blockCount*sizeof(real), cudaMemcpyDefault, r->ossBias);
  cudaMemsetAsync(m->dUdT_msld_d, 0, m->blockCount*sizeof(real), r->ossBias);
  oss_lambda_to_theta_force(system); // fill dUdT_msld_d
  cudaMemsetAsync(m->oss_bias_d, 0, m->siteCount*sizeof(real), r->ossBias);
  cudaMemsetAsync(m->meta_bias_d, 0, m->siteCount*sizeof(real), r->ossBias);
  cudaMemsetAsync(m->abf_bias_d, 0, m->siteCount*sizeof(real), r->ossBias);
  cudaMemsetAsync(m->LE_bias_d, 0, m->siteCount*sizeof(real), r->ossBias);
  cudaMemsetAsync(m->dGdF_d, 0, m->blockCount*sizeof(real), r->ossBias);
}

void Msld::log_sampling(System* system, int step){
  State* s = system->state;
  Run* r = system->run;

  if(step % (log_freq) != 0 || step == 0){ 
    return;
  }

  real ensemble_dUdT[total_T_bins], dUdT_var[total_T_bins], counts[total_T_bins], hist_1D[total_T_bins];
  real dGdF[blockCount];
  real oss_ttemper, meta_ttemper;
  cudaMemcpy(counts, histogram_1D_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(ensemble_dUdT, abf_ensemble_dUdT_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdT_var, abf_variance_dUdT_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dGdF, dGdF_d, blockCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(hist_1D, potential_1D_d, total_T_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&oss_ttemper, oss_min_V_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&meta_ttemper, meta_min_V_d, sizeof(real), cudaMemcpyDefault);
  recv_meta(); // bias, dUdT_msld, dUdL_msld, d2UdT2, dUdT_abf

  real M[total_T_bins]; 
  real R[siteCount];
  int theta_sweep[total_T_bins];
  int visited_bins[siteCount];
  cudaMemcpy(M, LE_M_d, LE_total_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(theta_sweep, LE_theta_sweep_d, LE_total_bins*sizeof(int), cudaMemcpyDefault);
  cudaMemcpy(visited_bins, LE_visited_bins_d, siteCount*sizeof(int), cudaMemcpyDefault);
  cudaMemcpy(R, LE_R_d, siteCount*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(theta_temp, theta_temp_d, siteCount*sizeof(real), cudaMemcpyDefault);
  system->state->recv_state();
  system->state->recv_lambda();

  int prev_subs = 1; // num of alchemical blocks already logged
  int prev_bins = 0;
  int LE_prev_bins = 0;
  int start = 0;
  for (int site = 1; site < siteCount; site++) { // Skip environment site
    printf("Step: %ld\n", r->step);
    printf("Site: %d, Theta: %f, Theta Temp: %f, Lambdas: [", site, system->state->theta[prev_subs], theta_temp[site] / samples);
    for(int i = prev_subs; i < prev_subs+blocksPerSite[site]; i++){ printf(" %f,", system->state->lambda[i]); }
    printf("] \n");
    real period = transition_w + plateau_w;
    printf("Sub Period: %f, Period: %f, T_bins: %d\n", period, site_period[site], T_bins[site]);
    real U = max(oss_bias[site] - temper_offset, 0.0) / (system->state->leapParms1->kT*temper_amount);
    real U_t = max(oss_ttemper - temper_offset, 0.0) / (system->state->leapParms1->kT*temper_amount);
    printf("OSS Bias: %f, OSS Tempering: %f, OSS Transition Tempering: %f, dGdF: [", oss_bias[site], exp(-U), exp(-U_t));
    for(int i = prev_subs; i < prev_subs+blocksPerSite[site]; i++){ printf(" %f,", dGdF[i]); }
    printf("] \n");
    U = max(meta_bias[site] - temper_offset, 0.0) / (system->state->leapParms1->kT*temper_amount);
    U_t = max(meta_ttemper - temper_offset, 0.0) / (system->state->leapParms1->kT*temper_amount);
    printf("Meta Bias: %f, Meta Tempering: %f, Meta Transition Tempering: %f, dMdL: TODO\n", meta_bias[site], exp(-U/temper_amount), U_t);
    printf("dUdT_abf: %f\n", dUdT_abf[site]);
    real rel[blocksPerSite[site]], var[blocksPerSite[site]];
    real dG = 0;
    real dG_var = 0;
    int count = 1;
    printf("TI dG i->i+1: [");
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
    printf("TI dG 0->i: [");
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
    printf("Site R: %f, k_LE*LE_f_red^R: %f\n", R[site], LE_k*pow(LE_f_red, R[site]));
    printf("M: [");
    for(int i = LE_prev_bins; i < LE_prev_bins + LE_bins[site]; i++){
      printf(" %f,", M[i]);
    }
    printf("] \n");
    printf("Visited Bins: %d, Sweep: [", visited_bins[site]);
    for(int i = LE_prev_bins; i < LE_prev_bins + LE_bins[site]; i++){
      printf(" %d,", theta_sweep[i]);
    }
    printf("] \n");
    LE_prev_bins += LE_bins[site];
    printf("\n");
    prev_subs += blocksPerSite[site];
    prev_bins += T_bins[site];
  }
}

/** 
 * Histogram defined to have elements for the first and last elements at max and min respectively. Width of the
 * first and last bins are half length. num_bins-1 whole bins fit in the [min, max) range. Uniform histogram.
 * Element is relative to min.
 * 
 * Safe because it won't index beyond what num_bins specifies.
 * 
 * Ex. Lambda bins: [0, .005), [.005, .015), ..., [.985, .995), [.995, 1.0 (0)) -> range [0,1), 101 bins
 *  Lambda Centers: [   0   ], [   .010   ], ..., [   .990   ], [   1.0   ]
 *    Lambda Index: [   0   ], [     1    ], ..., [    99    ], [   0 (100)   ] -> last is num_bins-2
 * 
 * Periodic just ties the first and last index to be the same value, so maximum this should resturn is num_bins-2.
*/
static __device__ int periodic_histogram_index(real val, int num_bins, real max, real min){
  real tmp = val - min;
  real range = max - min;
  real resolution = range / num_bins;
  int index = round(tmp/resolution);
  if(index >= num_bins){
    index -= num_bins;
  }
  if(index < 0){
    index += num_bins;
  }
  return index;
}

// This histogram index is used for non-periodic directions, where there exist bins on both edges of the period
static __device__ int histogram_index(real val, int num_bins, real max, real min){
  real tmp = val - min;
  real range = max - min;
  real resolution = range / (num_bins-1);
  return round(tmp/resolution);
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

__global__ void theta_temp_kernel(
  int nSite, int* blocksPerSite,
  // input
  real* theta_inv_sqrt_mass, real_v* theta_velocities, 
  // output
  real* theta_temps
){
  int i = blockIdx.x*blockDim.x+threadIdx.x + 1;
  if(i < nSite){
    int theta_id = 0;
    for(int j = 0; j < i; j++){
      theta_id += blocksPerSite[j];
    }
    real v_sqrtm = theta_velocities[theta_id] / theta_inv_sqrt_mass[theta_id];
    theta_temps[i] += v_sqrtm*v_sqrtm/kB;
  }
}

__global__ void add_sample_hist_kernel(
  real kT, int siteCount, int* blocksPerSite,
  int* T_bins, int* LE_bins, real* site_period,
  int dUdT_bins, real dUdT_max, real dUdT_min,
  real transition_w, real plateau_w, 
  // Input
  bool temper, bool transition_temper,
  bool abf_umbrella, bool abf_unweighted,
  real* thetas, real* dUdT_msld, 
  real tempering, real temper_offset, 
  real* oss_bias, real* meta_bias, 
  real* oss_max_potential, real* meta_potential_1D,
  real warmup_samples,
  real k_LE, real LE_f_red, 
  // Output - Meta
  real* theta_counts, real* histogram_2D, real* histogram_1D, 
  int* dUdT_sampled_max, int* dUdT_sampled_min,
  real* oss_min_V, real* meta_min_V,
  // Output - ABF
  real* weights, real* offsets, 
  real* weighted_dUdT, real* weighted_dUdT2, 
  real* ensemble_dUdT, real* variance_dUdT,
  // Output - LE
  real* R, int* visited_bins, 
  int* theta_sweep, real* M 
){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < siteCount && i != 0){ // none for environment
    int start_site = 0;
    int theta_start = 0;
    int LE_theta_start = 0;
    for(int j = 0; j < i; j++){
      start_site += blocksPerSite[j];
      theta_start += T_bins[j];
      LE_theta_start += LE_bins[j];
    }

    // Add sample to 1D and 2D Meta on (T, dU/dT)
    real T = thetas[start_site];
    real dUdT = 0;
    for(int j = 0; j < blocksPerSite[i]; j++){
      dUdT += dUdT_msld[start_site+j];
    }
    int X = periodic_histogram_index(T, T_bins[i], site_period[i], 0);
    int Y = histogram_index(dUdT, dUdT_bins, dUdT_max, dUdT_min);
    real T_res = site_period[i] / T_bins[i];
    real dUdT_res = (dUdT_max - dUdT_min) / (dUdT_bins-1.0);
    //printf("i: %d, T: %f, dU/dT: %f, X: %d, Y: %d -> (%f, %f), T_bins[i]: %d\n", i, T, dUdT, theta_start+X, Y, X*T_res, Y*dUdT_res, T_bins[i]);
    if(X >= 0 && X < T_bins[i] && Y >= 0 && Y <= dUdT_bins-1){
      theta_counts[theta_start + X] += 1.0;
      // Transition tempering
      oss_min_V[0] = 1e9;
      meta_min_V[0] = 1e9;
      for(int j = theta_start; j < theta_start + T_bins[i]; j++){
        if (oss_max_potential[j] < oss_min_V[0]){
          oss_min_V[0] = oss_max_potential[j];
        } 
        if (meta_potential_1D[j] < meta_min_V[0]){
          meta_min_V[0] = meta_potential_1D[j];
        }
      }
      // Meta
      real bias = transition_temper ? max(meta_min_V[0]-temper_offset, 0.0) : max(meta_bias[i]-temper_offset, 0.0);
      real decay = temper ? exp(-bias/(tempering*kT)) : 1.0;
      //printf("T: %f, start+X: %d, bias: %f, decay: %f, hist: %f\n", T, theta_start+X, meta_bias[i], decay, histogram_1D[theta_start+X]);
      histogram_1D[theta_start + X] += decay;
      // OSS
      bias = transition_temper ? max(oss_min_V[0]-temper_offset, 0.0) : max(oss_bias[i]-temper_offset, 0.0);
      decay = temper ? exp(-bias/(tempering*kT)) : 1.0;
      int hist_index = (theta_start+X)*dUdT_bins + Y;
      histogram_2D[hist_index] += decay;
      dUdT_sampled_max[theta_start+X] = Y > dUdT_sampled_max[theta_start+X] ? Y : dUdT_sampled_max[theta_start+X];
      dUdT_sampled_min[theta_start+X] = Y < dUdT_sampled_min[theta_start+X] ? Y : dUdT_sampled_min[theta_start+X];
    } 

    // Add sample to ABF for abf_umbrella or abf_unweighted options (abf_oss overwrites both if on)
    real weight = 1.0;
    if(abf_umbrella){
      real temper_correction = transition_temper ? 1.0 : (1+tempering)/tempering;
      real bias = temper_correction*(meta_bias[i] + oss_bias[i])/kT;
      if(bias > offsets[theta_start+X]){ // offsets make largest weight = 1
        real correction = exp(offsets[theta_start+X] - bias);
        weights[theta_start+X] *= correction;
        weighted_dUdT[theta_start+X] *= correction;
        weighted_dUdT2[theta_start+X] *= correction;
        offsets[theta_start+X] = bias;
      }
      weight = exp(bias-offsets[theta_start+X]);
    }
    if(abf_unweighted || abf_umbrella){
      weights[theta_start+X] += weight;
      weighted_dUdT[theta_start+X] += dUdT*weight;
      weighted_dUdT2[theta_start+X] += dUdT*dUdT*weight;
      ensemble_dUdT[theta_start+X] = weighted_dUdT[theta_start+X] / weights[theta_start+X];
      real ensemble_dUdT2 = weighted_dUdT2[theta_start+X] / weights[theta_start+X];
      if(theta_counts[theta_start+X] < .5*warmup_samples){
        real ramp = 0;
        ensemble_dUdT[theta_start+X] *= ramp;
        ensemble_dUdT2 *= ramp;
      } else if(theta_counts[theta_start+X] < warmup_samples){
        real ramp = 2.0*theta_counts[theta_start + X] / warmup_samples - 1.0;
        ensemble_dUdT[theta_start+X] *= ramp;
        ensemble_dUdT2 *= ramp;
      }
      variance_dUdT[theta_start+X] = ensemble_dUdT2 - ensemble_dUdT[theta_start+X];
    }

    // Add sample to 1D Local Elevation
    X = periodic_histogram_index(T, LE_bins[i], site_period[i], 0);
    //printf("T: %f, X: %d -> T: %f, LE_bins[i]: %d, \n", T, theta_start+X, X*(site_period[i]/LE_bins[i]), LE_bins[i]);
    // First and last bin are identical centers, don't include last, account for that in visited check
    if(X >= 0 && X < LE_bins[i]){ 
      // Order of these checks matters
      if (visited_bins[i] >= 2*LE_bins[i]) { // Double sweep complete, increment and clear mem
        R[i] += 1;
        for(int j = LE_theta_start; j < LE_theta_start + LE_bins[i]; j++){
          theta_sweep[j] = 0;
        }
        visited_bins[i] = 0;
      }
      if (theta_sweep[LE_theta_start + X] == 1 && visited_bins[i] >= LE_bins[i]){ // New bin sample on second sweep 
        theta_sweep[LE_theta_start + X] = 2;
        visited_bins[i] += 1;
      }
      if (theta_sweep[LE_theta_start + X] == 0){ // New bin sample on first sweep
        theta_sweep[LE_theta_start + X] = 1;
        visited_bins[i] += 1;
      }
      M[LE_theta_start + X] += k_LE*pow(LE_f_red, R[i]);
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
    samples++;
    add_sample_hist_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
      kT, siteCount, blocksPerSite_d,
      T_bins_d, LE_bins_d, site_period_d, 
      dUdT_bins, dUdT_max, dUdT_min,
      transition_w, plateau_w,
      // Input
      temper, transition_tempering,
      abf_umbrella, abf_unweighted,
      s->theta_fd, dUdT_msld_d, 
      temper_amount, temper_offset, 
      oss_bias_d, meta_bias_d,
      max_pot_d, potential_1D_d,
      abf_warmup_samples,
      LE_k, LE_f_red, 
      // Output - OSS
      theta_histogram_d,
      histogram_2D_d, histogram_1D_d,
      max_dUdT_index_d, min_dUdT_index_d, 
      oss_min_V_d, meta_min_V_d,
      // Output - ABF
      abf_weights_d, abf_offsets_d,
      abf_weighted_dUdT_d, abf_weighted_dUdT2_d, 
      abf_ensemble_dUdT_d, abf_variance_dUdT_d,
      // Output - LE
      LE_R_d, LE_visited_bins_d,
      LE_theta_sweep_d, LE_M_d);

    theta_temp_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
      siteCount, blocksPerSite_d,
      s->thetaInvsqrtMass_d, s->thetaVelocity_d,
      theta_temp_d);

    // updates potential with new sample (only where affected) & calculates <dU/dT> if abf_oss is on
    update_ABF_from_hist(system, 2*T_search+1, 2*dUdT_search+1, true);
  }

  if(step % write_freq == 0){ // write files every 1k samples (1k-10k steps or 1-20ps)
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
  real oss_k, 
  real* thetas, real* ensemble_dUdT, real* dUdT_msld, 
  // Outputs
  real_f* thetaForce, real* dUdT_abf,
  real_e* energy
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
    int bin = periodic_histogram_index(theta, T_bins[i], site_period[i], 0);
    // Lerp implementation of ABF along theta
    real res = site_period[i]/T_bins[i]; 
    real bin_center = bin*res;
    int hist_bin = prev_bins + bin;
    real dUdT = ensemble_dUdT[hist_bin];
    real dist = theta-bin_center;
    // PBC on T
    if(abs(dist) > site_period[i]/2.0){ // only the positive condition hits
      dist += dist > 0 ? -site_period[i] : site_period[i];
    }
    real partner_center, partner_dUdT, interp;
    real abf = 0.0;
    if(dist > 0){ // in upper half of bin
      partner_center = bin_center + res;
      if(bin + 1 >= T_bins[i]){
        hist_bin -= T_bins[i]; // next bin should be zero
      }
      partner_dUdT = ensemble_dUdT[hist_bin+1];
      interp = dist / res;
      abf = (1 - interp)*dUdT + interp*partner_dUdT;
    } else { // in lower half of bin
      partner_center = bin_center - res;
      if(bin - 1 < 0){ 
        hist_bin += T_bins[i]; // previous bin should be T_bins[i]-1
      }
      partner_dUdT = ensemble_dUdT[hist_bin-1];
      interp = (theta - partner_center) / res;
      abf = (1 - interp)*partner_dUdT + interp*dUdT;
    }
    // add -'ve
    abf = -abf;
    atomicAdd(&thetaForce[start], abf);
    dUdT_abf[i] = abf;
  }
}

__global__ void getforce_hist_kernel(
  int siteCount, int* blocksPerSite, 
  int* T_bins, real* site_period, 
  int dUdT_bins, real dUdT_max, real dUdT_min,
  real T_std, real dUdT_std,
  int T_search, int dUdT_search,
  real oss_weight, real meta_weight, real bias_mult,
  bool relative_indexing, // center on current (T, dUdT) or (0, 0)
  int vert_slices, int horz_slices,
  // Inputs
  bool energy, 
  real* thetas, real* dUdT_msld, real* histogram_2D, real* histogram_1D,
  // Outputs
  real* thetaForce, real* dGdF, 
  real* oss_bias, real* meta_bias,
  real* potential_2D, real* potential_1D
) {
  // Grid: Potential/Force evaluation at this grid point (if relative is on)
  int iSite = blockIdx.x + 1;  // Skip site 0
  int iT = blockIdx.y;         
  int idUdT = blockIdx.z;      
  // Thread: Eval points in square around grid point
  int iSearch_T = threadIdx.x;    // T search offset
  int iSearch_dUdT = threadIdx.y; // dUdT search offset
  if (iSite >= siteCount) return;
  if (energy && (iT >= vert_slices || idUdT >= horz_slices)) return; // kill overallocated CUDA grids
  
  int start_site = 0;
  int theta_start = 0;
  for(int j = 0; j < iSite; j++){
    start_site += blocksPerSite[j];
    theta_start += T_bins[j];
  }
  // Current (T, dUdT)
  real T = thetas[start_site];
  real dUdT = 0;
  for(int j = 0; j < blocksPerSite[iSite]; j++){
    dUdT += dUdT_msld[start_site+j];
  }
  if(!relative_indexing){ // Used when evaluating entire histogram
    T = 0;
    dUdT = 0;
  }
  real T_resolution = site_period[iSite]/T_bins[iSite];
  real dUdT_resolution = (dUdT_max-dUdT_min)/(dUdT_bins-1.0);
  if (energy) { // Re-center 
    // Offset Index -> new (T, dU/dT) center
    int T_index = periodic_histogram_index(T, T_bins[iSite], site_period[iSite], 0);
    T_index += -floor(vert_slices/2.0) + iT;
    T = T_index * T_resolution; 
    if (T < 0) { T += site_period[iSite]; } 
    if (T >= site_period[iSite]) { T -= site_period[iSite]; } 
    int dUdT_index = histogram_index(dUdT, dUdT_bins, dUdT_max, dUdT_min);
    dUdT_index += -floor(horz_slices/2.0) + idUdT;
    dUdT = dUdT_min + dUdT_index*dUdT_resolution;
  }
  
  // CUDA Grid: Center point on grid close to (T, dUdT) where we evalutate potential
  int X = periodic_histogram_index(T, T_bins[iSite], site_period[iSite], 0);
  int Y = histogram_index(dUdT, dUdT_bins, dUdT_max, dUdT_min);
  // A single thread from each CUDA grid sets potential to zero 
  if(energy && iSearch_T == 0 && iSearch_dUdT == 0){ 
    int output_index = (theta_start + X)*dUdT_bins + Y;
    potential_2D[output_index] = 0;
    // Only 1 CUDA grid per vertical slice needs to do this
    if(idUdT == 0){ 
      potential_1D[theta_start+X] = 0;
    }
  }
  __syncthreads();
  // CUDA Threads: search square region around center point
  int j = X - T_search + iSearch_T; 
  int true_j = j; 
  if (j < 0) { j += T_bins[iSite]; }  
  if (j >= T_bins[iSite]) { j -= T_bins[iSite]; } 
  int k = Y - dUdT_search + iSearch_dUdT;
  if (k >= 0 && k < dUdT_bins) {
    int T_index = j + theta_start; // shift into correct histogram for site
    int dUdL_index = k;
    
    real T_center = true_j * T_resolution; 
    real T_distance = (T - T_center);
    // PBC on T
    if(abs(T_distance) > site_period[iSite]/2.0){
      T_distance += T_distance > 0 ? -site_period[iSite] : site_period[iSite];
    }
    T_distance /= T_std;
    real T_gaussian = expf(-0.5*T_distance*T_distance);
    real dUdT_center = dUdT_min + k*dUdT_resolution;
    real dUdT_distance = (dUdT - dUdT_center) / dUdT_std;
    real dUdT_gaussian = expf(-0.5*dUdT_distance*dUdT_distance);
    
    int hist_index = T_index*dUdT_bins + dUdL_index;
    real oss = oss_weight * T_gaussian * dUdT_gaussian * histogram_2D[hist_index];
    real meta = meta_weight * T_gaussian * histogram_1D[T_index];
    if (energy) {
      int output_index = (theta_start + X)*dUdT_bins + Y; // output to CUDA grid point
      atomicAdd(&potential_2D[output_index], oss);
      // Energy: Many thread blocks per site hist, only 1 slice of 1 block does meta calc
      if(iSearch_dUdT == dUdT_search && idUdT == dUdT_search){
        atomicAdd(&potential_1D[theta_start+X], meta);
      }
    } else {
      real local_dUdT_force = -bias_mult*dUdT_distance/dUdT_std * oss;
      real local_T_force = -bias_mult*T_distance/T_std * oss;
      for(int l = 0; l < blocksPerSite[iSite]; l++){
        atomicAdd(&dGdF[start_site + l], local_dUdT_force);
      }
      atomicAdd(&thetaForce[start_site], local_T_force);
      atomicAdd(&oss_bias[iSite], oss);
      // Force: One thread block per site hist, only 1 slice does meta
      if(iSearch_dUdT == dUdT_search && idUdT == 0){
        local_T_force = -bias_mult*T_distance/T_std * meta;
        atomicAdd(&thetaForce[start_site], local_T_force);
        atomicAdd(&meta_bias[iSite], meta);
      }
    }
  }
  // TODO: Add reductions to avoid crowded atomic adds
}

__global__ void getforce_oss_linear(
  int blockCount,
  // Inputs
  real oss_k, real* dUdT_msld,
  // Outputs
  real* dGdF, real_e* energy
){
  int i=blockIdx.x*blockDim.x+threadIdx.x; // block
  if(i < blockCount && i != 0){
    dGdF[i] += oss_k;
    if(energy) atomicAdd(energy, oss_k*dUdT_msld[i]);
  }
}

// Out is a len=2 array for bias and derivative w.r.t. theta
__device__ void LE_bias(real theta, real* M, int bins, real period, real* out){
  out[0] = 0;
  out[1] = 1;
  real res = period / bins;
  int center_bin = periodic_histogram_index(theta, bins, period, 0); 
  for(int j = center_bin - 2; j <= center_bin + 2; j++){ 
      real bin_center = j*res;
      real dist = theta - bin_center; 
      // PBC on theta
      if(abs(dist) > period/2.0){ 
        dist += dist > 0 ? -period : period;
      }
      dist /= res; // normalize so delta res are 1
      // Compute cubic b-spline
      real t = abs(dist);
      real dtdx = dist > 0 ? 1/res : -1/res; // derivative at zero is zero anyway, so this doesn't matter
      real b_spline = 0;
      real db_spline = 0;
      if(t >= 1.0 && t <= 2.0){
        real tmp = 2.0-t;
        b_spline = tmp*tmp*tmp/4.0; 
        db_spline = (-3.0*tmp*tmp/4.0)*dtdx;
      } else if (t < 1.0){
        b_spline = (4.0+3.0*t*t*t-6.0*t*t)/4.0;
        db_spline = (9.0*t*t-12.0*t)/4.0*dtdx;
      }
      // Wrap with period
      int bin = j;
      if(j < 0){ bin += bins; }
      if(j >= bins){ bin -= bins; }
      out[0] += M[bin]*b_spline;
      out[1] += M[bin]*db_spline;
    }
}

__global__ void getforce_LE_kernel(
  int siteCount, int* blocksPerSite, 
  int* LE_bins, real* site_period,
  real transition_w, real plateau_w,
  // Inputs
  real* thetas, real* M,
  // Outputs,
  real* thetaForce, real_e* energy){
  int i=blockIdx.x*blockDim.x+threadIdx.x; // site
  if(i < siteCount && i != 0){
    // Count to first sub index
    int start = 0;
    int prev_bins = 0;
    for(int j = 0; j < i; j++){ 
      start += blocksPerSite[j];
      prev_bins += LE_bins[j];
    }
    real theta = thetas[start];
    // Local dot product of cubic b splines centered at M grid points evaluated at T
    real output[2]; // bias, dbdt
    int dv = theta / (transition_w + plateau_w); // transition number
    real md = fmod(theta, transition_w+plateau_w); // amount past beginning
    if(md >= 0 && md <= plateau_w){
      real start = dv*(transition_w + plateau_w);
      real end = start + plateau_w;
      real outA[2], outB[2];
      LE_bias(start, &M[prev_bins], LE_bins[i], site_period[i], outA);
      LE_bias(end, &M[prev_bins], LE_bins[i], site_period[i], outB);
      real m = (outB[0] - outA[0]) / (plateau_w); // slope
      output[0] = md*m + outA[0];
      output[1] = m;
    } else {
      LE_bias(theta, &M[prev_bins], LE_bins[i], site_period[i], output);
    }
    if(energy) atomicAdd(energy, output[0]);
    atomicAdd(&thetaForce[start], output[1]);
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
  if(restart_success){ // evaluate potential everywhere to get ABF (only once)
    restart_success=false;
    update_ABF_from_hist(system, total_T_bins, dUdT_bins, false);
  }

  if(LE){
    getforce_LE_kernel<<<(siteCount+BLMS-1)/BLMS, BLMS,shMem,stream>>>(
      siteCount, blocksPerSite_d,
      LE_bins_d, site_period_d,
      transition_w, plateau_w,
      // Inputs
      s->theta_fd, LE_M_d, 
      // Outputs
      s->thetaForce_d, pEnergy);
  }
  if (abf) {
    getforce_abf_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(
      siteCount, blocksPerSite_d, 
      T_bins_d, site_period_d, 
      // Inputs
      oss_k,
      s->theta_fd, abf_ensemble_dUdT_d, dUdT_msld_d, 
      // Outputs
      s->thetaForce_d, dUdT_abf_d, pEnergy);
  }
  // meta and oss bias mag need to get set
  if(oss || meta){
    getforce_oss_linear<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,stream>>>(blockCount, oss_k, dUdT_msld_d, dGdF_d, pEnergy);
    int vert_slices = 1;
    int horz_slices = 1;
    dim3 blockDim(2*T_search+1, 2*dUdT_search+1, 1); 
    dim3 gridDim(siteCount-1, 1, 1);
    getforce_hist_kernel<<<gridDim, blockDim, shMem, stream>>>(
      siteCount, blocksPerSite_d,
      T_bins_d, site_period_d, 
      dUdT_bins, dUdT_max, dUdT_min, 
      T_std, dUdT_std, T_search, dUdT_search, 
      oss_bias_mag, meta_bias_mag, bias_mult,
      true,
      vert_slices, horz_slices,
      // Inputs
      false,
      s->theta_fd, dUdT_msld_d, histogram_2D_d, histogram_1D_d,
      // Outputs
      s->thetaForce_d, dGdF_d, oss_bias_d, meta_bias_d,
      potential_2D_d, potential_1D_d);
  }
}

__global__ void hist_ensemble_dUdT_kernel(
    real kT, int nSite, real* thetas, int* siteBound,
    int* T_bins, real* site_periods,
    int dUdT_bins, real dUdT_max, real dUdT_min,
    int* dUdT_sampled_max, int* dUdT_sampled_min,
    int slice_count, real warmup_samples, 
    real temper_correction,
    // Inputs
    bool abf_oss,
    real* potential_grid, real* histogram, real* theta_counts,
    // Output
    real* ensemble_dUdT, real* variance_dUdT, real* max_potential
) {
  int site = threadIdx.x+1;
  int tid = blockIdx.x;
  if (site < nSite && tid < slice_count) { // each site
    real dUdT_res = (dUdT_max - dUdT_min) / (dUdT_bins - 1.0);
    real offset = 0.0;
    real weighted_dUdT = 0;
    real weighted_dUdT2 = 0;
    real Z = 0;
    real max_bias = 0;
    int start_1D = 0;
    for(int j = 0; j < site; j++){
      start_1D += T_bins[j];
    }
    int X = periodic_histogram_index(thetas[siteBound[site]], T_bins[site], site_periods[site], 0);
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
        real current_bias = temper_correction*potential_grid[grid_index]/kT;
        max_bias = current_bias > max_bias ? current_bias : max_bias;
        if(histogram[grid_index] < 1e-5 || !abf_oss){ continue; } // Don't include non-sampled regions
        if (current_bias > offset) {
          real correction = exp(offset - current_bias); 
          weighted_dUdT *= correction;
          weighted_dUdT2 *= correction;
          Z *= correction;
          offset = current_bias;
        }
        weighted_dUdT += dUdT * exp(current_bias - offset);
        weighted_dUdT2 += dUdT*dUdT*exp(current_bias - offset);
        Z += exp(current_bias - offset);
      }
      max_potential[start_1D+X] = max_bias*kT/temper_correction; // these things added back later if needed
      if(abf_oss){
        real scale = 1.0;
        if(theta_counts[start_1D+X] < .5*warmup_samples){
          scale = 0;
        } else if (theta_counts[start_1D+X] < warmup_samples){
          scale = 2.0*theta_counts[start_1D+X]/warmup_samples - 1.0;
        }
        ensemble_dUdT[start_1D+X] = Z > 1e-5 ? scale*weighted_dUdT/Z : 0.0;
        variance_dUdT[start_1D+X] = Z > 1e-5 ? scale*weighted_dUdT2/Z - ensemble_dUdT[start_1D+X] : 0.0;
      }
    } 
  }
}

// Compute potential and optional <dU/dT> from histogram for ABF for each site
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
    oss_bias_mag, meta_bias_mag, bias_mult,
    relative_indexing, vert_slices, horz_slices,
    // Inputs
    true,
    s->theta_fd, dUdT_msld_d, histogram_2D_d, histogram_1D_d,
    // Outputs
    s->thetaForce_d, dGdF_d, oss_bias_d, meta_bias_d,
    potential_2D_d, potential_1D_d);

  if(!abf_oss && !(oss && transition_tempering)){ return; } // don't need to do 2D grid <dU/dT> or min_potential
  real temper_correction = transition_tempering ? 1.0 : (1.0 + temper_amount)/temper_amount;

  // Compute <dU/dT> in region where potential got updated
  dim3 blockDim1(siteCount-1, 1, 1); 
  dim3 gridDim1(vert_slices, 1, 1);
  // relative_indexing doesn't really matter in X direction since it will get wrapped
  hist_ensemble_dUdT_kernel<<<gridDim1, blockDim1, 0, stream>>>(
    s->leapParms1->kT, siteCount, s->theta_fd, siteBound_d,
    T_bins_d, site_period_d, 
    dUdT_bins, dUdT_max, dUdT_min, 
    max_dUdT_index_d, min_dUdT_index_d, 
    vert_slices,
    abf_warmup_samples, 
    temper_correction,
    // Inputs
    abf_oss,
    potential_2D_d, histogram_2D_d, theta_histogram_d,
    // Outputs
    abf_ensemble_dUdT_d, abf_variance_dUdT_d, 
    max_pot_d);
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

__global__ void set_theta_kernel(int nL, real_x *theta, real_x theta_current, real* thetaForce, real* work){
  int i=blockIdx.x*blockDim.x+threadIdx.x+1;
  if(i < nL){
    theta[i] = theta_current;
    if(i == 1) {
      work[0] += thetaForce[1]; // only 1 site and L-LEUS
    }
  }
}

void Msld::calc_lambda_from_theta(cudaStream_t stream,System *system)
{
  State *s=system->state;
  if (!(fix || theta_slow_fix || theta_fix)) { // ffix
    calc_lambda_from_theta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,0,stream>>>(
      s->lambda_d,s->theta_d,siteCount,siteBound_d,fnex,
      L_LEUS, L_LEUS_function, xsin_n,
      plateau_w, transition_w,
      dLdT_d, d2LdT2_d);
  } else if (theta_slow_fix) { // only 2 sites exist
    real W;
    cudaMemcpyAsync(&W, W_d, sizeof(real), cudaMemcpyDefault, stream);
    theta_current += theta_delta;
    set_theta_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,0,stream>>>(blockCount-1, s->theta_d, theta_current, s->thetaForce_d, W_d);
    calc_lambda_from_theta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,0,stream>>>(
      s->lambda_d,s->theta_d,siteCount,siteBound_d,fnex,
      L_LEUS, L_LEUS_function, xsin_n,
      plateau_w, transition_w,
      dLdT_d, d2LdT2_d);
    if (system->run->step == system->run->nsteps-1){ 
      theta_slow_fix = false;
    }
  } else if (theta_fix){
    set_theta_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,0,stream>>>(blockCount-1, s->theta_d, theta_fix_value, s->thetaForce_d, W_d); 
    if (system->run->step == system->run->nsteps-1){ 
      theta_fix = false;
    }
  } else if (fix){
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
