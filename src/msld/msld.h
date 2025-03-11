#ifndef MSLD_MSLD_H
#define MSLD_MSLD_H

#include <vector>
#include <set>

#include "main/defines.h"

// Forward declarations
class System;

// CHARMM format: LDBV INDEX  I   J  CLASS  REF  CFORCE NPOWER
struct VariableBias {
  int i,j;
  int type;
  real l0;
  real k;
  int n;
};

class Msld {
public:
  int blockCount;
  int *atomBlock;
  int *lambdaSite;
  real *lambdaBias;
  real_x *theta;
  real_v *thetaVelocity;
  real *thetaMass;
  real *lambdaCharge;

  int *atomBlock_d;
  int *lambdaSite_d;
  real *lambdaBias_d;
  real *lambdaCharge_d;

  int siteCount;
  int *blocksPerSite;
  int *blocksPerSite_d;
  int *siteBound;
  int *siteBound_d;

  std::set<int> *atomsByBlock;

  int *rest;
  real restScaling;

  real gamma;
  real fnex;

  bool scaleTerms[6]; // bond,ureyb,angle,dihe,impr,cmap

  int variableBiasCount;
  std::vector<struct VariableBias> variableBias_tmp;
  struct VariableBias *variableBias;
  struct VariableBias *variableBias_d;

  // FE Estimation Variables -> need to be on/off before msld::init is called since no other init_XXX method call exists
  // Mem not allocated if not set since histogram so large
  bool update_fe_surface = true; // add samples to abf/meta/oss
  int sample_freq = 10;
  real* dGdF_d;
  real* dGdL_d;
  real* dU_msld_d;
  real* dG_imp_d; // -kbT*ln(p) where p is probability of a lambda bin uses 10 million samples
  real dG_imp_bins = 51;
  real* hist_potential_d; // [blockCount] potential from metadynamics
  real* step_force_d; // force from bias
  // Just in case we get ideas for later
  real L_max = 1.0;
  real L_min = 0.0;

  // Meta - uniform binning - use abf histogram?
  bool mirror_Lmin = false;
  bool mirror_Lmax = false;
  bool meta = false;
  int L_meta_bins = 501;
  real* meta_histogram_d;
  int* meta_index_d;

  // Histogram (2D meta) - uniform binning
  bool oss = false; // Perform Orthogonal Space Sampling force calculations
  bool oss_abf = false; // Use <dU/dL> from integration over histogram for ABF force
  int L_oss_bins = 501; // # of whole bins that fit in range [L_min, L_max]
  real* oss_ensemble_dUdL_d;
  real* oss_var_d;
  real* minL_maxdUdL_d; // tempering for each histogram
  real* oss_histogram_d; // stores sum of prefactors
  real* oss_potential_d; // same size as histogram
  int* oss_index_d; // index into lambda's histogram

  // Meta options
  bool temper = true;
  real tempering = 3.0; // constant for decay of bias magnitude
  real temper_min = 4.0; // add at least 4 kcal/mol (felt) bias for every l bin before tempering
  real gaussian_weight = .01;

  // Don't change?
  int dUdL_bins = 2001; // # of whole bins that fit in range [dUdL_min, dUdL_max]
  real dUdL_max = 1500;
  real dUdL_min = -500;
  real L_resolution = (abs(L_max)+abs(L_min))/L_oss_bins;
  real dUdL_resolution = (abs(dUdL_max)+abs(dUdL_min))/dUdL_bins;
  real L_std = 3*L_resolution;
  real dUdL_std = 3*dUdL_resolution;
  int L_search = 3.0*(L_std/L_resolution); // ~3 L std in each direction
  int dUdL_search = 3.0*(dUdL_std/dUdL_resolution); // ~3 dUdL std in each direction

  // ABF - uniform binning - separate from histogram estimation
  bool abf = false;
  bool tracking_only = false; // Don't apply ABF bias if this is true -> dominates abf & oss_abf flag
  int nFull = 10;
  int L_abf_bins = 51; // this is also the max index (51 leads to >.99 as last bin)
  int* abf_index_d; // index into abf histogram
  real* abf_histogram_d; // counts in bin -> also used for 1D meta
  real* ensemble_dUdL_d; // <dU/dL> from umbrella re-weighting of PBMetaD
  real* ensemble_dUdL2_d;
  real* ensemble_var_d;
  real* weights_d;
  real* partition_functions;
  real* partition_offsets;
  real* weighted_dUdL_d;
  real* weighted_dUdL2_d;
  real* offsets_d;
  real* average_dUdL_d;
  real* average_dUdL2_d;
  real* ave_var_d;

  int thetaCollBiasCount;
  real *kThetaCollBias;
  real *kThetaCollBias_d;
  real *nThetaCollBias;
  real *nThetaCollBias_d;
  int thetaIndeBiasCount;
  real *kThetaIndeBias;
  real *kThetaIndeBias_d;

  std::vector<Int2> softBonds;
  std::vector<std::vector<int> > atomRestraints;

  int atomRestraintCount;
  int *atomRestraintBounds;
  int *atomRestraintBounds_d;
  int *atomRestraintIdx;
  int *atomRestraintIdx_d;

  bool useSoftCore;
  bool useSoftCore14;
  int msldEwaldType; // 1=normal scaling 2=normal scaling squared self interactions 3=correct scaling

  real kRestraint;
  real kChargeRestraint;
  real softBondRadius;
  real softBondExponent;
  real softNotBondExponent;

  bool fix; // ffix

  Msld();
  ~Msld();

  bool check_soft(int *idx,int Nat);
  bool check_restrained(int atom);
  bool bonded_scaling(int *idx,int *siteBlock,int type,int Nat,int Nsc);
  void nonbonded_scaling(int *idx,int *siteBlock,int Nat);
  bool bond_scaling(int idx[2],int siteBlock[2]);
  bool ureyb_scaling(int idx[3],int siteBlock[2]);
  bool angle_scaling(int idx[3],int siteBlock[2]);
  bool dihe_scaling(int idx[4],int siteBlock[2]);
  bool impr_scaling(int idx[4],int siteBlock[2]);
  bool cmap_scaling(int idx[8],int siteBlock[3]);
  void nb14_scaling(int idx[2],int siteBlock[2]);
  // void nbex_scaling(int idx[2],int siteBlock[2]);
  bool nbex_scaling(int idx[2],int siteBlock[2]);
  void nbond_scaling(int idx[1],int siteBlock[1]);

  bool interacting(int i,int j);

  void initialize(System *system);

  void calc_lambda_from_theta(cudaStream_t stream,System *system);
  void init_lambda_from_theta(cudaStream_t stream,System *system);
  void calc_thetaForce_from_lambdaForce(cudaStream_t stream,System *system);
  void getforce_fixedBias(System *system,bool calcEnergy);
  void getforce_variableBias(System *system,bool calcEnergy);
  void getforce_thetaBias(System *system,bool calcEnergy);
  void getforce_atomRestraints(System *system,bool calcEnergy);
  void getforce_chargeRestraints(System *system,bool calcEnergy);

  // Implicit constraints entropy
  void calc_imp(System* system);
  void sub_imp_dGdL(System* system, cudaStream_t stream); // adds -dG_imp to dU_msld based on site

  // On the fly enhanced sampling
  void init_meta(System* system);
  void add_sample_meta(System *system);
  void get_force_meta(System* system, bool calcEnergy);

  void init_abf(System* system);
  void add_sample_abf(System *system);
  void getpotential_abf(System* system, real* potential_grid);
  void getforce_abf(System *system, bool calcEnergy);

  void init_oss(System* system);
  void add_sample_hist(System *system);
  void get_tempering_hist(System* system);
  void getpotential_hist(System* system);
  void getforce_hist(System *system, bool calcEnergy);
};

void parse_msld(char *line,System *system);

// Library functions
extern "C" {
  void blade_init_msld(System *system,int nblocks);
  void blade_dest_msld(System *system);
  void blade_add_msld_atomassignment(System *system,int atomIdx,int blockIdx);
  void blade_add_msld_initialconditions(System *system,int blockIdx,int siteIdx,double theta0,double thetaVelocity,double thetaMass,double fixBias,double blockCharge);
  void blade_add_msld_termscaling(System *system,int scaleBond,int scaleUrey,int scaleAngle,int scaleDihe,int scaleImpr,int scaleCmap);
  void blade_add_msld_flags(System *system,double gamma,double fnex,int useSoftCore,int useSoftCore14,int msldEwaldType,double kRestraint,double kChargeRestraint,double softBondRadius,double softBondExponent,double softNotBondExponent,int fix);
  void blade_add_msld_bias(System *system,int i,int j,int type,double l0,double k,int n);
  void blade_add_msld_thetacollbias(System *system,int sites,int i,double k,double n);
  void blade_add_msld_thetaindebias(System *system,int sites,int i,double k);
  void blade_add_msld_softbond(System *system,int i,int j);
  void blade_add_msld_atomrestraint(System *system);
  void blade_add_msld_atomrestraint_element(System *system,int i);
}

#endif
