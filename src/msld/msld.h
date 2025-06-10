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

  // FE Estimation Variables -> need to be on/off before msld::init is called
  bool update_fe_surface = true; // add samples to oss histogram
  bool tracking_only = false; // Collect samples in 1D & 2D, but don't use/calculate atomic/lambda forces
  int sample_freq = 10;
  int update_steps = -1; // timesteps of sampling while updating FE surface - default to 1/4th of simulation
  bool standard_tempering = true; // use exp(-max(0, hist_potential[i] - temper_min) / kT) vs. same thing with min_bias
  bool OSS_remove_bonded = true; // removes large fluctuations in dUdL around l=0 that can cause instabilities
  bool OSS_remove_abf = true; // remove abf force from oss_dUdL, hist density around zero if you don't
  bool ABF_remove_bonded = false; // Does not compute true free energy if true
  bool ABF_flatten_hist = false; // cancel average lambda forces from meta gaussians, makes algo very reliant on hist estimate
  real* dUdL_bonded_d; // [blockCount] lambda forces from bonds
  real* dUdL_alf_d; // [blockCount] lambda forces from alf
  real* dUdL_abf_d; // [blockCount] lambda force from ABF
  real* dUdL_msld_d; // [blockCount] lambda forces from force field 
  real* oss_dUdL_d; // [blockCount] lambda forces that OSS is interested in biasing (nonbonded)
  real* oss_dUdL; // [blockCount]

  // 1D histogramming and tracking of dU/dL distribution
  bool abf = false; // Subtract average dU/dL using
  int L_1D_bins = 51; // this is also the max index (51 leads to >.99 as last bin since first and last are half width)
  real* abf_TI_d; // [nL]
  real* dABF_dl_d; // [nL]
  real* histogram_1D_d; // [nL * L_1D_bins] counts in bin 
  real* average_dUdL_d; // [nL * L_1D_bins]
  real* average_dUdL2_d; // [nL * L_1D_bins]
  real* variance_dUdL_d; // [nL * L_1D_bins]
  real* weights_d; // [nL * L_1D_bins] sum_i (exp(bias_i - offset))
  real* weighted_dUdL_d; // [nL * L_1D_bins] sum_i (dUdL*exp(bias_i - offset)) 
  real* ensemble_dUdL_d; // [nL * L_1D_bins] sum_i (dUdL*exp(bias_i - offset)) / sum_i (exp(bias_i - offset))
  real* offsets_d; // [nL * L_1D_bins] offsets for each bin that cancels in ensemble_average

  real alpha = 1.0; // lambda^alpha scaling

  // Histogram (2D meta) - uniform binning - nL = blockCount-1
  bool oss = false; // Perform Orthogonal Space Sampling calculations
  real* oss_histogram_d; // [nL * L_oss_bins * dUdL_bins] stores sum of gaussian prefactors
  real* oss_potential_d; // [nL * L_oss_bins * dUdL_bins]
  real* dGdF_d; // [blockCount] Derivative of gaussians w.r.t. lambda force
  real* dGdL_d; // [blockCount] Derivative of gaussians w.r.t. lambda
  real* hist_potential_d; // [blockCount-1] potential from 2d metadynamics
  real* hist_potential; // [blockCount-1]
  real* min_bias_d; // [blockCount]

  // Grid & Meta Params (free means it is a free parameter)
  int L_oss_bins = 201; // free - # of whole bins that fit in range [L_min, L_max]
  int dUdL_bins = 2501; // free - # of whole bins that fit in range [dUdL_min, dUdL_max]
  real L_max = 1.0;
  real L_min = 0.0;
  real dUdL_max = 750; // free 
  real dUdL_min = -750; // free
  real L_resolution = (abs(L_max)+abs(L_min))/L_oss_bins;
  real dUdL_resolution = (abs(dUdL_max)+abs(dUdL_min))/dUdL_bins;
  real L_std = .01; // free
  real dUdL_std = 4.0; // free
  int L_search = 5.0*(L_std/L_resolution); 
  int dUdL_search = 5.0*(dUdL_std/dUdL_resolution); 
  bool temper = true; // Using defaults (2 kcal = .43, 4 kcal = .08, 6 kcal = .01)
  real tempering = 2.0; // free - exp(-max(0, pot - min) / kBT*tempering)
  real temper_min = 1.0; // free 
  bool mirror_Lmin = true; // free 
  bool mirror_Lmax = true; // free
  real gaussian_weight = .01; // free

  // GaMD Parameters - 3 Stages: [0, init), [init,equil), [equil, nStep)
  real* alchem_energy; // Internal energy of alchemical system
  real* alchem_energy_d;
  int init_steps=1000000;
  int equil_steps=init_steps+1000000;
  bool GaMD_total = false;
  bool GaMD_torsion = false;
  bool GaMD_alchem = false;
  bool GaMD_orth = false;
  bool GaMD_force = false; // calculate orth dGdF & add to dGdF array
  bool GaMD_low_threshold = true;
  const static int num_GaMD_stats = 7;
  const static int GaMD_modes = 4;
  int GaMD_samples = 0;
  double total_p_stats[num_GaMD_stats]; // [Vmin, Vmax, Vavg, Vstd, Vstd_max, E, k]
  double torsion_p_stats[num_GaMD_stats];
  double alchem_p_stats[num_GaMD_stats]; // Would prefer not do do this as it requires second codepath
  double* orth_p_stats; // Keep track of dU/dL
  real* GaMD_orth_boosts;
  real GaMD_bias_added[GaMD_modes]; // [dV_total, dV_torsion, dV_alchem]
  real* GaMD_torsion_force_d; // Force just due to torsions
  real* GaMD_alchem_force_d; // Force just due to alchemical non-bonded interactions (no overlap w/ torsion boost)

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

  // Remove alf from dUdL_msld, d
  void set_forces(System* system);

  // GaMD Functions
  void gamd_update(System* system, bool update_E_k);
  void gamd_reset(System* system);
  void getforce_gamd(System* system);
  void getforce_orth_GaMD(System* system);

  // OSS Functions
  void init_oss(System* system);
  void reset_1D(System* system);
  void add_sample(System *system, int step);
  void getpotential_hist(System* system);
  void getforce_hist(System *system, bool calcEnergy);
  void log_sampling(System *system, int step);
  void get_tempering_hist(System* system);

  // ABF Functions
  void getforce_abf(System* system, bool calcEnergy);

  void recv_meta();
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
