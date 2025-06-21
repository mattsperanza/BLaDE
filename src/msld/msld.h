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
  real* dUdL_bonded_d; // [blockCount] lambda forces from bonds
  real* dUdL_alf_d; // [blockCount] lambda forces from alf
  real* dUdL_abf_d; // [blockCount] lambda force from ABF, OPES
  real* dUdL_msld_d; // [blockCount] lambda forces from force field 
  real* dUdL_msld; // [blockCount]

  // 1D histogramming and tracking of dU/dL distribution
  bool abf = false; // Subtract average dU/dL using
  int L_1D_bins = 51; // this is also the max index (51 leads to >.99 as last bin since first and last are half width)
  int L_imp_bins = 401;
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
  // All paths sum_sites Ns*(Ns-1) -> each lambda in a site to every other lambda in that site, averages of its own forces 
  int path_count;
  int warmup_samples = 20; // linear ramp of <dU/dL> with how much abf sample weight you have (basically number of samples)
  // ABF alone doesn't work well when this is high since rare events are not capitalized on (might just be very slow idk)
  // Then again basing <dU/dL> on 1 sample may introduce artificial barriers if you sampled an outliner
  real edge_KDE_std = .05; // gaussians go to ~0 around 4*std
  // This means samples where sum k!=i,j(lmd k) < .08 have negligible weight
  real* path_samples_d; // [Ns*(Ns-1)] reduction of weights along each path, including prior
  real* path_sample_offsets_d;
  real* path_unsamples_d; // [Ns*(Ns-1)] reduction of unweights along each path, including prior
  real* path_weights_d; // [L_1D_bins * sum_sites Ns*(Ns-1)] gaussian weighted distances from edges with bias weighting
  real* path_unweights_d; // [L_1D_bins * sum_sites Ns*(Ns-1)] gaussian weighted distences from edges without bias weighting
  real* path_weight_offsets_d; // 
  real* path_weighted_dUdL_d; // [L_1D_bins * sum_sites Ns*(Ns-1)] weighted dU/dL
  real* path_weighted_dUdL2_d; // [L_1D_bins * sum_sites Ns*(Ns-1)] weighted dU/dL^2
  real* path_ensemble_dUdL_d; // [L_1D_bins * sum_sites Ns*(Ns-1)] <dU/dL> = sum(w*dU/dL) / sum(w)
  real* path_dUdL_variance_d; // [L_1D_bins * sum_sites Ns*(Ns-1)] <dU/dL^2> - <dU/dL>^2 this isn't the most stable estimator
  real* path_dUdL_diff_d;
  real* path_dUdL_diff2_d;
  real* path_ens_dUdL_diff_d;
  real* path_ens_dUdL_variance_d;
  real L_resolution = (abs(L_max)+abs(L_min))/L_1D_bins;
  real L_std = 4*L_resolution; // free
  int L_search = 4.0*(L_std/L_resolution); 

  real alpha = 1.0; // lambda^alpha scaling

  // Histogram (2D meta) -> uniform binning -> nL = blockCount-1
  bool oss = false; // Perform Orthogonal Space Sampling calculations
  bool opes = false; // 
  bool explore = false; // OPES explore vs OPES standard
  bool do_imp = false;
  real* p_imp_d; // [(nSite-1) * L_oss_bins] probability due to implicit constraints / max(p_imp_d)
  real* dGdF_d; // [blockCount] OSS chain rule multiplier, dGdF[i] * d2U/dlidX
  real* hist_potential_d; // [blockCount-1] potential from 2d metadynamics
  real* hist_potential; // [blockCount-1]
  real* min_bias_d; // [blockCount]

  real L_max = 1.0;
  real L_min = 0.0;
  real opes_dE = 5.0; // gamma * kT
  real opes_gamma = opes_dE * .6; // dE / kT
  real opes_eps = exp(-opes_dE / (.6 - .6 / opes_gamma));

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
  bool slow_fix; // Slowly tranform from WT to desired state, gets turned off automatically prior to end of this round of dynamics
  real_x *lambda_delta_d; // [blockCount] desired - WT / nsteps

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

  // ABF Functions
  void add_sample(System *system, int step);
  void log_sampling(System *system, int step);
  void calc_imp(System* system);
  void init_abf(System* system);
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
