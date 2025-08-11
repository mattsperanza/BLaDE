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

typedef enum leus_func {
  leus_linear,
  leus_cubic,
  leus_quintic,
  leus_septic,
  leus_sin2,
  leus_sin2x,
} Leus_func;

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

  real alpha = 1.0; // lambda^alpha scaling

  // MSLD L-LEUS style theta dynamics
  bool L_LEUS = true; // overrides new_implicit
  leus_func L_LEUS_function = leus_sin2x;
  real* dLdT_d; // first derivative of lambda w.r.t. theta
  real* d2LdT2_d; // second derivative of lambda w.r.t. theta
  real plateau_w = .1; 
  real transition_w = 2;
  real* site_period_d; // [0, site_period) range of theta sampling in each site
  real* site_period; // [0, site_period) range of theta sampling in each site

  // FE update & data
  bool update_fe_surface = true; // add samples to histogram
  int update_steps = -1; // timesteps of sampling while updating FE surface - default to 1/4th of simulation
  real* dUdL_msld_d; // [blockCount] lambda forces from force field 
  real* dUdT_msld_d; // [blockCount]
  real* dUdL_msld; // [blockCount]
  real* dUdT_msld; // [blockCount]

  // 2D histogramming of (T, dU/dT) for each site
  bool oss = false; // Perform Orthogonal Space Sampling calculations
  bool oss_theta = true;
  real* dGdF_d; // [blockCount] OSS chain rule multiplier, dGdF[i] * d2U/dTidX
  real* bias_potential_d; // [siteCount] total added bias potential at current step
  real* bias_potential; // [siteCount] total added bias potential current step
  // 2D histogram memory is layed out to give [nSite][theta][dU/dT] -> dU/dT is most continuous in memory
  real* oss_histogram_d; // [sum_sites(T_bins[i]*dUdT_bins)] sampled grid points including tempering weight
  real* oss_potential_d; // [sum_sites(T_bins[i]*dUdT_bins] potential from 2d metadynamics, used for <dU/dT> calculation
  real* oss_potential;
  // 1D memory f(theta)
  real* oss_theta_counts_d; // [Ns*2*L_bins*Ns] # of samples in each theta bin
  bool weighted_dUdL; // boltzmann weight ensemble average <dU/dT>
  real* oss_ensemble_dUdT_d; // [Ns*2*L_bins*Ns] <dU/dT> computed from histogram
  int* oss_dUdT_min_d; //[Ns*T_bins] index of minimum value dU/dT sample 
  int* oss_dUdT_max_d; //[Ns*T_bins] index of maximum value dU/dT sample
  real* oss_max_pot_d; //[Ns*T_bins] max potential at given X in histogram
  real warmup_samples = 0; // # of samples before <dU/dT> is fully subtracted off in ABF
  
  // Linear k*dU/dT bias
  real oss_k = .0; // normally just set this to be zero

  // Metadynamics adjustable parameters
  bool standard_tempering = false;
  int sample_freq = 5; // also affects how often <dU/dT> gets calculated (histogram potential evaluations can be expensive)
  real bias_mag = .05; // if it is zero we don't do expensive d2U/dTdX calculation
  real temper_amount = 3.0; 
  real temper_offset = 0.0;
  real T_std = .02; 
  real dUdT_std = 4.0;  
  real dUdT_max = 2000;
  real dUdT_min = -dUdT_max;
  // Derived or Fixed Parameters
  real dUdT_res; // dUdT_std / 2.0
  real T_res; // T_std / 2.0
  int dUdT_bins; 
  int* T_bins_d; // [siteCount] site_period / grid resolution + 1, should always be integer multiple
  int* T_bins;
  int T_search; // 5*T_std/res 
  int dUdT_search; // 5*dUdT_std/res

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

  // OSS/ABF Functions
  void init_oss(System* system);
  void log_sampling(System *system, int step);
  void add_sample(System *system, int step);
  void getforce_oss(System* system, bool calcEnergy);
  void get_ABF_from_hist(System* system);

  // L_LEUS
  void oss_lambda_to_theta_force(System* system);

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
