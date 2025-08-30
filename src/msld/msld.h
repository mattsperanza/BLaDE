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
  leus_sin2,
  leus_xsin,
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
  // Periodic function starts with sub 0 plateau, then transitions to sub 1, then sub 1 plateau
  bool L_LEUS = true; // overrides new_implicit
  leus_func L_LEUS_function = leus_xsin;
  real xsin_n = 1; // number of sub-plateaus in transition + 1 (only for xsin function)
  real* dLdT_d; // [blockCount] first derivative of lambda w.r.t. theta
  real* d2LdT2_d; // [blockCount] second derivative of lambda w.r.t. theta
  real plateau_w = .1; // Length of physical regions for each substituent
  real transition_w = 1.0; // Length of transition regions between sequential subsituents
  real* site_period_d; // [siteCount] [0, site_period) range of theta sampling in each site
  real* site_period; // [siteCount] [0, site_period) range of theta sampling in each site

  // FE update & data -> all forms of sampling are updated similtantiously whether set or not
  bool update_fe_surface = true; // add samples to histograms
  int sample_freq = 2; // <dU/dT> from 2D gets updated with every sample
  int update_steps = -1; // steps before turning update_fe_surface false - default to 1/4th of simulation
  real* dUdL_msld_d; // [blockCount] lambda forces from force field 
  real* dUdL_msld; // [blockCount]
  real* dUdT_msld_d; // [blockCount] theta forces from force field
  real* dUdT_msld; // [blockCount]
  real* theta_temp_d; // [blockCount] temperature of theta particles
  real* theta_temp;
  real samples=0;
  // Restarts & Logging
  bool restartable = false;
  bool restart_success = false;
  int log_freq = 1000; // log every # steps
  int write_freq = 1000; // write restart files every # steps 
  // Supported enhanced sampling options - these only work with L_LEUS interpolation function
  bool meta = false; // Feel samples from 1D histogram -> tempered on own potential
  bool oss = false; // Feel samples from 2D histogram -> tempered on own potential
  bool LE = false; // Feel samples from LE memory -> transition tempered
  bool abf = false; // Feel opposite of average force

  // 1D, along T
  real* theta_histogram_d; // [total_T_bins] sample count in each theta bin
  real* histogram_1D_d; // [total_T_bins] 1D meta tempered sample count in each theta bin
  real* potential_1D_d; // [total_T_bins] 1D meta potential
  real* max_pot_d; // [total_T_bins] max potential at given X in histogram
  int* min_dUdT_index_d; //[total_T_bins] index of minimum value dU/dT sample 
  int* max_dUdT_index_d; //[total_T_bins] index of maximum value dU/dT sample
  // 2D, along (T, dU/dT)
  real* histogram_2D_d; // [dUdT_bins*total_T_bins] 2D meta tempered sample count in (T, dU/dT) bin
  real* potential_2D_d; // [dUdT_bins*total_T_bins] 2D potential from OSS metadynamics
  // Grid adjustable parameters - used for 1D meta as well
  real T_std = .02; 
  real dUdT_max = 700;
  real dUdT_std = 2.0;  
  int bins_per_std = 2; 
  int n_std_search = 5; 
  // Grid derived Parameters
  real dUdT_min = -dUdT_max;
  real T_res; // T_std / bins_per_std
  real dUdT_res; // dUdT_std / bins_per_std
  int dUdT_bins; // abs(dUdT_max) + abs(dUdT_min) / T_res + 1
  int* T_bins_d; // [siteCount] site_period / grid resolution + 1, should always be integer multiple
  int* T_bins;
  int total_T_bins; // sum of T_bins
  int T_search; // n_std_search*T_std/res 
  int dUdT_search; // n_std_search*dUdT_std/res

  // 1D Meta
  real meta_bias_mag = .0; // Needs to be set if meta should do anything
  real* meta_bias_d; // [siteCount] bias due to 1D meta
  real* meta_bias;
  // 2D Meta 
  real oss_bias_mag = .0; // Needs to be set if oss should do anything
  bool oss_theta = true; // Compute dGdF*d2U/dXdT or dGdF*d2U/dXdL, L is not supported yet
  bool oss_force_test = false; // set to true to skip calculation steps during force testing
  real* oss_bias_d; // [siteCount] added bias potential at current step due to site histogram
  real* oss_bias; // [siteCount] added bias potential current step due to site histogram
  real* dGdF_d; // [blockCount] OSS chain rule multiplier due to gaussians, dGdF[i] * d2U/dTidX
  // Adds U_bias = oss_k*dU_msld/dT
  real oss_k = 0.0; 
  // Metadynamics adjustable parameters - 1D & 2D
  bool temper = true;
  real bias_mult = 1.0; // multiplier onto ost bias that bias_potential[i] does not pick up
  real temper_amount = 2.0; // exp(-bias_pot/(amount*kT))
  real temper_offset = 0; // exp(-max(0, bias_pot-offset)/(amount*kT))

  // ABF
  bool abf_oss = false; // compute weighted ABF from 2D histogram potential
  bool abf_umbrella = false; // compute weighted ABF from Torrie-Valleau reweighting - via offset exp sum
  bool abf_unweighted = true; // compute unweighted ABF
  real abf_warmup_samples = 100; // step-function after which ABF is added to theta force
  real* abf_weighted_dUdT_d; // [total_T_bins] sum(dU/dT*exp(bias/kT))
  real* abf_weighted_dUdT2_d; // total_T_bins] sum(dU/dT^2*exp(bias/kT))
  real* abf_weights_d; // [total_T_bins] sum(exp(bias/kT))
  real* abf_offsets_d;
  real* abf_ensemble_dUdT_d; // [total_T_bins] <dU/dT> computed from histogram or weighted_dUdT/weights_d
  real* abf_variance_dUdT_d; // [total_T_bins] <(dU/dT)^2> - <dU/dT>^2
  real* abf_bias_d; // [siteCount] bias from each ABF (how to compute)
  real* abf_bias;
  real* dUdT_abf_d; // [siteCount] force added from abf = -<dU/dT>
  real* dUdT_abf;
  
  // Local Elevation -> M[T] += LE_k * pow(f_red, R)
  real* LE_bias;
  real* LE_bias_d; // [site] bias from each LE bias
  real f_red = .6;
  real LE_k = .0293; // 1e-2 kJ/mol = .239e-2 kcal/mol
  real LE_T_res = .1; // Resolution of LE memory grid
  real LE_total_bins; // sum(site_period[i])/LE_T_res - total bins from all sites
  int* LE_bins_d; // [site] site_period[i]/LE_T_res - total bins in each site
  int* LE_bins;
  real* LE_R_d; // [site] starts at 1, add 1 for every double sweep across theta period
  int* LE_visited_bins_d; // [site] keeps track of sum of double sweep
  int* LE_theta_sweep_d; // [LE_total_bins] keeps track of double sweeps
  real* LE_M_d; // [LE_total_bins] Memory, theta bias is memory dotted with cubic bsplines

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

  // Enhanced Sampling Functions
  void init_enhanced_sampling(System* system);
  void copy_reset_memory(System* system);
  void getforce_bias(System* system, bool calcEnergy);
  void add_sample(System *system, int step);
  void log_sampling(System *system, int step);
  void update_ABF_from_hist(System* system, int vert_slices, int horz_slices, bool relative_indexing);
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
