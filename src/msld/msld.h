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

  // Histogram variables
  bool apply_histogram = true;
  int sampleFrequency = 10;
  int depth=2; // number of histograms to keep
  int first_half_bins=20; // number of bins assigned to the first half of the lambda range
  int second_half_bins=20;
  int total_bins; // total number of bins
  //TODO: set up system for swapping
  int sample_from=0;
  int accumulate_into=0; // which histogram index to accumulate into right now
  int accumulate_length=5000000; // how long to accumulate before combining data
  real *bin_edges; // edges of the bins 0-1, redefined for every site
  real *bin_edges_d;
  real lambda_std; // if dropping gaussians or some other kernel
  real dUdL_std;
  // Helper variables
  int* hist_index; // index to start of this lambda's histogram
  int* hist_index_d;
  real* step_force_d; // Force added on this step for each lambda, to correct force later
  real* step_potential_d; // same as above, only for potential
  real** partition_function; // sum of all weights for a given histogram = <exp(bias)>
  real** partition_function_d;
  real** partition_offset_d;
  //// Long, flat arrays of length nLambda*nBins
  real **histogram_counts; // number of occurrences in each bin
  real **histogram_counts_d;
  // TI-<dU/dL> estimation stuff
  int nFull = 0; // number of samples required in a bin to be 100% active, otherwise linear scaling
  real **ensemble_dUdL; // ensemble average dU/dL in each bin
  real **average_dUdL;
  real **ensemble_dUdL_d; // device pointer
  real **average_dUdL_d;
  real **integral_components; // area under the curve for this bin
  real **integral_components_d;
  real **offsets; // stores largest value of bias to save exp numerics
  real **offsets_d;
  real **weights; // sum of boltzmann weights = exp(beta*bias)
  real **weights_d;
  real **probability_distribution; // P(lambda) = <delta(lambda - lambda_i)exp(bias)> / <exp(bias)>
  real **probability_distribution_d;
  real **weighted_dUdL; // sum of boltzmann weighted dU/dL in each bin
  real **weighted_dUdL_d;
  real **dPdL;
  real **dPdL_d; // dP/dL
  real **weighted_dUbias_dL_d; // sum of boltzmann weighted dU_bias/dL in each bin
  real **weighted_partition_function_d; // sum of weighted dU_bias/dL in a histogram
  // variance = E[X^2] - E[X]^2
  real **ensemble_dUdL2; // ensemble average (dU/dL)^2 in each bin
  real **ensemble_dUdL2_d;
  real **variance; // variance of dU/dL in each bin
  real **variance_d;
  real **weighted_dUdL2; // sum of weighted ensemble_dUdL squares
  real **weighted_dUdL2_d;
  // OPES
  real **opes_potential; // V(lambda) = (1-1/gamma)/beta * log(P(lambda) + eps)
  real **opes_potential_d;
  real **opes_force;
  real **opes_force_d;
  real opes_barrier; // barrier for opes to overcome - defines gamma and esp parameters
  real opes_gamma;
  real opes_eps;

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

  // Histogram estimation functions
  static void assign_edges(int num_sites, const int *blocksPerSite, int first_half_bins, int second_half_bins, real *bin_edges);
  void add_sample(System *system);
  void getforce_histogram(System *system, bool calcEnergy);
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
