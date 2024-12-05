#ifndef MSLD_MSLD_H
#define MSLD_MSLD_H

#include <vector>
#include <set>

#include "main/defines.h"
#include "msld/HistogramEstimator.h"

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
  real *dGdLambda;
  real *dGdFl;
  // Histogram details
  bool histogram_switch = true;
  int sampleFrequency = 10;
  int* histIndex; // index to start of this lambda's histogram
  int depth; // number of histograms
  real *bin_edges; // edges of the bins 0-1
  real *bin_edges_d;
  real *bin_widths; // width of each bin, used for normalization & integration weighting
  real *bin_widths_d;
  int first_half_bins; // number of bins assigned to the first half of the lambda range
  int second_half_bins; // number of bins assigned to the other half of the lambda range
  int total_bins; // total number of bins
  int accumulate_into; // which histogram index to accumulate into
  // Longer TI & OST variables of length nLambda*nBins
  real **histogram_counts; // number of occurrences in each bin
  // <dU/dL> estimations
  real **dUdL; // ensemble average dU/dL in each bin
  real *dUdL_d; // device pointer
  real **d2UdL2; // ensemble average d2U/dL2 in each bin
  real *d2UdL2_d;
  real *offsets; // stores largest value of -beta*(U(x,l)-U_bias(x,l)) to save exp numerics
  real **log_weights; // sum of boltzmann log_weights = exp(-beta*(U(x,l)-U_bias(x,l)))
  real *log_weights_d;
  real **log_weighted_dUdL; // sum of boltzmann weighted dU/dL in each bin
  real *log_weighted_dUdL_d;
  real **log_weighted_d2UdL2; // optimize lambda profile with 2nd derivative information
  real *log_weighted_d2UdL2_d;
  // variance = E[X^2] - E[X]^2
  real **dUdL2; // ensemble average (dU/dL)^2 in each bin
  real **variance; // variance of dU/dL in each bin
  real *variance_d;
  real **log_weighted_dUdL2; // sum of weighted dUdL squares
  real *log_weighted_dUdL2_d;

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
  static inline int get_bin_index(real lambda, int total_bins, const real* bin_edges);
  static void assign_edges(int num_lambdas, int first, int second, real *edges, real* gaps);
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
