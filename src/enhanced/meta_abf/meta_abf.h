#ifndef META_ABF_H
#define META_ABF_H

#include "main/defines.h"
#include <string>

class System;

/*
  Tempered Metadynamics + Adaptive Biasing Force:

  U_{bias}(x) = U_{meta}(x) + U_{abf}(x)
                  
                   ___ i=search+curr_bin(x)
                   \
  U_{meta}(x) =     |   weight[i]*exp( ((x - bin_center[i])/s)^2 )
                   /
                   --- i=-search+curr_bin(x)

                    _ x
                   |
  U_{abf}(x) = -   |    <y> dx'
                   |
                   -  lb

  dU_{meta}/dXi = dU_{meta}/dx*cri
  dU_{abf}/dXi = dU_{abf}/dx*cri = -<y>*cri 

  ----------------
  For Spatial CVs:
  ----------------

  -------------------
  For Alchemical CVs:
  -------------------
  Potential looks like: 
  U = L0*U0 + L1*U1; 
  L1= L0 + L1 = 1;  0 <= Li <= 1;
  L0=f(T0,T1), L1=f(T0,T1);

  Choose x-axis:
  x can be     {  L1             |  T1-T0    |  T0      }
  lims         {  [0,1]          |  [lb,ub]  |  [0,ub)  }
  y is then    {  dU/dL1-dU/dL0  |  dU/dT1   |  dU/dT0  }
  
  For x=L1:
    Constrained boundary conditions on potential, metadynamics potential is mirrored
    x = 0*X[0] + 1*X[1], cr0=0, cr1=1
    y = -1*G[0] + 1*G[1], gScale0=-1, gScale1=1
    Gradient from Meta:
      [G[0], G[1]] += [dU_{meta}/dx*0, dU_{meta}/dx*1], cr0 = 0, cr1 = 1
    Gradient from ABF:
      dU_{abf}/dx = -<dU/dL1-dU/dL0>|x=x
      [G[0], G[1]] += [dU_{abf}/dx*0, dU_{abf}/dx*1], cr0 = 0, cr1 = 1

  For x=T2-T1: 
    Unconstrained boundary conditions on potential, metadynamics force set to zero outside of clamped region
    x = -1*X[0] + 1*X[1], cr0=-1, cr1=1
    y = 0*G[0] + 1*G[1], gScale0=0, gScale1=1
    Gradient from Meta:
      [G[0], G[1]] += [dU_{meta}/dx*-1, dU_{meta}/dx*1], cr0 = -1, cr1 = 1
    Gradient from ABF:
      dU_{abf}/dx = -<dU/dT1>|x=x = <dU/dT0>|x=x  (opposite as a result of implicit constraint)
      [G[0], G[1]] += [dU_{abf}/dx*-1, dU_{abf}/dx*1], cr0 = -1, cr1 = 1

  For T0:
    Periodic boundary conditions on potential
    x = 1*X[0] + 0*X[1], cr0=1, cr1=0
    y = 1*G[0] + 0*G[1], gScale0=1, gScale1=0
    Gradient from Meta:
      [G[0], G[1]] += [dU_{meta}/dx*1, dU_{meta}/dx*0], cr0 = 1, cr1 = 0
    Gradient from ABF:
      dU_{abf}/dx = -<dU/dT0>|x=x
      [G[0], G[1]] += [dU_{abf}/dx*1, dU_{abf}/dx*0], cr0 = 1, cr1 = 0

*/

typedef enum BoundaryConditions_MABF {
  unconstrained_bc, // [lb, ub] with clamped BC (i.e. U_{meta}(clamp(x, lb, ub)), zero force outside lb ub)
  constrained_bc, // [lb, ub] with mirrored BC
  periodic_bc, // [0, period) with periodic BC (last bin not included)
} BCs_MABF;

// Just pointers, don't free
struct AlchemicalCV_MABF {
  int target_site;
  real x, y;
  real *x_d, *y_d; // pointer to 1 real x coordinate and force
  real *sysX_d, *sysG_d; // pointer to first system coordinate and force
  real cr0, cr1; // x-axis (and chain rule) scaling like: x = cr0*X[0] + cr0*X[1]
  real gScale0, gScale1; // y-axis scaling like: y = gScale0*G[0] + gScale*G[1]
  bool lambda_space; // is abf on theta or lambda
};

// [lb, ub] -> bins centered on lb and ub with uniform spacing between
struct MetaABFGrid {
  real lb, ub, lb_clamp, ub_clamp;
  real *counts, *meta_weights, *y_m2, *y_avg;
  real *counts_d, *meta_weights_d, *y_m2_d, *y_avg_d;
  real bin_width;
  int n_bins;
  int bc; // see BoundaryConditions_MABF enum
};

struct MetaOptions_MABF {
  real w;
  real stdev, temper_factor;
  bool do_temper;
};

class MetaAdaptiveBiasingForce {
  public:
    MetaAdaptiveBiasingForce(){};
    ~MetaAdaptiveBiasingForce();
    void initialize(System* system);
    void restart(System* system);
    void compute_CV(System* system);
    void apply_chain_rule(System* system);
    void step_reset(System* system);
    bool init=false;

    // Options
    bool do_abf=true;
    bool do_meta=true;
    bool do_sample=true; 
    bool do_restart=true;

    AlchemicalCV_MABF alchemCV = {
      1, 0, 0, NULL, NULL, NULL, NULL,
      0, 1, -1, 1, true};
    MetaABFGrid grid = {0, 1, 0, 1, 
      NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
      0.005, (int)(1/0.005 + 1), constrained_bc};
    MetaOptions_MABF meta_options = {0.005, 0.01, 10, true};

    // ABF Stuff
    int abf_warmup=200; // number of samples before full activation
    real* y_std=NULL;
    // Metadynamics Stuff
    real search_std = 6; // look this many std for gaussians
    int half_search_bins; // = ceil(search_std*meta_std/bin_width) in each direction

    // General Stuff
    int sample_freq=10;
    int total_samples=0;
    real abfU, metaU, abfG, metaG;
    real* abfU_d, *metaU_d, *abfG_d, *metaG_d;
    real* tmpLmdF_d=NULL; // if alchemCV, this is length blockCount to modify lambda forces for theta force conversion
    real* tmpThetaF_d=NULL; // if alchemCV, this is length blockCount to put thetaForces into it

    // Restart 
    int write_restart_freq=1000;
    std::string fnm_meta_abf = "meta_abf.rst";
    FILE* fp_meta_abf = NULL;
    // Logging
    int log_freq=0;
};

void parse_meta_abf(char* line, MetaAdaptiveBiasingForce* meta_abf);
void getforce_meta_abf(System* system, int step, bool calcEnergy);
void sample_meta_abf(System* system, int step);
void log_meta_abf(System* system, int step);
void recv_meta_abf(System* system);
void write_meta_abf(std::string dir_name, System* system, int step);

#endif