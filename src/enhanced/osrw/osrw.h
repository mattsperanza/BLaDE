#ifndef OSRW_H
#define OSRW_H

#include "main/defines.h"
#include <string>

class System;

/*
  Transition Tempered 2D Metadynamics (L, dU/dL) + Adaptive Biasing Force along a single lambda dimension
  Supports calculations with multiple sites, but the OSRW can only be applied on a single site with 2 substituents
*/
class OrthogonalSpaceRandomWalk {
  public:
    OrthogonalSpaceRandomWalk(){};
    ~OrthogonalSpaceRandomWalk();
    void initialize(System* system);
    void restart(System* system);

    bool init=false;
    bool force_test=false;
    bool random_samples=false;

    // Options - Settings
    int target_site=1; // must have two substitiuents at this site
    bool do_abf=true;
    bool do_meta=true;
    bool do_sample=true; 
    bool do_restart=true;

    // Options - enhance specific DOF
    bool remove_bonded=false;
    bool remove_recip=false;
    bool sample_weighting=true;

    // Options - Sampling
    int sample_freq=10; // 1/step

    // Options - Grid/Meta
    real L_res = 0.005; // L in [0,1]
    real dUdL_res = 2.0;
    real dUdL_max = 2500;
    real meta_bias_mag=0.005; 
    real meta_L_std=0.01;
    real meta_dUdL_std=4.00;
    bool do_temper=true;
    real temper_factor=5; 
    int n_std_search = 6; 

    // Options - ABF
    int abf_warmup=100; // number of samples before full activation

    // Derived Parameters - Not optional
    real dUdL_min=-dUdL_max;
    int n_L_bins; 
    int n_dUdL_bins; 
    real bin_width_L;
    real bin_width_dUdL;
    int half_search_bins_L;
    int half_search_bins_dUdL;

    // Memory
    real* dUdL_copy=NULL;
    real* dUdL_copy_d=NULL;
    real* dGdF=NULL;   // OSRW chain rule multiplier due to gaussians, dGdF[i] * d2U/dLidX
    real* dGdF_d=NULL; // OSRW chain rule multiplier due to gaussians, dGdF[i] * d2U/dLidX
    // 2D, along (L, dU_sub/dL)
    real* lambda_counts_2D=NULL;
    real* lambda_counts_2D_d=NULL; // counts of number of samples in each 2D bin
    real* hist_weights_2D=NULL; 
    real* hist_weights_2D_d=NULL; // 2D tempered sample count in all (L, dU_sub/dL) bin
    real* potential_2D=NULL;
    real* potential_2D_d=NULL; // 2D potential from OSRW metadynamics
    real* avg_dUdL_2D=NULL; 
    real* avg_dUdL_2D_d=NULL; // average value of <dU_tot/dL> in each (L, dU_sub/dL) bin
    real* m2_dUdL_2D=NULL; 
    real* m2_dUdL_2D_d=NULL; 
    // 1D along L
    real* lambda_counts_1D=NULL;
    real* lambda_counts_1D_d=NULL;
    real* ensemble_dUdL=NULL;
    real* std_dUdL_d=NULL;
    real* std_dUdL=NULL;
    real* variance_dUdL_d=NULL;
    int* min_dUdL_id_d=NULL;
    int* max_dUdL_id_d=NULL;

    real* current_temper_bias_d=NULL;

    // Restart 
    int write_restart_freq=10000;
    std::string fnm_osrw_rst = "osrw.rst";
    FILE* fp_osrw_rst = NULL;
    // Logging
    int log_freq=500000;
};

// Pair OSRW (bonded streams) 
void getforce_nbex_oss(System *system);
void getforce_nb14_oss(System *system);
// Bonded OSRW
void getforce_bond_oss(System *system);
void getforce_angle_oss(System *system);
void getforce_dihe_oss(System *system);
void getforce_impr_oss(System *system);
void getforce_cmap_oss(System *system);
// nbrecip OSRW
void getforce_ewaldself_oss(System *system);
void getforce_ewald_oss(System *system);
// nbdirect OSRW
void getforce_nbdirect_oss_reduce(System *system);
void getforce_nbdirect_oss(System *system);

void parse_osrw(char* line, OrthogonalSpaceRandomWalk* osrw);
void getforce_osrw(System* system, int step, bool calcEnergy);
void sample_osrw(System* system, int step);
void log_osrw(System* system, int step);
void recv_osrw(System* system);
void write_osrw(std::string dir_name, System* system, int step);

#endif