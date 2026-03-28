#ifndef SIMULATED_TEMPERING_H
#define SIMULATED_TEMPERING_H

#include <string>
#include "main/defines.h"

class System;

enum mbarTerm {
  mbar_k,
  mbar_beta0,
  mbar_betak,
  mbar_Uss,
  mbar_Usu,
  mbar_end
};

class SimulatedTempering {
  public:
    SimulatedTempering(std::string potential);
    ~SimulatedTempering();
    void initialize(System* system, std::string output_dir); // output="nhcd"
    void recv_st();
    void write_g_iter(System* system, std::string output_dir);
    int read_restart(System* system, std::string output_dir); // output="nhcd/g_updates"
    bool solve_mbar(System* system);

    // U selection ("total" or "solute")
    std::string potential;

    real_e U; 
    real_e* U_d = NULL; // used to accumulate total potential
    real_e U_prime;
    real_e* U_prime_d = NULL;

    // Do not free, pointers to system->state->*
    real_e U_ss=0;
    real_e* U_ss_d = NULL;
    real_f* dU_ss_d = NULL;
    real_e U_su=0;
    real_e* U_su_d = NULL;
    real_f* dU_su_d = NULL;
    real_e* U_uu_d = NULL;
    real_f* dU_uu_d = NULL;
    // End do not free

    int low_idx; // used to isolate temperatures
    int high_idx; // used to isolate temperatures

    // Fixed variables
    int N_temp; 
    real* temperatures = NULL; // temperature range
    real* temperatures_d = NULL; // temperature range
    real* betas = NULL; // betas range
    real* betas_d = NULL; // betas range

    real* g = NULL; // current weights
    real* g_d = NULL; // current weights
    real scale_ss = 1.0;
    real* scale_ss_d = NULL; // stores B/B0, scale_su = sqrt(scale_ss)

    // Expanded ensemble
    int sample_freq = 10; // collect U_ss and U_su
    int temp_sample_freq = 10; // EE step 
    int temp_curr_idx; // current K index
    int* temp_curr_idx_d; // current K index

    /* MBAR weight updates:
       Initial starts with a fixed temp sweep,
       Followed by equilibrium iterations using fixed weights
       All info stored in g.dat and mbar.dat
    */
    int current_iter = 0;
    bool just_iterated = false;
    int total_iters = 20; // 2ns 
    int iter_history = 10; // look at last 10 mbar files (1ns)
    int iteration_length = 50000; // 100ps iterations
    int equil_length = 12500; // 25ps of each iteration for equil
    int collected_iter_samples = 0; // progress through this iteration

    // Derived quantities
    int steps_per_temp; 
    int samples_per_temp;
    int equil_per_temp; 
    int equil_samples_per_temp; 
    int iteration_samples;
    int equilibration_samples;
    int mbar_data_max_id = 0; // mbar data only filled up to this iteration
    int mbar_data_length;

    // 1D array with [[K, B0, BK, U_ss, U_su] for each sample for each iter]
    // Circular w.r.t. iteration, only includes up to iter_history-1 previous runs and current
    // Data collected in iteration zero is ignored in later iterations
    int sample_data_length = mbar_end;
    real* mbar_data = NULL; // [iter_history][samples][info] read in for restart
    real* g_data = NULL; // [iter_history][weights] also read in for restart

    // in "nhcd/"
    std::string output_dir;
    std::string fnm_betas = "betas.dat";
    FILE* fp_betas = NULL;
    std::string fnm_temps = "temps.dat";
    FILE* fp_temps = NULL;
    std::string fnm_current_T = "T.dat";
    FILE* fp_current_T = NULL;
    /*
      File formats:
      g.dat, written at beginning of each iteration:
      iteration g0 g1 g2 g3

      mbar.dat, written to with write_small_st():
      iteration K beta0 betak U_ss U_su

      Restarts depend on iteration in "g.dat" and reading corresponding data in "mbar.dat"
      A restart from the middle of an iteration restarts the iteration and deletes the unfinished data from "mbar.dat"
    */
    bool do_restart = true;
    std::string fnm_g = "g.dat";
    FILE* fp_g = NULL;
    std::string fnm_mbar = "mbar_rst.dat";
    FILE* fp_mbar = NULL;
    std::string fnm_mbar_iter = "mbar_prod0.dat"; // used to reweight 
    FILE* fp_mbar_iter = NULL;
};
    
void getforce_st(System* system, int step, bool calcEnergy);
void update_st(System* system);
void log_st(System* system);
void write_small_st(System* system, std::string output_dir);
void write_big_st(System* system, std::string output_dir);

#endif
