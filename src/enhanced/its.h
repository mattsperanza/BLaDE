#ifndef ITS_H
#define ITS_H

#include <string>
#include "main/defines.h"

class System;

class Its {
  public:
    Its(std::string potential);
    ~Its();
    void initialize(System* system);
    void recv_its();

    // U selection
    std::string potential;

    real_e U; // used to accumulate total potential
    real_e* U_d = NULL;
    real_e U_prime;
    real_e* U_prime_d = NULL;
    real* its_bias = NULL; // [N_temp real] bias relative to each temp
    real* its_bias_d = NULL; // [N_temp real] bias relative to each temp

    // Do not free, pointers to system->state->*
    real_e U_ss=0;
    real_e* U_ss_d = NULL;
    real_f* dU_ss_d = NULL;
    real_e U_su=0;
    real_e* U_su_d = NULL;
    real_f* dU_su_d = NULL;

    // Used for calculating bias potential
    int low_idx;
    int N_temp; // Number of temperatures to integrate over
    int N_temp_max;
    real* temperatures = NULL; // [N_temp real] integrated temperature range
    real* temperatures_d = NULL; // [N_temp real] integrated temperature range
    real* g_k = NULL;
    real* g_k_d = NULL;
    real weighted_beta;
    real* weighted_beta_d = NULL; // stores <B/B0>
    real weighted_root_beta;
    real* weighted_root_beta_d = NULL; // stores <sqrt(B/B0)>
    real* pHist_accum = NULL;
    real* pHist_accum_d = NULL; // [N_temp real] p(B | X)
    real* pHist = NULL;
    real* pHist_d = NULL;
    real correction_strength = 0; // 5 is a fine value (does this depend on number of temps?)
    real* correction_strength_d = NULL;

    // OnTheFly Weight Updates
    int update_steps = 1e9; // update until system->run->step > update_steps
    int steps_per_temp = 5000; // Add temp every x steps
    int sample_freq = 20; // update <U> with new sample every x steps
    real* expected_U = NULL; // <U> = weighted_U / weights 
    real* expected_U_d = NULL; // <U> = weighted_U / weights 
    real* weighted_U_d = NULL; // sum U*exp(B*U_bias)
    real* weights_d = NULL; // sum exp(B*U_bias)
    real* offsets_d = NULL;

    FILE* fp_beta = NULL;
    FILE* fp_g = NULL;
    FILE* fp_exp_U = NULL;
    FILE* fp_red_bias = NULL;
    FILE* fp_weighted_T = NULL;
};
    
void getforce_its(System* system);
void update_its(System* system);
void log_its(System* system);
void write_small_its(System* system, std::string output_dir);
void write_big_its(System* system, std::string output_dir);

#endif