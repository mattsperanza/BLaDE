#ifndef ITS_H
#define ITS_H

#include <string>
#include "main/defines.h"

class System;

class Its {
  public:
    Its(std::string potential);
    ~Its();
    void initialize();
    void recv_its(){};

    // U selection
    std::string potential;

    real_e U;
    real_e* U_d;
    real_e U_prime;
    real_e* U_prime_d = NULL;
    real* its_bias = NULL; // [N_temp real] bias relative to each temp
    real* its_bias_d = NULL; // [N_temp real] bias relative to each temp

    // Used for calculating bias potential
    int N_temp; // Number of temperatures to integrate over
    int N_temp_max;
    real* temperatures = NULL; // [N_temp real] integrated temperature range
    real* temperatures_d = NULL; // [N_temp real] integrated temperature range
    real* g_k = NULL;
    real* g_k_d = NULL;
    real alpha = 0.3; // arbitrarily set to lower to match REST2 kinda
    real* alpha_d = NULL;
    real weighted_beta;
    real* weighted_beta_d = NULL;
    real* pHist = NULL;
    real* pHist_d = NULL; // [N_temp real] p(B | X)
    real correction_strength = 0; // 5 is a fairly good value
    real* correction_strength_d = NULL;

    // OnTheFly Weight Updates
    int steps_per_temp = 5000; // Add temp every x steps
    int sample_freq = 20; // update <U> with new sample every x steps
    real* expected_U = NULL; // <U> = weighted_U / weights 
    real* expected_U_d = NULL; // <U> = weighted_U / weights 
    real* weighted_U_d = NULL; // sum U*exp(B*U_bias)
    real* weights_d = NULL; // sum exp(B*U_bias)
    real* offsets_d = NULL;
};
    
void getforce_its(System* system);
void update_its(System* system);
void log_its(System* system);

#endif