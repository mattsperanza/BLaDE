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

    // U selection
    std::string potential;

    real* its_bias = NULL; // [N_temp real] bias relative to each temp
    real* its_bias_d = NULL; // [N_temp real] bias relative to each temp

    // Used for calculating bias potential
    int N_temp; // Number of temperatures to integrate over
    int N_temp_max;
    real* temperatures = NULL; // [N_temp real] integrated temperature range
    real* temperatures_d = NULL; // [N_temp real] integrated temperature range
    real* g_k = NULL;
    real* g_k_d = NULL;
    real alpha = 0.4;
    real* alpha_d = NULL;
    real* weighted_beta = NULL;
    real* weighted_beta_d = NULL;
    real* pHist = NULL;
    real* pHist_d = NULL; // [N_temp real] p(B | X)

    // Continuous Weight Updates
    int add_temp_every = 5000;

    real* expected_U = NULL; // <U> = weighted_U / weights (cleared at beginning of each cycle)
    real* expected_U_d = NULL; // <U> = weighted_U / weights (cleared at beginning of each cycle)
    real* weighted_U = NULL; // sum U*exp(B*U_bias)
    real* weighted_U_d = NULL; // sum U*exp(B*U_bias)
    real* weights = NULL; // sum exp(B*U_bias)
    real* weights_d = NULL; // sum exp(B*U_bias)
    real* offsets = NULL;
    real* offsets_d = NULL;
    int sample_freq = 20; // every ten steps
};
    
void getforce_its(System* system);
void update_its(System* system);
void log_its(System* system);

#endif