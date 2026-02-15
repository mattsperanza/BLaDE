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
    real* temperatures = NULL; // [N_temp real] integrated temperature range
    real* g_k = NULL;
    real alpha = 0.4;
    real* alpha_d = NULL;

    // Iterative Weight Updates (2ns default)
    real update_iterations = 20;
    int steps_per_iter = 50000;
    real* expected_U = NULL; // <U> = weighted_U / weights (cleared at beginning of each cycle)
    real* expected_U_d = NULL; // <U> = weighted_U / weights (cleared at beginning of each cycle)
    real* weighted_U = NULL; // sum U*exp(B*U_bias)
    real* weighted_U_d = NULL; // sum U*exp(B*U_bias)
    real* weights = NULL; // sum exp(B*U_bias)
    real* weights_d = NULL; // sum exp(B*U_bias)
    real* offsets = NULL;
    real* offsets_d = NULL;
    int sample_freq = 10; // every ten steps
};
    
void getforce_its(System* system);
void update_its(System* system);

#endif