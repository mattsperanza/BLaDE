#ifndef LDYN_REST_H
#define LDYN_REST_H

#include <string>
#include "main/defines.h"

class System;

class Ldyn_rest {
  public:
    Ldyn_rest(real T);
    ~Ldyn_rest();
    void initialize(System* system);

    // alpha
    real alpha = 1.5;
    real T_high;

    // expected U
    int sample_freq = 1;
    int ramp_length = 100;
    int bin_count = 101;
    int* counts = NULL;
    int* counts_d = NULL;
    real* sum_dUdL = NULL;
    real* sum_dUdL_d = NULL;
    real* average_dUdL = NULL;
    real* average_dUdL_d = NULL;

    // Do not free, pointers to system->state->*
    real_e U_ss=0;
    real_e* U_ss_d = NULL;
    real_f* dU_ss_d = NULL;
    real_e U_su=0;
    real_e* U_su_d = NULL;
    real_f* dU_su_d = NULL;
};

void getforce_ldyn_rest(System* system);
void update_ldyn_rest(System* system);
void log_ldyn_rest(System* system);

#endif