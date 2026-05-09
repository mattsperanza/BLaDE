#ifndef WTM_ABF_H
#define WTM_ABF_H

#include "main/defines.h"

class System;

/*
  Transition Tempered Metadynamics + Adaptive Biasing Force along a single lambda dimension
  Supports calculations with multiple sites, but the WTM+ABF can only be applied on a single site with 2 substituents
*/
class MetaAdaptiveBiasingForce {
    MetaAdaptiveBiasingForce();
    ~MetaAdaptiveBiasingForce();
    void initialize(System* system);

    // Options
    bool do_abf;
    bool do_meta;
    int n_bins; // 101, bins
    int abf_warmup; // 100, samples
    int sample_freq; // 10, step
    real meta_weight; // 0.01, kcal/mol
    real temper_factor; // 5, scale weight by exp( -min_L(meta_weights)/(temper_factor*kT) )
    real temper_threshold; // 10, kcal/mol

    // ABF Memory
    real* counts;
    real* dUdL_sum;
    real* dUdL2_sum;
    real* dUdL_var;
    real* dUdL_ave;

    // Metadynamics Memory
    real* meta_weights;
};

void parse_wtm_abf(char* line, System* system);
void getforce_wtm_abf(System* system, int step, bool calcEnergy);
void sample_wtm_abf(System* system, int step);

#endif