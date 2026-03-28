#ifndef ENHANCED_H
#define ENHANCED_H

#include "main/defines.h"

class System;
class SimulatedTempering;
class Ldyn_rest;

class Enhanced {
  public:
    Enhanced();
    ~Enhanced();

    void initialize(System* system);

    bool init = false;
    bool active = false;
    bool updating = true; // don't collect samples (ex. during pressure coupling)
    SimulatedTempering* simulatedTempering = NULL;

    int log_freq = 100000; // 200ps
    int write_big_freq = 1000; // 20ps
    int write_small_freq = 10; // Same as lambda sampling
    std::string output_dir = "nhcd";

    bool separate_interactions = false;
    bool special_nbdirect = false;
    int nbrecip_mode = 0; // enum {correct, all, none}

    std::string primary_sele = "";
    int* atom_selection_primary = NULL;
    int* atom_selection_primary_d = NULL;
};

void parse_enhanced(char* line, System* system);
void getforce_enhanced(System* system, int step, bool calcEnergy);

#endif
