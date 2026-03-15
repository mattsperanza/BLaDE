#ifndef ENHANCED_H
#define ENHANCED_H

#include "main/defines.h"

class System;
class Its;
class Ldyn_rest;

class Enhanced {
  public:
    Enhanced();
    ~Enhanced();

    void initialize(System* system);

    bool init = false;
    bool active = false;
    bool updating = true; // don't collect samples (ex. during pressure coupling)
    Ldyn_rest* ldyn_rest = NULL;
    Its* its = NULL;

    int log_freq = 10000; // 20ps
    int write_small_freq = 100; // 2ps
    int write_big_freq = 10000; // 20ps
    std::string output_dir = "nhcd";

    bool separate_interactions = false;
    bool special_elec = false;
    bool osrw = false;

    std::string primary_sele = "";
    int* atom_selection_primary = NULL;
    int* atom_selection_primary_d = NULL;
    std::string secondary_sele = "";
    int* atom_selection_secondary = NULL;
};

void parse_enhanced(char* line, System* system);
void getforce_enhanced(System* system, int step, bool calcEnergy);

#endif
