#ifndef ENHANCED_H
#define ENHANCED_H

#include "main/defines.h"

class System;
class Its;

class Enhanced {
  public:
    Enhanced();
    ~Enhanced();

    void initialize(System* system);

    bool active = false;
    bool updating = true; // don't collect samples (ex. during pressure coupling)
    Its* its = NULL;

    int log_freq = 10000; // 20ps
    int write_small_freq = 100; // 2ps
    int write_big_freq = 10000; // 20ps
    std::string output_dir = "nhcd";

    bool separate_interactions = false;
    bool torsions_only = false;
    std::string primary_sele = "";
    int* atom_selection_primary = NULL;
    std::string secondary_sele = "";
    int* atom_selection_secondary = NULL;
};

void parse_enhanced(char* line, System* system);
void getforce_enhanced(System* system);

#endif
