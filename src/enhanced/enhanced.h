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
    Its* its = NULL;
};

void parse_enhanced(char* line, System* system);
void getforce_enhanced(System* system);

#endif
