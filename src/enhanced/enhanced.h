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

  private:
    Its* its;
};

void parse_enhanced(char* line, System* system);

#endif
