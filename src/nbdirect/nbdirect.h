#ifndef NBDIRECT_NBDIRECT_H
#define NBDIRECT_NBDIRECT_H

// Forward definitions
class System;

void getforce_nbdirect(System *system,bool calcEnergy);
void getforce_nbdirect_reduce(System *system,bool calcEnergy);

// Function calls for separated U_ss & U_su & dU/dXdL
void getforce_nbdirect_special(System *system, bool calcEnergy);
void getforce_nbdirect_reduce_special(System *system,bool calcEnergy);

#endif
