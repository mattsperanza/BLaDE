#ifndef NBDIRECT_NBDIRECT_H
#define NBDIRECT_NBDIRECT_H

// Forward definitions
class System;

void getforce_nbdirect(System *system,bool calcEnergy);
void getforce_nbdirect_reduce(System *system,bool calcEnergy);
void getforce_nbdirect_oss(System *system);
void getforce_nbdirect_oss_reduce(System *system);

#endif
