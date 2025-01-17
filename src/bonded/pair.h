#ifndef BONDED_PAIR_H
#define BONDED_PAIR_H

// Forward definitions
class System;

void getforce_nb14(System *system,bool calcEnergy);
void getforce_nbex(System *system,bool calcEnergy);

void getforce_nb14_oss(System *system);
void getforce_nbex_oss(System *system);

#endif
