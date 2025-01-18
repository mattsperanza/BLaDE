#ifndef NBRECIP_NBRECIP_H
#define NBRECIP_NBRECIP_H

// Forward definitions
class System;

void getforce_ewaldself(System *system,bool calcEnergy);
void getforce_ewald(System *system,bool calcEnergy);

void getforce_ewaldself_oss(System *system);
void getforce_ewald_oss(System *system);

#endif
