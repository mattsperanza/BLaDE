#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>

#include "io/io.h"
#include "system/system.h"
#include "system/selections.h"
#include "system/structure.h"
#include "system/state.h"
#include "system/potential.h"
#include "run/run.h"
#include "enhanced/simulated_tempering.h"

#include "enhanced/enhanced.h"
#include "main/gpu_check.h"

Enhanced::Enhanced(){
  simulatedTempering=NULL;
}

Enhanced::~Enhanced(){
  if(simulatedTempering) delete(simulatedTempering);
  if(atom_selection_primary) free(atom_selection_primary);
  if(atom_selection_primary_d) cudaFree(atom_selection_primary_d);
}

void parse_enhanced(char* line, System* system){
  char token[MAXLENGTHSTRING];
  int i,j;

  if (system->structure==NULL) {
    fatal(__FILE__,__LINE__,"selections cannot be defined until structure has been defined\n");
  }

  if (system->enhanced==NULL) {
    system->enhanced=new Enhanced();
  }

  io_nexta(line,token);
  system->enhanced->active = true;
  if(strcmp(token, "log_freq")==0){
    system->enhanced->log_freq = io_nexti(line);
  } else if (strcmp(token, "updating")==0){
    system->enhanced->updating = io_nextb(line);
  } else if (strcmp(token,"dnmo")==0){
    system->enhanced->output_dir = io_nexts(line);
  } else if (strcmp(token,"atom_selection")==0){
    std::string name=io_nexts(line);
    if (system->selections->selectionMap.count(name)==0) {
      fatal(__FILE__,__LINE__,"Selection %s not found\n",name.c_str());
    }
    system->enhanced->primary_sele=name;
    if (system->enhanced->atom_selection_primary) free(system->enhanced->atom_selection_primary);
    system->enhanced->atom_selection_primary=(int*)calloc(system->structure->atomList.size(),sizeof(int));
    for (i=0; i<system->structure->atomList.size(); i++) {
      system->enhanced->atom_selection_primary[i]=system->selections->selectionMap[name].boolSelection[i];
    }
    cudaMalloc(&system->enhanced->atom_selection_primary_d, system->structure->atomList.size()*sizeof(int));
    cudaMemcpy(system->enhanced->atom_selection_primary_d, system->enhanced->atom_selection_primary, system->structure->atomList.size()*sizeof(int), cudaMemcpyDefault);
  } else if (strcmp(token,"print_sele")==0){
    printf("Dumping enhanced selections! Will segfault if no selection is defined.\n");
    printf("Primary Selection Name: %s\n", system->enhanced->primary_sele.c_str());
    system->selections->dump();
  // ST Reading
  } else if (strcmp(token,"simulated_tempering")==0) {
    if(system->enhanced->simulatedTempering){
      printf("ST already exists!\n");
      exit(1);
    }
    std::string potential = io_nexts(line); 
    if(potential == "solute"){
      if(!system->enhanced->atom_selection_primary){
        printf("Primary atom selection is NULL!\n");
        exit(1);
      }
      system->enhanced->separate_interactions = true;
      system->enhanced->special_nbdirect = true;
      printf("Solute scaling requires separated nbdirect calculation. This may hurt performance!\n");
    }
    if(potential != "total" && potential != "solute"){
      printf("Unsuppored ST potential scaling: %s. Availible: total and solute\n");
      exit(1);
    }
    system->enhanced->simulatedTempering = new SimulatedTempering(potential);
    system->enhanced->active = true;
  } else if(strcmp(token,"st_temps_exp")==0) {
    // Optimal Exp spacing
    int num_temps = io_nexti(line); 
    real temp_low = io_nextf(line);
    real temp_high = io_nextf(line);
    real* temps = (real*) malloc(num_temps*sizeof(real));
    for(int i = 0; i < num_temps; i++){
      temps[i] = temp_low*pow(temp_high/temp_low, ((real)(i))/(num_temps-1));
    }
    printf("Auto ST Temps: [ ");
    for(int i = 0; i < num_temps; i++){
      printf("%f, ", temps[i]);
    }
    printf("] \n");
    system->enhanced->simulatedTempering->temperatures = temps;
    system->enhanced->simulatedTempering->N_temp = num_temps;
  } else if (strcmp(token, "st_temps_manual") == 0){
    // Manual temperature spacing
    int num_temps = io_nexti(line);
    real* temps = (real*) malloc(num_temps*sizeof(real));
    for(int i = 0; i < num_temps; i++){
      temps[i] = io_nextf(line);
    }
    printf("Manual ST Temps: [ ");
    for(int i = 0; i < num_temps; i++){
      printf("%f, ", temps[i]);
    }
    printf("] \n");
    system->enhanced->simulatedTempering->temperatures = temps;
    system->enhanced->simulatedTempering->N_temp = num_temps;
  } else if (strcmp(token, "st_iter") == 0){
    system->enhanced->simulatedTempering->total_iters = io_nexti(line);
  } else if (strcmp(token, "st_history") == 0){
    system->enhanced->simulatedTempering->iter_history = io_nexti(line);
  } else if (strcmp(token, "st_iter_steps") == 0){
    system->enhanced->simulatedTempering->iteration_length = io_nexti(line);
  } else if (strcmp(token, "st_iter_equil_steps") == 0){
    system->enhanced->simulatedTempering->equil_length = io_nexti(line);
  } else if (strcmp(token, "st_sample_freq") == 0){
    system->enhanced->simulatedTempering->sample_freq = io_nexti(line);
  } else if (strcmp(token, "st_temp_sample_freq") == 0){
    system->enhanced->simulatedTempering->temp_sample_freq = io_nexti(line);
  } else if (strcmp(token, "st_do_restart")==0){
    system->enhanced->simulatedTempering->do_restart = io_nextb(line);
  }
};

// Gets called each time a new run function is called
void Enhanced::initialize(System* system){
  if(simulatedTempering) { // reset pointers
    simulatedTempering->initialize(system, output_dir);
  }
  // TODO: Check settings and compatibility
  init = true;
}

void getforce_enhanced(System* system, int step, bool calcEnergy){
  Enhanced* es = system->enhanced;
  if (system->run->calcTermFlag[eeenhanced]==false) return;

  // ST should be last (to capture other bias in scaling?)
  if(es->simulatedTempering){
    getforce_st(system, step, calcEnergy); 
    // Don't update/log/write on pressure coupling moves
    if(es->updating && (step != 0 || step == system->run->step0) ){ 
      es->simulatedTempering->recv_st(); // fetch memory once
      update_st(system); // other internal update timing logic
      if (step % es->log_freq == 0) log_st(system);
      if (step % es->write_small_freq == 0) write_small_st(system, es->output_dir); 
    }
  }
}
