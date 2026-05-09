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

#include "enhanced/enhanced.h"
#include "main/gpu_check.h"

Enhanced::Enhanced(){
}

Enhanced::~Enhanced(){
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
  } else if (strcmp(token, "nbrecip_mode")==0){
    int sel = io_nexti(line);
    if(sel < 0 || sel > 2){
      printf("Only 0-2 nbrecip_mode supported {correct, on, off}!\n");
      exit(1);
    }
    system->enhanced->nbrecip_mode=sel;
  } else {
    printf("Didn't recognize option %s\n", token);
  }
};

// Gets called each time a new run function is called
void Enhanced::initialize(System* system){
  
  init = true;
}

void getforce_enhanced(System* system, int step, bool calcEnergy){
  Enhanced* es = system->enhanced;
  if (system->run->calcTermFlag[eeenhanced]==false) return;

}