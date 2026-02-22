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
#include "enhanced/its.h"

#include "enhanced/enhanced.h"

Enhanced::Enhanced(){
  its=NULL;
}

Enhanced::~Enhanced(){
  if(its) delete(its);
  if(atom_selection_primary) delete(atom_selection_primary);
  if(atom_selection_secondary) delete(atom_selection_secondary);
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
  } else if (strcmp(token,"atom_selection_2") == 0){
    std::string name=io_nexts(line);
    if (system->selections->selectionMap.count(name)==0) {
      fatal(__FILE__,__LINE__,"Selection %s not found\n",name.c_str());
    }
    system->enhanced->secondary_sele=name;
    if (system->enhanced->atom_selection_secondary) free(system->enhanced->atom_selection_secondary);
    system->enhanced->atom_selection_secondary=(int*)calloc(system->structure->atomList.size(),sizeof(int));
    for (i=0; i<system->structure->atomList.size(); i++) {
      system->enhanced->atom_selection_secondary[i]=system->selections->selectionMap[name].boolSelection[i];
    }
  } else if (strcmp(token,"print_sele")==0){
    printf("Dumping enhanced selections! Will segfault if no selection is defined.\n");
    printf("Primary Selection: %s, Secondary Selection: %s\n", system->enhanced->primary_sele.c_str(), system->enhanced->secondary_sele.c_str());
    system->selections->dump();
  // ITS Reading
  } else if (strcmp(token,"its")==0) {
    if(system->enhanced->its){
      printf("ITS already exists!\n");
      exit(1);
    }
    std::string potential = io_nexts(line); 
    if(potential == "rest"){
      if(!system->enhanced->atom_selection_primary){
        printf("Primary atom selection is NULL!\n");
        exit(1);
      }
      system->enhanced->separate_interactions = true;
      system->enhanced->special_nbdirect = true;
      printf("REST scaling requires separated nbdirect calculation. This may hurt performance!\n");
    }
    if(potential == "torsion"){
      if(!system->enhanced->atom_selection_primary){
        printf("Primary atom selection is NULL!\n");
        exit(1);
      }
      system->enhanced->separate_interactions = true;
      system->enhanced->special_nbdirect = false;
      printf("ITS on selected torsion potential!\n");
    }
    if(potential != "total" && potential != "torsion" && potential != "rest"){
      printf("Unsuppored ITS potential scaling: %s. Availible: total, torsion\n");
      exit(1);
    }
    system->enhanced->its = new Its(potential);
    system->enhanced->active = true;
  } else if(strcmp(token,"its_temps")==0) {
    if(!system->enhanced->its) {
      printf("ITS not defined yet!\n"); 
      exit(1);
    }
    int num_temps = io_nexti(line);
    real temp_low = io_nextf(line);
    real temp_high = io_nextf(line);
    real* temps = (real*) malloc(num_temps*sizeof(real));
    for(int i = 0; i < num_temps; i++){
      temps[i] = temp_low*pow(temp_high/temp_low, ((real)i)/(num_temps-1));
    }
    printf("ITS Temps: [ ");
    for(int i = 0; i < num_temps; i++){
      printf("%f, ", temps[i]);
    }
    printf("] \n");
    system->enhanced->its->temperatures = temps;
    system->enhanced->its->N_temp = num_temps;
  } else if (strcmp(token, "its_steps_per") == 0){
    if(!system->enhanced->its) {
      printf("ITS not defined yet!\n"); 
      exit(1);
    }
    system->enhanced->its->steps_per_temp = io_nexti(line);
  } else if (strcmp(token, "its_flattening_strength") == 0){
    if(!system->enhanced->its) {
      printf("ITS not defined yet!\n"); 
      exit(1);
    }
    system->enhanced->its->correction_strength = io_nextf(line);
  } else if (strcmp(token, "its_sample_freq") == 0){
    if(!system->enhanced->its) {
      printf("ITS not defined yet!"); 
      exit(1);
    }
    system->enhanced->its->sample_freq = io_nexti(line);
  } else if (strcmp(token, "its_update_steps")==0){
    if(!system->enhanced->its) {
      printf("ITS not defined yet!"); 
      exit(1);
    }
    system->enhanced->its->update_steps=io_nexti(line);
  }
};

// Gets called each time a new run function is called
void Enhanced::initialize(System* system){
  if(its) {
    its->initialize(system);
  }
  // TODO: Check settings and compatibility
}

void getforce_enhanced(System* system){
  Enhanced* es = system->enhanced;

  // ITS should be last (capture to capture other bias in scaling)
  if(es->its){
    getforce_its(system); 
    if(es->updating){
      update_its(system); // internal update timing logic
      if (system->run->step % es->log_freq == 0) log_its(system);
      if (system->run->step % es->write_small_freq == 0) write_small_its(system, es->output_dir); 
      if (system->run->step % es->write_big_freq == 0) write_big_its(system, es->output_dir); 
    }
  }
}