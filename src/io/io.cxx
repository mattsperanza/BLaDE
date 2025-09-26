#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
// For arrested_development
#include <signal.h>
#include <unistd.h>

#include "main/defines.h"
#include "system/system.h"

// parse_whatever
#include "io/io.h"

#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <cstdio>  

#include "io/variables.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "system/selections.h"
#include "msld/msld.h"
#include "system/potential.h"
#include "system/state.h"
#include "run/run.h"
#include "xdr/xdrfile.h"
#include "xdr/xdrfile_xtc.h"

void fatal(const char* fnm,int i,const char* format, ...)
{
  va_list args;

  va_start(args,format);
  fprintf(stdout,"FATAL ERROR:\n");
  fprintf(stdout,"%s:%d\n",fnm,i);
  vfprintf(stdout,format,args);
  va_end(args);

  exit(1);
}

void arrested_development(System *system,int howLong) {
  int i;
  char hostname[MAXLENGTHSTRING];
  for (i=0; i<system->idCount; i++) {
#pragma omp barrier
    if (i==system->id) {
      gethostname(hostname,MAXLENGTHSTRING);
      fprintf(stderr,"PID %d rank %d host %s\n",getpid(),i,hostname);
    }
#pragma omp barrier
  }
  sleep(howLong);
}

FILE* fpopen(const char* fnm,const char* type)
{
  FILE *fp;

  fprintf(stdout,"Opening file %s for %s\n",fnm,type);
  fp=fopen(fnm,type);
  if (fp==NULL) {
    fatal(__FILE__,__LINE__,"Error: Unable to open file %s\n",fnm);
  }

  return fp;
}

// for positive i, deletes i characters from beginning of line
void io_shift(char *line,int i)
{
  if (i>0) {
    memmove(line,line+i,strlen(line+i)+1);
  } else {
    memmove(line-i,line,strlen(line)+1);
  }
}

// Read the next string in the line to token, and shift line to after string
void io_nexta(char *line,char *token)
{
  int ntoken, nchar;

  ntoken=sscanf(line,"%s%n",token,&nchar);
  if (ntoken==1 && token[0]!='!') {
    io_shift(line,nchar);
  } else {
    token[0]='\0';
  }
}

// Get next word without advancing
std::string io_peeks(char *line)
{
  char token[MAXLENGTHSTRING];
  int ntoken, nchar;
  std::string output;

  ntoken=sscanf(line,"%s%n",token,&nchar);
  if (ntoken==1 && token[0]!='!') {
    ;
  } else {
    token[0]='\0';
  }

  output=token;
  return output;
}

std::string io_nexts(char *line)
{
  char token[MAXLENGTHSTRING];
  int ntoken, nchar;
  std::string output;

  ntoken=sscanf(line,"%s%n",token,&nchar);
  if (ntoken==1 && token[0]!='!') {
    io_shift(line,nchar);
  } else {
    token[0]='\0';
  }

  output=token;
  return output;
}

std::string io_uppers(std::string input)
{
  char token[MAXLENGTHSTRING];
  int i;
  std::string output;

  if (MAXLENGTHSTRING<=input.length()) fatal(__FILE__,__LINE__,"Error: string to uppercase is too long: %s\n",input.c_str());
  for (i=0; i<input.length(); i++) {
    token[i]=toupper(input.c_str()[i]);
  }
  token[i]='\0';

  output=token;
  return output;
}

bool io_nextb(char *line)
{
  std::string booleanString=io_nexts(line);

  if (booleanString=="on" || booleanString=="true" || booleanString=="yes" || booleanString=="1" || booleanString=="T") {
    return true;
  } else if (booleanString=="off" || booleanString=="false" || booleanString=="no" || booleanString=="0" || booleanString=="F") {
    return false;
  } else {
    fatal(__FILE__,__LINE__,"Error: could not convert string %s to boolean value\n",booleanString.c_str());
  }
  return false;
}

// Read the next int in the line and shift line to after string
int io_nexti(char *line)
{
  int ntoken, nchar;
  int output;

  ntoken=sscanf(line,"%d%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  } else {
    fatal(__FILE__,__LINE__,"Error: failed to read int from: %s\n",line);
  }
  return output;
}

int io_nexti(char *line,int output)
{
  int ntoken, nchar;

  ntoken=sscanf(line,"%d%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  }
  return output;
}

int io_nexti(char *line,FILE *fp,const char *tag)
{
  int ntoken, nchar;
  int output;

  ntoken=0;
  while (ntoken!=1) {
    ntoken=sscanf(line,"%d%n",&output,&nchar);
    if (ntoken==1) {
      io_shift(line,nchar);
      return output;
    }
    if(!fgets(line, MAXLENGTHSTRING, fp)) {
      fatal(__FILE__,__LINE__,"End of file while searching for int value for %s\n",tag);
    }
  }
  return output;
}

real io_nextf(char *line)
{
  int ntoken, nchar;
  double output; // Intentional double

  ntoken=sscanf(line,"%lg%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  } else {
    fatal(__FILE__,__LINE__,"Error: failed to read double from: %s\n",line);
  }
  return output; // Cast to real
}

real io_nextf(char *line,real input)
{
  int ntoken, nchar;
  double output=input; // Intentional double

  ntoken=sscanf(line,"%lg%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  }
  return output; // Cast to real
}

real io_nextf(char *line,FILE *fp,const char *tag)
{
  int ntoken, nchar;
  double output; // Intentional double

  ntoken=0;
  while (ntoken!=1) {
    ntoken=sscanf(line,"%lg%n",&output,&nchar);
    if (ntoken==1) {
      io_shift(line,nchar);
      return output;
    }
    if(!fgets(line, MAXLENGTHSTRING, fp)) {
      fatal(__FILE__,__LINE__,"End of file while searching for real value for %s\n",tag);
    }
  }
  return output; // Cast to real
}

void io_strncpy(char *targ,char *dest,int n)
{
  strncpy(targ,dest,n);
  targ[n]='\0';
}

void interpretter(const char *fnm,System *system)
{
  FILE *fp;
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];
  system->control.push_back(Control());
  int level=system->control.size();

  fp=fpopen(fnm,"r");
  system->control[level-1].fp=fp;

  fgetpos(fp,&system->control[level-1].fp_pos);
  // fsetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    fprintf(stdout,"IN%d> %s",level,line);
    system->variables->substitute(line);
    io_nexta(line,token);
    system->parse_system(line,token,system);
    fgetpos(fp,&system->control[level-1].fp_pos);
    // fsetpos(fp,&fp_pos);
  }

  fclose(fp);
  system->control.pop_back();
}

void print_xtc(int step,System *system)
{
  XDRFILE *fp=system->run->fpXTC;
  float box[3][3]={{0,0,0},{0,0,0},{0,0,0}};
  int i,j,N;

  N=system->state->atomCount;
  if (system->state->typeBox) {
    box[0][0]=system->state->tricBox.a.x/(10*ANGSTROM);
    box[1][0]=system->state->tricBox.b.x/(10*ANGSTROM);
    box[1][1]=system->state->tricBox.b.y/(10*ANGSTROM);
    box[2][0]=system->state->tricBox.c.x/(10*ANGSTROM);
    box[2][1]=system->state->tricBox.c.y/(10*ANGSTROM);
    box[2][2]=system->state->tricBox.c.z/(10*ANGSTROM);
  } else {
    box[0][0]=system->state->orthBox.x/(10*ANGSTROM);
    box[1][1]=system->state->orthBox.y/(10*ANGSTROM);
    box[2][2]=system->state->orthBox.z/(10*ANGSTROM);
  }

  real_x (*x)[3]=system->state->position;
  float (*xXTC)[3]=system->state->positionXTC;
  for (i=0; i<N; i++) {
    for (j=0; j<3; j++) {
      xXTC[i][j]=x[i][j]/(10*ANGSTROM);
    }
  }
//  extern int write_xtc(XDRFILE *xd,
//                       int natoms,int step,real time,
//                       matrix box,rvec *x,real prec);
  write_xtc(fp,N,step,(float) (step*system->run->dt/PICOSECOND),box,xXTC,1000.0);
}

void print_lmd(int step,System *system)
{
  real_x *l=system->state->lambda;
  int i;

  if (system->run->hrLMD) {
    FILE *fp=system->run->fpLMD;
    fprintf(fp,"%10d",step);
    for (i=1; i<system->state->lambdaCount; i++) {
      fprintf(fp," %8.6f",(real)l[i]);
    }
    fprintf(fp,"\n");
  } else {
    XDRFILE *fp=system->run->fpXLMD;
    xdrfile_write_int(&system->state->lambdaCount-1,1,fp);
#if defined DOUBLE || defined DOUBLE_X
    for (i=1; i<system->state->lambdaCount; i++) {
      float lf=l[i];
      xdrfile_write_float(&lf,1,fp);
    }
#else
    xdrfile_write_float(&l[1],system->state->lambdaCount-1,fp);
#endif
  }
}

void print_nrg(int step,System *system)
{
  FILE *fp=system->run->fpNRG;
  real_e *e=system->state->energy;
  int i;

  fprintf(fp,"%10d",step);
  for (i=0; i<eeend; i++) {
    fprintf(fp," %12.4f",e[i]);
  }
  fprintf(fp,"\n");
}

void display_nrg(System *system)
{
  real_e *e=system->state->energy;
  int i;

  for (i=0; i<eeend; i++) {
    fprintf(stdout," %12.4f",e[i]);
  }
  fprintf(stdout,"\n");
}

void print_meta(int step, System* system, bool enhanced_variables){
  FILE *fp=system->run->fpMTD_LMD;
  // Header for lmd file lines: Sites: #, [subs_per_site]
  // Format: Step num, data for site 1, data for site 2, ...,
  if(step == 0){
    fprintf(fp, "Info: %d,", system->msld->siteCount);
    for(int i = 0; i < system->msld->siteCount; i++){
      fprintf(fp, "%d, ", system->msld->blocksPerSite[i]);
    }
    fprintf(fp, "\n");
  }
  for(int i = 0; i < system->msld->blockCount; i++){ 
    fprintf(fp, "%f ", system->state->lambda[i]);
  }
  fprintf(fp, "\n\n");

  fp=system->run->fpMTD_THETA;
  for(int i = 0; i < system->msld->blockCount; i++){
    fprintf(fp, "%f ", system->state->theta[i]);
  }
  fprintf(fp, "\n\n");

  fp=system->run->fpLMD_F;
  for(int i = 0; i < system->msld->blockCount; i++){
    fprintf(fp, "%f ", system->state->lambdaForce[i]);
  }
  fprintf(fp, "\n\n");

  fp=system->run->fpTHETA_F;
  for(int i = 0; i < system->msld->blockCount; i++){
    fprintf(fp, "%f ", system->state->thetaForce[i]);
  }
  fprintf(fp, "\n\n");

  if(enhanced_variables){
    fp=system->run->fpMTD_dUdL;
    for(int i = 0; i < system->msld->blockCount; i++){
      fprintf(fp, "%f ", system->msld->dUdL_msld[i]);
    }
    fprintf(fp, "\n\n");

    fp=system->run->fpMTD_dUdT;
    for(int i = 0; i < system->msld->blockCount; i++){
      fprintf(fp, "%f ", system->msld->dUdT_msld[i]);
    }
    fprintf(fp, "\n\n");
  
    fp=system->run->fpMTD_dUdT_abf;
    for(int i = 0; i < system->msld->siteCount; i++){
      fprintf(fp, "%f ", system->msld->dUdT_abf[i]);
    }
    fprintf(fp, "\n\n");
  
    fp=system->run->fpMTD_HIST;
    for(int i = 0; i < system->msld->siteCount; i++){
      fprintf(fp, "%f ", system->msld->oss_bias[i]);
    }
    fprintf(fp, "\n\n");

    fp=system->run->fpMTD_BIAS;
    fprintf(fp, "%f ", 0); // TODO: Calculate this
    fprintf(fp, "\n\n");
  }

  fflush(NULL);
}

void print_dynamics_output(int step,System *system)
{
  if (system->id==0) {
    if (step % system->run->freqXTC == 0) {
      system->state->recv_position();
      system->state->prettify_position(system);
      print_xtc(step,system);
    }
    if (step % system->run->freqLMD == 0) {
      system->state->recv_lambda();
      print_lmd(step,system);
    }
    if (step % system->run->freqNRG == 0) {
      system->state->recv_energy();
      print_nrg(step,system);
    }
    if (step % system->run->freqMTD == 0) {
      system->state->recv_lambda();
      // Only write things like dUdL_msld/dUdT_msld/bias_pot if they are calculated
      bool enhanced_variables = (system->msld->oss || system->msld->LE || system->msld->abf || system->msld->meta);
      if(enhanced_variables) { system->msld->recv_meta(); }
      print_meta(step, system, enhanced_variables);
    }
  }
}

void write_checkpoint_file(const char *fnm,System *system)
{
  FILE *fp;
  int i;

  if (system->id==0) {
    fp=fpopen(fnm,"w");

    system->state->recv_state();

    fprintf(fp,"Step %d\n",system->run->step0);

    fprintf(fp,"Position %d\n",system->state->atomCount);
    for (i=0; i<system->state->atomCount; i++) {
      fprintf(fp,"%f ",system->state->position[i][0]);
      fprintf(fp,"%f ",system->state->position[i][1]);
      fprintf(fp,"%f\n",system->state->position[i][2]);
    }

    fprintf(fp,"Velocity %d\n",system->state->atomCount);
    for (i=0; i<system->state->atomCount; i++) {
      fprintf(fp,"%f ",system->state->velocity[i][0]);
      fprintf(fp,"%f ",system->state->velocity[i][1]);
      fprintf(fp,"%f\n",system->state->velocity[i][2]);
    }

    fprintf(fp,"ThetaPos %d\n",system->state->lambdaCount);
    for (i=0; i<system->state->lambdaCount; i++) {
      fprintf(fp,"%f\n",system->state->theta[i]);
    }

    fprintf(fp,"ThetaVel %d\n",system->state->lambdaCount);
    for (i=0; i<system->state->lambdaCount; i++) {
      fprintf(fp,"%f\n",system->state->thetaVelocity[i]);
    }

    fprintf(fp,"Box\n");
    fprintf(fp,"%f %f %f\n",system->state->box.a.x,system->state->box.a.y,system->state->box.a.z);
    fprintf(fp,"%f %f %f\n",system->state->box.b.x,system->state->box.b.y,system->state->box.b.z);

    fclose(fp);
  }
}

void write_1D(Msld* msld, std::ofstream& file, real* T_1D, int* lengths){
  Msld* m = msld;
  int accum=0;
  for (int i = 1; i < m->siteCount; i++) {
    file << "# Site " << i << " Bins " << lengths[i] << "\n";
    for (int j = accum; j < accum+lengths[i]; j++) {
        file << i << ", " << j << ", " << T_1D[j] << "\n";
    }
    accum+=lengths[i];
  }
}

void write_2D(Msld* msld, std::ofstream& file, real* T_2D, int* lengths, int vert_bins){
  Msld* m = msld;
  real accum=0;
  for (int i = 1; i < m->siteCount; i++) {
    file << "# Theta " << i << " T_bins: " << lengths[i] << " T_range: [" << 0 << ", " << m->site_period[i] << "]"
       << " dUdT_bins: " << vert_bins << " dUdT_range: [" << m->dUdT_min << ", " << m->dUdT_max << "] " << "\n";
    for (int j = 0; j < lengths[i]; j++) {
      for (int k = 0; k < vert_bins; k++) {
        int idx = (accum+j)*vert_bins + k;
        file << i << ", " << j << ", " << k << ", " << T_2D[idx] << "\n";
      }
    }
    accum+= lengths[i];
  }
}

void write_histogram_file(System* system, std::string file_name, bool potential) {
  // Write to temporary file
  std::string temp_file_name = file_name + ".tmp";
  
  // Print path to file
  char cwd[1024];
  std::ofstream file(temp_file_name);
  if (!file) {
    std::cerr << "Error: Unable to open file " << temp_file_name << " for writing." << std::endl;
    return;
  }
  Msld* m = system->msld;
  if (m->oss || m->LE || m->abf || m->meta) {
    // Print average dUdL
    int nS = m->siteCount;
    file << "# Num_Sites: " << nS-1 << "\n"; // will be read in as num histograms
  
    // Write 1D info
    int total_bins = m->total_T_bins;
    real* T_1D = (real*)malloc(total_bins * sizeof(real));

    file << "# Theta Counts\n";
    cudaMemcpy(T_1D, m->theta_histogram_d, total_bins*sizeof(real), cudaMemcpyDefault);
    write_1D(m, file, T_1D, m->T_bins);

    // ABF
    file << "# ABF <dU/dT>\n";
    cudaMemcpy(T_1D, m->abf_ensemble_dUdT_d, total_bins*sizeof(real), cudaMemcpyDefault);
    write_1D(m, file, T_1D, m->T_bins);

    file << "# ABF Weights\n";
    cudaMemcpy(T_1D, m->abf_weights_d, total_bins*sizeof(real), cudaMemcpyDefault);
    write_1D(m, file, T_1D, m->T_bins);

    file << "# ABF Offsets\n";
    cudaMemcpy(T_1D, m->abf_offsets_d, total_bins*sizeof(real), cudaMemcpyDefault);
    write_1D(m, file, T_1D, m->T_bins);

    file << "# ABF Weighted dU/dT\n";
    cudaMemcpy(T_1D, m->abf_weighted_dUdT_d, total_bins*sizeof(real), cudaMemcpyDefault);
    write_1D(m, file, T_1D, m->T_bins);

    file << "# ABF Weighted dU/dT^2\n";
    cudaMemcpy(T_1D, m->abf_weighted_dUdT2_d, total_bins*sizeof(real), cudaMemcpyDefault);
    write_1D(m, file, T_1D, m->T_bins);

    file << "# ABF Weighted dU/dT^2\n";
    cudaMemcpy(T_1D, m->abf_weighted_dUdT2_d, total_bins*sizeof(real), cudaMemcpyDefault);
    write_1D(m, file, T_1D, m->T_bins);
    free(T_1D);

    // LE
    file << "# R counter\n";
    real* R = (real*) malloc(m->siteCount*sizeof(real));
    cudaMemcpy(R, m->LE_R_d, m->siteCount*sizeof(real), cudaMemcpyDefault);
    for(int i = 0; i < m->siteCount; i++){
      file << i << ", " << R[i] << "\n";
    }
    free(R);

    file << "# LE Memory\n";
    int LE_bins = m->LE_total_bins;
    real* LE_T = (real*) malloc(LE_bins*sizeof(real));
    cudaMemcpy(LE_T, m->LE_M_d, LE_bins*sizeof(real), cudaMemcpyDefault);
    write_1D(m, file, LE_T, m->LE_bins);
    free(LE_T);

    real* hist_2D = (real*)malloc(total_bins*m->dUdT_bins*sizeof(real));
    real* hist_2D_d = potential ? m->potential_2D_d : m->hist_weights_2D_d;
    cudaMemcpy(hist_2D, hist_2D_d, total_bins*m->dUdT_bins*sizeof(real), cudaMemcpyDefault);
    file << "# Histogram Potential\n";
    write_2D(m, file, hist_2D, m->T_bins, m->dUdT_bins);
    free(hist_2D);
  }

  file.close();
  
  // Atomic
  if (std::rename(temp_file_name.c_str(), file_name.c_str()) != 0) {
    std::cerr << "Error: Failed to rename " << temp_file_name << " to " << file_name << std::endl;
  }
}

// Helper: read one 1D block into host array
void read_1D(Msld* m, std::ifstream& file, real* host_array, int* lengths) {
  int accum = 0;
  std::string line;
  for (int i = 1; i < m->siteCount; i++) {
    // Read site header line
    if (!std::getline(file, line)) return;
    int site, bins;
    sscanf(line.c_str(), "# Site %d Bins %d", &site, &bins);
    // Now read bins lines
    for (int j = 0; j < bins; j++) {
      if (!std::getline(file, line)) return;
      int isite, idx;
      real val;
      char comma;
      std::stringstream ss(line);
      ss >> isite >> comma >> idx >> comma >> val;
      host_array[accum + j] = val;
    }
    accum += bins;
  }
}

void read_2D(Msld* m, std::ifstream& file, real* host_array_2D, int* lengths, int vert_bins){
  int accum = 0;
  std::string line;
  for (int i = 1; i < m->siteCount; i++) {
    // Read site descriptor line
    std::getline(file, line);
    for (int j = 0; j < lengths[i]; j++) {
      for (int k = 0; k < vert_bins; k++) {
        if (!std::getline(file, line)) break;
        int isite, jj, kk;
        real val;
        char comma;
        std::stringstream ss(line);
        ss >> isite >> comma >> jj >> comma >> kk >> comma >> val;
        int idx = (accum + jj) * vert_bins + kk;
        host_array_2D[idx] = val;
      }
    }
    accum += lengths[i];
  }
}


bool read_histogram_file(System* system, std::string file_name) {
    std::ifstream file(file_name);
    if (!file) {
        std::cerr << "Error: Unable to open file " << file_name << " for reading." << std::endl;
        return true; // file doesn't exist, nothing is wrong
    }

    Msld* m = system->msld;
    std::string line;
    int nSites = 0;

    int total_bins = m->total_T_bins;
    real* T_1D = (real*)malloc(total_bins * sizeof(real));
    int counter = -1;
    while (std::getline(file, line)) {
        counter++;
        if (line.rfind("# Num_Sites:", 0) == 0) {
            sscanf(line.c_str(), "# Num_Sites: %d", &nSites);
        } else if (line.find("# Theta Counts") != std::string::npos) {
            read_1D(m, file, T_1D, m->T_bins);
            cudaMemcpy(m->theta_histogram_d, T_1D, total_bins*sizeof(real), cudaMemcpyDefault);
        }
        else if (line.find("# ABF <dU/dT>") != std::string::npos){
          read_1D(m, file, T_1D, m->T_bins);
          cudaMemcpy(m->abf_ensemble_dUdT_d, T_1D, total_bins*sizeof(real), cudaMemcpyDefault);
        }
        else if (line.find("# ABF Weights") != std::string::npos) {
            read_1D(m, file, T_1D, m->T_bins);
            cudaMemcpy(m->abf_weights_d, T_1D, total_bins*sizeof(real), cudaMemcpyDefault);
        }
        else if (line.find("# ABF Offsets") != std::string::npos) {
            read_1D(m, file, T_1D, m->T_bins);
            cudaMemcpy(m->abf_offsets_d, T_1D, total_bins*sizeof(real), cudaMemcpyDefault);
        }
        else if (line.find("# ABF Weighted dU/dT") != std::string::npos) {
            read_1D(m, file, T_1D, m->T_bins);
            cudaMemcpy(m->abf_weighted_dUdT_d, T_1D, total_bins*sizeof(real), cudaMemcpyDefault);
        }
        else if (line.find("# ABF Weighted dU/dT^2") != std::string::npos) {
            read_1D(m, file, T_1D, m->T_bins);
            cudaMemcpy(m->abf_weighted_dUdT2_d, T_1D, total_bins*sizeof(real), cudaMemcpyDefault);
        }
        else if (line.find("# R counter") != std::string::npos) {
            // Parse LE_R_d (no bins, just siteCount lines)
            real* R = (real*)malloc(m->siteCount*sizeof(real));
            for (int i = 0; i < m->siteCount; i++) {
                if (!std::getline(file, line)) break;
                int idx;
                real val;
                char comma;
                std::stringstream ss(line);
                ss >> idx >> comma >> val;
                R[idx] = val;
            }
            cudaMemcpy(m->LE_R_d, R, m->siteCount*sizeof(real), cudaMemcpyDefault);
            free(R);
        }
        else if (line.find("# LE Memory") != std::string::npos) {
            int LE_bins = m->LE_total_bins;
            real* LE_T = (real*)malloc(LE_bins*sizeof(real));
            read_1D(m, file, LE_T, m->LE_bins);
            cudaMemcpy(m->LE_M_d, LE_T, LE_bins*sizeof(real), cudaMemcpyDefault);
            free(LE_T);
        }
        else if (line.find("# Histogram Potential") != std::string::npos) {
            // Read restartable 2D histogram data
            real* hist_2D = (real*)malloc(total_bins*m->dUdT_bins*sizeof(real));
            read_2D(m, file, hist_2D, m->T_bins, m->dUdT_bins);
            cudaMemcpy(m->hist_weights_2D_d, hist_2D, total_bins*m->dUdT_bins*sizeof(real), cudaMemcpyDefault);
            free(hist_2D);
        } else if (!file.eof()){
          // incorrect settings if reading things in wrong order - kill restart attempt
          std::cerr << "Error reading item #" << counter << ": " << line << std::endl;
          return false;
        }
    }
    free(T_1D);
    file.close();
    return true;
}

void read_checkpoint_file(const char *fnm,System *system)
{
  FILE *fp;
  int i;
  double v;

  if (system->id==0) {
    fp=fpopen(fnm,"r");
    fscanf(fp,"Step %ld\n",&system->run->step0);

    fscanf(fp,"Position %d\n",&i);
    if (i!=system->state->atomCount) {
      fatal(__FILE__,__LINE__,"Checkpoint file is not compatible with system setup. Wrong number of atoms\n");
    }
    for (i=0; i<system->state->atomCount; i++) {
      fscanf(fp,"%lf ",&v);
      system->state->position[i][0]=v;
      fscanf(fp,"%lf ",&v);
      system->state->position[i][1]=v;
      fscanf(fp,"%lf\n",&v);
      system->state->position[i][2]=v;
    }

    fscanf(fp,"Velocity %d\n",&i);
    if (i!=system->state->atomCount) {
      fatal(__FILE__,__LINE__,"Checkpoint file is not compatible with system setup. Wrong number of atoms\n");
    }
    for (i=0; i<system->state->atomCount; i++) {
      fscanf(fp,"%lf ",&v);
      system->state->velocity[i][0]=v;
      fscanf(fp,"%lf ",&v);
      system->state->velocity[i][1]=v;
      fscanf(fp,"%lf\n",&v);
      system->state->velocity[i][2]=v;
    }

    fscanf(fp,"ThetaPos %d\n",&i);
    if (i!=system->state->lambdaCount) {
      fatal(__FILE__,__LINE__,"Checkpoint file is not compatible with system setup. Wrong number of alchemical coordinates\n");
    }
    for (i=0; i<system->state->lambdaCount; i++) {
      fscanf(fp,"%lf\n",&v);
      system->state->theta[i]=v;
    }

    fscanf(fp,"ThetaVel %d\n",&i);
    if (i!=system->state->lambdaCount) {
      fatal(__FILE__,__LINE__,"Checkpoint file is not compatible with system setup. Wrong number of alchemical coordinates\n");
    }
    for (i=0; i<system->state->lambdaCount; i++) {
      fscanf(fp,"%lf\n",&v);
      system->state->thetaVelocity[i]=v;
    }

    fscanf(fp,"Box\n");
    fscanf(fp,"%lf\n",&v);
    system->state->box.a.x=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.a.y=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.a.z=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.b.x=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.b.y=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.b.z=v;

    fclose(fp);

    system->state->send_state();
    if (system->msld->fix) { // ffix
      cudaMemcpy(system->state->lambda_d,system->state->theta,system->state->lambdaCount*sizeof(real_x),cudaMemcpyHostToDevice);
    }
    system->msld->calc_lambda_from_theta(0,system);
  }
}

void blade_interpretter(const char *fnm,System *system)
{
  system+=omp_get_thread_num();
  interpretter(fnm,system);
}
