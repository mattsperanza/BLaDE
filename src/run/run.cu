#include <omp.h>
#include <cuda_runtime.h>
#include <string.h>

#include "run/run.h"

#include <float.h>

#include "system/system.h"
#include "io/io.h"
#include "msld/msld.h"
#include "system/state.h"
#include "system/potential.h"
#include "system/selections.h"
#include "holonomic/rectify.h"
#include "domdec/domdec.h"
#include "main/gpu_check.h"

#include <iostream>    // For std::cout, std::cerr
#include <fstream>     // For std::ofstream
#include <unistd.h>    // For getcwd


#ifdef REPLICAEXCHANGE
#include <mpi.h>
#endif



// #warning "Hardcoded serial kernels"
// #define PROFILESERIAL

// Class constructors
Run::Run(System *system)
{
  step0=0;
  nsteps=5000;
  dt=0.001*PICOSECOND; // ps
  T=300; // K
  gamma=1.0/PICOSECOND; // ps^-1
  fnmXTC="default.xtc";
  fnmLMD="default.lmd";
  fnmNRG="default.nrg";
  fnmCPI="";
  fnmCPO="default.cpt";
  fpXTC=NULL;
  fpXLMD=NULL;
  fpLMD=NULL;
  fpNRG=NULL;
  freqXTC=1000;
  freqLMD=10;
  freqNRG=10;
  hrLMD=true;
  prettyXTC=false;
// Nonbonded options
  betaEwald=1/(3.2*ANGSTROM); // rCut=10*ANGSTROM, erfc(betaEwald*rCut)=1e-5
  rCut=10*ANGSTROM;
  rSwitch=8.5*ANGSTROM;
  vfSwitch=true;
  usePME=true;
  gridSpace=1.0*ANGSTROM;
  grid[0]=-1;
  grid[1]=-1;
  grid[2]=-1;
  orderEwald=6;

  cutoffs.betaEwald=betaEwald;
  cutoffs.rCut=rCut;
  cutoffs.rSwitch=rSwitch;

  shakeTolerance=2e-7; // floating point precision is only 1.2e-7

  freqNPT=50;
  volumeFluctuation=100*ANGSTROM*ANGSTROM*ANGSTROM;
  pressure=1*ATMOSPHERE;

  // minimization options
  dxAtomMax=0.1*ANGSTROM;
  dxRMSInit=0.05*ANGSTROM;
  dxRMS=dxRMSInit;
  minType=esd; // enum steepest descent

  domdecHeuristic=true;

  termStringToInt.clear();
  termStringToInt["bond"]=eebond;
  termStringToInt["angle"]=eeangle;
  termStringToInt["urey"]=eeurey;
  termStringToInt["dihe"]=eedihe;
  termStringToInt["impr"]=eeimpr;
  termStringToInt["cmap"]=eecmap;
  termStringToInt["nb14"]=eenb14;
  termStringToInt["nbdirect"]=eenbdirect;
  termStringToInt["nbrecip"]=eenbrecip;
  termStringToInt["nbrecipself"]=eenbrecipself;
  termStringToInt["nbrecipexcl"]=eenbrecipexcl;
  termStringToInt["lambda"]=eelambda;
  termStringToInt["bias"]=eebias;
  termStringToInt["potential"]=eepotential;
  termStringToInt["kinetic"]=eekinetic;
  termStringToInt["total"]=eetotal;
  calcTermFlag.clear();
  for (int i=0; i<eeend; i++) {
    calcTermFlag[i]=true;
  }

#ifdef REPLICAEXCHANGE
  fnmREx="default.rex";
  fpREx=NULL;
  freqREx=-1;
  MPI_Comm_rank(MPI_COMM_WORLD, &replica);
#endif

#ifdef PROFILESERIAL
  updateStream=0;
  bondedStream=0;
  biaspotStream=0;
  nbdirectStream=0;
  nbrecipStream=0;

  abfBias=0;
  ossBias=0;
  ossBonded=0;
  ossDirect=0;
  ossRecip=0;
#else
  cudaStreamCreate(&updateStream);
  cudaStreamCreate(&bondedStream);
  cudaStreamCreate(&biaspotStream);
  cudaStreamCreate(&nbdirectStream);
  cudaStreamCreate(&nbrecipStream);
  // ABF Bias Streams
  cudaStreamCreate(&abfBias);
  // Orthogonal Bias streams
  cudaStreamCreate(&ossBias);
  cudaStreamCreate(&ossBonded);
  cudaStreamCreate(&ossDirect);
  cudaStreamCreate(&ossRecip);

  // Set priorities if desired:
  // int low,high;
  // cudaDeviceGetStreamPriorityRange(&low,&high);
  // cudaStreamCreateWithPriority(&nbdirectStream,cudaStreamDefault,low);
  // cudaStreamCreateWithPriority(&nbrecipStream,cudaStreamDefault,high);
#endif
  cudaEventCreate(&forceBegin);
  cudaEventCreate(&bondedComplete);
  cudaEventCreate(&biaspotComplete);
  cudaEventCreate(&nbdirectComplete);
  cudaEventCreate(&nbrecipComplete);
  // cudaEventCreate(&forceComplete);
  cudaEventCreate(&communicate);

  // ABF Bias Events
  cudaEventCreate(&abfBiasComplete);
  // Orthogonal Bias events
  cudaEventCreate(&ossForceBegin);
  cudaEventCreate(&ossBiasComplete);
  cudaEventCreate(&ossBondedComplete);
  cudaEventCreate(&ossDirectComplete);
  cudaEventCreate(&ossRecipComplete);


  if (system->idCount>0) {
    communicate_omp=(cudaEvent_t*)calloc(system->idCount,sizeof(cudaEvent_t));
#pragma omp barrier
    system->message[system->id]=(void*)&communicate;
#pragma omp barrier
    for (int i=0; i<system->idCount; i++) {
      communicate_omp[i]=((cudaEvent_t*)(system->message[i]))[0];
    }
#pragma omp barrier
  } else {
    communicate_omp=NULL;
  }

  setup_parse_run();
}

Run::~Run()
{
  if (fpXTC) xdrfile_close(fpXTC);
  if (fpXLMD) xdrfile_close(fpXLMD);
  if (fpLMD) fclose(fpLMD);
  if (fpNRG) fclose(fpNRG);
#ifndef PROFILESERIAL
  cudaStreamDestroy(updateStream);
#endif
  cudaEventDestroy(forceBegin);
  // cudaEventDestroy(forceComplete);

#ifndef PROFILESERIAL
  cudaStreamDestroy(bondedStream);
  cudaStreamDestroy(biaspotStream);
  cudaStreamDestroy(nbdirectStream);
  cudaStreamDestroy(nbrecipStream);
#endif
  cudaEventDestroy(bondedComplete);
  cudaEventDestroy(biaspotComplete);
  cudaEventDestroy(nbdirectComplete);
  cudaEventDestroy(nbrecipComplete);
  cudaEventDestroy(communicate);
  if (communicate_omp) free(communicate_omp);
}



// Parsing functions
void parse_run(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  if (!system->run) {
    system->run=new Run(system);
  }

  io_nexta(line,token);
  name=token;
  if (system->run->parseRun.count(name)==0) name="";
  // So much for function pointers being elegant.
  // call the function pointed to by: system->run->parseRun[name]
  // within the object: system->run
  // with arguments: (line,token,system)
  (system->run->*(system->run->parseRun[name]))(line,token,system);
}

void Run::setup_parse_run()
{
  parseRun[""]=&Run::error;
  helpRun[""]="If you see this string, something went wrong.\n";
  parseRun["help"]=&Run::help;
  helpRun["help"]="?run help [directive]> Prints help on run directive, including a list of subdirectives. If a subdirective is listed, this prints help on that specific subdirective.\n";
  parseRun["print"]=&Run::dump;
  helpRun["print"]="?run print> Prints out some of the data in the run data structure\n";
  parseRun["reset"]=&Run::reset;
  helpRun["reset"]="?run reset> Resets the run data structure to it's default values\n";
  parseRun["setvariable"]=&Run::set_variable;
  helpRun["setvariable"]="?run setvariable \"name\" \"value\"> Set the variable \"name\" to \"value\". Available \"name\"s are: dt (time step in ps), nsteps (number of steps of dynamics to run), fnmxtc (filename for the coordinate output), fnmlmd (filename for the lambda output), fnmnrg (filename for the energy output)\n";
  parseRun["setterm"]=&Run::set_term;
  helpRun["setterm"]="?run setterm [term] [on|off]> Turn terms (including bond, angle, dihe, impr, nb14 nbdirect, nbrecip, nbrecipself, nbrecipexcl, lambda, and bias) on or off\n";
  parseRun["energy"]=&Run::energy;
  helpRun["energy"]="?run energy> Calculate energy of current conformation or from fnmcpi checkpoint in file\"\n";
  parseRun["test"]=&Run::test;
  helpRun["test"]="?run test [arguments]> Test first derivatives using finite differences. Valid arguments are \"alchemical [difference]\" and \"spatial [selection] [difference]\"\n";
  parseRun["minimize"]=&Run::minimize;
  helpRun["minimize"]="?run minimize> Minimize structure with the options set by \"run setvariable\"\n";
  parseRun["dynamics"]=&Run::dynamics;
  helpRun["dynamics"]="?run dynamics> Run dynamics with the options set by \"run setvariable\"\n";
}

void Run::help(char *line,char *token,System *system)
{
  std::string name=io_nexts(line);
  if (name=="") {
    fprintf(stdout,"?run> Available directives are:\n");
    for (std::map<std::string,std::string>::iterator ii=helpRun.begin(); ii!=helpRun.end(); ii++) {
      fprintf(stdout," %s",ii->first.c_str());
    }
    fprintf(stdout,"\n");
  } else if (helpRun.count(token)==1) {
    fprintf(stdout,helpRun[name].c_str());
  } else {
    error(line,token,system);
  }
}

void Run::error(char *line,char *token,System *system)
{
  fatal(__FILE__,__LINE__,"Unrecognized token after run: %s\n",token);
}

void Run::dump(char *line,char *token,System *system)
{
  fprintf(stdout,"RUN PRINT> dt=%f (time step input in ps)\n",dt/PICOSECOND);
  fprintf(stdout,"RUN PRINT> T=%f (temperature in K)\n",T);
  fprintf(stdout,"RUN PRINT> gamma=%f (friction input in ps^-1)\n",gamma*PICOSECOND);
  fprintf(stdout,"RUN PRINT> nsteps=%d (number of time steps for dynamics)\n",nsteps);
  fprintf(stdout,"RUN PRINT> fnmxtc=%s (file name for coordinate trajectory)\n",fnmXTC.c_str());
  fprintf(stdout,"RUN PRINT> fnmlmd=%s (file name for lambda trajectory)\n",fnmLMD.c_str());
  fprintf(stdout,"RUN PRINT> fnmnrg=%s (file name for energy output)\n",fnmNRG.c_str());
  fprintf(stdout,"RUN PRINT> fnmcpi=%s (file name for reading checkpoint in, null means start without checkpoint)\n",fnmCPI.c_str());
  fprintf(stdout,"RUN PRINT> fnmcpo=%s (file name for writing out checkpoint file for later continuation)\n",fnmCPO.c_str());
  fprintf(stdout,"RUN PRINT> betaEwald=%f (input 1/invbetaewald in A^-1)\n",betaEwald*ANGSTROM);
  fprintf(stdout,"RUN PRINT> rcut=%f (input in A)\n",rCut/ANGSTROM);
  fprintf(stdout,"RUN PRINT> rswitch=%f (input in A)\n",rSwitch/ANGSTROM);
  fprintf(stdout,"RUN PRINT> vfswitch=%d\n",vfSwitch);
  fprintf(stdout,"RUN PRINT> usepme=%d\n",usePME);
  fprintf(stdout,"RUN PRINT> gridspace=%f (For PME - input in A)\n",gridSpace/ANGSTROM);
  fprintf(stdout,"RUN PRINT> grid=[%d %d %d] (For PME if gridspace<0)\n",grid[0],grid[1],grid[2]);
  fprintf(stdout,"RUN PRINT> orderewald=%d (PME interpolation order, dimensionless. 4, 6, 8, or 10 supported, 6 recommended)\n",orderEwald);
  fprintf(stdout,"RUN PRINT> shaketolerance=%f (For use with shake - dimensionless - do not go below 1e-7 with single precision)\n",shakeTolerance);
  fprintf(stdout,"RUN PRINT> freqnpt=%d (frequency of pressure coupling moves. 10 or less reproduces bulk dynamics, OpenMM often uses 100)\n",freqNPT);
  fprintf(stdout,"RUN PRINT> volumefluctuation=%f (rms volume move for pressure coupling, input in A^3, recommend sqrt(V*(1 A^3)), rms fluctuations are typically sqrt(V*(2 A^3))\n",volumeFluctuation/(ANGSTROM*ANGSTROM*ANGSTROM));
  fprintf(stdout,"RUN PRINT> pressure=%f (pressure for pressure coupling, input in atmospheres)\n",pressure/ATMOSPHERE);
  fprintf(stdout,"RUN PRINT> dxatommax=%f (Maximum minimization atom displacement in A)\n",dxAtomMax/ANGSTROM);
  fprintf(stdout,"RUN PRINT> dxrmsinit=%f (Starting minimization rms displacement in A)\n",dxRMSInit/ANGSTROM);
  fprintf(stdout,"RUN PRINT> mintype=%d (minimization algorithm. 0 is steepest descent, etc)\n",minType);
  fprintf(stdout,"RUN PRINT> domdecheuristic=%d (use heuristics for domdec limits without checking their validity)\n",(int)domdecHeuristic);
#ifdef REPLICAEXCHANGE
  fprintf(stdout,"RUN PRINT> fnmrex=%s (file name for replica exchange)\n",fnmREx.c_str());
  fprintf(stdout,"RUN PRINT> freqrex=%d (frequency of replica exchange attempts. Use {rexrank} (NYI) to access 0 ordinalized replica index in script)\n",freqREx);
#endif
}

void Run::reset(char *line,char *token,System *system)
{
  delete system->run;
  system->run=NULL;
}

void Run::set_variable(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (strcmp(token,"dt")==0) {
    dt=PICOSECOND*io_nextf(line);
  } else if (strcmp(token,"nsteps")==0) {
    nsteps=io_nexti(line);
  } else if (strcmp(token,"fnmxtc")==0) {
    if (fpXTC) xdrfile_close(fpXTC);
    fpXTC=NULL;
    fnmXTC=io_nexts(line);
  } else if (strcmp(token,"fnmlmd")==0) {
    if (fpXLMD) xdrfile_close(fpXLMD);
    fpXLMD=NULL;
    if (fpLMD) fclose(fpLMD);
    fpLMD=NULL;
    fnmLMD=io_nexts(line);
  } else if (strcmp(token,"fnmnrg")==0) {
    if (fpNRG) fclose(fpNRG);
    fpNRG=NULL;
    fnmNRG=io_nexts(line);
  } else if (strcmp(token,"fnmcpi")==0) {
    fnmCPI=io_nexts(line);
  } else if (strcmp(token,"fnmcpo")==0) {
    fnmCPO=io_nexts(line);
  } else if (strcmp(token,"freqxtc")==0) {
    freqXTC=io_nexti(line);
  } else if (strcmp(token,"freqlmd")==0) {
    freqLMD=io_nexti(line);
  } else if (strcmp(token,"freqnrg")==0) {
    freqNRG=io_nexti(line);
  } else if (strcmp(token,"hrlmd")==0) {
    hrLMD=io_nextb(line);
  } else if (strcmp(token,"prettyxtc")==0) {
    prettyXTC=io_nextb(line);
  } else if (strcmp(token,"T")==0) {
    T=io_nextf(line);
  } else if (strcmp(token,"gamma")==0) {
    gamma=io_nextf(line)/PICOSECOND;
  } else if (strcmp(token,"invbetaewald")==0) {
    betaEwald=1/(io_nextf(line)*ANGSTROM);
    cutoffs.betaEwald=betaEwald;
  } else if (strcmp(token,"rcut")==0) {
    rCut=io_nextf(line)*ANGSTROM;
    cutoffs.rCut=rCut;
  } else if (strcmp(token,"rswitch")==0) {
    rSwitch=io_nextf(line)*ANGSTROM;
    cutoffs.rSwitch=rSwitch;
  } else if (strcmp(token,"vfswitch")==0) {
    vfSwitch=io_nextb(line);
  } else if (strcmp(token,"usepme")==0) {
    usePME=io_nextb(line);
  } else if (strcmp(token,"gridspace")==0) {
    gridSpace=io_nextf(line)*ANGSTROM;
  } else if (strcmp(token,"grid")==0) {
    grid[0]=io_nexti(line);
    grid[1]=io_nexti(line);
    grid[2]=io_nexti(line);
    gridSpace=-1;
  } else if (strcmp(token,"orderewald")==0) {
    orderEwald=io_nexti(line);
    if ((orderEwald/2)*2!=orderEwald) fatal(__FILE__,__LINE__,"orderEwald (%d) must be even\n",orderEwald);
    if (orderEwald<4 || orderEwald>8) fatal(__FILE__,__LINE__,"orderEwald (%d) must be 4, 6, or 8\n",orderEwald);
  } else if (strcmp(token,"shaketolerance")==0) {
    shakeTolerance=io_nextf(line);
  } else if (strcmp(token,"freqnpt")==0) {
    freqNPT=io_nexti(line);
  } else if (strcmp(token,"volumefluctuation")==0) {
    volumeFluctuation=io_nextf(line)*ANGSTROM*ANGSTROM*ANGSTROM;
  } else if (strcmp(token,"pressure")==0) {
    pressure=io_nextf(line)*ATMOSPHERE;
  } else if (strcmp(token,"dxatommax")==0) {
    dxAtomMax=io_nextf(line)*ANGSTROM;
  } else if (strcmp(token,"dxrmsinit")==0) {
    dxRMSInit=io_nextf(line)*ANGSTROM;
  } else if (strcmp(token,"mintype")==0) {
    std::string minString=io_nexts(line);
    if (strcmp(minString.c_str(),"sd")==0) {
      minType=esd;
    } else if (strcmp(minString.c_str(),"sdfd")==0) {
      minType=esdfd;
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized token %s for minimization type minType. Options are: sd or sdfd\n",minString.c_str());
    }
  } else if (strcmp(token,"domdecheuristic")==0) {
    domdecHeuristic=io_nextb(line);
#ifdef REPLICAEXCHANGE
  } else if (strcmp(token,"fnmrex")==0) {
    if (fpREx) fclose(fpREx);
    fpREx=NULL;
    fnmREx=io_nexts(line);
  } else if (strcmp(token,"freqrex")==0) {
    freqREx=io_nexti(line);
#endif
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token %s in run setvariable command\n",token);
  }
}

void Run::set_term(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (termStringToInt.count(token)) {
    calcTermFlag[termStringToInt[token]]=io_nextb(line);
  } else {
    fatal(__FILE__,__LINE__,"No such energy term %s to be turned on or off\n",token);
  }
}

__global__
void shift_kernel(real_x *x,real_x dx)
{
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if (i==0) {
    x[0]+=dx;
  }
}

__global__ void shift_lambda(real* position, real dl, int index) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if (i==0) {
    position[index] += dl;
  }
}

void Run::energy(char *line,char *token,System *system)
{
  dynamics_initialize(system);
  system->potential->calc_force(0,system);
  system->state->recv_energy();
  print_nrg(0,system);
  dynamics_finalize(system);
}

void test_OST_force(System *system) {
  // This is way too small with float precision
  real dl = .001;
  bool flags[eeend];
  for (int i = 0; i < eeend; i++) {
    flags[i] = system->run->calcTermFlag[i];
    system->run->calcTermFlag[i] = false;
  }

  printf("Starting OSS Force Test...\n");
  bool ossPrior = system->msld->oss;
  int len = system->state->lambdaCount+3*system->state->atomCount; // theta forces at end
  for (int i = 0; i < eeend; i++) {
    printf("Term %d Numerical Test: \n", i);
    system->run->calcTermFlag[i] = true;
    // Calculate ref dU(X, L) to subtract from force w/ oss
    real_f dU[len];
    system->msld->oss = false;
    gpuCheck(cudaPeekAtLastError());
    system->potential->calc_force(1, system);
    cudaDeviceSynchronize();
    system->msld->oss = true;
    cudaMemcpy(dU, system->state->forceBuffer_d, len*sizeof(real), cudaMemcpyDeviceToHost);
    double sum = 0;
    double num_sum = 0;
    for (int j = 1; j < system->state->lambdaCount; j++) { // skip environment lambda
      // Set dGdF[:] = 0 and dGdF[j] = e * pi
      real pi_e = M_E * M_PI;
      real_f dGdF[system->state->lambdaCount];
      memset(dGdF, 0, system->state->lambdaCount*sizeof(real));
      dGdF[j] = pi_e;
      cudaMemcpy(system->msld->dGdF_d, dGdF, system->state->lambdaCount*sizeof(real_f), cudaMemcpyDefault);
      system->msld->oss = false;
      // Numerical Force = dU(X, L+dl) - dU(X, L-dl) / 2*dl
      // L+dl
      real_f tmp_high[len];
      // lambda block at beginning of position buffer, everything else is index into this array
      cudaDeviceSynchronize();
      shift_lambda<<<1, 1>>>(system->state->positionBuffer_fd, dl, j);
      cudaDeviceSynchronize();
      gpuCheck(cudaPeekAtLastError());
      system->potential->calc_force(1, system);
      cudaDeviceSynchronize();
      cudaMemcpy(tmp_high, system->state->forceBuffer_d, len*sizeof(real_f), cudaMemcpyDefault);
      // L-dl
      real_f tmp_low[len];
      cudaDeviceSynchronize();
      shift_lambda<<<1, 1>>>(system->state->positionBuffer_fd, -2.0*dl, j);
      cudaDeviceSynchronize();
      system->potential->calc_force(1, system);
      cudaDeviceSynchronize();
      cudaMemcpy(tmp_low, system->state->forceBuffer_d, len*sizeof(real_f), cudaMemcpyDefault);
      // Reset and diff
      real_f d2U_numeric[len];
      cudaDeviceSynchronize();
      shift_lambda<<<1, 1>>>(system->state->positionBuffer_fd, dl, j);
      for (int k = 0; k < len; k++) {
        d2U_numeric[k] = (tmp_high[k] - tmp_low[k]) / (2.0*dl);
        num_sum += abs(d2U_numeric[k]);
      }
      // Analytic Force
      real_f d2U_analytic[len];
      system->msld->oss = true;
      cudaDeviceSynchronize();
      system->potential->calc_force(1, system);
      cudaDeviceSynchronize();
      cudaMemcpy(d2U_analytic, system->state->forceBuffer_d, len*sizeof(real_f), cudaMemcpyDefault);
      cudaDeviceSynchronize();
      for (int k = 0; k < len; k++) {
        d2U_analytic[k] = (d2U_analytic[k] - dU[k]) / pi_e;
        sum += abs(d2U_analytic[k]);
      }
      // Check if they match
      for (int k = 0; k < len; k++) {
        real_f diff = abs(d2U_analytic[k] - d2U_numeric[k]);
        real_f tol = .1;
        if (diff > tol) { // floating point ops (like expf) can cause float errors
          printf("Numerical derivatives test %d failed (tol = %f, dl = %f) at force array index %d for lambda %d! \n",
            i, tol, dl, k, j);
          printf("Num lambdas: %d \n", system->state->lambdaCount);
          real l_sum = 0;
          system->state->recv_lambda();
          printf("Lambdas: [ ");
          for (int l = 1; l < system->state->lambdaCount; l++) {
            printf("%.3f, ", system->state->lambda[l]);
            l_sum += system->state->lambda[l];
          }
          printf("] --> Sum: %.12f\n", l_sum);
          real lmd = system->state->lambda[j];
          system->state->recv_position();
          real x = system->state->positionBuffer[k];
          printf("Position[%d] = %f\n", k, x);
          printf("Analytic dU(X,L+dl):          %15.8f\n", tmp_high[k]);
          printf("Analytic dU(X,L=%.2f):        %15.8f\n", lmd, dU[k]);
          printf("Analytic dU(X,L-dl):          %15.8f\n", tmp_low[k]);
          printf("\n|Diff|:                       %15.8f\n", diff);
          printf("Numeric d2U:                  %15.8f\n", d2U_numeric[k]);
          printf("Analytic d2U:                 %15.8f\n", d2U_analytic[k]);
          printf("Scaling num->analytic:        %15.8f\n", d2U_analytic[k]/d2U_numeric[k]);
          printf("Scaling analytic->num:        %15.8f\n\n", d2U_numeric[k]/d2U_analytic[k]);
          //printf("Exiting...\n");
          //exit(1);
        }
      }
    }
    printf("Total Forces Calculated: %lf analytic, %lf numeric\n\n", sum, num_sum);
    gpuCheck(cudaPeekAtLastError());
    system->run->calcTermFlag[i] = false;
  }

  // Just in case things get run after this
  system->msld->oss = ossPrior;
  for (int i = 0; i < eeend; i++) { 
    system->run->calcTermFlag[i] = flags[i];
  }
}

void test_OSS_conservation(System* system) {
  int nL = system->state->lambdaCount-1; // no environment

  // NPT/NVT for 100k steps adding bias
  int total = 1000010;
  int updating = total * .5;
  printf("Running %d steps with bias+update and %d steps just bias to equilibrate!\n", updating, total-updating);
  system->run->freqNRG = 1;
  for (int step=0; step<total; step++) { 
    system->run->step = step;
    system->domdec->update_domdec(system,(step%system->domdec->freqDomdec)==0);
    system->potential->calc_force(step,system);
    gpuCheck(cudaPeekAtLastError());
    system->state->update(step,system);
    gpuCheck(cudaPeekAtLastError());

    if(step == updating){
      printf("\n\n\nTurning off FE estimation updating!\n\n\n");
      system->msld->update_fe_surface = false;
    }
    system->state->recv_energy();
    if (isnan(system->state->energy[eetotal]) || isinf(system->state->energy[eetotal])) {
      printf("Something went wrong!!\n");
      printf("Energies: \n");
      for (int i = 0; i < eetotal; i++) {
        printf("Term %d: %f\n", i, system->state->energy[i]);
      }
      exit(-1);
    }

    if(step % 1000 == 0){
      printf("Step: %d, Pot: %f, Kin: %f, Tot: %f\n",step,system->state->energy[eepotential],system->state->energy[eekinetic],system->state->energy[eetotal]);
      real dUdL[nL+1], dU_msld[nL+1];
      cudaMemcpy(dUdL, system->state->lambdaForce_d, (nL+1)*sizeof(real), cudaMemcpyDefault);
      cudaMemcpy(dU_msld, system->msld->dU_msld_d, (nL+1)*sizeof(real), cudaMemcpyDefault);
      printf("Lambda Force: [ ");
      for (int i = 0; i < nL+1; i++) {
        printf("%f -> %f, ", dU_msld[i], dUdL[i]);
      }
      printf(" ]\n");
    }
  }

  // Output Histogram data to histograms.txt
  //write_histogram_file(system, "histograms.txt");

  // Turn on NVE for 500k w/ oss & abf update off
  printf("Running 500k steps with oss to check energy conservation!\n");
  system->run->freqNPT = 0; // turn off pressure coupling
  system->state->leapParms1->gamma = 0; // turn off langevin
  system->state->recv_energy(); // get kinetic energy
  real eStart = system->state->energy[eetotal];
  system->msld->update_fe_surface = false;
  printf("Starting energy conservation with:\n Pot: %f\n, Kin: %f\n, Tot: %f\n",
   system->state->energy[eepotential],
   system->state->energy[eekinetic],
   system->state->energy[eetotal]);
  for (int step=1; step<1000000; step++) {
    system->domdec->update_domdec(system,(step%system->domdec->freqDomdec)==0);
    system->potential->calc_force(step,system);
    gpuCheck(cudaPeekAtLastError());
    system->state->update(step,system);
    print_dynamics_output(step,system);
    system->state->recv_energy();
    // 1 kT = .5922 kcal/mol at 298K?
    real diff = system->state->energy[eetotal] - eStart;
    real drift = abs(system->state->energy[eetotal] - eStart) * (1/.5922);
    real drift_step_dof = drift / (step*system->state->leapParms1->dt/1000) / (system->state->atomCount*3 + system->state->lambdaCount - 2);
    printf("Step: %d, Pot: %f, Kin: %f, Tot: %f, Start Tot: %f, Tot - Start (kcal/mol): %f, |Drift| (kT): %f, Drift/step/dof: %f\n",
      step,
      system->state->energy[eepotential],
      system->state->energy[eekinetic],
      system->state->energy[eetotal],
      eStart, diff,
      drift, drift_step_dof);
    if (isnan(system->state->energy[eetotal]) || isinf(system->state->energy[eetotal])) {
      printf("Something went wrong!!\n");
      printf("Energies: \n");
      for (int i = 0; i < eetotal; i++) {
        printf("Term %d: %f\n", i, system->state->energy[i]);
      }
      exit(-1);
    }
  }
}

void Run::test(char *line,char *token,System *system)
{
  std::string testType=io_nexts(line);
  std::string name; // (selection name for spatial test)
  real dx;
  int i,j,ij,s;
  int ij0,imax,jmax;
  real_e F,E[2];

  // Initialize data structures
  dynamics_initialize(system);

  // Calculate forces
  //system->potential->calc_force(0,system);
  // Save position and forces
  //system->state->backup_position();

  if (testType=="alchemical") {
    dx=io_nextf(line); // dimensionless
    ij0=0;
    imax=system->state->lambdaCount;
    jmax=1;
  } else if (testType=="spatial") {
    name=io_nexts(line);
    if (system->selections->selectionMap.count(name)==0) {
      fatal(__FILE__,__LINE__,"Error: selection %s not found for spatial derivative testing\n",name.c_str());
    }
    dx=io_nextf(line)*ANGSTROM; // ANGSTROM units
    ij0=system->state->lambdaCount;
    imax=system->state->atomCount;
    jmax=3;
  } else if (testType=="oss_force") {
    test_OST_force(system);
    return;
  } else if (testType=="oss_energy_cons") {
    if(system->msld->oss || system->msld->abf){
      test_OSS_conservation(system);
    } else {
      printf("OSS or ABF boolean not set!!!\n");
      printf("OSS or ABF boolean not set!!!\n");
      printf("OSS or ABF boolean not set!!!\n");
      printf("Not running energy conservation test!!!\n");
    }
    return;
  } else if (testType=="print_histogram") {
    write_histogram_file(system, "histograms.txt", true);
    return;
  } else {
    fatal(__FILE__,__LINE__,"Error: test type %s does not match alchemical, spatial, oss_force, or oss_energy_cons \n",testType.c_str());
  }

  for (i=0; i<imax; i++) {
    if (jmax==1 || system->selections->selectionMap[name].boolSelection[i]) {
      for (j=0; j<jmax; j++) {
        ij=ij0+i*jmax+j;
        for (s=0; s<2; s++) {
          // Shift ij by (s-0.5)*dx
          shift_kernel<<<1,1>>>(&system->state->positionBuffer_d[ij],(s-0.5)*dx);
          
          // Calculate energy
          system->domdec->update_domdec(system,0);
          system->potential->calc_force(0,system);

          // Save relevant data
          if (system->id==0) {
            system->state->recv_energy();
            E[s]=system->state->energy[eepotential];
          }

          // Restore positions
          system->state->restore_position();
        }
        if (system->id==0) {
          cudaMemcpy(&F,&system->state->forceBuffer_d[ij],sizeof(real),cudaMemcpyDeviceToHost);
          fprintf(stdout,"ij=%7d, Emin=%20.16g, Emax=%20.16g, (Emax-Emin)/dx=%20.16g, force=%20.16g\n",ij,E[0],E[1],(E[1]-E[0])/dx,F);
        }
      }
    }
  }

  dynamics_finalize(system);
}

void Run::minimize(char *line,char *token,System *system)
{
  dynamics_initialize(system);
  system->state->min_init(system);

  for (step=0; step<nsteps; step++) {
    system->domdec->update_domdec(system,true); // true to always update neighbor list
    system->potential->calc_force(0,system); // step 0 to always calculate energy
    system->state->min_move(step,nsteps,system);
    print_dynamics_output(step,system);
    gpuCheck(cudaPeekAtLastError());
  }

  system->state->min_dest(system);
  dynamics_finalize(system);
}

void Run::dynamics(char *line,char *token,System *system)
{
  clock_t t1,t2;

  // Initialize data structures
  dynamics_initialize(system);

  // Run dynamics
  t1=clock();
  for (step=step0; step<step0+nsteps; step++) {
    if (system->verbose>0) {
      fprintf(stdout,"Step %d\n",step);
    }
    system->domdec->update_domdec(system,(step%system->domdec->freqDomdec)==0);
    system->potential->calc_force(step,system);
    system->state->update(step,system);
#warning "Need to copy coordinates before update"
    print_dynamics_output(step,system);
    gpuCheck(cudaPeekAtLastError());
  }
  t2=clock();
// Note: omp_get_wtime may be of more interest when parallelizing
  fprintf(stdout,"Elapsed dynamics time: %f\n",(t2-t1)*1.0/CLOCKS_PER_SEC);

  dynamics_finalize(system);
}

void Run::dynamics_initialize(System *system)
{
  // Open files
  if (!fpXTC) {
    fpXTC=xdrfile_open(fnmXTC.c_str(),"w");
    if (!fpXTC) {
      fatal(__FILE__,__LINE__,"Failed to open XTC file %s\n",fnmXTC.c_str());
    }
  }
  if (hrLMD) {
    if (!fpLMD) fpLMD=fpopen(fnmLMD.c_str(),"w");
  } else {
    if (!fpXLMD) fpXLMD=xdrfile_open(fnmLMD.c_str(),"w");
    if (!fpXLMD) fatal(__FILE__,__LINE__,"Failed to open LMD file %s\n",fnmLMD.c_str());
  }
  if (!fpNRG) fpNRG=fpopen(fnmNRG.c_str(),"w");
#ifdef REPLICAEXCHANGE
  if (!fpREx && freqREx>0) fpREx=fpopen(fnmREx.c_str(),"w");
#endif

  // Finish setting up MSLD
  system->msld->initialize(system);

  // Set up update structures
  if (system->state) delete system->state;
  system->state=new State(system);
  system->state->initialize(system);

  // Set up potential structures
  if (system->potential) delete system->potential;
  system->potential=new Potential();
  system->potential->initialize(system);

  // Rectify bond constraints
  holonomic_rectify(system);

  // Read checkpoint
  if (fnmCPI!="") {
    read_checkpoint_file(fnmCPI.c_str(),system);
  }

  // Set up domain decomposition
  if (system->domdec) delete system->domdec;
  system->domdec=new Domdec();
  system->domdec->initialize(system);

  cudaDeviceSynchronize();
#pragma omp barrier
  gpuCheck(cudaPeekAtLastError());
#pragma omp barrier
}

void Run::dynamics_finalize(System *system)
{
  step0=step;
  write_checkpoint_file(fnmCPO.c_str(),system);
  system->state->save_state(system);  
}



void blade_init_run(System *system)
{
  system+=omp_get_thread_num();
  if (system->run) {
    delete(system->run);
  }
  system->run=new Run(system);
}

void blade_dest_run(System *system)
{
  system+=omp_get_thread_num();
  if (system->run) {
    delete(system->run);
  }
  system->run=NULL;
}

void blade_add_run_flags(System *system,
  double gamma,
  double betaEwald,
  double rCut,
  double rSwitch,
  int vdWfSwitch,
  int elecPME,
  double gridSpace,
  int gridx,
  int gridy,
  int gridz,
  int orderEwald,
  double shakeTolerance)
{
  system+=omp_get_thread_num();
  system->run->gamma=gamma;

  system->run->betaEwald=betaEwald;
  system->run->rCut=rCut;
  system->run->rSwitch=rSwitch;
  system->run->vfSwitch=vdWfSwitch==1;
  system->run->usePME=elecPME==1;
  system->run->gridSpace=gridSpace; // grid spacing for PME calculation
  system->run->grid[0]=gridx; // if gridSpace is negative, use these values
  system->run->grid[1]=gridy; // if gridSpace is negative, use these values
  system->run->grid[2]=gridz; // if gridSpace is negative, use these values
  system->run->orderEwald=orderEwald; // interpolation order (4, 6, or 8 typically)
  system->run->shakeTolerance=shakeTolerance;

  system->run->cutoffs.betaEwald=betaEwald;
  system->run->cutoffs.rCut=rCut;
  system->run->cutoffs.rSwitch=rSwitch;
}

void blade_add_run_dynopts(System *system,
  int step,
  int step0,
  int nsteps,
  double dt,
  double T,
  int freqNPT,
  double volumeFluctuation,
  double pressure)
{
  system+=omp_get_thread_num();
  system->run->step=step; // current step
  system->run->step0=step0; // starting step
  system->run->nsteps=nsteps; // steps in next dynamics call
  system->run->dt=dt;
  system->run->T=T;

  system->run->freqNPT=freqNPT;
  system->run->volumeFluctuation=volumeFluctuation;
  system->run->pressure=pressure;
}

void blade_run_energy(System *system)
{
  system+=omp_get_thread_num();
  
  if (!system->run) {
    system->run=new Run(system);
  }
  system->run->dynamics_initialize(system);
  system->potential->calc_force(0,system);
  system->state->recv_energy();
  system->run->dynamics_finalize(system);
}
