#include "enhanced/ldyn_rest.h"
#include "enhanced/enhanced.h"
#include "system/state.h"
#include "system/system.h"
#include "main/real3.h"
#include "run/run.h"
#include "system/potential.h"

Ldyn_rest::Ldyn_rest(real T){
   this->T_high=T; 
}

Ldyn_rest::~Ldyn_rest(){
  if(counts) free(counts);
  if(counts_d) cudaFree(counts_d);
  if(sum_dUdL) free(sum_dUdL);
  if(sum_dUdL_d) cudaFree(sum_dUdL_d);
  if(average_dUdL) free(average_dUdL);
  if(average_dUdL_d) cudaFree(average_dUdL_d);
}

void Ldyn_rest::initialize(System* system){
    Ldyn_rest* ld_rest = system->enhanced->ldyn_rest;

    ld_rest->U_ss_d = system->state->U_ss_d; // (includes torsions)
    ld_rest->dU_ss_d = system->state->dU_ss_buffer_d;
    ld_rest->U_su_d = system->state->U_su_d;
    ld_rest->dU_su_d = system->state->dU_su_buffer_d;

    ld_rest->counts = (int*) calloc(2*ld_rest->bin_count, sizeof(int));
    cudaMalloc(&ld_rest->counts_d, 2*ld_rest->bin_count*sizeof(int));
    cudaMemset(ld_rest->counts_d, 0, 2*ld_rest->bin_count*sizeof(int));
    ld_rest->sum_dUdL = (real*) calloc(2*ld_rest->bin_count, sizeof(real));
    cudaMalloc(&ld_rest->sum_dUdL_d, 2*ld_rest->bin_count*sizeof(real));
    cudaMemset(ld_rest->sum_dUdL_d, 0, 2*ld_rest->bin_count*sizeof(real));
    ld_rest->average_dUdL = (real*)calloc(2*ld_rest->bin_count, sizeof(real));
    cudaMalloc(&ld_rest->average_dUdL_d, 2*ld_rest->bin_count*sizeof(real));
    cudaMemset(ld_rest->average_dUdL_d, 0, 2*ld_rest->bin_count*sizeof(real));

    if (system->state->lambdaCount > 3){
      printf("LDYN_REST does not yet work with multiple lambdas.\n");
      exit(1);
    }
};

__global__ void ldyn_rest_force_kernel(
  int DOF, int L_DOF, real_x* lambdas,
  real_e* U_sel, real_e* U_int, 
  real_f* dU_sel, real_f* dU_int, 
  real T_high, real T0, real alpha,
  real_f* forceBuffer, real_e* energy
){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if(i < DOF){
    // Compute f(L) = 4*sum(li*(1-li))/N
    real f = 0;
    for(int j = 1; j < L_DOF; j++){
        real lj = lambdas[j];
        f += lj*(1.0-lj);
    }
    f *= 4.0/(L_DOF-1.0);

    // Scalings
    real scale_ss = (T_high - T0) / T_high;
    real_f U_ss = U_sel[0];
    real_f dU_ss = dU_sel[i];
    real scale_su = pow(scale_ss, alpha);
    real_f U_su = U_int[0];
    real_f dU_su = dU_int[i];

    // Lambda Force (if i < L_DOF)
    real dfdi = i < L_DOF && i != 0 ? 4.0/(L_DOF-1.0)*(1.0-2.0*lambdas[i]) : 0; // cancel product rule for spatial DOF
    real U = U_ss*scale_ss + U_su*scale_su;
    real pUpi = dU_ss*scale_ss + dU_su*scale_su;
    forceBuffer[i] -= dfdi*U + f*pUpi; // Product rule

    if(i==0){
      // Compute energy = (U_ss*c1 + U_su*c2) * f(l)
      real lEnergy = -f*U;
      atomicAdd(energy, lEnergy);
    }
  }
}

// This histogram index is used for non-periodic directions, where there exist bins on both edges of the period
static __device__ int histogram_index(real val, int num_bins, real max, real min){
  real tmp = val - min;
  real range = max - min;
  real resolution = range / (num_bins-1);
  return round(tmp/resolution);
}

__global__ void ldyn_rest_abf_kernel(
  int L_DOF, real_x* lambdas, real_f* lambdaForce,
  int* count, real* sum_dUdL, real* average_dUdL,
  int ramp_length, int bin_count 
){
  // Add lambda 1 forces
  int id = histogram_index(lambdas[1], bin_count, 1.0, 0);
  sum_dUdL[id] += lambdaForce[1];
  count[id] += 1;
  average_dUdL[id] = sum_dUdL[id] / count[id];
  real abf = 0;
  average_dUdL[id] *= count[id] < ramp_length ? count[id]/ramp_length : 1.0;
  //printf("L0: %f, bin: %d, dUdL-<dUdL>, %f, count: %d\n", lambdas[1], id, lambdaForce[1]-average_dUdL[id], count[id]);
  // Subtract average forces
  atomicAdd(&lambdaForce[1], -average_dUdL[id]);

  // Add lambda 2 forces
  id = bin_count + histogram_index(lambdas[2], bin_count, 1.0, 0);
  sum_dUdL[id] += lambdaForce[2];
  count[id] += 1;
  average_dUdL[id] = sum_dUdL[id] / count[id];
  average_dUdL[id] *= count[id] < ramp_length ? count[id]/ramp_length : 1.0;
  //printf("L1: %f, bin: %d, dUdL-<dUdL>, %f, count: %d\n", lambdas[2], id, lambdaForce[2]-average_dUdL[id], count[id]);
  // Subtract average forces
  atomicAdd(&lambdaForce[2], -average_dUdL[id]);

  id = histogram_index(lambdas[1], bin_count, 1.0, 0); 
  real L_std = .01;
  real L_res = 1.0/(bin_count-1.0);
  int search = 5*ceil(L_std/L_res);
  real bias_mag = .05;
  real U = 0;
  real dUdL = 0;
  for(int j = id-search; j <= id+search; j++){
    int j_mir = j;
    if(j < 0){ j_mir = -j; }
    if(j >= bin_count){ j_mir = bin_count - 2 - (j - bin_count);}
    real L_center = j*L_res;
    real dist = (lambdas[1] - L_center)/L_std;
    real gaussian = bias_mag*exp((real)(-0.5) * dist * dist);
    gaussian *= count[j_mir];
    U += gaussian;
    dUdL += -dist/L_std * gaussian;
  }
  //printf("U: %f, dUdL: %f\n", U, dUdL);
  dUdL /= 2;
  atomicAdd(&lambdaForce[1], dUdL);
  atomicAdd(&lambdaForce[2], -dUdL);
}

void getforce_ldyn_rest(System* system){
  cudaStream_t stream=0;
  State *s = system->state;
  Ldyn_rest* ld = system->enhanced->ldyn_rest;
  int shMem = BLMS*sizeof(real)/32;

  if (system->run) {
    stream = system->run->enhancedStream;
  }

  int dof = 3*s->atomCount + s->lambdaCount;
  ldyn_rest_force_kernel<<<(dof+BLMS-1)/BLMS,BLMS, 0,stream>>>(
    dof, s->lambdaCount, s->lambda_d,
    s->U_ss_d, s->U_su_d,
    s->dU_ss_buffer_d, s->dU_su_buffer_d,
    ld->T_high, system->run->T, ld->alpha,
    s->forceBuffer_d, &s->energy_d[eeenhanced]);

};

void update_ldyn_rest(System* system){
  cudaStream_t stream=0;
  State *s = system->state;
  Ldyn_rest* ld = system->enhanced->ldyn_rest;
  int shMem = BLMS*sizeof(real)/32;

  if (system->run) {
    stream = system->run->enhancedStream;
  }

  if(system->run->step % ld->sample_freq == 0){
    ldyn_rest_abf_kernel<<<1, 1, 0, stream>>>(
      s->lambdaCount, s->lambda_d, s->lambdaForce_d, 
      ld->counts_d, ld->sum_dUdL_d, ld->average_dUdL_d,
      ld->ramp_length, ld->bin_count);
  }
  
}

void log_ldyn_rest(System* system){
  Ldyn_rest* ld = system->enhanced->ldyn_rest;
  cudaMemcpy(ld->average_dUdL, ld->average_dUdL_d, 2*ld->bin_count*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(ld->counts, ld->counts_d, 2*ld->bin_count*sizeof(int), cudaMemcpyDefault);

  if(system->run->step == 0){
    return;
  }

  real sum = 0;
  printf("Step: %d\n", system->run->step);
  printf("Counts: [ ");
  for(int i = 0; i < ld->bin_count; i++){
    printf("%d, ", ld->counts[i]);
  }
  printf("]\n");
  printf("<dUdL1-dUdL2>: [ ");
  for(int i = 0; i < ld->bin_count; i++){
    real dUdL1 = (ld->average_dUdL[i]-ld->average_dUdL[2*ld->bin_count-1-i]); 
    printf("%.2f, ", dUdL1);
  }
  printf("]\n");
  printf("dG: [ 0, ");
  for(int i = 0; i < ld->bin_count-1; i++){
    real dUdL1 = -(ld->average_dUdL[i]-ld->average_dUdL[2*ld->bin_count-1-i]); 
    real dUdL2 = -(ld->average_dUdL[i+1]-ld->average_dUdL[2*ld->bin_count-1-(i+1)]); 
    real w = (1.0/(ld->bin_count-1.0));
    sum += w/2*(dUdL1+dUdL2);
    printf("%.2f, ", sum);
  }
  printf("]\n\n");
};
