#include "enhanced/its.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/state.h"
#include "system/system.h"
#include "enhanced/enhanced.h"
#include "system/potential.h"
#include "main/gpu_check.h"

Its::Its(std::string potential){
    this->potential = potential;
    // temperatures' memory alloc & N_temp set in enhanced.cu
}

Its::~Its(){
    if(temperatures) free(temperatures);
    if(temperatures_d) cudaFree(temperatures_d);
    if(its_bias_d) cudaFree(its_bias_d);
    if(its_bias) free(its_bias);
    if(expected_U_d) cudaFree(expected_U_d);
    if(expected_U) free(expected_U);
    if(weighted_U_d) cudaFree(weighted_U_d);
    if(weighted_U) free(weighted_U);
    if(weights_d) cudaFree(weights_d);
    if(weights) free(weights);
    if(alpha_d) cudaFree(alpha_d);
    if(pHist_d) cudaFree(pHist_d);
}

void Its::initialize(){
    if(!temperatures){
      printf("Didn't set temperature range! Use \"enhanced its_temps {N_temp} {T_low} {T_high}\"");
      exit(1);
    }

    // Just assume that if this isn't set all of them are or aren't
    if(!expected_U){
      // N_temp initialized in enhanced.cu
      cudaMalloc(&temperatures_d, N_temp*sizeof(real));
      cudaMemcpy(temperatures_d, temperatures, N_temp*sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&g_k_d, N_temp*sizeof(real));
      cudaMemset(g_k_d, 0, N_temp*sizeof(real));
      g_k = (real*)calloc(N_temp,sizeof(real));
      cudaMalloc(&its_bias_d, N_temp*sizeof(real));
      its_bias = (real*)calloc(N_temp,sizeof(real));
      cudaMalloc(&expected_U_d, N_temp*sizeof(real));
      cudaMemset(expected_U_d, 0, N_temp*sizeof(real));
      expected_U=(real*)calloc(N_temp, sizeof(real));
      cudaMalloc(&weighted_U_d, N_temp*sizeof(real));
      cudaMemset(weighted_U_d, 0, N_temp*sizeof(real));
      weighted_U=(real*)calloc(N_temp, sizeof(real));
      cudaMalloc(&weights_d, N_temp*sizeof(real));
      cudaMemset(weights_d, 0, N_temp*sizeof(real));
      weights=(real*)calloc(N_temp, sizeof(real));
      cudaMalloc(&offsets_d, N_temp*sizeof(real));
      cudaMemset(offsets_d, 0, N_temp*sizeof(real));
      offsets=(real*)calloc(N_temp, sizeof(real));
      for(int i = 0; i < N_temp; i++){
        offsets[i] = -1e9; // offset is largest U value observed
      }
      cudaMemcpy(offsets_d, offsets, N_temp*sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&alpha_d, sizeof(real));
      cudaMemcpy(alpha_d, &alpha_d, sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&weighted_beta_d, sizeof(real));
      weighted_beta = (real*) calloc(1, sizeof(real));
      cudaMalloc(&pHist_d, N_temp*sizeof(real));
      cudaMemset(pHist_d, 0, N_temp*sizeof(real));
      pHist_d = (real*) calloc(1, sizeof(real));
    }

    N_temp_max = N_temp;
    N_temp = 2; // slowly add additional temperatures
}

__global__ void its_logsumexp_kernel(
  int N_temps, real sim_kT, 
  real_e* U_sele, real_e* U_int, real_e* U_solvent,
  real* temps, real* g, real* alpha,
  real* weighted_beta, real* its_bias,
  real* pHist
){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  real_e U_ss=0;
  real_e U_su=0;
  real_e U_uu=0;
  if(U_sele){
    U_ss = U_sele[0];
  }
  if(U_int){
    U_su = U_int[0];
  }
  if(U_solvent){
    U_uu = U_solvent[0];
  }

  real beta_0 = 1.0/sim_kT;
  //printf("U_ss: %f, U_su: %f, U_uu: %f\n", U_ss, U_su, U_uu);
  // Compute the max [beta_k*U_ss + ((1-alpha)*beta_0 + alpha*beta_k)*U_su + beta_0*U_uu + g_k]
  real_e c = -1e9;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k*U_ss + ((1-alpha[0])*beta_0 + alpha[0]*beta_k)*U_su + beta_0*U_uu) + g[i];
    if(exp_arg > c){
      c = exp_arg;
    }
  }
  //printf("Max b*U: %f\n", c);

  // -beta_0 U' = ln(sum_k exp(-beta_k*U + g))
  //            = ln(exp(c) * sum_k exp(-beta_k*U + g - c))
  real exp_sum = 0;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k*U_ss + ((1-alpha[0])*beta_0 + alpha[0]*beta_k)*U_su + beta_0*U_uu) + g[i];
    exp_sum += exp(exp_arg-c);
  }
  real U_prime = -(c+log(exp_sum))/beta_0;

  // exp(-bj*U) = exp(bj*U_bias)*(-b0*U')
  // U_bias = (b0/bk)U' - U
  real U = U_ss + U_su + U_uu;
  *weighted_beta = 0;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    its_bias[i] = (beta_0/beta_k)*U_prime - U;
    real_e exp_arg = -(beta_k*U_ss + ((1-alpha[0])*beta_0 + alpha[0]*beta_k)*U_su + beta_0*U_uu) + g[i];
    real weight = exp(exp_arg-c) / exp_sum;
    pHist[i] += weight;
    *weighted_beta += beta_k*weight;
  }
  *weighted_beta /= beta_0;
  real eff_beta = *weighted_beta*beta_0;
  real eff_T = 1.0/(kB*eff_beta);
}

__global__ void its_update_force_kernel(
  int DOF,  real* weighted_beta, real* alpha,
  real* dU_sel, real* dU_int, real* dU_solv, 
  real* forceBuffer){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if(i < DOF){
    // Check NaN
    real dU_ss = 0;
    real dU_su = 0;
    real dU_uu = 0;
    if(dU_sel){
      dU_ss = dU_sel[i];
    }
    if(dU_int){
      dU_su = dU_sel[i];
    }
    if(dU_solv){
      dU_uu = dU_solv[i];
    }
    real heated = weighted_beta[0]*(dU_ss + alpha[0]*dU_su);
    real bath = dU_uu + (1-alpha[0])*dU_su;
    forceBuffer[i] = heated + bath;
  }
}

__global__ void reduce_total_energy_kernel(real_e* energy){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  for(int i = 0; i < eepotential; i++){
    energy[eepotential] += energy[i];
  }
}

void getforce_its(System* system){
  cudaStream_t stream=0;
  State *s = system->state;
  Its* it = system->enhanced->its;
  int shMem = BLMS*sizeof(real)/32;

  if (system->run) {
    stream = system->run->enhancedStream;
  }

  // Compute bias and forces due to ITS bias with weights
  real kT = kB*system->run->T;
  real_e* U_ss = NULL;
  real* dU_ss= NULL;
  real_e* U_su = NULL;
  real* dU_su= NULL;
  real_e* U_uu = NULL;
  real* dU_uu = NULL;
  if(it->potential == "total"){
    // Reduce energy_d to eepotential
    reduce_total_energy_kernel<<<1,1,0,stream>>>(s->energy_d);
    U_ss = s->energy_d + eepotential;
    dU_ss = s->forceBuffer_d;
  }
  if(it->potential == "torsion"){
    U_ss = s->U_torsion_d;
    dU_ss = s->torsionForceBuffer_d;
  }
  if(it->potential == "rest"){
    // U_ss = s->U_ss_d;
    // dU_ss = s->dU_ss_d;
    // U_su = s->U_su_d;
    // dU_su = s->dU_su_d;
    // U_uu = s->U_uu_d;
    // dU_uu = s->dU_uu_d;
  }
  // Compute <B>
  its_logsumexp_kernel<<<1,1,shMem,stream>>>(
    it->N_temp, kT, 
    U_ss, U_su, U_uu, 
    it->temperatures_d, it->g_k_d, it->alpha_d, 
    it->weighted_beta_d, it->its_bias_d, it->pHist_d);

  // dU/dX = <B>/B0*(dU_ss/dX + a*dU_su/dX) + (1-a)*dU_su/dX + a*dU_uu/dX
  int dof = 3*s->atomCount + s->lambdaCount;
  its_update_force_kernel<<<(dof+BLMS-1)/BLMS,BLMS, 0,stream>>>(
    dof, it->weighted_beta_d, it->alpha_d,
    dU_ss, dU_su, dU_uu,
    s->forceBuffer_d
  );
}

__global__ void update_its_expectation_kernel(
  int N_temps, real* temps, 
  real_e* U_sele, real_e* U_int, 
  real* alpha, real* its_bias,
  real* expected_U, real* weighted_U,
  real* weights, real* offsets
){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  real_e U_ss=0;
  real_e U_su=0;
  if(U_sele){
    U_ss = U_sele[0];
  }
  if(U_int){
    U_su = U_int[0];
  }

  for(int i = 0; i < N_temps; i++){
    real U = U_ss + alpha[0]*U_su;
    real beta_k = 1.0/(kB*temps[i]);
    real reduced_bias = beta_k*its_bias[i];
    if(reduced_bias > offsets[i]){
      // update offset
      real correction = exp(offsets[i] - reduced_bias);
      weights[i] *= correction;
      weighted_U[i] *= correction;
      offsets[i] = reduced_bias;
    }
    weights[i] += exp(reduced_bias - offsets[i]);
    weighted_U[i] += U * exp(reduced_bias - offsets[i]);
    expected_U[i] = weighted_U[i] / weights[i]; // offset cancels
  }
}

__global__ void update_weights_kernel(int N_temps, real* temps, real* expected_U, real* pBeta, real* g){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  g[0] = 0.0;
  real kbi = 1/kB;
  
  if (N_temps < 2){
    return;
  } else if (N_temps == 2) {
    g[1] = (kbi/temps[1] - kbi/temps[0]) * (expected_U[0] + expected_U[1]) / 2.0;
    return;
  }

  real x1 = kbi/temps[0];
  real x2 = kbi/temps[1];
  real x3 = kbi/temps[2];
  real y1 = expected_U[0];
  real y2 = expected_U[1];
  real y3 = expected_U[2];
  real h12 = x2 - x1;
  real h23 = x3 - x2;
  real h13 = x3 - x1;
  real I01 = (h12 / 6.0) *((3.0 - h12 / h13) * y1 + (3.0 + (h12*h12)/(h23*h13) + h12/h13) * y2 - (h12*h12)/(h23*h13) * y3);
  // Integral 1 → 2  (swap roles of 1 and 3)
  real I12 = (h23 / 6.0) * ((3.0 - h23 / h13) * y3+ (3.0 + (h23*h23)/(h12*h13) + h23/h13) * y2 - (h23*h23)/(h12*h13) * y1);
  g[1] = I01;
  g[2] = I01 + I12;
  // ---- Remaining intervals ----
  //printf("g_k: [ %f, %f, %f, ", g[0], g[1], g[2]);
  for (int i = 2; i < N_temps - 1; i++)
  {
    real x1 = kbi/temps[i-1];
    real x2 = kbi/temps[i];
    real x3 = kbi/temps[i+1];

    real y1 = expected_U[i-1];
    real y2 = expected_U[i];
    real y3 = expected_U[i+1];

    real h12 = x2 - x1;
    real h23 = x3 - x2;
    real h13 = x3 - x1;

    // Integral i → i+1
    real I = (h23 / 6.0) * ((3.0 - h23 / h13) * y3 + (3.0 + (h23*h23)/(h12*h13) + h23/h13) * y2 - (h23*h23)/(h12*h13) * y1);

    g[i+1] = g[i] + I;
    //printf("%f, ", g[i+1]);
  }
  //printf("] \n");
};

void update_its(System* system){
  cudaStream_t stream=0;
  State *s = system->state;
  Its* it = system->enhanced->its;
  int shMem = BLMS*sizeof(real)/32;

  if (system->run) {
    stream = system->run->enhancedStream;
  }

  if (system->run->step != 0 && system->run->step % it->sample_freq == 0){
    // Compute bias and forces due to ITS bias with weights
    real kT = kB*system->run->T;
    real_e* U_ss = NULL;
    real* dU_ss= NULL;
    real_e* U_su = NULL;
    real* dU_su= NULL;
    real_e* U_uu = NULL;
    real* dU_uu = NULL;
    if(it->potential == "total"){
      // Reduce energy_d to eepotential already complete
      U_ss = s->energy_d + eepotential;
      dU_ss = s->forceBuffer_d;
    }
    if(it->potential == "torsion"){
      U_ss = s->U_torsion_d;
      dU_ss = s->torsionForceBuffer_d;
    }
    if(it->potential == "rest"){
      // U_ss = s->U_ss_d;
      // dU_ss = s->dU_ss_d;
      // U_su = s->U_su_d;
      // dU_su = s->dU_su_d;
      // U_uu = s->U_uu_d;
      // dU_uu = s->dU_uu_d;
    }

    update_its_expectation_kernel<<<1,1, 0, stream>>>(
      it->N_temp, it->temperatures_d, 
      U_ss, U_su, 
      it->alpha_d, it->its_bias_d,
      it->expected_U_d, it->weighted_U_d,
      it->weights_d, it->offsets_d);

    update_weights_kernel<<<1,1,0,stream>>>(it->N_temp, it->temperatures_d, it->expected_U_d, it->pHist_d, it->g_k_d);
  }

  if(system->run->step != 0 && system->run->step % it->add_temp_every == 0 && it->N_temp < it->N_temp_max){
    printf("Add New Temp!!!!!!!!!!!!!!!!!!!!!\n\n");
    // TODO: Quadratic extrapolation of <U>
    it->N_temp++;
  }
}

void log_its(System* system){

}