#include "enhanced/its.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/state.h"
#include "system/system.h"
#include "enhanced/enhanced.h"
#include "system/potential.h"

Its::Its(std::string potential){
    this->potential = potential;
    // temperatures memory alloc in enhanced.cu
}

Its::~Its(){
    if(temperatures) free(temperatures);
    if(its_bias_d) free(its_bias_d);
    if(its_bias) free(its_bias);
    if(expected_U_d) cudaFree(expected_U_d);
    if(expected_U) free(expected_U);
    if(weighted_U_d) cudaFree(weighted_U_d);
    if(weighted_U) free(weighted_U);
    if(weights_d) cudaFree(weights_d);
    if(weights) free(weights);
    if(alpha_d) cudaFree(alpha_d);
}

void Its::initialize(){
    if(!temperatures){
        printf("Didn't set temperature range! Use \"enhanced its_temps {N_temp} {T_low} {T_high}\"");
    }

    // Just assume that if this isn't set all of them are or aren't
    if(!expected_U){
        cudaMalloc(&its_bias_d, N_temp*sizeof(real));
        its_bias = (real*)calloc(N_temp,sizeof(real));
        cudaMalloc(&expected_U_d, N_temp*sizeof(real));
        cudaMemset(expected_U_d, 0, N_temp*sizeof(real));
        expected_U=(real*)calloc(N_temp, sizeof(real));
        cudaMalloc(&weighted_U_d, N_temp*sizeof(real));
        cudaMemset(weighted_U, 0, N_temp*sizeof(real));
        weighted_U=(real*)calloc(N_temp, sizeof(real));
        cudaMalloc(&weights_d, N_temp*sizeof(real));
        cudaMemset(weights_d, 0, N_temp*sizeof(real));
        weights=(real*)calloc(N_temp, sizeof(real));
        cudaMalloc(&alpha_d, sizeof(real));
        cudaMemcpy(alpha_d, &alpha_d, sizeof(real), cudaMemcpyDefault);
    }
}

__global__ void getforce_its_kernel(
    int N_temps, real sim_kT, 
    real_e* U_sele, real_e* U_int, real_e* U_solvent,
    real* temps, real* g, real* alpha,
    real* its_bias
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
  // Compute the max [beta_k*U_ss + ((1-alpha)*beta_0 + alpha*beta_k)*U_su + beta_0*U_uu + g_k]
  real_e max_beta_U = -1e9;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k*U_ss + ((1-alpha[0])*beta_0 + alpha[0]*beta_k)*U_su + beta_0*U_uu) + g[i];
    if(exp_arg > max_beta_U){
        max_beta_U = exp_arg;
    }
  }

  // -beta_0 U' = ln(sum_k exp(-beta_k*U + g))
  //            = ln(exp(c) * sum_k exp(-beta_k*U + g - c))
  real U_prime = max_beta_U;
  real exp_sum = 0;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k*U_ss + ((1-alpha[0])*beta_0 + alpha[0]*beta_k)*U_su + beta_0*U_uu) + g[i];
    exp_sum += exp_arg;
  }
  U_prime += log(exp_sum);
  U_prime /= -beta_0;

  // exp(-bj*U) = exp(bj*U_bias)*(-b0*U')
  // U_bias = (b0/bk)U' - U
  real U = U_ss + U_su + U_uu;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    its_bias[i] = (beta_0/beta_k) * U_prime - U;
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
    real_e* U_su = NULL;
    real_e* U_uu = NULL;
    if(it->potential == "total"){
        U_ss = s->energy_d + eepotential;
    }
    if(it->potential == "torsion"){
        U_ss = s->U_torsion_d;
    }
    if(it->potential == "rest"){
        // Need to assign U_ss = U_pp + U_tors, U_su = U_pw, U_uu = U_ww
    }
    getforce_its_kernel<<<1,1,shMem,stream>>>(it->N_temp, kT, U_ss, U_su, U_uu, it->temperatures, it->g_k, it->alpha_d, it->its_bias_d);
}

void update_its(System* system){
    Its* it = system->enhanced->its;
    int shMem = BLMS*sizeof(real)/32;

}