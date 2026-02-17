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
    if(weights_d) cudaFree(weights_d);
    if(offsets_d) cudaFree(offsets_d);
    if(alpha_d) cudaFree(alpha_d);
    if(pHist_d) cudaFree(pHist_d);
    if(U_prime_d) cudaFree(U_prime_d);
    if(g_k) free(g_k);
    if(g_k_d) cudaFree(g_k_d);
    if(weighted_beta_d) cudaFree(weighted_beta_d);
    if(U_d) cudaFree(U_d);
    if(correction_strength_d) cudaFree(correction_strength_d);
    if(fp_beta) fclose(fp_beta);
    if(fp_exp_U) fclose(fp_exp_U);
    if(fp_g) fclose(fp_g);
    if(fp_red_bias) fclose(fp_red_bias);
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
      cudaMalloc(&weights_d, N_temp*sizeof(real));
      cudaMemset(weights_d, 0, N_temp*sizeof(real));
      cudaMalloc(&offsets_d, N_temp*sizeof(real));
      cudaMemset(offsets_d, 0, N_temp*sizeof(real));
      real offsets[N_temp];
      for(int i = 0; i < N_temp; i++){
        offsets[i] = -1e9; // offset is largest U value observed
      }
      cudaMemcpy(offsets_d, offsets, N_temp*sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&alpha_d, sizeof(real));
      cudaMemcpy(alpha_d, &alpha_d, sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&weighted_beta_d, sizeof(real));
      cudaMalloc(&pHist_d, N_temp*sizeof(real));
      cudaMemset(pHist_d, 0, N_temp*sizeof(real));
      pHist = (real*) calloc(N_temp, sizeof(real));
      cudaMalloc(&U_prime_d, sizeof(real_e));
      cudaMalloc(&U_d, sizeof(real_e));
      cudaMalloc(&correction_strength_d, sizeof(real));
      cudaMemcpy(correction_strength_d, &correction_strength, sizeof(real), cudaMemcpyDefault);
    }

    N_temp_max = N_temp;
    N_temp = 2; // slowly add additional temperatures
}

__global__ void its_logsumexp_kernel(
  // Input
  int N_temps, real sim_kT, 
  real_e* U_sele, real_e* U_int, real_e* U_solvent,
  real* temps, real* g, real* alpha,
  // Output
  real* weighted_beta, real* its_bias,
  real* pHist, real_e* U_prime
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

  // Compute the max [beta_k*U_ss + ((1-alpha)*beta_0 + alpha*beta_k)*U_su + beta_0*U_uu + g_k]
  real beta_0 = 1.0/sim_kT;
  real_e c = -1e9;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k*U_ss + ((1-alpha[0])*beta_0 + alpha[0]*beta_k)*U_su + beta_0*U_uu) + g[i];
    if(exp_arg > c){
      c = exp_arg;
    }
  }
  // -beta_0 U' = ln(sum_k exp(-beta_k*U + g))
  //            = ln(exp(c) * sum_k exp(-beta_k*U + g - c))
  real exp_sum = 0;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k*U_ss + ((1-alpha[0])*beta_0 + alpha[0]*beta_k)*U_su + beta_0*U_uu) + g[i];
    exp_sum += exp(exp_arg-c);
  }
  *U_prime = -(c+log(exp_sum))/beta_0;
  // exp(-bj*U) = exp(bj*U_bias)*(-b0*U')
  // U_bias = (b0/bk)U' - U
  real U = U_ss + U_su + U_uu;
  *weighted_beta = 0;
  for(int i = 0; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    its_bias[i] = (beta_0/beta_k)*(*U_prime + g[i]/beta_0) - U;
    real_e exp_arg = -(beta_k*U_ss + ((1-alpha[0])*beta_0 + alpha[0]*beta_k)*U_su + beta_0*U_uu) + g[i];
    real weight = exp(exp_arg-c) / exp_sum;
    pHist[i] += weight;
    *weighted_beta += beta_k*weight;
  }
  *weighted_beta /= beta_0;
}

__global__ void its_update_force_kernel(
  int DOF,  real* weighted_beta, real* alpha,
  real* dU_sel, real* dU_int, real* dU_solv, 
  bool torsion,
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
      dU_su = dU_int[i];
    }
    if(dU_solv){
      dU_uu = dU_solv[i];
    }
    real heated = weighted_beta[0]*(dU_ss + alpha[0]*dU_su);
    real bath = dU_uu + (1-alpha[0])*dU_su;
    if(!torsion){
      forceBuffer[i] = heated + bath;
    } else {
      // remove existing forces and replace with scaled version
      forceBuffer[i] += (weighted_beta[0] - 1)*dU_ss;
    }
  }
}

__global__ void reduce_total_energy_kernel(real_e* energy, real_e* U){
  *U=0;
  for(int i = 0; i < eepotential; i++){
    *U += energy[i];
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
  bool torsion = false;
  if(it->potential == "total"){
    // Reduce energy_d to eepotential
    reduce_total_energy_kernel<<<1,1,0,stream>>>(s->energy_d, it->U_d);
    U_ss = it->U_d;
    dU_ss = s->forceBuffer_d;
  }
  else if(it->potential == "torsion"){
    U_ss = s->U_torsion_d;
    dU_ss = s->torsionForceBuffer_d;
    torsion=true;
  }
  else if(it->potential == "rest"){
    // U_ss = s->U_ss_d; (needs to include torsions)
    // dU_ss = s->dU_ss_d;
    // U_su = s->U_su_d;
    // dU_su = s->dU_su_d;
    // U_uu = s->U_uu_d;
    // dU_uu = s->dU_uu_d;
  }
  // Compute <B>
  its_logsumexp_kernel<<<1,1,shMem,stream>>>(
    it->N_temp, kT, U_ss, U_su, U_uu, 
    it->temperatures_d, it->g_k_d, it->alpha_d, 
    it->weighted_beta_d, it->its_bias_d, it->pHist_d, it->U_prime_d);

  // dU/dX = <B>/B0*(dU_ss/dX + a*dU_su/dX) + (1-a)*dU_su/dX + a*dU_uu/dX
  int dof = 3*s->atomCount + s->lambdaCount;
  its_update_force_kernel<<<(dof+BLMS-1)/BLMS,BLMS, 0,stream>>>(
    dof, it->weighted_beta_d, it->alpha_d,
    dU_ss, dU_su, dU_uu, torsion,
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

  if(i < N_temps){
    real_e U_ss=0;
    real_e U_su=0;
    if(U_sele){
      U_ss = U_sele[0];
    }
    if(U_int){
      U_su = U_int[0];
    }
  
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

__global__ void update_weights_kernel(int N_temps, real* temps, real* expected_U, real* correction_strength, real* pBeta, real* g){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  real eps = 1e-10;
  real scale = *correction_strength;
  real uniform_FE = -log(1.0/N_temps);
  real norm = pBeta[0];
  g[0] = 0;
  real kbi = 1/kB;
  
  if (N_temps < 2){
    return;
  } else if (N_temps == 2) {
    g[1] = (kbi/temps[1] - kbi/temps[0]) * (expected_U[0] + expected_U[1]) / 2.0;
    norm = pBeta[0] + pBeta[1];
    g[0] += scale*(-log(pBeta[0]/norm + eps) - uniform_FE);
    g[1] += scale*(-log(pBeta[1]/norm + eps) - uniform_FE);
    return;
  }

  // See scipy cumulative simpsons
  real x1 = kbi/temps[0];
  real x2 = kbi/temps[1];
  real x3 = kbi/temps[2];
  real y1 = expected_U[0];
  real y2 = expected_U[1];
  real y3 = expected_U[2];
  real h12 = x2 - x1;
  real h23 = x3 - x2;
  real h13 = x3 - x1;
  real I01 = (h12/6.0)*((3.0 - h12/h13)*y1 + (3.0 + (h12*h12)/(h23*h13) + h12/h13)*y2 - (h12*h12)/(h23*h13)*y3);
  real I12 = (h23/6.0)*((3.0 - h23/h13)*y3 + (3.0 + (h23*h23)/(h12*h13) + h23/h13)*y2 - (h23*h23)/(h12*h13)*y1);
  g[1] = I01;
  g[2] = I01 + I12;
  norm += pBeta[1] + pBeta[2];
  // ---- Remaining intervals ----
  for (int i = 2; i < N_temps - 1; i++) {
    real x1 = kbi/temps[i-1];
    real x2 = kbi/temps[i];
    real x3 = kbi/temps[i+1];
    real y1 = expected_U[i-1];
    real y2 = expected_U[i];
    real y3 = expected_U[i+1];
    real h12 = x2 - x1;
    real h23 = x3 - x2;
    real h13 = x3 - x1;
    real I = (h23/6.0)*((3.0 - h23/h13)*y3 + (3.0 + (h23*h23)/(h12*h13) + h23/h13)*y2 - (h23*h23)/(h12*h13)*y1);
    g[i+1] = g[i] + I;
    norm += pBeta[i+1];
  }

  for(int i = 0; i < N_temps; i++){
    g[i] += scale*(-log(pBeta[i]/norm + eps) - uniform_FE); // lower the weight of higher sampled temps
  }
};

void update_its(System* system){
  cudaStream_t stream=0;
  State *s = system->state;
  Its* it = system->enhanced->its;
  int shMem = BLMS*sizeof(real)/32;

  if (system->run) {
    stream = system->run->enhancedStream;
  }

  if(system->run->step % it->sample_freq == 0){
    // Compute bias and forces due to ITS bias with weights
    real kT = kB*system->run->T;
    real_e* U_ss = NULL;
    real_e* U_su = NULL;
    real_e* U_uu = NULL;
    if(it->potential == "total"){
      // Reduce energy_d to eepotential already complete
      U_ss = it->U_d;
    }
    if(it->potential == "torsion"){
      U_ss = s->U_torsion_d;
    }
    if(it->potential == "rest"){
      // U_ss = s->U_ss_d;
      // dU_ss = s->dU_ss_d;
      // U_su = s->U_su_d;
      // dU_su = s->dU_su_d;
      // U_uu = s->U_uu_d;
      // dU_uu = s->dU_uu_d;
    }
  
    update_its_expectation_kernel<<<(it->N_temp+BLMS-1)/BLMS,BLMS,0,stream>>>(
      it->N_temp, it->temperatures_d, 
      U_ss, U_su, 
      it->alpha_d, it->its_bias_d,
      it->expected_U_d, it->weighted_U_d,
      it->weights_d, it->offsets_d);
  
    update_weights_kernel<<<1,1,0,stream>>>(
      it->N_temp, it->temperatures_d, 
      it->expected_U_d, 
      it->correction_strength_d, it->pHist_d, 
      it->g_k_d);
  }

  // This needs to go after the update (since its_bias was not filled)
  if(system->run->step % it->steps_per_temp == 0 && it->N_temp < it->N_temp_max){
    it->N_temp++;
  }
}

void Its::recv_its(){
  int size = N_temp_max*sizeof(real);
  cudaMemcpy(&U, U_d, sizeof(real_e), cudaMemcpyDefault);
  cudaMemcpy(&U_prime, U_prime_d, sizeof(real_e), cudaMemcpyDefault);
  cudaMemcpy(&weighted_beta, weighted_beta_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(pHist, pHist_d, size, cudaMemcpyDefault);
  cudaMemcpy(expected_U, expected_U_d, size, cudaMemcpyDefault);
  cudaMemcpy(g_k, g_k_d, size, cudaMemcpyDefault);
  cudaMemcpy(its_bias, its_bias_d, size, cudaMemcpyDefault);
}

void log_its(System* system){
  State *s = system->state;
  Its* it = system->enhanced->its;

  if(it->potential == "torsion"){
    cudaMemcpy(&it->U, s->U_torsion_d, sizeof(real_e), cudaMemcpyDefault);
  }

  real beta0= 1.0/(kB*system->run->T);
  real eff_beta = it->weighted_beta*(1.0/(kB*system->run->T));
  real eff_temp = 1.0/(kB*eff_beta);
  printf("Step: %d, U: %.2f, U': %.2f, Force Scale: %.2f, <beta>: %.2f, <T>: %.2f\n", 
    system->run->step, it->U, it->U_prime, eff_beta/beta0, eff_beta, eff_temp);
  printf("Accessible Temperatures: %d / %d\n", it->N_temp, it->N_temp_max);
  real sum = 0;
  for(int i = 0; i < it->N_temp; i++){
    sum+= it->pHist[i];
  }
  printf("P(beta): [ ");
  for(int i = 0; i < it->N_temp; i++){
    printf("%.3f, ", it->pHist[i]/sum);
  }
  printf("]\n");
  printf("Flattening correction: [ ");
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->correction_strength*(-log(it->pHist[i]/sum + 1e-10) + log(1.0/it->N_temp)));
  }
  printf("]\n");
  printf("its_bias: [");
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->its_bias[i]);
  }
  printf("]\n");

  printf("<U>_k: [ ");
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->expected_U[i]);
  }
  printf("]\n");

  printf("g_k: [ ");
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->g_k[i]);
  }
  printf("]\n");
  printf("\n");
}

void write_small_its(System* system, std::string output_dir){
  Its* it = system->enhanced->its;
  int size = it->N_temp_max*sizeof(real);
  it->recv_its();
  FILE* f;

  // Temps
  if(!it->fp_beta){ // only do this once
    std::string fnm_temps      = output_dir + "/betas.dat";
    it->fp_beta = fopen(fnm_temps.c_str(), "w");
    if(!it->fp_beta){
      printf("Error opening %s. Please make directory!\n", fnm_temps.c_str());
      exit(1);
      return;
    }
    for(int i = 0; i < it->N_temp_max; i++){
      real beta_k = 1.0/(kB*it->temperatures[i]);
      fprintf(it->fp_beta, "%f ", beta_k);
    }
    fprintf(it->fp_beta, "\n");
  }
  // g_k
  if(!it->fp_g){
    std::string fnm_g_k        = output_dir + "/g.dat";
    it->fp_g = fopen(fnm_g_k.c_str(), "w");
    if(!it->fp_g){
      printf("Error opening %s. Please make directory!\n", fnm_g_k.c_str());
      exit(1);
      return;
    }
  }
  fprintf(it->fp_g, "%d ", system->run->step);
  for(int i = 0; i < it->N_temp_max; i++){
    fprintf(it->fp_g, "%f ", it->g_k[i]);
  }
  fprintf(it->fp_g, "\n");
  // <U>
  if(!it->fp_exp_U){
    std::string fnm_expected_U = output_dir + "/expected_U.dat";
    it->fp_exp_U = fopen(fnm_expected_U.c_str(), "w");
    if(!it->fp_exp_U){
      printf("Error opening %s. Please make directory!\n", fnm_expected_U.c_str());
      exit(1);
      return;
    }
  }
  fprintf(it->fp_exp_U, "%d ", system->run->step);
  for(int i = 0; i < it->N_temp_max; i++){
    fprintf(it->fp_exp_U, "%f ", it->expected_U[i]);
  }
  fprintf(it->fp_exp_U, "\n");
  // Bk*U_bias
  if(!it->fp_red_bias){
     std::string fnm_its_bias   = output_dir + "/reduced_bias.dat";
     it->fp_red_bias = fopen(fnm_its_bias.c_str(), "w");
     if(!it->fp_red_bias){
       printf("Error opening %s. Please make directory!\n", fnm_its_bias.c_str());
       exit(1);
       return;
     }
  }
  fprintf(it->fp_red_bias, "%d ", system->run->step);
  for(int i = 0; i < it->N_temp_max; i++){
    real beta_k = 1.0/(kB*it->temperatures[i]);
    fprintf(it->fp_red_bias, "%f ", it->its_bias[i]);
  }
  fprintf(it->fp_red_bias, "\n");
}


void write_big_its(System* system, std::string output_dir){};