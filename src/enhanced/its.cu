#include "enhanced/its.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/state.h"
#include "system/system.h"
#include "enhanced/enhanced.h"
#include "system/potential.h"
#include "main/gpu_check.h"
#include "main/real3.h"

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
    if(pBeta_d) cudaFree(pBeta_d);
    if(pBeta) cudaFree(pBeta);
    if(wl_hist_d) cudaFree(wl_hist_d);
    if(wl_hist) cudaFree(wl_hist);
    if(U_prime_d) cudaFree(U_prime_d);
    if(g) free(g);
    if(g_d) cudaFree(g_d);
    if(scale_ss_d) cudaFree(scale_ss_d);
    if(scale_su_d) cudaFree(scale_su_d);
    if(U_d) cudaFree(U_d);
    if(fp_beta) fclose(fp_beta);
    if(fp_exp_U) fclose(fp_exp_U);
    if(fp_g) fclose(fp_g);
    if(fp_red_bias) fclose(fp_red_bias);
}

void Its::initialize(System* system){
    if(!temperatures){
      printf("Didn't set temperature range! Use \"enhanced its_temps {N_temp} {T_low} {T_high}\"\n");
      exit(1);
    } 
    
    // system->state pointers need to exist
    // need to check if U_su_d or dU_su_d exist
    if(potential == "total"){
      // Reduce energy_d to eepotential
      U_ss_d = U_d;
      dU_ss_d = system->state->forceBuffer_d;
    } else if (potential == "torsion"){
      U_ss_d = system->state->U_ss_d;
      dU_ss_d = system->state->dU_ss_buffer_d;
    } else if(potential == "rest"){
      U_ss_d = system->state->U_ss_d; // (includes torsions)
      dU_ss_d = system->state->dU_ss_buffer_d;
      U_su_d = system->state->U_su_d;
      dU_su_d = system->state->dU_su_buffer_d;
      U_uu_d = system->state->U_uu_d;
      dU_uu_d = system->state->dU_uu_buffer_d;
    }

    if(system->enhanced->init){ return; } // don't overallocate or reset mem/temps

    // Just assume that if this isn't set all of them are or aren't
    if(!expected_U){
      // N_temp initialized in enhanced.cu
      cudaMalloc(&temperatures_d, N_temp*sizeof(real));
      cudaMemcpy(temperatures_d, temperatures, N_temp*sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&g_d, N_temp*sizeof(real));
      cudaMemset(g_d, 0, N_temp*sizeof(real));
      g = (real*)calloc(N_temp,sizeof(real));
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
      cudaMalloc(&scale_ss_d, sizeof(real));
      cudaMemcpy(scale_ss_d, &scale_ss, sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&scale_su_d, sizeof(real));
      cudaMemcpy(scale_su_d, &scale_su, sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&pBeta_d, N_temp*sizeof(real));
      cudaMemset(pBeta_d, 0, N_temp*sizeof(real));
      pBeta = (real*) calloc(N_temp, sizeof(real));
      for(int i = 0; i < N_temp; i++){ pBeta[i] = 1.0; }
      cudaMemcpy(pBeta_d, pBeta, N_temp*sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&wl_hist_d, N_temp*sizeof(real));
      cudaMemset(wl_hist_d, 0, N_temp*sizeof(real));
      wl_hist = (real*) calloc(N_temp, sizeof(real));
      cudaMalloc(&wl_f_d, N_temp*sizeof(real));
      cudaMemset(wl_f_d, 0, N_temp*sizeof(real));
      wl_f = (real*) calloc(N_temp, sizeof(real));
      cudaMalloc(&wl_inc_d, sizeof(real));
      cudaMemcpy(wl_inc_d, &wl_inc, sizeof(real), cudaMemcpyDefault);
      cudaMalloc(&U_prime_d, sizeof(real_e));
      cudaMalloc(&U_d, sizeof(real_e));
    }

    N_temp_max = N_temp;
    N_temp = 1; // slowly add additional temperatures
    low_idx=0;

    if(update_steps < (N_temp_max-1)*steps_per_temp){
      update_steps = N_temp_max*steps_per_temp;
      printf("Set ITS update_steps < temp_count*steps_per_temp. Increasing update_steps to %d!", update_steps);
    }
}

__global__ void ee_sample_betas_kernel(
  int low_idx, int N_temps, real_e sim_T, 
  real_e* U_tot, real_e* U_sele, real_e* U_int, 
  real* temps, real* g, real alpha,
  real_e rand_number,
  // Output
  real* scale_ss, real* scale_su, 
  real* its_bias, real* pHist,
  real_e* enhanced_energy, real_e* U_prime){

  real_e U_ss=U_sele[0];
  real_e U_su=0;
  real_e U_uu=0;
  if(U_int){
    U_su = U_int[0];
  }
  // Compute the max exponent
  real_e beta_0 = 1.0/(kB*sim_T);
  real_e c = -1e9;
  for(int i = low_idx; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k-beta_0)*U_ss - (beta_k*pow(beta_k/beta_0, (double)alpha-1.0) - beta_0)*U_su + g[i];
    if(exp_arg > c){
      c = exp_arg;
    }
  }
  real_e exp_sum = 0;
  for(int i = low_idx; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k-beta_0)*U_ss - (beta_k*pow(beta_k/beta_0, (double)alpha-1.0) - beta_0)*U_su + g[i];
    exp_sum += exp(exp_arg-c);
  }
  // r < sum k ( p(beta_k) )
  real_e p_sum = 0;
  real_e new_beta = 0;
  bool found = false;
  for(int i = low_idx; i < N_temps; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -(beta_k-beta_0)*U_ss - (beta_k*pow(beta_k/beta_0, (double)alpha-1.0) - beta_0)*U_su + g[i];
    p_sum += exp(exp_arg-c) / exp_sum;
    if (p_sum > rand_number && !found){
      new_beta=beta_k;
      pHist[i] = 1;
      found = true;
      //printf("New Temp %d: %f, p_range: [%f < rnd: %f < %f] \n", i, temps[i], p_sum-exp(exp_arg-c)/exp_sum, rand_number, p_sum);
    } else {
      pHist[i] = 0.0;
    }
  }
  scale_ss[0] = new_beta/beta_0;
  scale_su[0] = pow(new_beta/beta_0, alpha);
}

__global__ void expanded_bias_kernel(
  real_e* U_tot, real_e* U_sele, real_e* U_int, 
  real* scale_ss, real* scale_su, 
  real_e* enhanced_energy, real_e* U_prime
){
  real_e U_ss=U_sele[0];
  real_e U_su=0;
  if(U_int){
    U_su = U_int[0];
  }
  real_e U_extra = (scale_ss[0] - 1.0)*U_ss + (scale_su[0]-1.0)*U_su;
  U_prime[0] = U_tot[0]+U_extra;
  atomicAdd(enhanced_energy, U_extra);
}

__global__ void update_forces_kernel(
  int DOF,  int nL, real alpha,
  real* scale_ss, real* scale_su,
  real_f* dU_sel, real_f* dU_int, real_f* dU_solv,
  real_f* forceBuffer 
){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if(i < DOF){
    // Check NaN
    real_f dU_ss = dU_sel[i];
    real_f dU_su = 0;
    real_f dU_uu = 0;
    if(dU_int){
      dU_su = dU_int[i];
    }
    // Already contains dU_uu
    // Remove exising and replace with scaled versions of dU_ss & dU_su
    //if(i < nL){ //&& abs(dU_ss - forceBuffer[i]) > 1e-3){
    //  printf("diff: %f\n", dU_ss - forceBuffer[i]);
    //}
    forceBuffer[i] += (scale_ss[0]-1)*dU_ss + (scale_su[0]-1)*dU_su;
  }
}

__global__ void reduce_total_energy_kernel(real_e* energy, real_e* U){
  *U=0;
  for(int i = 0; i < eepotential; i++){
    *U += energy[i];
  }
}

void getforce_its(System* system, int step, bool calcEnergy){
  cudaStream_t stream=0;
  State *s = system->state;
  Its* it = system->enhanced->its;
  int shMem = BLMS*sizeof(real)/32;

  if (system->run) {
    stream = system->run->enhancedStream;
  }

  if (it->expanded_ensemble){ // always true for now
    // Calculate total potential prior to ITS
    reduce_total_energy_kernel<<<1,1,0,stream>>>(s->energy_d, it->U_d);
    if((step != 0 && step % it->sample_freq == 0) || system->run->step == system->run->step0){
      real_x rand_num = ((double)rand())/RAND_MAX;
      ee_sample_betas_kernel<<<1,1,shMem,stream>>>(
        it->low_idx, it->N_temp, system->run->T, it->U_d, it->U_ss_d, it->U_su_d, 
        it->temperatures_d, it->g_d, it->alpha,
        rand_num,
        it->scale_ss_d, it->scale_su_d,
        it->its_bias_d, it->pBeta_d,
        &(s->energy_d[eeenhanced]), it->U_prime_d);
    }
    if(calcEnergy){
      // update bias due to expanded ensemble
      expanded_bias_kernel<<<1,1,0,stream>>>(
        it->U_d, it->U_ss_d, it->U_su_d,
        it->scale_ss_d, it->scale_su_d,
        &(s->energy_d[eeenhanced]), it->U_prime_d);
    }
  } 

  // dU'/dX = dU/dX + (<B>/B0-1)*dU_ss/dX + (<sqrt(B/B0)>-1)*dU_su/dX
  int dof = 3*s->atomCount + s->lambdaCount;
  update_forces_kernel<<<(dof+BLMS-1)/BLMS,BLMS, 0,stream>>>(
    dof, s->lambdaCount, it->alpha,
    it->scale_ss_d, it->scale_su_d,
    it->dU_ss_d, it->dU_su_d, it->dU_uu_d,
    s->forceBuffer_d);
}

__global__ void update_abf_wl_kernel(
  int low_idx, int N_temps, real sys_kT,
  real* temps, real alpha,
  real_e* U_sele, real_e* U_int, 
  real* pHist, real* wl_hist, real* wl_f, real* wl_inc,
  real* expected_U, real* weighted_U,
  real* weights
){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i >= low_idx && i < N_temps){
    // Update ABF <U>
    real_e U_ss=U_sele[0];
    real_e U_su=0;
    if(U_int){
      U_su = U_int[0];
    }
    real beta_0 = 1.0/sys_kT;
    real beta_k = 1.0/(kB*temps[i]);
    real U = U_ss + alpha*pow((beta_k/beta_0), alpha-1)*U_su;
    weighted_U[i] += U * pHist[i];
    weights[i] += pHist[i];
    expected_U[i] = weighted_U[i] / weights[i];
    // Update Wang-Landau
    wl_hist[i] += pHist[i];
    wl_f[i] -= (*wl_inc)*pHist[i];
  }
}

__global__ void wl_inc_kernel(int N_temps, real* wl_hist, real* wl_inc, real wl_ratio, real wl_alpha){
  int total = 0;
  for(int i = 0; i < N_temps; i++){
    total += wl_hist[i];
  }
  real average = total/N_temps;
  bool reset = true;
  for (int i = 0; i < N_temps; i++){
    real ratio = wl_hist[i]/average;
    reset = reset && ratio > wl_ratio;
  }
  if (reset){
    *wl_inc *= wl_alpha;
    for(int i = 0; i < N_temps; i++){
      wl_hist[i] = 0;
    }
  }
}

__global__ void update_weights_kernel(int N_temps, real* temps, real* expected_U, real* wl_f, real* g){
  real kbi = 1/kB;
  g[0] = 0;

  if (N_temps < 2){
    return;
  } else if (N_temps == 2) {
    g[0] += wl_f[0];
    g[1] = (kbi/temps[1] - kbi/temps[0]) * (expected_U[0] + expected_U[1]) / 2.0;
    g[1] += wl_f[1];
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
  }
  for(int i = 0; i < N_temps; i++){
    g[i] += wl_f[i];
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

  // Only update until update_steps
  if (system->run->step - system->run->step0 > it->update_steps){
    return;
  }

  if(system->run->step % it->sample_freq == 0){
    // Compute bias and forces due to ITS bias with weights
    real kT = kB*system->run->T;
    update_abf_wl_kernel<<<(it->N_temp+BLMS-1)/BLMS,BLMS,0,stream>>>(
      it->low_idx, it->N_temp, kT,
      it->temperatures_d, it->alpha,
      it->U_ss_d, it->U_su_d, 
      it->pBeta_d, 
      it->wl_hist_d, it->wl_f_d, it->wl_inc_d,
      it->expected_U_d, it->weighted_U_d,
      it->weights_d);

    // if we are out of fixed temp sampling
    if(it->low_idx == 0 && it->N_temp == it->N_temp_max){
      wl_inc_kernel<<<1, 1, 0, stream>>>(
        it->N_temp, it->wl_hist_d, it->wl_inc_d, it->wl_ratio, it->wl_alpha);
    }
  
    update_weights_kernel<<<1,1,0,stream>>>(
      it->N_temp, it->temperatures_d, 
      it->expected_U_d, 
      it->wl_f_d,
      it->g_d);
  }

  // This needs to go after the update (since its_bias/pHist was not filled)
  if(system->run->step % it->steps_per_temp == 0 && it->N_temp <= it->N_temp_max){
    if(it->N_temp == it->N_temp_max){
      it->low_idx = 0;
    } else {
      it->low_idx++;
      it->N_temp++;
    }
  }
}

void Its::recv_its(){
  int size = N_temp_max*sizeof(real);
  cudaMemcpy(&U, U_d, sizeof(real_e), cudaMemcpyDefault);
  cudaMemcpy(&U_prime, U_prime_d, sizeof(real_e), cudaMemcpyDefault);
  cudaMemcpy(&U_ss, U_ss_d, sizeof(real_e), cudaMemcpyDefault);
  if(U_su_d) cudaMemcpy(&U_su, U_su_d, sizeof(real_e), cudaMemcpyDefault);

  cudaMemcpy(&scale_ss, scale_ss_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&scale_su, scale_su_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(pBeta, pBeta_d, size, cudaMemcpyDefault);
  cudaMemcpy(expected_U, expected_U_d, size, cudaMemcpyDefault);
  cudaMemcpy(g, g_d, size, cudaMemcpyDefault);
  cudaMemcpy(its_bias, its_bias_d, size, cudaMemcpyDefault);
  cudaMemcpy(wl_hist, wl_hist_d, size, cudaMemcpyDefault);
  cudaMemcpy(wl_f, wl_f_d, size, cudaMemcpyDefault);
  cudaMemcpy(&wl_inc, wl_inc_d, sizeof(real), cudaMemcpyDefault);
}

void log_its(System* system){
  State *s = system->state;
  Its* it = system->enhanced->its;
  it->recv_its();

  real beta0= 1.0/(kB*system->run->T);
  real eff_beta = it->scale_ss*(1.0/(kB*system->run->T));
  real eff_temp = 1.0/(kB*eff_beta);
  real root_beta = it->scale_su*sqrt(1.0/(kB*system->run->T));
  printf("Step: %d, U: %.2f, U_ss: %.2f, U_su: %.2f, U': %.2f, U'-U: %.2f\n", 
    system->run->step, it->U, it->U_ss, it->U_su, it->U_prime, it->U_prime - it->U);
  printf("Force Scale ss: %.2f, Force Scale su: %.2f, <beta>: %.2f, <T>: %.2f\n", 
    it->scale_ss, it->scale_su, eff_beta, eff_temp);
  printf("Accessible Temps: %d / %d [ ", it->N_temp, it->N_temp_max);
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->temperatures[i]);
  }
  printf("]\n");
  printf("Wang-Landau Delta: %.3e\n", it->wl_inc);
  real ave = 0;
  for(int i = 0; i < it->N_temp; i++){
    ave+=it->wl_hist[i];
  }
  ave /= it->N_temp;
  printf("Wang-Landau Ratios (eta > %.2f): [ ", it->wl_ratio);
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->wl_hist[i]/ave);
  }
  printf("]\n");
  printf("Wang-Landau FE - g0: [ ");
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->wl_f[i]-it->wl_f[0]);
  }
  printf("]\n");
  printf("<U>_k: [ ");
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->expected_U[i]);
  }
  printf("]\n");
  printf("g_k: [ ");
  for(int i = 0; i < it->N_temp; i++){
    printf("%.2f, ", it->g[i]);
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
    }
    for(int i = 0; i < it->N_temp_max; i++){
      real beta_k = 1.0/(kB*it->temperatures[i]);
      fprintf(it->fp_beta, "%f ", beta_k);
    }
    fprintf(it->fp_beta, "\n");
    fflush(it->fp_beta);
  }
  // g_k
  if(!it->fp_g){
    std::string fnm_g_k        = output_dir + "/g.dat";
    it->fp_g = fopen(fnm_g_k.c_str(), "w");
    if(!it->fp_g){
      printf("Error opening %s. Please make directory!\n", fnm_g_k.c_str());
      exit(1);
    }
  }
  fprintf(it->fp_g, "%d ", system->run->step);
  for(int i = 0; i < it->N_temp_max; i++){
    fprintf(it->fp_g, "%f ", it->g[i]);
  }
  fprintf(it->fp_g, "\n");
  fflush(it->fp_g);
  // <U>
  if(!it->fp_exp_U){
    std::string fnm_expected_U = output_dir + "/expected_U.dat";
    it->fp_exp_U = fopen(fnm_expected_U.c_str(), "w");
    if(!it->fp_exp_U){
      printf("Error opening %s. Please make directory!\n", fnm_expected_U.c_str());
      exit(1);
    }
  }
  fprintf(it->fp_exp_U, "%d ", system->run->step);
  for(int i = 0; i < it->N_temp_max; i++){
    fprintf(it->fp_exp_U, "%f ", it->expected_U[i]);
  }
  fprintf(it->fp_exp_U, "\n");
  fflush(it->fp_exp_U);
  // Bk*U_bias
  if(!it->fp_red_bias){
     std::string fnm_its_bias   = output_dir + "/reduced_bias.dat";
     it->fp_red_bias = fopen(fnm_its_bias.c_str(), "w");
     if(!it->fp_red_bias){
       printf("Error opening %s. Please make directory!\n", fnm_its_bias.c_str());
       exit(1);
     }
  }
  fprintf(it->fp_red_bias, "%d ", system->run->step);
  for(int i = 0; i < it->N_temp_max; i++){
    real beta_k = 1.0/(kB*it->temperatures[i]);
    fprintf(it->fp_red_bias, "%f ", beta_k*it->its_bias[i]);
  }
  fprintf(it->fp_red_bias, "\n");
  fflush(it->fp_red_bias);
  // <T>
  if(!it->fp_weighted_T){
     std::string fnm_T   = output_dir + "/T.dat";
     it->fp_weighted_T = fopen(fnm_T.c_str(), "w");
     if(!it->fp_weighted_T){
       printf("Error opening %s. Please make directory!\n", fnm_T.c_str());
       exit(1);
     }
  }
  fprintf(it->fp_weighted_T, "%d ", system->run->step);
  real eff_beta = it->scale_ss*(1.0/(kB*system->run->T));
  real eff_temp = 1.0/(kB*eff_beta);
  fprintf(it->fp_weighted_T, "%f ", eff_temp);
  fprintf(it->fp_weighted_T, "\n");
  fflush(it->fp_weighted_T);
  // Potentials
  if(!it->fp_potentials){
     std::string fnm_pot   = output_dir + "/potentials.dat";
     it->fp_potentials = fopen(fnm_pot.c_str(), "w");
     if(!it->fp_potentials){
       printf("Error opening %s. Please make directory!\n", fnm_pot.c_str());
       exit(1);
     }
  }
  fprintf(it->fp_potentials, "%d ", system->run->step);
  fprintf(it->fp_potentials, "%f %f %f %f\n", it->U, it->U_ss, it->U_su, it->U_prime);
  fflush(it->fp_potentials);
}

void write_big_its(System* system, std::string output_dir){};
