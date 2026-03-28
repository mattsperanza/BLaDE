#include "enhanced/simulated_tempering.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/state.h"
#include "system/system.h"
#include "enhanced/enhanced.h"
#include "system/potential.h"
#include "main/gpu_check.h"
#include "main/real3.h"
#include <unistd.h> // for ftruncate

SimulatedTempering::SimulatedTempering(std::string potential){
    this->potential = potential;
    // temperatures' memory alloc & N_temp set in enhanced.cu
}

SimulatedTempering::~SimulatedTempering(){
  // Heap Allocated Memory
  if(U_d) cudaFree(U_d);
  if(U_prime_d) cudaFree(U_prime_d);
  if(temperatures) free(temperatures);
  if(temperatures_d) cudaFree(temperatures_d);
  if(betas) free(betas);
  if(betas_d) cudaFree(betas_d);
  if(g) free(g);
  if(g_d) cudaFree(g_d);
  if(scale_ss_d) cudaFree(scale_ss_d);
  if(temp_curr_idx_d) cudaFree(temp_curr_idx_d);
  if(mbar_data) free(mbar_data);
  if(g_data) free(g_data);

  // Files
  if(fp_betas) fclose(fp_betas);
  if(fp_temps) fclose(fp_temps);
  if(fp_current_T) fclose(fp_current_T);
  if(fp_g) fclose(fp_g);
  if(fp_mbar) fclose(fp_mbar);
}

void SimulatedTempering::initialize(System* system, std::string output_dir){
  if(!temperatures){
    printf("Didn't set temperature range! Use \"enhanced st_temps_exp {N_temp} {T_low} {T_high}\"\n");
    exit(1);
  } 
  
  // system->state pointers need to exist
  // these are updated each dynamics call
  if(potential == "total"){
    // Reduce energy_d to eepotential
    U_ss_d = U_d;
    dU_ss_d = system->state->forceBuffer_d;
  } else if(potential == "solute"){
    U_ss_d = system->state->U_ss_d; // (includes torsions)
    dU_ss_d = system->state->dU_ss_buffer_d;
    U_su_d = system->state->U_su_d;
    dU_su_d = system->state->dU_su_buffer_d;
    U_uu_d = system->state->U_uu_d;
    dU_uu_d = system->state->dU_uu_buffer_d;
  }

  std::string mbar_rst = output_dir + "/" + fnm_mbar;
  if(mbar_rst == fnm_mbar_iter){
    printf("MBAR restart file name: %s is reserved. Please rename st_mbar_file.\n", mbar_rst.c_str());
    exit(1);
  }

  if(system->enhanced->init){ return; } // don't overallocate or reset mem
  // Everything below should not be updated each dynamics call

  cudaMalloc(&U_d, sizeof(real_e));
  cudaMalloc(&U_prime_d, sizeof(real_e));
  cudaMalloc(&temperatures_d, N_temp*sizeof(real));
  cudaMemcpy(temperatures_d, temperatures, N_temp*sizeof(real), cudaMemcpyDefault);
  betas = (real*) malloc(N_temp*sizeof(real));
  for(int i = 0; i < N_temp; i++){
    betas[i] = 1.0/(kB*temperatures[i]);
  }
  cudaMalloc(&betas_d, N_temp*sizeof(real));
  cudaMemcpy(betas_d, betas, N_temp*sizeof(real), cudaMemcpyDefault);
  cudaMalloc(&g_d, N_temp*sizeof(real));
  cudaMemset(g_d, 0, N_temp*sizeof(real));
  g = (real*)calloc(N_temp,sizeof(real));
  cudaMalloc(&scale_ss_d, sizeof(real));
  cudaMemcpy(scale_ss_d, &scale_ss, sizeof(real), cudaMemcpyDefault);
  cudaMalloc(&temp_curr_idx_d, sizeof(int));

  // Init timing and MBAR iteration stuff
  steps_per_temp = iteration_length/N_temp; // only for iter=0
  samples_per_temp = steps_per_temp/sample_freq;
  equil_per_temp = equil_length/N_temp; // only for iter=0
  equil_samples_per_temp = equil_per_temp/sample_freq;
  printf("For iteration 0: steps_per_temp: %d -> %d samples, equil_per_temp: %d -> %d samples\n", steps_per_temp, samples_per_temp, equil_per_temp, equil_samples_per_temp); 
  iteration_samples = iteration_length/sample_freq;
  equilibration_samples = equil_length/sample_freq;
  if(equil_length>iteration_length){
    printf("Error: equil_length: %d > iter_length: %d\n", equil_length, iteration_length);
    exit(1);
  }
  mbar_data_length = iter_history*iteration_samples*sample_data_length;
  mbar_data = (real*) calloc(mbar_data_length, sizeof(real));
  g_data = (real*) calloc(iter_history*N_temp, sizeof(real));
  printf("For iteration n: iteration_steps: %d -> %d samples, equilibration_steps: %d -> %d samples, mbar_samples: %d, mbar_data_length: %d\n", 
    iteration_length, iteration_samples, equil_length, equilibration_samples, iter_history*iteration_samples, mbar_data_length);

  // Check for and read from restarts, only if do_restart is true
  current_iter = read_restart(system, output_dir);
  printf("Starting at iter: %d\n", current_iter);
  cudaMemcpy(g_d, g, N_temp*sizeof(real), cudaMemcpyDefault);

  // Slowly add additional temperatures if no restarts exist
  if (current_iter == 0){
    printf("Doing fixed temp sampling for first iteration!\n");
    low_idx=0;
    high_idx=1; 
  } else {
    printf("Skipping fixed temp sampling!\n");
    low_idx=0;
    high_idx=N_temp;
  }
}

__global__ void ee_sample_betas_kernel(
  int low_idx, int high_idx, real_e sim_T, 
  real_e* U_tot, real_e* U_sele, real_e* U_int, 
  real* temps, real* g, 
  real_e rand_number, 
  // Output
  int* temp_index, 
  real* scale_ss, real_e* enhanced_energy, real_e* U_prime){

  real_e U_ss=U_sele[0];
  real_e U_su=0;
  if(U_int){
    U_su = U_int[0];
  }
  /*
    Make 1 MCMC move to new bias/temperature (REST2), sample from stationary distribution of betas (independence gibbs)
    U = U_ss + U_su + U_uu 
    U_eff = U_uu + bk/b0*U_ss + sqrt(bk/b0)*U_su
    prob(bk|X) = exp(-b0*(U_eff + gk/b0)) / sum_j [ exp(-b0*(U_eff + gj/b0)) ]
    prob(bk|X) = exp(-b0*U_bk + gk) / sum_j [ exp(-b0*U_bj + gj) ]  // using exp(-b0*U_uu + b0*U_uu) = 1
    -b0*U_bk = -bk*U_ss - sqrt(bk*b0)*U_su
    g = -ln(int [ exp(-b0*U_eff) ]) = -ln(Z(bk)) // see later explanation in update_expectation_kernel

    In practice I do: 
      U_eff = U + Ub 
    where 
      Ub = (bk/b0-1)*U_ss + (sqrt(bk/b0)-1)*U_su 
    to cancel existing forces/potential, but this doesn't change this above criteria
  */

  // Factor out a constant c which will cancel
  real_e beta_0 = 1.0/(kB*sim_T);
  real_e c = -1e9;
  for(int i = low_idx; i < high_idx; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -beta_k*U_ss - sqrt(beta_k*beta_0)*U_su + g[i];
    if(exp_arg > c){
      c = exp_arg;
    }
  }
  // Compute weights (largest weight is 1)
  real_e exp_sum = 0;
  for(int i = low_idx; i < high_idx; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -beta_k*U_ss - sqrt(beta_k*beta_0)*U_su + g[i];
    exp_sum += exp(exp_arg-c);
  }
  // r < sum k ( p(beta_k) )
  real_e p_sum = 0;
  real_e new_beta = 0;
  for(int i = low_idx; i < high_idx; i++){
    real_e beta_k = 1.0/(kB*temps[i]);
    real_e exp_arg = -beta_k*U_ss - sqrt(beta_k*beta_0)*U_su + g[i];
    p_sum += exp(exp_arg-c) / exp_sum;
    if (p_sum > rand_number){
      new_beta=beta_k;
      *temp_index = i;
      break;
    }
  }
  scale_ss[0] = new_beta/beta_0;
}

__global__ void expanded_bias_kernel(
  real_e* U_tot, real_e* U_sele, real_e* U_int, 
  real* scale_ss, real_e* enhanced_energy, real_e* U_prime
){
  real_e U_ss=U_sele[0];
  real_e U_su=0;
  if(U_int){
    U_su = U_int[0];
  }
  real_e U_extra = (scale_ss[0] - 1.0)*U_ss + (sqrt(scale_ss[0]) - 1.0)*U_su;
  U_prime[0] = U_tot[0]+U_extra;
  atomicAdd(enhanced_energy, U_extra);
}

__global__ void update_forces_kernel(
  int DOF,  int nL, 
  real_f* dU_sel, real_f* dU_int, real_f* dU_solv,
  real* scale_ss,
  real_f* forceBuffer 
){
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if(i < DOF){
    // Check NaN
    real_f dU_ss = dU_sel[i];
    real_f dU_su = 0;
    if(dU_int){
      dU_su = dU_int[i];
    }
    //if(i < nL){
      //printf("i: %d, force_buff: %f, dU_ss: %f\n", i, forceBuffer[i], dU_ss);
    //}
    // Remove exising forces and replace with scaled versions of dU_ss & dU_su
    forceBuffer[i] += (scale_ss[0]-1)*dU_ss + (sqrt(scale_ss[0])-1)*dU_su;
  }
}

__global__ void reduce_total_energy_kernel(real_e* energy, real_e* U){
  *U=0;
  for(int i = 0; i < eepotential; i++){
    *U += energy[i];
  }
}

void getforce_st(System* system, int step, bool calcEnergy){
  cudaStream_t stream=0;
  State *s = system->state;
  SimulatedTempering* st = system->enhanced->simulatedTempering;
  int shMem = BLMS*sizeof(real)/32;

  if (system->run) {
    stream = system->run->enhancedStream;
  }

  reduce_total_energy_kernel<<<1,1,0,stream>>>(s->energy_d, st->U_d);

  // Don't sample betas on pressure coupling steps unless it is the first step
  if((step != 0 && step % st->temp_sample_freq == 0) || system->run->step == system->run->step0){
    real_x rand_num = ((double)rand())/RAND_MAX;
    ee_sample_betas_kernel<<<1,1,shMem,stream>>>(
      // Input
      st->low_idx, st->high_idx, system->run->T, 
      st->U_d, st->U_ss_d, st->U_su_d, 
      st->temperatures_d, st->g_d, 
      rand_num, 
      // Output
      st->temp_curr_idx_d, 
      st->scale_ss_d, 
      &(s->energy_d[eeenhanced]), st->U_prime_d);
  }

  if(calcEnergy){
    // update bias due to expanded ensemble
    expanded_bias_kernel<<<1,1,0,stream>>>(
      st->U_d, st->U_ss_d, st->U_su_d,
      st->scale_ss_d, &(s->energy_d[eeenhanced]), st->U_prime_d);
  }

  // dU'/dX = dU/dX + (<B>/B0-1)*dU_ss/dX + (<sqrt(B/B0)>-1)*dU_su/dX
  int dof = 3*s->atomCount + s->lambdaCount;
  update_forces_kernel<<<(dof+BLMS-1)/BLMS,BLMS, 0,stream>>>(
    // Input
    dof, s->lambdaCount, 
    st->dU_ss_d, st->dU_su_d, st->dU_uu_d,
    st->scale_ss_d, 
    // Output
    s->forceBuffer_d);
}

bool iterate_mbar(int N_temp, real* nk, real* uk, real* g){
  int N_samples = 0;
  for(int i = 0; i < N_temp; i++){
    N_samples += nk[i];
  }

  int max_iter = 500;
  real tol = 1e-3; // max_k |gi,k - gi-1,k|

  /* iterate eq. 11 from shirts and chodera
    fi = -ln[ sum_n( exp(-u_ni)/sum_l{ N_l*exp(fl-u_nl) } ) ]
    arg_ni = -u_ni - max_l - ln( sum_l{ N_l*exp(fl-u_nl-max_l) } )
    fi = -ln[ sum_n( exp(arg_ni) ) ]
    fi = -max_n - ln[ sum_n ( exp(arg_ni - max_n) ) ]
  */
  real* args = (real*) calloc(N_temp*N_samples, sizeof(real));
  bool converged = false;
  for(int iter = 0; iter < max_iter; iter++){
    // Compute arg_ni elements and respective maxes
    real max_n[N_temp];
    for(int k = 0; k < N_temp; k++){
      max_n[k] = -1e9;
    }
    for(int n = 0; n < N_samples; n++){
      real max_l = -1e9;
      for(int k = 0; k < N_temp; k++){
        args[n*N_temp + k] = -uk[n*N_temp + k];
        if(g[k]-uk[n*N_temp+k] > max_l){
          max_l = g[k]-uk[n*N_temp+k];
        }
      }
      real sum = 0;
      for(int k = 0; k < N_temp; k++){
        sum += nk[k]*exp(g[k] - uk[n*N_temp+k] - max_l);
      }
      real ln_sum = log(sum);
      for(int k = 0; k < N_temp; k++){
        args[n*N_temp+k] -= max_l + ln_sum;
        if(args[n*N_temp+k] > max_n[k]){
          max_n[k] = args[n*N_temp+k];
        }
      }
    }
    // update f
    real f[N_temp];
    for(int k = 0; k < N_temp; k++){
      f[k] = -max_n[k];
      real sum = 0;
      for(int n = 0; n < N_samples; n++){
        sum += exp(args[n*N_temp + k] - max_n[k]);
      }
      f[k] -= log(sum);
    }
    // Subtract f[0] from all elements & compute max delta
    real max_delta = 0;
    real tmp = f[0];
    //printf("MBAR %d: [", iter);
    for(int k = 0; k < N_temp; k++){
      f[k] -= tmp;
      real abs_delta = abs(f[k] - g[k]); 
      real omega = 1.0; // SOR
      g[k] = omega*f[k] + (1.0-omega)*g[k];
      if(abs_delta > max_delta){
        max_delta = abs_delta;
      }
      //printf("%f, ", g[k]);
    }
    //printf("]\n");
    // check delta condition
    if(max_delta < tol){
      converged=true;
      break;
    }
  }
  free(args);
  printf("N_k: [");
  for(int i = 0; i < N_temp; i++){
    printf(" %f, ", nk[i]);
  }
  printf("]\n");
  printf("g_k: [");
  for(int i = 0; i < N_temp; i++){
    printf(" %f, ", g[i]);
  }
  printf("]\n");

  return converged;
}

bool SimulatedTempering::solve_mbar(System* system){
  bool success = false;
  // Filter and copy data into iteration form
  real beta0 = 1.0/(kB*system->run->T);
  real* uk_tmp = NULL;
  real* nk_tmp = (real*)calloc(N_temp,sizeof(real));
  real* g_tmp = (real*)calloc(N_temp,sizeof(real));
  memcpy(g_tmp,g, N_temp*sizeof(real)); // initial guess of weights as current weights
  int sample_count = 0; // sample count
  // TODO: Come back to clean this up
  if(current_iter == 0){ // iter_history = 0
    int n_samples = N_temp * (samples_per_temp - equil_samples_per_temp);
    int uk_len = n_samples * N_temp;
    uk_tmp = (real*)malloc(uk_len * sizeof(real));
    for(int i = 0; i < N_temp; i++){
      int skip = i * samples_per_temp + equil_samples_per_temp;  // skip equil for this temp
      for(int j = 0; j < (samples_per_temp - equil_samples_per_temp); j++){
        int idx_mbar = (skip + j) * sample_data_length;
        real U_ss = mbar_data[idx_mbar + mbar_Uss];
        real U_su = mbar_data[idx_mbar + mbar_Usu];
        nk_tmp[(int)round(mbar_data[idx_mbar + mbar_k])]++;
        for(int k = 0; k < N_temp; k++){
          real reduced_pot = betas[k]*U_ss + sqrt(betas[k]*beta0)*U_su;
          uk_tmp[sample_count*N_temp + k] = reduced_pot;
          g_tmp[k] -= reduced_pot;
        }
        sample_count++;
      }
    }
    // Initialize weights to <-reduced_pot>
    real rel = g_tmp[0]/nk_tmp[0];
    for(int i = 0; i < N_temp; i++){
      g_tmp[i] /= nk_tmp[i];
      g_tmp[i] -= rel;
    }
    if (sample_count != n_samples){
      printf("Something wrong 0!!\n");
    }
  } else {
    int iter_of_data = min(current_iter+1, iter_history);
    int n_samples_per_iter = (iteration_samples - equilibration_samples);
    int uk_len = n_samples_per_iter*N_temp*iter_of_data;
    uk_tmp = (real*)malloc(uk_len * sizeof(real));
    for(int i = current_iter; i >= max(current_iter-(iter_history-1), 0); i--){
      int eff_i = i % iter_history;
      int skip = eff_i*iteration_samples + equilibration_samples;
      for(int j = 0; j < n_samples_per_iter; j++){
        int idx_mbar = (skip + j) * sample_data_length;
        real U_ss = mbar_data[idx_mbar + mbar_Uss];
        real U_su = mbar_data[idx_mbar + mbar_Usu];
        nk_tmp[(int)round(mbar_data[idx_mbar + mbar_k])]++;
        for(int k = 0; k < N_temp; k++){
          real reduced_pot = betas[k]*U_ss + sqrt(betas[k]*beta0)*U_su;
          uk_tmp[sample_count*N_temp + k] = reduced_pot;// - g_data[eff_i*N_temp + k];
        }
        sample_count++;
      }
    }
    if (sample_count != n_samples_per_iter*iter_of_data){
      printf("Something wrong 1!!\n");
    }
  } 

  iterate_mbar(N_temp, nk_tmp, uk_tmp, g_tmp);

  // Update CPU mem (for next iter)
  memcpy(&g_data[((current_iter+1)%iter_history)*N_temp], g_tmp, N_temp*sizeof(real)); 
  memcpy(g, g_tmp, N_temp*sizeof(real));
  free(uk_tmp);
  free(nk_tmp);
  free(g_tmp);
  return success;
}

void update_st(System* system){
  cudaStream_t stream=0;
  State *s = system->state;
  SimulatedTempering* st = system->enhanced->simulatedTempering;
  int shMem = BLMS*sizeof(real)/32;

  if (system->run) {
    stream = system->run->enhancedStream;
  }

  if(system->run->step % st->sample_freq == 0 
      && st->current_iter < st->total_iters){ // don't collect samples in equilibrium sampling stage (mbar_data not large enough)
    // load sample data into mbar_data
    int effective_iter = st->current_iter % st->iter_history;
    int idx = st->sample_data_length*(effective_iter*st->iteration_samples + st->collected_iter_samples);
    st->mbar_data[idx+mbar_k] = st->temp_curr_idx;
    st->mbar_data[idx+mbar_beta0] = 1.0/(kB*system->run->T);
    st->mbar_data[idx+mbar_betak] = st->betas[st->temp_curr_idx];
    st->mbar_data[idx+mbar_Uss] = st->U_ss;
    st->mbar_data[idx+mbar_Usu] = st->U_su;
    st->collected_iter_samples++;
    if (st->collected_iter_samples == st->iteration_samples){
      // run MBAR on data
      st->solve_mbar(system); // fills g and g_data with updated free energies
      // write update to weights
      cudaMemcpy(st->g_d, st->g, st->N_temp*sizeof(real), cudaMemcpyDefault);
      // write update to weights file
      st->write_g_iter(system , system->enhanced->output_dir);
      // TODO
      st->collected_iter_samples=0;
      st->current_iter++;
      st->just_iterated = true;
    }
  }

  // Iteration through temperatures
  if((system->run->step - system->run->step0) % st->steps_per_temp == 0 
  && system->run->step != system->run->step0 && st->high_idx <= st->N_temp){
    if(st->high_idx == st->N_temp){
      st->low_idx = 0;
    } else {
      st->low_idx++;
      st->high_idx++;
    }
  }
}

void SimulatedTempering::recv_st(){
  int size = N_temp*sizeof(real);
  cudaMemcpy(&U, U_d, sizeof(real_e), cudaMemcpyDefault);
  cudaMemcpy(&U_prime, U_prime_d, sizeof(real_e), cudaMemcpyDefault);
  cudaMemcpy(&U_ss, U_ss_d, sizeof(real_e), cudaMemcpyDefault);
  if(U_su_d) {
    cudaMemcpy(&U_su, U_su_d, sizeof(real_e), cudaMemcpyDefault);
  } else {
    U_su = 0;
  }
  cudaMemcpy(&scale_ss, scale_ss_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&temp_curr_idx, temp_curr_idx_d, sizeof(int), cudaMemcpyDefault);
  cudaMemcpy(g, g_d, size, cudaMemcpyDefault);
}

void log_st(System* system){}

void SimulatedTempering::write_g_iter(System* system, std::string output_dir){
  SimulatedTempering* st = system->enhanced->simulatedTempering;
  const char* mode = st->do_restart ? "a" : "w";
  if(!st->fp_g){
     std::string fnm = output_dir + "/" + st->fnm_g;
     st->fp_g = fopen(fnm.c_str(), mode);
     if(!st->fp_g){
       printf("Error opening %s. Please make directory!\n", fnm.c_str());
       exit(1);
     }
  }
  // 0 iteration g is all zeros, not included in file
  // write next iteration weights
  fprintf(st->fp_g, "%d ", st->current_iter+1);
  for(int i = 0; i < st->N_temp; i++){
    fprintf(st->fp_g, "%f ", st->g[i]);
  }
  fprintf(st->fp_g, "\n");
  fflush(st->fp_g);
}

void write_small_st(System* system, std::string output_dir){
  SimulatedTempering* st = system->enhanced->simulatedTempering;
  int size = st->N_temp*sizeof(real);

  // Temps
  if(!st->fp_temps){ // only do this once
    std::string fnm = output_dir + "/" + st->fnm_temps;
    st->fp_temps= fopen(fnm.c_str(), "w");
    if(!st->fp_temps){
      printf("Error opening %s. Please make directory!\n", fnm.c_str());
      exit(1);
    }
    for(int i = 0; i < st->N_temp; i++){
      fprintf(st->fp_temps, "%f ", st->temperatures[i]);
    }
    fprintf(st->fp_temps, "\n");
    fflush(st->fp_temps);
  }
  // Betas
  if(!st->fp_betas){ // only do this once
    std::string fnm = output_dir + "/" + st->fnm_betas;
    st->fp_betas = fopen(fnm.c_str(), "w");
    if(!st->fp_betas){
      printf("Error opening %s. Please make directory!\n", fnm.c_str());
      exit(1);
    }
    for(int i = 0; i < st->N_temp; i++){
      real beta_k = 1.0/(kB*st->temperatures[i]);
      fprintf(st->fp_betas, "%f ", beta_k);
    }
    fprintf(st->fp_betas, "\n");
    fflush(st->fp_betas);
  }
  const char* mode = st->do_restart ? "a" : "w";
  // Current T
  if(!st->fp_current_T){
     std::string fnm = output_dir + "/" + st->fnm_current_T;
     st->fp_current_T = fopen(fnm.c_str(), mode); // when restarting, this fp_current_T will have extra lines
     if(!st->fp_current_T){
       printf("Error opening %s. Please make directory!\n", fnm.c_str());
       exit(1);
     }
  }
  fprintf(st->fp_current_T, "%d ", system->run->step);
  real eff_beta = st->scale_ss*(1.0/(kB*system->run->T));
  real eff_temp = 1.0/(kB*eff_beta);
  fprintf(st->fp_current_T, "%f ", eff_temp);
  fprintf(st->fp_current_T, "\n");
  fflush(st->fp_current_T);
  // MBAR restart File 
  if(!st->fp_mbar){
     std::string fnm = output_dir + "/" + st->fnm_mbar;
     st->fp_mbar = fopen(fnm.c_str(), mode);
     if(!st->fp_mbar){
       printf("Error opening %s. Please make directory!\n", fnm.c_str());
       exit(1);
     }
     if(mode == "a"){fprintf(st->fp_mbar, "\n");}
  }
  int correction = st->just_iterated ? 1 : 0;
  fprintf(st->fp_mbar, "%d %d %f %f %f %f\n", st->current_iter - correction, st->temp_curr_idx, 1.0/(kB*system->run->T), st->betas[st->temp_curr_idx], st->U_ss, st->U_su);
  fflush(st->fp_mbar);
  // g.dat written separately as iterations complete
  // MBAR reweighting file(s)
  if(!st->fp_mbar_iter){
     std::string fnm = st->fnm_mbar_iter;
     st->fp_mbar_iter = fopen(fnm.c_str(), "w"); // always write
     if(!st->fp_mbar_iter){
       printf("Error opening %s. Please make directory!\n", fnm.c_str());
       exit(1);
     }
  }
  correction = st->just_iterated ? 1 : 0;
  fprintf(st->fp_mbar_iter, "%d %d %f %f %f %f\n", st->current_iter - correction, st->temp_curr_idx, 1.0/(kB*system->run->T), st->betas[st->temp_curr_idx], st->U_ss, st->U_su);
  fflush(st->fp_mbar_iter);
  st->just_iterated = false;
}

// written by Claude, then checked that it gave correct behavior 
int SimulatedTempering::read_restart(System* system, std::string output_dir) {
  if (!do_restart) return 0;

  std::string g_path    = output_dir + "/" + fnm_g;
  std::string mbar_path = output_dir + "/" + fnm_mbar;
  std::string T_path    = output_dir + "/" + fnm_current_T;

  // --- Check for inconsistent state: mbar.dat exists but g.dat does not ---
  FILE* fp_mbar_check = fopen(mbar_path.c_str(), "r");
  FILE* fp_g_check    = fopen(g_path.c_str(), "r");
  if (fp_mbar_check && !fp_g_check) {
    printf("Error: %s exists but %s does not. Cannot restart safely. Starting fresh!\n",
           mbar_path.c_str(), g_path.c_str());
    if (fp_mbar_check) fclose(fp_mbar_check);
    do_restart = false;
    return 0;
  }//13.7470
  if (fp_mbar_check) fclose(fp_mbar_check);
  if (!fp_g_check) {
    printf("No %s found, starting from scratch.\n", g_path.c_str());
    return 0;
  }
  fclose(fp_g_check);

  // --- Step 1: Read g.dat ---
  FILE* fp_g_r = fopen(g_path.c_str(), "r");
  int restart_iter = 0;
  int lines_read = 0;
  while (true) {
    int iter_num;
    if (fscanf(fp_g_r, "%d", &iter_num) != 1) break;
    bool ok = true;
    for (int k = 0; k < N_temp; k++) {
      double val;
      if (fscanf(fp_g_r, "%lf", &val) != 1) { ok = false; break; }
      g_data[(iter_num % iter_history) * N_temp + k] = (real)val;
    }
    if (!ok) break;
    restart_iter = iter_num;
    lines_read++;
  }
  fclose(fp_g_r);

  if (lines_read == 0) return 0;

  memcpy(g, &g_data[(restart_iter % iter_history) * N_temp], N_temp * sizeof(real));
  printf("Restarting from iteration %d, g: [", restart_iter);
  for (int k = 0; k < N_temp; k++) printf("%f ", g[k]);
  printf("]\n");

  if (restart_iter >= total_iters) {
    printf("Iterations complete (%d/%d), skipping mbar.dat read.\n",
           restart_iter, total_iters);
    return restart_iter;
  }

  // --- Step 2: Read mbar.dat ---
  FILE* fp_mbar_r = fopen(mbar_path.c_str(), "r");
  if (!fp_mbar_r) {
    printf("No %s found, starting fresh mbar data.\n", mbar_path.c_str());
    return restart_iter;
  }

  int* sample_counts   = (int*)calloc(iter_history, sizeof(int));
  long last_valid_pos  = 0;
  int  valid_mbar_samples = 0;

  while (true) {
    int iter_mbar, k_idx;
    double beta0_val, betak_val, U_ss_val, U_su_val;
    if (fscanf(fp_mbar_r, "%d %d %lf %lf %lf %lf",
               &iter_mbar, &k_idx,
               &beta0_val, &betak_val,
               &U_ss_val, &U_su_val) != 6) break;
    if (iter_mbar >= restart_iter) break;

    int eff_i = iter_mbar % iter_history;
    int s = sample_counts[eff_i];
    if (s < iteration_samples) {
      int idx = sample_data_length * (eff_i * iteration_samples + s);
      mbar_data[idx + mbar_k]     = (real)k_idx;
      mbar_data[idx + mbar_beta0] = (real)beta0_val;
      mbar_data[idx + mbar_betak] = (real)betak_val;
      mbar_data[idx + mbar_Uss]   = (real)U_ss_val;
      mbar_data[idx + mbar_Usu]   = (real)U_su_val;
      sample_counts[eff_i]++;
    }
    valid_mbar_samples++;
    last_valid_pos = ftell(fp_mbar_r);
  }
  fclose(fp_mbar_r);

  for (int i = 0; i < iter_history; i++) {
    if (sample_counts[i] > 0 && sample_counts[i] != iteration_samples) {
      printf("Warning: slot %d has %d samples, expected %d\n",
             i, sample_counts[i], iteration_samples);
    }
  }
  free(sample_counts);

  // --- Step 3: Truncate mbar.dat and current_T.dat if iterations incomplete ---
  if (restart_iter < total_iters) {
    // Truncate mbar.dat
    FILE* ft = fopen(mbar_path.c_str(), "r+");
    if (ft) {
      ftruncate(fileno(ft), last_valid_pos);
      fclose(ft);
      printf("Truncated %s to %ld bytes (%d samples)\n",
             mbar_path.c_str(), last_valid_pos, valid_mbar_samples);
    }

    // Truncate current_T.dat to same number of lines
    FILE* fp_T = fopen(T_path.c_str(), "r");
    if (fp_T) {
      long T_valid_pos = 0;
      int lines = 0;
      int c;
      while (lines < valid_mbar_samples && (c = fgetc(fp_T)) != EOF) {
        if (c == '\n') {
          lines++;
          T_valid_pos = ftell(fp_T);
        }
      }
      fclose(fp_T);
      if (lines == valid_mbar_samples) {
        FILE* ft_T = fopen(T_path.c_str(), "r+");
        if (ft_T) {
          ftruncate(fileno(ft_T), T_valid_pos);
          fclose(ft_T);
          printf("Truncated %s to %ld bytes (%d samples)\n",
                 T_path.c_str(), T_valid_pos, valid_mbar_samples);
        }
      } else {
        printf("Warning: %s has fewer lines (%d) than valid mbar samples (%d)\n",
               T_path.c_str(), lines, valid_mbar_samples);
      }
    }
  } 

  return restart_iter;
}


void write_big_st(System* system, std::string output_dir){};
