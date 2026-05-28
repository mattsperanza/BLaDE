#include "enhanced/osrw/osrw.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/state.h"
#include "system/system.h"
#include "enhanced/enhanced.h"
#include "system/potential.h"
#include "main/gpu_check.h"
#include "main/real3.h"
#include "io/io.h"
#include "main/gpu_check.h"
#include <string>
#include <stdlib.h>

OrthogonalSpaceRandomWalk::~OrthogonalSpaceRandomWalk(){
  if(dUdL_copy) free(dUdL_copy);
  if(dUdL_copy_d) cudaFree(dUdL_copy_d);
  if(dGdF) free(dGdF);
  if(dGdF_d) cudaFree(dGdF_d);
  // 2D
  if(lambda_counts_2D) free(lambda_counts_2D);
  if(lambda_counts_2D_d) cudaFree(lambda_counts_2D_d);
  if(hist_weights_2D) free(hist_weights_2D);
  if(hist_weights_2D_d) cudaFree(hist_weights_2D_d);
  if(potential_2D) free(potential_2D);
  if(potential_2D_d) cudaFree(potential_2D_d);
  if(avg_dUdL_2D) free(avg_dUdL_2D);
  if(avg_dUdL_2D_d) cudaFree(avg_dUdL_2D_d);
  if(m2_dUdL_2D) free(m2_dUdL_2D);
  if(m2_dUdL_2D_d) cudaFree(m2_dUdL_2D_d);
  // 1D
  if(lambda_counts_1D) free(lambda_counts_1D);
  if(lambda_counts_1D_d) cudaFree(lambda_counts_1D_d);
  if(ensemble_dUdL) free(ensemble_dUdL);
  if(std_dUdL_d) cudaFree(std_dUdL_d);
  if(std_dUdL) free(std_dUdL);
  if(variance_dUdL_d) cudaFree(variance_dUdL_d);
  if(min_dUdL_id_d) cudaFree(min_dUdL_id_d);
  if(max_dUdL_id_d) cudaFree(max_dUdL_id_d);
  if(current_temper_bias_d) cudaFree(current_temper_bias_d);
};

void parse_osrw(char* line, OrthogonalSpaceRandomWalk* osrw){
  char token[MAXLENGTHSTRING];
  io_nexta(line, token);
  // Switches
  if(strcmp(token, "target_site") == 0){
    osrw->target_site = io_nexti(line);
  } else if(strcmp(token, "do_abf") == 0){
    osrw->do_abf = io_nextb(line);
  } else if(strcmp(token, "do_meta") == 0){
    osrw->do_meta = io_nextb(line);
  } else if(strcmp(token, "do_sample") == 0){
    osrw->do_sample = io_nextb(line);
  } else if(strcmp(token, "do_restart") == 0){
    osrw->do_restart = io_nextb(line);
  } else if(strcmp(token, "do_temper") == 0){
    osrw->do_temper = io_nextb(line);
  } else if(strcmp(token, "remove_bonded") == 0){
    osrw->remove_bonded = io_nextb(line);
  } else if(strcmp(token, "remove_recip") == 0){
    osrw->remove_recip = io_nextb(line);
  } else if(strcmp(token, "sample_weighting") == 0){
    osrw->sample_weighting = io_nextb(line);
  } else if(strcmp(token, "sample_freq") == 0){
    osrw->sample_freq = io_nexti(line);
  // Grid / metadynamics
  } else if(strcmp(token, "L_res") == 0){
    if(!osrw->init){ osrw->L_res = io_nextf(line); }
    else { printf("!!!!! Cannot change L_res after initialization!\n"); }
  } else if(strcmp(token, "dUdL_res") == 0){
    if(!osrw->init){ osrw->dUdL_res = io_nextf(line); }
    else { printf("!!!!! Cannot change dUdL_res after initialization!\n"); }
  } else if(strcmp(token, "dUdL_max") == 0){
    if(!osrw->init){ osrw->dUdL_max = io_nextf(line); }
    else { printf("!!!!! Cannot change dUdL_max after initialization!\n"); }
  } else if(strcmp(token, "meta_bias_mag") == 0){
    osrw->meta_bias_mag = io_nextf(line);
  } else if(strcmp(token, "meta_L_std") == 0){
    osrw->meta_L_std = io_nextf(line);
  } else if(strcmp(token, "meta_dUdL_std") == 0){
    osrw->meta_dUdL_std = io_nextf(line);
  } else if(strcmp(token, "temper_factor") == 0){
    osrw->temper_factor = io_nextf(line);
    if(osrw->temper_factor < 1 || abs(osrw->temper_factor-1) < 1e-4){
      printf("Temper factor less than or too close to 1.\n");
      exit(1);
    }
  } else if(strcmp(token, "n_std_search") == 0){
    osrw->n_std_search = io_nexti(line);
  // ABF
  } else if(strcmp(token, "abf_warmup") == 0){
    osrw->abf_warmup = io_nexti(line);
  // Restart / logging
  } else if(strcmp(token, "write_restart_freq") == 0){
    osrw->write_restart_freq = io_nexti(line);
  } else if(strcmp(token, "log_freq") == 0){
    osrw->log_freq = io_nexti(line);
  } else {
    printf("OSRW: didn't recognize option %s\n", token);
    exit(1);
  }
};

// This only gets called the first time enhanced->initialize() gets called
void OrthogonalSpaceRandomWalk::initialize(System* system){
  printf("Initializing OSRW!\n");
  // ----- Derived grid parameters -----
  dUdL_min = -dUdL_max;
  n_L_bins = (int) round(1.0 / L_res) + 1; // bins on both edges of [0,1]
  n_dUdL_bins = (int) round((dUdL_max - dUdL_min) / dUdL_res) + 1; // bins on both edges
  if(n_L_bins < 10 || n_dUdL_bins < 10){
    printf("OSRW: please choose reasonable bin resolutions (need >=2 bins per axis)!\n");
    printf("Exiting...\n"); exit(1);
  }
  bin_width_L    = 1.0 / (n_L_bins - 1);
  bin_width_dUdL = (dUdL_max - dUdL_min) / (n_dUdL_bins - 1);
  // Need at least 2 bins per std so gaussians are resolved
  if(meta_L_std < bin_width_L/2.0 || meta_dUdL_std < bin_width_dUdL/2.0){
    printf("OSRW: meta_L_std/meta_dUdL_std must be at least half the bin width (>=2 bins/std)!\n");
    printf("Exiting...\n"); exit(1);
  }
  half_search_bins_L    = (int) ceil(n_std_search * meta_L_std    / bin_width_L);
  half_search_bins_dUdL = (int) ceil(n_std_search * meta_dUdL_std / bin_width_dUdL);
  if(half_search_bins_L > n_L_bins-1){
    printf("OSRW: requested L search wider than the [0,1] interval. The reflecting boundary "
           "does not support multiple reflections. Reduce meta_L_std or n_std_search!\n");
    printf("Exiting...\n"); exit(1);
  }

  int n2D = n_L_bins * n_dUdL_bins;
  // 2D memory - (L, dU_sub/dL)
  lambda_counts_2D = (real*)calloc(n2D, sizeof(real));
  cudaMalloc(&lambda_counts_2D_d, n2D*sizeof(real));
  cudaMemcpy(lambda_counts_2D_d, lambda_counts_2D, n2D*sizeof(real), cudaMemcpyDefault);
  hist_weights_2D = (real*)calloc(n2D, sizeof(real));
  cudaMalloc(&hist_weights_2D_d, n2D*sizeof(real));
  cudaMemcpy(hist_weights_2D_d, hist_weights_2D, n2D*sizeof(real), cudaMemcpyDefault);
  potential_2D = (real*)calloc(n2D, sizeof(real));
  cudaMalloc(&potential_2D_d, n2D*sizeof(real));
  cudaMemcpy(potential_2D_d, potential_2D, n2D*sizeof(real), cudaMemcpyDefault);
  avg_dUdL_2D = (real*)calloc(n2D, sizeof(real));
  cudaMalloc(&avg_dUdL_2D_d, n2D*sizeof(real));
  cudaMemcpy(avg_dUdL_2D_d, avg_dUdL_2D, n2D*sizeof(real), cudaMemcpyDefault);
  m2_dUdL_2D = (real*)calloc(n2D, sizeof(real));
  cudaMalloc(&m2_dUdL_2D_d, n2D*sizeof(real));
  cudaMemcpy(m2_dUdL_2D_d, m2_dUdL_2D, n2D*sizeof(real), cudaMemcpyDefault);
  // 1D memory - L
  lambda_counts_1D = (real*)calloc(n_L_bins, sizeof(real));
  cudaMalloc(&lambda_counts_1D_d, n_L_bins*sizeof(real));
  cudaMemcpy(lambda_counts_1D_d, lambda_counts_1D, n_L_bins*sizeof(real), cudaMemcpyDefault);
  ensemble_dUdL = (real*)calloc(n_L_bins, sizeof(real));
  cudaMalloc(&std_dUdL_d, n_L_bins*sizeof(real));
  cudaMemcpy(std_dUdL_d, ensemble_dUdL, n_L_bins*sizeof(real), cudaMemcpyDefault);
  std_dUdL = (real*)calloc(n_L_bins, sizeof(real));
  cudaMalloc(&variance_dUdL_d, n_L_bins*sizeof(real));
  cudaMemset(variance_dUdL_d, 0, n_L_bins*sizeof(real));
  dUdL_copy = (real*)calloc(system->msld->blockCount, sizeof(real));
  cudaMalloc(&dUdL_copy_d, system->msld->blockCount*sizeof(real));
  cudaMemcpy(dUdL_copy_d, dUdL_copy, system->msld->blockCount*sizeof(real), cudaMemcpyDefault);

  int tmp[n_L_bins];
  for(int i = 0; i < n_L_bins; i++){ tmp[i] = n_dUdL_bins; }
  cudaMalloc(&min_dUdL_id_d, n_L_bins*sizeof(int));
  cudaMemcpy(min_dUdL_id_d, tmp, n_L_bins*sizeof(int), cudaMemcpyDefault);
  cudaMalloc(&max_dUdL_id_d, n_L_bins*sizeof(int));
  cudaMemset(max_dUdL_id_d, 0, n_L_bins*sizeof(int));

  dGdF = (real*)calloc(system->msld->blockCount, sizeof(real));
  cudaMalloc(&dGdF_d, system->msld->blockCount*sizeof(real));
  cudaMemset(dGdF_d, 0, system->msld->blockCount*sizeof(real));

  cudaMalloc(&current_temper_bias_d, sizeof(real));
  cudaMemset(current_temper_bias_d, 0, sizeof(real));

  if(do_restart) restart(system);

  // ----- Validate target site -----
  if(target_site <= 0 || target_site >= system->msld->siteCount){
    printf("OSRW: choose a valid site! %d is not a valid site!\n", target_site);
    exit(1);
  }
  if(system->msld->blocksPerSite[target_site] != 2){
    printf("OSRW can only be applied on a site with exactly 2 substituents!\n");
    exit(1);
  }

  init = true;
};

// bin index where bin centers range evenly spaced [fmin, fmax]
static int __device__ histogram_index(real F, int n_bins, real fmin, real fmax){
  real res = (fmax-fmin)/(n_bins-1.0);
  return (int)round((F-fmin)/res);
}

// One block per L-search bin; threads sweep the dU/dL search window.
void __global__ getforce_osrw_meta_kernel(
  int n_L, int n_F, real fmin, real fmax,
  int search_L, int search_F,
  real bias_mag, real L_std, real dUdL_std,
  real* lambda, real* dUdL_tot, real* dUdL_bonded, real* dUdL_recip, real* dUdL_restrain,
  real* hist_weights_2D,
  bool update_potential, 
  real* current_bias, real kT, real temper_factor,
  // Outputs
  real* hist_potential_2D,
  real* lambdaForce, real* dGdF, real_e* energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy = 0;
  int len_y = 2*search_F+1; // column length
  int len_x = 2*search_L+1; // row length

  if (i <= (len_x)*(len_y)){ 
    int thr_x = i / len_y; // which column
    int thr_y = i % len_y; // which row
    int iL = thr_x - search_L; // L offset
    int s  = thr_y - search_F; // dU/dL offset
  
    real L = lambda[1];
    real F = dUdL_tot[1] - dUdL_tot[0];
    if(dUdL_bonded){ F -= (dUdL_bonded[1] - dUdL_bonded[0]); }
    if(dUdL_recip){ F -= (dUdL_recip[1] - dUdL_recip[0]); }
    if(dUdL_restrain) {F -= dUdL_restrain[1] - dUdL_restrain[0];}
  
    real bin_width_L = 1.0/(n_L-1.0);
    real bin_width_F = (fmax-fmin)/(n_F-1.0);
  
    int X = histogram_index(L, n_L, 0.0, 1.0);
    int Y = histogram_index(F, n_F, fmin, fmax);
  
    int jb = X + iL;
    int k  = Y + s;
  
    // reflecting boundary in L
    real L_center = jb*bin_width_L;
    real F_center = fmin + k*bin_width_F;
    // Get weight from mirrored bin
    real mirror_factor = 1.0;
    if(jb < 0){ jb = -jb; }
    if(jb >= n_L){ jb = (n_L-1) - (jb-(n_L-1)); }
    if(jb == 0 || jb == n_L-1){ mirror_factor = 2.0; } // edge double counting 
    if(k >= 0 && k < n_F && jb >= 0 && jb < n_L){ // don't index outside
      real dL = (L - L_center)/L_std;
      real dF = (F - F_center)/dUdL_std;
      real gL = exp(-0.5*dL*dL);
      real gF = exp(-0.5*dF*dF);
      if (!update_potential) { // force call, compute force from grid of bins
        real h  = hist_weights_2D[jb*n_F + k];
        real bias = bias_mag*mirror_factor*h*gL*gF;
        lEnergy = bias;
        real dGdL = -(dL/L_std)*bias; // dV/dL
        real dGdF_local = -(dF/dUdL_std)*bias; // dV/dF
        atomicAdd(&lambdaForce[1], dGdL);
        atomicAdd(&dGdF[1], dGdF_local);
        atomicAdd(&dGdF[0], -dGdF_local);
      } else { // update potential call, compute potential contribution of new sample at (L, F)
        real factor = exp(-current_bias[0]/(temper_factor-1.0)*kT);
        atomicAdd(&hist_potential_2D[jb*n_F + k], bias_mag*factor*mirror_factor*gL*gF); // mirror makes this a race
      }
    }
  }
  if(energy){
    real_sum_reduce(lEnergy, sEnergy, energy);
  }
}

void __global__ reweight_dUdL_kernel(
  int n_L, int n_F, int L_search, real* lambdas, 
  real kT, real temper_correction, 
  real* counts_2D, real* avg_2D, real* m2_2D, real* potential_2D,
  int* min_dUdL_id, int* max_dUdL_id,
  bool sample_weighting,
  // Outputs
  real* ensemble_dUdL, real* std_dUdL)
{
  int X = blockIdx.x*blockDim.x + threadIdx.x; // L bin
  int curr_bin = histogram_index(lambdas[1], n_L, 0.0, 1.0);
  X -= (L_search-1)/2; // [-half, half]
  X = curr_bin + X;
  if(X < 0 || X >= n_L) return;

  real offset = 0.0;
  real wsum = 0.0; // sum weights
  real w_dUdL = 0.0; // sum w * <dU/dL>
  real w_dUdL2 = 0.0; // sum w * <dU/dL>^2 (for variance)
  int low_id = min_dUdL_id[X];
  int high_id = max_dUdL_id[X];
  for(int Y = low_id; Y <= high_id; Y++){
    int idx = X*n_F + Y;
    if(counts_2D[idx] < 1e-5){ continue; } // skip unsampled bins
    real bias = temper_correction * potential_2D[idx] / kT;
    if(bias > offset){
      real corr = exp(offset - bias);
      wsum *= corr;
      w_dUdL *= corr;
      w_dUdL2 *= corr;
      offset = bias;
    }
    real w = exp(bias - offset);
    w *= sample_weighting ? counts_2D[idx] : 1.0;
    real mean = avg_2D[idx];
    wsum += w;
    w_dUdL += w * mean;
    w_dUdL2 += w * mean * mean;
  }

  real ens = 0.0, ens2 = 0.0;
  if(wsum > 1e-5){
    ens  = w_dUdL / wsum;
    ens2 = w_dUdL2 / wsum;
  }
  std_dUdL[X] = sqrt(ens2 - ens*ens);
  ensemble_dUdL[X] = ens;
}

void update_osrw_potential(System* system){
  OrthogonalSpaceRandomWalk* osrw = system->enhanced->osrw;
  State* state = system->state;
  Run* run = system->run;
  // Build potential everywhere
  int id = system->msld->siteBound[osrw->target_site];
  if(osrw->do_meta){ // if meta is off, <dU/dL> should be calculated with uniform weights (pot_2D=0 everywhere)
    int nL = 2*osrw->half_search_bins_L + 1;
    int nF = 2*osrw->half_search_bins_dUdL + 1;
    real* bonded = (osrw->remove_bonded && state->dUdL_BA_d) ? & state->dUdL_BA_d[id] : NULL;
    real* recip = (osrw->remove_recip && state->dUdL_recip_d) ? & state->dUdL_recip_d[id] : NULL;
    real* restrain = state->dUdL_restrain_d ? &state->dUdL_restrain_d[id] : NULL;
    getforce_osrw_meta_kernel<<<(nL*nF+BLBO-1)/BLBO,BLBO, 0, run->enhancedStream>>>(
      osrw->n_L_bins, osrw->n_dUdL_bins, osrw->dUdL_min, osrw->dUdL_max,
      osrw->half_search_bins_L, osrw->half_search_bins_dUdL,
      osrw->meta_bias_mag,osrw->meta_L_std, osrw->meta_dUdL_std,
      &state->lambda_fd[id], &state->lambdaForce_d[id], bonded, recip, restrain,
      osrw->hist_weights_2D_d,
      true, // update potential
      osrw->current_temper_bias_d, kB*system->run->T, osrw->temper_factor,
      // Outputs (lambda force at the site, and dGdF at the site)
      osrw->potential_2D_d,
      &state->lambdaForce_d[id], &osrw->dGdF_d[id], NULL);
  }

  // Eq: 7 in: Annu. Rev. Phys. Chem. 2016. 67:159–84 to convert WT potential into -'ve FES
  // Large temper_factor => correction = 1  (untempered limit)
  // Small temper_factor => correction = large (unbiased limit)
  real temper_correction = osrw->do_temper ? 1.0/(1.0 - 1.0/osrw->temper_factor) : 1.0;
  int L_update = 2*osrw->half_search_bins_L + 1;
  reweight_dUdL_kernel<<<(L_update+BLBO-1)/BLBO, BLBO, 0, run->enhancedStream>>>(
    osrw->n_L_bins, osrw->n_dUdL_bins, L_update, &state->lambda_fd[id], kB*run->T, temper_correction, 
    osrw->lambda_counts_2D_d, osrw->avg_dUdL_2D_d, osrw->m2_dUdL_2D_d, osrw->potential_2D_d,
    osrw->min_dUdL_id_d, osrw->max_dUdL_id_d,
    osrw->sample_weighting,
    osrw->std_dUdL_d, osrw->variance_dUdL_d);
}

void __global__ add_sample_osrw_kernel(
  int n_L, int n_F, real fmin, real fmax,
  real kT, bool do_temper, 
  real temper_factor, real* potential_2D,
  real* lambda, real* lambdaForce, real* dUdL_BA, real* dUdL_recip, real* dUdL_restrain,
  int* min_dUdL_id, int* max_dUdL_id,
  // Outputs
  real* counts_1D, real* counts_2D, real* weights_2D,
  real* avg_2D, real* m2_2D, real* current_temper_bias)
{
  real L = 1.0 - lambda[0];
  real dUdL_total = lambdaForce[1] - lambdaForce[0];
  real dUdL_cv = dUdL_total;
  if(dUdL_BA){
    dUdL_cv -= (dUdL_BA[1] - dUdL_BA[0]);
  }
  if (dUdL_recip){
    dUdL_cv -= (dUdL_recip[1] - dUdL_recip[0]);
  }
  if (dUdL_restrain){ 
    dUdL_cv -= dUdL_restrain[1]-dUdL_restrain[0];
  }
  int X = histogram_index(L, n_L, 0, 1);
  int Y = histogram_index(dUdL_cv, n_F, fmin, fmax);
  if(X < 0 || X >= n_L || Y < 0 || Y >= n_F){ printf("Out of bounds sample!!\n"); return; } // clip out-of-range dU/dL
  int idx = X*n_F + Y;

  if(Y < min_dUdL_id[X]){ min_dUdL_id[X] = Y; }
  if(Y > max_dUdL_id[X]){ max_dUdL_id[X] = Y; }

  // 1D and 2D counts
  counts_1D[X] += 1.0;
  counts_2D[idx] += 1.0;

  // 2D tracking of <dU_total/dL> in each bin
  real prev_delta = dUdL_total - avg_2D[idx];
  avg_2D[idx] += prev_delta / counts_2D[idx];
  m2_2D[idx]  += prev_delta * (dUdL_total - avg_2D[idx]);

  // Add meta sample to (L, dU_cv/dL) - Tempered - Eq: 13 in: Annu. Rev. Phys. Chem. 2016. 67:159–84
  current_temper_bias[0] = potential_2D[idx];
  real factor = do_temper ? exp(-current_temper_bias[0]/((temper_factor-1.0)*kT)) : 1.0;
  weights_2D[idx] += factor;
}

void sample_osrw(System* system, int step){
  OrthogonalSpaceRandomWalk* osrw = system->enhanced->osrw;
  State* state = system->state;
  Run* run = system->run;

  if(osrw->do_sample && step % osrw->sample_freq == 0){
    int id = system->msld->siteBound[osrw->target_site];
    real* bonded = (osrw->remove_bonded && state->dUdL_BA_d) ? &state->dUdL_BA_d[id] : NULL;
    real* recip = (osrw->remove_recip && state->dUdL_recip_d) ? &state->dUdL_recip_d[id] : NULL;
    real* restrain = state->dUdL_restrain_d ? &state->dUdL_restrain_d[id] : NULL;
    add_sample_osrw_kernel<<<1, 1, 0, run->enhancedStream>>>(
      osrw->n_L_bins, osrw->n_dUdL_bins, osrw->dUdL_min, osrw->dUdL_max,
      kB*run->T, osrw->do_temper, 
      osrw->temper_factor, osrw->potential_2D_d,
      &state->lambda_fd[id], &state->lambdaForce_d[id], bonded, recip, restrain,
      osrw->min_dUdL_id_d, osrw->max_dUdL_id_d,
      // Outputs
      osrw->lambda_counts_1D_d, osrw->lambda_counts_2D_d, osrw->hist_weights_2D_d,
      osrw->avg_dUdL_2D_d, osrw->m2_dUdL_2D_d, osrw->current_temper_bias_d);
    update_osrw_potential(system);
  }
};

void __global__ getforce_osrw_abf_kernel(
  int n_bins, real* lambda, real* dUdL_avg, real* counts, int abf_warmup,
  real* lambdaForce, real_e* energy){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  if (i < n_bins){
    real L = 1.0-lambda[0];
    real bin_width = (1.0)/(n_bins-1.0);
    real dUdL_curr = dUdL_avg[i];
    dUdL_curr *= counts[i] < abf_warmup ? counts[i]/abf_warmup : 1;
    if (i >= 1){ // Each thread computes integral from previous bin to this bin
      real dUdL_prev = dUdL_avg[i-1];
      dUdL_prev *= counts[i-1] < abf_warmup ? counts[i-1]/abf_warmup : 1;
      lEnergy = bin_width*(dUdL_curr+dUdL_prev)/2.0; // trapezoid up to lambda
      if(L >= (i-1.0)*bin_width && L < i*bin_width){ // L is between last bin center and current bin center
        real interp = (L-(i-1.0)*bin_width)/bin_width;
        real dUdL_up = (1.0-interp)*dUdL_prev + interp*dUdL_curr;
        real width = (L-(i-1.0)*bin_width);
        lEnergy = width*(dUdL_prev+dUdL_up)/2.0;
        atomicAdd(&lambdaForce[0], dUdL_up); 
      } else if(L <= (i-1)*bin_width){ // L is less than lower bin center
        lEnergy = 0;
      }
    }
  }
  if (energy){
    // ABF adds -'ve F(L)
    real_sum_reduce(-lEnergy,sEnergy,energy);
  }
};

void getforce_osrw(System* system, int step, bool calcEnergy){
  OrthogonalSpaceRandomWalk* osrw = system->enhanced->osrw;
  State* state = system->state;
  Run* r = system->run;

  int shMem = 0;
  real_e* pEnergy = NULL;
  if(calcEnergy){
    shMem = BLBO*sizeof(real)/32;
    pEnergy = state->energy_d + eeenhanced;
  }

  int id = system->msld->siteBound[osrw->target_site];

  if(osrw->do_abf && !osrw->force_test){
    getforce_osrw_abf_kernel<<<(osrw->n_L_bins+BLBO-1)/BLBO, BLBO, shMem, r->enhancedStream>>>(
      osrw->n_L_bins, &state->lambda_fd[id], osrw->std_dUdL_d, osrw->lambda_counts_1D_d, osrw->abf_warmup,
      &state->lambdaForce_d[id], pEnergy);
  }

  if(osrw->do_meta){
    if (!osrw->force_test){
      int nL = 2*osrw->half_search_bins_L + 1;
      int nF = 2*osrw->half_search_bins_dUdL + 1;
      real* bonded = osrw->remove_bonded && state->dUdL_BA_d ? &state->dUdL_BA_d[id] : NULL;
      real* recip = osrw->remove_recip && state->dUdL_recip_d ? &state->dUdL_recip_d[id] : NULL;
      real* restrain = state->dUdL_restrain_d ? &state->dUdL_restrain_d[id] : NULL;
      cudaMemsetAsync(osrw->dGdF_d, 0, system->msld->blockCount*sizeof(real), r->enhancedStream); // reset dGdF memory
      getforce_osrw_meta_kernel<<<(nL*nF+BLBO-1)/BLBO,BLBO, shMem, r->enhancedStream>>>(
        osrw->n_L_bins, osrw->n_dUdL_bins, osrw->dUdL_min, osrw->dUdL_max,
        osrw->half_search_bins_L, osrw->half_search_bins_dUdL,
        osrw->meta_bias_mag,osrw->meta_L_std, osrw->meta_dUdL_std,
        &state->lambda_fd[id], &osrw->dUdL_copy_d[id], bonded, recip, restrain,
        osrw->hist_weights_2D_d,
        false, // update potential
        osrw->current_temper_bias_d, kB*system->run->T, osrw->temper_factor,
        // Outputs (lambda force at the site, and dGdF at the site)
        osrw->potential_2D_d,
        &state->lambdaForce_d[id], &osrw->dGdF_d[id], pEnergy);
    }

    // Wait on enhancedStream complete then launch on respective kernel streams
    cudaEventRecord(r->enhancedComplete, r->enhancedStream);
    int helper=(system->idCount==2); // 0 unless there are 2 GPUs, then it's 1.
    if (system->id == helper) {
      cudaStreamWaitEvent(r->bondedStream, r->enhancedComplete, 0);
      if(!osrw->remove_bonded){
        getforce_bond_oss(system);
        getforce_angle_oss(system);
      }
      getforce_impr_oss(system);
      getforce_dihe_oss(system);
      getforce_cmap_oss(system);
      getforce_nb14_oss(system);
      if(!osrw->remove_recip){
        getforce_nbex_oss(system);
      }
      cudaEventRecord(r->bondedComplete, r->bondedStream);
      cudaStreamWaitEvent(r->updateStream, r->bondedComplete, 0);
    }
    if (system->id>=0) {
      cudaStreamWaitEvent(r->nbdirectStream, r->enhancedComplete, 0);
      getforce_nbdirect_oss(system);
      cudaEventRecord(r->nbdirectComplete, r->nbdirectStream);
      cudaStreamWaitEvent(r->updateStream, r->nbdirectComplete, 0);
    }
    if (system->id==0 && !osrw->remove_recip) {
      cudaStreamWaitEvent(r->nbrecipStream, r->enhancedComplete, 0);
      getforce_ewaldself_oss(system);
      getforce_ewald_oss(system);
      cudaEventRecord(r->nbrecipComplete, r->nbrecipStream);
      cudaStreamWaitEvent(r->updateStream, r->nbrecipComplete, 0);
    }
  }
};

// ===========================================================================
// Receive / logging
// ===========================================================================
void recv_osrw(System* system){
  OrthogonalSpaceRandomWalk* osrw = system->enhanced->osrw;
  int n2D = osrw->n_L_bins * osrw->n_dUdL_bins;
  cudaMemcpy(osrw->lambda_counts_2D, osrw->lambda_counts_2D_d, n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(osrw->hist_weights_2D,  osrw->hist_weights_2D_d,  n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(osrw->potential_2D,     osrw->potential_2D_d,     n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(osrw->avg_dUdL_2D,      osrw->avg_dUdL_2D_d,      n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(osrw->m2_dUdL_2D,       osrw->m2_dUdL_2D_d,       n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(osrw->lambda_counts_1D, osrw->lambda_counts_1D_d, osrw->n_L_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(osrw->ensemble_dUdL,    osrw->std_dUdL_d,    osrw->n_L_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(osrw->std_dUdL,    osrw->variance_dUdL_d,    osrw->n_L_bins*sizeof(real), cudaMemcpyDefault);
};

static void osrw_print_real_array(real* arr, int len){
  if(arr){
    printf("[ ");
    for(int i = 0; i < len; i++){
      printf(i == len-1 ? "%7.2f " : "%7.2f, ", arr[i]);
    }
    printf("]");
  }
}

void log_osrw(System* system, int step){
  OrthogonalSpaceRandomWalk* osrw = system->enhanced->osrw;
  State* state = system->state;
  Msld* msld = system->msld;
  if(step % osrw->log_freq == 0){
    state->recv_energy();
    if(!osrw->do_sample){ printf("OSRW: NOT ADDING SAMPLES!!!!\n"); }
    printf("Step %d, U_enhanced: %8.2f:\n", step, state->energy[eeenhanced]);
    recv_osrw(system);
    state->recv_lambda();
    cudaMemcpy(state->lambdaForce, state->lambdaForce_d, state->lambdaCount*sizeof(real), cudaMemcpyDefault);
    int id = msld->siteBound[osrw->target_site];
    real L = 1.0 - state->lambda[id];
    real dUdL = state->lambdaForce[id+1] - state->lambdaForce[id];
    printf("Target Site: %d, Ref Block: %d, L: %3.2f, dU/dL: %8.2f\n",
           osrw->target_site, id, L, dUdL);

    if(osrw->do_temper){
      real kT = kB*system->run->T;
      real mb;
      cudaMemcpy(&mb, osrw->current_temper_bias_d, sizeof(real), cudaMemcpyDefault);
      real factor = exp(-mb/(kT*(osrw->temper_factor-1)));
      printf("Current Tempering Bias: %5.2f, Temper Factor: %5.2f, Decay: %5.2f\n",
             mb, osrw->temper_factor, factor);
    }
    if(osrw->do_abf){
      printf("Counts: ");
      osrw_print_real_array(osrw->lambda_counts_1D, osrw->n_L_bins);
      printf("\n");
      printf("<dU/dL>: ");
      osrw_print_real_array(osrw->ensemble_dUdL, osrw->n_L_bins);
      printf("\n");
      printf("std dU/dL: ");
      osrw_print_real_array(osrw->std_dUdL, osrw->n_L_bins);
      printf("\n");
      // dG 0->1 via trapezoidal integral of <dU/dL> over L in [0,1]
      real sum = 0;
      real bw = 1.0/(osrw->n_L_bins-1.0);
      for(int i = 0; i < osrw->n_L_bins-1; i++){
        sum += bw*(osrw->ensemble_dUdL[i] + osrw->ensemble_dUdL[i+1])/2.0;
      }
      printf("dG sub0->sub1: %f\n", sum);
    }
    printf("\n");
  }
};

// ===========================================================================
// Restart file IO
// ---------------------------------------------------------------------------
// Single file. Contains everything needed to resume exactly, INCLUDING the
// metadynamics potential. We persist the potential because it is maintained
// incrementally during the run (every new sample reconvolves only the affected
// columns), so there is no routine that re-evaluates it across the whole grid
// at load time. Writing it out lets a restart pick up the exact surface.
//
// 2D arrays are written as (n_L_bins rows) x (n_dUdL_bins cols) grids, one L
// row per line, so the file is human-readable and trivial to load (e.g. the
// companion read_osrw_restart.m MATLAB script).
// ===========================================================================
static void osrw_write_grid(FILE* fp, const char* name, real* host, int n_L, int n_F){
  fprintf(fp, "%s\n", name);
  for(int x = 0; x < n_L; x++){
    for(int y = 0; y < n_F; y++){
      fprintf(fp, "%.10g%c", host[x*n_F + y], y == n_F-1 ? '\n' : ' ');
    }
  }
}

void write_osrw(std::string dir_name, System* system, int step){
  OrthogonalSpaceRandomWalk* osrw = system->enhanced->osrw;
  if(step % osrw->write_restart_freq != 0) return;
  recv_osrw(system);
  int n_L = osrw->n_L_bins;
  int n_F = osrw->n_dUdL_bins;

  std::string filename = dir_name + "/" + osrw->fnm_osrw_rst;
  FILE* fp = fopen(filename.c_str(), "w");
  if(!fp){
    printf("OSRW: could not open %s for writing!\nExiting...\n", filename.c_str());
    exit(1);
  }
  /*
    File structure:
      target_site  <int>
      n_L_bins     <int>
      n_dUdL_bins  <int>
      dUdL_max     <real>
      L_res        <real>
      dUdL_res     <real>
      lambda_counts_1D
      <n_L values, one line>
      ensemble_dUdL
      <n_L values, one line>
      lambda_counts_2D
      <n_L rows x n_F cols>
      hist_weights_2D
      <n_L rows x n_F cols>
      potential_2D
      <n_L rows x n_F cols>
      avg_dUdL_2D
      <n_L rows x n_F cols>
      m2_dUdL_2D
      <n_L rows x n_F cols>
  */
  fprintf(fp, "target_site %d\n", osrw->target_site);
  fprintf(fp, "n_L_bins %d\n",    n_L);
  fprintf(fp, "n_dUdL_bins %d\n", n_F);
  fprintf(fp, "dUdL_max %.10g\n", osrw->dUdL_max);
  fprintf(fp, "L_res %.10g\n",    osrw->L_res);
  fprintf(fp, "dUdL_res %.10g\n", osrw->dUdL_res);

  // 1D arrays (one line each)
  fprintf(fp, "lambda_counts_1D\n");
  for(int i = 0; i < n_L; i++){ fprintf(fp, "%.10g%c", osrw->lambda_counts_1D[i], i==n_L-1?'\n':' '); }
  fprintf(fp, "ensemble_dUdL\n");
  for(int i = 0; i < n_L; i++){ fprintf(fp, "%.10g%c", osrw->ensemble_dUdL[i], i==n_L-1?'\n':' '); }

  // 2D arrays (grid form)
  osrw_write_grid(fp, "lambda_counts_2D", osrw->lambda_counts_2D, n_L, n_F);
  osrw_write_grid(fp, "hist_weights_2D",  osrw->hist_weights_2D,  n_L, n_F);
  osrw_write_grid(fp, "potential_2D",     osrw->potential_2D,     n_L, n_F);
  osrw_write_grid(fp, "avg_dUdL_2D",      osrw->avg_dUdL_2D,      n_L, n_F);
  osrw_write_grid(fp, "m2_dUdL_2D",       osrw->m2_dUdL_2D,       n_L, n_F);

  fflush(fp);
  fclose(fp);
};

void OrthogonalSpaceRandomWalk::restart(System* system){
  std::string fnm = system->enhanced->output_dir + "/" + fnm_osrw_rst;
  FILE* fp = fopen(fnm.c_str(), "r");
  if(!fp){
    printf("OSRW: no restart file found (%s), starting fresh.\n", fnm.c_str());
    return;
  }
  printf("OSRW: reading restart file %s\n", fnm.c_str());

  int n_L = n_L_bins;
  int n_F = n_dUdL_bins;
  int n2D = n_L * n_F;

  // Header scalars are read with fscanf; grids are read value-by-value with
  // fscanf too (whitespace-insensitive, so newlines between rows don't matter).
  char token[MAXLENGTHSTRING];
  while(fscanf(fp, "%s", token) == 1){
    if(strcmp(token, "target_site") == 0){
      int v; fscanf(fp, "%d", &v);
      if(v != target_site){
        printf("OSRW warning: restart target_site (%d) differs from current (%d). Using current.\n", v, target_site);
      }
    } else if(strcmp(token, "n_L_bins") == 0){
      int v; fscanf(fp, "%d", &v);
      if(v != n_L){ printf("OSRW error: restart n_L_bins (%d) != current (%d)!\nExiting...\n", v, n_L); exit(1); }
    } else if(strcmp(token, "n_dUdL_bins") == 0){
      int v; fscanf(fp, "%d", &v);
      if(v != n_F){ printf("OSRW error: restart n_dUdL_bins (%d) != current (%d)!\nExiting...\n", v, n_F); exit(1); }
    } else if(strcmp(token, "dUdL_max") == 0){
      double v; fscanf(fp, "%lf", &v);
      if(fabs(v - dUdL_max) > 1e-6){ printf("OSRW warning: restart dUdL_max (%g) != current (%g).\n", v, dUdL_max); }
    } else if(strcmp(token, "L_res") == 0){
      double v; fscanf(fp, "%lf", &v); // informational
    } else if(strcmp(token, "dUdL_res") == 0){
      double v; fscanf(fp, "%lf", &v); // informational
    } else if(strcmp(token, "lambda_counts_1D") == 0){
      for(int i = 0; i < n_L; i++){ double v; fscanf(fp, "%lf", &v); lambda_counts_1D[i] = (real)v; }
    } else if(strcmp(token, "ensemble_dUdL") == 0){
      for(int i = 0; i < n_L; i++){ double v; fscanf(fp, "%lf", &v); ensemble_dUdL[i] = (real)v; }
    } else if(strcmp(token, "lambda_counts_2D") == 0){
      for(int i = 0; i < n2D; i++){ double v; fscanf(fp, "%lf", &v); lambda_counts_2D[i] = (real)v; }
    } else if(strcmp(token, "hist_weights_2D") == 0){
      for(int i = 0; i < n2D; i++){ double v; fscanf(fp, "%lf", &v); hist_weights_2D[i] = (real)v; }
    } else if(strcmp(token, "potential_2D") == 0){
      for(int i = 0; i < n2D; i++){ double v; fscanf(fp, "%lf", &v); potential_2D[i] = (real)v; }
    } else if(strcmp(token, "avg_dUdL_2D") == 0){
      for(int i = 0; i < n2D; i++){ double v; fscanf(fp, "%lf", &v); avg_dUdL_2D[i] = (real)v; }
    } else if(strcmp(token, "m2_dUdL_2D") == 0){
      for(int i = 0; i < n2D; i++){ double v; fscanf(fp, "%lf", &v); m2_dUdL_2D[i] = (real)v; }
    } else {
      printf("OSRW restart: ignoring unrecognized token '%s'\n", token);
    }
  }
  fclose(fp);

  // Reconstruct the sampled dU/dL index extent for each L bin and a global
  // min metadynamics bias from the loaded grids (these are derived, not stored).
  int* tmp_min = (int*)malloc(n_L*sizeof(int));
  int* tmp_max = (int*)malloc(n_L*sizeof(int));
  for(int x = 0; x < n_L; x++){
    tmp_min[x] = n_F-1; // so empty columns have min > max and get skipped
    tmp_max[x] = 0;
    bool any = false;
    for(int y = 0; y < n_F; y++){
      int idx = x*n_F + y;
      if(lambda_counts_2D[idx] > 0.5){
        if(y < tmp_min[x]) tmp_min[x] = y;
        if(y > tmp_max[x]) tmp_max[x] = y;
        any = true;
      }
    }
    if(!any){ tmp_min[x] = n_F-1; tmp_max[x] = 0; } // explicit empty marker
  }

  // Push restored host memory to the GPU
  cudaMemcpy(lambda_counts_1D_d, lambda_counts_1D, n_L*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(std_dUdL_d,    ensemble_dUdL,    n_L*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(lambda_counts_2D_d, lambda_counts_2D, n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(hist_weights_2D_d,  hist_weights_2D,  n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(potential_2D_d,     potential_2D,     n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(avg_dUdL_2D_d,      avg_dUdL_2D,      n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(m2_dUdL_2D_d,       m2_dUdL_2D,       n2D*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(min_dUdL_id_d,      tmp_min,          n_L*sizeof(int),  cudaMemcpyDefault);
  cudaMemcpy(max_dUdL_id_d,      tmp_max,          n_L*sizeof(int),  cudaMemcpyDefault);

  free(tmp_min);
  free(tmp_max);

  printf("OSRW: restart complete.\n");
};