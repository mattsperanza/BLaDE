#include "enhanced/meta_abf/meta_abf.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/state.h"
#include "system/system.h"
#include "enhanced/enhanced.h"
#include "system/potential.h"
#include "main/gpu_check.h"
#include "main/real3.h"
#include "io/io.h"
#include <string>

MetaAdaptiveBiasingForce::~MetaAdaptiveBiasingForce(){
  if(grid.counts) free(grid.counts);
  if(grid.counts_d) cudaFree(grid.counts_d);
  if(grid.meta_weights) free(grid.meta_weights);
  if(grid.meta_weights_d) cudaFree(grid.meta_weights_d);
  if(grid.y_m2) free(grid.y_m2);
  if(grid.y_m2_d) cudaFree(grid.y_m2_d);
  if(grid.y_avg) free(grid.y_avg);
  if(grid.y_avg_d) cudaFree(grid.y_avg_d);

  if(alchemCV.x_d) cudaFree(alchemCV.x_d);
  if(alchemCV.y_d) cudaFree(alchemCV.y_d);

  if(y_std) free(y_std);
  if(abfU_d) cudaFree(abfU_d);
  if(abfG_d) cudaFree(abfG_d);
  if(metaU_d) cudaFree(metaU_d);
  if(metaG_d) cudaFree(metaG_d);
  if(tmpThetaF_d) cudaFree(tmpThetaF_d);
  if(tmpLmdF_d) cudaFree(tmpLmdF_d);
};

void parse_meta_abf(char* line, MetaAdaptiveBiasingForce* meta_abf){
  char token[MAXLENGTHSTRING];
  io_nexta(line, token);
  if(strcmp(token, "sample_freq") == 0){
    meta_abf->sample_freq=io_nexti(line);
  } else if (strcmp(token, "bin_width") == 0){
    if(!meta_abf->init){
      meta_abf->grid.bin_width=io_nextf(line);
    } else {
      printlog("!!!!! Cannot change n_bins after initialization, leaving with %d bins!\n", meta_abf->grid.n_bins);
    }
  } else if (strcmp(token, "target_site")==0){
    meta_abf->alchemCV.target_site=io_nextb(line);
  } else if (strcmp(token, "do_meta")==0){
    meta_abf->do_meta=io_nextb(line);
  } else if (strcmp(token, "do_abf")==0){
    meta_abf->do_abf=io_nextb(line);
  } else if (strcmp(token, "do_temper")==0){
    meta_abf->meta_options.do_temper=io_nextb(line);
  } else if (strcmp(token, "do_sample")==0){
    meta_abf->do_sample=io_nextb(line);
  } else if (strcmp(token, "do_restart")==0){
    meta_abf->do_restart=io_nextb(line);
  } else if (strcmp(token, "lambda_space")==0){
    meta_abf->alchemCV.lambda_space=io_nextb(line);
  } else if (strcmp(token, "abf_warmup")==0){
    meta_abf->abf_warmup=io_nexti(line);
  } else if (strcmp(token, "temper_factor")==0){
    meta_abf->meta_options.temper_factor=io_nextf(line);
    if(meta_abf->meta_options.temper_factor < 1.0 || fabs(meta_abf->meta_options.temper_factor-1)< 1e-4){
      printlog("Temper factor too small!\n");
      exit(1);
    }
  } else if (strcmp(token, "meta_w") == 0){
    meta_abf->meta_options.w=io_nextf(line);
  } else if (strcmp(token, "meta_std")==0){
    meta_abf->meta_options.stdev = io_nextf(line);
  } else if (strcmp(token, "write_restart_freq")==0){
    meta_abf->write_restart_freq=io_nexti(line);
  } else if (strcmp(token, "log_freq")==0){
    meta_abf->log_freq=io_nexti(line);
  } else {
    printlog("Didn't recognize option: %s\n", token);
    exit(1);
  }
};

// This only gets called the first time enhanced->initialize() gets called
void MetaAdaptiveBiasingForce::initialize(System* system){
  if(!init){
    printlog("Initializing Meta-ABF!\n");
    // CV Memory
    // Alchemical CV
    int site = alchemCV.target_site;
    if (site <= 0 || site >= system->msld->siteCount){
      printlog("Choose a valid site! %d is not a valid site!\n", alchemCV.target_site);
      exit(1);
    }
    if (system->msld->blocksPerSite[site] > 2){ // TODO: Don't exit if using L-LEUS and theta biasing
      printlog("Cannot do meta-abf on lambda space with Nsub=%d, only 2 subs works!\n", system->msld->blocksPerSite[site]);
      exit(1);
    }
    cudaMalloc(&alchemCV.x_d, sizeof(real));
    cudaMalloc(&alchemCV.y_d, sizeof(real));
    cudaMalloc(&tmpThetaF_d, system->msld->blockCount*sizeof(real));
    cudaMalloc(&tmpLmdF_d, system->msld->blockCount*sizeof(real));
    int blockStart = system->msld->siteBound[site];
    if(alchemCV.lambda_space){
      alchemCV.sysX_d = &system->state->lambda_fd[blockStart];
      alchemCV.sysG_d = &system->state->lambdaForce_d[blockStart];
      alchemCV.cr0 = 0; alchemCV.cr1 = 1;
      alchemCV.gScale0 = -1; alchemCV.gScale1 = 1;
      grid.lb = 0; grid.ub = 1;
      grid.lb_clamp = 0; grid.ub_clamp = 0; // ignored
      grid.bc = constrained_bc;
    } else {
      if (!system->msld->new_implicit){
        printlog("Cannot do theta ABF without pwise or l-leus constraint!");
        exit(1);
      }
      alchemCV.sysX_d = &system->state->theta_fd[blockStart];
      alchemCV.sysG_d = &tmpThetaF_d[blockStart];
      real fbw = 2*2*system->msld->well_width; // nsubs=2
      real excess = 5/sqrt(system->msld->well_k); // hist extends n=10 std beyond the flat bottom region
      // if using pwise constraint: T1-T0 -> unconstrained
      alchemCV.cr0 = -1; alchemCV.cr1 = 1;
      alchemCV.gScale0 = 0; alchemCV.gScale1 = 1;
      grid.lb = -fbw - excess; grid.ub = fbw + excess;
      //grid.lb = floor(grid.lb/grid.bin_width)*grid.bin_width; //
      //grid.ub = ceil(grid.ub/grid.bin_width)*grid.bin_width;
      grid.lb_clamp = -1; grid.ub_clamp = 1;
      grid.bc = unconstrained_bc;
      // if using lleus constraint: T1 -> periodic
      // tmp storage for theta forces
    }
    // Grid Memory
    real range = grid.ub - grid.lb;
    grid.n_bins = round(range/(grid.bin_width));
    grid.n_bins += grid.bc != periodic_bc ? 1 : 0;
    printlog("bin_width: %f, n_bins: %d\n", grid.bin_width, grid.n_bins);
    int n_bins = grid.n_bins;
    grid.counts = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&grid.counts_d, n_bins*sizeof(real));
    cudaMemcpy(grid.counts_d, grid.counts, n_bins*sizeof(real), cudaMemcpyDefault);
    grid.y_m2 = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&grid.y_m2_d, n_bins*sizeof(real));
    cudaMemcpy(grid.y_m2_d, grid.y_m2, n_bins*sizeof(real), cudaMemcpyDefault);
    grid.y_avg = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&grid.y_avg_d, n_bins*sizeof(real));
    cudaMemcpy(grid.y_avg_d, grid.y_avg, n_bins*sizeof(real), cudaMemcpyDefault);
    grid.meta_weights = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&grid.meta_weights_d, n_bins*sizeof(real));
    cudaMemcpy(grid.meta_weights_d, grid.meta_weights, n_bins*sizeof(real), cudaMemcpyDefault);

    // Other Memory
    y_std = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&abfU_d, sizeof(real));
    cudaMalloc(&abfG_d, sizeof(real));
    cudaMalloc(&metaU_d, sizeof(real));
    cudaMalloc(&metaG_d, sizeof(real));

    // Read restart files
    if (do_restart) restart(system);
    if (system->msld->blocksPerSite[alchemCV.target_site] != 2){
      printlog("Meta ABF cannot be used on a site with more than 2 substituents!\n");
      exit(1);
    }
    // Update with current options
    real bin_width = grid.bin_width;
    half_search_bins = ceil(search_std*meta_options.stdev/bin_width);
    if (do_meta && half_search_bins >= grid.n_bins){
      printlog("Requested search longer then 2 grid widths. Please reduce meta_std or increase range!\n");
      exit(1);
    }
  } else { // Re-init system pointers
    int blockStart = system->msld->siteBound[alchemCV.target_site];
    if(alchemCV.lambda_space){
      alchemCV.sysX_d = &system->state->lambda_fd[blockStart];
      alchemCV.sysG_d = &system->state->lambdaForce_d[blockStart];
    } else {
      alchemCV.sysX_d = &system->state->theta_fd[blockStart];
      alchemCV.sysG_d = &tmpThetaF_d[blockStart];
    }
  }
  init=true;
};

void MetaAdaptiveBiasingForce::step_reset(System* system){
  cudaMemsetAsync(abfU_d, 0, sizeof(real), system->run->enhancedStream);
  cudaMemsetAsync(abfG_d, 0, sizeof(real), system->run->enhancedStream);
  cudaMemsetAsync(metaU_d, 0, sizeof(real), system->run->enhancedStream);
  cudaMemsetAsync(metaG_d, 0, sizeof(real), system->run->enhancedStream);
  cudaMemcpyAsync(tmpLmdF_d, system->state->lambdaForce_d, system->msld->blockCount*sizeof(real), cudaMemcpyDefault, system->run->enhancedStream);
  cudaMemsetAsync(tmpThetaF_d, 0, system->msld->blockCount*sizeof(real), system->run->enhancedStream);
}

void __global__ compute_alchemCV_kernel(AlchemicalCV_MABF cv){
  cv.x_d[0] = cv.cr0*cv.sysX_d[0] + cv.cr1*cv.sysX_d[1];
  cv.y_d[0] = cv.gScale0*cv.sysG_d[0] + cv.gScale1*cv.sysG_d[1];
}

void MetaAdaptiveBiasingForce::compute_CV(System* system){
  Run* r = system->run;
  // if (alchemCV){}
  if (!alchemCV.lambda_space){
    system->msld->calc_thetaForce_from_lambdaForce(r->enhancedStream,system,tmpLmdF_d,tmpThetaF_d);
    int blockStart = system->msld->siteBound[alchemCV.target_site];
    alchemCV.sysG_d = &tmpThetaF_d[blockStart]; // TODO: fix this, this is jank
    compute_alchemCV_kernel<<<1, 1, 0, r->enhancedStream>>>(alchemCV);
    alchemCV.sysG_d = &system->state->thetaForce_d[blockStart]; // TODO: fix this, this is jank
  } else {
    compute_alchemCV_kernel<<<1, 1, 0, r->enhancedStream>>>(alchemCV);
  }
  // if (spatialCV){
  //   In the future this might compute other CV (COM/dihe/rij) for eABF
  // }
}

// TODO: Handle periodicity and out of bounds samples
int __host__ __device__ get_histogram_index(real bin_w, real x, real l, real u){
  if (x < l || x > u) { return 0; } // o.b. samples map to bin 0
  return (int)round((x-l)/bin_w);
}

void __global__ getforce_abf_kernel(
  real* cv, MetaABFGrid gd, int abf_warmup, 
  real* abfG, real* abf_bias, real_e* energy){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  int n_bins = gd.n_bins;
  real bin_width = gd.bin_width;
  real* y_avg = gd.y_avg_d;
  real* counts = gd.counts_d;

  if (i < n_bins){
    real x = cv[0];
    real y_curr = y_avg[i];
    y_curr *= counts[i] < abf_warmup && abf_warmup > 0 ? counts[i]/abf_warmup : 1;
    if (i >= 1){ // Each thread computes integral from previous bin to this bin
      real y_prev = y_avg[i-1];
      y_prev *= counts[i-1] < abf_warmup && abf_warmup > 0 ? counts[i-1]/abf_warmup : 1;
      lEnergy = -bin_width*(y_curr+y_prev)/2.0; // trapezoid up to lambda
      real c_prev = gd.lb+(i-1)*bin_width;
      if(x >= c_prev && x < gd.lb+i*bin_width){ // L is between last bin center and current bin center
        real interp = (x-c_prev)/bin_width;
        real y_up = (1.0-interp)*y_prev + interp*y_curr;
        real width = x-c_prev;
        lEnergy = -width*(y_prev+y_up)/2.0;
        atomicAdd(abfG, -y_up); 
      } else if(x <= gd.lb+(i-1)*bin_width){ // L is less than lower bin center
        lEnergy = 0;
      }
    }
  }
  real_sum_reduce(lEnergy,sEnergy,abf_bias);
  if (energy){
    // ABF adds -'ve F(L)
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
};

void __global__ getforce_meta_kernel(
  int n_search, real* cv, 
  MetaABFGrid mabf_grid, MetaOptions_MABF meta_options,
  real* metaG, real* current_meta_bias, real_e* energy){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  int n_bins = mabf_grid.n_bins;
  real meta_std = meta_options.stdev;
  real* meta_weights = mabf_grid.meta_weights_d;
  real bin_width = mabf_grid.bin_width;

  if (i < n_search){
    real L = cv[0];
    real dClamp_dx = 1;
    real mirror_factor = 1;
    if(mabf_grid.bc == unconstrained_bc){
      if (L < mabf_grid.lb_clamp){ L = mabf_grid.lb_clamp; dClamp_dx = 0; }
      else if (L > mabf_grid.ub_clamp){ L = mabf_grid.ub_clamp; dClamp_dx = 0; }
    }

    int half_search = (n_search-1)/2;
    int bin = get_histogram_index(mabf_grid.bin_width, L, mabf_grid.lb, mabf_grid.ub);
    int my_bin = bin + (i-half_search);
    real my_bin_center = mabf_grid.lb + my_bin*bin_width; // don't update this with mirror
    if (mabf_grid.bc == constrained_bc){
      if (my_bin < 0){ // lower mirror
        my_bin = -my_bin;
      } // mirror doesn't handle multiple reflections
      if (my_bin >= n_bins){ // upper mirror
        int overshoot = my_bin - (n_bins-1);
        my_bin = (n_bins-1) - overshoot; // max_id - overshoot
      }
      mirror_factor = my_bin == 0 || my_bin == n_bins-1 ? 2 : 1;
    }
    if (my_bin >= 0 && my_bin < n_bins){
      real dist = (L-my_bin_center)/meta_std;
      real gauss = exp(-.5*dist*dist);
      // first and last bins should have their weights doubled from contribution on other side of the mirror
      lEnergy = mirror_factor*meta_weights[my_bin]*gauss;
      real dUdL = -dist/meta_std*lEnergy*dClamp_dx; // need variance in denom
      atomicAdd(metaG, dUdL); 
    }
  }
  real_sum_reduce(lEnergy,sEnergy,current_meta_bias);
  if (energy){
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
};

// This function does not modify alchemCV x or y values
void getforce_meta_abf(System* system, int step, bool calcEnergy){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  State* state = system->state;
  Run* run = system->run;
  int shMem=BLBO*sizeof(real)/32;
  real_e *pEnergy=NULL;
  if (calcEnergy) {
    pEnergy=state->energy_d+eeenhanced;
  }

  if (m_abf->do_abf) {
    getforce_abf_kernel<<<(m_abf->grid.n_bins+BLBO-1)/BLBO,BLBO,shMem,run->enhancedStream>>>(
      m_abf->alchemCV.x_d, m_abf->grid, m_abf->abf_warmup,  // In
      m_abf->abfG_d, m_abf->abfU_d, pEnergy); // Out
  }
  if (m_abf->do_meta) {
    int bins = 2*m_abf->half_search_bins + 1;
    getforce_meta_kernel<<<(bins+BLBO-1)/BLBO,BLBO,shMem,run->enhancedStream>>>(
      bins, m_abf->alchemCV.x_d, 
      m_abf->grid, m_abf->meta_options,
      m_abf->metaG_d, m_abf->metaU_d, pEnergy);
  }
  gpuCheck(cudaPeekAtLastError());
};

void __global__ chainRule_alchemCV_kernel(AlchemicalCV_MABF cv, real* abfG, real* metaG){
  real totalG = abfG[0] + metaG[0];
  atomicAdd(&cv.sysG_d[0], cv.cr0*totalG);
  atomicAdd(&cv.sysG_d[1], cv.cr1*totalG);
}

void MetaAdaptiveBiasingForce::apply_chain_rule(System* system){
  chainRule_alchemCV_kernel<<<1, 1, 0, system->run->enhancedStream>>>(alchemCV, abfG_d, metaG_d);
}

void __global__ add_sample_abf(AlchemicalCV_MABF cv,MetaABFGrid gd){
    // stable online average and std - Welford's
    int bin = get_histogram_index(gd.bin_width, cv.x_d[0], gd.lb, gd.ub);
    real prev_delta = cv.y_d[0] - gd.y_avg_d[bin];
    gd.counts_d[bin] += 1;
    gd.y_avg_d[bin] += prev_delta/gd.counts_d[bin];
    gd.y_m2_d[bin] += prev_delta*(cv.y_d[0]-gd.y_avg_d[bin]);
}

void __global__ add_sample_meta(
  real kT, AlchemicalCV_MABF cv, real* metaU,
  MetaABFGrid gd, MetaOptions_MABF m_opt){
    int bin = get_histogram_index(gd.bin_width, cv.x_d[0], gd.lb, gd.ub);
    real factor = m_opt.do_temper ? exp(-metaU[0]/((m_opt.temper_factor-1.0)*kT)): 1.0;
    gd.meta_weights_d[bin] += m_opt.w*factor;
}

// gefforce doesn't touch x or y values, so this is safe to call after getforce
void sample_meta_abf(System* system, int step){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  Run* run = system->run;

  if(m_abf->do_sample && step % m_abf->sample_freq == 0){
    if(m_abf->do_abf){
      add_sample_abf<<<1, 1, 0, run->enhancedStream>>>(m_abf->alchemCV,m_abf->grid);
    }
    if(m_abf->do_meta){
      add_sample_meta<<<1, 1, 0, run->enhancedStream>>>(kB*run->T, m_abf->alchemCV, m_abf->metaU_d, m_abf->grid, m_abf->meta_options);
    }
  }
  gpuCheck(cudaPeekAtLastError());
};

void recv_meta_abf(System* system){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;

  MetaABFGrid* grid = &m_abf->grid;
  int nb = grid->n_bins;
  cudaMemcpy(grid->counts, grid->counts_d, nb*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(grid->y_m2, grid->y_m2_d, nb*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(grid->y_avg, grid->y_avg_d, nb*sizeof(real), cudaMemcpyDefault);
  for(int i = 0; i < nb; i++){
    m_abf->y_std[i] = grid->counts[i]>1e-4 ? sqrt(grid->y_m2[i]/grid->counts[i]) : 0;
  }
  cudaMemcpy(grid->meta_weights, grid->meta_weights_d, nb*sizeof(real), cudaMemcpyDefault);

  cudaMemcpy(&m_abf->alchemCV.x, m_abf->alchemCV.x_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&m_abf->alchemCV.y, m_abf->alchemCV.y_d, sizeof(real), cudaMemcpyDefault);

  cudaMemcpy(&m_abf->abfU, m_abf->abfU_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&m_abf->abfG, m_abf->abfG_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&m_abf->metaU, m_abf->metaU_d, sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&m_abf->metaG, m_abf->metaG_d, sizeof(real), cudaMemcpyDefault);
};

void print_real_array(real* arr, int len){
  if(arr){
    printlog("[ ");
    for(int i = 0; i < len; i++){
      if(i == len-1){
        printlog("%7.2f ", arr[i]);
      } else {
        printlog("%7.2f, ", arr[i]);
      }
    }
    printlog("]");
  }
}

void log_meta_abf(System* system, int step){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  State* state = system->state;
  if(m_abf->log_freq != 0 && step % m_abf->log_freq == 0){
    recv_meta_abf(system);
    state->recv_energy();

    if(!m_abf->do_sample){ printlog("NOT ADDING SAMPLES!!!!\n"); }
    printlog("Step %d = %.3f ns:\n", step, step*system->run->dt/PICOSECOND*(1.0/1000.0)); // nstep*(dt in ps)*(ps_to_ns)
    printlog("x: %.2f (%.4e), y: %.2f, bin: %d\n", 
      m_abf->alchemCV.x, m_abf->alchemCV.x, m_abf->alchemCV.y, 
      get_histogram_index(m_abf->grid.bin_width, m_abf->alchemCV.x, m_abf->grid.lb, m_abf->grid.ub));
    printlog("x bounds: [%.4f, %.4f], bin_width: %.4f, n_bins: %d\n", m_abf->grid.lb, m_abf->grid.ub, m_abf->grid.bin_width, m_abf->grid.n_bins);
    printlog("U_abf: %.2f: U_meta: %.2f, U_enhanced: %.2f, ", m_abf->abfU, m_abf->metaU, system->state->energy[eeenhanced]);
    printlog("dUdx_abf: %.2f: dUdx_meta: %.2f\n", m_abf->abfG, m_abf->metaG);
    if (m_abf->do_meta){
      if (m_abf->meta_options.do_temper){
        real kT = kB*system->run->T;
        real factor = exp(-m_abf->metaU/(kT*(m_abf->meta_options.temper_factor-1.0)));
        printlog("Meta Temper Factor: %5.2f,  Meta Decay Factor: %5.2f\n", m_abf->meta_options.temper_factor, factor);
      }
      printlog("Meta Weights: ");
      print_real_array(m_abf->grid.meta_weights, m_abf->grid.n_bins);
      printlog("\n");
    }
    if (m_abf->do_abf){
      printlog("counts(x): ");
      print_real_array(m_abf->grid.counts, m_abf->grid.n_bins);
      printlog("\n");
      printlog("<y>(x): ");
      print_real_array(m_abf->grid.y_avg, m_abf->grid.n_bins);
      printlog("\n");
      printlog("std[y](x): ");
      print_real_array(m_abf->y_std, m_abf->grid.n_bins);
      printlog("\n");
      real dG_TI = 0;
      real bin_width = m_abf->grid.bin_width;
      for(int i = 0; i < m_abf->grid.n_bins-1; i++){
        dG_TI += bin_width*(m_abf->grid.y_avg[i] + m_abf->grid.y_avg[i+1])/2.0;
      }
      printlog("dG_{%.3f->%.3f,TI}: %f\n", m_abf->grid.lb, m_abf->grid.ub, dG_TI);
      // index zero is l0=1, index n_bins-1 is l0=0
    }
    printlog("\n");
  }
};

// As usual, claude wrote the file write and reads
void write_meta_abf(std::string dir_name, System* system, int step){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  if (step % m_abf->write_restart_freq == 0){
    std::string filename = dir_name + "/" +  m_abf->fnm_meta_abf;
    std::string tmp_filename = filename + "_tmp";
    recv_meta_abf(system);
    FILE* fp = fopen(tmp_filename.c_str(), "w");
    if(!fp){
      printlog("Error: could not open %s for writing!\n", filename.c_str());
      printlog("Exiting...\n"); exit(1);
    }
    /*
      File structure:
      target_site #
      n_bins #
      counts # # # # ...
      dUdL_m2 # # # # ...
      dUdL2_sum # # # # ...
      meta_weights # # # # ...
    */
    fprintf(fp, "target_site %d\n", m_abf->alchemCV.target_site);
    fprintf(fp, "bc %d\n",          m_abf->grid.bc);
    fprintf(fp, "lb %f\n",          m_abf->grid.lb);
    fprintf(fp, "ub %f\n",          m_abf->grid.ub);
    fprintf(fp, "lb_clamp %f\n",    m_abf->grid.lb_clamp);
    fprintf(fp, "ub_clamp %f\n",    m_abf->grid.ub_clamp);
    fprintf(fp, "bin_width %f\n",   m_abf->grid.bin_width);

    int nb = m_abf->grid.n_bins;
    fprintf(fp, "counts");
    for(int i = 0; i < nb; i++){
      fprintf(fp, " %f", m_abf->grid.counts[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "dUdL_m2");
    for(int i = 0; i < nb; i++){
      fprintf(fp, " %f", m_abf->grid.y_m2[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "dUdL_avg");
    for(int i = 0; i < nb; i++){
      fprintf(fp, " %f", m_abf->grid.y_avg[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "meta_weights");
    for(int i = 0; i < nb; i++){
      fprintf(fp, " %f", m_abf->grid.meta_weights[i]);
    }
    fprintf(fp, "\n");
    fflush(fp);
    fclose(fp);
    if(rename(tmp_filename.c_str(), filename.c_str()) != 0){              
      printlog("Error: could not move %s to %s!\n", tmp_filename.c_str(), filename.c_str());  
      printlog("Exiting...\n"); exit(1);                                
    }
  }
};

void MetaAdaptiveBiasingForce::restart(System* system){
  std::string fnm = system->enhanced->output_dir + "/" + fnm_meta_abf;
  FILE* fp = fopen(fnm.c_str(), "r");
  if(!fp){
    printlog("MetaABF: output dir (%s)\n", system->enhanced->output_dir.c_str());
    printlog("MetaABF: no restart file found (%s), starting fresh.\n", fnm.c_str());
    return;
  }
  printlog("MetaABF: reading restart file %s\n", fnm.c_str());

  // Width-limited "%s" so the token read can never overflow `token`.
  char token[MAXLENGTHSTRING];
  char tokfmt[16];
  snprintf(tokfmt, sizeof(tokfmt), "%%%ds", MAXLENGTHSTRING - 1);

  // Reads n_bins values into arr, into a double temp first so it works
  // whether `real` is float or double. Fails loudly on a short read.
  auto read_array = [&](real* arr, const char* name){
    for(int i = 0; i < grid.n_bins; i++){
      double v;
      if(fscanf(fp, " %lf", &v) != 1){
        printlog("Error: failed reading '%s' value %d of %d.\n", name, i, grid.n_bins);
        fclose(fp); exit(1);
      }
      arr[i] = (real)v;
    }
    printlog("Read '%s' from file: ", name);
    print_real_array(arr, grid.n_bins);
    printlog("\n");
  };

  while(fscanf(fp, tokfmt, token) == 1){
    if(strcmp(token, "target_site") == 0){
      int read_target_site;
      if(fscanf(fp, " %d", &read_target_site) != 1){
        printlog("Error: failed reading target_site value.\n"); fclose(fp); exit(1);
      }
      if(read_target_site != alchemCV.target_site){
        printlog("Warning: restart target_site (%d) differs from current target_site (%d). Using current.\n",
               read_target_site, alchemCV.target_site);
      }
    } else if(strcmp(token, "bc") == 0){
      int read_bc;
      if(fscanf(fp, " %d", &read_bc) != 1){ printlog("Error: failed reading bin_width value.\n"); fclose(fp); exit(1);}
      if(read_bc != grid.bc){
        printlog("Error: restart bc (%d) does not match current bc (%d)!\n", read_bc, grid.bc); fclose(fp); exit(1);
      }
    } else if(strcmp(token, "bin_width") == 0){
      double read_bin_width;
      if(fscanf(fp, " %lf", &read_bin_width) != 1){ printlog("Error: failed reading bin_width value.\n"); fclose(fp); exit(1); }
      if(fabs(read_bin_width-grid.bin_width) > 1e-4){
        printlog("Error: restart n_bins (%f) does not match current n_bins (%f)!\n", read_bin_width, grid.bin_width); fclose(fp); exit(1);
      }
      // Bin count should be identical if bc and bin_width are the same
    } else if(strcmp(token, "lb") == 0){
      double read_lb;
      if(fscanf(fp, " %lf", &read_lb) != 1){ printlog("Error: failed reading lb value.\n"); fclose(fp); exit(1);}
      if(fabs(read_lb-grid.lb) > 1e-4){
        printlog("Error: restart lb (%f) does not match current lb (%f)!\n", (float)read_lb, grid.lb); fclose(fp); exit(1);
      }
    } else if(strcmp(token, "ub") == 0){
      double read_ub;
      if(fscanf(fp, " %lf", &read_ub) != 1){ printlog("Error: failed reading ub value.\n"); fclose(fp); exit(1); }
      if(fabs(read_ub-grid.ub) > 1e-4){
        printlog("Error: restart ub (%f) does not match current ub (%f)!\n", (float)read_ub, grid.ub); fclose(fp); exit(1);
      }
    } else if(strcmp(token, "lb_clamp") == 0){
      double read_lb_clamp;
      if(fscanf(fp, " %lf", &read_lb_clamp) != 1){ printlog("Error: failed reading lb_clamp value.\n"); fclose(fp); exit(1); }
      if(fabs(read_lb_clamp-grid.lb_clamp) > 1e-4){
        printlog("Error: restart lb_clamp (%f) does not match current lb_clamp (%f)!\n", (float)read_lb_clamp, grid.lb_clamp); fclose(fp); exit(1);
      }
    } else if(strcmp(token, "ub_clamp") == 0){
      double read_ub_clamp;
      if(fscanf(fp, " %lf", &read_ub_clamp) != 1){ printlog("Error: failed reading ub_clamp value.\n"); fclose(fp); exit(1); }
      if(fabs(read_ub_clamp-grid.ub_clamp) > 1e-4){
        printlog("Error: restart ub_clamp (%f) does not match current ub_clamp (%f)!\n", (float)read_ub_clamp, grid.ub_clamp); fclose(fp); exit(1);
      }
    } else if(strcmp(token, "counts") == 0){
      read_array(grid.counts, "counts");
    } else if(strcmp(token, "dUdL_m2") == 0){
      read_array(grid.y_m2, "dUdL_m2");
    } else if(strcmp(token, "dUdL_avg") == 0){
      read_array(grid.y_avg, "dUdL_avg");
    } else if(strcmp(token, "meta_weights") == 0){
      read_array(grid.meta_weights, "meta_weights");
    }
    // Unrecognized tokens are consumed and ignored; because fscanf("%s")
    // always advances, stray tokens/values can't cause an infinite loop.
  }
  fclose(fp);

  for(int i = 0; i < grid.n_bins; i++){
    y_std[i] = fabs(grid.counts[i]) > 1e-3 ? sqrt(grid.y_m2[i]/grid.counts[i]) : 0;
  }

  // Update GPU memory with values from file
  int nb = grid.n_bins;
  cudaMemcpy(grid.counts_d, grid.counts, nb*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(grid.y_m2_d, grid.y_m2, nb*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(grid.y_avg_d, grid.y_avg, nb*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(grid.meta_weights_d, grid.meta_weights, nb*sizeof(real), cudaMemcpyDefault);
  printlog("MetaABF: restart complete.\n");
};