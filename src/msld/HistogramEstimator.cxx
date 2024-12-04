//
// Created by matthew-speranza on 12/3/24.
//

#include "HistogramEstimator.h"

#include <cassert>

#include "system/state.h"

void HistogramEstimator::initialize(int lambda_id, int num_lambdas, int first_half_bins, int second_half_bins){
  this->lambda_id = lambda_id;
  this->first_half_bins = first_half_bins;
  this->second_half_bins = second_half_bins;
  this->total_bins = first_half_bins + second_half_bins;
  this->bin_edges = static_cast<real *>(calloc(this->total_bins + 1, sizeof(real)));
  this->bin_widths = static_cast<real *>(calloc(this->total_bins, sizeof(real)));
  this->assign_edges(num_lambdas, first_half_bins, second_half_bins, bin_edges, bin_widths);
  // Depth allows for biases to be sampled without affecting the potential
  this->depth = 1;
  this->accumulate_into = 0;
  this->histogram_counts = static_cast<real **>(calloc(depth, sizeof(real **)));
  this->weights = static_cast<real **>(calloc(depth, sizeof(real **)));
  this->weighted_dUdL = static_cast<real **>(calloc(depth, sizeof(real **)));
  this->weighted_d2UdL2 = static_cast<real **>(calloc(depth, sizeof(real **)));
  this->weighted_dUdL2 = static_cast<real **>(calloc(depth, sizeof(real **)));
  for (int i = 0; i < depth; i++){
    this->histogram_counts[i] = static_cast<real *>(calloc(this->total_bins, sizeof(real)));
    this->weights[i] = static_cast<real *>(calloc(this->total_bins, sizeof(real)));
    this->weighted_dUdL[i] = static_cast<real *>(calloc(this->total_bins, sizeof(real)));
    this->weighted_d2UdL2[i] = static_cast<real *>(calloc(this->total_bins, sizeof(real)));
    this->weighted_dUdL2[i] = static_cast<real *>(calloc(this->total_bins, sizeof(real)));
  }
}

void HistogramEstimator::destroy(){
  free(this->bin_edges);
  for (int i = 0; i < 1; i++){
    free(this->histogram_counts[i]);
    free(this->weights[i]);
    free(this->weighted_dUdL[i]);
    free(this->weighted_d2UdL2[i]);
    free(this->weighted_dUdL2[i]);
  }
  free(this->histogram_counts);
  free(this->weights);
  free(this->weighted_dUdL);
  free(this->weighted_d2UdL2);
  free(this->weighted_dUdL2);
}

real beta_cdf_inverse(const real y, const int b){
  return 1 - static_cast<real>(pow(1 - y, 1 / b));
}

/*
 * Uniform multi-dim distribution of samples creates a marginal beta distribution. This takes the form of:
 *
 *  p(x) = (a+b)!/(a!b!) * x^(a-1) * (1-x)^(b-1)
 *
 * In this case, a = 1 and b = nLamdas - 1. We want each bin in the lower half to have the
 * same probability density. To do this we integrate the beta distribution from 0 to x to get
 * the CDF and take evenly spaced samples of that function's inverse. This can be solved analytically
 * to give:
 *
 * x = 1 - (1-y)^(1/b)
 *
 * The second half of the bins are assigned linearly from the first half to 1.
*/
void HistogramEstimator::assign_edges(const int num_lambdas, const int first, const int second, real *edges, real *gaps){
  int total = first + second;
  for (int i = 0; i < first; i++){
    edges[i] = beta_cdf_inverse(static_cast<real>(i) / static_cast<real>(first) / 2, num_lambdas-1);
    if (i > 0){
      gaps[i-1] = edges[i] - edges[i-1];
    }
  }
  real rest = 1.0 - edges[first-1];
  real gap = rest / second;
  for (int i = first; i < total; i++){
    edges[i] = (i-first+1)*gap + edges[first];
    gaps[i] = edges[i] - edges[i-1];
  }
}

// This is unlikely to be worth speeding up w/ binary search
int HistogramEstimator::get_bin_index(real lambda){
  for (int i = 0; i < this->total_bins; i++){
    if (lambda < this->bin_edges[i+1]){ // i+1 always in range for edges
      return i;
    }
  }
  // Out of range?
  return this->total_bins-1;
}


void HistogramEstimator::add_sample(real lambda, real dUdL, real d2UdL2, real potEnergy, real temp){
  int bin = this->get_bin_index(lambda);
  this->histogram_counts[accumulate_into][bin] += 1;
  real beta = 1 / (kB * temp);
  if (-beta*potEnergy > offset) {
    real new_offset = -beta*potEnergy;
    this->weights[accumulate_into][bin] = exp(log(weights[accumulate_into][bin])-offset+new_offset);
    this->weighted_dUdL[accumulate_into][bin] = exp(log(weighted_dUdL[accumulate_into][bin])-offset+new_offset);
    this->weighted_d2UdL2[accumulate_into][bin] = exp(log(weighted_d2UdL2[accumulate_into][bin])-offset+new_offset);
    this->weighted_dUdL2[accumulate_into][bin] = exp(log(weighted_dUdL2[accumulate_into][bin])-offset+new_offset);
    offset = new_offset;
  }
  this->weights[accumulate_into][bin] += exp(-beta * potEnergy - offset);
  this->weighted_dUdL[accumulate_into][bin] += dUdL * exp(-beta * potEnergy - offset);

  this->weighted_d2UdL2[accumulate_into][bin] += d2UdL2 * exp(-beta * potEnergy - offset);
  this->weighted_dUdL2[accumulate_into][bin] += dUdL * dUdL * exp(-beta * potEnergy - offset);
}

// Need to calc energy when adding sample
void HistogramEstimator::getforce_histogram(System* system, bool calcEnergy){
  system->state->recv_lambda();
  system->state->recv_lambda_force(false);
  real lambda = system->state->lambda[this->lambda_id];
  // This value is in range for bins, +1 is in range for edges, but + 2 may be out of range
  int bin = this->get_bin_index(lambda);
  // Lerp implementation of ABF - values are defined on the bin's lower edges to avoid edge cases
  real low_edge = this->bin_edges[bin];
  real high_edge = this->bin_edges[bin+1];
  real dUdL_low = this->weighted_dUdL[accumulate_into][bin] / this->weights[accumulate_into][bin];
  int id = bin+1 >= total_bins ? bin : bin + 1; // last edge has same value as last bin
  real dUdL_high = this->weighted_dUdL[accumulate_into][id] / this->weights[accumulate_into][id];
  real interp = (lambda - low_edge) / (high_edge - low_edge);
  real dUdL = (1-interp) * dUdL_low + interp * dUdL_high;
  system->state->lambdaForce[this->lambda_id] -= dUdL;
  system->state->recv_lambda_force(true); // send force
}