//
// Created by matthew-speranza on 12/3/24.
//

#ifndef HISTOGRAMESTIMATOR_H
#define HISTOGRAMESTIMATOR_H

#include "main/defines.h"
#include "system/system.h"
// Forward declarations
class System;

enum HistogramEstimatorType {
  ABF, // Adaptive Biasing Force with linear interpolation - subtract <dU/dL> from each lambda force
  ABF_SPLINE, // ABF with second order spline interpolation (only with OST)
  CZAR, // Corrected Z-averaged restraint - probabilitic ABF
  OPES // On-the-fly Probability Enhanced Sampling - probabilistic metadynamics
};

/**
* Estimates 1D profile of lambda variables constrained [0,1]
*/
class HistogramEstimator {
  public:
    // Basic info
    int lambda_id; // which lambda this estimator is for
    int depth; // number of histograms
    int accumulate_into; // which histogram index to accumulate into
    real **histogram_counts; // number of occurences in each bin
    real *bin_edges; // edges of the bins 0-1
    real *bin_widths; // width of each bin, used for normalization & integration weighting
    int first_half_bins; // number of bins assigned to the first half of the lambda range
    int second_half_bins; // number of bins assigned to the other half of the lambda range
    int total_bins; // total number of bins

    // <dU/dL> estimations
    real **dUdL; // ensemble average dU/dL in each bin
    real **dUdL_d; // device pointer
    real **d2UdL2; // ensemble average d2U/dL2 in each bin
    real **d2UdL2_d;
    real offset; // stores largest value of -beta*(U(x,l)-U_bias(x,l)) for numerics
    real **weights; // sum of boltzmann weights = exp(-beta*(U(x,l)-U_bias(x,l)))
    real **weighted_dUdL; // sum of boltzmann weighted dU/dL in each bin
    real **weighted_d2UdL2; // optimize lambda profile with 2nd derivative information
    // variance = E[X^2] - E[X]^2
    real **dUdL2; // ensemble average (dU/dL)^2 in each bin
    real **variance; // variance of dU/dL in each bin
    real **variance_d;
    real **weighted_dUdL2; // sum of weighted dUdL squares

    HistogramEstimator();
    ~HistogramEstimator();

    void initialize(int num_lambdas, int lambda_id, int first_half_bins, int second_half_bins);
    void destroy();
    void add_sample(real lambda, real dUdL, real d2UdL2, real potEnergy, real temp);
    void send_averages();
    void getforce_histogram(System* system, bool calcEnergy);

  private:
    int get_bin_index(real lambda);
    static void assign_edges(int num_lambdas, int first, int second, real *edges, real* gaps);
};



#endif //HISTOGRAMESTIMATOR_H
