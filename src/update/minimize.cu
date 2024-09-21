#include <cassert>
#include <cuda_runtime.h>
#include <math.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <stdlib.h>

#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "holonomic/holonomic.h"
#include "io/io.h"
#include "main/real3.h"

__global__ void copy_float_onto_double(float* in, double* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) {
    out[i] = (double)in[i];
  }
}

__global__ void copy_double_onto_float(double* in, float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) {
    out[i] = (float)in[i];
  }
}

__global__ void update_lbfgs_position(double* x, float* search, float stepSize, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) {
    x[i] += stepSize * search[i];
  }
}

void lbfgsForward(float* q, float* rho, float* alpha, float* s, float* y, int mIndex, int N, int nLam,cublasHandle_t handle) {
  // Launch kernels for alpha and update q
  // alpha = s . q
  cublasSdot_v2(handle, 3*N+nLam, s+mIndex*(3*N+nLam), 1, q, 1, alpha+mIndex);
  // q = q - rho * alpha * y
  alpha[mIndex] *= -rho[mIndex];
  cublasSaxpy_v2(handle, 3*N, alpha+mIndex, y+mIndex*(3*N+nLam), 1, q, 1); // writes over q
}

void lbfgsBackward(float* q, float* rho, float* alpha, float* s, float* y, int mIndex, int N, int nLam, cublasHandle_t handle) {
  // Launch kernels for beta and update q
  // beta = y . q
  float beta = 0.0;
  cublasSdot_v2(handle, 3*N+nLam, y+mIndex*(3*N+nLam), 1, q, 1, &beta);
  // q = q + (alpha - rho * beta) * s
  beta = alpha[mIndex] - rho[mIndex] * beta;
  cublasSaxpy_v2(handle, 3*N+nLam, &beta, s+mIndex*(3*N+nLam), 1, q, 1); // writes over q
}

void lbfgsUpdate(LeapState* ls, float* rho, float* gamma, const float* previous_p, float* s, const float* previous_g,
  float* y, const int mID, const int N, cublasHandle_t handle, System* system) {
  // Launch kernels for rho, alpha, s, y calculations after line search_d
  float nOne = -1.0;
  // s = x - prev_x
  copy_double_onto_float<<<(3*N+system->state->lambdaCount+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32,system->run->updateStream>>>
    (ls->x, s + mID*(3*N+system->state->lambdaCount), 3*N+system->state->lambdaCount);
  cublasSaxpy_v2(handle, N, &nOne, previous_p, 1, s + mID*(3*N+system->state->lambdaCount), 1);
  // y = f - prev_f
  cudaMemcpy(y + mID*N, ls->f, 3*N*sizeof(float), cudaMemcpyDeviceToDevice);
  cublasSaxpy_v2(handle, N, &nOne, previous_g, 1, y + mID*(3*N+system->state->lambdaCount), 1);
  system->state->recv_lbfgs();
  // These are computed for the next iteration
  // gamma = abs(s . y / y . y)
  float numerator, denominator;
  cublasSdot_v2(handle, N, s+mID*N, 1, y+mID*(3*N+system->state->lambdaCount), 1, &numerator);
  cublasSdot_v2(handle, N, y+mID*N, 1, y+mID*(3*N+system->state->lambdaCount), 1, &denominator);
  system->state->recv_lbfgs();
  if(numerator == 0 || denominator == 0) { // isinf() from c++11, cuda 10.0 supports c++14
    fprintf(stdout, "Skipped update of rho and gamma in L-BFGS to recover.\n");
    return; // don't update gamma or rho if it will break algo
  }
  *gamma = abs(numerator / denominator);
  // rho = 1 / (y . s)
  rho[mID] = 1 / numerator;
}

/**
 * 1. Compute new L-BFGS step direction
 *   Pseudocode from wikipedia:
 *   q = g.i // search direction to be updated
 *   for j = i-1 to i-m:
 *     <<<forward kernel>>>
 *     alpha.j = rho.j * s.j.T * q // dot product
 *     q = q - alpha.j * y.j // vector scale & subtraction
 *   gamma.i = s.i-1.T * y.i-1 / y.i-1.T * y.i-1 // dot products in numerator and denominator
 *   q = gamma.i * q
 *   for j = i-m to i-1:
 *     <<<backward kernel>>>
 *     beta = rho.j * y.j.T * q // dot product
 *     q = q + (alpha.j - beta) * s.j // vector scale & addition
 *   q = -q  // negate applied above instead of here most likely
 *   <<<update kernel>>>
 *   gamma = s.i.T * y.i / y.i.T * y.i
 *   rho.j = 1 / (y.j.T * s.j)
 */
void lbfgsDirection(System* system, int step) {
  cudaMemcpy(system->state->search_d, system->state->leapState->f, (3*system->state->atomCount+system->state->lambdaCount)*sizeof(float),
         cudaMemcpyDeviceToDevice);
  system->state->recv_state();
  system->state->recv_lbfgs();
  if(step == 0) { // Just do a steepest descent step w/ a line search_d
    // Normalize search direction
    float normal = 1.0f;
    cublasSnrm2_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount, system->state->search_d, 1, &normal);
    system->state->recv_lbfgs();
    assert(normal != 0.0f);
    normal = -1.0f / normal; // Reverse direction
    cublasSscal_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount, &normal, system->state->search_d, 1);
    system->state->recv_lbfgs();
    return;
  }
  int stepM = step + min(system->state->m, step); // as many as you have
  for(int i = step; i < stepM; i++) {
    int id = (i-1) % system->state->m; // index into length m arrays
    lbfgsForward(system->state->search_d, system->state->rho, system->state->alpha,
                 system->state->position_residuals_d, system->state->gradient_residuals_d, id,
                 system->state->atomCount, system->state->lambdaCount, system->state->cublasHandle);
  }
  system->state->recv_lbfgs();
  // Apply gamma
  cublasSscal_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount, &system->state->gamma, system->state->search_d, 1);
  system->state->recv_lbfgs();
  for(int i = stepM; i > step; i--) {
    int id = (i-1) % system->state->m;
    lbfgsBackward(system->state->search_d, system->state->rho, system->state->alpha,
                  system->state->position_residuals_d, system->state->gradient_residuals_d, id,
                  system->state->atomCount, system->state->lambdaCount, system->state->cublasHandle);
  }
  system->state->recv_lbfgs();
  // Normalize search direction
  float normal = 1.0f;
  cublasSnrm2_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount, system->state->search_d, 1, &normal);
  assert(normal != 0.0f);
  normal = -1.0f / normal; // Reverse direction
  cublasSscal_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount, &normal, system->state->search_d, 1);
  system->state->recv_lbfgs();
}


// phi = f(X+alpha*D), phiDot = f(X+alpha*D)/d(X+alpha*D) . d(X+alpha*D)/d(alpha)
void phi(float trialStepSize, System* system, float* posGrad) { // posGrad[0] = phi(a) posGrad[1] = phi'(a)
  // Trial move x = x0 + alpha * search_d
  system->state->recv_lbfgs();
  cublasSaxpy_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount, &trialStepSize,
                 system->state->search_d, 1, system->state->prev_position_d, 1);
  // Lose double precision here, unavoidable
  copy_float_onto_double<<<(3*system->state->atomCount+system->state->lambdaCount+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32,system->run->updateStream>>>
      (system->state->prev_position_d, system->state->leapState->x, 3*system->state->atomCount+system->state->lambdaCount);
  system->state->recv_lbfgs();
  system->domdec->update_domdec(system,false);
  system->potential->calc_force(0, system);
  system->state->recv_energy();
  posGrad[0] = system->state->energy[eepotential]; // phi
  // Move back to x0
  trialStepSize = -trialStepSize;
  cublasSaxpy_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount, &trialStepSize,
                 system->state->search_d, 1, system->state->prev_position_d, 1);
  copy_float_onto_double<<<(3*system->state->atomCount+system->state->lambdaCount+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32,system->run->updateStream>>>
      (system->state->prev_position_d, system->state->leapState->x, 3*system->state->atomCount+system->state->lambdaCount);
  // phiDot
  cublasSdot_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount,
    system->state->leapState->f, 1, system->state->search_d, 1, posGrad+1); // gradient
  float gnorm = 0.0;
  cublasSnrm2_v2(system->state->cublasHandle, 3*system->state->atomCount+system->state->lambdaCount, system->state->leapState->f, 1, &gnorm);
  posGrad[1] = posGrad[1] / gnorm; // f projected onto search direction - zero at a minimum
}

/**
 * Overwrites phiGrad at c
 */
void update(float a, float b, float c, float* ab, float* phiGrad, const float* oldPhiGrad, System* system) {
  if(c < a || c > b) { // U0 c isn't part of (a,b)
    ab[0] = a;
    ab[1] = b;
    return;
  }
  phi(c, system, phiGrad);
  if(phiGrad[1] >= 0.0) { // U1
    ab[0] = a;
    ab[1] = c;
    return;
  }
  float epsK = 1e-6;
  if(phiGrad[1] < 0 && phiGrad[0] <= oldPhiGrad[0]+epsK) { // U2
    ab[0] = c;
    ab[1] = b;
    return;
  }
  if(phiGrad[1] < 0 && phiGrad[0] > oldPhiGrad[0]+epsK) { // U3
    ab[0] = a;
    ab[1] = c;
    float theta = .5;
    int iter = 0;
    while(iter < 10) { // Binary search between a & c
      iter++;
      float d = (1 - theta) * ab[0] + theta * ab[1];
      float phiD[2];
      phi(d, system, phiD);
      if(phiD[1] >= 0) { // U3a - end binary search when grad at d is first positive
        ab[1] = d;
        return;
      }
      if(phiD[1] < 0 && phiD[0] <= oldPhiGrad[0] + epsK) { // U3b
        ab[0] = d;
        continue;
      }
      if(phiD[1] < 0 && phiD[0] > oldPhiGrad[0] + epsK) { // U3c
        ab[1] = d;
      }
    }
  }
}

/**
 * secant(a,b) = a*phi'(b) - b*phi'(a) / phi'(b) - phi'(a)
 * Two energy and gradient evals.
 */
float secant(real a, real b, System* system) {
  float phiA[2], phiB[2];
  phi(a, system, phiA);
  phi(b, system, phiB);
  return (a*phiB[1] - b*phiA[1]) / (phiB[1] - phiA[1]);
}

/**
 * Modifies phiGrad
 */
void secant2(float a, float b, float* ab, float* phiGrad, const float* oldPhiGrad, System* system) {
  float c = secant(a, b, system); // S1
  float AB[2];
  update(a, b, c, AB, phiGrad, oldPhiGrad, system); // S1
  float cbar = 0.0;
  if(c == AB[1]) { // S2
    cbar = secant(b, AB[1], system);
  }
  if(c == AB[0]) { // S3
    cbar = secant(a, AB[0], system);
  }
  if(c == AB[0] || c == AB[1]) { // S4.1
    update(AB[0], AB[1], cbar, ab, phiGrad, oldPhiGrad, system);
    return;
  }
  ab[0] = AB[0]; // S4.2
  ab[1] = AB[1];
}

/**
 * Modifies phiGrad
 */
void bracket(float c, float* ab, float* phiGrad, const float* oldPhiGrad, System* system) {
  float ci = 1e-8; // smallest possible step size
  float cj = c; // Initial guess
  phi(c, system, phiGrad);
  int iter = 0;
  while(iter < 10) {
    iter++;
    constexpr float rho = 5;
    if(phiGrad[1] >= 0) { // B1
      ab[0] = ci;
      ab[1] = cj;
      return;
    }
    constexpr float epsK = 1e-6;
    if(phiGrad[0] <= oldPhiGrad[0]+epsK) { // B1, i < j
      ci = cj;
    }
    if(phiGrad[1] < 0 && phiGrad[0] > oldPhiGrad[0] + epsK) { // B2
      ab[0] = 1e-8;
      ab[1] = cj;
      update(ab[0], ab[1], cj, ab, phiGrad, oldPhiGrad, system);
      return;
    }
    cj = rho * cj; // B3
    phi(cj, system, phiGrad);
  }
}

/**
 * Finds optimal step size along by performing energy/gradient evaluations along the direction.
 * @param system
 */
void wolfeLineSearch(System* system, int step) {
  // Move current position & grad into prev_position_d & prev_gradient_d
  // Position needs to be cast to float as it is copied
  copy_double_onto_float<<<(3*system->state->atomCount+system->state->lambdaCount+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32,system->run->updateStream>>>
      (system->state->leapState->x, system->state->prev_position_d, 3*system->state->atomCount+system->state->lambdaCount);
  cudaMemcpy(system->state->prev_gradient_d, system->state->leapState->f,
    (3*system->state->atomCount+system->state->lambdaCount)*sizeof(float),cudaMemcpyDeviceToDevice);
  system->state->recv_lbfgs();

  // Search from Hager-Zhang
  float psi2 = 2.0;
  float c = step == 0 ? 1 : psi2 * system->state->stepSize; // L0
  float phiGrad[2], oldPhiGrad[2]; // phi(c), phi'(c)
  phi(0, system, phiGrad);
  oldPhiGrad[0] = phiGrad[0];
  oldPhiGrad[1] = phiGrad[1];
  float ab[2];
  bracket(c, ab, phiGrad, oldPhiGrad, system);
  float aj = ab[0];
  float bj = ab[1];
  int maxIter = 10;
  for(int i = 0; i < maxIter; i++) {
    secant2(aj, bj, ab, phiGrad, oldPhiGrad, system); // L1
    float gamma = .66;
    if(ab[1] - ab[0] > gamma*(bj - aj)) { // L2
      c = (ab[0] + ab[1]) / 2;
      update(ab[0], ab[1], c, ab, phiGrad, oldPhiGrad, system);
    }
    aj = ab[0]; // l3
    bj = ab[1];
    phi(c, system, phiGrad);
    // Check (approximate) wolf conditions depending on i
    float del = .1;
    float sigma = .9; // phiGrad hasn't been changed since update
    bool wolfe1 = phiGrad[0] - oldPhiGrad[0] <= del * c * oldPhiGrad[1];
    bool wolfe2 = phiGrad[1] >= sigma * oldPhiGrad[1];
    bool approxWolfe = (2*del - 1) * oldPhiGrad[1] >= phiGrad[1] && phiGrad[1] >= sigma * oldPhiGrad[1];
    if(wolfe1 && wolfe2) {
      break;
    }
    if(i == maxIter-1) {
      //fprintf(stdout, "Reached max line search iterations! Defaulting to smallest step size possible!");
      c = 1e-8;
    }
  }
  system->state->stepSize = c;
}

__global__ void no_mass_weight_kernel(int N,real* masses,real* ones)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real m;

  if (i<N) {
    m=masses[i];
    ones[i]=(isfinite(m)?1:m);
  }
}

__global__ void sd_acceleration_kernel(int N,struct LeapState ls)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real ism;

  if (i<N) {
    ism=ls.ism[i];
    // No acceleration for massless virtual particles
    ls.v[i]=(isfinite(ism)?-ls.f[i]*ls.ism[i]*ls.ism[i]:0);
  }
}

__global__ void sd_scaling_kernel(int N,struct LeapState ls,real_e *grads2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3 grad;
  real grad2=0;
  extern __shared__ real sGrad2[];

  if (i<N) {
    grad.x=ls.v[i+0];
    grad.y=ls.v[i+1];
    grad.z=ls.v[i+2];
    grad2 =grad.x*grad.x;
    grad2+=grad.y*grad.y;
    grad2+=grad.z*grad.z;
  }

  real_sum_reduce(grad2/N,sGrad2,grads2);
  real_max_reduce(grad2,sGrad2,grads2+1);
}

__global__ void sd_position_kernel(int N,struct LeapState ls,real_v *v,real scale,real_x *bx)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real_x x;

  if (i<N) {
    x=ls.x[i];
    if (bx) bx[i]=x;
    ls.x[i]=x+scale*v[i];
  }
}

__global__ void sdfd_dotproduct_kernel(int N,struct LeapState ls,real_v *minDirection,real_e *dot)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lDot=0;
  extern __shared__ real sDot[];

  if (i<N) {
    lDot=ls.v[i]*minDirection[i];
  }

  real_sum_reduce(3*lDot/N,sDot,dot);
}

void State::min_init(System *system)
{
  // Set masses to 1 for shake during minimization, except virtual sites
  cudaMalloc(&(leapState->ism),(3*atomCount+lambdaCount)*sizeof(real));
  no_mass_weight_kernel<<<(3*atomCount+lambdaCount+BLUP-1)/BLUP,BLUP,
    0,system->run->updateStream>>>
    (3*atomCount+lambdaCount,invsqrtMassBuffer_d,leapState->ism);
  system->run->minType = elbfgs;
  if (system->run->minType==elbfgs) {
    system->state->cublasHandle = nullptr;
    cublasCreate_v2(&cublasHandle);
    cublasSetStream_v2(cublasHandle, system->run->updateStream);
    cudaMalloc(&search_d, (3*atomCount+lambdaCount)*sizeof(float));
    search = (float*)calloc(sizeof(float), 3*atomCount+lambdaCount);
    cudaMalloc(&prev_position_d, (3*atomCount+lambdaCount)*sizeof(float));
    prev_position = (float*)calloc(sizeof(float), 3*atomCount+lambdaCount);
    cudaMalloc(&position_residuals_d, (3*atomCount+lambdaCount)*m*sizeof(float)); // indexed like [m][3*atomCount]
    position_residuals = (float*)calloc(m*sizeof(float), 3*atomCount+lambdaCount);
    cudaMalloc(&prev_gradient_d, (3*atomCount+lambdaCount)*sizeof(float));
    prev_gradient = (float*)calloc(sizeof(float), 3*atomCount+lambdaCount);
    cudaMalloc(&gradient_residuals_d, (3*atomCount+lambdaCount)*m*sizeof(float)); // indexed like [m][3*atomCount]
    gradient_residuals = (float*)calloc(m*sizeof(float), 3*atomCount+lambdaCount);
    rho = (float*)malloc(m*sizeof(float)); // [m]
    alpha = (float*)malloc(m*sizeof(float)); // [m]
    cudaMalloc(&grads2_d,2*sizeof(real_e));
    cudaMemset(grads2_d,0,2*sizeof(real_e));
  }
  if (system->run->minType==esd || system->run->minType==esdfd) {
    cudaMalloc(&grads2_d,2*sizeof(real_e));
    cudaMemset(grads2_d,0,2*sizeof(real_e));
  }
  if (system->run->minType==esdfd) {
    cudaMalloc(&minDirection_d,(3*atomCount+lambdaCount)*sizeof(real_v));
  }
}

void State::min_dest(System *system)
{
  // Set masses back
  cudaFree(leapState->ism);
  leapState->ism=invsqrtMassBuffer_d;
  if (system->run->minType==esd || system->run->minType==esdfd) {
    cudaFree(grads2_d);
  }
  if (system->run->minType==esdfd) {
    cudaFree(minDirection_d);
  }
}

void State::min_move(int step,int nsteps,System *system)
{
  Run *r=system->run;
  real_e grads2[2];
  real_e currEnergy;
  real_e gradDot[1];
  real scaling, rescaling;
  real frac;

  if (r->minType == elbfgs) {
    if(system->id==0 && system->state->stepSize > 1e-8) { // skip rest of minimization if last step was too small
      recv_energy();
      if (system->verbose>0) display_nrg(system);
      prevEnergy=energy[eepotential];
      // Compute L-BFGS direction & step size
      lbfgsDirection(system, step);
      wolfeLineSearch(system, step);
      // Move with step size
      // X = X + alpha * D
      update_lbfgs_position<<<(3*atomCount+lambdaCount+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32,r->updateStream>>>
          (leapState->x, search_d, stepSize, 3*atomCount+lambdaCount);
      system->domdec->update_domdec(system,false); // true to always update neighbor list
      system->potential->calc_force(0, system);
      recv_energy();
      currEnergy = energy[eepotential];
      float diff = currEnergy - prevEnergy;

      // rms calculation
      sd_scaling_kernel<<<(3*atomCount+lambdaCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (3*atomCount+lambdaCount,*leapState,grads2_d);
      cudaMemcpy(grads2,grads2_d,2*sizeof(real_e),cudaMemcpyDeviceToHost);
      cudaMemset(grads2_d,0,2*sizeof(real_e));
      fprintf(stdout,"rmsgrad = %f\n",sqrt(grads2[0]));
      fprintf(stdout,"maxgrad = %f\n",sqrt(grads2[1]));
      if(currEnergy > prevEnergy) {
        fprintf(stdout, "Minimization step # %d increased potential by %f kcal/mol\n", step, diff);
      } else {
        fprintf(stdout, "Min step: %f\n", diff);
      }
      //Update L-BFGS history
      lbfgsUpdate(system->state->leapState, system->state->rho, &system->state->gamma,
                  system->state->prev_position_d, system->state->position_residuals_d,
                  system->state->prev_gradient_d, system->state->gradient_residuals_d, step % system->state->m,
                  system->state->atomCount, system->state->cublasHandle, system);
    } else if(system->state->stepSize != 0.0) {
      fprintf(stdout, "Skipping the rest of the minimization steps after %d due to convergence!\n", step);
      fprintf(stdout, "Final Energy: %f\n", prevEnergy);
      system->state->stepSize = 0.0;
    }
  } else if (r->minType==esd) {
    if (system->id==0) {
      recv_energy();
      if (system->verbose>0) display_nrg(system);
      currEnergy=energy[eepotential];
      if (step==0) {
        r->dxRMS=r->dxRMSInit;
      } else if (currEnergy<prevEnergy) {
        r->dxRMS*=1.2;
      } else {
        r->dxRMS*=0.5;
      }
      prevEnergy=currEnergy;
      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      holonomic_velocity(system);
      sd_scaling_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (3*atomCount,*leapState,grads2_d);
      cudaMemcpy(grads2,grads2_d,2*sizeof(real_e),cudaMemcpyDeviceToHost);
      cudaMemset(grads2_d,0,2*sizeof(real_e));
      fprintf(stdout,"rmsgrad = %f\n",sqrt(grads2[0]));
      fprintf(stdout,"maxgrad = %f\n",sqrt(grads2[1]));
      recv_energy();
      real currE = energy[eepotential];
      fprintf(stdout, "energy = %f\n", currE);
      // scaling factor to achieve desired rms displacement
      scaling=r->dxRMS/sqrt(grads2[0]);
      // ratio of allowed maximum displacement over actual maximum displacement
      rescaling=r->dxAtomMax/(scaling*sqrt(grads2[1]));
      fprintf(stdout,"scaling = %f, rescaling = %f\n",scaling,rescaling);
      // decrease scaling factor if actual max violates allowed max
      if (rescaling<1) {
        scaling*=rescaling;
        r->dxRMS*=rescaling;
      }
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,leapState->v,scaling,positionCons_d);
      holonomic_position(system);
    }
  } else if (r->minType==esdfd) {
    if (system->id==0) {
      recv_energy();
      if (system->verbose>0) display_nrg(system);
      if (step==0) {
        r->dxRMS=r->dxRMSInit;
      }
      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      holonomic_velocity(system);
      cudaMemcpy(minDirection_d,leapState->v,3*atomCount*sizeof(real_v),cudaMemcpyDeviceToDevice);
      sd_scaling_kernel<<<(atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (atomCount,*leapState,grads2_d);
      cudaMemcpy(grads2,grads2_d,2*sizeof(real_e),cudaMemcpyDeviceToHost);
      cudaMemset(grads2_d,0,2*sizeof(real_e));
      fprintf(stdout,"rmsgrad = %f\n",sqrt(grads2[0]));
      fprintf(stdout,"maxgrad = %f\n",sqrt(grads2[1]));
      // scaling factor to achieve desired rms displacement
      scaling=r->dxRMS/sqrt(grads2[0]);
      // ratio of allowed maximum displacement over actual maximum displacement
      rescaling=r->dxAtomMax/(scaling*sqrt(grads2[1]));
      fprintf(stdout,"scaling = %f, rescaling = %f\n",scaling,rescaling);
      // decrease scaling factor if actual max violates allowed max
      if (rescaling<1) {
        scaling*=rescaling;
        r->dxRMS*=rescaling;
      }
      backup_position();
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,leapState->v,scaling,positionCons_d);
      holonomic_position(system);
    }
    system->domdec->update_domdec(system,false); // false, no need to update neighbor list
    system->potential->calc_force(0,system); // step 0 to always calculate energy
    if (system->id==0) {
      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      holonomic_velocity(system);
      sdfd_dotproduct_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (3*atomCount,*leapState,minDirection_d,grads2_d);
      cudaMemcpy(gradDot,grads2_d,sizeof(real_e),cudaMemcpyDeviceToHost);
      cudaMemset(grads2_d,0,2*sizeof(real_e));
      // grads2[0] is F*F, gradDot is F*Fnew
      frac=grads2[0]/(grads2[0]-gradDot[0]);
      fprintf(stdout,"F(x0)*dx = %f, F(x0+dx)*dx = %f, frac = %f\n",grads2[0],gradDot[0],frac);
      if (frac>1.44 || frac<0) {
        r->dxRMS*=1.2;
      } else if (frac<0.25) {
        r->dxRMS*=0.5;
      } else {
        r->dxRMS*=sqrt(frac);
      }
      frac*=(nsteps-step)/(1.0*nsteps);
      if (frac>1 || frac<0) frac=1;
      restore_position();
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,minDirection_d,frac*scaling,positionCons_d);
      holonomic_position(system);
    }
  } else {
    fatal(__FILE__,__LINE__,"Error: Unrecognized minimization type %d\n",r->minType);
  }
}
