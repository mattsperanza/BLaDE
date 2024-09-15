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

void lbfgsForward(double* q, double* rho, double* alpha, double* s, double* y, int mIndex, int N, cublasHandle_t handle) {
  // Launch kernels for alpha and update q
  // alpha = s . q
  cublasDdot_v2(handle, 3*N, s+mIndex*3*N, 1, q, 1, alpha+mIndex);
  // q = q - rho * alpha * y
  alpha[mIndex] *= -rho[mIndex]; // How do I get this onto the gpu?!
  cublasDaxpy_v2(handle, 3*N, alpha+mIndex, y+mIndex*3*N, 1, q, 1); // writes over q
}

void lbfgsBackward(double* q, double* rho, double* alpha, double* s, double* y, int mIndex, int N, cublasHandle_t handle) {
  // Launch kernels for beta and update q
  // beta = y . q
  double beta = 0.0;
  cublasDdot_v2(handle, 3*N, y+mIndex*3*N, 1, q, 1, &beta);
  // q = q + (alpha - rho * beta) * s
  beta = alpha[mIndex] - rho[mIndex] * beta;
  cublasDaxpy_v2(handle, 3*N, &beta, s+mIndex*3*N, 1, q, 1); // writes over q
}

void lbfgsUpdate(LeapState* ls, double* rho, double* gamma, const double* previous_p, double* s, const double* previous_g,
  double* y, const int mID, const int N, cublasHandle_t handle) {
  // Launch kernels for rho, alpha, s, y calculations after line search_d
  double nOne = -1.0;
  // s = x - prev_x
  cudaMemcpy(s + mID*N, ls->x, 3*N*sizeof(real), cudaMemcpyDeviceToDevice);
  cublasDaxpy_v2(handle, N, &nOne, previous_p, 1, s + mID*N, 1);
  // y = f - prev_f
  cudaMemcpy(y + mID*N, ls->f, 3*N*sizeof(real), cudaMemcpyDeviceToDevice);
  cublasDaxpy_v2(handle, N, &nOne, previous_g, 1, y + mID*N, 1);
  // These are computed for the next iteration
  // gamma = abs(s . y / y . y)
  double numerator, denominator;
  cublasDdot_v2(handle, N, s+mID*N, 1, y+mID*N, 1, &numerator);
  cublasDdot_v2(handle, N, y+mID*N, 1, y+mID*N, 1, &denominator);
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
  cudaMemcpy(system->state->search_d, system->state->leapState->f, 3*system->state->atomCount*sizeof(double),
         cudaMemcpyDeviceToDevice);
  if(step == 0) { // Just do a steepest descent step w/ a line search_d
    return;
  }
  int stepM = step + min(system->state->m, step); // as many as you have
  for(int i = step; i < stepM; i++) {
    int id = (i-1) % system->state->m; // index into length m arrays
    lbfgsForward(system->state->search_d, system->state->rho, system->state->alpha,
                 system->state->position_residuals_d, system->state->gradient_residuals_d, id,
                 system->state->atomCount, system->state->cublasHandle);
  }
  // Apply gamma
  cublasDscal_v2(system->state->cublasHandle, system->state->atomCount, &system->state->gamma, system->state->search_d, 1);
  for(int i = stepM; i > step; i--) {
    int id = (i-1) % system->state->m;
    lbfgsBackward(system->state->search_d, system->state->rho, system->state->alpha,
                  system->state->position_residuals_d, system->state->gradient_residuals_d, id,
                  system->state->atomCount, system->state->cublasHandle);
  }
  double nOne = -1.0;
  cublasDscal_v2(system->state->cublasHandle, system->state->atomCount, &nOne, system->state->search_d, 1);
}


// phi = f(X+alpha*D), phiDot = f(X+alpha*D)/d(X+alpha*D) . d(X+alpha*D)/d(alpha)
// d(X+alpha*D)/d(alpha) = D
void phi(double trialStepSize, System* system, double* posGrad) { // posGrad[0] = phi(a) posGrad[1] = phi'(a)
  // Trial move x0 = x + alpha * search_d
  cublasDaxpy_v2(system->state->cublasHandle, system->state->atomCount, &trialStepSize,
                 system->state->search_d, 1, system->state->prev_position_d, 1);
  system->potential->calc_force(0, system);
  // Move back to x0
  trialStepSize = -trialStepSize;
  cublasDaxpy_v2(system->state->cublasHandle, system->state->atomCount, &trialStepSize,
                 system->state->search_d, 1, system->state->prev_position_d, 1);
  posGrad[0] = *system->state->energy;
  cublasDdot_v2(system->state->cublasHandle, system->state->atomCount,
    system->state->prev_gradient_d, 1, system->state->search_d, 1, posGrad+1); // gradient
}

/**
 * Overwrites phiGrad at c
 */
void update(double a, double b, double c, double* ab, double* phiGrad, const double* oldPhiGrad, System* system) {
  if(c < a && c > b) { // U0
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
  double epsK = 1e-6;
  if(phiGrad[1] < 0 && phiGrad[0] <= oldPhiGrad[0]+epsK) { // U2
    ab[0] = c;
    ab[1] = b;
    return;
  }
  if(phiGrad[1] < 0 && phiGrad[0] > oldPhiGrad[0]+epsK) { // U3
    ab[0] = a;
    ab[1] = c;
    double theta = .5;
    while(true) {
      double d = (1 - theta) * ab[0] + theta * ab[1];
      double phiD[2];
      phi(d, system, phiD);
      if(phiD[1] >= 0) { // U3a
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
double secant(real a, real b, System* system) {
  double phiA[2], phiB[2];
  phi(a, system, phiA);
  phi(b, system, phiB);
  return (a*phiB[1] - b*phiA[1]) / (phiB[1] - phiA[1]);
}

/**
 * Modifies phiGrad
 */
void secant2(double a, double b, double* ab, double* phiGrad, const double* oldPhiGrad, System* system) {
  double c = secant(a, b, system); // S1
  double AB[2];
  update(a, b, c, AB, phiGrad, oldPhiGrad, system); // S1
  double cbar = 0.0;
  if(c == AB[1]) { // S2
    cbar = secant(b, AB[1], system);
  }
  if(c == AB[0]) { // S3
    cbar = secant(AB[0], a, system);
  }
  if(c == AB[0] || c == AB[1]) { // S4.1
    update(AB[0], AB[0], cbar, ab, phiGrad, oldPhiGrad, system);
    return;
  }
  ab[0] = AB[0]; // S4.2
  ab[1] = AB[1];
}

/**
 * Modifies phiGrad
 */
void bracket(double c, double* ab, double* phiGrad, const double* oldPhiGrad, System* system) {
  double cj = c;
  double ci = c;
  phi(c, system, phiGrad);
  while(true) {
    constexpr double rho = 5;
    if(phiGrad[1] >= 0) { // B1
      ab[0] = ci;
      ab[1] = cj;
    }
    constexpr double epsK = 1e-6;
    if(phiGrad[0] <= oldPhiGrad[0]+epsK) { // B1, i < j
      ci = cj;
    }
    if(phiGrad[1] < 0 && phiGrad[0] > oldPhiGrad[0] + epsK) { // B2
      ab[0] = 0;
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
  cudaMemcpy(system->state->prev_position_d, system->state->leapState->x, 3*system->state->atomCount*sizeof(double),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(system->state->prev_gradient_d, system->state->leapState->f, 3*system->state->atomCount*sizeof(double),
             cudaMemcpyDeviceToDevice);

  // Search from Hager-Zhang
  double psi2 = 2.0;
  double c = step == 0 ? 1.0f : psi2 * system->state->stepSize; // L0
  double phiGrad[2], oldPhiGrad[2]; // phi(c), phi'(c)
  phi(c, system, phiGrad);
  oldPhiGrad[0] = phiGrad[0];
  oldPhiGrad[1] = phiGrad[1];
  double ab[2];
  bracket(c, ab, phiGrad, oldPhiGrad, system);
  double aj = ab[0];
  double bj = ab[1];
  int maxIter = 25;
  for(int i = 0; i < maxIter; i++) {
    secant2(aj, bj, ab, phiGrad, oldPhiGrad, system); // L1
    double gamma = .66;
    if(ab[0] - ab[1] > gamma*(bj - aj)) { // L2
      c = (ab[0] + ab[1]) / 2;
      update(ab[0], ab[1], c, ab, phiGrad, oldPhiGrad, system);
    }
    aj = ab[0]; // l3
    bj = ab[1];
    // Check (approximate) wolf conditions depending on i
    double del = .1;
    double sigma = .9; // phiGrad hasn't been changed since update
    bool wolfe1 = phiGrad[0] - oldPhiGrad[0] <= del * c * oldPhiGrad[1];
    bool wolfe2 = phiGrad[1] >= sigma * oldPhiGrad[1];
    bool approxWolfe = (2*del - 1) * oldPhiGrad[1] >= phiGrad[1] && phiGrad[1] >= sigma * oldPhiGrad[1];
    if(wolfe1 && wolfe2) {
      break;
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
    grad.x=ls.v[3*i+0];
    grad.y=ls.v[3*i+1];
    grad.z=ls.v[3*i+2];
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
    cudaMalloc(&search_d, 3*atomCount*sizeof(double));
    cudaMalloc(&prev_position_d, 3*atomCount*sizeof(double));
    cudaMalloc(&position_residuals_d, 3*atomCount*m*sizeof(double)); // indexed like [m][3*atomCount]
    cudaMalloc(&prev_gradient_d, 3*atomCount*sizeof(double));
    cudaMalloc(&gradient_residuals_d, 3*atomCount*m*sizeof(double)); // indexed like [m][3*atomCount]
    rho = (double*)malloc(m*sizeof(double)); // [m]
    alpha = (double*)malloc(m*sizeof(double)); // [m]
  }
  if (system->run->minType==esd || system->run->minType==esdfd) {
    cudaMalloc(&grads2_d,2*sizeof(real_e));
    cudaMemset(grads2_d,0,2*sizeof(real_e));
  }
  if (system->run->minType==esdfd) {
    cudaMalloc(&minDirection_d,3*atomCount*sizeof(real_v));
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
    if(system->id==0) {
      recv_energy();
      if (system->verbose>0) display_nrg(system);
      prevEnergy=energy[eepotential];
      // Compute L-BFGS direction & step size
      lbfgsDirection(system, step);
      wolfeLineSearch(system, step);
      // Move with step size
      // X = X + alpha * D
      cublasDaxpy_v2(cublasHandle, 3*atomCount, &stepSize, search_d, 1, leapState->x, 1);
      // Update L-BFGS history
      lbfgsUpdate(system->state->leapState, system->state->rho, &system->state->gamma,
                  system->state->prev_position_d, system->state->position_residuals_d,
                  system->state->prev_gradient_d, system->state->gradient_residuals_d, (step-1) % system->state->m,
                  system->state->atomCount, system->state->cublasHandle);
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
