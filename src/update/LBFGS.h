
#ifndef LBFGS_H
#define LBFGS_H

#include <type_traits>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <functional>

using namespace std;

template <typename T> class LBFGS {
    static_assert(std::is_floating_point<T>::value, "L-BFGS can only be used with floating point types");

public:
    LBFGS(int m, int DOF, std::function<T(T*, T*)> user_grad) : m(m), DOF(DOF), grad(user_grad) {
        minimized = false;
        min_step_size = sizeof(T) == 4 ? 1e-4 : 1e-7;
        max_step_size = 1.0;
        rho = (T*) calloc(m, sizeof(T));
        alpha = (T*) calloc(m, sizeof(T));
        gamma = (T*) calloc(m, sizeof(T));
        beta = (T*) calloc(m, sizeof(T));
        step_size = 0.0;
        steepest_descent_steps = 0;
        linesearch_failures = 0;
         
        search = (T*) calloc(DOF, sizeof(T));
        prev_positions = (T*) calloc(DOF, sizeof(T));
        prev_gradient = (T*) calloc(DOF, sizeof(T));

        q = (T*) calloc(DOF, sizeof(T));
        x_plus_step = (T*) calloc(DOF, sizeof(T));
        s = (T*) calloc(m*DOF, sizeof(T));
        y = (T*) calloc(m*DOF, sizeof(T));
    }

    ~LBFGS() {
        free(rho);
        free(alpha);
        free(gamma);
        free(beta);
        free(G);
        free(X);
        free(search);
        free(prev_positions);
        free(prev_gradient);
        free(q);
        free(x_plus_step);
        free(s);
        free(y);
    }

/**
 * Steepest decent step
 * 
 * @param X Initial Position
 * @param G Initial Position Grad
 */
void init(T* X, T* G, T steepest_descent_step_size) {
    // f(X0), g(X0)
    T f_initial = grad(X, G);
    //real p0[50], p1[50];
    for (int i = 0; i < DOF; ++i) {
        prev_positions[i] = X[i];
        prev_gradient[i] = G[i];
    }
    for (int i = 0; i < DOF; ++i) {
        q[i] = G[i];
    }
    steepest_descent_step_size = new_linesearch(f_initial, X, G);
    printf("steepest descent step size: %f\n", steepest_descent_step_size);
    for (int i = 0; i < DOF; ++i) {
        X[i] += steepest_descent_step_size * -q[i];
    }
}

/**
 * 1. Compute new L-BFGS step direction
 *   Pseudocode from wikipedia:
 *   q = g.i // search direction to be updated
 *   for j = i-1 to i-m:
 *     alpha.j = rho.j * s.j.T * q // dot product
 *     q = q - alpha.j * y.j // vector scale & subtraction
 *   gamma.i = s.i-1.T * y.i-1 / y.i-1.T * y.i-1 // dot products in numerator and denominator
 *   q = gamma.i * q
 *   for j = i-m to i-1:
 *     beta = rho.j * y.j.T * q // dot product
 *     q = q + (alpha.j - beta) * s.j // vector scale & addition
 *   q = -q  // negate applied above instead of here most likely
 *   gamma = s.i.T * y.i / y.i.T * y.i
 *   rho.j = 1 / (y.j.T * s.j)
 */
void minimize_step(T* X, T* G) {
    if (minimized){
        return;
    }
    // Eval function at X
    T p0 = grad(X, G);
    for (int i = 0; i < DOF; ++i) {
        q[i] = G[i];
    }
    update(X, G);
    if (minimized){
        return;
    }
    if (std::isnan(rho[m-1]) || std::isinf(rho[m-1])) {
        printf("ERROR: Dividing by zero. Resetting and doing steepest descent\n");
        for (int i = 0; i < this->m*DOF; ++i) {
            s[i] = 0;
            y[i] = 0;
        }
        for (int i = 0; i < this->m; ++i) {
            rho[i] = 0;
            alpha[i] = 0;
            gamma[i] = 0;
        }
        init(X, G, 0.0001);
        ++steepest_descent_steps;
        printf("Steepest descent steps: %i\n", steepest_descent_steps);
        return;
    }
    for (int i = m - 1; i >= 0; --i) {
        alpha[i] = rho[i] * dot_product(s + i * DOF, q, DOF);
        for (int j = 0; j < DOF; ++j) {
            q[j] -= alpha[i] * y[i * DOF + j];
        }
    }
    T gamma = dot_product(s + (m - 1) * DOF, y + (m - 1) * DOF, DOF) / dot_product(y + (m - 1) * DOF, y + (m - 1) * DOF, DOF);
    for (int j = 0; j < DOF; ++j) {
        q[j] *= gamma;
    }
    for (int i = 0; i < m; ++i) {
        T beta = rho[i] * dot_product(y + i * DOF, q, DOF);
        for (int j = 0; j < DOF; ++j) {
            q[j] += (alpha[i] - beta) * s[i * DOF + j];
        }
    }
    // min_a f(X + a*q)
    printf("Old function value: %f\n", p0);
    /*
    if (dot_product(q, G, DOF) < 0) {
        printf("Warning: search direction points up, using steepest descent");
        ++steepest_descent_steps;
        printf("Steepest descent steps: %i\n", steepest_descent_steps);
        for (int i = 0; i < DOF; ++i) {
            q[i] = G[i];
        }
    }*/
    step_size = this->new_linesearch(p0, X, G); 
    grad(X, G);
    std::cout << "step size: " << step_size << "\n";
    // Update system positions
    for (int j = 0; j < DOF; ++j) {
        X[j] += (step_size * -q[j]);
    }
    grad(X, G);
    T G_magnitude = 0;
    for (int i = 0; i < DOF; ++i) {
        G_magnitude += G[i]*G[i];
    }
    G_magnitude = sqrt(G_magnitude);
    printf("Magnitude of gradient: %f\n", G_magnitude);
    printf("New function value: %f\n", grad(X, G));
}

void update(T* X, T* G) {
    real prev[50], current[50];
    memcpy(prev, prev_gradient, 50*sizeof(real));
    memcpy(current, G, 50*sizeof(real));
    for (int i = 0; i < ((m - 1) * DOF); ++i) {
        s[i] = s[i + DOF];
        y[i] = y[i + DOF];
    }
    for (int i = 0; i < DOF; ++i) {
        s[(m - 1) * DOF + i] = X[i] - prev_positions[i];
        y[(m - 1) * DOF + i] = G[i] - prev_gradient[i];
    }
    real y50[50];
    memcpy(y50, y, 50*sizeof(real));
    for (int i = 0; i < m - 1; ++i) {
        rho[i] = rho[i + 1];
    }
    double s_dot_y = dot_product((s + (m - 1) * DOF), (y + (m - 1) * DOF), DOF);
    if (abs(s_dot_y) <= 1e-5) {
        std::cout << "Error: Dividing by zero.\n";
    }
    rho[m - 1] = 1 / s_dot_y;
    
    for (int i = 0; i < DOF; ++i) {
        prev_positions[i] = X[i];
        prev_gradient[i] = G[i];
    }
}

T linesearch(T p0, T* X, T* G) {
    int max_it = 1000;
    T c1 = 1e-4;
    T c2 = 0.7;
    T tau = 0.75;
    T m_ls = 0.0;
    for (int i = 0; i < DOF; ++i) {
        m_ls += G[i] * (-q[i]);
    }
    if (m_ls >= 0 ) {
        printf("ERROR: m should be negative\n");
         printf("m: %f\n", m_ls);

        for (int i = 0; i < this->m*DOF; ++i) {
            s[i] = 0;
            y[i] = 0;
        }
        for (int i = 0; i < this->m; ++i) {
            rho[i] = 0;
            alpha[i] = 0;
            gamma[i] = 0;
        }

        return 0;
    }
    printf("m: %f\n", m_ls); 
    T step_size = 1;
    printf("q[3]: %f, g[3] %f\n", q[3], G[3]);

    for (int i = 0; i < max_it; ++i) {
        for (int j = 0; j < DOF; ++j) {
            x_plus_step[j] = X[j] - (step_size * q[j]);
        }
        T new_value = grad(x_plus_step, G);
        if (p0 - new_value >= step_size * -c1 * m_ls) {
            printf("Armijo satisfied\n");
            return step_size;
        } 
        else {
            step_size *= tau;
        }
    }
    return 0;
}
T new_linesearch(T p0, T* X, T* G) {
    T c1 = 1e-4;
    T c2 = 0.9;
    T m_ls = 0.0;
    T alpha_low = 0;
    T alpha_high = 100;
    T alpha = (alpha_low + alpha_high) / 2;
    T max_it = 100;

    for (int i = 0; i < DOF; ++i) {
        m_ls += G[i] * (-q[i]); // m_ls = grad . search direction
    }
    if (m_ls >= 0 ) {
        printf("ERROR: m should be negative\n");
        printf("m: %f\n", m_ls);
         
        for (int i = 0; i < this->m*DOF; ++i) {
            s[i] = 0;
            y[i] = 0;
        }
        for (int i = 0; i < this->m; ++i) {
            rho[i] = 0;
        }
        return 0;
    }
    for (int i = 0; i < max_it; ++i) {
        alpha = (alpha_low + alpha_high) / 2; //step size
        if (alpha < 1e-20) {
            printf("ERROR: linesearch failed because step is too small\n");
            ++linesearch_failures;
            printf("# of linesearch failures: %i\n", linesearch_failures);
            return 0;
        }
        for (int j = 0; j < DOF; ++j) {
            x_plus_step[j] = X[j] - (alpha * q[j]);
        }
        grad(x_plus_step, G);
        if (p0 - grad(x_plus_step, G) < alpha * -c1 *m_ls) { // 1st wolfe not satisfied
            alpha_high = alpha;
            continue;
        }
        if(abs((dot_product(q, G, DOF))) > abs(c2 * m_ls)) { // 2nd wolfe not satisfied (strong wolfe)
            alpha_low = alpha;
            continue;
        }
        return alpha;
    } 
    printf("Warning: linesearch failed\n");
    ++linesearch_failures;
    printf("Upper bound: %f, Lower bound: %f, step: %f\n", alpha_high, alpha_low, alpha);
    printf("# of linesearch failures: %i\n", linesearch_failures);
    return alpha;
}


private:
    int m; // Number of previous gradients to use for hessian approximation (5-7)
    int DOF; // Degrees of freedom
    T min_step_size; // terminate minimization with steps smaller than this number
    T max_step_size; // ensures that lbfgs doesn't overshoot minimum
    bool minimized;
    int steepest_descent_steps;
    int linesearch_failures;

    T *beta; 
    T *gamma; // s projected onto y
    T *rho; // [m] rho_{i} = (s_{i}^T * y_{i}
    T *alpha; // [m] alpha_{i} = rho_{i} * s_{i}^T * y_{i}

    T *X;
    T *G;
    T *search; // [DOF] L-BFGS search direction
    T *prev_positions; // [DOF] x_{i-1}
    T *prev_gradient; // [DOF] g_{i-1}

    T* q;
    T* x_plus_step;
    T *s; // [m*DOF] x_{i+1} - x_{i} = s_{i}
    T *y; // [m*DOF] grad_{i+1} - grad_{i} = y_{i}
    T step_size;

    std::function<T(T*, T*)> grad;

    T dot_product(T* a, T* b, int n) {
        T result = 0;
        for (int i = 0; i < n; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
};

#endif // LBFGS