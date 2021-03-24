#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>
#include <limits>
#include <iostream>
#include <chrono>
#define N 256
#define TPB 256
#define TRIALS_PER_THREAD 12800


__global__ void gpu_random(curandState *states, float *device_pi_estimates) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int seed = id; // different seed per thread
    float x, y, z;
    int p_in_circle = 0;

    curand_init(seed, id, 0, &states[id]);  // 	Initialize CURAND

    for(int i = 0; i < TRIALS_PER_THREAD; i++) {
        x = curand_uniform (&states[id]);
        y = curand_uniform (&states[id]);
        z = x * x + y * y;
        if (z <= 1.0) {
            p_in_circle ++;
        }
    }

    device_pi_estimates[id] = 4.0 * (float)p_in_circle / (float)TRIALS_PER_THREAD;
}

int main(int argc, char const *argv[]) {
    // init cuda random state
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, N*sizeof(curandState));

    // init calculation of Pi using Monte Carlo
    float *d_pi, *pi;
    cudaMalloc(&d_pi, sizeof(float) * N);
    pi = (float*)malloc(sizeof(float) * N);

    // call kernel
    gpu_random <<<(N+TPB-1)/TPB, TPB>>>(dev_random, d_pi);

    // very important...
    cudaDeviceSynchronize();

    // copy back to host
    cudaMemcpy(pi, d_pi, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // calculate pi on host
    float sum_pi = 0;
    float ave_pi = 0;
    for (int i = 0; i < N; i++) {
        sum_pi += pi[i];
    }
    ave_pi = sum_pi / float(N);

    int n_samples = N * TRIALS_PER_THREAD;

    printf("The computed Pi with %d samples is %f\n", n_samples, ave_pi);

    return 0;
}
