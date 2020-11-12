#include <stdio.h>
#include <limits>
#include <iostream>
#include <chrono>
#define N 256
#define TPB 256

__device__ float plus(float a, float b)
{
    return a + b;
}

__global__ void distanceKernel(float *d_out, float *d_x, float *d_y)
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("Adding %f and %f \n", d_x[i], d_y[i]);
    d_out[i] = d_x[i] + d_y[i];
    //printf("Finished from thread %d \n", i);
    //printf("%f.\n", d_out[i]);
}

int main()
{
    // Declare a pointer for an array of floats
    float *d_out = 0;
    float *d_x = 0;
    float *d_y = 0;

    // Allocate device memory to store the output array
    cudaMalloc(&d_out, 100000000*sizeof(float));
    cudaMalloc(&d_x, 100000000*sizeof(float));
    cudaMalloc(&d_y, 100000000*sizeof(float));

    float *x = (float*) malloc(100000000*sizeof(float)); for(int i=0; i < 100000000; i++) x[i] = 100.0;
    float *y = (float*) malloc(100000000*sizeof(float)); for(int i=0; i < 100000000; i++) y[i] = 100.1;

    // CPU add
    auto start = std::chrono::steady_clock::now();
    for (int i=0;i < 100000000;i++) {
        float res = x[i] + y[i];
    //    printf("%f\n", res);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "CPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

    cudaMemcpy(d_x, x, 100000000*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, 100000000*sizeof(float),cudaMemcpyHostToDevice);

    //my_kernel <<<Dg, Db>>>(arg1, arg2, ...)
    // Dg = number of thread blocks
    // Db = number of threads per block
    // The total number of threads is Dg*Db.
    start = std::chrono::steady_clock::now();
    distanceKernel<<<(N+TPB-1)/TPB, TPB>>>(d_out, d_x, d_y);
    end = std::chrono::steady_clock::now();

    std::cout << "GPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

    // Free the memory
    cudaFree(d_out);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
