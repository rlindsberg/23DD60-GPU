#include <stdio.h>
#define N 20
#define TPB 20

__device__ float plus(float a, float b)
{
    return a + b;
}

__global__ void distanceKernel(float *d_out, float *d_x, float *d_y)
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Adding %f and %f \n", d_x[i], d_y[i]);
    d_out[i] = d_x[i] + d_y[i];
    printf("Finished from thread %d \n", i);
    printf("%f.\n", d_out[i]);
}

int main()
{

    // Declare a pointer for an array of floats
    float *d_out = 0;
    // Allocate device memory to store the output array
    cudaMalloc(&d_out, 20*sizeof(float));
    float *d_x = 0;
    cudaMalloc(&d_x, 20*sizeof(float));
    float *d_y = 0;
    cudaMalloc(&d_y, 20*sizeof(float));

    float *x = (float*) malloc(20*sizeof(float)); for(int i=0; i < 20; i++) x[i] = 100.0;
    float *y = (float*) malloc(20*sizeof(float)); for(int i=0; i < 20; i++) y[i] = 100.1;

    // CPU add
    for (int i=0;i < 20;i++) {
        float res = x[i] + y[i];
    //    printf("%f\n", res);
    }

    cudaMemcpy(d_x, x, 20*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, 20*sizeof(float),cudaMemcpyHostToDevice);

    //my_kernel <<<Dg, Db>>>(arg1, arg2, ...)
    // Dg = number of thread blocks
    // Db = number of threads per block
    // The total number of threads is Dg*Db.
    distanceKernel<<<N/TPB, TPB>>>(d_out, d_x, d_y);

    // Free the memory
    cudaFree(d_out);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
