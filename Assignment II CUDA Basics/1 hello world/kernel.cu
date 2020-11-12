#include <stdio.h>
#define N 256
#define TPB 256

__global__ void hello(float *d_out)
{
    printf("Hello world from thread %d \n", threadIdx.x);
}

int main()
{

  // Declare a pointer for an array of floats
  float *d_out = 0;

  // Allocate device memory to store the output array
  cudaMalloc(&d_out, N*sizeof(float));

  hello<<<N/TPB, TPB>>>(d_out);

  cudaFree(d_out); // Free the memory
  return 0;
}

