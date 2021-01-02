#include <stdio.h>
#define N 24
#define TSPB 6

__global__ void hello(float *d_out)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("thread x %d \n", index_x);
}

int main()
{

  // Declare a pointer for an array of floats
  float *d_out = 0;

  // Allocate device memory to store the output array
  cudaMalloc(&d_out, N*sizeof(float));

  dim3     grid(1);
  grid = dim3(3,2);
  dim3 tbp = dim3(18,1);
// TBP is bloxkDim.x
  hello<<<grid, tbp>>>(d_out);

  cudaFree(d_out); // Free the memory
  return 0;
}
