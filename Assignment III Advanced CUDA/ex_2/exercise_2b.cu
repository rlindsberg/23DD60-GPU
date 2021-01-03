
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TPB 128
#define N 100000000
#define ITERATION 10
#define PRIME_NUMBER 97
#define ERROR 1e-5

// Particle object
struct Particle{
    float3 pos;
    float3 vel;
};

__host__ bool compare(Particle* par1, Particle* par2) {
    for (int i = 0; i < N; i++) {
        bool a = fabs(par1[i].pos.x - par2[i].pos.x) < ERROR;
        //printf("comapre %f %f %d\n", par1[i].pos.x, par2[i].pos.x,a);
        if (fabs(par1[i].pos.x - par2[i].pos.x) > ERROR || fabs(par1[i].pos.y - par2[i].pos.y) > ERROR || fabs(par1[i].pos.z - par2[i].pos.z) > ERROR) {
            return false;
        }
    }
    return true;
}

// Random between 1 and -1
void randomParticle(Particle* par) {
    srand(time(NULL));
    for (int i = 0; i < N;i++) {
        par[i].pos.x = (float)(rand() / (float)RAND_MAX);
        par[i].pos.y = (float)(rand() / (float)RAND_MAX);
        par[i].pos.z = (float)(rand() / (float)RAND_MAX);
        par[i].vel.x = (float)(rand() / (float)RAND_MAX);
        par[i].vel.y = (float)(rand() / (float)RAND_MAX);
        par[i].vel.z = (float)(rand() / (float)RAND_MAX);
        //printf("init pos%f vel%f \n", par[i].pos.x, par[i].vel.x);
    }
}

__host__ __device__ float randomVelocity(int i, int j) {
    return (float)((i * j) % PRIME_NUMBER) / (float)PRIME_NUMBER;
}

void h_updateParticle(Particle* par) {
    for (int j = 0; j < ITERATION; j++) {
        for (int i = 0; i < N; i++) {
            par[i].pos.x += par[i].vel.x;
            par[i].pos.y += par[i].vel.y;
            par[i].pos.z += par[i].vel.z;
            //printf("CPU pos:%f vel:%f Idx:%d ITER:%d \n", par[i].pos.x, par[i].vel.x, i, j);
            par[i].vel.x = randomVelocity(i, j);
            par[i].vel.y = randomVelocity(i, j);
            par[i].vel.z = randomVelocity(i, j);
        }
    }
}

__host__ __device__ void printParticle(Particle* par) {
    for (int i = 0; i < N; i++) {
        printf("pos:%f vel:%f\n", par[i].pos.x, par[i].vel.x);
    }
}

__global__ void kernel(Particle* par, int iteration)
{
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    else
        //printf("adding %f and %f resulting in %f\n", par[i].pos.x, par[i].vel.x, par[i].pos.x + par[i].vel.x);
        par[i].pos.x += par[i].vel.x;
        par[i].pos.y += par[i].vel.y;
        par[i].pos.z += par[i].vel.z;
        //printf("GPU pos:%f vel:%f Idx:%d ITER:%d \n", par[i].pos.x, par[i].vel.x, i, iteration);
        par[i].vel.x = randomVelocity(i, iteration);
        par[i].vel.y = randomVelocity(i, iteration);
        par[i].vel.z = randomVelocity(i, iteration);
        
}

void checkCudaError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}


void g_updateParticle(Particle* par) {
    for (int j = 0; j < ITERATION; j++) {

        kernel <<<((N + TPB - 1) / TPB), TPB >>> (par, j);
        cudaDeviceSynchronize();
        checkCudaError();
    }
}


int main()
{
    Particle* par;
    cudaMallocManaged(&par, N * sizeof(Particle));
    randomParticle(par);

    g_updateParticle(par);
    
    cudaFree(par);

    return 0;
}
