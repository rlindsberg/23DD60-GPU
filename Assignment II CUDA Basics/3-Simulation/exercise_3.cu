#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TPB 16
#define N 2000
#define ITERATION 10000
#define PRIME_NUMBER 97
#define ERROR 1e-6

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
    //return (float)((i * j) % PRIME_NUMBER) / (float)PRIME_NUMBER;
    return 1.0;
}

void h_updateParticle(Particle* par) {
    for (int j = 0; j < ITERATION; j++) {
        for (int i = 0; i < N; i++) {
            par[i].pos.x += par[i].vel.x;
            par[i].pos.y += par[i].vel.y;
            par[i].pos.z += par[i].vel.z;

            par[i].vel.x = randomVelocity(i, j);
            par[i].vel.y = randomVelocity(i, j);
            par[i].vel.z = randomVelocity(i, j);
            //printf("CPU pos:%f vel:%f Idx:%d ITER:%d \n", par[i].pos.x, par[i].vel.x, i, j);
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
    else {
        //printf("adding %f and %f resulting in %f\n", par[i].pos.x, par[i].vel.x, par[i].pos.x += par[i].vel.x);
        par[i].pos.x += par[i].vel.x;
        par[i].pos.y += par[i].vel.y;
        par[i].pos.z += par[i].vel.z;

        par[i].vel.x = randomVelocity(i, iteration);
        par[i].vel.y = randomVelocity(i, iteration);
        par[i].vel.z = randomVelocity(i, iteration);
        //printf("GPU pos:%f vel:%f Idx:%d ITER:%d \n", par[i].pos.x, par[i].vel.x, i, iteration);
    }
}

void g_updateParticle(Particle* par) {
    for (int j = 0; j < ITERATION; j++) {

        kernel << <(N + TPB - 1) / TPB, TPB >> > (par, j);
        cudaDeviceSynchronize();
    }
}

int main()
{
    // CPU init
    Particle* h_par = (Particle*)malloc(N * sizeof(Particle));
    randomParticle(h_par);

    // CPU implementation
    clock_t begin = clock();
    h_updateParticle(h_par);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("CPU run time:%f N: %d Iter: %d TPB: %d\n", time_spent, N, ITERATION, TPB);

    // GPU init
    Particle* d_par = 0;
    Particle* d_res = (Particle*)malloc(N * sizeof(Particle));
    cudaMalloc(&d_par, N * sizeof(Particle));
    begin = clock();
    cudaMemcpy(d_par, h_par, N * sizeof(Particle), cudaMemcpyHostToDevice);

    // GPU implementation

    g_updateParticle(d_par);

    cudaMemcpy(d_res, d_par, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("GPU run time:%f N: %d Iter: %d TPB: %d\n", time_spent, N, ITERATION, TPB);

    // Compare results

    bool result = compare(h_par, d_res);
    printf("Same answer? %s\n", result ? "true" : "false");

    free(h_par);
    free(d_res);
    cudaFree(d_par);

    /* clock
    clock_t begin = clock();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    */


    return 0;
}
