
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

void g_updateParticle(Particle* d_par, Particle* h_par) {
    for (int j = 0; j < ITERATION; j++) {
        cudaMemcpy(d_par, h_par, N * sizeof(Particle), cudaMemcpyHostToDevice);
        kernel <<<((N + TPB - 1) / TPB), TPB >>> (d_par, j);
        cudaDeviceSynchronize();
        cudaMemcpy(h_par, d_par, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    }
}

int main()
{
    // CPU and GPU init
    //Particle* h_par = (Particle*)malloc(N * sizeof(Particle));
    Particle* h_par = 0;
    cudaHostAlloc(&h_par, N * sizeof(Particle), cudaHostAllocDefault);
    randomParticle(h_par);

    Particle* d_par = 0;
    //Particle* d_res = (Particle*)malloc(N * sizeof(Particle));
    
    //clock_t begin = clock();
    cudaMalloc(&d_par, N * sizeof(Particle));
    //cudaMemcpy(d_par, h_par, N * sizeof(Particle), cudaMemcpyHostToDevice);
    //clock_t end = clock();
    //double time_spent_memcpy = (double)(end - begin) / CLOCKS_PER_SEC;

    // correct memcpy?
    /*cudaMemcpy(d_res, d_par, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    printf("cpu: %f gpu: %f\n", h_par[0].pos.x, d_res[0].pos.x);*/

    // CPU implementation
    /*
    begin = clock();
    h_updateParticle(h_par);
    end = clock();
    double time_spent_cpu = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("CPU run time:%f N: %d Iter: %d TPB: %d\n", time_spent_cpu, N, ITERATION, TPB);
    */
    
    // GPU implementation
    //begin = clock();

    g_updateParticle(d_par, h_par);
    //cudaMemcpy(d_res, d_par, N * sizeof(Particle), cudaMemcpyDeviceToHost);

    //end = clock();
    //double time_spent_gpu = (double)(end - begin) / CLOCKS_PER_SEC;
    //printf("GPU run time:%f N: %d Iter: %d TPB: %d\n", time_spent_gpu + time_spent_memcpy, N, ITERATION, TPB);

    // Compare results

    //bool result = compare(h_par, d_res);
    //printf("Same answer? %s\n", result ? "true" : "false");
    
    cudaFreeHost(h_par);
    //free(d_res);
    cudaFree(d_par);

    /* clock
    clock_t begin = clock();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    */
   

    return 0;
}
