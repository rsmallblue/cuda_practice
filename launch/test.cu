#include <stdio.h>

__managed__ unsigned long long starttime;

__device__ unsigned long long globaltime(void)
{
    unsigned long long time;
    asm("mov.u64  %0, %%globaltimer;" : "=l"(time));
    return time;
}

__device__ unsigned int smid(void)
{
    unsigned int sm;
    asm("mov.u32  %0, %%smid;" : "=r"(sm));
    return sm;
}

__global__ void logkernel(void)
{
    unsigned long long t = globaltime();
    unsigned long long t0 = atomicCAS(&starttime, 0ull, t);
    if (t0==0) t0 = t;
    printf("Started block x: %2u y: %2u z: %2u on SM %2u at %llu.\n", blockIdx.x, blockIdx.y, blockIdx.z, smid(), t - t0);
}

int main(void)
{
    starttime = 0;
    dim3 GridDim(16, 80, 8);
    logkernel<<<GridDim, 1, 49152>>>();
    cudaDeviceSynchronize();

    return 0;
}