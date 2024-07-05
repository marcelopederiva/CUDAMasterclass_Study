#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void print_details()
{   

    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n ", 
        threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main()
{  
    // int nx, ny, nz;
    // nx = 8;
    // ny = 8;
    // nz = 8;
    dim3 block(2,2,2);
    dim3 grid(4, 4, 4);

    print_details <<<grid, block>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}