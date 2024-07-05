#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>

// Kernel function to add the elements of two arrays
__global__ void sum_arrays_gpu(int* a, int* b, int* c, int size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size)
    {
        c[index] = a[index] + b[index];
    }
}

// CPU function to add the elements of two arrays
void sum_arrays_cpu(int* a, int* b, int* c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// Declaration of compare_arrays function (should be defined in common.h or another source file)
void compare_arrays(int* arr1, int* arr2, int size);

int main()
{
    int size = 1000;
    int block_size = 128;
    int NO_BYTE = size * sizeof(int);

    int* h_a = (int*)malloc(NO_BYTE);
    int* h_b = (int*)malloc(NO_BYTE);
    int* h_c = (int*)malloc(NO_BYTE);
    int* gpu_results = (int*)malloc(NO_BYTE);

    // Initialize random seed
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; i++)
    {
        h_a[i] = (int)(rand() & 0xFF);
        h_b[i] = (int)(rand() & 0xFF);
    }

    // Compute sum on CPU
    sum_arrays_cpu(h_a, h_b, h_c, size);

    // Zero out the results array for GPU
    memset(gpu_results, 0, NO_BYTE);

    // Allocate memory on the device
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, NO_BYTE);
    cudaMalloc((void**)&d_b, NO_BYTE);
    cudaMalloc((void**)&d_c, NO_BYTE);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, NO_BYTE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NO_BYTE, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);
    sum_arrays_gpu <<<grid, block>>> (d_a, d_b, d_c, size);
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(gpu_results, d_c, NO_BYTE, cudaMemcpyDeviceToHost);

    // Compare the results
    compare_arrays(gpu_results, h_c, size);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(gpu_results);
    free(h_c);

    return 0;
}
