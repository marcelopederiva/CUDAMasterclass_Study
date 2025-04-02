#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"
#include <stdio.h>

//for random intialize
#include <stdlib.h>
#include <time.h>

//for memset
#include <cstring>

#include "common.h"
#include <vector>
#include <iomanip> 

__global__ void sum_arrays_gpu(int * a, int * b, int* c, int * d, int size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < size)
		d[index] = a[index] + b[index] + c[index];
}

void sum_arrays_cpu(int * a, int * b, int * c, int * d, int size)
{
	for (int i = 0; i < size; i++)
	{
		d[i] = a[i] + b[i] + c[i];
	}
}

int main(){
    int size = 1 << 22;
	
    //number of bytes needed to hold element count
	size_t NO_BYTES = size * sizeof(int);

    // host pointers
	int *h_a, *h_b, *h_c, *gpu_result, *cpu_result;
    //allocate memory for host size pointers
	h_a = (int *)malloc(NO_BYTES);
	h_b = (int *)malloc(NO_BYTES);
    h_c = (int *)malloc(NO_BYTES);
	cpu_result = (int *)malloc(NO_BYTES);

    //initialize h_a and h_b arrays randomly
	time_t t;
	srand((unsigned)time(&t));

    for (size_t i = 0; i < size; i++)
	{
		h_a[i] = (int)(rand() & 0xFF);
		//h_a[i] = 1;
	}
    for (size_t i = 0; i < size; i++)
	{
		h_b[i] = (int)(rand() & 0xFF);
		//h_a[i] = 1;
	}

    for (size_t i = 0; i < size; i++)
	{
		h_c[i] = (int)(rand() & 0xFF);
		//h_a[i] = 1;
	}

    // memset(gpu_result, 0, NO_BYTES);
	memset(cpu_result, 0, NO_BYTES);
    
    //summation in CPU
	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	sum_arrays_cpu(h_a, h_b, h_c, cpu_result, size);
	cpu_end = clock();
    std::cout << "CPU sum time: " << 1000.0*(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC)<< " ms\n";

    // GPU STEP
    gpu_result = (int *)malloc(NO_BYTES);

    int *d_a, *d_b, *d_c, *d_d;
	gpuErrchk(cudaMalloc((int **)&d_a, NO_BYTES));
	gpuErrchk(cudaMalloc((int **)&d_b, NO_BYTES));
	gpuErrchk(cudaMalloc((int **)&d_c, NO_BYTES));
    gpuErrchk(cudaMalloc((int **)&d_d, NO_BYTES));

    //execution time measuring in GPU
    std::vector<int> block_size = {64, 128, 256, 512};
    std::vector<clock_t> mem_htod_start(block_size.size()), mem_htod_end(block_size.size());
    std::vector<clock_t> mem_dtoh_start(block_size.size()), mem_dtoh_end(block_size.size());
    std::vector<clock_t> gpu_start(block_size.size()), gpu_end(block_size.size());
    
    for (size_t i = 0; i < block_size.size(); i++){

        mem_htod_start[i] = clock();
        gpuErrchk(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_c, h_c, NO_BYTES, cudaMemcpyHostToDevice));
        mem_htod_end[i] = clock();



        dim3 block(block_size[i]);
        dim3 grid((size / block.x) + 1);
        gpu_start[i] = clock();
        sum_arrays_gpu << <grid, block >> > (d_a, d_b, d_c, d_d, size);
        gpuErrchk(cudaDeviceSynchronize());
        gpu_end[i] = clock();


        mem_dtoh_start[i] = clock();
        gpuErrchk(cudaMemcpy(gpu_result, d_d, NO_BYTES, cudaMemcpyDeviceToHost));
        mem_dtoh_end[i] = clock();


    }
    
    printf("************************************\n");
    for (size_t i=0; i< block_size.size(); i++){
        std::cout << "Block: " << block_size[i] << "\n";
        std::cout << "Mem transfer host to device: " << std::right << std::setw(5) << std::fixed << std::setprecision(3) << (double)((double)1000.0*(mem_htod_end[i] - mem_htod_start[i]) / CLOCKS_PER_SEC) << " ms\n";

        std::cout << "GPU kernel execution time: " << std::right << std::setw(7) << std::fixed << std::setprecision(3) << (double)((double)1000.0*(gpu_end[i] - gpu_start[i]) / CLOCKS_PER_SEC) << " ms\n";
        std::cout << "Mem transfer device to host: " << std::right << std::setw(5) << std::fixed << std::setprecision(3) << (double)((double)1000.0*(mem_dtoh_end[i] - mem_dtoh_start[i]) / CLOCKS_PER_SEC) << " ms\n";

        std::cout << "Total GPU time: " << std::right << std::setw(18) << std::fixed << std::setprecision(3) << (double)((double)(1000.0*((mem_htod_end[i] - mem_htod_start[i])+ (gpu_end[i] - gpu_start[i])+ (mem_dtoh_end[i] - mem_dtoh_start[i]))) / CLOCKS_PER_SEC) << " ms\n";
        std::cout << "_______________________________________\n";
    }
    

    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_c));
    gpuErrchk(cudaFree(d_d));

    free(h_a);
    free(h_b);
    free(h_c);
    free(gpu_result);
}