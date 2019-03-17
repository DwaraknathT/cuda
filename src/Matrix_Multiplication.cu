#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include"cuda_runtime.h"
#include"cuda.h"
#include "device_launch_parameters.h"
using namespace std;

#define blocksize = 16

//Multiplication Kernel 
__global__ void MatMulKernel(int *a, int *b, int *c, int m, int n, int k)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if ((row < m) && (col < n))
	{
		int pvalue = 0;
		for (int i = 0; i < n; ++i)
		{
			pvalue += a[row*k + i] * b[i*k + col];
		}
		c[row*k + col] = pvalue;
	}
}

int main(int argc, char const *argv)
{
	int m, n, k;
	//fix the seed 
	srand(1);
	cout << "enter m,n,k";
	cin >> m >> n >> k;
	int *h_a, *h_b, *h_c;
	cudaMallocHost((void**)&h_a, sizeof(int)*m*n);
	cudaMallocHost((void**)&h_b, sizeof(int)*m*n);
	cudaMallocHost((void**)&h_c, sizeof(int)*m*n);
	//initlialize a 
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			h_a[i*n + j] = rand() % 1024;
		}
	}
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			h_b[i*n + j] = rand() % 1024;
		}
	}
	float gpu_elapsed_time;
	//cuda events to count
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//allocate deice space 
	int *d_a, *d_b, *d_c;
	cudaMalloc((void**)&d_a, sizeof(int)*m*n);
	cudaMalloc((void**)&d_b, sizeof(int)*m*n);
	cudaMalloc((void**)&d_c, sizeof(int)*m*n);
	//Move data from host to device 
	cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, h_b, sizeof(int)*m*n, cudaMemcpyHostToDevice);
	//Launch the kernel 
	dim3 dimGrid((int)(m / 32), (int)(m / 32), 1);
	dim3 dimBlock(32, 32, 1);
	MatMulKernel << <dimGrid, dimBlock >> > (d_a, d_b, d_c, m, n, k);
	//copy the result back 
	cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	//terminate couting 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time, start, stop);
	printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time);
	//free the memory 
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);

	return 0;
}
