#include<iostream>
#include<cuda.h>

using namespace std;

//kernel 
__global__ vecAdd(float *a, float *b, float c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        c[i] = a[i] + b[i];
    }
}
void vecAdd(float *a, float *b, float *c, int n)
{
    //allocate mem and move data 
    float *d_a, *d_b, *d_c;
    int size = n*sizeof(float);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);

    //move data
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    //launch the kernel 
    vecAdd<<<ceil(n/256.0), 256>>> (d_a, d_b, d_c);
    //copy answer back 
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    //free memeory 
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

