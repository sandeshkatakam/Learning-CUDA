// CUDA Program to Calculate Vector Addition for Two Vectors
// Author: Sandesh Katakam

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CHECK_ERROR(call) { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
exit(err); \
} \
}

// compute vector sum C = A+B
// each thread performs one pair-wise addition
__global__ // executed on the device, only callable from the host
void vecAddKernel(float *A, float *B, float *C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

__global__
void VecAdd(float* A_h, float* B_h, float* C_h, int n){

    int size = n*sizeof(n);

    float *A_d, float *B_d, float* C_d;

    // Allocate the Variables on the Device (Device Variables)
    CHECK_ERROR(cudaMalloc(void**)&A_d, size);
    CHECK_ERROR(cudaMalloc(void**)&B_d, size);
    CHECK_ERROR(cudaMalloc(void**)&C_d, size);
    
    // Copy Variables from host to Device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice);

    // Invoke the CUDA Kernel
    vecAddKernel<<<ceil(n/256.0), 256.0>>>(A_d, B_d, C_d, n);


    // Another way of Invoking the CUDA Kernel with Grids and Block size Specified
    	// Kernel invocation with 256 threads
	// dim3 dimGrid(ceil(n / 256.0),1,1);
	// dim3 dimBlock((256.0),1,1);

    
    // Copy C_d from device to host C_h
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

VecAdd_host(float* A_h, float* B_h, float* C_h, int n){
    for (i = 0; i<n; i++){
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main(){

    // Create host Vectors A_h, B_h and C_h
    A_h = (float*)malloc(sizeof(float)*n);
    B_h = (float*)malloc(sizeof(float)*n);
    C_h = (float*)malloc(sizeof(float)*n);

    // Fill the Host Vectors A_h and B_h with random numbers
    srand(time(NULL));
    for (int i =0; i <= n; i++){
		h_A[i] = ((((float)rand() / (float)(RAND_MAX)) * 100));
		h_B[i] = ((((float)rand() / (float)(RAND_MAX)) * 100));
    }

    // Call VecAdd CUDA Kernel (Invocation from the Main)
    VecAdd(A_h, B_h, C_h, n);

    // Check if the Result of the Vector Sum matches with that of CPU Kernel
    bool valid = true;
    for (int i = 0; i < n ; i++){
        if (A_h[i] + B_h[i] != C_h[i]){
            valid = false;
            break;
        }
    }

    if (!valid){
        printf("The Result is not Correct \n");
    }else{
        printf("The Sum Computed Matches and is Correct \n");
    }

    // Free Host Memory Variables
    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}