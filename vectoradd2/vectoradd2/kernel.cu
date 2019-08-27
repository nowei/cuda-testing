#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
//#include "cuda.h"
//#include "cuda_runtime_api.h"
//#include "cuda_device_runtime_api.h"
// Each thread performs one pair-wise addition
__global__
void vecAddKernel(const float* A, const float* B, float* C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) C[i] = A[i] + B[i];
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n) {
	int size = n * sizeof(float);
	float* d_A, * d_B, * d_C;
	cudaError_t err;
	err = cudaMalloc((void**)& d_A, size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d \nYou can cry now", cudaGetErrorString(err), __FILE__, __LINE__);
		return;
	}
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)& d_B, size);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)& d_C, size);

	vecAddKernel <<<ceil(n / 256.0), 256 >>> (d_A, d_B, d_C, n);

	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}

int main(void) {
	int SIZE = 25600;
	float* a, * b, * c;
	a = (float*)malloc(SIZE * sizeof(float));
	b = (float*)malloc(SIZE * sizeof(float));
	c = (float*)malloc(SIZE * sizeof(float));
	for (int i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	vecAdd(a,b,c,SIZE);
	printf("c[5] = %f", c[5]);
	free(a);
	free(b);
	free(c);
}