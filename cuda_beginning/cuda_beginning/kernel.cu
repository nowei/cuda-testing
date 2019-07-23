// CUDACast #2
#include <stdio.h>

#define SIZE 1024

// Enable to launch on GPU
// tells compiler that function is going to be executed
// on the gpu and callable on the host
__global__ void VectorAdd(int* a, int* b, int* c, int n) {
	// Select the thread index to figure out how to index into vector 
	int i = threadIdx.x;
	if (i < n)
		c[i] = a[i] + b[i];
}

int main() {
	int* a, * b, * c;
	int* d_a, * d_b, * d_c;
	a = (int*)malloc(SIZE * sizeof(int));
	b = (int*)malloc(SIZE * sizeof(int));
	c = (int*)malloc(SIZE * sizeof(int));

	// Must allocate memory on GPU
	cudaMalloc(&d_a, SIZE * sizeof(int));
	cudaMalloc(&d_b, SIZE * sizeof(int));
	cudaMalloc(&d_c, SIZE * sizeof(int));
	
	for (int i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	// Copies values to GPU
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// Launch VectorAdd kernel on GPU
	// <<<BLOCKS, #Threads>>>
	VectorAdd<<< 1, SIZE >>>(d_a, d_b, d_c, SIZE);

	// Copy from GPU back to CPU
	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// Check results
	for (int i = 0; i < 10; i++)
		printf("c[%d] = %d\n", i, c[i]);
	
	// Free from CPU
	free(a);
	free(b);
	free(c);

	// Free from cuda
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}