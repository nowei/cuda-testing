/*
 * Initial analysis: This is good if it is faster to load values separately,
 * since each thread is tasked with loading one value. The overall multiplication
 * will be the same. 
 */

// Goals is to not leave threads idling, i.e. they can't do anything while
// waiting for other threads to do things; depends on configuration of the 
// gpus to see if threads are idling with these settings
__global__ void MatrixMulKernel2(float* M, float* N, float* P, int Width) {
    __shared__ float ds_M[BLOCK_WIDTH][BLOCK_WIDTH/2];
    __shared__ float ds_N[BLOCK_WIDTH/2][BLOCK_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0;
    // Loop over the M and N tiles required to compute the P element
    // Have to load separately, so need to go two times over.
    for (int p = 0; p < 2*Width/BLOCK_WIDTH; ++p) {
        // Collaborative loading of M and N tiles into shared memory
        if ( tx < BLOCK_WIDTH/2) {
            // Load each long up-down tile
            ds_M[ty][tx] = M[Row*Width + p*BLOCK_WIDTH/2+tx];
        } else {
            // Loads each long left-right tile
            // - tx - BLOCK_WIDTH/2 because we must load the other half 
            // - p*BLOCK_WIDTH/2+(tx-BLOCK_WIDTH/2)*width is (how many blocks down + 
            //   how many single rows down) * width
            ds_N[tx - BLOCK_WIDTH/2][ty] = N[(p*BLOCK_WIDTH/2+(tx-BLOCK_WIDTH/2))*Width + bx * blockDim.x + ty];
        }
        __syncthreads(); // Makes it so everything is ready at a time
        // Depends on how many threads there are. 

        for (int i = 0; i < BLOCK_WIDTH/2; ++i) 
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
        __synchthreads();
    }
    P[Row*Width+Col] = Pvalue;
}