// Note, does all the same multiplications, but does it a block [tile]
// at a time instead of all at once, i.e. loads two blocks, one from M
// and one from N and then adds all the calculations for each row * col
// dot product result and keeps doing that until we reach the end of each
// block, then we do it for each block. After that, we store the result
// into the respective element in P. 

// Note, how does this work for different dimensioned matrices?

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y; 
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0;
    // Loop over the M and N tiles required to compute the P element
    for (int p = 0; p < Width/TILE_WIDTH; ++p) { 
        // For each tile, respectively load in all of the parts of 
        // the tile so that you have two TILE_WIDTH * TILE_WIDTH
        // tiles. 

        // Collaborative loading of M and N tiles into shared memory
        // Thread loads two values
        ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];   // Note the Row
        ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*Width + Col]; // Note the Col
        __syncthreads();

        // Multiply all values along a row and respective column
        // and keep the amount stored. 
        for (int i = 0; i < TILE_WIDTH; ++i)
            // TILE_WIDTH multiplications 
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
        __synchthreads();
    }
    // By the end, we will have done this for each respective set of tiles
    P[Row*Width+Col] = Pvalue;
}