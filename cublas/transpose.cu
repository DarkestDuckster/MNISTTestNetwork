#include "transpose.hu"

#define TILE_DIM 32
#define BLOCK_HEIGHT 8

__global__ void
naive_matrix_transpose(float *input, int axis_0, int axis_1, float *output)
{
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int i = 0; i < TILE_DIM && y + i < axis_1 && x < axis_0; i += BLOCK_HEIGHT) {
    tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * axis_0 + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int i = 0; i < TILE_DIM && y + i < axis_1 && x < axis_0; i += BLOCK_HEIGHT) {
    output[(y + i) * axis_0 + x] = tile[(threadIdx.x)][threadIdx.y + i];
  }
}

CudaMatrix *
transposeCudaMatrix(CudaMatrix *matrix)
{
  int ax_0 = matrix->dimension_sizes[0];
  int ax_1 = matrix->dimension_sizes[1];

  dim3 block_siz (TILE_DIM, BLOCK_HEIGHT);
  dim3 block_num (ceil(ax_0 / ((float) TILE_DIM)), ceil(ax_1 / ((float) TILE_DIM)));


  CudaMatrix *ret = create2dCudaMatrix(ax_1, ax_0);
  naive_matrix_transpose<<<block_num, block_siz>>>(matrix->ptr, ax_0, ax_1, ret->ptr);
  return ret;
}
