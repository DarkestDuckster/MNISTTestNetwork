
#include "cublasmethods.hu"
#include "cudamethods.hu"
#include <stdio.h>

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

void
forwardDense(DenseInfo *dense, cublasHandle_t handle)
{
  float alpha = 1.0, beta = 0.0;
  CUBLAS_ERR_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      dense->weights->dimension_sizes[1],
      dense->input_matrix->dimension_sizes[0],
      dense->weights->dimension_sizes[0],
      &alpha,
      (float*)dense->weights->ptr,
      dense->weights->dimension_sizes[1],
      (float*)dense->input_matrix->ptr,
      dense->weights->dimension_sizes[0],
      &beta,
      (float*)dense->output_matrix->ptr,
      dense->output_matrix->dimension_sizes[1]
  ));
}

__device__ void
warpReduce(volatile float *sdata, int tid, int bid, int size)
{
  if (bid + 32 < size) sdata[tid] += sdata[tid + 32];
  if (bid + 16 < size) sdata[tid] += sdata[tid + 16];
  if (bid + 8 < size) sdata[tid] += sdata[tid + 8];
  if (bid + 4 < size) sdata[tid] += sdata[tid + 4];
  if (bid + 2 < size) sdata[tid] += sdata[tid + 2];
  if (bid + 1 < size) sdata[tid] += sdata[tid + 1];
}

__global__ void
naive_sum(float *input, int size, float *out)
{
  const unsigned int tid = threadIdx.x;
  const unsigned int bid = blockIdx.x * blockDim.x * 2 + tid;
  extern __shared__ float sdata[];
  if (!(bid < size)) return;
  sdata[tid] = input[bid];
  if (bid + blockDim.x < size) sdata[tid] += input[bid + blockDim.x];
  __syncthreads();
  for (unsigned int offset = blockDim.x/2; offset > 32; offset /= 2) {
    if (tid < offset && bid + offset < size) sdata[tid] += sdata[tid + offset];
    __syncthreads();
  }
  if (tid < 32) warpReduce(sdata, tid, bid, size);
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

CudaMatrix *
getMatrixSum(CudaMatrix *input)
{
  int block_siz = 128;
  int block_num = ceil(input->size / (block_siz * 2.0));
  int block_mem = block_siz * sizeof(float);
  CudaMatrix *ret = create1dCudaMatrix(block_num);
  naive_sum<<<block_num, block_siz, block_mem>>>(input->ptr, input->size, ret->ptr);
  while (ret->size > 1) {
    printf("Reduced problem to size: %d\n",ret->size);
    block_num = ceil(ret->size / (block_siz * 2.0));
    CudaMatrix *tmp = create1dCudaMatrix(block_num);
    naive_sum<<<block_num, block_siz, block_mem>>>(ret->ptr, ret->size, tmp->ptr);
    ret = tmp;
    cudaDeviceSynchronize();
  }
  return ret;
}

/* If Python really had a garbage collector, it would have collected my code by now. */

DenseInfo *
setupDenseInfo(CudaMatrix *input, int output_size)
{
  CudaMatrix *output;
  CudaMatrix *weights, *biases;
  DenseInfo *ret = (DenseInfo*) malloc(sizeof *ret);
  int input_size = 1;
  int i;
  for (i = 1; i < input->num_dimensions; i++) {
    input_size *= input->dimension_sizes[i];
  }

  weights = create2dCudaMatrix(input_size, output_size);
  biases = create2dCudaMatrix(1, output_size);
  output = create2dCudaMatrix(input->dimension_sizes[0], weights->dimension_sizes[1]);

  ret->input_matrix = input;
  ret->output_matrix = output;
  ret->weights = weights;
  ret->biases = biases;
  return ret;
}
