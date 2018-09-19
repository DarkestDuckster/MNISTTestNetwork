
#include "cublasmethods.hu"
#include "cudamethods.hu"
#include <stdio.h>

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
