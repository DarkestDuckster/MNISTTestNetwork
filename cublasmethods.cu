
#include "cublasmethods.hu"
#include "cudamethods.hu"
#include <stdio.h>

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

__global__ void
naive_softmax_cross_entropy(float *in, int size, float *out)
{

}

__global__ void
naive_sum(float *in, int size, float *out)
{
  const unsigned int tid = threadIdx.x;
  const unsigned int bid = blockIdx.x * blockDim.x + tid;
  extern __shared__ float sdata[];
  sdata[tid] = in[bid];
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;

    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

CudaMatrix *
getMatrixSum(CudaMatrix *input)
{
  CudaMatrix *ret = create1dCudaMatrix(1);
  naive_sum<<<input->size / 64.0 + 1, 64, 64 * sizeof(float)>>>(input->ptr, input->size, ret->ptr);
  return ret;
}

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
