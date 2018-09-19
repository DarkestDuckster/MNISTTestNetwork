#include "dense.hu"

__global__ void
naive_bias_add(float *in, int size, float *bias, int bias_size)
{
  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(bid < size)) return;
  int bias_offset = bid - (bid / bias_size) * bias_size;
  in[bid] += bias[bias_offset];
}

int
checkBiasShape(CudaMatrix *A, CudaMatrix *bias)
{
  if (!(A->num_dimensions == bias->num_dimensions && A->num_dimensions > 1)) return 0;
  if (!(bias->dimension_sizes[0] == 1)) return 0;
  for (int i = 1; i < A->num_dimensions; i++)
    if (!(A->dimension_sizes[i] == bias->dimension_sizes[i])) return 0;
  return 1;
}

void
addBias(CudaMatrix *input, CudaMatrix *bias)
{
  checkBiasShape(input, bias);
  int block_siz = 128;
  int block_num = input->size / block_siz + 1;
  naive_bias_add<<<block_num, block_siz>>>(input->ptr, input->size, bias->ptr, bias->size);
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
  addBias(dense->output_matrix, dense->biases);
}

void
backwardDense(DenseInfo *dense, cublasHandle_t handle)
{
  float alpha = 1.0, beta = 0.0;
  CUBLAS_ERR_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      dense->weights->dimension_sizes[0],
      dense->output_matrix->dimension_sizes[0],
      dense->weights->dimension_sizes[1],
      &alpha,
      (float*)dense->weights->ptr,
      dense->weights->dimension_sizes[0],
      (float*)dense->output_matrix->ptr,
      dense->weights->dimension_sizes[1],
      &beta,
      (float*)dense->backward_matrix->ptr,
      dense->backward_matrix->dimension_sizes[1]
  ));
}

DenseInfo *
setupDenseInfo(CudaMatrix *input, int output_size)
{
  CudaMatrix *output, *back;
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
  back = createCudaMatrixLike(input);

  ret->input_matrix = input;
  ret->output_matrix = output;
  ret->weights = weights;
  ret->biases = biases;
  ret->backward_matrix = back;
  return ret;
}
