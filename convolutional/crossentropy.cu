#include "crossentropy.hu"

void
forwardCrossEntropy(CrossEntropyInfo *cross_entropy, cudnnHandle_t handle)
{
  float alpha = 1.0, beta = 0.0;
  CUDNN_ERR_CHECK(cudnnSoftmaxForward(
      handle,
      cross_entropy->softmax.algorithm,
      cross_entropy->softmax.mode,
      &alpha,
      cross_entropy->softmax.in,
      cross_entropy->input_matrix->ptr,
      &beta,
      cross_entropy->softmax.out,
      cross_entropy->output_matrix->ptr
  ))
}

__global__ void
naive_backward_cross_entropy(float *in, int *one_hot_classes, float batches, int size, float *out)
{
  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(bid < size)) return;
  out[bid] = (in[bid] - one_hot_classes[bid]) / batches;
}

void
backwardCrossEntropy(CrossEntropyInfo *cross_entropy, int *one_hot_classes)
{
  int block_siz = 128;
  int block_num = cross_entropy->output_matrix->size / block_siz + 1;
  naive_backward_cross_entropy<<<block_num, block_siz>>>
            (cross_entropy->output_matrix->ptr, one_hot_classes, cross_entropy->output_matrix->dimension_sizes[0],
              cross_entropy->output_matrix->size, cross_entropy->backward_matrix->ptr);
}

CrossEntropyInfo *
setupCrossEntropyInfo(CudaMatrix *input)
{
  CrossEntropyInfo *ret = (CrossEntropyInfo*) malloc(sizeof *ret);
  ret->softmax.mode = CUDNN_SOFTMAX_MODE_INSTANCE;
  ret->softmax.algorithm = CUDNN_SOFTMAX_FAST;
  setupTensorDescriptor(&ret->softmax.in, input->dimension_sizes[0], input->dimension_sizes[1],
                                          1,                         1);
  setupTensorDescriptor(&ret->softmax.out, input->dimension_sizes[0], input->dimension_sizes[1],
                                          1,                         1);
  ret->input_matrix = input;
  CudaMatrix *out = createCudaMatrixLike(input);
  ret->output_matrix = out;
  CudaMatrix *back = createCudaMatrixLike(input);
  ret->backward_matrix = back;
  return ret;
}
