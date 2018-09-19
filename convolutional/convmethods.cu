
#include <stdio.h>
#include "convmethods.hu"


void
setupTensorDescriptor(cudnnTensorDescriptor_t *tensor, int n, int c, int h, int w)
{
  CUDNN_ERR_CHECK(cudnnCreateTensorDescriptor(tensor));
  CUDNN_ERR_CHECK(cudnnSetTensor4dDescriptor(
                  *tensor,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT,
                  n, c,
                  h, w
  ));
}

void
findBestAlgorithm(ConvInfo *network, cudnnHandle_t handle, int algorithms_to_search)
{
  int algorithm_request_count = algorithms_to_search;
  int returned_algorithms = 0;
  cudnnConvolutionFwdAlgoPerf_t *performance_results = (cudnnConvolutionFwdAlgoPerf_t *)
                                      malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t) * algorithm_request_count);

  CUDNN_ERR_CHECK(cudnnFindConvolutionForwardAlgorithm(
                  handle,
                  network->in,
                  network->filter,
                  network->convolution,
                  network->out,
                  algorithm_request_count,
                  &returned_algorithms,
                  performance_results
  ));

  network->algorithm = performance_results[0].algo;
  CUDNN_ERR_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                  handle,
                  network->in,
                  network->filter,
                  network->convolution,
                  network->out,
                  network->algorithm,
                  &network->workspace.size
  ));
  network->workspace.workspace_pointer = createCudaMemory(network->workspace.size);
  free(performance_results);
}

void
destroyNetwork(ConvInfo *network)
{
  CUDNN_ERR_CHECK(cudnnDestroyTensorDescriptor(network->in));
  CUDNN_ERR_CHECK(cudnnDestroyTensorDescriptor(network->out));
  CUDNN_ERR_CHECK(cudnnDestroyFilterDescriptor(network->filter));
  CUDNN_ERR_CHECK(cudnnDestroyConvolutionDescriptor(network->convolution));
  CUDNN_ERR_CHECK(cudnnDestroyActivationDescriptor(network->activation));
  CUDNN_ERR_CHECK(cudnnDestroyTensorDescriptor(network->bias));
  free(network);
}

void
destroyPool(PoolInfo *pool)
{
  CUDNN_ERR_CHECK(cudnnDestroyTensorDescriptor(pool->in));
  CUDNN_ERR_CHECK(cudnnDestroyTensorDescriptor(pool->out));
  CUDNN_ERR_CHECK(cudnnDestroyPoolingDescriptor(pool->pooling));
}

cudnnStatus_t
forwardPool(PoolInfo *pool, cudnnHandle_t handle)
{
  float alpha = 1.0, beta = 0.0;
  return cudnnPoolingForward(
      handle,
      pool->pooling,
      &alpha,
      pool->in,
      pool->input_matrix->ptr,
      &beta,
      pool->out,
      pool->output_matrix->ptr
  );
}

cudnnStatus_t
forwardNetwork(ConvInfo *network, cudnnHandle_t handle)
{
  float alpha1 = 1.0, alpha2 = 0.0;
  return cudnnConvolutionBiasActivationForward(
      handle,
      &alpha1,
      network->in,
      network->input_matrix->ptr,
      network->filter,
      network->weights->ptr,
      network->convolution,
      network->algorithm,
      network->workspace.workspace_pointer,
      network->workspace.size,
      &alpha2,
      network->out,
      network->output_matrix->ptr,
      network->bias,
      network->biases->ptr,
      network->activation,
      network->out,
      network->output_matrix->ptr
  );
}

CudaMatrix *
getFilterOutput(CudaMatrix *in, int out_channels, int filter_size, int padding_size, int strides)
{
  int width = in->dimension_sizes[3], height = in->dimension_sizes[2];
  int new_width = ceil((width + padding_size * 2 - (filter_size - 1)) / strides);
  int new_height = ceil((height + padding_size * 2 - (filter_size - 1)) / strides);
  CudaMatrix *ret = create4dCudaMatrix(in->dimension_sizes[0], out_channels, new_width, new_height);
  return ret;
}


PoolInfo *
setupPoolInfo(CudaMatrix *input, int pool_size, int padding_size, int stride)
{
  CudaMatrix *output = getFilterOutput(input, input->dimension_sizes[1], pool_size, padding_size, stride);
  PoolInfo *ret = (PoolInfo*) malloc(sizeof *ret);
  ret->input_matrix = input;
  ret->output_matrix = output;
  CUDNN_ERR_CHECK(cudnnCreatePoolingDescriptor(&ret->pooling));
  CUDNN_ERR_CHECK(cudnnSetPooling2dDescriptor(
                  ret->pooling,
                  CUDNN_POOLING_MAX,
                  CUDNN_NOT_PROPAGATE_NAN,
                  pool_size, pool_size,
                  padding_size, padding_size,
                  stride, stride
  ));
  setupTensorDescriptor(&ret->in, input->dimension_sizes[0], input->dimension_sizes[1],
                                  input->dimension_sizes[2], input->dimension_sizes[3]);
  setupTensorDescriptor(&ret->out, output->dimension_sizes[0], output->dimension_sizes[1],
                                   output->dimension_sizes[2], output->dimension_sizes[3]);
  return ret;
}

ConvInfo *
setupNetworkInfo(CudaMatrix *input, int out_channels, int kernel_size, int padding_size, int stride)
{
  CudaMatrix *output = getFilterOutput(input, out_channels, kernel_size, padding_size, stride);
  CudaMatrix *weights = create4dCudaMatrix(output->dimension_sizes[1], input->dimension_sizes[1], kernel_size, kernel_size);
  CudaMatrix *biases = create4dCudaMatrix(1, output->dimension_sizes[1], 1, 1);
  ConvInfo *ret = (ConvInfo*) malloc(sizeof *ret);
  ret->input_matrix = input;
  ret->output_matrix = output;
  ret->weights = weights;
  ret->biases = biases;
  setupTensorDescriptor(&ret->in, input->dimension_sizes[0], input->dimension_sizes[1],
                                  input->dimension_sizes[2], input->dimension_sizes[3]);
  setupTensorDescriptor(&ret->out, output->dimension_sizes[0], output->dimension_sizes[1],
                                   output->dimension_sizes[2], output->dimension_sizes[3]);
  CUDNN_ERR_CHECK(cudnnCreateFilterDescriptor(&ret->filter));
  CUDNN_ERR_CHECK(cudnnSetFilter4dDescriptor(
                  ret->filter,
                  CUDNN_DATA_FLOAT,
                  CUDNN_TENSOR_NCHW,
                  weights->dimension_sizes[0],
                  weights->dimension_sizes[1],
                  weights->dimension_sizes[2],
                  weights->dimension_sizes[3]
  ));
  CUDNN_ERR_CHECK(cudnnCreateConvolutionDescriptor(&ret->convolution));
  CUDNN_ERR_CHECK(cudnnSetConvolution2dDescriptor(
                  ret->convolution,
                  padding_size, padding_size,
                  stride, stride,
                  1, 1,
                  CUDNN_CONVOLUTION,
                  CUDNN_DATA_FLOAT
  ));
  CUDNN_ERR_CHECK(cudnnSetConvolutionMathType(ret->convolution, CUDNN_TENSOR_OP_MATH));
  setupTensorDescriptor(&ret->bias, biases->dimension_sizes[0], biases->dimension_sizes[1],
                                      biases->dimension_sizes[2], biases->dimension_sizes[3]);
  CUDNN_ERR_CHECK(cudnnCreateActivationDescriptor(&ret->activation));
  CUDNN_ERR_CHECK(cudnnSetActivationDescriptor(
                  ret->activation,
                  CUDNN_ACTIVATION_RELU,
                  CUDNN_NOT_PROPAGATE_NAN,
                  0
  ));
  return ret;
}
