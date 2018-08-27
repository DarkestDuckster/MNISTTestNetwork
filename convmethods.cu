
#include <stdio.h>
#include <cudnn.h>
#include "convmethods.hu"



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

  printf("%d Algorithms Returned\n",returned_algorithms);
  {
    int i;
    for (i = 0; i < returned_algorithms; i++) {
      if (performance_results[i].status != CUDNN_STATUS_SUCCESS) {
        printf("Algorithm%d Error: %s\n", i, cudnnGetErrorString(performance_results[i].status));
      }
      else {
        printf("Algorithm %d took %fs to finish\n", i, performance_results[i].time);
        printf("Required %ld bytes of workspace\n", performance_results[i].memory);
      }
    }
  }
  //network->algorithm = performance_results[0].algo;
  network->algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  CUDNN_ERR_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                  handle,
                  network->in,
                  network->filter,
                  network->convolution,
                  network->out,
                  network->algorithm,
                  &network->workspace.size
  ));
  printf("Fastest algorithm requires %ld bytes of space\n", network->workspace.size);
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

void
forwardNetwork(ConvInfo *network, cudnnHandle_t handle, void *in, void *weights, void *bias, void *output)
{
  float alpha1 = 1.0, alpha2 = 0.0;
  CUDNN_ERR_CHECK(cudnnConvolutionBiasActivationForward(
      handle,
      &alpha1,
      network->in,
      in,
      network->filter,
      weights,
      network->convolution,
      network->algorithm,
      network->workspace.workspace_pointer,
      network->workspace.size,
      &alpha2,
      network->out,
      output,
      network->bias,
      bias,
      network->activation,
      network->out,
      output
    ));
}

void
forwardPool(PoolInfo *pool, cudnnHandle_t handle, void *in, void *out)
{
  float alpha = 1.0, beta = 0.0;
  CUDNN_ERR_CHECK(cudnnPoolingForward(
      handle,
      pool->pooling,
      &alpha,
      pool->in,
      in,
      &beta,
      pool->out,
      out
    ));
}

PoolInfo *
setupPoolInfo(int channels, int input_width, int input_height, int batch_size, int pool_size, int padding_size, int strides)
{
  if (padding_size != 0) printf("Still not accounting for padding sizes!\n");
  PoolInfo *ret = (PoolInfo *) malloc(sizeof(PoolInfo));
  CUDNN_ERR_CHECK(cudnnCreatePoolingDescriptor(&ret->pooling));
  CUDNN_ERR_CHECK(cudnnSetPooling2dDescriptor(
                  ret->pooling,
                  CUDNN_POOLING_MAX,
                  CUDNN_NOT_PROPAGATE_NAN,
                  pool_size, pool_size,
                  padding_size, padding_size,
                  strides, strides
  ));
  CUDNN_ERR_CHECK(cudnnCreateTensorDescriptor(&ret->in));
  CUDNN_ERR_CHECK(cudnnSetTensor4dDescriptor(
                  ret->in,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT,
                  batch_size,
                  channels,
                  input_height,
                  input_width
  ));
  CUDNN_ERR_CHECK(cudnnCreateTensorDescriptor(&ret->out));
  CUDNN_ERR_CHECK(cudnnSetTensor4dDescriptor(
                  ret->out,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT,
                  batch_size,
                  channels,
                  ceil((input_height - pool_size - 1) / strides),
                  ceil((input_width - pool_size - 1) / strides)
  ));
  return ret;
}

ConvInfo *
setupNetworkInfo(int in_channels, int out_channels, int image_width, int image_height,
                      int batch_size, int kernel_size, int padding_size, int stride)
{
  if (stride != 1) printf("Warning! Stride not implemented.\n");
  ConvInfo *ret = (ConvInfo *) malloc(sizeof(ConvInfo));
  CUDNN_ERR_CHECK(cudnnCreateTensorDescriptor(&ret->in));
  CUDNN_ERR_CHECK(cudnnSetTensor4dDescriptor(
                  ret->in,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT,
                  batch_size,
                  in_channels,
                  image_height,
                  image_width
  ));
  CUDNN_ERR_CHECK(cudnnCreateTensorDescriptor(&ret->out));
  CUDNN_ERR_CHECK(cudnnSetTensor4dDescriptor(
                  ret->out,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT,
                  batch_size,
                  out_channels,
                  image_height,
                  image_width
  ));
  CUDNN_ERR_CHECK(cudnnCreateFilterDescriptor(&ret->filter));
  CUDNN_ERR_CHECK(cudnnSetFilter4dDescriptor(
                  ret->filter,
                  CUDNN_DATA_FLOAT,
                  CUDNN_TENSOR_NCHW,
                  out_channels,
                  in_channels,
                  kernel_size,
                  kernel_size
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
  CUDNN_ERR_CHECK(cudnnCreateTensorDescriptor(&ret->bias));
  CUDNN_ERR_CHECK(cudnnSetTensor4dDescriptor(
                  ret->bias,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_DATA_FLOAT,
                  1,
                  out_channels,
                  1,
                  1
  ));
  CUDNN_ERR_CHECK(cudnnCreateActivationDescriptor(&ret->activation));
  CUDNN_ERR_CHECK(cudnnSetActivationDescriptor(
                  ret->activation,
                  CUDNN_ACTIVATION_RELU,
                  CUDNN_NOT_PROPAGATE_NAN,
                  0
  ));
  return ret;
}
