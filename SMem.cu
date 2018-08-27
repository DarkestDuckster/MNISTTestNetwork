#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdint.h>
#include "fileloader.hu"
#include "convmethods.hu"
#include "cudamethods.hu"

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE ((IMAGE_WIDTH * IMAGE_HEIGHT) * sizeof(float))
#define NUM_LABELS 10
#define LABEL_SIZE (NUM_LABELS * sizeof(float))
#define BATCH_SIZE 64


int main(int argc, char **argv)
{
  int little_endian = isLittleEndian();
  if (little_endian) printf("This machine uses little endian representation\n");
  else printf("This machine uses big endian representation\n");
  float *labels;
  float *images;
  images = (float *)readFile(TEST_IMAGES);
  labels = (float *)readFile(TEST_LABELS);




  cudaStream_t memH2DStream, memD2HStream, computeStream;
  cudaStreamCreate(&memH2DStream);
  cudaStreamCreate(&memD2HStream);
  cudaStreamCreate(&computeStream);


  int kernel_size = 5, first_network_channels = 32, second_network_channels = 64;
  int pool_size = 2, pool_stride = 2;
  int first_dense_output_number = 1024;
  int first_network_weight_size = kernel_size * kernel_size * first_network_channels;
  int second_network_weight_size = kernel_size * kernel_size * second_network_channels;
  int first_pool_output_dimension = ((IMAGE_HEIGHT - (pool_size - 1)) / pool_stride + 1);
  int second_pool_output_dimension = ((first_pool_output_dimension - (pool_size - 1)) / pool_stride + 1);
  int first_pool_output_size = (BATCH_SIZE
                                * first_network_channels
                                * first_pool_output_dimension
                                * first_pool_output_dimension);
  int second_pool_output_size = (BATCH_SIZE
                                * second_network_channels
                                * second_pool_output_dimension
                                * second_pool_output_dimension);

  void *d_batch_images = getCudaPointer(IMAGE_SIZE * BATCH_SIZE), *d_batch_labels = getCudaPointer(LABEL_SIZE * BATCH_SIZE);
  void *d_first_network_output = getCudaPointer(IMAGE_SIZE * BATCH_SIZE * first_network_channels);
  void *d_first_network_weights = getCudaPointer(first_network_weight_size * sizeof(float));
  void *d_first_network_bias = getCudaPointer(first_network_channels * sizeof(float));
  void *d_first_pool_output = getCudaPointer(first_pool_output_size * sizeof(float));
  void *d_second_network_output = getCudaPointer(first_pool_output_size * sizeof(float));
  void *d_second_network_weights = getCudaPointer(second_network_weight_size * sizeof(float));
  void *d_second_network_bias = getCudaPointer(second_network_channels * sizeof(float));
  void *d_second_pool_output = getCudaPointer(second_pool_output_size * sizeof(float));

  cudaEvent_t memory_event;
  cudaEventCreate(&memory_event);

  ConvInfo *first_network = setupNetworkInfo(
                            1,          // In Channels
                            first_network_channels,         // Out Channels
                            IMAGE_WIDTH,
                            IMAGE_HEIGHT,
                            BATCH_SIZE,
                            kernel_size,          // Kernel Size
                            2,          // Padding Size
                            1);         // Stride Length
  ConvInfo *second_network = setupNetworkInfo(
                            first_network_channels,
                            second_network_channels,
                            first_pool_output_dimension,
                            first_pool_output_dimension,
                            BATCH_SIZE,
                            kernel_size,
                            2,
                            1);
  PoolInfo *first_pool = setupPoolInfo(
                            first_network_channels,
                            IMAGE_WIDTH,
                            IMAGE_HEIGHT,
                            BATCH_SIZE,
                            pool_size,
                            0,
                            pool_stride);
  PoolInfo *second_pool = setupPoolInfo(
                            second_network_channels,
                            first_pool_output_dimension,
                            first_pool_output_dimension,
                            BATCH_SIZE,
                            pool_size,
                            0,
                            pool_stride);
  cudnnHandle_t cudnnHandle;
  CUDNN_ERR_CHECK(cudnnCreate(&cudnnHandle));
  CUDNN_ERR_CHECK(cudnnSetStream(cudnnHandle, computeStream));
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  cublasSetStream(cublasHandle, computeStream);

  findBestAlgorithm(first_network, cudnnHandle, 3);
  first_network->workspace.workspace_pointer = getCudaPointer(first_network->workspace.size);
  findBestAlgorithm(second_network, cudnnHandle, 3);
  second_network->workspace.workspace_pointer = getCudaPointer(second_network->workspace.size);
  initializeMemory<<<first_network_weight_size / 256 + 1, 256>>>((float *)d_first_network_weights, first_network_weight_size, 0.01);
  initializeMemory<<<first_network_channels / 256 + 1, 256>>>((float *)d_first_network_bias, first_network_channels, 0);
  initializeMemory<<<second_network_weight_size / 256 + 1, 256>>>((float *)d_second_network_weights, second_network_weight_size, 0.01);
  initializeMemory<<<second_network_channels / 256 + 1, 256>>>((float *)d_second_network_bias, second_network_channels, 0);

  CUDA_ERR_CHECK(cudaMemcpyAsync(d_batch_images, images, IMAGE_SIZE * BATCH_SIZE, cudaMemcpyHostToDevice, memH2DStream));
  CUDA_ERR_CHECK(cudaMemcpyAsync(d_batch_labels, labels, LABEL_SIZE * BATCH_SIZE, cudaMemcpyHostToDevice, memH2DStream));
  CUDA_ERR_CHECK(cudaEventRecord(memory_event, memH2DStream));
  CUDA_ERR_CHECK(cudaStreamWaitEvent(computeStream, memory_event, 0));

  forwardNetwork(first_network, cudnnHandle, d_batch_images, d_first_network_weights, d_first_network_bias, d_first_network_output);
  forwardPool(first_pool, cudnnHandle, d_first_network_output, d_first_pool_output);
  forwardNetwork(second_network, cudnnHandle, d_first_pool_output, d_second_network_weights, d_second_network_bias, d_second_network_output);
  forwardPool(second_pool, cudnnHandle, d_second_network_output, d_second_pool_output);




  // Wait for finish
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
  CUDA_ERR_CHECK(cudaPeekAtLastError());



  // Destroy cudnn components
  CUDNN_ERR_CHECK(cudnnDestroy(cudnnHandle));
  destroyNetwork(first_network);



  // Destroy cuda Components
  CUDA_ERR_CHECK(cudaStreamDestroy(memH2DStream));
  CUDA_ERR_CHECK(cudaStreamDestroy(memD2HStream));
  CUDA_ERR_CHECK(cudaStreamDestroy(computeStream));
  destroyCudaPointers();
  exit(EXIT_SUCCESS);
}
