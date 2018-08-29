#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdint.h>
#include "fileloader.hu"
#include "convmethods.hu"
#include "cublasmethods.hu"
#include "cudamethods.hu"
#include <time.h>

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_CHANNELS 1
#define IMAGE_SIZE ((IMAGE_WIDTH * IMAGE_HEIGHT) * sizeof(float))
#define NUM_LABELS 10
#define LABEL_SIZE (NUM_LABELS * sizeof(float))
#define BATCH_SIZE 64


int main(int argc, char **argv)
{
  int little_endian = isLittleEndian();
  if (little_endian) printf("This machine uses little endian representation\n");
  else printf("This machine uses big endian representation\n");
  //float *images;
  //float *labels;
  //images = (float *)readFile(TEST_IMAGES);
  //labels = (float *)readFile(TEST_LABELS);

  cudaStream_t computeStream;
  cudaStreamCreate(&computeStream);
  //int network_kernel_size = 5, network_padding_size = 2, network_stride_size = 1;
  //int pool_filter_size = 2, pool_padding_size = 0, pool_stride_size = 2;
  //int first_network_channels = 32, second_network_channels = 64;
  //int dense_output_size = 1024, logits_output_size = 10;

  cudnnHandle_t cudnnHandle;
  cudnnCreate(&cudnnHandle);
  cudnnSetStream(cudnnHandle, computeStream);
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  cublasSetStream(cublasHandle, computeStream);


  //CudaMatrix *matrix = create4dCudaMatrix(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
  //CUDA_ERR_CHECK(cudaMemcpyAsync(matrix->ptr, images, matrix->size, cudaMemcpyHostToDevice, computeStream));
  //ConvInfo *first_network = setupNetworkInfo(matrix, first_network_channels,
  //                                              network_kernel_size, network_padding_size, network_stride_size);
  //PoolInfo *first_pool = setupPoolInfo(first_network->output_matrix, pool_filter_size, pool_padding_size, pool_stride_size);
  //ConvInfo *second_network = setupNetworkInfo(first_pool->output_matrix, second_network_channels,
  //                                              network_kernel_size, network_padding_size, network_stride_size);
  //PoolInfo *second_pool = setupPoolInfo(second_network->output_matrix, pool_filter_size, pool_padding_size, pool_stride_size);
  //DenseInfo *first_dense = setupDenseInfo(second_pool->output_matrix, dense_output_size);
  //DenseInfo *second_dense = setupDenseInfo(first_dense->output_matrix, logits_output_size);

  //findBestAlgorithm(first_network, cudnnHandle, 3);
  //findBestAlgorithm(second_network, cudnnHandle, 3);
  //CUDA_ERR_CHECK(cudaDeviceSynchronize());

  CudaMatrix *A = create2dCudaMatrix(4, 2);
  initializeCudaMatrix(A, 1);
  CudaMatrix *B = getMatrixSum(A);
  printCudaMatrix(A);
  printCudaMatrix(B);





  // Wait for finish
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
  CUDA_ERR_CHECK(cudaPeekAtLastError());


  // Destroy cuda Components
  destroyCudaPointers();
  exit(EXIT_SUCCESS);
}
