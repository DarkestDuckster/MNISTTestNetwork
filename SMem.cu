#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdint.h>
#include "fileloader.hu"
#include "convolutional/crossentropy.hu"
#include "cublas/dense.hu"
#include "cudamethods.hu"
#include <time.h>

#define START_TIMING clock_t start = clock();
#define STOP_TIMING clock_t end = clock();
#define GET_TIME printf("Time taken was %f\n",(float)(end - start) / CLOCKS_PER_SEC);

int main(int argc, char **argv)
{
  cublasHandle_t handle_b;
  cublasCreate(&handle_b);
  cudnnHandle_t handle_e;
  cudnnCreate(&handle_e);


  CudaMatrix *A = create2dCudaMatrix(2,3);
  int *B;
  CUDA_ERR_CHECK(cudaMalloc(&B, 4 * sizeof *B));
  int b[2][2] = {{0, 1},{1, 0}};
  CUDA_ERR_CHECK(cudaMemcpy(B, b, 4 * sizeof(int), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  initializeCudaMatrix(A, 0, 1);
  printCudaMatrix(A);
  DenseInfo *dense = setupDenseInfo(A, 2);
  initializeCudaMatrix(dense->weights, 1);
  initializeCudaMatrix(dense->biases, 0);
  CrossEntropyInfo *loss = setupCrossEntropyInfo(dense->output_matrix);
  cudaDeviceSynchronize();

  forwardDense(dense, handle_b);
  printCudaMatrix(dense->output_matrix);
  backwardDense(dense, handle_b);
  printCudaMatrix(dense->backward_matrix);
  //forwardCrossEntropy(loss, handle_e);
  //cudaDeviceSynchronize();
  //backwardCrossEntropy(loss, B);
  //cudaDeviceSynchronize();
  //printCudaMatrix(loss->backward_matrix);




  // Wait for finish
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
  CUDA_ERR_CHECK(cudaPeekAtLastError());



  // Destroy cuda Components
  destroyCudaPointers();
  exit(EXIT_SUCCESS);
}
