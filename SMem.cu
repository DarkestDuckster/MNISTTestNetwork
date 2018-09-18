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
#define START_TIMING clock_t start = clock();
#define STOP_TIMING clock_t end = clock();
#define GET_TIME printf("Time taken was %f\n",(float)(end - start) / CLOCKS_PER_SEC);

int main(int argc, char **argv)
{

  // Wait for finish
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
  CUDA_ERR_CHECK(cudaPeekAtLastError());



  // Destroy cuda Components
  destroyCudaPointers();
  exit(EXIT_SUCCESS);
}
