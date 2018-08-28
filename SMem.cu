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

  


  // Wait for finish
  CUDA_ERR_CHECK(cudaDeviceSynchronize());
  CUDA_ERR_CHECK(cudaPeekAtLastError());


  // Destroy cuda Components
  destroyCudaPointers();
  exit(EXIT_SUCCESS);
}
