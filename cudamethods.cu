
#include "cudamethods.hu"
#include <cuda_runtime.h>
#include <stdio.h>

CudaQueueElement *first = NULL;

__global__ void
initializeMemory(float *dst, int size, float val = 0)
{
  int tid = threadIdx.x;
  int tidx = blockDim.x * blockIdx.x + tid;
  if (tidx < size) {
    dst[tidx] = val;
  }
}

void *
getCudaPointer(size_t size)
{
  void *ret;
  CUDA_ERR_CHECK(cudaMalloc(&ret, size));
  CudaQueueElement *new_element = (CudaQueueElement *) malloc(sizeof(CudaQueueElement));
  new_element->next = NULL;
  new_element->ptr = ret;
  if (first == NULL) first = new_element;
  else {
    CudaQueueElement *current = first;
    while(current->next != NULL) {
      current = current->next;
    }
    current->next = new_element;
  }
  return ret;
}

void
destroyCudaPointers(void)
{
  CudaQueueElement *current = first, *next;
  while (current != NULL) {
    next = current->next;
    CUDA_ERR_CHECK(cudaFree(current->ptr));
    free(current);
    current = next;
  }
}
