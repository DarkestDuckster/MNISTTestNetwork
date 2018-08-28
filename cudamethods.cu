
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

void
addCudaElement(void *element, Destructor destructor_pointer)
{
  void *ret;
  CUDA_ERR_CHECK(cudaMalloc(&ret, size));
  CudaQueueElement *new_element = (CudaQueueElement *) malloc(sizeof(CudaQueueElement));
  new_element->next = NULL;
  if (first == NULL) first = new_element;
  else {
    CudaQueueElement *current = first;
    while(current->next != NULL) {
      current = current->next;
    }
    current->next = new_element;
  }
  new_element->element = element;
  new_element->destructor_pointer = destructor_pointer;

  return ret;
}

void
destroyCudaMatrix(void *ptr)
{
  CudaMatrix *matrix = (CudaMatrix*) ptr;
  CUDA_ERR_CHECK(cudaFree(matrix->ptr));
  free(matrix->dimension_sizes);
}

// This method steals the dimension_sizes pointer from caller.
CudaMatrix *
createCudaMatrix(int num_dimensions, int *dimension_sizes)
{
  CudaMatrix *ret;
  int i, n = 1;
  ret = (CudaMatrix *) malloc(sizeof *ret);
  ret.num_dimensions = num_dimensions;
  ret.dimension_sizes = dimension_sizes;
  for (i = 0; i < num_dimensions; i++) {
    n *= dimension_sizes[i];
  }
  ret.ptr = CUDA_ERR_CHECK(cudaMalloc(n * sizeof(float)));
  addCudaElement(ret, &destroyCudaMatrix);

}


void
destroyCudaPointers(void)
{
  CudaQueueElement *current = first, *next;
  while (current != NULL) {
    next = current->next;
    current->destructor_pointer(current->element);
    free(current);
    current = next;
  }
}
