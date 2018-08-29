
#include "cudamethods.hu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdarg.h>

CudaQueueElement *first = NULL;

__global__ void
initializeMemory(float *dst, int size, float constant_val, float scaling_val)
{
  int tid = threadIdx.x;
  int tidx = blockDim.x * blockIdx.x + tid;
  if (tidx < size) {
    dst[tidx] = constant_val + scaling_val * tidx;
  }
}

void
initializeCudaMatrix(CudaMatrix *matrix, float constant_val, float scaling_val)
{
  initializeMemory<<<matrix->size / 64.0 + 1, 64>>>(matrix->ptr, matrix->size, constant_val, scaling_val);
}

void
printCudaMatrix(CudaMatrix *matrix)
{
  int n = matrix->dimension_sizes[0];
  int m = matrix->num_dimensions == 2 ? matrix->dimension_sizes[1] : 1;
  float *tmp = (float*) malloc(matrix->size * sizeof *tmp);
  CUDA_ERR_CHECK(cudaMemcpy(tmp, matrix->ptr, matrix->size * sizeof *matrix->ptr, cudaMemcpyDeviceToHost));
  printf("Printing CudaMatrix %p with sizes %dx%d\n",matrix, n, m);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%.3f, ",tmp[i * n + j]);
    }
    printf("\n");
  }
  free(tmp);
}

void
addCudaElement(void *element, Destructor destructor_pointer)
{
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
}

void
destroyCudaMatrix(void *ptr)
{
  CudaMatrix *matrix = (CudaMatrix*) ptr;
  CUDA_ERR_CHECK(cudaFree(matrix->ptr));
  free(matrix->dimension_sizes);
}

void
destroyCudaMemory(void *ptr)
{
  CUDA_ERR_CHECK(cudaFree(ptr));
}

CudaMatrix *
createNdCudaMatrix(int n, ...)
{
  va_list args;
  va_start(args, n);
  int *dims = (int*) malloc(n * sizeof *dims);
  for (int i = 0; i < n; i++) {
    dims[i] = va_arg(args, int);
  }
  va_end(args);
  CudaMatrix *ret = createCudaMatrix(n, dims);
  return ret;
}

CudaMatrix *
create1dCudaMatrix(int a)
{
  int dims = 1, *dim = (int*) malloc(dims * sizeof *dim);
  dim[0] = a;
  CudaMatrix *ret = createCudaMatrix(dims, dim);
  return ret;
}

CudaMatrix *
create2dCudaMatrix(int a, int b)
{
  int dims = 2, *dim = (int*) malloc(dims * sizeof *dim);
  dim[0] = a;
  dim[1] = b;
  CudaMatrix *ret = createCudaMatrix(dims, dim);
  return ret;
}

CudaMatrix *
create3dCudaMatrix(int a, int b, int c)
{
  int dims = 3, *dim = (int*) malloc(dims * sizeof *dim);
  dim[0] = a;
  dim[1] = b;
  dim[2] = c;
  CudaMatrix *ret = createCudaMatrix(dims, dim);
  return ret;
}

CudaMatrix *
create4dCudaMatrix(int a, int b, int c, int d)
{
  int dims = 4, *dim = (int*) malloc(dims * sizeof *dim);
  dim[0] = a;
  dim[1] = b;
  dim[2] = c;
  dim[3] = d;
  CudaMatrix *ret = createCudaMatrix(dims, dim);
  return ret;
}

// This method steals the dimension_sizes pointer from caller.
CudaMatrix *
createCudaMatrix(int num_dimensions, int *dimension_sizes)
{
  CudaMatrix *ret;
  int n = 1;
  ret = (CudaMatrix *) malloc(sizeof *ret);
  ret->num_dimensions = num_dimensions;
  ret->dimension_sizes = dimension_sizes;
  for (int i = 0; i < num_dimensions; i++) {
    n *= dimension_sizes[i];
  }
  ret->size = n;
  CUDA_ERR_CHECK(cudaMalloc(&ret->ptr, n * sizeof *ret->ptr));
  addCudaElement(ret, &destroyCudaMatrix);
  return ret;
}

void *
createCudaMemory(size_t size)
{
  void *ptr;
  CUDA_ERR_CHECK(cudaMalloc(&ptr, size));
  addCudaElement(ptr, &destroyCudaMemory);
  return ptr;
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
