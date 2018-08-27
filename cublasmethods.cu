
#include "cublasmethods.hu"

void
cublasDot(cublasHandle_t handle, void *A, void *B, void *out, int m, int n, int k)
{
  float alpha = 1.0, beta = 0.0;
  CUBLAS_ERR_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      &alpha,
      A, m,
      B, n,
      &beta,
      out, m
  ))
}
