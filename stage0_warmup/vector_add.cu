/*
 * Stage 0 — vector_add: 一维向量加法
 *
 * 每个线程算一个元素: C[i] = A[i] + B[i]
 *
 * 核心概念:
 *   threadIdx / blockIdx / blockDim 的关系
 *   cudaMalloc / cudaMemcpy / cudaFree 流程
 */

#include "check.h"
#include <cstdio>
#include <cstdlib>

#define N (1 << 24)  // 16M elements

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  // host allocation
  float *h_a = new float[N];
  float *h_b = new float[N];
  float *h_c = new float[N];
  for (int i = 0; i < N; i++) {
    h_a[i] = (float)i;
    h_b[i] = (float)(i * 2);
  }

  // device allocation
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

  // launch
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // copy back & verify
  CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) errors++;
  }
  printf("vector_add: %d errors out of %d\n", errors, N);

  // cleanup
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
