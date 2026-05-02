/*
 * Stage 0 — image_invert: 二维图像取反
 *
 * 每个线程处理一个像素: output = 255 - input
 *
 * 核心概念:
 *   二维 grid/block 的线程索引
 *   ix = blockIdx.x * blockDim.x + threadIdx.x
 *   iy = blockIdx.y * blockDim.y + threadIdx.y
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "check.h"
#include <cstdio>
#include <cstdlib>

const int W = 1024, H = 1024;

__global__ void invert(const unsigned char *src, unsigned char *dst, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    int idx = y * w + x;
    dst[idx] = 255 - src[idx];
  }
}

int main() {
  size_t size = W * H * sizeof(unsigned char);

  // generate a grayscale gradient on host
  unsigned char *h_src = new unsigned char[W * H];
  unsigned char *h_dst = new unsigned char[W * H];
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      h_src[y * W + x] = (unsigned char)((x + y) * 255 / (W + H));
    }
  }

  unsigned char *d_src, *d_dst;
  CUDA_CHECK(cudaMalloc(&d_src, size));
  CUDA_CHECK(cudaMalloc(&d_dst, size));
  CUDA_CHECK(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((W + 15) / 16, (H + 15) / 16);
  invert<<<blocks, threads>>>(d_src, d_dst, W, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost));

  stbi_write_png("image_invert.png", W, H, 1, h_dst, W);
  printf("image_invert: saved image_invert.png (%dx%d)\n", W, H);

  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  delete[] h_src;
  delete[] h_dst;
  return EXIT_SUCCESS;
}
