/*
 * Stage 0 — sobel: Sobel 边缘检测
 *
 * 每个线程读自己及 8 个邻居像素，做 Sobel 卷积
 *
 * 核心概念:
 *   Stencil 模式: 线程访问邻居数据
 *   边界处理: clamp 到图像边缘
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "check.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

const int W = 1024, H = 1024;

__global__ void sobel(const unsigned char *src, unsigned char *dst, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  // clamp-to-edge helper
  auto at = [=](int cx, int cy) -> int {
    cx = min(max(cx, 0), w - 1);
    cy = min(max(cy, 0), h - 1);
    return (int)src[cy * w + cx];
  };

  int gx = -at(x - 1, y - 1) + at(x + 1, y - 1)
           - 2 * at(x - 1, y) + 2 * at(x + 1, y)
           - at(x - 1, y + 1) + at(x + 1, y + 1);

  int gy = -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1)
           + at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1);

  int mag = (int)sqrtf((float)(gx * gx + gy * gy));
  dst[y * w + x] = (unsigned char)min(mag, 255);
}

int main() {
  size_t size = W * H * sizeof(unsigned char);

  // generate a synthetic test pattern (circle + rectangle)
  unsigned char *h_src = new unsigned char[W * H];
  unsigned char *h_dst = new unsigned char[W * H];
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      // white circle on gray background
      float cx = x - W / 2.0f, cy = y - H / 2.0f;
      bool in_circle = (cx * cx + cy * cy) < (200 * 200);
      // white rectangle
      bool in_rect =
          (x > 600 && x < 850 && y > 200 && y < 600);
      h_src[y * W + x] = (in_circle || in_rect) ? 255 : 64;
    }
  }

  unsigned char *d_src, *d_dst;
  CUDA_CHECK(cudaMalloc(&d_src, size));
  CUDA_CHECK(cudaMalloc(&d_dst, size));
  CUDA_CHECK(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((W + 15) / 16, (H + 15) / 16);
  sobel<<<blocks, threads>>>(d_src, d_dst, W, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost));

  stbi_write_png("sobel_input.png", W, H, 1, h_src, W);
  stbi_write_png("sobel_output.png", W, H, 1, h_dst, W);
  printf("sobel: saved sobel_input.png, sobel_output.png (%dx%d)\n", W, H);

  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  delete[] h_src;
  delete[] h_dst;
  return EXIT_SUCCESS;
}
