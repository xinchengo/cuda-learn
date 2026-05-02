/*
 * Stage 1 — heat_euler: 2D 热传导方程 (显式 Euler)
 *
 * ∂u/∂t = α ∇²u
 *
 * 核心概念:
 *   每个线程 = 一个格点，每个 kernel launch = 一个时间步
 *   Ping-pong buffer: 两个 device 数组交替读写
 *   Stencil Laplacian: 读上下左右四个邻居
 *
 * 离散化:
 *   u_new[i,j] = u[i,j] + α*dt/h² * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "check.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

const int W = 512, H = 512;

__global__ void heatStep(const float *cur, float *next, int w, int h, float coeff) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  // clamp-to-edge boundary
  auto at = [=](int cx, int cy) -> float {
    cx = min(max(cx, 0), w - 1);
    cy = min(max(cy, 0), h - 1);
    return cur[cy * w + cx];
  };

  float laplacian = at(x + 1, y) + at(x - 1, y) + at(x, y + 1) + at(x, y - 1) - 4.0f * at(x, y);
  next[y * w + x] = cur[y * w + x] + coeff * laplacian;
}

void saveFrame(const float *h_f, int w, int h, const char *filename) {
  unsigned char *img = new unsigned char[w * h];
  for (int i = 0; i < w * h; i++) {
    float v = h_f[i] * 255.0f;
    img[i] = (unsigned char)min(max(v, 0.0f), 255.0f);
  }
  stbi_write_png(filename, w, h, 1, img, w);
  delete[] img;
}

int main() {
  // unit grid spacing: dx = 1
  // stability: alpha*dt <= 0.25
  const int steps = 2000;
  const float alpha = 2.0f, dt = 0.1f;
  const float coeff = alpha * dt;  // = 0.2, stable

  printf("heat_euler: %d steps, alpha=%.1f, dt=%.1f, coeff=%.2f (stable if <= 0.25)\n",
         steps, alpha, dt, coeff);

  size_t size = W * H * sizeof(float);

  // initial condition: hot center, cold edges
  float *h_cur = new float[W * H];
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float cx = (x - W / 2.0f) / (W / 8.0f);
      float cy = (y - H / 2.0f) / (H / 8.0f);
      h_cur[y * W + x] = expf(-(cx * cx + cy * cy));
    }
  }

  float *d_cur, *d_next;
  CUDA_CHECK(cudaMalloc(&d_cur, size));
  CUDA_CHECK(cudaMalloc(&d_next, size));
  CUDA_CHECK(cudaMemcpy(d_cur, h_cur, size, cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((W + 15) / 16, (H + 15) / 16);

  saveFrame(h_cur, W, H, "heat_euler_0000.png");

  for (int t = 0; t < steps; t++) {
    heatStep<<<blocks, threads>>>(d_cur, d_next, W, H, coeff);
    // swap buffers: next becomes cur for the next step
    float *tmp = d_cur; d_cur = d_next; d_next = tmp;

    if (t % 500 == 499) {
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_cur, d_cur, size, cudaMemcpyDeviceToHost));
      char fname[64];
      sprintf(fname, "heat_euler_%04d.png", t + 1);
      saveFrame(h_cur, W, H, fname);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  printf("heat_euler: saved heat_euler_0000.png ~ heat_euler_%04d.png\n", steps);

  CUDA_CHECK(cudaFree(d_cur));
  CUDA_CHECK(cudaFree(d_next));
  delete[] h_cur;
  return EXIT_SUCCESS;
}
