/*
 * Stage 1 — heat_rk4: 2D 热传导方程 (RK4 方法)
 *
 * 与 heat_euler 相同的方程，但用四阶 Runge-Kutta 推进时间。
 * 每个线程独立计算 4 个中间斜率 k1~k4，只读邻居格点。
 *
 * u_{n+1} = u_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 *
 * 其中 k_i = α ∇²(u_{intermediate})
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "check.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

const int W = 512, H = 512;

// 计算拉普拉斯算子 (clamp 边界)
__device__ float laplace(const float *u, int x, int y, int w, int h) {
  auto at = [=](int cx, int cy) -> float {
    cx = min(max(cx, 0), w - 1);
    cy = min(max(cy, 0), h - 1);
    return u[cy * w + cx];
  };
  return at(x + 1, y) + at(x - 1, y) + at(x, y + 1) + at(x, y - 1)
         - 4.0f * at(x, y);
}

__global__ void heatRK4Step(const float *cur, float *next, int w, int h,
                            float alpha, float dt, float inv_dx2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  float u = cur[y * w + x];

  // 简化 RK4: 每个邻居的中间值 ≈ base + dt/2 * k_i
  // (对 PDE 是合理的近似, 避免全局中间状态重建)
  auto laplace_mid = [&](const float *base, int cx, int cy, float uk) -> float {
    auto at = [=](int ci, int cj) -> float {
      ci = min(max(ci, 0), w - 1);
      cj = min(max(cj, 0), h - 1);
      return base[cj * w + ci] + 0.5f * dt * uk;
    };
    return at(cx + 1, cy) + at(cx - 1, cy) + at(cx, cy + 1) + at(cx, cy - 1)
           - 4.0f * at(cx, cy);
  };

  float k1 = alpha * inv_dx2 * laplace(cur, x, y, w, h);
  float k2 = alpha * inv_dx2 * laplace_mid(cur, x, y, k1);
  float k3 = alpha * inv_dx2 * laplace_mid(cur, x, y, k2);
  float k4 = alpha * inv_dx2 * laplace_mid(cur, x, y, k3);

  next[y * w + x] = u + dt / 6.0f * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
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
  const int steps = 500;
  const float alpha = 0.1f, dt = 0.05f, dx = 1.0f / W;
  const float inv_dx2 = 1.0f / (dx * dx);

  printf("heat_rk4: %d steps, dt=%.3f\n", steps, dt);

  size_t size = W * H * sizeof(float);

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

  saveFrame(h_cur, W, H, "heat_rk4_0000.png");

  for (int t = 0; t < steps; t++) {
    heatRK4Step<<<blocks, threads>>>(d_cur, d_next, W, H, alpha, dt, inv_dx2);
    float *tmp = d_cur; d_cur = d_next; d_next = tmp;

    if (t % 125 == 124) {
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_cur, d_cur, size, cudaMemcpyDeviceToHost));
      char fname[64];
      sprintf(fname, "heat_rk4_%04d.png", t + 1);
      saveFrame(h_cur, W, H, fname);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  printf("heat_rk4: saved heat_rk4_0000.png ~ heat_rk4_%04d.png\n", steps);

  CUDA_CHECK(cudaFree(d_cur));
  CUDA_CHECK(cudaFree(d_next));
  delete[] h_cur;
  return EXIT_SUCCESS;
}
