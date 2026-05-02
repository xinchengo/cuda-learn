/*
 * Stage 1 — grayscott: Gray-Scott 反应-扩散模型
 *
 * ∂u/∂t = D_u ∇²u - u·v² + F·(1-u)
 * ∂v/∂t = D_v ∇²v + u·v² - (F+k)·v
 *
 * 双场耦合 PDE，4 种典型斑图:
 *   F=0.04,  k=0.06  → spots
 *   F=0.035, k=0.065 → stripes
 *   F=0.025, k=0.06  → mazes
 *   F=0.012, k=0.05  → solitons
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda/cmath>

// =================================================================
//  Laplacian helpers (device)
// =================================================================

// TODO: 实现 __device__ 函数 laplacian9  (9-point stencil, O(h⁴))
//       推导 ∇²f ≈ [4(N+S+E+W) + (NE+NW+SE+SW) - 20·C] / (6·h²)
//       但 Gray-Scott 通常 h=1, 梯度系数被吸收到 D 中，除以 dx² 即可

__device__ float laplacian5(const float *f, int idx, int W) {
  // TODO: 返回 5-point Laplacian (除以 h², 这里 h=1)
  //       即 f[idx-W] + f[idx+W] + f[idx-1] + f[idx+1] - 4*f[idx]
  return 0.0f;
}

__device__ float laplacian9(const float *f, int idx, int W) {
  // TODO: 返回 9-point Laplacian (除以 6·h²)
  //       [4*(f[idx-W]+f[idx+W]+f[idx-1]+f[idx+1])
  //        + (f[idx-W-1]+f[idx-W+1]+f[idx+W-1]+f[idx+W+1])
  //        - 20*f[idx]] / 6.0f
  return 0.0f;
}

// =================================================================
//  反应项 (device): 每个时间积分方法都需要重复计算
// =================================================================

struct Reaction {
  float du, dv;
};

__device__ Reaction reaction(float u, float v, float F, float k) {
  // TODO: 计算反应项
  //       du = -u*v*v + F*(1 - u)
  //       dv =  u*v*v - (F + k)*v
  return {0.0f, 0.0f};
}

// =================================================================
//  CUDA kernel — Euler forward (1 阶)
// =================================================================

__global__ void gsEuler(const float *u, const float *v,
                        float *u_new, float *v_new,
                        int W, int H, float dt,
                        float Du, float Dv, float F, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= W - 1 || j >= H - 1) return;
  int idx = j * W + i;

  // TODO: Euler 前向步
  //  1. 计算 Laplacian:  lap_u = laplacian5(u, idx, W)
  //                       lap_v = laplacian5(v, idx, W)
  //  2. 计算反应项:      auto r = reaction(u[idx], v[idx], F, k)
  //  3. 更新:           u_new[idx] = u[idx] + dt * (Du * lap_u + r.du)
  //                     v_new[idx] = v[idx] + dt * (Dv * lap_v + r.dv)
  //  4. Clamp:          u_new[idx] = fminf(fmaxf(u_new[idx], 0.0f), 1.0f)
  //                     v_new[idx] = fminf(fmaxf(v_new[idx], 0.0f), 1.0f)

  u_new[idx] = u[idx];
  v_new[idx] = v[idx];
}

// =================================================================
//  CUDA kernel — RK2 (中点法, 2 阶)
// =================================================================

__global__ void gsRK2(const float *u, const float *v,
                      float *u_new, float *v_new,
                      int W, int H, float dt,
                      float Du, float Dv, float F, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= W - 1 || j >= H - 1) return;
  int idx = j * W + i;

  float u0 = u[idx], v0 = v[idx];

  // TODO: RK2 中点法
  //  k1:
  //    lap_u = laplacian5(u, idx, W)
  //    lap_v = laplacian5(v, idx, W)
  //    auto r1 = reaction(u0, v0, F, k)
  //    k1_u = Du*lap_u + r1.du
  //    k1_v = Dv*lap_v + r1.dv
  //
  //  中点近似 (用邻居的中间值):
  //    对邻居使用 u + dt/2 * k1_?   近似
  //    或者直接在 local 计算: u_mid = u0 + dt/2 * k1_u
  //    lap_u2 ≈ laplacian5(u, idx, W)  // 简化: 复用原 Laplacian
  //    (实际 RK2 对 PDE 通常用邻居近似, 见 heat_rk4.cu 的做法)
  //    auto r2 = reaction(u0 + dt/2*k1_u, v0 + dt/2*k1_v, F, k)
  //    k2_u = Du*lap_u2 + r2.du
  //    k2_v = Dv*lap_v2 + r2.dv
  //
  //    u_new[idx] = u0 + dt * k2_u
  //    v_new[idx] = v0 + dt * k2_v

  u_new[idx] = u0;
  v_new[idx] = v0;
}

// =================================================================
//  CUDA kernel — RK4 (4 阶)
// =================================================================

__global__ void gsRK4(const float *u, const float *v,
                      float *u_new, float *v_new,
                      int W, int H, float dt,
                      float Du, float Dv, float F, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= W - 1 || j >= H - 1) return;
  int idx = j * W + i;

  float u0 = u[idx], v0 = v[idx];

  // TODO: RK4
  //  k1:
  //    lap_u = laplacian5(u, idx, W), lap_v = laplacian5(v, idx, W)
  //    auto r1 = reaction(u0, v0, F, k)
  //    k1_u = Du*lap_u + r1.du  ,  k1_v = Dv*lap_v + r1.dv
  //
  //  k2 (用 u0 + dt/2*k1_ ?, 并对邻居做类似近似):
  //    auto r2 = reaction(u0 + dt/2*k1_u, v0 + dt/2*k1_v, F, k)
  //    k2_u = Du*lap_u + r2.du  (简化: 复用 Laplacian)
  //    k2_v = Dv*lap_v + r2.dv
  //
  //  k3:
  //    auto r3 = reaction(u0 + dt/2*k2_u, v0 + dt/2*k2_v, F, k)
  //    k3_u = Du*lap_u + r3.du
  //    k3_v = Dv*lap_v + r3.dv
  //
  //  k4:
  //    auto r4 = reaction(u0 + dt*k3_u, v0 + dt*k3_v, F, k)
  //    k4_u = Du*lap_u + r4.du
  //    k4_v = Dv*lap_v + r4.dv
  //
  //  u_new[idx] = u0 + dt/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u)
  //  v_new[idx] = v0 + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)

  u_new[idx] = u0;
  v_new[idx] = v0;
}

// =================================================================
//  CUDA host helpers
// =================================================================

void gsInit(float *&d_u, float *&d_v, float *&d_u_new, float *&d_v_new,
            int W, int H, const float *h_u_init, const float *h_v_init) {
  int n = W * H;
  // TODO: 分配 4 块 device 内存并拷贝初始场
  // cudaMalloc + cudaMemcpy + cudaMemcpy  (u_new/v_new 用 memset 0)
  (void)d_u; (void)d_v; (void)d_u_new; (void)d_v_new;
  (void)n; (void)h_u_init; (void)h_v_init;
}

void gsStep(float *&d_u, float *&d_v, float *&d_u_new, float *&d_v_new,
            int W, int H, float dt, float Du, float Dv, float F, float k,
            int method) {
  dim3 blockSize(16, 16);
  dim3 gridSize(cuda::ceil_div(W - 2, blockSize.x),
                cuda::ceil_div(H - 2, blockSize.y));

  // TODO: 根据 method 调用不同的 kernel
  //   0 = Euler, 1 = RK2, 2 = RK4
  //   gsEuler/gsRK2/gsRK4<<<gridSize, blockSize>>>(d_u, d_v, d_u_new, d_v_new, ...)

  // TODO: 交换指针 (u ↔ u_new, v ↔ v_new)
  //   std::swap(d_u, d_u_new); std::swap(d_v, d_v_new);
  (void)W; (void)H; (void)dt; (void)Du; (void)Dv; (void)F; (void)k; (void)method;
}

void gsFree(float *d_u, float *d_v, float *d_u_new, float *d_v_new) {
  // TODO: cudaFree 四块
  (void)d_u; (void)d_v; (void)d_u_new; (void)d_v_new;
}

void gsCopyToHost(float *h_u, float *h_v,
                  const float *d_u, const float *d_v,
                  int W, int H) {
  // TODO: cudaMemcpy d_u→h_u, d_v→h_v
  (void)h_u; (void)h_v; (void)d_u; (void)d_v; (void)W; (void)H;
}

// =================================================================
//  Render: 显示 v 场 (斑图主体), 颜色映射到 [0,1]
// =================================================================

static void render(const float *v, int W, int H, cv::Mat &display) {
  cv::Mat gray(H, W, CV_8UC1);
  for (int j = 0; j < H; ++j)
    for (int i = 0; i < W; ++i) {
      float val = v[j * W + i];
      gray.at<uint8_t>(j, i) = (uint8_t)(fminf(fmaxf(val, 0.0f), 1.0f) * 255.0f);
    }
  cv::applyColorMap(gray, display, cv::COLORMAP_INFERNO);
}

// =================================================================
//  Trackbar state — 运行时实时调节参数
// =================================================================

struct Params {
  float F = 0.04f;
  float k = 0.06f;
  float Du = 0.16f;
  float Dv = 0.08f;
  int   method = 0;  // 0=Euler, 1=RK2, 2=RK4
};

static Params g_params;

static void onTrackbar(int, void*) {
  // 空回调，参数直接从 g_params 读取
}

static void createTrackbars(const char *win) {
  cv::createTrackbar("F * 1000", win, nullptr, 100, onTrackbar);
  cv::createTrackbar("k * 1000", win, nullptr, 100, onTrackbar);
  cv::createTrackbar("Du * 100", win, nullptr, 100, onTrackbar);
  cv::createTrackbar("Dv * 100", win, nullptr, 100, onTrackbar);
  cv::createTrackbar("method",    win, nullptr, 2,   onTrackbar);

  // 初始值
  cv::setTrackbarPos("F * 1000", win, (int)(g_params.F * 1000));
  cv::setTrackbarPos("k * 1000", win, (int)(g_params.k * 1000));
  cv::setTrackbarPos("Du * 100", win, (int)(g_params.Du * 100));
  cv::setTrackbarPos("Dv * 100", win, (int)(g_params.Dv * 100));
  cv::setTrackbarPos("method",   win, g_params.method);
}

static void readTrackbars(const char *win) {
  g_params.F  = cv::getTrackbarPos("F * 1000", win) / 1000.0f;
  g_params.k  = cv::getTrackbarPos("k * 1000", win) / 1000.0f;
  g_params.Du = cv::getTrackbarPos("Du * 100", win) / 100.0f;
  g_params.Dv = cv::getTrackbarPos("Dv * 100", win) / 100.0f;
  g_params.method = cv::getTrackbarPos("method", win);
}

// =================================================================
//  Main loop
// =================================================================

static bool g_paused = true;
static bool g_running = true;

void gsMainLoop(float *d_u, float *d_v, float *d_u_new, float *d_v_new,
                int W, int H) {
  int n = W * H;
  float dt = 0.02f;

  cv::Mat display(H, W, CV_8UC3);
  float *h_v = new float[n];

  cv::namedWindow("Gray-Scott", cv::WINDOW_NORMAL);
  createTrackbars("Gray-Scott");

  int step = 0;
  while (g_running) {
    readTrackbars("Gray-Scott");

    int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      g_running = false;
    } else if (key == ' ') {
      g_paused = !g_paused;
      std::cout << (g_paused ? "[PAUSED]" : "[RUNNING]")
                << "  step=" << step
                << "  F=" << g_params.F
                << "  k=" << g_params.k
                << "  Du=" << g_params.Du
                << "  Dv=" << g_params.Dv
                << "  method="
                << (g_params.method == 0 ? "Euler" :
                    g_params.method == 1 ? "RK2" : "RK4")
                << "\n";
    } else if (key == 'r' || key == 'R') {
      // TODO: 重置初始场 (全 1 的 u, 中心小 patch 的 v)
      step = 0;
    } else if (key == '1') {
      g_params.F = 0.04f; g_params.k = 0.06f;
      g_params.Du = 0.16f; g_params.Dv = 0.08f;
      std::cout << "[preset] spots\n";
      cv::setTrackbarPos("F * 1000", "Gray-Scott", (int)(g_params.F * 1000));
      cv::setTrackbarPos("k * 1000", "Gray-Scott", (int)(g_params.k * 1000));
    } else if (key == '2') {
      g_params.F = 0.035f; g_params.k = 0.065f;
      g_params.Du = 0.16f; g_params.Dv = 0.08f;
      std::cout << "[preset] stripes\n";
      cv::setTrackbarPos("F * 1000", "Gray-Scott", (int)(g_params.F * 1000));
      cv::setTrackbarPos("k * 1000", "Gray-Scott", (int)(g_params.k * 1000));
    } else if (key == '3') {
      g_params.F = 0.025f; g_params.k = 0.06f;
      g_params.Du = 0.16f; g_params.Dv = 0.08f;
      std::cout << "[preset] mazes\n";
      cv::setTrackbarPos("F * 1000", "Gray-Scott", (int)(g_params.F * 1000));
      cv::setTrackbarPos("k * 1000", "Gray-Scott", (int)(g_params.k * 1000));
    }

    if (g_paused) {
      gsCopyToHost(nullptr, h_v, nullptr, d_v, W, H);
      render(h_v, W, H, display);
      cv::setWindowTitle("Gray-Scott", "Gray-Scott [paused]");
      cv::imshow("Gray-Scott", display);
      continue;
    }

    // TODO: 调用 gsStep(d_u, d_v, d_u_new, d_v_new, W, H,
    //                    dt, g_params.Du, g_params.Dv,
    //                    g_params.F, g_params.k, g_params.method)
    ++step;

    if (step % 4 == 0) {
      gsCopyToHost(nullptr, h_v, nullptr, d_v, W, H);
      render(h_v, W, H, display);
      char title[64];
      snprintf(title, sizeof(title),
               "Gray-Scott [step %d | %s]",
               step,
               g_params.method == 0 ? "Euler" :
               g_params.method == 1 ? "RK2" : "RK4");
      cv::setWindowTitle("Gray-Scott", title);
      cv::imshow("Gray-Scott", display);
    }
  }

  cv::destroyAllWindows();
  delete[] h_v;
}

// =================================================================
//  main
// =================================================================

int main(int argc, char **argv) {
  int W = 256, H = 256;

  std::cout << "Gray-Scott | " << W << "x" << H << " | CUDA\n";
  std::cout << "  SPACE = play/pause\n"
            << "  R     = reset\n"
            << "  1     = spots  (F=0.04,  k=0.06)\n"
            << "  2     = stripes(F=0.035,k=0.065)\n"
            << "  3     = mazes  (F=0.025,k=0.06)\n"
            << "  trackbars = F, k, Du, Dv, method\n"
            << "  ESC/Q = quit\n";

  int n = W * H;

  // 初始场: u 全 1, v 中心一小块随机
  float *h_u_init = new float[n];
  float *h_v_init = new float[n];
  for (int i = 0; i < n; ++i) {
    h_u_init[i] = 1.0f;
    h_v_init[i] = 0.0f;
  }
  // v: 中央方形区域随机扰动
  int cx = W / 2, cy = H / 2, r = 10;
  // TODO: 用确定的种子或 rand() 生成初始 v 斑块
  srand(42);
  for (int j = cy - r; j <= cy + r; ++j)
    for (int i = cx - r; i <= cx + r; ++i)
      if (i >= 1 && i < W - 1 && j >= 1 && j < H - 1)
        h_v_init[j * W + i] = (rand() % 100) / 100.0f;

  float *d_u, *d_v, *d_u_new, *d_v_new;
  gsInit(d_u, d_v, d_u_new, d_v_new, W, H, h_u_init, h_v_init);
  gsMainLoop(d_u, d_v, d_u_new, d_v_new, W, H);
  gsFree(d_u, d_v, d_u_new, d_v_new);

  delete[] h_u_init;
  delete[] h_v_init;
  return 0;
}
