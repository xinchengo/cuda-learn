/*
 * Stage 1 — grayscott: Gray-Scott 反应-扩散模型
 *
 * ∂u/∂t = D_u ∇²u - u·v² + F·(1-u)
 * ∂v/∂t = D_v ∇²v + u·v² - (F+k)·v
 *
 * 双场耦合 PDE，典型斑图:
 *   F=0.04,  k=0.06  → spots
 *   F=0.035, k=0.065 → stripes
 *   F=0.025, k=0.06  → mazes
 */

#include <argparse/argparse.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda/cmath>

// =================================================================
//  Device helpers
// =================================================================

__device__ float laplacian5(const float *f, int idx, int W) {
  return f[idx - W] + f[idx + W] + f[idx - 1] + f[idx + 1] - 4.0f * f[idx];
}

__device__ float laplacian9(const float *f, int idx, int W) {
  return (4.0f * (f[idx - W] + f[idx + W] + f[idx - 1] + f[idx + 1])
          + (f[idx - W - 1] + f[idx - W + 1] + f[idx + W - 1] + f[idx + W + 1])
          - 20.0f * f[idx]) / 6.0f;
}

struct Reaction { float du, dv; };

__device__ Reaction reaction(float u, float v, float F, float k) {
  float uvv = u * v * v;
  return { -uvv + F * (1.0f - u), uvv - (F + k) * v };
}

// =================================================================
//  Kernels — Euler / RK2 / RK4
// =================================================================

__global__ void gsEuler(const float *u, const float *v,
                        float *u_new, float *v_new,
                        int W, int H, float dt,
                        float Du, float Dv, float F, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= W - 1 || j >= H - 1) return;
  int idx = j * W + i;

  float u0 = u[idx], v0 = v[idx];
  float lap_u = laplacian5(u, idx, W);
  float lap_v = laplacian5(v, idx, W);
  auto r = reaction(u0, v0, F, k);

  float un = u0 + dt * (Du * lap_u + r.du);
  float vn = v0 + dt * (Dv * lap_v + r.dv);

  u_new[idx] = fminf(fmaxf(un, 0.0f), 1.0f);
  v_new[idx] = fminf(fmaxf(vn, 0.0f), 1.0f);
}

__global__ void gsRK2(const float *u, const float *v,
                      float *u_new, float *v_new,
                      int W, int H, float dt,
                      float Du, float Dv, float F, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= W - 1 || j >= H - 1) return;
  int idx = j * W + i;

  float u0 = u[idx], v0 = v[idx];
  float lap_u = laplacian5(u, idx, W);
  float lap_v = laplacian5(v, idx, W);

  // k1
  auto r1 = reaction(u0, v0, F, k);
  float k1_u = Du * lap_u + r1.du;
  float k1_v = Dv * lap_v + r1.dv;

  // k2 (反应项用中点值重新算, Laplacian 近似不变)
  auto r2 = reaction(u0 + 0.5f * dt * k1_u, v0 + 0.5f * dt * k1_v, F, k);
  float k2_u = Du * lap_u + r2.du;
  float k2_v = Dv * lap_v + r2.dv;

  float un = u0 + dt * k2_u;
  float vn = v0 + dt * k2_v;

  u_new[idx] = fminf(fmaxf(un, 0.0f), 1.0f);
  v_new[idx] = fminf(fmaxf(vn, 0.0f), 1.0f);
}

__global__ void gsRK4(const float *u, const float *v,
                      float *u_new, float *v_new,
                      int W, int H, float dt,
                      float Du, float Dv, float F, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= W - 1 || j >= H - 1) return;
  int idx = j * W + i;

  float u0 = u[idx], v0 = v[idx];
  float lap_u = laplacian5(u, idx, W);
  float lap_v = laplacian5(v, idx, W);

  // k1
  auto r1 = reaction(u0, v0, F, k);
  float k1_u = Du * lap_u + r1.du;
  float k1_v = Dv * lap_v + r1.dv;

  // k2
  auto r2 = reaction(u0 + 0.5f * dt * k1_u, v0 + 0.5f * dt * k1_v, F, k);
  float k2_u = Du * lap_u + r2.du;
  float k2_v = Dv * lap_v + r2.dv;

  // k3
  auto r3 = reaction(u0 + 0.5f * dt * k2_u, v0 + 0.5f * dt * k2_v, F, k);
  float k3_u = Du * lap_u + r3.du;
  float k3_v = Dv * lap_v + r3.dv;

  // k4
  auto r4 = reaction(u0 + dt * k3_u, v0 + dt * k3_v, F, k);
  float k4_u = Du * lap_u + r4.du;
  float k4_v = Dv * lap_v + r4.dv;

  float un = u0 + dt / 6.0f * (k1_u + 2.0f * k2_u + 2.0f * k3_u + k4_u);
  float vn = v0 + dt / 6.0f * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);

  u_new[idx] = fminf(fmaxf(un, 0.0f), 1.0f);
  v_new[idx] = fminf(fmaxf(vn, 0.0f), 1.0f);
}

// =================================================================
//  Host helpers
// =================================================================

void gsInit(float *&d_u, float *&d_v, float *&d_u_new, float *&d_v_new,
            int W, int H, const float *h_u_init, const float *h_v_init) {
  int n = W * H;
  cudaMalloc(&d_u,     n * sizeof(float));
  cudaMalloc(&d_v,     n * sizeof(float));
  cudaMalloc(&d_u_new, n * sizeof(float));
  cudaMalloc(&d_v_new, n * sizeof(float));
  cudaMemcpy(d_u,     h_u_init, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v,     h_v_init, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u_new, h_u_init, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_new, h_v_init, n * sizeof(float), cudaMemcpyHostToDevice);
}

void gsStep(float *&d_u, float *&d_v, float *&d_u_new, float *&d_v_new,
            int W, int H, float dt, float Du, float Dv, float F, float k,
            int method) {
  dim3 blockSize(16, 16);
  dim3 gridSize(cuda::ceil_div(W - 2, blockSize.x),
                cuda::ceil_div(H - 2, blockSize.y));

  if (method == 0)
    gsEuler<<<gridSize, blockSize>>>(d_u, d_v, d_u_new, d_v_new, W, H, dt, Du, Dv, F, k);
  else if (method == 1)
    gsRK2<<<gridSize, blockSize>>>(d_u, d_v, d_u_new, d_v_new, W, H, dt, Du, Dv, F, k);
  else
    gsRK4<<<gridSize, blockSize>>>(d_u, d_v, d_u_new, d_v_new, W, H, dt, Du, Dv, F, k);

  std::swap(d_u, d_u_new);
  std::swap(d_v, d_v_new);
}

void gsFree(float *d_u, float *d_v, float *d_u_new, float *d_v_new) {
  cudaFree(d_u); cudaFree(d_v); cudaFree(d_u_new); cudaFree(d_v_new);
}

void gsCopyToHost(float *h_u, float *h_v,
                  const float *d_u, const float *d_v,
                  int W, int H) {
  int n = W * H;
  if (h_u) cudaMemcpy(h_u, d_u, n * sizeof(float), cudaMemcpyDeviceToHost);
  if (h_v) cudaMemcpy(h_v, d_v, n * sizeof(float), cudaMemcpyDeviceToHost);
}

// =================================================================
//  Render
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
//  Trackbar
// =================================================================

struct Params {
  float F = 0.04f;
  float k = 0.06f;
  float Du = 0.16f;
  float Dv = 0.08f;
  int method = 0;  // 0=Euler, 1=RK2, 2=RK4
};

static Params g_params;

static void onTrackbar(int, void*) {}

static void createTrackbars(const char *win) {
  cv::createTrackbar("F * 1000", win, nullptr, 100, onTrackbar);
  cv::createTrackbar("k * 1000", win, nullptr, 100, onTrackbar);
  cv::createTrackbar("Du * 100", win, nullptr, 100, onTrackbar);
  cv::createTrackbar("Dv * 100", win, nullptr, 100, onTrackbar);
  cv::createTrackbar("method",    win, nullptr, 2,   onTrackbar);

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

static void syncTrackbars(const char *win) {
  cv::setTrackbarPos("F * 1000", win, (int)(g_params.F * 1000));
  cv::setTrackbarPos("k * 1000", win, (int)(g_params.k * 1000));
  cv::setTrackbarPos("Du * 100", win, (int)(g_params.Du * 100));
  cv::setTrackbarPos("Dv * 100", win, (int)(g_params.Dv * 100));
  cv::setTrackbarPos("method",   win, g_params.method);
}

// =================================================================
//  Main loop
// =================================================================

static bool g_paused = false;
static bool g_running = true;
static bool g_mouse_down = false;

struct MouseCtx {
  int w, h;
  float *h_v;
  float *d_v;
};

static void paintBrush(float *v, int w, int h, int x, int y) {
  constexpr int R = 3;
  for (int dy = -R; dy <= R; ++dy) {
    for (int dx = -R; dx <= R; ++dx) {
      int nx = x + dx, ny = y + dy;
      if (nx < 1 || nx >= w - 1 || ny < 1 || ny >= h - 1) continue;
      float rr = (float)(dx * dx + dy * dy) / (R * R);
      // 画 v 场: 中心 1.0, u 对应位置也设为随机
      float val = 1.0f * expf(-2.0f * rr);
      if (val > v[ny * w + nx])
        v[ny * w + nx] = val;
    }
  }
}

static void onMouse(int event, int x, int y, int flags, void *userdata) {
  MouseCtx *c = (MouseCtx *)userdata;
  if (!c || !c->h_v) return;

  if (event == cv::EVENT_LBUTTONDOWN) {
    g_paused = true;
    g_mouse_down = true;
    paintBrush(c->h_v, c->w, c->h, x, y);
    // upload 9x9 region to d_v
    for (int dy = -4; dy <= 4; ++dy) {
      int ny = y + dy;
      if (ny < 1 || ny >= c->h - 1) continue;
      int nx0 = (x - 4 < 1) ? 1 : x - 4;
      int nx1 = (x + 4 >= c->w - 1) ? c->w - 2 : x + 4;
      int len = nx1 - nx0 + 1;
      cudaMemcpy(c->d_v + ny * c->w + nx0,
                 c->h_v + ny * c->w + nx0,
                 len * sizeof(float), cudaMemcpyHostToDevice);
    }
  } else if (event == cv::EVENT_MOUSEMOVE && g_mouse_down) {
    paintBrush(c->h_v, c->w, c->h, x, y);
    for (int dy = -4; dy <= 4; ++dy) {
      int ny = y + dy;
      if (ny < 1 || ny >= c->h - 1) continue;
      int nx0 = (x - 4 < 1) ? 1 : x - 4;
      int nx1 = (x + 4 >= c->w - 1) ? c->w - 2 : x + 4;
      int len = nx1 - nx0 + 1;
      cudaMemcpy(c->d_v + ny * c->w + nx0,
                 c->h_v + ny * c->w + nx0,
                 len * sizeof(float), cudaMemcpyHostToDevice);
    }
  } else if (event == cv::EVENT_LBUTTONUP) {
    g_mouse_down = false;
  }
}

static void resetFields(float *d_u, float *d_v, float *d_u_new, float *d_v_new,
                         const float *h_u_init, const float *h_v_init, int n) {
  cudaMemcpy(d_u,     h_u_init, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v,     h_v_init, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u_new, h_u_init, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_new, h_v_init, n * sizeof(float), cudaMemcpyHostToDevice);
}

void gsMainLoop(float *d_u, float *d_v, float *d_u_new, float *d_v_new,
                int W, int H, int substeps,
                const float *h_u_init, const float *h_v_init) {
  int n = W * H;
  float dt = 0.02f;

  cv::Mat display(H, W, CV_8UC3);
  float *h_v = new float[n];

  MouseCtx mc{ W, H, h_v, d_v };

  cv::namedWindow("Gray-Scott", cv::WINDOW_NORMAL);
  createTrackbars("Gray-Scott");
  cv::setMouseCallback("Gray-Scott", onMouse, &mc);

  int step = 0;
  while (g_running) {
    readTrackbars("Gray-Scott");

    int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      g_running = false;
    } else if (key == ' ' && !g_mouse_down) {
      g_paused = !g_paused;
      std::cout << (g_paused ? "[PAUSED]" : "[RUNNING]")
                << "  step=" << step
                << "  F=" << g_params.F
                << "  k=" << g_params.k
                << "  Du=" << g_params.Du
                << "  Dv=" << g_params.Dv
                << "  substeps=" << substeps
                << "  method="
                << (g_params.method == 0 ? "Euler" :
                    g_params.method == 1 ? "RK2" : "RK4")
                << std::endl;
    } else if (key == 82 && substeps < 256) {
      ++substeps;
      std::cout << "substeps=" << substeps << std::endl;
    } else if (key == 84 && substeps > 1) {
      --substeps;
      std::cout << "substeps=" << substeps << std::endl;
    } else if (key == 'r' || key == 'R') {
      resetFields(d_u, d_v, d_u_new, d_v_new, h_u_init, h_v_init, n);
      step = 0;
      std::cout << "[RESET]" << std::endl;
    } else if (key == '1') {
      g_params.F = 0.04f; g_params.k = 0.06f;
      g_params.Du = 0.16f; g_params.Dv = 0.08f;
      resetFields(d_u, d_v, d_u_new, d_v_new, h_u_init, h_v_init, n);
      step = 0;
      std::cout << "[preset] spots" << std::endl;
      syncTrackbars("Gray-Scott");
    } else if (key == '2') {
      g_params.F = 0.035f; g_params.k = 0.065f;
      g_params.Du = 0.16f; g_params.Dv = 0.08f;
      resetFields(d_u, d_v, d_u_new, d_v_new, h_u_init, h_v_init, n);
      step = 0;
      std::cout << "[preset] stripes" << std::endl;
      syncTrackbars("Gray-Scott");
    } else if (key == '3') {
      g_params.F = 0.025f; g_params.k = 0.06f;
      g_params.Du = 0.16f; g_params.Dv = 0.08f;
      resetFields(d_u, d_v, d_u_new, d_v_new, h_u_init, h_v_init, n);
      step = 0;
      std::cout << "[preset] mazes" << std::endl;
      syncTrackbars("Gray-Scott");
    }

    if (g_paused) {
      gsCopyToHost(nullptr, h_v, nullptr, d_v, W, H);
      render(h_v, W, H, display);
      char title[64];
      snprintf(title, sizeof(title),
               "Gray-Scott [paused]  F=%.3f k=%.3f  %s",
               g_params.F, g_params.k,
               g_params.method == 0 ? "Euler" :
               g_params.method == 1 ? "RK2" : "RK4");
      cv::setWindowTitle("Gray-Scott", title);
      cv::imshow("Gray-Scott", display);
      continue;
    }

    for (int s = 0; s < substeps; ++s) {
      gsStep(d_u, d_v, d_u_new, d_v_new, W, H,
             dt, g_params.Du, g_params.Dv,
             g_params.F, g_params.k, g_params.method);
      ++step;
    }

    gsCopyToHost(nullptr, h_v, nullptr, d_v, W, H);
    render(h_v, W, H, display);
    char title[64];
    snprintf(title, sizeof(title),
             "Gray-Scott [step %d | %s | x%d]",
             step,
             g_params.method == 0 ? "Euler" :
             g_params.method == 1 ? "RK2" : "RK4",
             substeps);
    cv::setWindowTitle("Gray-Scott", title);
    cv::imshow("Gray-Scott", display);
  }

  cv::destroyAllWindows();
  delete[] h_v;
}

// =================================================================
//  main
// =================================================================

int main(int argc, char **argv) {
  int W = 256, H = 256;
  int substeps = 16;

  argparse::ArgumentParser program("grayscott");
  program.add_description("Gray-Scott reaction-diffusion (CUDA)");

  program.add_argument("-W", "--width")
      .default_value(256).help("Grid width").store_into(W);
  program.add_argument("-H", "--height")
      .default_value(256).help("Grid height").store_into(H);
  program.add_argument("-n", "--substeps")
      .default_value(16).help("Simulation steps per frame").store_into(substeps);
  program.add_argument("--F")
      .default_value(0.04f).help("Feed rate").store_into(g_params.F);
  program.add_argument("--k")
      .default_value(0.06f).help("Kill rate").store_into(g_params.k);
  program.add_argument("--Du")
      .default_value(0.16f).help("Diffusion u").store_into(g_params.Du);
  program.add_argument("--Dv")
      .default_value(0.08f).help("Diffusion v").store_into(g_params.Dv);
  program.add_argument("-m", "--method")
      .default_value(0).help("0=Euler, 1=RK2, 2=RK4").store_into(g_params.method);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << "\n" << program;
    return 1;
  }

  std::cout << "Gray-Scott | " << W << "x" << H
            << " | substeps=" << substeps
            << " | F=" << g_params.F
            << " k=" << g_params.k
            << " Du=" << g_params.Du
            << " Dv=" << g_params.Dv
            << " | method="
            << (g_params.method == 0 ? "Euler" :
                g_params.method == 1 ? "RK2" : "RK4")
            << std::endl;
  std::cout << "  SPACE = play/pause   R = reset\n"
            << "  1 = spots   2 = stripes   3 = mazes\n"
            << "  ARROWS = substeps +/-   trackbars = F,k,Du,Dv,method\n"
            << "  ESC/Q = quit" << std::endl;

  int n = W * H;

  float *h_u_init = new float[n];
  float *h_v_init = new float[n];
  for (int i = 0; i < n; ++i) {
    h_u_init[i] = 1.0f;
    h_v_init[i] = 0.0f;
  }

  int cx = W / 2, cy = H / 2, r = 10;
  srand(42);
  for (int j = cy - r; j <= cy + r; ++j)
    for (int i = cx - r; i <= cx + r; ++i)
      if (i >= 1 && i < W - 1 && j >= 1 && j < H - 1)
        h_v_init[j * W + i] = (rand() % 100) / 100.0f;

  float *d_u, *d_v, *d_u_new, *d_v_new;
  gsInit(d_u, d_v, d_u_new, d_v_new, W, H, h_u_init, h_v_init);
  gsMainLoop(d_u, d_v, d_u_new, d_v_new, W, H, substeps, h_u_init, h_v_init);
  gsFree(d_u, d_v, d_u_new, d_v_new);

  delete[] h_u_init;
  delete[] h_v_init;
  return 0;
}
