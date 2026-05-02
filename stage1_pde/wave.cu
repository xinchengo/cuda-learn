/*
 * Stage 1 — wave: 2D 波动方程
 *
 * ∂²u/∂t² = c² ∇²u
 *
 * 三步法 (Verlet / leapfrog):
 *   u_new = 2·u - u_old + c²·dt²·∇²u
 */

#include <argparse/argparse.hpp>

#include <cmath>
#include <cstring>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda/cmath>

int H, W, Steps;

// =================================================================
//  CUDA kernel
// =================================================================

__global__ void waveStep5(const float *u, const float *u_old, float *u_new,
                          int W, int H, float coeff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= W - 1 || j >= H - 1) return;
  int idx = j * W + i;
  float stencil = u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1] - 4.0f * u[idx];
  u_new[idx] = 2.0f * u[idx] - u_old[idx] + coeff * stencil;
}

__global__ void waveStep9(const float *u, const float *u_old, float *u_new,
                          int W, int H, float coeff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= W - 1 || j >= H - 1) return;
  int idx = j * W + i;
  float c6 = coeff / 6.0f;
  float stencil =
      4.0f * (u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1]) +
      (u[idx - W - 1] + u[idx - W + 1] + u[idx + W - 1] + u[idx + W + 1]) -
      20.0f * u[idx];
  u_new[idx] = 2.0f * u[idx] - u_old[idx] + c6 * stencil;
}

// =================================================================
//  CPU step
// =================================================================

void step5_cpu(const float *u, const float *u_old, float *u_new,
               int W, int H, float coeff) {
  for (int j = 1; j < H - 1; ++j)
    for (int i = 1; i < W - 1; ++i) {
      int idx = j * W + i;
      float stencil = u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1] - 4.0f * u[idx];
      u_new[idx] = 2.0f * u[idx] - u_old[idx] + coeff * stencil;
    }
}

void step9_cpu(const float *u, const float *u_old, float *u_new,
               int W, int H, float coeff) {
  float c6 = coeff / 6.0f;
  for (int j = 1; j < H - 1; ++j)
    for (int i = 1; i < W - 1; ++i) {
      int idx = j * W + i;
      float stencil =
          4.0f * (u[idx - W] + u[idx + W] + u[idx - 1] + u[idx + 1]) +
          (u[idx - W - 1] + u[idx - W + 1] + u[idx + W - 1] + u[idx + W + 1]) -
          20.0f * u[idx];
      u_new[idx] = 2.0f * u[idx] - u_old[idx] + c6 * stencil;
    }
}

// =================================================================
//  CUDA host helpers
// =================================================================

void cudaWaveInit(float *&d_u_old, float *&d_u, float *&d_u_new,
                  int W, int H) {
  int n = W * H;
  cudaMalloc(&d_u_old, n * sizeof(float));
  cudaMalloc(&d_u,     n * sizeof(float));
  cudaMalloc(&d_u_new, n * sizeof(float));
  cudaMemset(d_u_old, 0, n * sizeof(float));
  cudaMemset(d_u,     0, n * sizeof(float));
  cudaMemset(d_u_new, 0, n * sizeof(float));
}

void cudaWaveStep(float *&d_u_old, float *&d_u, float *&d_u_new,
                  int W, int H, float coeff, bool use9pt) {
  dim3 blockSize(16, 16);
  dim3 gridSize(cuda::ceil_div(W - 2, blockSize.x),
                cuda::ceil_div(H - 2, blockSize.y));
  if (use9pt)
    waveStep9<<<gridSize, blockSize>>>(d_u, d_u_old, d_u_new, W, H, coeff);
  else
    waveStep5<<<gridSize, blockSize>>>(d_u, d_u_old, d_u_new, W, H, coeff);
  std::swap(d_u_old, d_u);
  std::swap(d_u, d_u_new);
}

void cudaWaveFree(float *d_u_old, float *d_u, float *d_u_new) {
  cudaFree(d_u_old); cudaFree(d_u); cudaFree(d_u_new);
}

// =================================================================
//  Shared state
// =================================================================

static bool g_paused = false;
static bool g_running = true;
static bool g_mouse_down = false;

struct MouseCtx {
  int w, h;
  float *h_paint;  // 涂抹累计 —— host 端
  float *h_u;      // 当前场快照 —— host 端
  float *d_u;      // CUDA 端当前场，CPU 模式为 nullptr
};

// =================================================================
//  Render: 仿真场 + 涂抹预览叠加
// =================================================================

static void render(const float *u, const float *paint,
                   int W, int H, cv::Mat &display, cv::Mat &gray) {
  constexpr float vmax = 0.8f;
  constexpr float scale = 255.0f / (2.0f * vmax);

  for (int j = 0; j < H; ++j)
    for (int i = 0; i < W; ++i) {
      float v = u[j * W + i];
      if (v > vmax) v = vmax;
      if (v < -vmax) v = -vmax;
      gray.at<uint8_t>(j, i) = (uint8_t)((v + vmax) * scale);
    }
  cv::applyColorMap(gray, display, cv::COLORMAP_JET);

  // 涂抹预览：按强度 alpha 混合叠加
  if (paint) {
    for (int j = 1; j < H - 1; ++j)
      for (int i = 1; i < W - 1; ++i) {
        float v = paint[j * W + i];
        if (v == 0.0f) continue;
        float a = fminf(fabsf(v) * 5.0f, 1.0f);
        cv::Vec3b &px = display.at<cv::Vec3b>(j, i);
        if (v > 0) {
          // 暖红 (BGR: 0, 0, 255 = 纯红)
          px[0] = (uint8_t)(px[0] * (1 - a) + 0 * a);
          px[1] = (uint8_t)(px[1] * (1 - a) + 0 * a);
          px[2] = (uint8_t)(px[2] * (1 - a) + 255 * a);
        } else {
          // 冷蓝 (BGR: 255, 0, 0 = 纯蓝)
          px[0] = (uint8_t)(px[0] * (1 - a) + 255 * a);
          px[1] = (uint8_t)(px[1] * (1 - a) + 0 * a);
          px[2] = (uint8_t)(px[2] * (1 - a) + 0 * a);
        }
      }
  }
}

// =================================================================
//  Brush: 涂抹到 h_paint (不碰仿真场)
// =================================================================

static void paintBrush(float *paint, int w, int h, int x, int y) {
  constexpr int R = 2;
  for (int dy = -R; dy <= R; ++dy) {
    for (int dx = -R; dx <= R; ++dx) {
      int nx = x + dx, ny = y + dy;
      if (nx < 1 || nx >= w - 1 || ny < 1 || ny >= h - 1) continue;
      float rr = (float)(dx * dx + dy * dy) / (R * R);
      float val = 0.15f * expf(-2.0f * rr);
      paint[ny * w + nx] = val;
    }
  }
}

static void onMouse(int event, int x, int y, int flags, void *userdata) {
  MouseCtx *c = (MouseCtx *)userdata;
  if (!c || !c->h_paint) return;

  if (event == cv::EVENT_LBUTTONDOWN) {
    g_paused = true;
    g_mouse_down = true;
    paintBrush(c->h_paint, c->w, c->h, x, y);
  } else if (event == cv::EVENT_MOUSEMOVE && g_mouse_down) {
    paintBrush(c->h_paint, c->w, c->h, x, y);
  } else if (event == cv::EVENT_LBUTTONUP) {
    g_mouse_down = false;
  }
}

// =================================================================
//  Apply: 将 h_paint 高斯模糊后叠加到仿真场，然后清空
// =================================================================

static void applyPaint(float *u, float *paint, float *d_u,
                       int W, int H) {
  int n = W * H;
  cv::Mat src(H, W, CV_32F, paint);
  cv::Mat blurred;
  cv::GaussianBlur(src, blurred, cv::Size(11, 11), 3.0);

  for (int j = 1; j < H - 1; ++j)
    for (int i = 1; i < W - 1; ++i) {
      int idx = j * W + i;
      u[idx] += blurred.at<float>(j, i);
    }

  std::memset(paint, 0, n * sizeof(float));

  if (d_u)
    cudaMemcpy(d_u, u, n * sizeof(float), cudaMemcpyHostToDevice);
}

// =================================================================
//  Main loops
// =================================================================

void cudaMainLoop(int maxSteps, float coeff, bool use9pt,
                  float *d_u_old, float *d_u, float *d_u_new,
                  int W, int H) {
  int n = W * H;
  cv::Mat display(H, W, CV_8UC3);
  cv::Mat gray(H, W, CV_8UC1);
  float *h_u = new float[n]();
  float *h_paint = new float[n]();

  MouseCtx mc{ W, H, h_paint, h_u, d_u };
  cv::namedWindow("Wave Simulation", cv::WINDOW_NORMAL);
  cv::setMouseCallback("Wave Simulation", onMouse, &mc);

  int step = 0;
  bool bounded = (maxSteps > 0);
  while (g_running && (!bounded || step < maxSteps)) {
    int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') { g_running = false; }
    else if (key == ' ' && !g_mouse_down) {
      bool was = g_paused;
      g_paused = !g_paused;
      if (was && !g_paused)  // resume: apply paint
        applyPaint(h_u, h_paint, d_u, W, H);
    }
    else if (key == 'r' || key == 'R') {
      cudaMemset(d_u_old, 0, n * sizeof(float));
      cudaMemset(d_u,     0, n * sizeof(float));
      cudaMemset(d_u_new, 0, n * sizeof(float));
      std::memset(h_paint, 0, n * sizeof(float));
      step = 0;
    }

    cudaMemcpy(h_u, d_u, n * sizeof(float), cudaMemcpyDeviceToHost);

    if (g_paused) {
      render(h_u, h_paint, W, H, display, gray);
      cv::setWindowTitle("Wave Simulation", "Wave [paused]");
      cv::imshow("Wave Simulation", display);
      continue;
    }

    if (!g_mouse_down)
      cudaWaveStep(d_u_old, d_u, d_u_new, W, H, coeff, use9pt);
    ++step;

    if (step % 4 == 0) {
      render(h_u, h_paint, W, H, display, gray);
      char title[64];
      snprintf(title, sizeof(title), "Wave [step %d]", step);
      cv::setWindowTitle("Wave Simulation", title);
      cv::imshow("Wave Simulation", display);
    }
  }

  cv::destroyAllWindows();
  delete[] h_u;
  delete[] h_paint;
}

void cpuMainLoop(int maxSteps, float coeff, bool use9pt,
                 float *u, float *u_old, float *u_new,
                 int W, int H) {
  int n = W * H;
  cv::Mat display(H, W, CV_8UC3);
  cv::Mat gray(H, W, CV_8UC1);
  float *h_paint = new float[n]();

  MouseCtx mc{ W, H, h_paint, u, nullptr };
  cv::namedWindow("Wave Simulation", cv::WINDOW_NORMAL);
  cv::setMouseCallback("Wave Simulation", onMouse, &mc);

  int step = 0;
  bool bounded = (maxSteps > 0);
  while (g_running && (!bounded || step < maxSteps)) {
    int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') { g_running = false; }
    else if (key == ' ' && !g_mouse_down) {
      bool was = g_paused;
      g_paused = !g_paused;
      if (was && !g_paused)
        applyPaint(u, h_paint, nullptr, W, H);
    }
    else if (key == 'r' || key == 'R') {
      std::memset(u,     0, n * sizeof(float));
      std::memset(u_old, 0, n * sizeof(float));
      std::memset(u_new, 0, n * sizeof(float));
      std::memset(h_paint, 0, n * sizeof(float));
      step = 0;
    }

    if (g_paused) {
      render(u, h_paint, W, H, display, gray);
      cv::setWindowTitle("Wave Simulation", "Wave [paused]");
      cv::imshow("Wave Simulation", display);
      continue;
    }

    if (!g_mouse_down) {
      if (use9pt) step9_cpu(u, u_old, u_new, W, H, coeff);
      else        step5_cpu(u, u_old, u_new, W, H, coeff);

      for (int j = 0; j < H; ++j)
        u_new[j * W] = u_new[j * W + W - 1] = 0.0f;
      for (int i = 0; i < W; ++i)
        u_new[i] = u_new[(H - 1) * W + i] = 0.0f;

      std::swap(u_old, u);
      std::swap(u, u_new);
    }
    ++step;

    if (step % 4 == 0) {
      render(u, h_paint, W, H, display, gray);
      char title[64];
      snprintf(title, sizeof(title), "Wave [step %d]", step);
      cv::setWindowTitle("Wave Simulation", title);
      cv::imshow("Wave Simulation", display);
    }
  }

  cv::destroyAllWindows();
  delete[] h_paint;
}

// =================================================================
//  main
// =================================================================

int main(int argc, char **argv) {
  constexpr float c = 1.0f;
  constexpr float h_val = 1.0f;
  constexpr float dt = 0.1f;

  argparse::ArgumentParser program("wave");
  program.add_description("2D wave equation — paint & play");

  program.add_argument("-W", "--width")
      .default_value(256).help("Grid width").store_into(W);
  program.add_argument("-H", "--height")
      .default_value(256).help("Grid height").store_into(H);
  program.add_argument("-s", "--steps")
      .default_value(-1).help("Max steps (default: run forever)").store_into(Steps);

  auto &stencil = program.add_mutually_exclusive_group(true);
  stencil.add_argument("--5-point").help("5-point stencil (2nd order)").flag();
  stencil.add_argument("--9-point").help("9-point stencil (4th order)").flag();

  auto &device = program.add_mutually_exclusive_group(true);
  device.add_argument("--cpu").help("Run on CPU").flag();
  device.add_argument("--cuda").help("Run on CUDA GPU").flag();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << "\n" << program;
    return 1;
  }

  bool use9pt  = program.get<bool>("--9-point");
  bool useCuda = program.get<bool>("--cuda");

  std::cout << "Wave | " << W << "x" << H << " | "
            << (use9pt ? "9-point" : "5-point") << " | "
            << (useCuda ? "CUDA" : "CPU") << "\n";
  std::cout << "LMB=paint  SPACE=play  R=reset  ESC=quit\n";

  float coeff = (c * dt / h_val) * (c * dt / h_val);
  int n = W * H;

  float *u = new float[n]();
  float *u_old = new float[n]();
  float *u_new = new float[n]();

  if (useCuda) {
    float *d_u_old, *d_u, *d_u_new;
    cudaWaveInit(d_u_old, d_u, d_u_new, W, H);
    cudaMainLoop(Steps, coeff, use9pt, d_u_old, d_u, d_u_new, W, H);
    cudaWaveFree(d_u_old, d_u, d_u_new);
  } else {
    cpuMainLoop(Steps, coeff, use9pt, u, u_old, u_new, W, H);
  }

  delete[] u;
  delete[] u_old;
  delete[] u_new;

  return 0;
}
