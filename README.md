# cuda-learn

从零开始学 CUDA，以数值计算和图像处理为主线。

## 学习路径

### Stage 0 — CUDA 热身：掌握核心范式

| 练习 | 学到什么 |
|------|----------|
| `vector_add` | 一维线程索引 `threadIdx.x + blockIdx.x * blockDim.x`；host↔device 内存管理；kernel launch |
| `image_invert` | 二维线程索引 `(x,y)`；每个线程一个像素的并行模式 |
| `sobel` | **Stencil** 模式——每个线程访问邻居像素（边界处理）；这是后续 PDE 的基础算子 |

### Stage 1 — 2D 微分方程：Stencil 的正经用途

| 练习 | 学到什么 |
|------|----------|
| `heat_euler` | 2D 热传导方程，显式 Euler；ping-pong buffer；每个线程=一个格点，每个 kernel launch=一个时间步 |
| `heat_rk4` | 同一方程用 RK4；每个线程独立算 k1~k4，只读邻居（stencil），无需全局同步 |
| `wave` | 2D 波动方程（双曲型）；三步法（存两个历史帧） |
| `grayscott` | Gray-Scott 反应-扩散模型；双场耦合，视觉效果极佳 |

### Stage 2 — Horn-Schunck 光流

| 练习 | 学到什么 |
|------|----------|
| `horn_schunck` | 计算 I_x/I_y/I_t → 迭代求解 (u,v) → 每个时间步=一次 stencil（邻域平均）+ 逐点更新 |

### Stage 3 — 视频插帧

| 练习 | 学到什么 |
|------|----------|
| `frame_interp` | 双向光流 → warp（双线性采样）→ 融合；把 stage 2 的光流用起来 |

---

## CUDA 核心概念清单

整个学习路径只涉及四个核心机制（反复出现，无需更多）：

1. **二维线程索引** — `ix = blockIdx.x * blockDim.x + threadIdx.x`
2. **Stencil 访问** — 线程读取邻居格点（边界用 clamp/周期/镜像）
3. **逐元素独立计算** — 每个线程算自己的，不通信
4. **内存管理** — `cudaMalloc / cudaMemcpy / kernel launch / cudaFree`

## 构建

```bash
mkdir build && cd build
cmake ..
make -j
```

可选依赖：`libsdl2-dev`（用于交互式可视化，未安装时回退到 PNG 文件输出）
