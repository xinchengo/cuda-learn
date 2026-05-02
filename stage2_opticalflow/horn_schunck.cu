/*
 * Stage 2 — horn_schunck: Horn-Schunck 光流
 *
 * 1. 计算时空导数 I_x, I_y, I_t (图像梯度 + 帧差)
 * 2. 迭代求解光流 (u, v):
 *
 *    u_bar = 4-邻域平均(u)
 *    v_bar = 4-邻域平均(v)
 *
 *    denom = λ + I_x² + I_y²
 *    u_new = u_bar - I_x * (I_x*u_bar + I_y*v_bar + I_t) / denom
 *    v_new = v_bar - I_y * (I_x*u_bar + I_y*v_bar + I_t) / denom
 *
 * 每次迭代 = 一次 kernel launch (stencil 邻域平均 + 逐点更新)
 *
 * TODO: 实现完整的 Horn-Schunck 光流
 */

#include <cstdio>

int main() {
  printf("horn_schunck: TODO — Horn-Schunck optical flow not yet implemented\n");
  return 0;
}
