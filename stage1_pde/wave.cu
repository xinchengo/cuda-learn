/*
 * Stage 1 — wave: 2D 波动方程
 *
 * ∂²u/∂t² = c² ∇²u
 *
 * 离散化 (二阶中心差分):
 *   u_new = 2*u - u_old + (c*dt/h)² * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
 *
 * TODO: 实现三步法 —— 维护 u_new, u, u_old 三个 buffer
 */

#include <cstdio>

int main() {
  printf("wave: TODO — 2D wave equation not yet implemented\n");
  return 0;
}
