/*
 * Stage 1 — grayscott: Gray-Scott 反应-扩散模型
 *
 * ∂u/∂t = D_u ∇²u - u*v² + F*(1-u)
 * ∂v/∂t = D_v ∇²v + u*v² - (F+k)*v
 *
 * 双场耦合 stencil，每个线程更新两个化学浓度 (u, v)
 *
 * TODO: 实现 dual-field stencil kernel
 */

#include <cstdio>

int main() {
  printf("grayscott: TODO — Gray-Scott reaction-diffusion not yet implemented\n");
  return 0;
}
