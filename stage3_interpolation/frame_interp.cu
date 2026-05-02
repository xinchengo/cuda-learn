/*
 * Stage 3 — frame_interp: 视频插帧
 *
 * 1. 对相邻两帧算双向 Horn-Schunck 光流
 * 2. 用 forward flow 把 I_t warp 到中间时刻
 * 3. 用 backward flow 把 I_{t+1} warp 到中间时刻
 * 4. 融合两路 warp 结果
 *
 * TODO: 实现基于光流的帧插值
 */

#include <cstdio>

int main() {
  printf("frame_interp: TODO — frame interpolation not yet implemented\n");
  return 0;
}
