#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
struct InkRuntimeArgs { uint count, boundary; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_copy_bf16(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  // boundary=3 gathers the third of four completed blocks per row into a contiguous bank.
  const uint src = a.boundary == 0u ? i : (i / 2304u * 4u + a.boundary - 1u) * 2304u + i % 2304u;
  y[i] = x[src];
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_copy_f32(const device float* x [[buffer(0)]], device float* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) y[i] = x[i];
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_widen(const device bfloat* x [[buffer(0)]], device float* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) y[i] = float(x[i]);
}
// The old three convolution rows and this launch's fresh rows, rounded exactly as glue's BF16 convolution.
inline float ink2_tap(const device bfloat* old, const device bfloat* fresh, uint i, uint d) {
  return i < 3u ? float(old[i * 2304u + d]) : float(fresh[size_t(i - 3u) * 2304u + d]);
}
inline bfloat ink2_conv(const device bfloat* old, const device bfloat* fresh, const device bfloat* w, uint row, uint d) {
  float sum = ink2_tap(old, fresh, row + 3u, d);
  for (uint tap = 0; tap < 4u; ++tap) sum += float(w[d * 4u + tap]) * ink2_tap(old, fresh, row + tap, d);
  return bfloat(precise::divide(sum, precise::sqrt(5.0f)));
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_attention_partial(const device bfloat* p [[buffer(0)]], const device bfloat* old [[buffer(1)]],
                                      const device bfloat* fresh [[buffer(2)]], const device bfloat* w [[buffer(3)]],
                                      device bfloat* blocks [[buffer(4)]], device float* t [[buffer(5)]],
                                      constant InkRuntimeArgs& a [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / 2304u, d = i % 2304u;
  const float x = float(ink2_conv(old, fresh, w, row, d));
  t[i] = a.boundary ? x : float(p[i]) + x;
  if (a.boundary) blocks[row * 4u * 2304u + (a.boundary - 1u) * 2304u + d] = p[i];
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_mlp_partial(const device float* t [[buffer(0)]], const device bfloat* old [[buffer(1)]],
                               const device bfloat* fresh [[buffer(2)]], const device bfloat* w [[buffer(3)]],
                               device bfloat* p [[buffer(4)]], device float* wide [[buffer(5)]],
                               constant InkRuntimeArgs& a [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const bfloat v = bfloat(t[i] + float(ink2_conv(old, fresh, w, i / 2304u, i % 2304u)));
  p[i] = v;
  wide[i] = float(v);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_capture_tap(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / 2304, d = i % 2304;
  y[row * 4 * 2304 + a.boundary * 2304 + d] = x[i];
}
