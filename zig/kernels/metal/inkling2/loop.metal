#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

// Cross-pass dots use fixed-order FP32 accumulation and SIMD reductions; no BF16 value accumulates.
constant float INKLING_LOOP_EPS = 1e-6f;
constant float INKLING_LOOP_SCALE = 0x1.555556p-6f; // binary32(1 / 48)
constant uint INKLING_LOOP_H = 2304;
constant uint INKLING_LOOP_RANK = 64;
constant uint INKLING_LOOP_MESSAGES = 48;

// stride: floats from one row's m to the next (the messages interleave 48 entries a row).
struct InkLoopArgs { uint rows, stride; };

kernel void tf_inkling2_loop_messages(
    const device bfloat* bank0 [[buffer(0)]], const device bfloat* bank1 [[buffer(1)]],
    const device float* q_mlp [[buffer(2)]], const device float* kv [[buffer(3)]],
    device float* out [[buffer(4)]], constant InkLoopArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint work [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = work / INKLING_LOOP_MESSAGES;
  const uint entry = work - row * INKLING_LOOP_MESSAGES;
  if (row >= a.rows) return;
  const uint layer = entry / 4;
  const uint role = entry - 4 * layer;
  const device float* queries = (role == 0 || role == 3) ? q_mlp : kv;
  // q/mlp and k/v occupy even/odd rows respectively.
  const device float* query = queries + size_t(2 * layer + (role >= 2 ? 1 : 0)) * INKLING_LOOP_H;
  const device bfloat* banks[2] = { bank0 + size_t(row) * INKLING_LOOP_H,
                                    bank1 + size_t(row) * INKLING_LOOP_H };
  threadgroup float sums[2 * 8], inverse[2], dots[2 * 8], weights[2];
  for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint d = t; d < INKLING_LOOP_H; d += 256) {
      const float x = float(banks[s][d]);
      ss += x * x;
    }
    const float reduced = simd_sum(ss);
    if (lane == 0) sums[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 8; ++j) ss += sums[s * 8 + j];
    inverse[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 0; s < 2; ++s) {
    float dot = 0.0f;
    for (uint d = t; d < INKLING_LOOP_H; d += 256) {
      const float beta = float(banks[s][d]) * inverse[s];
      dot += query[d] * beta;
    }
    const float reduced = simd_sum(dot);
    if (lane == 0) dots[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float logits[2];
    float peak = -INFINITY;
    for (uint s = 0; s < 2; ++s) {
      float dot = 0.0f;
      for (uint j = 0; j < 8; ++j) dot += dots[s * 8 + j];
      weights[s] = dot * INKLING_LOOP_SCALE;
      logits[s] = weights[s];
      peak = max(peak, logits[s]);
    }
    float den = 0.0f;
    for (uint s = 0; s < 2; ++s) {
      weights[s] = precise::exp(logits[s] - peak);
      den += weights[s];
    }
    for (uint s = 0; s < 2; ++s) weights[s] = precise::divide(weights[s], den);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    const float beta0 = float(banks[0][d]) * inverse[0];
    const float beta1 = float(banks[1][d]) * inverse[1];
    out[(size_t(row) * INKLING_LOOP_MESSAGES + entry) * INKLING_LOOP_H + d] =
        weights[0] * beta0 + weights[1] * beta1;
  }
}


// A row's whole gate in one launch: h and m to unit RMS BF16, the rank-64 zeta dots, the four-step blend,
// and when the caller passes a norm the row's RMS scale against it to BF16. The phases keep the old three
// launches' reduction order; `normed` picks the last phase. `out` always receives the FP32 blend row.
struct InkGateArgs { uint rows, stride, normed; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_gate(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
    const device bfloat* down [[buffer(2)]], const device bfloat* forget [[buffer(3)]],
    const device bfloat* gain [[buffer(4)]], device float* out [[buffer(5)]],
    const device bfloat* norm_w [[buffer(6)]], device bfloat* norm_out [[buffer(7)]],
    constant InkGateArgs& a [[buffer(8)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device float* inputs[2] = { h + size_t(row) * INKLING_LOOP_H,
                                    m + size_t(row) * a.stride };
  threadgroup float sums[2 * 8], inverse[2], part[8];
  threadgroup bfloat normalized[2 * INKLING_LOOP_H], zeta[INKLING_LOOP_RANK];
  threadgroup float blended[INKLING_LOOP_H];
  for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint d = t; d < INKLING_LOOP_H; d += 256) {
      const float x = inputs[s][d];
      ss += x * x;
    }
    const float reduced = simd_sum(ss);
    if (lane == 0) sums[s * 8 + sg] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 8; ++j) ss += sums[s * 8 + j];
    inverse[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    normalized[d] = bfloat(inputs[0][d] * inverse[0]);
    normalized[INKLING_LOOP_H + d] = bfloat(inputs[1][d] * inverse[1]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // A simdgroup takes eight of the ranks in sequence, each dot in the old launch's lane order.
  for (uint band = 0; band < 8; ++band) {
    const uint r = band * 8 + sg;
    const device bfloat* w = down + size_t(r) * (2 * INKLING_LOOP_H);
    float acc = 0.0f;
    for (uint i = 8 * lane; i < 2 * INKLING_LOOP_H; i += 256) {
      const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
      const float4 x0 = float4(*(const threadgroup bfloat4*)(normalized + i)), x1 = float4(*(const threadgroup bfloat4*)(normalized + i + 4));
      acc += dot(w0, x0) + dot(w1, x1);
    }
    const float sum = simd_sum(acc);
    if (lane == 0) zeta[r] = bfloat(sum);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Each output's forget and gain dots over zeta in rank order, then the four-step blend.
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    float fdot = 0.0f, gdot = 0.0f;
    for (uint r = 0; r < INKLING_LOOP_RANK; ++r) {
      const float z = float(zeta[r]);
      fdot += float(forget[size_t(d) * INKLING_LOOP_RANK + r]) * z;
      gdot += float(gain[size_t(d) * INKLING_LOOP_RANK + r]) * z;
    }
    const float f = precise::tanh(fdot);
    const float g = precise::tanh(gdot);
    // Four separately rounded binary32 steps; do not fold these into one expression.
    const float keep = 1.0f - f;
    const float left = keep * h[size_t(row) * INKLING_LOOP_H + d];
    const float right = g * m[size_t(row) * a.stride + d];
    const float v = left + right;
    blended[d] = v;
    out[size_t(row) * INKLING_LOOP_H + d] = v;
  }
  if (a.normed == 0u) return;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ns = 0.0f;
  for (uint d = t; d < INKLING_LOOP_H; d += 256) ns = fma(blended[d], blended[d], ns);
  const float nreduced = simd_sum(ns);
  if (lane == 0) part[sg] = nreduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ntotal = 0.0f;
  for (uint j = 0; j < 8; ++j) ntotal += part[j];
  const float ninv = precise::rsqrt(ntotal / 2304.0f + INKLING_LOOP_EPS);
  for (uint d = t; d < INKLING_LOOP_H; d += 256)
    norm_out[size_t(row) * INKLING_LOOP_H + d] = bfloat(float(norm_w[d]) * (blended[d] * ninv));
}
