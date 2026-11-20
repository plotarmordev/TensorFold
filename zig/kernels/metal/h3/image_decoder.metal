// MiniMax-H3-Image's image decoder (the VAE's 2D ViT decoder) in bf16 with fp32 sums, on any Apple GPU.
#include <metal_stdlib>
using namespace metal;
// The products' fragments: the M5's tensor units, or simdgroup matrices where TF_SIMD_FRAGS is set.
#include "../nax.h"

#define H3D_UNROLL _Pragma("clang loop unroll(full)")
// Keys a lane holds in the attention: 9 a lane, 288 a simdgroup, past a 16 x 16 tile's 261 tokens.
#define H3D_SLOTS 9

// One parameter block for every kernel: the pass's tiles and their geometry, a product's size, the norms' epsilon.
struct H3dShape {
  uint tiles;
  uint h;
  uint w;
  uint tokens;
  uint rows;
  uint cols;
  uint m;
  uint n;
  uint k;
  uint mode;
  float eps;
  float scale;
};

// Each tile's cells rounded to bf16, then post_quant_conv (24 to 24 with bias) to bf16: tok [tiles * h * w, 24].
[[max_total_threads_per_threadgroup(192)]] [[kernel]] void h3d_post_quant(
    const device float* lat [[buffer(0)]], const device uint2* org [[buffer(1)]], const device bfloat* pw [[buffer(2)]],
    const device bfloat* pb [[buffer(3)]], device bfloat* tok [[buffer(4)]], constant H3dShape& p [[buffer(5)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint cells = p.h * p.w;
  if (at.x >= 24 || at.y >= p.tiles * cells) return;
  const uint tile = at.y / cells;
  const uint i = at.y % cells;
  const ulong r = org[tile].x + i / p.w;
  const ulong c = org[tile].y + i % p.w;
  const ulong plane = ulong(p.rows) * p.cols;
  float acc = 0.0f;
  for (uint k = 0; k < 24; k++) acc += float(pw[at.x * 24 + k]) * float(bfloat(lat[k * plane + r * p.cols + c]));
  tok[ulong(at.y) * 24 + at.x] = bfloat(acc + float(pb[at.x]));
}

// A tile's sequence: x_embedder (24 to 1,024 with bias) on its cells to bf16, then the 4 registers and a zero token.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void h3d_embed(
    const device bfloat* tok [[buffer(0)]], const device bfloat* ew [[buffer(1)]], const device bfloat* eb [[buffer(2)]],
    const device bfloat* reg [[buffer(3)]], device bfloat* x [[buffer(4)]], constant H3dShape& p [[buffer(5)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= 1024 || at.y >= p.tiles * p.tokens) return;
  const uint cells = p.h * p.w;
  const uint tile = at.y / p.tokens;
  const uint t = at.y % p.tokens;
  bfloat v = bfloat(0.0f);
  if (t < cells) {
    const device bfloat* in = tok + (ulong(tile) * cells + t) * 24;
    float acc = 0.0f;
    for (uint k = 0; k < 24; k++) acc += float(ew[at.x * 24 + k]) * float(in[k]);
    v = bfloat(acc + float(eb[at.x]));
  } else if (t < cells + 4) {
    v = reg[(t - cells) * 1024 + at.x];
  }
  x[ulong(at.y) * 1024 + at.x] = v;
}

// The sum of `v` over a threadgroup of 256 (8 simdgroups), the same on every thread.
inline float h3d_total(float v, threadgroup float* part, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  H3D_UNROLL
  for (uint i = 0; i < 8; i++) total += part[i];
  return total;
}

// RMSNorm, one row of 1,024 a threadgroup of 256: y = bf16((x * rsqrt(mean(x^2) + eps)) * g), all in fp32.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void h3d_rms_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* g [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant H3dShape& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8];
  const device bfloat* xr = x + ulong(row) * 1024;
  float v[4];
  float sq = 0.0f;
  H3D_UNROLL
  for (uint i = 0; i < 4; i++) {
    v[i] = float(xr[t + 256 * i]);
    sq += v[i] * v[i];
  }
  const float inv = precise::rsqrt(h3d_total(sq, part, lane, sg) / 1024.0f + p.eps);
  H3D_UNROLL
  for (uint i = 0; i < 4; i++) y[ulong(row) * 1024 + t + 256 * i] = bfloat(v[i] * inv * float(g[t + 256 * i]));
}

// norm_out: LayerNorm in fp32 (weight and bias), then bf16 as proj_out's input. One row a threadgroup of 256.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void h3d_layer_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* g [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant H3dShape& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8];
  threadgroup float squares[8];
  const device bfloat* xr = x + ulong(row) * 1024;
  float v[4];
  float s = 0.0f;
  H3D_UNROLL
  for (uint i = 0; i < 4; i++) {
    v[i] = float(xr[t + 256 * i]);
    s += v[i];
  }
  const float mean = h3d_total(s, sums, lane, sg) / 1024.0f;
  float sq = 0.0f;
  H3D_UNROLL
  for (uint i = 0; i < 4; i++) sq += (v[i] - mean) * (v[i] - mean);
  const float rstd = precise::rsqrt(h3d_total(sq, squares, lane, sg) / 1024.0f + p.eps);
  H3D_UNROLL
  for (uint i = 0; i < 4; i++) {
    const uint c = t + 256 * i;
    y[ulong(row) * 1024 + c] = bfloat((v[i] - mean) * rstd * float(g[c]) + float(b[c]));
  }
}

// y = bf16(x w^T + b), x [m, k] and w [n, k] bf16, fp32 sums on nax.h's fragments; mode 1 adds bf16(y s) into y's
// old value (LayerScale); mode 2 is w1's SwiGLU: w and b hold [gate n | value n] and y = bf16(bf16(SiLU(gate)) * value),
// so the 2n-wide rows never reach memory. A threadgroup spans 64 rows x 64 columns, each of its four warps 32 x 32.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void h3d_gemm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    const device bfloat* s [[buffer(3)]], device bfloat* y [[buffer(4)]], constant H3dShape& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using namespace tfp;
  const int row = int(tg.y) * 64 + 32 * int(sg >> 1);
  const int col = int(tg.x) * 64 + 32 * int(sg & 1);
  const bool gated = p.mode == 2;
  const short2 home = frag_home(ushort(lane));
  const device bfloat* A = x + ulong(row) * p.k;
  const device bfloat* B = w + ulong(col) * p.k;
  const device bfloat* V = B + ulong(p.n) * p.k;
  frag<float> acc[2][2];
  frag<float> val[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) TF_UNROLL for (short n = 0; n < 2; n++) {
    acc[m][n] = frag<float>(0.0f);
    val[m][n] = frag<float>(0.0f);
  }
  for (int k0 = 0; k0 < int(p.k); k0 += 32) {
    frag<bfloat> a[2][2];
    frag<bfloat> wb[2][2];
    TF_UNROLL
    for (short m = 0; m < 2; m++) TF_UNROLL for (short k2 = 0; k2 < 2; k2++)
      frag_get_in(a[m][k2], A, int(p.k), 16 * m, k0 + 16 * k2, home, int(p.m) - row, 1 << 30);
    TF_UNROLL
    for (short k2 = 0; k2 < 2; k2++) TF_UNROLL for (short n = 0; n < 2; n++)
      frag_get(wb[k2][n], B, int(p.k), 16 * n, k0 + 16 * k2, home);
    TF_UNROLL
    for (short k2 = 0; k2 < 2; k2++) TF_UNROLL for (short m = 0; m < 2; m++)
      mma_16x32<false, true>(acc[m][0], acc[m][1], a[m][k2], wb[k2][0], wb[k2][1]);
    if (gated) {
      TF_UNROLL
      for (short k2 = 0; k2 < 2; k2++) TF_UNROLL for (short n = 0; n < 2; n++)
        frag_get(wb[k2][n], V, int(p.k), 16 * n, k0 + 16 * k2, home);
      TF_UNROLL
      for (short k2 = 0; k2 < 2; k2++) TF_UNROLL for (short m = 0; m < 2; m++)
        mma_16x32<false, true>(val[m][0], val[m][1], a[m][k2], wb[k2][0], wb[k2][1]);
    }
  }
  TF_UNROLL
  for (short m = 0; m < 2; m++) TF_UNROLL for (short h2 = 0; h2 < 2; h2++) TF_UNROLL for (short e = 0; e < 8; e++) {
    const int rr = row + 16 * m + home.y + (e >> 2) * 8;
    if (rr >= int(p.m)) continue;
    const int cc = col + 16 * h2 + home.x + TF_COL(e);
    const float out = float(bfloat(acc[m][h2][e] + float(b[cc])));
    const ulong at = ulong(rr) * p.n + uint(cc);
    if (p.mode == 0) {
      y[at] = bfloat(out);
    } else if (gated) {
      const float value = float(bfloat(val[m][h2][e] + float(b[p.n + cc])));
      y[at] = bfloat(float(bfloat(out / (1.0f + precise::exp(-out)))) * value);
    } else {
      y[at] = bfloat(float(y[at]) + float(bfloat(out * float(s[cc]))));
    }
  }
}

// Per row and head: q and k RMS-normed (no weight) to bf16, their first 32 dims turned half-style in bf16; v copied.
[[max_total_threads_per_threadgroup(32)]] [[kernel]] void h3d_heads(
    const device bfloat* qkv [[buffer(0)]], const device bfloat* cs [[buffer(1)]], const device bfloat* sn [[buffer(2)]],
    device bfloat* q [[buffer(3)]], device bfloat* k [[buffer(4)]], device bfloat* v [[buffer(5)]],
    constant H3dShape& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.x;
  const uint head = tg.y;
  const uint tile = row / p.tokens;
  const uint t = row % p.tokens;
  const device bfloat* src = qkv + ulong(row) * 3072 + head * 192;
  const ulong to = ((ulong(tile) * 16 + head) * p.tokens + t) * 64;
  const float c = float(cs[t * 32 + lane]);
  const float s = float(sn[t * 32 + lane]);
  for (uint part = 0; part < 2; part++) {
    const float a = float(src[part * 64 + lane]);
    const float z = float(src[part * 64 + 32 + lane]);
    const float inv = precise::rsqrt(simd_sum(a * a + z * z) / 64.0f + p.eps);
    const float na = float(bfloat(a * inv));
    const float other = simd_shuffle_xor(na, 16);
    const float turned = lane < 16 ? -other : other;
    device bfloat* out = (part == 0 ? q : k) + to;
    out[lane] = bfloat(float(bfloat(na * c)) + float(bfloat(turned * s)));
    out[32 + lane] = bfloat(z * inv);
  }
  v[to + lane] = src[128 + lane];
  v[to + 32 + lane] = src[160 + lane];
}

// softmax(q k^T / 8) v, a simdgroup a query over its tile's keys; fp32 scores, softmax and sums. [T / 8, 16, tiles].
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void h3d_attention(
    const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
    device bfloat* o [[buffer(3)]], constant H3dShape& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float qs[8 * 64];
  threadgroup float ps[8 * 32 * H3D_SLOTS];
  const uint T = p.tokens;
  const uint query = tg.x * 8 + sg;
  if (query >= T) return;
  const uint head = tg.y;
  const uint tile = tg.z;
  const ulong base = (ulong(tile) * 16 + head) * T * 64;
  threadgroup float* qr = qs + sg * 64;
  threadgroup float* pr = ps + sg * 32 * H3D_SLOTS;
  qr[lane] = float(q[base + ulong(query) * 64 + lane]);
  qr[lane + 32] = float(q[base + ulong(query) * 64 + 32 + lane]);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float sc[H3D_SLOTS];
  float top = -INFINITY;
  H3D_UNROLL
  for (uint i = 0; i < H3D_SLOTS; i++) {
    const uint key = lane + 32 * i;
    sc[i] = -INFINITY;
    if (key < T) {
      const device bfloat4* kr = (const device bfloat4*)(k + base + ulong(key) * 64);
      const threadgroup float4* q4 = (const threadgroup float4*)qr;
      float4 dot = float4(0.0f);
      H3D_UNROLL
      for (uint d = 0; d < 16; d++) dot += float4(kr[d]) * q4[d];
      sc[i] = (dot.x + dot.y + dot.z + dot.w) * p.scale;
      top = max(top, sc[i]);
    }
  }
  top = simd_max(top);
  float total = 0.0f;
  H3D_UNROLL
  for (uint i = 0; i < H3D_SLOTS; i++) {
    const uint key = lane + 32 * i;
    if (key < T) {
      const float e = precise::exp(sc[i] - top);
      pr[key] = e;
      total += e;
    }
  }
  total = simd_sum(total);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float2 a = float2(0.0f);
  for (uint key = 0; key < T; key++) {
    const device bfloat2* vr = (const device bfloat2*)(v + base + ulong(key) * 64 + 2 * lane);
    a += float2(vr[0]) * pr[key];
  }
  const ulong at = (ulong(tile) * T + query) * 1024 + head * 64 + 2 * lane;
  o[at] = bfloat(a.x / total);
  o[at + 1] = bfloat(a.y / total);
}
