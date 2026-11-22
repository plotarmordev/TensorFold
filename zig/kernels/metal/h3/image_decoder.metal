// MiniMax-H3-Image's image decoder (the VAE's 2D ViT decoder) in bf16 with fp32 sums: tensor-unit products where
// H3D_TENSOR is set (the M5), main's simdgroup-matrix products and separate SwiGLU on every other Apple GPU.
#include <metal_stdlib>
using namespace metal;
#if H3D_TENSOR
// The products' fragments on the M5's tensor units.
#include "../nax.h"
#endif

#define H3D_UNROLL _Pragma("clang loop unroll(full)")
// The attention's key blocks: 9 of 32 keys, 288 past a 16 x 16 tile's 261 tokens, staged in fp32 rows of 68.
#define H3D_BLOCKS 9
#define H3D_LD 68

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

#if H3D_TENSOR
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
      frag_get_t(wb[k2][n], B, int(p.k), 16 * n, k0 + 16 * k2, home);
    TF_UNROLL
    for (short k2 = 0; k2 < 2; k2++) TF_UNROLL for (short m = 0; m < 2; m++)
      mma_16x32<false, true>(acc[m][0], acc[m][1], a[m][k2], wb[k2][0], wb[k2][1]);
    if (gated) {
      TF_UNROLL
      for (short k2 = 0; k2 < 2; k2++) TF_UNROLL for (short n = 0; n < 2; n++)
        frag_get_t(wb[k2][n], V, int(p.k), 16 * n, k0 + 16 * k2, home);
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
#else
// y = bf16(x w^T + b), x [m, k] and w [n, k] bf16, fp32 sums; mode 1 adds bf16(y s) into y's old value (LayerScale).
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void h3d_gemm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    const device bfloat* s [[buffer(3)]], device bfloat* y [[buffer(4)]], constant H3dShape& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint BK = 32;
  constexpr uint LD = BK + 4;
  threadgroup float buf[2 * 64 * LD];
  threadgroup float* as = buf;
  threadgroup float* bs = buf + 64 * LD;
  const uint m0 = tg.y * 64;
  const uint n0 = tg.x * 64;
  // a thread stages 16 values of one row of each operand; a simdgroup owns a 32 x 32 quarter of the output
  const uint lr = t / 2;
  const uint lc = (t % 2) * 16;
  const uint sr = (sg / 2) * 32;
  const uint sc = (sg % 2) * 32;
  const bool live = m0 + lr < p.m;
  const device bfloat4* xa = (const device bfloat4*)(x + ulong(live ? m0 + lr : 0) * p.k + lc);
  const device bfloat4* wb = (const device bfloat4*)(w + ulong(n0 + lr) * p.k + lc);
  simdgroup_float8x8 acc[4][4];
  H3D_UNROLL
  for (uint i = 0; i < 4; i++) {
    H3D_UNROLL
    for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);
  }
  for (uint k0 = 0; k0 < p.k; k0 += BK) {
    threadgroup float4* da = (threadgroup float4*)(as + lr * LD + lc);
    threadgroup float4* db = (threadgroup float4*)(bs + lr * LD + lc);
    H3D_UNROLL
    for (uint i = 0; i < 4; i++) {
      da[i] = live ? float4(xa[k0 / 4 + i]) : float4(0.0f);
      db[i] = float4(wb[k0 / 4 + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    H3D_UNROLL
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 am[4];
      simdgroup_float8x8 bm[4];
      H3D_UNROLL
      for (uint i = 0; i < 4; i++) simdgroup_load(am[i], as + (sr + i * 8) * LD + kk, LD);
      H3D_UNROLL
      for (uint j = 0; j < 4; j++) simdgroup_load(bm[j], bs + (sc + j * 8) * LD + kk, LD, ulong2(0, 0), true);
      H3D_UNROLL
      for (uint i = 0; i < 4; i++) {
        H3D_UNROLL
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // the 64 x 64 sums through threadgroup memory (the staging buffers are free now), then the epilogue
  threadgroup float* cs = buf;
  H3D_UNROLL
  for (uint i = 0; i < 4; i++) {
    H3D_UNROLL
    for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], cs + (sr + i * 8) * 64 + sc + j * 8, 64);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = t; e < 64 * 64; e += 128) {
    const uint row = m0 + e / 64;
    if (row >= p.m) continue;
    const uint col = n0 + e % 64;
    const ulong at = ulong(row) * p.n + col;
    const float out = float(bfloat(cs[e] + float(b[col])));
    if (p.mode == 0) {
      y[at] = bfloat(out);
    } else {
      y[at] = bfloat(float(y[at]) + float(bfloat(out * float(s[col]))));
    }
  }
}
#endif

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

// Keys or values [key0, key0 + 32) of one (tile, head) into `kv` in fp32, zero past T. 128 threads.
inline void h3d_stage(threadgroup float* kv, const device bfloat* src, uint key0, uint T, uint tid) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 32 * 16; i += 128) {
    const uint key = key0 + i / 16;
    const float4 x = key < T ? float4(((const device bfloat4*)(src + ulong(key) * 64))[i % 16]) : float4(0.0f);
    *(threadgroup float4*)(kv + (i / 16) * H3D_LD + (i % 16) * 4) = x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// softmax(q k^T / 8) v on 8 x 8 simdgroup matrices; fp32 scores, softmax and sums. Each simdgroup holds 8 queries'
// scores, then their weights, in registers; only k and v pass through threadgroup memory. [T / 32, 16, tiles].
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void h3d_attention(
    const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
    device bfloat* o [[buffer(3)]], constant H3dShape& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float kv[32 * H3D_LD];
  const uint T = p.tokens;
  const ulong base = (ulong(tg.z) * 16 + tg.y) * T * 64;
  // This lane's place in an 8 x 8 fragment: row `r`, columns `c` and `c + 1`; a row's lanes differ by ^1 and ^8.
  const uint r = (lane / 16) * 4 + (lane / 2) % 4;
  const uint c = (lane / 8) % 2 * 4 + (lane % 2) * 2;
  const uint query = tg.x * 32 + sg * 8 + r;
  simdgroup_float8x8 qf[8];
  const device bfloat* qr = q + base + ulong(min(query, T - 1)) * 64 + c;
  H3D_UNROLL
  for (uint d = 0; d < 8; d++) {
    qf[d].thread_elements()[0] = float(qr[d * 8]);
    qf[d].thread_elements()[1] = float(qr[d * 8 + 1]);
  }
  simdgroup_float8x8 s[H3D_BLOCKS][4];
  H3D_UNROLL
  for (uint b = 0; b < H3D_BLOCKS; b++) {
    h3d_stage(kv, k + base, b * 32, T, tid);
    H3D_UNROLL
    for (uint j = 0; j < 4; j++) {
      s[b][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      H3D_UNROLL
      for (uint d = 0; d < 8; d++) {
        simdgroup_float8x8 kt;
        simdgroup_load(kt, kv + j * 8 * H3D_LD + d * 8, H3D_LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[b][j], qf[d], kt, s[b][j]);
      }
    }
  }
  float top = -INFINITY;
  H3D_UNROLL
  for (uint b = 0; b < H3D_BLOCKS; b++) {
    H3D_UNROLL
    for (uint j = 0; j < 4; j++) {
      H3D_UNROLL
      for (uint e = 0; e < 2; e++) {
        const float x = b * 32 + j * 8 + c + e < T ? s[b][j].thread_elements()[e] * p.scale : -INFINITY;
        s[b][j].thread_elements()[e] = x;
        top = max(top, x);
      }
    }
  }
  top = max(top, simd_shuffle_xor(top, 1));
  top = max(top, simd_shuffle_xor(top, 8));
  float total = 0.0f;
  H3D_UNROLL
  for (uint b = 0; b < H3D_BLOCKS; b++) {
    H3D_UNROLL
    for (uint j = 0; j < 4; j++) {
      H3D_UNROLL
      for (uint e = 0; e < 2; e++) {
        const float w = precise::exp(s[b][j].thread_elements()[e] - top);
        s[b][j].thread_elements()[e] = w;
        total += w;
      }
    }
  }
  total += simd_shuffle_xor(total, 1);
  total += simd_shuffle_xor(total, 8);
  simdgroup_float8x8 acc[8];
  H3D_UNROLL
  for (uint d = 0; d < 8; d++) acc[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  H3D_UNROLL
  for (uint b = 0; b < H3D_BLOCKS; b++) {
    h3d_stage(kv, v + base, b * 32, T, tid);
    H3D_UNROLL
    for (uint j = 0; j < 4; j++) {
      H3D_UNROLL
      for (uint d = 0; d < 8; d++) {
        simdgroup_float8x8 vf;
        simdgroup_load(vf, kv + j * 8 * H3D_LD + d * 8, H3D_LD);
        simdgroup_multiply_accumulate(acc[d], s[b][j], vf, acc[d]);
      }
    }
  }
  if (query >= T) return;
  device bfloat* out = o + (ulong(tg.z) * T + query) * 1024 + tg.y * 64 + c;
  H3D_UNROLL
  for (uint d = 0; d < 8; d++) {
    out[d * 8] = bfloat(acc[d].thread_elements()[0] / total);
    out[d * 8 + 1] = bfloat(acc[d].thread_elements()[1] / total);
  }
}

// act = bf16(bf16(SiLU(gate)) * value), w1's rows read as [gate 4,096 | value 4,096]. Threads [4,096, m].
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void h3d_swiglu(
    const device bfloat* wide [[buffer(0)]], device bfloat* act [[buffer(1)]], constant H3dShape& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= 4096 || at.y >= p.m) return;
  const device bfloat* r = wide + ulong(at.y) * 8192;
  const float g = float(r[at.x]);
  const float silu = float(bfloat(g / (1.0f + precise::exp(-g))));
  act[ulong(at.y) * 4096 + at.x] = bfloat(silu * float(r[4096 + at.x]));
}
