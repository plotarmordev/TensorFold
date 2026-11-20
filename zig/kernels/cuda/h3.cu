// MiniMax H3 / FastH3's transformer blocks on CUDA in the checkpoint's bf16: the row, routing and attention kernels.
#include "h3_attention.cuh"
#include "common/reduce.cuh"

namespace tf_h3 {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }

// A value every thread of the block agrees on: the sum of what each holds.
using tf_common::block_sum_serial;

using tf_common::warp_max;
using tf_common::warp_sum;

// The modulated norm of one value, rounded to bf16 where torch's bf16 ops round: norm, 1 + scale, product, shift.
__device__ __forceinline__ float modulated(float x, float inv, float w, float scale, float shift) {
    return bf(bf(bf(x * inv * w) * bf(1.0f + bf(scale))) + bf(shift));
}

} // namespace tf_h3

using namespace tf_h3;

extern "C" {

__global__ void __launch_bounds__(256) h3_norm(__nv_bfloat16* X, const __nv_bfloat16* Y, const __nv_bfloat16* NW,
                                               const float* GT, const float* TAB, const int* LINE, __nv_bfloat16* O,
                                               int C, int gate, int scale, int shift, float eps) {
    __shared__ float scratch[8];
    const int row = blockIdx.x, line = LINE[row];
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    float sq = 0.0f;
    if (gate >= 0) {
        const float* g = GT + (static_cast<size_t>(line) * 6 + gate) * C;
        const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float v = bf(f(x[c]) + f(y[c]) * bf(g[c]));
            x[c] = __float2bfloat16(v);
            sq += v * v;
        }
    } else {
        for (int c = threadIdx.x; c < C; c += blockDim.x) sq += f(x[c]) * f(x[c]);
    }
    const float inv = rsqrtf(block_sum_serial(sq, scratch) / float(C) + eps);
    const float* s = TAB + (static_cast<size_t>(line) * 6 + scale) * C;
    const float* h = TAB + (static_cast<size_t>(line) * 6 + shift) * C;
    __nv_bfloat16* o = O + static_cast<size_t>(row) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) o[c] = __float2bfloat16(modulated(f(x[c]), inv, f(NW[c]), s[c], h[c]));
}

// X += GT[line, part] * Y in one rounding, as torch's addcmul: the last block's MLP branch. Grid [M], 256 threads.
__global__ void __launch_bounds__(256) h3_gate_add(__nv_bfloat16* X, const __nv_bfloat16* Y, const float* GT,
                                                   const int* LINE, int C, int part) {
    const int row = blockIdx.x;
    const float* g = GT + (static_cast<size_t>(LINE[row]) * 6 + part) * C;
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) x[c] = __float2bfloat16(f(x[c]) + f(y[c]) * bf(g[c]));
}

// q, k, v to bf16 in tile order: head RMSNorm, then split-half rotary rounded as the reference's bf16 tables round it.
__global__ void __launch_bounds__(256) h3_heads(
        const __nv_bfloat16* QKV, const __nv_bfloat16* NQ, const __nv_bfloat16* NK, const float* COS, const float* SIN,
        const int* ROWOF, __nv_bfloat16* TQ, __nv_bfloat16* TK, __nv_bfloat16* TV, int H, int stride, int ROT, int slots,
        float eps) {
    const int tile = blockIdx.x, s = threadIdx.x >> 2, c0 = (threadIdx.x & 3) * 32, slot = tile * 64 + s;
    const int row = ROWOF[slot];
    const bool live = row >= 0;
    const float* cs = COS + static_cast<size_t>(live ? row : 0) * ROT;
    const float* sn = SIN + static_cast<size_t>(live ? row : 0) * ROT;
    for (int head = 0; head < H; ++head) {
        const __nv_bfloat16* qr = QKV + static_cast<size_t>(live ? row : 0) * stride + head * 128;
        const __nv_bfloat16* kr = qr + H * 128;
        const __nv_bfloat16* vr = kr + H * 128;
        float q[32], k[32], sq = 0.0f, sk = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            q[j] = live ? f(qr[c0 + j]) : 0.0f;
            k[j] = live ? f(kr[c0 + j]) : 0.0f;
            sq += q[j] * q[j];
            sk += k[j] * k[j];
        }
        // a row's four quarters sit in neighbouring lanes
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        sk += __shfl_xor_sync(0xffffffffu, sk, 2);
        if (!live) continue;
        const float qi = rsqrtf(sq / 128.0f + eps), ki = rsqrtf(sk / 128.0f + eps);
        const size_t at = (static_cast<size_t>(head) * slots + slot) * 128 + c0;
#pragma unroll
        for (int j = 0; j < 32; j += 2) {
            float a[2], b[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = c0 + j + e;
                a[e] = bf(q[j + e] * qi * f(NQ[c]));
                b[e] = bf(k[j + e] * ki * f(NK[c]));
                if (c < 2 * ROT) {
                    const bool low = c < ROT;
                    const int p = low ? c + ROT : c - ROT, t = low ? c : c - ROT;
                    const float sign = low ? -1.0f : 1.0f, co = bf(cs[t]), si = bf(sn[t]);
                    const float qp = bf(f(qr[p]) * qi * f(NQ[p])), kp = bf(f(kr[p]) * ki * f(NK[p]));
                    a[e] = bf(a[e] * co) + bf(sign * qp * si);
                    b[e] = bf(b[e] * co) + bf(sign * kp * si);
                }
            }
            *reinterpret_cast<uint32_t*>(TQ + at + j) = pack2(a[0], a[1]);
            *reinterpret_cast<uint32_t*>(TK + at + j) = pack2(b[0], b[1]);
            *reinterpret_cast<uint32_t*>(TV + at + j) = *reinterpret_cast<const uint32_t*>(vr + c0 + j);
        }
    }
}

// A bf16 from a pack2 word by element index.
__device__ __forceinline__ float unpack(const uint32_t* p, int e) {
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(p + (e >> 1));
    return f(e & 1 ? v.y : v.x);
}

// h3_heads over an int8 product's int32 sums: each value is dequantized as h3_dequantize rounds it
// (ACC SX[row] SW[col] to bf16), then the same head RMSNorm and split-half rotary run, so TQ/TK/TV match
// h3_dequantize + h3_heads without the bf16 qkvc rows between them. Once per row, like h3_heads — card 120
// measured folding this into attention's tile loop 11x slower. Grid [tiles], 256 threads.

// q/k in registers rounded to bf16 two a word, rows read 16B at a time; card 128's 252-register float arrays spilled.

// A quarter's rotary partner lies in another quarter: the partner's int32 group is reloaded and dequantized again.
__global__ void __launch_bounds__(256) h3_heads_i8(
        const int* ACC, const float* SX, const float* SW,
        const __nv_bfloat16* NQ, const __nv_bfloat16* NK, const float* COS, const float* SIN,
        const int* ROWOF, __nv_bfloat16* TQ, __nv_bfloat16* TK, __nv_bfloat16* TV, int H, int stride, int ROT, int slots,
        float eps) {
    const int tile = blockIdx.x, s = threadIdx.x >> 2, c0 = (threadIdx.x & 3) * 32, slot = tile * 64 + s;
    const int row = ROWOF[slot];
    const bool live = row >= 0;
    const float sx = SX[live ? row : 0];
    const float* cs = COS + static_cast<size_t>(live ? row : 0) * ROT;
    const float* sn = SIN + static_cast<size_t>(live ? row : 0) * ROT;
    // the rotary partner of c0 + 32h + j sits ±ROT over; ROT is a four-multiple, so each 4-wide group is one side
    for (int head = 0; head < H; ++head) {
        const int* qr = ACC + static_cast<size_t>(live ? row : 0) * stride + head * 128;
        const int* kr = qr + H * 128;
        const int* vr = kr + H * 128;
        const float* wq = SW + head * 128;
        const float* wk = wq + H * 128;
        const float* wv = wk + H * 128;
        // q and k held rounded to bf16 (two a word), so the norm sums and the rotary see h3_dequantize's values
        uint32_t qp[16], kp[16];
        float sq = 0.0f, sk = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int4 a4 = live ? reinterpret_cast<const int4*>(qr + c0)[i] : make_int4(0, 0, 0, 0);
            const int4 b4 = live ? reinterpret_cast<const int4*>(kr + c0)[i] : make_int4(0, 0, 0, 0);
            const float4 wa = live ? reinterpret_cast<const float4*>(wq + c0)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 wb = live ? reinterpret_cast<const float4*>(wk + c0)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
            const float q0 = bf(a4.x * sx * wa.x), q1 = bf(a4.y * sx * wa.y), q2 = bf(a4.z * sx * wa.z), q3 = bf(a4.w * sx * wa.w);
            const float k0 = bf(b4.x * sx * wb.x), k1 = bf(b4.y * sx * wb.y), k2 = bf(b4.z * sx * wb.z), k3 = bf(b4.w * sx * wb.w);
            sq += q0 * q0;
            sq += q1 * q1;
            sq += q2 * q2;
            sq += q3 * q3;
            sk += k0 * k0;
            sk += k1 * k1;
            sk += k2 * k2;
            sk += k3 * k3;
            qp[2 * i] = pack2(q0, q1);
            qp[2 * i + 1] = pack2(q2, q3);
            kp[2 * i] = pack2(k0, k1);
            kp[2 * i + 1] = pack2(k2, k3);
        }
        // a row's four quarters sit in neighbouring lanes
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        sk += __shfl_xor_sync(0xffffffffu, sk, 2);
        if (!live) continue;
        const float qi = rsqrtf(sq / 128.0f + eps), ki = rsqrtf(sk / 128.0f + eps);
        const size_t at = (static_cast<size_t>(head) * slots + slot) * 128 + c0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 wv4 = reinterpret_cast<const float4*>(wv + c0)[i];
            const int4 va = reinterpret_cast<const int4*>(vr + c0)[i];
            float a[4], b[4], vc[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int c = c0 + 4 * i + e;
                a[e] = bf(unpack(qp, 4 * i + e) * qi * f(NQ[c]));
                b[e] = bf(unpack(kp, 4 * i + e) * ki * f(NK[c]));
            }
            vc[0] = bf(va.x * sx * wv4.x);
            vc[1] = bf(va.y * sx * wv4.y);
            vc[2] = bf(va.z * sx * wv4.z);
            vc[3] = bf(va.w * sx * wv4.w);
            const int cb = c0 + 4 * i;
            if (cb < 2 * ROT) {
                const bool grp_low = cb < ROT; // uniform per group: cb and ROT are multiples of four
                const int p0 = grp_low ? cb + ROT : cb - ROT;
                const int t0 = grp_low ? cb : cb - ROT;
                const int4 pa = *reinterpret_cast<const int4*>(qr + p0);
                const int4 pb = *reinterpret_cast<const int4*>(kr + p0);
                const float4 wpa = *reinterpret_cast<const float4*>(wq + p0);
                const float4 wpb = *reinterpret_cast<const float4*>(wk + p0);
                const uint2 npa = *reinterpret_cast<const uint2*>(NQ + p0);
                const uint2 npb = *reinterpret_cast<const uint2*>(NK + p0);
                const uint2 npa2 = *reinterpret_cast<const uint2*>(NQ + p0 + 2);
                const uint2 npb2 = *reinterpret_cast<const uint2*>(NK + p0 + 2);
                const float4 co = *reinterpret_cast<const float4*>(cs + t0);
                const float4 si = *reinterpret_cast<const float4*>(sn + t0);
                const float sign = grp_low ? -1.0f : 1.0f;
                const float qpp[4] = {pa.x * sx * wpa.x, pa.y * sx * wpa.y, pa.z * sx * wpa.z, pa.w * sx * wpa.w};
                const float kpp[4] = {pb.x * sx * wpb.x, pb.y * sx * wpb.y, pb.z * sx * wpb.z, pb.w * sx * wpb.w};
                const float npav[4] = {f(unpack(&npa.x, 0)), f(unpack(&npa.x, 1)), f(unpack(&npa.y, 0)), f(unpack(&npa.y, 1))};
                const float npbv[4] = {f(unpack(&npb.x, 0)), f(unpack(&npb.x, 1)), f(unpack(&npb.y, 0)), f(unpack(&npb.y, 1))};
                const float npa2v[4] = {f(unpack(&npa2.x, 0)), f(unpack(&npa2.x, 1)), f(unpack(&npa2.y, 0)), f(unpack(&npa2.y, 1))};
                const float npb2v[4] = {f(unpack(&npb2.x, 0)), f(unpack(&npb2.x, 1)), f(unpack(&npb2.y, 0)), f(unpack(&npb2.y, 1))};
                const float cov[4] = {co.x, co.y, co.z, co.w}, siv[4] = {si.x, si.y, si.z, si.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float nq_ = e < 2 ? npav[e] : npa2v[e - 2], nk_ = e < 2 ? npbv[e] : npb2v[e - 2];
                    const float qp_ = bf(bf(qpp[e]) * qi * nq_), kp_ = bf(bf(kpp[e]) * ki * nk_);
                    a[e] = bf(a[e] * bf(cov[e])) + bf(sign * qp_ * bf(siv[e]));
                    b[e] = bf(b[e] * bf(cov[e])) + bf(sign * kp_ * bf(siv[e]));
                }
            }
            *reinterpret_cast<uint2*>(TQ + at + 4 * i) = make_uint2(pack2(a[0], a[1]), pack2(a[2], a[3]));
            *reinterpret_cast<uint2*>(TK + at + 4 * i) = make_uint2(pack2(b[0], b[1]), pack2(b[2], b[3]));
            *reinterpret_cast<uint2*>(TV + at + 4 * i) = make_uint2(pack2(vc[0], vc[1]), pack2(vc[2], vc[3]));
        }
    }
}

// Each tile's mean q, k and v over its real rows, in fp32. QP, KP, VP: (H, tiles, 128). Grid [tiles, H], 32 threads.
__global__ void h3_pool(const __nv_bfloat16* TQ, const __nv_bfloat16* TK, const __nv_bfloat16* TV, const int* SIZES,
                        float* QP, float* KP, float* VP, int slots, int tiles) {
    const int tile = blockIdx.x, head = blockIdx.y, size = SIZES[tile], c0 = 4 * threadIdx.x;
    const size_t first = static_cast<size_t>(head) * slots + tile * 64, ht = static_cast<size_t>(head) * tiles + tile;
    float q[4] = {0, 0, 0, 0}, k[4] = {0, 0, 0, 0}, v[4] = {0, 0, 0, 0};
    for (int s = 0; s < size; ++s) {
        const size_t at = (first + s) * 128 + c0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] += f(TQ[at + j]);
            k[j] += f(TK[at + j]);
            v[j] += f(TV[at + j]);
        }
    }
    const float n = float(size);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        QP[ht * 128 + c0 + j] = q[j] / n;
        KP[ht * 128 + c0 + j] = k[j] / n;
        VP[ht * 128 + c0 + j] = v[j] / n;
    }
}

// Tile against tile: S[h, tq, tk] = QP[h, tq] . KP[h, tk] / sqrt(128), a warp a pair. Grid [tiles / 8, query tiles, H].
__global__ void __launch_bounds__(256) h3_tile_scores(const float* QP, const float* KP, float* S, int tiles) {
    const int tk = blockIdx.x * 8 + (threadIdx.x >> 5), tq = blockIdx.y, head = blockIdx.z, c0 = 4 * (threadIdx.x & 31);
    if (tk >= tiles) return;
    const float* q = QP + (static_cast<size_t>(head) * tiles + tq) * 128 + c0;
    const float* k = KP + (static_cast<size_t>(head) * tiles + tk) * 128 + c0;
    const float sum = warp_sum(q[0] * k[0] + q[1] * k[1] + q[2] * k[2] + q[3] * k[3]);
    if ((threadIdx.x & 31) == 0) S[(static_cast<size_t>(head) * tiles + tq) * tiles + tk] = sum * 0.08838834764831845f;
}

// A video tile's keys: the prefix, then its KEEP best tiles (NaN as -inf, ties to the lower tile) in tile order.
__global__ void h3_topk(const float* S, int* IDX, int tiles, int P0, int KEEP) {
    const int NV = tiles - P0, tq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float* s = S + (static_cast<size_t>(head) * tiles + P0 + tq) * tiles + P0;
    int* out = IDX + (static_cast<size_t>(head) * NV + tq) * (P0 + KEEP);
    for (int i = lane; i < P0; i += 32) out[i] = i;
    auto key = [&](int j) {
        const float v = s[j];
        const uint32_t b = __float_as_uint(isnan(v) ? -INFINITY : v);
        return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    };
    uint32_t thr = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = thr | (1u << bit);
        uint32_t mine = 0;
        for (int j = lane; j < NV; j += 32) mine += key(j) >= cand;
        if (__reduce_add_sync(0xffffffffu, mine) >= uint32_t(KEEP)) thr = cand;
    }
    // a kept tile's place is the number kept before it: lanes count in order, each after the lanes before it
    uint32_t above = 0;
    for (int j = lane; j < NV; j += 32) above += key(j) > thr;
    const int ties = KEEP - int(__reduce_add_sync(0xffffffffu, above));
    int kept = 0, tied = 0;
    for (int j0 = 0; j0 < NV; j0 += 32) {
        const int j = j0 + lane;
        const uint32_t k = j < NV ? key(j) : 0u;
        const uint32_t eq = __ballot_sync(0xffffffffu, j < NV && k == thr);
        const uint32_t gt = __ballot_sync(0xffffffffu, j < NV && k > thr);
        // of this round's equal scores, the first `ties - tied` are kept
        const int my_tie = tied + __popc(eq & ((1u << lane) - 1));
        const uint32_t take = __ballot_sync(0xffffffffu, (gt >> lane & 1) || ((eq >> lane & 1) && my_tie < ties));
        if (take >> lane & 1) out[P0 + kept + __popc(take & ((1u << lane) - 1))] = P0 + j;
        kept += __popc(take);
        tied += __popc(eq);
    }
}

// The pooled branch: C[h, tq] = softmax over every tile of S[h, tq] times VP[h], in fp32. Grid [query tiles, H], 32.
__global__ void h3_coarse(const float* S, const float* VP, float* C, int tiles) {
    const int tq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x, c0 = 4 * lane;
    const float* s = S + (static_cast<size_t>(head) * tiles + tq) * tiles;
    float top = -3.0e38f;
    for (int i = lane; i < tiles; i += 32) top = fmaxf(top, s[i]);
    top = warp_max(top);
    float mass = 0.0f;
    for (int i = lane; i < tiles; i += 32) mass += expf(s[i] - top);
    const float inv = 1.0f / warp_sum(mass);
    float acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < tiles; ++i) {
        const float w = expf(s[i] - top);
        const float* v = VP + (static_cast<size_t>(head) * tiles + i) * 128 + c0;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] += w * v[j];
    }
    float* o = C + (static_cast<size_t>(head) * tiles + tq) * 128 + c0;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = acc[j] * inv;
}

// The pooled branch gated into the attention rows in place: Y[r, c] += bf16(C[c / 128, tile of r] G[r, c]). Grid [M].
__global__ void __launch_bounds__(256) h3_gate_mix(__nv_bfloat16* Y, const __nv_bfloat16* G, const float* C,
                                                   const int* SLOT, int W, int stride, int tiles) {
    const int row = blockIdx.x, tile = SLOT[row] >> 6;
    __nv_bfloat16* y = Y + static_cast<size_t>(row) * W;
    const __nv_bfloat16* g = G + static_cast<size_t>(row) * stride;
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        const float cc = C[(static_cast<size_t>(c >> 7) * tiles + tile) * 128 + (c & 127)];
        y[c] = __float2bfloat16(f(y[c]) + bf(cc * f(g[c])));
    }
}

// silu(gate) * value as torch rounds it in bf16, from the first projection's rows [gate | value]. Grid-stride.
__global__ void __launch_bounds__(256) h3_swiglu(const __nv_bfloat16* F, __nv_bfloat16* A, int M, int N) {
    const size_t total = static_cast<size_t>(M) * N;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t r = i / N, n = i % N;
        const float g = f(F[r * 2 * N + n]), v = f(F[r * 2 * N + N + n]);
        A[i] = __float2bfloat16(bf(g / (1.0f + expf(-g))) * v);
    }
}

// A warp's most chunks of a quantized row: K up to 8 * 14 * 128 = 14,336 (fc1's half-row, the widest input).
constexpr int quantize_chunks = 14;

// h3_swiglu over an int8 product's int32 sums [gate | value]: each sum to bf16 as h3_dequantize rounds it, then the
// same silu and product, so A matches the two kernels' output without the bf16 rows between them.

// Grid [M], a row a block of 256 threads reading 4-wide int4/float4 groups — the grid-stride loop was the cost.
__global__ void __launch_bounds__(256) h3_swiglu_int8(const int* ACC, const float* SX, const float* SW, __nv_bfloat16* A, int M, int N) {
    const int r = blockIdx.x, groups = N >> 2;
    const int* acc = ACC + static_cast<size_t>(r) * 2 * N;
    const float s = SX[r];
#pragma unroll
    for (int i = 0; i < quantize_chunks; ++i) {
        const int g = threadIdx.x + 256 * i;
        if (g < groups) {
            const int4 ag = reinterpret_cast<const int4*>(acc + 4 * g)[0];
            const int4 av = reinterpret_cast<const int4*>(acc + N + 4 * g)[0];
            const float4 wg = *reinterpret_cast<const float4*>(SW + 4 * g);
            const float4 wv = *reinterpret_cast<const float4*>(SW + N + 4 * g);
            const float gg[4] = {bf(ag.x * s * wg.x), bf(ag.y * s * wg.y), bf(ag.z * s * wg.z), bf(ag.w * s * wg.w)};
            const float vv[4] = {bf(av.x * s * wv.x), bf(av.y * s * wv.y), bf(av.z * s * wv.z), bf(av.w * s * wv.w)};
            uint2 out;
            out.x = pack2(bf(gg[0] / (1.0f + expf(-gg[0]))) * vv[0], bf(gg[1] / (1.0f + expf(-gg[1]))) * vv[1]);
            out.y = pack2(bf(gg[2] / (1.0f + expf(-gg[2]))) * vv[2], bf(gg[3] / (1.0f + expf(-gg[3]))) * vv[3]);
            *reinterpret_cast<uint2*>(A + static_cast<size_t>(r) * N + 4 * g) = out;
        }
    }
}

// A warp's 128 values, four a lane in order, through the orthonormal Walsh-Hadamard transform in place.
__device__ __forceinline__ void hadamard128(float v[4]) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const bool high = lane & o;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float p = __shfl_xor_sync(0xffffffffu, v[i], o);
            v[i] = high ? p - v[i] : v[i] + p;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764831845f; // 1 / sqrt(128)
}

// Lane's four values of a row's 128-wide chunk `c`, rotated.
__device__ __forceinline__ void rotated_chunk(const __nv_bfloat16* x, int c, float v[4]) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(x + c * 128 + (threadIdx.x & 31) * 4);
    const float2 lo = __bfloat1622float2(p[0]), hi = __bfloat1622float2(p[1]);
    v[0] = lo.x;
    v[1] = lo.y;
    v[2] = hi.x;
    v[3] = hi.y;
    hadamard128(v);
}

// Each bf16 row of X (M, K) Hadamard-rotated in 128-wide chunks, then to int8 Q at the row's absmax / 127 in S.
// A row and its weights' rows rotate alike, so their products' sums are the unrotated ones. The rotated chunks stay in
// registers between the absmax and the rounding, so X is read once. Grid [M], 256 threads.
__global__ void __launch_bounds__(256) h3_quantize(const __nv_bfloat16* X, int8_t* Q, float* S, int K) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    const __nv_bfloat16* x = X + row * K;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, chunks = K >> 7;
    float most = 0.f, v[quantize_chunks][4];
#pragma unroll
    for (int i = 0; i < quantize_chunks; ++i) {
        const int c = warp + 8 * i;
        if (c < chunks) {
            rotated_chunk(x, c, v[i]);
            for (int j = 0; j < 4; ++j) most = fmaxf(most, fabsf(v[i][j]));
        }
    }
    most = tf_common::block_max_tree(most, scratch, 8);
    const float scale = most > 0.f ? most / 127.f : 1.f;
    if (threadIdx.x == 0) S[row] = scale;
#pragma unroll
    for (int i = 0; i < quantize_chunks; ++i) {
        const int c = warp + 8 * i;
        if (c < chunks) {
            const char4 q = make_char4(__float2int_rn(v[i][0] / scale), __float2int_rn(v[i][1] / scale), __float2int_rn(v[i][2] / scale), __float2int_rn(v[i][3] / scale));
            *reinterpret_cast<char4*>(Q + row * K + c * 128 + lane * 4) = q;
        }
    }
}

// h3_norm and h3_quantize as one launch for the int8 path: the gated add and its squares run exactly as h3_norm
// runs them (the same thread/stride and block_sum_serial order); h3_quantize's rotation, absmax and rounding run
// unchanged on the normed values, which each lane recomputes as modulated() rounded to bf16 — the same values
// h3_quantize reads from h3_norm's output row. The normed row is never materialized: the int8 product is its
// only reader, so the unfused pair's bf16 round trip through a row buffer drops out. Grid [M], 256 threads.
__global__ void __launch_bounds__(256) h3_norm_quant(
    __nv_bfloat16* X, const __nv_bfloat16* Y, const __nv_bfloat16* NW,
    const float* GT, const float* TAB, const int* LINE,
    int8_t* Q, float* S, int C, int gate, int scale, int shift, float eps) {
    __shared__ float scratch[32];
    const int row = blockIdx.x, line = LINE[row];
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    float sq = 0.0f;
    if (gate >= 0) {
        const float* g = GT + (static_cast<size_t>(line) * 6 + gate) * C;
        const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float v = bf(f(x[c]) + f(y[c]) * bf(g[c]));
            x[c] = __float2bfloat16(v);
            sq += v * v;
        }
    } else {
        for (int c = threadIdx.x; c < C; c += blockDim.x) sq += f(x[c]) * f(x[c]);
    }
    const float inv = rsqrtf(block_sum_serial(sq, scratch) / float(C) + eps);
    const float* s = TAB + (static_cast<size_t>(line) * 6 + scale) * C;
    const float* h = TAB + (static_cast<size_t>(line) * 6 + shift) * C;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, chunks = C >> 7;
    float most = 0.f, v[quantize_chunks][4];
#pragma unroll
    for (int i = 0; i < quantize_chunks; ++i) {
        const int c = warp + 8 * i;
        if (c < chunks) {
            const int base = c * 128 + lane * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                v[i][j] = f(__float2bfloat16(modulated(f(x[base + j]), inv, f(NW[base + j]), s[base + j], h[base + j])));
            hadamard128(v[i]);
            for (int j = 0; j < 4; ++j) most = fmaxf(most, fabsf(v[i][j]));
        }
    }
    most = tf_common::block_max_tree(most, scratch, 8);
    const float row_scale = most > 0.f ? most / 127.f : 1.f;
    if (threadIdx.x == 0) S[row] = row_scale;
#pragma unroll
    for (int i = 0; i < quantize_chunks; ++i) {
        const int c = warp + 8 * i;
        if (c < chunks) {
            const char4 q = make_char4(__float2int_rn(v[i][0] / row_scale), __float2int_rn(v[i][1] / row_scale), __float2int_rn(v[i][2] / row_scale), __float2int_rn(v[i][3] / row_scale));
            *reinterpret_cast<char4*>(Q + row * C + c * 128 + lane * 4) = q;
        }
    }
}

// Y (M, N) = bf16(ACC SX[row] SW[col]): an int8 product's int32 sums back to bf16 rows. Grid-stride, four a thread.
__global__ void __launch_bounds__(256) h3_dequantize(const int* ACC, const float* SX, const float* SW, __nv_bfloat16* Y, int M, int N) {
    const size_t quads = static_cast<size_t>(M) * N / 4;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t at = i * 4, r = at / N, n = at % N;
        const int4 a = reinterpret_cast<const int4*>(ACC)[i];
        const float4 w = *reinterpret_cast<const float4*>(SW + n);
        const float s = SX[r];
        __nv_bfloat162* y = reinterpret_cast<__nv_bfloat162*>(Y + at);
        y[0] = __floats2bfloat162_rn(a.x * s * w.x, a.y * s * w.y);
        y[1] = __floats2bfloat162_rn(a.z * s * w.z, a.w * s * w.w);
    }
}

__global__ void __launch_bounds__(128) h3_attention(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, const int* LIST, const int* SIZES,
        const int* ROWOF, __nv_bfloat16* Y, int slots, int heads, int queries, int keys, int first_query, int per_query,
        float scale) {
    attention_body(Q, K, V, LIST, SIZES, ROWOF, Y, slots, heads, queries, keys, first_query, per_query, scale);
}

// The stream's rows FIRST.. from (rows, width) f32 rows R: bf16(sum over k of R[r, k] W[n, k] + B[n]), summed one k at
// a time as rows_reference.rowsIn sums them. Grid [rows], 256 threads; width at most 128.
__global__ void __launch_bounds__(256) h3_rows_in(const float* R, const float* W, const float* B, __nv_bfloat16* X,
                                                  int width, int C, int first) {
    __shared__ float in[128];
    const int r = blockIdx.x;
    for (int k = threadIdx.x; k < width; k += blockDim.x) in[k] = R[static_cast<size_t>(r) * width + k];
    __syncthreads();
    for (int n = threadIdx.x; n < C; n += blockDim.x) {
        const float* w = W + static_cast<size_t>(n) * width;
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) sum = __fadd_rn(sum, __fmul_rn(in[k], w[k]));
        X[static_cast<size_t>(first + r) * C + n] = __float2bfloat16(__fadd_rn(sum, B[n]));
    }
}

// The stream's row FIRST + r through the final norm, modulated by its line of the (times, 2 C) table TAB, then the
// step's head into V (rows, width): rows_reference.finalNorm and rowsOut in their own sum orders (the squares one
// at a time, the head's 32 lane sums then added in lane order). With ACC null V = row · W + B, the on-grid
// step's plain head; else V = ACC + SCALE · (row · W + B), the off-grid steps' blend of the distilled
// heads their span covers (mmh3's step_heads: the run chains one launch a spanned head). Grid [rows],
// 256 threads, C floats of shared.
__global__ void __launch_bounds__(256) h3_rows_out(const __nv_bfloat16* X, const __nv_bfloat16* NW, const __nv_bfloat16* TAB,
                                                   const int* LINE, const float* W, const float* B, const float* ACC,
                                                   float* V, int C, int width, int first, float eps, float scale) {
    extern __shared__ float normed[];
    __shared__ float inv;
    const int r = blockIdx.x, row = first + r, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) normed[c] = f(x[c]);
    __syncthreads();
    if (threadIdx.x == 0) {
        float sq = 0.0f;
        for (int c = 0; c < C; ++c) sq = __fadd_rn(sq, __fmul_rn(normed[c], normed[c]));
        inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sq, static_cast<float>(C)), eps)));
    }
    __syncthreads();
    const __nv_bfloat16* tab = TAB + static_cast<size_t>(LINE[row]) * 2 * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) normed[c] = modulated(normed[c], inv, f(NW[c]), f(tab[C + c]), f(tab[c]));
    __syncthreads();
    for (int n = warp; n < width; n += blockDim.x >> 5) {
        const float* w = W + static_cast<size_t>(n) * C;
        float part = 0.0f;
        for (int k = lane; k < C; k += 32) part = __fadd_rn(part, __fmul_rn(normed[k], w[k]));
        float sum = 0.0f;
        for (int l = 0; l < 32; ++l) sum = __fadd_rn(sum, __shfl_sync(0xffffffffu, part, l));
        if (lane == 0) {
            const float head = __fadd_rn(sum, B[n]);
            V[static_cast<size_t>(r) * width + n] = ACC == nullptr ? head : __fadd_rn(ACC[static_cast<size_t>(r) * width + n], __fmul_rn(scale, head));
        }
    }
}

// X += delta V, the Euler step, in picture.euler's two roundings. Grid-stride.
__global__ void __launch_bounds__(256) h3_euler(float* X, const float* V, float delta, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) X[i] = __fadd_rn(X[i], __fmul_rn(delta, V[i]));
}

} // extern "C"
