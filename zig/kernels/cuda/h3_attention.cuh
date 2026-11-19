// MiniMax H3's tile-routed attention in bf16 on mma.sync m16n8k16: fp32 scores, an fp32 online softmax, fp32 sums.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "common/qmm_frag.cuh"

namespace tf_h3 {

using tf_common::smem;
using tf_common::cp16;
using tf_common::cp16z;
__device__ __forceinline__ void commit() { tf_common::cp_commit(); }
template <int N>
__device__ __forceinline__ void wait() { tf_common::cp_wait<N>(); }
using qmm_frag::ldmatrix4;
using qmm_frag::ldmatrix4t;
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    qmm_frag::mma(d, a, b0, b1);
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }

constexpr int head_dim = 128, tile_rows = 64;
// Bytes of one head's tile in shared memory, and the attention's dynamic shared bytes: q, k and v (two blocks an SM).
constexpr int tile_bytes = tile_rows * head_dim * 2;
constexpr int attention_shared = 3 * tile_bytes;

// A row is 256 bytes in 16 chunks; eight rows read at one chunk fall in different banks.
__device__ __forceinline__ int at(int r, int c) { return r * 256 + ((c ^ (r & 7)) << 4); }

// The fused variant's prologue inputs: the block's fused QKV rows, the head norms and the (rows, rot) rotary tables.
struct HeadsPrologue {
    const __nv_bfloat16* __restrict__ qkv;
    const __nv_bfloat16* __restrict__ nq;
    const __nv_bfloat16* __restrict__ nk;
    const float* __restrict__ cosine;
    const float* __restrict__ sine;
    int width;
    int rot;
    float eps;
};

// A 64-slot tile into `to` in the fragment layout: either the heads pass's tile-order rows `from` at absolute
// `tile`, or field `f` (0 q, 1 k, 2 v) of the fused QKV rows, gathered by `rowof` with zero fill for its -1 slots.
template <bool fused>
__device__ __forceinline__ void load_tile(unsigned char* to, const __nv_bfloat16* from, const HeadsPrologue& pro,
        const int* rowof, int head, int heads, int hs, int tile, int f) {
    const int tid = threadIdx.x;
    if (fused) {
        const __nv_bfloat16* src = pro.qkv + (static_cast<size_t>(f) * heads + head) * head_dim;
        for (int c = tid; c < tile_rows * 16; c += 128) {
            const int row = rowof[c >> 4];
            cp16z(to + at(c >> 4, c & 15), src + static_cast<size_t>(row < 0 ? 0 : row) * pro.width + (c & 15) * 8, row >= 0);
        }
        return;
    }
    const __nv_bfloat16* src = from + (static_cast<size_t>(hs) + static_cast<size_t>(tile) * tile_rows) * head_dim;
    for (int c = tid; c < tile_rows * 16; c += 128) cp16(to + at(c >> 4, c & 15), src + c * 8);
}

// h3_heads' q/k pass over the tile in `sm` in its roundings: the 32-channel share of each lane quad (t & 3), the
// row's sum of squares by the quad's two shuffles, then every channel's bf16 norm, split-half rotary and bf16
// store. The rotary's partner channel reads shared memory before any lane writes, so a __syncwarp between the
// reads and the stores is the whole inter-lane discipline; the caller's __syncthreads guards the mma reads.
__device__ __forceinline__ void heads_tile(unsigned char* sm, const HeadsPrologue& pro, const __nv_bfloat16* nw,
        const int* rowof) {
    const int tid = threadIdx.x, lane = tid & 31, c0 = (lane & 3) * 32;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int s = (tid >> 2) + half * 32, row = rowof[s], at_row = row < 0 ? 0 : row;
        const float* cs = pro.cosine + static_cast<size_t>(at_row) * pro.rot;
        const float* sn = pro.sine + static_cast<size_t>(at_row) * pro.rot;
        float x[32], px[32], sq = 0.0f;
        // this lane's 32 channels and its rotary partners, all read before the quad's first store
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 v = *reinterpret_cast<const uint4*>(sm + at(s, (c0 >> 3) + i));
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 w = __bfloat1622float2(p[j]);
                x[i * 8 + j * 2] = w.x;
                x[i * 8 + j * 2 + 1] = w.y;
            }
        }
#pragma unroll
        for (int j = 0; j < 32; ++j) sq += x[j] * x[j];
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int c = c0 + j;
            px[j] = 0.0f;
            if (c < 2 * pro.rot) {
                const int p = c < pro.rot ? c + pro.rot : c - pro.rot;
                px[j] = f(*reinterpret_cast<const __nv_bfloat16*>(sm + at(s, p >> 3) + (p & 7) * 2));
            }
        }
        __syncwarp();
        const float inv = rsqrtf(sq / 128.0f + pro.eps);
#pragma unroll
        for (int j = 0; j < 32; j += 2) {
            float a[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = c0 + j + e;
                a[e] = bf(x[j + e] * inv * f(nw[c]));
                if (c < 2 * pro.rot) {
                    const bool low = c < pro.rot;
                    const int p = low ? c + pro.rot : c - pro.rot, t = low ? c : c - pro.rot;
                    const float sign = low ? -1.0f : 1.0f, co = bf(cs[t]), si = bf(sn[t]);
                    const float partner = bf(px[j + e] * inv * f(nw[p]));
                    a[e] = bf(a[e] * co) + bf(sign * partner * si);
                }
            }
            *reinterpret_cast<uint32_t*>(sm + at(s, (c0 + j) >> 3) + ((c0 + j) & 7) * 2) = pack2(a[0], a[1]);
        }
    }
}

// softmax(q k^T scale) v over a query tile's key tiles in bf16, fp32 scores and sums; grid [query tiles, H] of 128.
// `fused` gathers the fused QKV rows and folds h3_heads' norm and rotary into the arriving tiles; else Q, K and V
// are h3_heads' tile-order buffers. The mma pipeline and the softmax are the same either way.
template <bool fused>
__device__ __forceinline__ void attention_body(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
        const HeadsPrologue& pro, const int* __restrict__ LIST, const int* __restrict__ SIZES,
        const int* __restrict__ ROWOF, __nv_bfloat16* __restrict__ Y, int slots, int heads, int queries, int keys,
        int first_query, int per_query, float scale) {
    constexpr int D = head_dim, T = tile_rows, THREADS = 128;
    extern __shared__ __align__(128) unsigned char buf[];
    unsigned char* const qsm = buf;
    unsigned char* const ksm = buf + tile_bytes;
    unsigned char* const vsm = buf + 2 * tile_bytes;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, qt = first_query + blockIdx.x;
    const int hs = head * slots;
    const int* list = LIST + (per_query ? (static_cast<size_t>(head) * queries + blockIdx.x) * keys : 0);

    auto load = [&](unsigned char* to, const __nv_bfloat16* from, int tile, int f) {
        load_tile<fused>(to, from, pro, ROWOF + tile * T, head, heads, hs, tile, f);
    };

    // a key tile's k is fetched while the tile before it is weighed against its values, and its v while its scores form
    load(qsm, Q, qt, 0);
    load(ksm, K, list[0], 1);
    commit();

    const int row_a = (lane & 7) + ((lane >> 3) & 1) * 8, chunk_a = lane >> 4;
    const int row_b = (lane & 7) + ((lane >> 4) & 1) * 8, chunk_b = (lane >> 3) & 1;
    const int r0 = warp * 16 + (lane >> 2), c0 = (lane & 3) * 2;
    const float sl = scale * 1.4426950408889634f;
    float out[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j][0] = out[j][1] = out[j][2] = out[j][3] = 0.0f;
    float top0 = -1e30f, top1 = -1e30f, mass0 = 0.0f, mass1 = 0.0f;
    uint32_t qa[8][4];

    for (int sel = 0; sel < keys; ++sel) {
        const int tile = list[sel];
        wait<0>();
        __syncthreads();
        load(vsm, V, tile, 2);
        commit();
        if (fused) {
            if (sel == 0) heads_tile(qsm, pro, pro.nq, ROWOF + qt * T);
            heads_tile(ksm, pro, pro.nk, ROWOF + tile * T);
            __syncthreads();
        }
        if (sel == 0) {
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) ldmatrix4(qa[ks], qsm + at(warp * 16 + row_a, ks * 2 + chunk_a));
        }
        const unsigned char* pk = ksm;
        const unsigned char* pv = vsm;
        const int size = SIZES[tile];

        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b[4];
                ldmatrix4(b, pk + at(j * 16 + row_b, ks * 2 + chunk_b));
                mma_bf16(s[2 * j], qa[ks], b[0], b[1]);
                mma_bf16(s[2 * j + 1], qa[ks], b[2], b[3]);
            }
        }
        __syncthreads();
        if (sel + 1 < keys) load(ksm, K, list[sel + 1], 1);
        commit();
        // each row's 64 scores sit in the four lanes of one group: its maximum needs two shuffles
        float m0 = -1e30f, m1 = -1e30f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + c0;
            s[j][0] = col < size ? s[j][0] * sl : -1e30f;
            s[j][1] = col + 1 < size ? s[j][1] * sl : -1e30f;
            s[j][2] = col < size ? s[j][2] * sl : -1e30f;
            s[j][3] = col + 1 < size ? s[j][3] * sl : -1e30f;
            m0 = fmaxf(m0, fmaxf(s[j][0], s[j][1]));
            m1 = fmaxf(m1, fmaxf(s[j][2], s[j][3]));
        }
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        const float n0 = fmaxf(top0, m0), n1 = fmaxf(top1, m1);
        const float f0 = exp2f(top0 - n0), f1 = exp2f(top1 - n1);
        // the weights in fp32 for the softmax's mass, rounded to bf16 as the value product's A operand
        float l0 = 0.0f, l1 = 0.0f;
        uint32_t pa[4][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float p0 = exp2f(s[j][0] - n0), p1 = exp2f(s[j][1] - n0);
            const float p2 = exp2f(s[j][2] - n1), p3 = exp2f(s[j][3] - n1);
            l0 += p0 + p1;
            l1 += p2 + p3;
            pa[j >> 1][(j & 1) * 2] = pack2(p0, p1);
            pa[j >> 1][(j & 1) * 2 + 1] = pack2(p2, p3);
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        mass0 = mass0 * f0 + l0;
        mass1 = mass1 * f1 + l1;
        top0 = n0;
        top1 = n1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            out[j][0] *= f0;
            out[j][1] *= f0;
            out[j][2] *= f1;
            out[j][3] *= f1;
        }
        wait<1>();
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int jd = 0; jd < 8; ++jd) {
                uint32_t b[4];
                ldmatrix4t(b, pv + at(kk * 16 + row_a, jd * 2 + chunk_a));
                mma_bf16(out[2 * jd], pa[kk], b[0], b[1]);
                mma_bf16(out[2 * jd + 1], pa[kk], b[2], b[3]);
            }
        }
    }
    const int to0 = ROWOF[qt * T + r0], to1 = ROWOF[qt * T + r0 + 8];
    const float i0 = 1.0f / mass0, i1 = 1.0f / mass1;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int col = head * D + j * 8 + c0;
        if (to0 >= 0) *reinterpret_cast<uint32_t*>(Y + static_cast<size_t>(to0) * heads * D + col) = pack2(out[j][0] * i0, out[j][1] * i0);
        if (to1 >= 0) *reinterpret_cast<uint32_t*>(Y + static_cast<size_t>(to1) * heads * D + col) = pack2(out[j][2] * i1, out[j][3] * i1);
    }
}

} // namespace tf_h3
