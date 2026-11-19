// MiniMax H3's tile-routed attention in bf16 on mma.sync m16n8k16: fp32 scores, an fp32 online softmax, fp32 sums.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "common/qmm_frag.cuh"

namespace tf_h3 {

using tf_common::smem;
using tf_common::cp16;
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

constexpr int head_dim = 128, tile_rows = 64;
// Bytes of one head's tile in shared memory; the attention stages a key tile and two value tiles in the same
// 48 KB the q, k and v tiles used, the queries held in registers instead of a shared tile.
constexpr int tile_bytes = tile_rows * head_dim * 2;
constexpr int attention_shared = 3 * tile_bytes;

// softmax(q k^T scale) v over a query tile's key tiles in bf16, fp32 scores and sums; grid [query tiles, H] of 128.
// The value tile is double buffered: the next tile's v is fetched while this tile's scores and softmax run, so the
// product no longer waits on a just-in-time load. The queries sit in registers, their smem tile now the second v.
__device__ __forceinline__ void attention_body(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
        const int* __restrict__ LIST, const int* __restrict__ SIZES, const int* __restrict__ ROWOF,
        __nv_bfloat16* __restrict__ Y, int slots, int heads, int queries, int keys, int first_query, int per_query,
        float scale) {
    constexpr int D = head_dim, T = tile_rows, THREADS = 128;
    extern __shared__ __align__(128) unsigned char buf[];
    unsigned char* const ksm = buf;
    unsigned char* const vsm[2] = {buf + tile_bytes, buf + 2 * tile_bytes};
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, qt = first_query + blockIdx.x;
    const size_t hs = static_cast<size_t>(head) * slots;
    const int* list = LIST + (per_query ? (static_cast<size_t>(head) * queries + blockIdx.x) * keys : 0);

    // A row is 256 bytes in 16 chunks; eight rows read at one chunk fall in different banks.
    auto at = [](int r, int c) { return r * 256 + ((c ^ (r & 7)) << 4); };
    auto load = [&](unsigned char* to, const __nv_bfloat16* from, int tile) {
        const __nv_bfloat16* src = from + (hs + static_cast<size_t>(tile) * T) * D;
        for (int c = tid; c < T * 16; c += THREADS) cp16(to + at(c >> 4, c & 15), src + c * 8);
    };

    // The query tile's fragments, from global once into registers. A lane's two rows and 16-byte columns are what
    // ldmatrix of the tile would hand it, so the score product's operands are unchanged.
    const int r0 = warp * 16 + (lane >> 2), qd = lane & 3;
    const int qr0 = qt * T + r0, qr1 = qr0 + 8;
    const bool have0 = ROWOF[qr0] >= 0, have1 = ROWOF[qr1] >= 0;
    const __nv_bfloat16* q0 = Q + (hs + static_cast<size_t>(have0 ? qr0 : qr1)) * D;
    const __nv_bfloat16* q1 = Q + (hs + static_cast<size_t>(have1 ? qr1 : qr0)) * D;
    uint32_t qa[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const uint32_t* p0 = reinterpret_cast<const uint32_t*>(q0 + ks * 16);
        const uint32_t* p1 = reinterpret_cast<const uint32_t*>(q1 + ks * 16);
        qa[ks][0] = have0 ? p0[qd] : 0u;
        qa[ks][1] = have1 ? p1[qd] : 0u;
        qa[ks][2] = have0 ? p0[qd + 4] : 0u;
        qa[ks][3] = have1 ? p1[qd + 4] : 0u;
    }

    // a key tile's k is fetched while the tile before it is weighed against its values, and its v a tile ahead
    load(ksm, K, list[0]);
    commit();
    load(vsm[0], V, list[0]);
    commit();
    if (keys > 1) {
        load(vsm[1], V, list[1]);
        commit();
    }

    const int row_b = (lane & 7) + ((lane >> 4) & 1) * 8, chunk_b = (lane >> 3) & 1;
    const int row_v = (lane & 7) + ((lane >> 3) & 1) * 8, chunk_v = lane >> 4;
    const int c0 = (lane & 3) * 2;
    const float sl = scale * 1.4426950408889634f;
    float out[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j][0] = out[j][1] = out[j][2] = out[j][3] = 0.0f;
    float top0 = -1e30f, top1 = -1e30f, mass0 = 0.0f, mass1 = 0.0f;

    for (int sel = 0; sel < keys; ++sel) {
        wait<0>();
        __syncthreads();
        unsigned char* const pk = ksm;
        unsigned char* const pv = vsm[sel & 1];
        const int size = SIZES[list[sel]];

        // the score product, each b fragment fetched while the one before it is still in the tensor pipe
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
        {
            uint32_t b[4], bn[4];
            ldmatrix4(b, pk + at(row_b, chunk_b));
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int ks = i / 4, j = i % 4;
                if (i + 1 < 32) {
                    const int nks = (i + 1) / 4, nj = (i + 1) % 4;
                    ldmatrix4(bn, pk + at(nj * 16 + row_b, nks * 2 + chunk_b));
                }
                mma_bf16(s[2 * j], qa[ks], b[0], b[1]);
                mma_bf16(s[2 * j + 1], qa[ks], b[2], b[3]);
                b[0] = bn[0]; b[1] = bn[1]; b[2] = bn[2]; b[3] = bn[3];
            }
        }
        __syncthreads();
        if (sel + 1 < keys) load(ksm, K, list[sel + 1]);
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
        // the value tile landed two tiles ago, so the product reads it with no wait past the top of the tile; each v
        // fragment is fetched while the one before it is still in the tensor pipe, then the slot takes the next v
        {
            uint32_t b[4], bn[4];
            ldmatrix4t(b, pv + at(row_v, chunk_v));
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int kk = i / 8, jd = i % 8;
                if (i + 1 < 32) {
                    const int nkk = (i + 1) / 8, njd = (i + 1) % 8;
                    ldmatrix4t(bn, pv + at(nkk * 16 + row_v, njd * 2 + chunk_v));
                }
                mma_bf16(out[2 * jd], pa[kk], b[0], b[1]);
                mma_bf16(out[2 * jd + 1], pa[kk], b[2], b[3]);
                b[0] = bn[0]; b[1] = bn[1]; b[2] = bn[2]; b[3] = bn[3];
            }
        }
        __syncthreads();
        if (sel + 2 < keys) load(pv, V, list[sel + 2]);
        commit();
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
