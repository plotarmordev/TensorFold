// Dense H3 attention: int8 QK and unsigned-P/signed-V m16n8k32 products, fp32 online softmax.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
// tile_rows / head_dim and the cp.async helpers (cp16, commit, wait, pack2) live with the bf16 body
#include "h3_attention.cuh"

namespace tf_h3 {

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_u8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Q, K and V each occupy one 8 KiB shared tile.
constexpr int i8_tile_bytes = tile_rows * head_dim;
constexpr int i8_attention_shared = 3 * i8_tile_bytes;
static_assert(i8_attention_shared == 24576, "int8 attention's shared bytes");

// Dense int8 tiles and scales with h3_attention's online softmax and grid [query tiles,H] of 128.
__device__ __forceinline__ void attention_body_i8(
        const int8_t* __restrict__ Q8, const int8_t* __restrict__ K8, const int8_t* __restrict__ V8,
        const float* __restrict__ QS, const float* __restrict__ KS, const float* __restrict__ VS,
        const int* __restrict__ LIST, const int* __restrict__ SIZES, const int* __restrict__ ROWOF,
        __nv_bfloat16* __restrict__ Y, int slots, int heads, int queries, int keys, int first_query, int per_query,
        float scale) {
    constexpr int D = head_dim, T = tile_rows, THREADS = 128;
    extern __shared__ __align__(128) unsigned char buf[];
    unsigned char* const qsm = buf;
    unsigned char* const ksm = buf + i8_tile_bytes;
    unsigned char* const vsm = buf + 2 * i8_tile_bytes;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, qt = first_query + blockIdx.x;
    const size_t hs = static_cast<size_t>(head) * slots;

    // a row is 128 bytes in eight 16-byte chunks; the same (row & 7) xor swizzle as h3_attention's
    auto at = [](int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); };
    // V's 64-byte rows use (dim >> 1) & 3 so an eight-row fragment never repeats a bank phase.
    auto vat = [](int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); };
    auto load_qk = [&](unsigned char* to, const int8_t* from, int tile) {
        const int8_t* src = from + (hs + static_cast<size_t>(tile) * T) * D;
        for (int c = tid; c < T * 8; c += THREADS) cp16(to + at(c >> 3, c & 7), src + c * 16);
    };
    const int* list = LIST + (per_query ? (static_cast<size_t>(head) * queries + blockIdx.x) * keys : 0);
    const float* qsc = QS + hs;
    const float* kst = KS + hs;
    const float* vst = VS + static_cast<size_t>(head) * (slots / T) * D;
    auto load_v = [&](int tile) {
        const int8_t* src = V8 + (hs + static_cast<size_t>(tile) * T) * D;
        for (int c = tid; c < D * 4; c += THREADS) cp16(vsm + vat(c >> 2, c & 3), src + c * 16);
    };
    // Each fragment register is an aligned u32 at its swizzled row/channel quad.
    auto aread = [&](const unsigned char* sm, int r, int e) {
        return *reinterpret_cast<const uint32_t*>(sm + at(r, e >> 4) + (e & 15));
    };

    load_qk(qsm, Q8, qt);
    load_qk(ksm, K8, list[0]);
    commit();

    const int r0 = warp * 16 + (lane >> 2), c0 = (lane & 3) * 2, p = (lane & 3) * 4;
    // A m16n8k32 u32s hold rows r0 / r0 + 8 and key quads p, p + 16 of each 32-key chunk
    const int sa = (lane & ~3) + 2 * (lane & 1), sb = sa + 1;
    const float sl = scale * 1.4426950408889634f;
    float out[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j][0] = out[j][1] = out[j][2] = out[j][3] = 0.0f;
    float top0 = -1e30f, top1 = -1e30f, mass0 = 0.0f, mass1 = 0.0f;
    uint32_t qa[4][4];

    for (int sel = 0; sel < keys; ++sel) {
        const int tile = list[sel];
        wait<0>();
        __syncthreads();
        load_v(tile);
        commit();
        if (sel == 0) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                qa[ks][0] = aread(qsm, r0, ks * 32 + p);
                qa[ks][1] = aread(qsm, r0 + 8, ks * 32 + p);
                qa[ks][2] = aread(qsm, r0, ks * 32 + p + 16);
                qa[ks][3] = aread(qsm, r0 + 8, ks * 32 + p + 16);
            }
        }
        const int size = SIZES[tile];
        const float* ksc = kst + static_cast<size_t>(tile) * T;
        int si[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) si[j][0] = si[j][1] = si[j][2] = si[j][3] = 0;
        // QK^T: b regs hold key n = j * 8 + lane >> 2's channel quads p (b0) and p + 16 (b1) of chunk ks
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int n = j * 8 + (lane >> 2);
                mma_s8(si[j], qa[ks], aread(ksm, n, ks * 32 + p), aread(ksm, n, ks * 32 + p + 16));
            }
        }
        __syncthreads();
        if (sel + 1 < keys) load_qk(ksm, K8, list[sel + 1]);
        commit();

        const float q0 = qsc[qt * T + r0] * sl, q1 = qsc[qt * T + r0 + 8] * sl;
        // row's scores sit in its four lanes, so the maximum needs two shuffles
        float m0 = -1e30f, m1 = -1e30f;
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + c0;
            s[j][0] = col < size ? float(si[j][0]) * q0 * ksc[col] : -1e30f;
            s[j][1] = col + 1 < size ? float(si[j][1]) * q0 * ksc[col + 1] : -1e30f;
            s[j][2] = col < size ? float(si[j][2]) * q1 * ksc[col] : -1e30f;
            s[j][3] = col + 1 < size ? float(si[j][3]) * q1 * ksc[col + 1] : -1e30f;
            m0 = fmaxf(m0, fmaxf(s[j][0], s[j][1]));
            m1 = fmaxf(m1, fmaxf(s[j][2], s[j][3]));
        }
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        const float n0 = fmaxf(top0, m0), n1 = fmaxf(top1, m1);
        const float f0 = exp2f(top0 - n0), f1 = exp2f(top1 - n1);
        // Keep exact fp32 mass while u8 codes round(p*255) feed the value product.
        float l0 = 0.0f, l1 = 0.0f;
        uint32_t wlo[8], whi[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float p0 = exp2f(s[j][0] - n0), p1 = exp2f(s[j][1] - n0);
            const float p2 = exp2f(s[j][2] - n1), p3 = exp2f(s[j][3] - n1);
            l0 += p0 + p1;
            l1 += p2 + p3;
            // two consecutive keys' codes byte-packed: the gather below reads them as u16 pairs
            wlo[j] = __float2uint_rn(p0 * 255.0f) | (__float2uint_rn(p1 * 255.0f) << 8);
            whi[j] = __float2uint_rn(p2 * 255.0f) | (__float2uint_rn(p3 * 255.0f) << 8);
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

        const float* vsc = vst + static_cast<size_t>(tile) * D;
        int acc[16][4];
#pragma unroll
        for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            // Gather each key quad's four codes from the two lanes holding its u16 pairs.
            const uint32_t pair[4] = {
                (wlo[kk * 4] & 0xffffu) | (wlo[kk * 4 + 1] << 16),
                (whi[kk * 4] & 0xffffu) | (whi[kk * 4 + 1] << 16),
                (wlo[kk * 4 + 2] & 0xffffu) | (wlo[kk * 4 + 3] << 16),
                (whi[kk * 4 + 2] & 0xffffu) | (whi[kk * 4 + 3] << 16),
            };
            const int sh16 = (lane & 2) << 3;
            const uint32_t pa[4] = {
                uint32_t(uint16_t(__shfl_sync(0xffffffffu, pair[0], sa) >> sh16)) | (uint32_t(uint16_t(__shfl_sync(0xffffffffu, pair[0], sb) >> sh16)) << 16),
                uint32_t(uint16_t(__shfl_sync(0xffffffffu, pair[1], sa) >> sh16)) | (uint32_t(uint16_t(__shfl_sync(0xffffffffu, pair[1], sb) >> sh16)) << 16),
                uint32_t(uint16_t(__shfl_sync(0xffffffffu, pair[2], sa) >> sh16)) | (uint32_t(uint16_t(__shfl_sync(0xffffffffu, pair[2], sb) >> sh16)) << 16),
                uint32_t(uint16_t(__shfl_sync(0xffffffffu, pair[3], sa) >> sh16)) | (uint32_t(uint16_t(__shfl_sync(0xffffffffu, pair[3], sb) >> sh16)) << 16),
            };
            // PV: b regs hold dim n = j * 8 + lane >> 2's key quads p (b0) and p + 16 (b1) of chunk kk
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int n = j * 8 + (lane >> 2);
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(vsm + vat(n, 2 * kk) + p);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(vsm + vat(n, 2 * kk + 1) + p);
                mma_u8(acc[j], pa, b0, b1);
            }
        }

#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float vf0 = vsc[j * 8 + c0] * (1.0f / 255.0f);
            const float vf1 = vsc[j * 8 + c0 + 1] * (1.0f / 255.0f);
            out[j][0] += float(acc[j][0]) * vf0;
            out[j][1] += float(acc[j][1]) * vf1;
            out[j][2] += float(acc[j][2]) * vf0;
            out[j][3] += float(acc[j][3]) * vf1;
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
