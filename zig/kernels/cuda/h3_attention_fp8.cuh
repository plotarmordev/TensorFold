// MiniMax H3's dense attention on mma.sync m16n8k32: h3_heads' q/k RMSNorm and rotary run in the tile's
// prologue from the raw QKV rows; q and k quantize to int8 at each row's absmax (per-tile-absmax e4m3 measured
// a 0.87 latent cosine — four significand bits is too coarse for Q·K) while the weighed scores and v stay e4m3
// like mmh3's split. The row scales land in shared qsc/ksc and fold into the score rescale. The fragments of
// m16n8k32.row.col (PTX ISA): a0..a3 hold rows lane>>2 and +8 at channel quads lane&3 and +16 elements; b0, b1
// hold column lane>>2 at channel quads lane&3 and +16 — every register one aligned u32 in the swizzled tile.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "mma4.cuh"

namespace tf_h3 {

__device__ __forceinline__ float fp8_bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float fp8_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ uint32_t fp8_pack4(float a, float b, float c, float d) {
    // element k sits in byte k of the fragment's u32
    return static_cast<uint32_t>(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3))
        | (static_cast<uint32_t>(__nv_cvt_float_to_fp8(b, __NV_SATFINITE, __NV_E4M3)) << 8)
        | (static_cast<uint32_t>(__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3)) << 16)
        | (static_cast<uint32_t>(__nv_cvt_float_to_fp8(d, __NV_SATFINITE, __NV_E4M3)) << 24);
}
__device__ __forceinline__ uint32_t i8_pack4(float a, float b, float c, float d) {
    return (__float2int_rn(a) & 0xff) | ((__float2int_rn(b) & 0xff) << 8) | ((__float2int_rn(c) & 0xff) << 16) | ((__float2int_rn(d) & 0xff) << 24);
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int fp8_tile_bytes = tile_rows * head_dim;      // 8 KB a tile — int8 for q/k, e4m3 for v
constexpr int fp8_stage_bytes = tile_rows * head_dim * 2; // a tile's raw bf16 staging
constexpr int fp8_score_bytes = tile_rows * tile_rows;    // the weighed scores' 4 KB (e4m3)
// q, k, v and the weighed scores' byte tiles plus the one bf16 stage the arriving tiles share: 44 KB, two CTAs an SM.
constexpr int fp8_attention_shared = 3 * fp8_tile_bytes + fp8_stage_bytes + fp8_score_bytes;
static_assert(fp8_attention_shared == 45056, "fp8 attention's shared bytes");


// The prologue's weights: the block's fused QKV rows, the head norms and the (rows, rot) rotary tables.
struct Fp8Prologue {
    const __nv_bfloat16* __restrict__ qkv;
    const __nv_bfloat16* __restrict__ nq;
    const __nv_bfloat16* __restrict__ nk;
    const float* __restrict__ cosine;
    const float* __restrict__ sine;
    const int* __restrict__ rowof;
    int width;
    int heads;
    int rot;
};

// softmax(q k^T scale) v over a query tile's key tiles in e4m3, fp32 scores and sums; grid [query tiles, H]
// of 128 threads. The prologue's norm and rotary reproduce h3_heads' arithmetic; dead slots read row 0's
// tables, write zero codes and stay masked out of the result.
__device__ __forceinline__ void attention_body_fp8(
        const Fp8Prologue& pro, const int* __restrict__ LIST, const int* __restrict__ SIZES,
        __nv_bfloat16* __restrict__ Y, int queries, int keys, int first_query, int per_query, float scale) {
    constexpr int D = head_dim, T = tile_rows, THREADS = 128;
    constexpr float EPS = 1e-5f, PACK = 448.0f;
    extern __shared__ __align__(128) unsigned char buf[];
    __shared__ float red[8];
    __shared__ float qsc[tile_rows], ksc[tile_rows]; // the rows' amax/127 dequant factors
    unsigned char* const qsm = buf;
    unsigned char* const ksm = buf + fp8_tile_bytes;
    unsigned char* const vsm = buf + 2 * fp8_tile_bytes; // dim-major: row d holds the 64 keys' dim d
    unsigned char* const stage = buf + 3 * fp8_tile_bytes;
    unsigned char* const psm = stage + fp8_stage_bytes; // the weighed scores, keys minor
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, qt = first_query + blockIdx.x;
    const int* list = LIST + (per_query ? (static_cast<size_t>(head) * queries + blockIdx.x) * keys : 0);
    // swizzled byte address of row r's 16-byte chunk c: eight chunks a row in an fp8 tile, sixteen in the
    // stage, four in the dim-major v and the score tiles' 64-byte rows
    auto at = [](int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); };
    auto at16 = [](int r, int c) { return r * 256 + ((c ^ (r & 7)) << 4); };
    auto vat = [](int r, int c) { return r * 64 + ((c ^ (r & 3)) << 4); };
    // the staged tile's bf16 element (row, channel)
    auto selem = [&](int s, int c) { return fp8_f(*reinterpret_cast<const __nv_bfloat16*>(stage + at16(s, c >> 3) + (c & 7) * 2)); };
    // the tile's rows from field f (0 q, 1 k, 2 v) of the fused QKV rows into the stage
    auto load_stage = [&](const int* rowof, int field) {
        const size_t off = static_cast<size_t>(field) * pro.heads + head;
        for (int c = tid; c < T * 16; c += THREADS) {
            const int row = rowof[c >> 4];
            cp16(stage + at16(c >> 4, c & 15), pro.qkv + (static_cast<size_t>(row < 0 ? 0 : row) * pro.width + off * D) + (c & 15) * 8);
        }
        commit();
    };
    // h3_heads' one channel normed and rotated, in its roundings; the partner channel p = c +/- rot reads
    // the staged row too.
    auto normed = [&](const __nv_bfloat16* nw, const float* cs, const float* sn, int s, int c, float inv) {
        float v = fp8_bf(selem(s, c) * inv * fp8_f(nw[c]));
        if (c < 2 * pro.rot) {
            const bool low = c < pro.rot;
            const int pp = low ? c + pro.rot : c - pro.rot, t = low ? c : c - pro.rot;
            const float sign = low ? -1.0f : 1.0f, co = fp8_bf(cs[t]), si = fp8_bf(sn[t]);
            const float partner = fp8_bf(selem(s, pp) * inv * fp8_f(nw[pp]));
            v = fp8_bf(v * co) + fp8_bf(sign * partner * si);
        }
        return v;
    };
    // a row's rsqrt from the lane's 32 channels, summed across the lane quad
    auto row_inv = [&](int s, int c0) {
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) sq += selem(s, c0 + j) * selem(s, c0 + j);
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        return rsqrtf(sq / 128.0f + EPS);
    };

    // The stage's tile to int8 in `to` at each row's absmax (the dequant factor amax/127 lands in `sc`), with the
    // q/k norm and rotary. h3_heads' split: thread t owns the 32-channel chunk (t&3)*32 of rows t>>2 and t>>2+32.
    auto quantize_qk = [&](unsigned char* to, int field, const int* rowof, float* sc) {
        const __nv_bfloat16* nw = field == 0 ? pro.nq : pro.nk;
        const int c0 = (tid & 3) * 32;
        for (int half = 0; half < 2; ++half) {
            const int s = (tid >> 2) + half * 32, row = rowof[s];
            // every lane joins row_inv's quad shuffles; a dead row's garbage never leaves its quad
            const float inv = row_inv(s, c0);
            const int at_row = row < 0 ? 0 : row;
            const float* cs = pro.cosine + static_cast<size_t>(at_row) * pro.rot;
            const float* sn = pro.sine + static_cast<size_t>(at_row) * pro.rot;
            float vals[32], amax = 0.0f;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                vals[j] = row < 0 ? 0.0f : normed(nw, cs, sn, s, c0 + j, inv);
                amax = fmaxf(amax, fabsf(vals[j]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            if ((tid & 3) == 0) sc[s] = amax * (1.0f / 127.0f); // dequant factor; 0 on a dead or all-zero row
            const float iscale = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const int e = c0 + j;
                *reinterpret_cast<uint32_t*>(to + at(s, e >> 4) + (e & 15)) = i8_pack4(vals[j] * iscale, vals[j + 1] * iscale, vals[j + 2] * iscale, vals[j + 3] * iscale);
            }
        }
    };
    // The stage's tile to e4m3 in `to` without a norm (v), dim-major for the PV fragment; the tile's absmax
    // `vm` also scales the P codes, so the value product's multiplier is the constant 1 / 448.
    auto quantize_v = [&](unsigned char* to, const int* rowof) {
        const int s = tid >> 1, c0 = (tid & 1) * 64;
        const int row = rowof[s];
        float most = 0.0f;
        if (row >= 0) {
#pragma unroll
            for (int j = 0; j < 64; ++j) most = fmaxf(most, fabsf(selem(s, c0 + j)));
        }
        most = tf_common::block_max_tree(most, red, 4);
        const float iscale = most > 0.0f ? PACK / most : 1.0f;
#pragma unroll
        for (int j = 0; j < 64; j += 4) {
            const int e = c0 + j;
            const float v0 = row < 0 ? 0.0f : selem(s, e + 0);
            const float v1 = row < 0 ? 0.0f : selem(s, e + 1);
            const float v2 = row < 0 ? 0.0f : selem(s, e + 2);
            const float v3 = row < 0 ? 0.0f : selem(s, e + 3);
            const uint32_t q = fp8_pack4(v0 * iscale, v1 * iscale, v2 * iscale, v3 * iscale);
            // dim-major at 64-byte rows: key s's byte sits in chunk s>>4 swizzled by the dim
            to[vat(e + 0, s >> 4) + (s & 15)] = q & 0xff;
            to[vat(e + 1, s >> 4) + (s & 15)] = (q >> 8) & 0xff;
            to[vat(e + 2, s >> 4) + (s & 15)] = (q >> 16) & 0xff;
            to[vat(e + 3, s >> 4) + (s & 15)] = q >> 24;
        }
        return most; // the P codes' multiplier
    };

    // the query tile's q rows stage, norm, rotate and quantize once
    load_stage(pro.rowof + qt * T, 0);
    wait<0>();
    __syncthreads();
    quantize_qk(qsm, 0, pro.rowof + qt * T, qsc);
    __syncthreads();

    // fragment reads: element e of row r sits at byte e&15 of swizzled chunk e>>4; a lane's quads are
    // p = (lane&3)*4 apart, so every fragment register is one aligned u32 at (r, chunk p>>4, byte p)
    auto aread = [&](const unsigned char* sm, int r, int e) {
        return *reinterpret_cast<const uint32_t*>(sm + at(r, e >> 4) + (e & 15));
    };
    const int ra = warp * 16 + (lane >> 2), p = (lane & 3) * 4;
    uint32_t qa[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        qa[ks][0] = aread(qsm, ra, ks * 32 + p);
        qa[ks][1] = aread(qsm, ra + 8, ks * 32 + p);
        qa[ks][2] = aread(qsm, ra, ks * 32 + p + 16);
        qa[ks][3] = aread(qsm, ra + 8, ks * 32 + p + 16);
    }

    const int r0 = warp * 16 + (lane >> 2), c0 = (lane & 3) * 2;
    const float sl = scale * 1.4426950408889634f;
    float out[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j][0] = out[j][1] = out[j][2] = out[j][3] = 0.0f;
    float top0 = -1e30f, top1 = -1e30f, mass0 = 0.0f, mass1 = 0.0f;

    // the first key tile's stage load runs beside the fragment setup; each later tile's load is issued during
    // the previous tile's score staging and value product
    load_stage(pro.rowof + list[0] * T, 1);
    for (int sel = 0; sel < keys; ++sel) {
        const int tile = list[sel];
        const int size = SIZES[tile];
        const int* krow = pro.rowof + tile * T;
        wait<0>();
        __syncthreads();
        quantize_qk(ksm, 1, krow, ksc);
        __syncthreads();
        // v's rows stage under the score product; the stage is free once k's reads ended above
        load_stage(krow, 2);
        int si[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) si[j][0] = si[j][1] = si[j][2] = si[j][3] = 0;
        // QK^T: b regs hold key n = j*8 + lane>>2's channel quads lane&3 (b0) and +16 (b1) of chunk ks
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int n = j * 8 + (lane >> 2);
                mma_s8(si[j], qa[ks], aread(ksm, n, ks * 32 + p), aread(ksm, n, ks * 32 + p + 16));
            }
        }
        __syncthreads();

        // softmax, h3_attention's order: the row's 64 scores sit in its four lanes; the int sums dequantize
        // by their q and k rows' amax/127 factors on the way in
        const float q0 = qsc[r0] * sl, q1 = qsc[r0 + 8] * sl;
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
        float mine[8][4], l0 = 0.0f, l1 = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            mine[j][0] = exp2f(s[j][0] - n0);
            mine[j][1] = exp2f(s[j][1] - n0);
            mine[j][2] = exp2f(s[j][2] - n1);
            mine[j][3] = exp2f(s[j][3] - n1);
            l0 += mine[j][0] + mine[j][1];
            l1 += mine[j][2] + mine[j][3];
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
        wait<0>();
        __syncthreads();
        const float vm = quantize_v(vsm, krow);
        __syncthreads();
        // the next key tile's stage load overlaps the score staging and the value product
        if (sel + 1 < keys) load_stage(pro.rowof + list[sel + 1] * T, 1);


        // P in e4m3, scaled by vm so the value product's multiplier stays the constant 1 / 448: a lane's
        // u16 of keys j*8+c0..+1 stays inside one 16-byte chunk (j*8 mod 16 is 0 or 8, c0 even).
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int e = j * 8 + c0;
            const uint32_t lo = fp8_pack4(mine[j][0] * vm, mine[j][1] * vm, 0.0f, 0.0f);
            const uint32_t hi = fp8_pack4(mine[j][2] * vm, mine[j][3] * vm, 0.0f, 0.0f);
            *reinterpret_cast<uint16_t*>(psm + vat(r0, e >> 4) + (e & 15)) = static_cast<uint16_t>(lo);
            *reinterpret_cast<uint16_t*>(psm + vat(r0 + 8, e >> 4) + (e & 15)) = static_cast<uint16_t>(hi);
        }
        __syncthreads();
        // P's a fragments read rows r0 and r0+8 at key quads lane&3 and +16 of each k32 chunk
        uint32_t pv[2][4];
        auto pread = [&](int r, int e) { return *reinterpret_cast<const uint32_t*>(psm + vat(r, e >> 4) + (e & 15)); };
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            pv[kk][0] = pread(r0, kk * 32 + p);
            pv[kk][1] = pread(r0 + 8, kk * 32 + p);
            pv[kk][2] = pread(r0, kk * 32 + p + 16);
            pv[kk][3] = pread(r0 + 8, kk * 32 + p + 16);
        }
        // PV: b regs hold dim n = j*8 + lane>>2's key quads 4p..4p+3 (b0) and +16..+19 (b1) of chunk kk
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int n = j * 8 + (lane >> 2);
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(vsm + vat(n, 2 * kk) + p);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(vsm + vat(n, 2 * kk + 1) + p);
                mma4::mma_fp8(out[j], pv[kk], b0, b1);
            }
        }
        __syncthreads();
    }
    const int to0 = pro.rowof[qt * T + r0], to1 = pro.rowof[qt * T + r0 + 8];
    // the PV mma's codes carry a factor 448 (p·vm times v·448/vm): it comes back out here
    const float i0 = 1.0f / (mass0 * PACK), i1 = 1.0f / (mass1 * PACK);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int col = head * D + j * 8 + c0;
        if (to0 >= 0) *reinterpret_cast<uint32_t*>(Y + static_cast<size_t>(to0) * pro.heads * D + col) = pack2(out[j][0] * i0, out[j][1] * i0);
        if (to1 >= 0) *reinterpret_cast<uint32_t*>(Y + static_cast<size_t>(to1) * pro.heads * D + col) = pack2(out[j][2] * i1, out[j][3] * i1);
    }
}

} // namespace tf_h3
