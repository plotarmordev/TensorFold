// MiniMax H3 / FastH3's transformer blocks on CUDA in the checkpoint's bf16: the row, routing and attention kernels.
#include "h3_attention.cuh"
#include "common/reduce.cuh"
#include "common/qmm_frag.cuh"

namespace h3_gemm {
using namespace qmm_frag;

// One m16n8k32 int8 mma step on the packed code words of a 32-value slice.
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// C (M, N) bf16 = X (M, K) . W (N, K)^T of rotated int8 rows. Activations carry one fp32 scale a 128-wide
// chunk in XS (M, K/128) — the per-chunk fold keeps a loud chunk from shrinking its neighbours; weights
// carry one fp32 scale a row in WS (N), applied once in the epilogue. The int32 sums stay in registers
// across K; the mma's own tile writes bf16, no int32 round trip through a second kernel.
// A stage is one K chunk deep: BM rows of X, BN rows of W and this chunk's BM activation scales.
// Grid tiles (M, N) in bands of `group` for L2 reuse.
template <int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void gemm(const int8_t* __restrict__ X, const float* __restrict__ XS,
        const int8_t* __restrict__ W, const float* __restrict__ WS, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, int group) {
    constexpr int GS = 128, THREADS = WM * WN * 32, MT = BM / WM / 16, NT = BN / WN / 8;
    constexpr int SX = BM * GS, SW = BN * GS;            // one 128-value chunk's codes a row
    constexpr int STAGE = SX + SW + BM * 4;              // codes, then this chunk's activation scales
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K >> 7;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    auto load = [&](int s, int g) {
        unsigned char* p = buf + s * STAGE;
        for (int c = tid; c < SX / 16; c += THREADS) {          // 8 chunks of 16 bytes a 128-byte row
            const int r = c >> 3, ch = c & 7, row = m0 + r;
            cp16z(p + r * GS + swz<8>(r, ch) * 16, X + (size_t)min(row, M - 1) * K + g * GS + ch * 16, row < M);
        }
        for (int c = tid; c < SW / 16; c += THREADS) {
            const int r = c >> 3, ch = c & 7, col = n0 + r;
            cp16z(p + SX + r * GS + swz<8>(r, ch) * 16, W + (size_t)min(col, N - 1) * K + g * GS + ch * 16, col < N);
        }
        float* ps = reinterpret_cast<float*>(p + SX + SW);
        for (int c = tid; c < BM / 4; c += THREADS) {           // 4 scales a 16-byte copy
            const int r = c * 4;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int idx = m0 + r + e;
                ps[r + e] = idx < M ? __ldg(XS + (size_t)idx * KG + g) : 0.f;
            }
        }
    };
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        qmm_frag::commit();
    }
    for (int it = 0; it < KG; ++it) {
        qmm_frag::wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        qmm_frag::commit();
        const unsigned char* p = buf + (it % STAGES) * STAGE;
        const float* ps = reinterpret_cast<const float*>(p + SX + SW);
        // this chunk's activation scales for the warp's rows, and the magic-bias correction once each
        float ra[MT][2], rc[MT][2];
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float s = ps[wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8];
                ra[i][h] = s;
                rc[i][h] = -12582912.0f * s;
            }
        // a chunk is four 32-deep mma steps. kt outer, jj inner: a's fragment is loaded once a step, and
        // each n pair's int32 sums fold into fp32 right after its two mma — iacc lives across one step of
        // one pair, so the fold runs four times a chunk at the same chunk scale. A 32-deep int32 sum
        // stays exact in the 12582912.0f bias.
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ldmatrix4(a[i], p + r * GS + swz<8>(r, kt * 2 + (lane >> 4)) * 16);
            }
#pragma unroll
            for (int jj = 0; jj < NT; jj += 2) {              // an x4 load feeds two n8 operands
                const int half = lane >> 4;                    // lanes 16+ take the j+1 weight rows
                const int c = wn * (BN / WN) + (jj + half) * 8 + (lane & 7);
                uint32_t b[4];
                ldmatrix4(b, p + SX + c * GS + swz<8>(c, kt * 2 + ((lane >> 3) & 1)) * 16);
                int iacc[MT][2][4];
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int p2 = 0; p2 < 4; ++p2) iacc[i][0][p2] = iacc[i][1][p2] = 0x4B400000;
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    mma_s8(iacc[i][0], a[i], b[0], b[1]);
                    mma_s8(iacc[i][1], a[i], b[2], b[3]);
                }
                // drop the bias and fold at the chunk's activation scale: acc += (iacc - magic) * xs
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float s = ra[i][h], cc = rc[i][h];
                        acc[i][jj][2 * h] = fmaf(fmaf(__int_as_float(iacc[i][0][2 * h]), s, cc), 1.0f, acc[i][jj][2 * h]);
                        acc[i][jj][2 * h + 1] = fmaf(fmaf(__int_as_float(iacc[i][0][2 * h + 1]), s, cc), 1.0f, acc[i][jj][2 * h + 1]);
                        acc[i][jj + 1][2 * h] = fmaf(fmaf(__int_as_float(iacc[i][1][2 * h]), s, cc), 1.0f, acc[i][jj + 1][2 * h]);
                        acc[i][jj + 1][2 * h + 1] = fmaf(fmaf(__int_as_float(iacc[i][1][2 * h + 1]), s, cc), 1.0f, acc[i][jj + 1][2 * h + 1]);
                    }
            }
        }
    }
    qmm_frag::wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            const float w0 = __ldg(WS + min(col, N - 1)), w1 = __ldg(WS + min(col + 1, N - 1));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M || col >= N) continue;
                __nv_bfloat16* dst = C + (size_t)row * N + col;
                dst[0] = __float2bfloat16(acc[i][j][2 * h] * w0);
                if (col + 1 < N) dst[1] = __float2bfloat16(acc[i][j][2 * h + 1] * w1);
            }
        }
}

} // namespace h3_gemm

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

// A warp's most chunks of a quantized row: K up to 8 * 14 * 128 = 14,336, the widest projection input.
constexpr int quantize_chunks = 14;

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

// Each bf16 row of X (M, K) Hadamard-rotated in 128-wide chunks, then to int8 Q at each chunk's own absmax
// / 127 in S (M, K/128): the activation-side counterpart of h3_quantize's per-row scale. A chunk sits whole
// in a warp, so its absmax is a warp reduce. Grid [M], 256 threads.
__global__ void __launch_bounds__(256) h3_quantize_chunks(const __nv_bfloat16* X, int8_t* Q, float* S, int K) {
    const size_t row = blockIdx.x;
    const __nv_bfloat16* x = X + row * K;
    const int chunks = K >> 7;
    for (int c = threadIdx.x >> 5; c < chunks; c += blockDim.x >> 5) {
        float v[4];
        rotated_chunk(x, c, v);
        float most = 0.f;
        for (int j = 0; j < 4; ++j) most = fmaxf(most, fabsf(v[j]));
        most = warp_max(most);
        const float scale = most > 0.f ? most / 127.f : 1.f;
        if ((threadIdx.x & 31) == 0) S[row * chunks + c] = scale;
        const char4 q = make_char4(__float2int_rn(v[0] / scale), __float2int_rn(v[1] / scale),
                                   __float2int_rn(v[2] / scale), __float2int_rn(v[3] / scale));
        *reinterpret_cast<char4*>(Q + row * K + c * 128 + (threadIdx.x & 31) * 4) = q;
    }
}

extern "C" __global__ void __launch_bounds__(384) h3_gemm_int8(const int8_t* __restrict__ X, const float* __restrict__ XS,
        const int8_t* __restrict__ W, const float* __restrict__ WS, __nv_bfloat16* __restrict__ C,
        int M, int N, int K, int group) {
    h3_gemm::gemm<192, 128, 6, 2, 2>(X, XS, W, WS, C, M, N, K, group); // 2 stages of a 192x128 tile, 12 warps an SM
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
// at a time, the head's 32 lane sums then added in lane order). Grid [rows], 256 threads, C floats of shared.
__global__ void __launch_bounds__(256) h3_rows_out(const __nv_bfloat16* X, const __nv_bfloat16* NW, const __nv_bfloat16* TAB,
                                                   const int* LINE, const float* W, const float* B, float* V, int C,
                                                   int width, int first, float eps) {
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
        if (lane == 0) V[static_cast<size_t>(r) * width + n] = __fadd_rn(sum, B[n]);
    }
}

// X += delta V, the Euler step, in picture.euler's two roundings. Grid-stride.
__global__ void __launch_bounds__(256) h3_euler(float* X, const float* V, float delta, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) X[i] = __fadd_rn(X[i], __fmul_rn(delta, V[i]));
}

} // extern "C"
