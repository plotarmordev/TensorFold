// Inkling 2 Mini's DSpark draft head (inkling2_dspark): the block's norms, attention and Markov chain, and greedy picks
// that stay on the device. The projections run on the target's bf16 GEMM (qmmf); these kernels join them. fp32 sums.
#include <cstdint>
#include <cuda_bf16.h>

namespace dsp {

constexpr int H = 2304, Q = 3072, KV = 1024, D = 128, R = 256, RNN = 5120;
constexpr int kRows = 8, kWindow = 1024;
constexpr float kEps = 1e-6f;
constexpr int kThreads = 256, kWarps = kThreads / 32;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// A 256-thread block's sum; every thread gets it.
__device__ __forceinline__ float blockSum(float v, float* sh) {
    v = warpSum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < kWarps; ++i) t += sh[i];
    return t;
}
// Rotate-half RoPE, theta 1e6: d's partner is d +- 64.
__device__ __forceinline__ float rope(float x, float other, int d, uint32_t pos) {
    const float a = static_cast<float>(pos) * powf(1e6f, -static_cast<float>(d % 64) / 64.0f);
    float s, c;
    sincosf(a, &s, &c);
    return d < 64 ? x * c - other * s : other * s + x * c;
}
// A 128-wide head row normed by w and RoPE'd at pos, one warp: each lane holds dims lane+32j.
__device__ __forceinline__ void headRope(const uint16_t* in, const uint16_t* w, uint16_t* out, uint32_t pos) {
    const int lane = threadIdx.x & 31;
    float x[4], ss = 0.0f;
    for (int j = 0; j < 4; ++j) {
        x[j] = bf(in[lane + 32 * j]);
        ss += x[j] * x[j];
    }
    const float scale = rsqrtf(warpSum(ss) / D + kEps);
    for (int j = 0; j < 4; ++j) x[j] *= scale * bf(w[lane + 32 * j]);
    for (int j = 0; j < 4; ++j) out[lane + 32 * j] = tobf(rope(x[j], x[(j + 2) % 4], lane + 32 * j, pos));
}

}  // namespace dsp

using namespace dsp;

// One row a block: the four taps mixed by fusion f's softmax weights, then normed by its dflare row. taps [rows][4][H];
// c [4][rows][H]; newest [H] takes fusion 3's last row.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_dsp_fuse(const uint16_t* taps, const uint16_t* fuse, const uint16_t* dflare, uint16_t* c, uint16_t* newest, int rows) {
    __shared__ float sh[kWarps];
    const int f = blockIdx.y, row = blockIdx.x;
    float alpha[4], peak = -INFINITY, sum = 0.0f;
    for (int j = 0; j < 4; ++j) peak = fmaxf(peak, alpha[j] = bf(fuse[f * 4 + j]));
    for (int j = 0; j < 4; ++j) sum += alpha[j] = expf(alpha[j] - peak);
    float m[9], ss = 0.0f;
    for (int n = 0; n < 9; ++n) {
        const int d = n * kThreads + threadIdx.x;
        m[n] = 0.0f;
        for (int j = 0; j < 4; ++j) m[n] += alpha[j] / sum * bf(taps[(static_cast<size_t>(row) * 4 + j) * H + d]);
        ss += m[n] * m[n];
    }
    const float scale = rsqrtf(blockSum(ss, sh) / H + kEps);
    for (int n = 0; n < 9; ++n) {
        const int d = n * kThreads + threadIdx.x;
        const uint16_t out = tobf(m[n] * scale * bf(dflare[f * H + d]));
        c[(static_cast<size_t>(f) * rows + row) * H + d] = out;
        if (f == 3 && row + 1 == rows) newest[d] = out;
    }
}

// Context K (normed, RoPE'd) and V of rows at positions base.. into a layer's ring; one warp a (row, kv head).
extern "C" __global__ void __launch_bounds__(32) tf_ink2_dsp_ctx(const uint16_t* k, const uint16_t* v, const uint16_t* knorm, uint16_t* ring_k, uint16_t* ring_v, uint32_t base, int ring) {
    const int row = blockIdx.x, h = blockIdx.y;
    const size_t at = (static_cast<size_t>((base + row) % ring)) * KV + h * D;
    headRope(k + static_cast<size_t>(row) * KV + h * D, knorm, ring_k + at, base + row);
    for (int j = 0; j < 4; ++j) ring_v[at + threadIdx.x + 32 * j] = v[static_cast<size_t>(row) * KV + h * D + threadIdx.x + 32 * j];
}

// Block row 0: the target's normed embedding of the pending token (read on the device); rows 1..7 the mask embedding.
// Writes the fp32 residual and its first input norm.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_dsp_init(const uint32_t* pending, const uint16_t* embed, const uint16_t* embed_norm, float embed_eps, const uint16_t* mask, const uint16_t* norm, float* resid, uint16_t* x) {
    __shared__ float sh[kWarps];
    const int row = blockIdx.x;
    const uint16_t* src = row == 0 ? embed + static_cast<size_t>(*pending) * H : mask;
    float e[9], ss = 0.0f;
    for (int n = 0; n < 9; ++n) {
        e[n] = bf(src[n * kThreads + threadIdx.x]);
        ss += e[n] * e[n];
    }
    if (row == 0) {
        const float scale = rsqrtf(blockSum(ss, sh) / H + embed_eps);
        ss = 0.0f;
        for (int n = 0; n < 9; ++n) {
            e[n] = bf(tobf(e[n] * scale * bf(embed_norm[n * kThreads + threadIdx.x])));
            ss += e[n] * e[n];
        }
    }
    const float scale = rsqrtf(blockSum(ss, sh) / H + kEps);
    for (int n = 0; n < 9; ++n) {
        const int d = n * kThreads + threadIdx.x;
        resid[row * H + d] = e[n];
        x[row * H + d] = tobf(e[n] * scale * bf(norm[d]));
    }
}

// resid += y, then x = norm(resid); one row a block.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_dsp_add_norm(const uint16_t* y, const uint16_t* norm, float* resid, uint16_t* x) {
    __shared__ float sh[kWarps];
    const int row = blockIdx.x;
    float r[9], ss = 0.0f;
    for (int n = 0; n < 9; ++n) {
        const int d = row * H + n * kThreads + threadIdx.x;
        r[n] = resid[d] + bf(y[d]);
        resid[d] = r[n];
        ss += r[n] * r[n];
    }
    const float scale = rsqrtf(blockSum(ss, sh) / H + kEps);
    for (int n = 0; n < 9; ++n) x[row * H + n * kThreads + threadIdx.x] = tobf(r[n] * scale * bf(norm[n * kThreads + threadIdx.x]));
}

// The block's Q and K heads normed and RoPE'd at anchor + row in place (headRope reads every dim before its warp sum);
// one warp a (row, head), heads 0..23 Q, 24..31 K.
extern "C" __global__ void __launch_bounds__(32) tf_ink2_dsp_qk(uint16_t* q, uint16_t* k, const uint16_t* qnorm, const uint16_t* knorm, uint32_t anchor) {
    const int row = blockIdx.x, h = blockIdx.y;
    uint16_t* at = h < 24 ? q + static_cast<size_t>(row) * Q + h * D : k + static_cast<size_t>(row) * KV + (h - 24) * D;
    headRope(at, h < 24 ? qnorm : knorm, at, anchor + row);
}

// Attention of each (row, q head): the context ring's keys from `first` to `committed` (local layers keep the last
// 1024), then the block's 8 rows, no mask; online softmax over 32-key tiles, one warp a (row, head), 4 dims a lane.
extern "C" __global__ void __launch_bounds__(32) tf_ink2_dsp_attn(const uint16_t* q, const uint16_t* bk, const uint16_t* bv, const uint16_t* ring_k, const uint16_t* ring_v, uint16_t* out, uint32_t from, uint32_t committed, int ring, int local) {
    const int row = blockIdx.x, h = blockIdx.y, kv = h / 3, lane = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(D));
    float qd[4], acc[4] = {0, 0, 0, 0}, m = -INFINITY, l = 0.0f;
    for (int j = 0; j < 4; ++j) qd[j] = bf(q[static_cast<size_t>(row) * Q + h * D + lane + 32 * j]);
    const uint32_t first = local && committed > from + kWindow ? committed - kWindow : from;
    const uint32_t held = committed - first, keys = held + kRows;
    auto row_of = [&](const uint16_t* ring_x, const uint16_t* bx, uint32_t key) {
        return key < held ? ring_x + static_cast<size_t>((first + key) % ring) * KV + kv * D : bx + static_cast<size_t>(key - held) * KV + kv * D;
    };
    for (uint32_t t = 0; t < keys; t += 32) {
        // each lane scores one key of the tile (past the end, the last key masked), then the warp folds them in order
        const uint32_t key = min(t + lane, keys - 1);
        const uint16_t* kp = row_of(ring_k, bk, key);
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += __shfl_sync(0xffffffffu, qd[d / 32], d % 32) * bf(kp[d]);
        const float s = t + lane < keys ? dot * scale : -INFINITY;
        const int n = min(32, static_cast<int>(keys - t));
        for (int i = 0; i < n; ++i) {
            const float si = __shfl_sync(0xffffffffu, s, i);
            const uint16_t* vp = row_of(ring_v, bv, t + i);
            const float mn = fmaxf(m, si), c = expf(m - mn), p = expf(si - mn);
            l = l * c + p;
            for (int j = 0; j < 4; ++j) acc[j] = acc[j] * c + p * bf(vp[lane + 32 * j]);
            m = mn;
        }
    }
    for (int j = 0; j < 4; ++j) out[static_cast<size_t>(row) * Q + h * D + lane + 32 * j] = tobf(acc[j] / l);
}

// SwiGLU of the block's gate and up rows [rows][I].
extern "C" __global__ void tf_ink2_dsp_swiglu(const uint16_t* gate, const uint16_t* up, uint16_t* act, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = bf(gate[i]);
    act[i] = tobf(g / (1.0f + expf(-g)) * bf(up[i]));
}

// The Markov chain's step-independent part of z for each step's row: rnn columns 512.. (y_s) and 2816.. (newest) plus
// the bias; one warp a (step, channel). y is the block's rows 1..7.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_dsp_kappa(const uint16_t* y, const uint16_t* newest, const uint16_t* rnn, const uint16_t* bias, float* kappa) {
    const int c = blockIdx.x * kWarps + (threadIdx.x >> 5), step = blockIdx.y, lane = threadIdx.x & 31;
    const uint16_t* w = rnn + static_cast<size_t>(c) * RNN;
    float sum = 0.0f;
    for (int d = lane; d < H; d += 32) sum += bf(y[static_cast<size_t>(step + 1) * H + d]) * bf(w[512 + d]) + bf(newest[d]) * bf(w[2816 + d]);
    sum = warpSum(sum);
    if (lane == 0) kappa[step * 3 * R + c] = sum + bf(bias[c]);
}

// One Markov step on the device: z from the state and the previous token's markov_w1 row (read from `prev`), the
// gated state update, then h = y_s + hidden_out lambda + bias in bf16 for the unembed GEMM. One block of 768 threads.
extern "C" __global__ void __launch_bounds__(3 * R) tf_ink2_dsp_step(const uint32_t* prev, const uint16_t* markov, const uint16_t* rnn, const float* kappa, const uint16_t* hidden_w, const uint16_t* hidden_b, const uint16_t* y, float* state, uint16_t* h) {
    __shared__ float s_in[2 * R], z[3 * R];
    const int c = threadIdx.x;
    if (c < R) {
        s_in[c] = state[c];
        s_in[R + c] = bf(markov[static_cast<size_t>(*prev) * R + c]);
    }
    __syncthreads();
    const uint16_t* w = rnn + static_cast<size_t>(c) * RNN;
    float sum = kappa[c];
    for (int d = 0; d < 2 * R; ++d) sum += s_in[d] * bf(w[d]);
    z[c] = sum;
    __syncthreads();
    if (c < R) {
        const float gamma = 1.0f / (1.0f + expf(-z[c]));
        state[c] = gamma * s_in[c] + (1.0f - gamma) * tanhf(z[R + c]);
        z[c] = tanhf(z[2 * R + c]);  // lambda
    }
    __syncthreads();
    for (int d = c; d < H; d += 3 * R) {
        float v = bf(y[d]) + bf(hidden_b[d]);
        for (int r = 0; r < R; ++r) v += bf(hidden_w[static_cast<size_t>(d) * R + r]) * z[r];
        h[d] = tobf(v);
    }
}

// Softcap the step's logits in place and pick the greedy token into pick[0] (the next step's `prev`); one block.
extern "C" __global__ void __launch_bounds__(1024) tf_ink2_dsp_pick(float* logits, int vocab, float cap, uint32_t* pick) {
    __shared__ float best[32];
    __shared__ int at[32];
    float b = -INFINITY;
    int id = 0;
    for (int v = threadIdx.x; v < vocab; v += 1024) {
        const float x = cap * tanhf(logits[v] / cap);
        logits[v] = x;
        if (x > b) b = x, id = v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, b, o);
        const int oi = __shfl_xor_sync(0xffffffffu, id, o);
        if (ob > b || (ob == b && oi < id)) b = ob, id = oi;
    }
    if ((threadIdx.x & 31) == 0) best[threadIdx.x >> 5] = b, at[threadIdx.x >> 5] = id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; ++w)
            if (best[w] > best[0] || (best[w] == best[0] && at[w] < at[0])) best[0] = best[w], at[0] = at[w];
        *pick = at[0];
    }
}
