// MiniMax-H3-Image's prompt stage on CUDA: the glue between the checkpoint's bf16 cuBLASLt products,
// rounding exactly as the Metal kernels (zig/kernels/metal/h3/h3_prompt.metal) and the host reference
// (zig/src/families/h3/prompt_reference.zig) do: bf16 in, fp32 sums, every output rounded once to bf16.
#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>
#include "h3_int8.cuh" // hadamard128 and rotated_chunk, the rotation h3p_quantize quantizes under

typedef __nv_bfloat16 bf16;

static __device__ float f(bf16 v) { return __bfloat162float(v); }
static __device__ bf16 h(float v) { return __float2bfloat16_rn(v); }

// 1 / sqrt(x) and exp(x) one rounding from exact, as the host reference's float32 run computes them;
// rsqrtf and expf are up to 2 ULPs off, an error every value of a norm or softmax row carries.
static __device__ float inv_sqrt(float x) { return __fdiv_rn(1.0f, __fsqrt_rn(x)); }
static __device__ float exact_exp(float x) { return (float)exp((double)x); }

// One RMSNorm value: mode 0 bf16(w * bf16(x * r)) as the Qwen text model rounds, mode 1 bf16(x * r * w)
// as H3's blocks round. The weight is fp32: checkpoint bf16 values convert exactly.
static __device__ float h3p_normed(float x, float r, float w, int mode) {
    return mode == 0 ? f(h(__fmul_rn(w, f(h(__fmul_rn(x, r)))))) : f(h(__fmul_rn(__fmul_rn(x, r), w)));
}

// y = RMSNorm(x) by w, x bf16, w fp32, y bf16: squares summed in the host's order, as a tree flips bf16 outputs.
extern "C" __global__ void __launch_bounds__(256) h3p_norm(
        const bf16* x, const float* w, bf16* y, int width, int mode, float eps) {
    __shared__ float r;
    const int row = blockIdx.x;
    const bf16* xr = x + (size_t)row * width;
    bf16* yr = y + (size_t)row * width;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < width; ++i) s = __fadd_rn(s, __fmul_rn(f(xr[i]), f(xr[i])));
        r = inv_sqrt(__fadd_rn(s / (float)width, eps));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += 256) yr[i] = h(h3p_normed(f(xr[i]), r, w[i], mode));
}

// q then k heads of 128 normed in place (mode as h3p_norm) and turned by cos, sin [rows, 64]; a warp a head, in order.
extern "C" __global__ void __launch_bounds__(256) h3p_heads(
        bf16* q, bf16* k, const float* wq, const float* wk, const float* cs, const float* sn,
        int heads, int kv_heads, int q_row, int k_row, int head, int mode, int rotary, float eps) {
    const int row = blockIdx.x;
    const int hh = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (hh >= heads + kv_heads) return;
    const int lane = threadIdx.x & 31;
    const bool is_q = hh < heads;
    bf16* x = is_q ? q + (size_t)row * q_row + (size_t)hh * head
                   : k + (size_t)row * k_row + (size_t)(hh - heads) * head;
    const float* w = is_q ? wq : wk;
    float ss = 0.0f;
    for (int i = 0; i < 128; ++i) ss = __fadd_rn(ss, __fmul_rn(f(x[i]), f(x[i])));
    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = f(x[lane * 4 + i]);
    __syncwarp();
    const float r = inv_sqrt(__fadd_rn(ss / 128.0f, eps));
    for (int i = 0; i < 4; ++i) v[i] = h3p_normed(v[i], r, w[lane * 4 + i], mode);
    if (rotary) {
        // channel c pairs with c + 64, 16 lanes away: rotate-half reads -x[c + 64] below the half
        // and x[c - 64] above; both multiply's ends round to bf16 before the add, as the Metal kernel's
        const float sign = lane < 16 ? -1.0f : 1.0f;
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], 16);
            const size_t at = (size_t)row * 64 + (lane & 15) * 4 + i;
            v[i] = __fadd_rn(f(h(__fmul_rn(v[i], cs[at]))), f(h(__fmul_rn(sign * o, sn[at]))));
        }
    }
    for (int i = 0; i < 4; ++i) x[lane * 4 + i] = h(v[i]);
}

// One key's score: q . k over 128 in order, times scale, as the host reference's.
static __device__ float h3p_score(const bf16* q, const bf16* k, float scale) {
    float s = 0.0f;
    for (int i = 0; i < 128; ++i) s = __fadd_rn(s, __fmul_rn(f(q[i]), f(k[i])));
    return __fmul_rn(s, scale);
}

// o = softmax(q k^T / sqrt(128)) v in bf16, a warp a row's head: the top score, then the host's in-order sums.
extern "C" __global__ void __launch_bounds__(256) h3p_attention(
        const bf16* q, const bf16* k, const bf16* v, bf16* o,
        int rows, int heads, int kv_heads, int q_row, int k_row, int v_row, int head,
        int causal, float scale) {
    const int row = blockIdx.x;
    const int hh = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (hh >= heads) return;
    const int lane = threadIdx.x & 31;
    const int kv = hh / (heads / kv_heads);
    const bf16* qp = q + (size_t)row * q_row + (size_t)hh * head;
    const bf16* kp = k + (size_t)kv * head;
    const bf16* vp = v + (size_t)kv * head + lane * 4;
    const int end = causal ? row + 1 : rows;
    float top = -INFINITY;
    for (int j = lane; j < end; j += 32) top = fmaxf(top, h3p_score(qp, kp + (size_t)j * k_row, scale));
    for (int k_ = 16; k_; k_ >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, k_));
    float l = 0.0f;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int j0 = 0; j0 < end; j0 += 32) {
        // the warp's next 32 keys' weights, one a lane, then read back in key order
        const int j = j0 + lane;
        const float mine = j < end ? exact_exp(__fadd_rn(h3p_score(qp, kp + (size_t)j * k_row, scale), -top)) : 0.0f;
        for (int u = 0; u < 32 && j0 + u < end; ++u) {
            const float e = __shfl_sync(0xffffffffu, mine, u);
            l = __fadd_rn(l, e);
            const bf16* vr = vp + (size_t)(j0 + u) * v_row;
            for (int i = 0; i < 4; ++i) acc[i] = __fadd_rn(acc[i], __fmul_rn(e, f(vr[i])));
        }
    }
    bf16* out = o + (size_t)row * heads * 128 + (size_t)hh * 128 + lane * 4;
    for (int i = 0; i < 4; ++i) out[i] = h(__fdiv_rn(acc[i], l));
}

// o [rows, width] = silu(g) * v rounded to bf16: mode 0 bf16(bf16(g / e) v) as the text model rounds,
// mode 1 bf16(bf16(g * bf16(1 / e)) v) as H3's blocks do; g and v bf16, their rows may differ in
// stride (the refiner's fused fc1 rows hold them side by side).
extern "C" __global__ void __launch_bounds__(256) h3p_glu(
        const bf16* g, const bf16* v, bf16* o, int width, int g_row, int v_row, int mode) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int row = blockIdx.y;
    if (c >= width) return;
    const float x = f(g[(size_t)row * g_row + c]);
    const float e = 1.0f + exact_exp(-x);
    const float s = mode == 0 ? f(h(__fdiv_rn(x, e))) : f(h(__fmul_rn(x, f(h(__fdiv_rn(1.0f, e))))));
    o[(size_t)row * width + c] = h(__fmul_rn(s, f(v[(size_t)row * v_row + c])));
}


// x = bf16(x + bf16(y)): round the fp32 product before the residual add.
extern "C" __global__ void __launch_bounds__(256) h3p_add(bf16* x, const float* y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] = h(__fadd_rn(f(x[i]), f(h(y[i]))));
}

// x rows idx[i] = bf16(x + y[i]): DeepStack's features added at the image pads, y bf16.
extern "C" __global__ void __launch_bounds__(256) h3p_scatter_add(
        bf16* x, const bf16* y, const unsigned* idx, int width) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int i = blockIdx.y;
    if (c >= width) return;
    const size_t to = (size_t)idx[i] * width + c;
    x[to] = h(__fadd_rn(f(x[to]), f(y[(size_t)i * width + c])));
}

// The embedding gather: x rows = table's rows `ids`, bf16 copied as stored.
extern "C" __global__ void __launch_bounds__(256) h3p_embed(
        const bf16* table, const unsigned* ids, bf16* x, int width) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int row = blockIdx.y;
    if (c >= width) return;
    x[(size_t)row * width + c] = table[(size_t)ids[row] * width + c];
}

// Vision rows spliced into the stream: x rows pads[i] = rows[i], whole bf16 rows.
extern "C" __global__ void __launch_bounds__(256) h3p_splice(
        bf16* x, const bf16* rows_, const unsigned* pads, int width) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int i = blockIdx.y;
    if (c >= width) return;
    x[(size_t)pads[i] * width + c] = rows_[(size_t)i * width + c];
}

// y [rows, n] fp32 = x [rows, k] bf16 . w [n, k] bf16^T, each K sum in the host reference's order.
// The text encoder's short rows make a sequential dot product preferable to a differently ordered tiled GEMM:
// its bf16 rounding points then coincide with the reference's instead of failing a zero-noise vision baseline.
// y [rows,n] = x [rows,k] . w [n,k]^T: a warp owns one output column and its lanes load the
// column's k weights in coalesced 256-element windows (eight a lane, one 16-byte load), so a
// column's row is read once per 32-row group instead of once per row and 512 bytes stay in flight
// per warp. Each lane sums its own row over those windows in the reference's k order, eight at a
// time with the loads issued together, so every output's fp32 sum is the sequential dot the host
// computes. A 32-element window left the stage memory-latency bound: 45 GiB took 5.5 s, 8 GB/s.
extern "C" __global__ void __launch_bounds__(256) h3p_linear(
        const bf16* x, const bf16* w, float* y, int rows, int n, int k) {
    __shared__ __align__(16) bf16 win[8][256]; // one window a warp
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.x * 8 + warp;
    if (col >= n) return;
    const bf16* wr = w + (size_t)col * k;
    // k a multiple of 8 keeps the column's start and every eight-element step 16-byte aligned
    const bool vec = ((size_t)wr & 15) == 0;
    for (int base = 0; base < rows; base += 32) {
        const int row = base + lane;
        const bf16* xr = x + (size_t)row * k;
        float sum = 0.0f;
        for (int i0 = 0; i0 < k; i0 += 256) {
            bf16* dst = win[warp] + lane * 8;
            const int at = i0 + lane * 8;
            if (vec && at + 8 <= k) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(wr + at);
            else for (int u = 0; u < 8; ++u) dst[u] = at + u < k ? wr[at + u] : h(0.0f);
            __syncwarp();
            const int jn = min(256, k - i0);
            for (int u = 0; u < jn; u += 8) {
                // the row's next eight x values, then their products in the reference's k order
                float xv[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                const int atx = i0 + u;
                if (row < rows) {
                    if (vec && atx + 8 <= k) {
                        __align__(16) bf16 xs[8];
                        *reinterpret_cast<uint4*>(xs) = *reinterpret_cast<const uint4*>(xr + atx);
                        for (int i = 0; i < 8; ++i) xv[i] = f(xs[i]);
                    } else for (int i = 0; i < 8; ++i) if (atx + i < k) xv[i] = f(xr[atx + i]);
                    for (int i = 0; i < 8 && u + i < jn; ++i)
                        sum = __fadd_rn(sum, __fmul_rn(xv[i], f(win[warp][u + i])));
                }
            }
            __syncwarp();
        }
        if (row < rows) y[(size_t)row * n + col] = sum;
    }
}

// Each bf16 row of X (M, K) rotated in 128-wide chunks and quantized to int8 at the row's absmax / 127, the scale in
// S — h3_quantize's math with the row streamed twice instead of register-resident, so any multiple-of-128 K fits
// (the text encoder's down product reads 25,600 wide). A row and its weights' rows rotate alike, so their products'
// sums are the unrotated ones. Grid [M], 256 threads.
extern "C" __global__ void __launch_bounds__(256) h3p_quantize(const bf16* X, signed char* Q, float* S, int K) {
    __shared__ float warp_max[8];
    __shared__ float scale;
    const size_t row = blockIdx.x;
    const bf16* x = X + row * K;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, chunks = K >> 7;
    float most = 0.0f;
    for (int c = warp; c < chunks; c += 8) {
        float v[4];
        rotated_chunk(x, c, v);
        for (int j = 0; j < 4; ++j) most = fmaxf(most, fabsf(v[j]));
    }
    for (int o = 16; o; o >>= 1) most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, o));
    if (lane == 0) warp_max[warp] = most;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = warp_max[0];
        for (int w = 1; w < 8; ++w) m = fmaxf(m, warp_max[w]);
        scale = m > 0.0f ? m / 127.0f : 1.0f;
        S[row] = scale;
    }
    __syncthreads();
    for (int c = warp; c < chunks; c += 8) {
        float v[4];
        rotated_chunk(x, c, v);
        const char4 q = make_char4(__float2int_rn(v[0] / scale), __float2int_rn(v[1] / scale), __float2int_rn(v[2] / scale), __float2int_rn(v[3] / scale));
        *reinterpret_cast<char4*>(Q + row * K + c * 128 + lane * 4) = q;
    }
}

// Y (M, N) = bf16(ACC SX[row] SW[col]): an int8 product's int32 sums back to bf16 rows, h3_dequantize's rounding.
// With `add` set the product is the delta of a residual instead: Y = bf16(Y + bf16(product)), the rounding h3p_add
// gives the bf16 path's fp32 product, so the int8 layer's o and down products add exactly as the bf16 ones do.
// Grid-stride, four a thread.
extern "C" __global__ void __launch_bounds__(256) h3p_dequantize(
        const int* ACC, const float* SX, const float* SW, bf16* Y, int M, int N, int add) {
    const size_t quads = static_cast<size_t>(M) * N / 4;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t at = i * 4, r = at / N, n = at % N;
        const int4 a = reinterpret_cast<const int4*>(ACC)[i];
        const float4 w = *reinterpret_cast<const float4*>(SW + n);
        const float s = SX[r];
        const float p[4] = { a.x * s * w.x, a.y * s * w.y, a.z * s * w.z, a.w * s * w.w };
        bf16* y = Y + at;
        if (add)
            for (int j = 0; j < 4; ++j) y[j] = h(__fadd_rn(f(y[j]), f(h(p[j]))));
        else {
            __nv_bfloat162* pair = reinterpret_cast<__nv_bfloat162*>(y);
            pair[0] = __floats2bfloat162_rn(p[0], p[1]);
            pair[1] = __floats2bfloat162_rn(p[2], p[3]);
        }
    }
}

// y [n] = bf16(x [n]): a product's fp32 rows at their one rounding to bf16, the reference's every output.
extern "C" __global__ void __launch_bounds__(256) h3p_round(const float* x, bf16* y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = h(x[i]);
}
