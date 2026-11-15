// MiniMax-H3-Image's prompt stage on CUDA: the glue between the checkpoint's bf16 cuBLASLt products,
// rounding exactly as the Metal kernels (zig/kernels/metal/h3/h3_prompt.metal) and the host reference
// (zig/src/families/h3/prompt_reference.zig) do: bf16 in, fp32 sums, every output rounded once to bf16.
#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>

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

// y = RMSNorm(x) by w with fp32 statistics, its squares summed in order as the host reference sums
// them: a tree's other order moves the scale a ULP and flips bf16 outputs. x bf16, w fp32, y bf16.
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

// The text model's heads: q then k heads of 128 normed in place (mode as h3p_norm), turned by
// cos, sin [rows, 64]. Heads sit `head` apart in rows `q_row`/`k_row` wide; a warp a head, each
// lane summing the head's 128 squares in order as the host reference does.
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

// o [rows, heads * 128] = softmax(q k^T / sqrt(128)) v over earlier rows (causal) or all, rounded to
// bf16 as the reference stores it. The reference's two passes in its order: the top score (order-free,
// lanes split the keys), then each key's weight once, the weight sum and each lane's 4 value sums over
// the keys in order. A warp a row's head; q, k, v, o bf16; heads sit `head` apart in rows q_row/k_row/v_row.
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
extern "C" __global__ void __launch_bounds__(256) h3p_linear(
        const bf16* x, const bf16* w, float* y, int rows, int n, int k) {
    const int at = blockIdx.x * 256 + threadIdx.x;
    if (at >= rows * n) return;
    const int row = at / n;
    const int col = at % n;
    const bf16* xr = x + (size_t)row * k;
    const bf16* wr = w + (size_t)col * k;
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum = __fadd_rn(sum, __fmul_rn(f(xr[i]), f(wr[i])));
    y[at] = sum;
}

// y [n] = bf16(x [n]): a product's fp32 rows at their one rounding to bf16, the reference's every output.
extern "C" __global__ void __launch_bounds__(256) h3p_round(const float* x, bf16* y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = h(x[i]);
}
