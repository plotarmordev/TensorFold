// MiniMax-H3-Image's timestep embedder and AdaLN projections; checkpoint weights stay in their stored dtype.
#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>

__device__ float h3m_read(const void* p, size_t i, int wide) {
    return wide ? static_cast<const float*>(p)[i] : __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

extern "C" __global__ void __launch_bounds__(256) h3m_sinusoid(
        const float* times, float* out, int dim, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * dim) return;
    const int half = dim / 2, c = i % dim;
    const float freq = expf(-logf(10000.0f) * float(c % half) / float(half));
    const float angle = times[i / dim] * freq;
    out[i] = c < half ? cosf(angle) : sinf(angle);
}

// The reference embedder sums in binary64, rounds to float, then applies SiLU in float.
extern "C" __global__ void __launch_bounds__(256) h3m_embed(
        const float* x, const void* w, const void* b, float* out, int rows, int n, int k, int ww, int bw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const int r = i / n, c = i % n;
    double sum = 0;
    for (int j = 0; j < k; ++j)
        sum += double(x[size_t(r) * k + j]) * double(h3m_read(w, size_t(c) * k + j, ww));
    const float v = float(sum + double(h3m_read(b, c, bw)));
    out[i] = v / (1.0f + expf(-v));
}

// One warp per output; the 32 partials meet in lane order, as rows_reference.modulation does. The tables are
// bf16: the reference rounds every AdaLN value there.
extern "C" __global__ void __launch_bounds__(256) h3m_project(
        const float* x, const void* w, const void* b, __nv_bfloat16* out, int rows, int n, int k, int ww, int bw) {
    const int lane = threadIdx.x % 32;
    const int c = blockIdx.x * 8 + threadIdx.x / 32;
    const int r = blockIdx.y;
    if (c >= n || r >= rows) return;
    float sum = 0;
    for (int j = lane; j < k; j += 32) {
        const float v = __bfloat162float(__float2bfloat16_rn(x[size_t(r) * k + j]));
        sum = __fadd_rn(sum, __fmul_rn(v, h3m_read(w, size_t(c) * k + j, ww)));
    }
    float total = 0;
    for (int j = 0; j < 32; ++j) total = __fadd_rn(total, __shfl_sync(0xffffffffu, sum, j));
    if (lane == 0) out[size_t(r) * n + c] = __float2bfloat16_rn(__fadd_rn(total, h3m_read(b, c, bw)));
}
