// The int8 path's shared rotation: the 128-wide Hadamard transform both operands quantize under (mmh3
// ConvRot's outlier spreading). Included by h3.cu (h3_quantize, h3_norm_quant) and h3_image_prompt.cu
// (h3p_quantize), so both fatbins quantize under the same rotation.
#pragma once
#include <cuda_bf16.h>

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
