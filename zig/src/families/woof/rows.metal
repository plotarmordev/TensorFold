// Woof's ops.rows on the GPU: y[r][o] = x[r] . W[o] (+ b[o]) (+ res[r][o]), W packed in panels as ops.Linear.
#include <metal_stdlib>
using namespace metal;

struct Args {
    uint out;
    uint in;
    uint n;
    uint x_stride;
    uint y_stride;
    uint has_b;
    uint has_res;
};

constant uint lanes = 16;
constant uint parts = 16;
constant uint block = 8;

// A group: one panel's 16 outputs for `block` rows; part q sums inputs q, q + parts, ..., then the parts sum in order.
kernel void woof_rows(device const float* w [[buffer(0)]], device const float* b [[buffer(1)]], device const float* x [[buffer(2)]],
                      device const float* res [[buffer(3)]], device float* y [[buffer(4)]], constant Args& a [[buffer(5)]],
                      uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[parts][block][lanes];
    const uint j = tid % lanes;
    const uint q = tid / lanes;
    const uint r0 = group.y * block;
    device const float* col = w + group.x * a.in * lanes + j;
    uint rs[block];
    for (uint k = 0; k < block; k++) rs[k] = min(r0 + k, a.n - 1) * a.x_stride;
    float acc[block] = {};
    for (uint i = q; i < a.in; i += parts) {
        const float wv = col[i * lanes];
        for (uint k = 0; k < block; k++) acc[k] = fma(x[rs[k] + i], wv, acc[k]);
    }
    for (uint k = 0; k < block; k++) sums[q][k][j] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= block * lanes) return;
    const uint k = tid / lanes;
    const uint o = group.x * lanes + j;
    if (o >= a.out || r0 + k >= a.n) return;
    float s = 0;
    for (uint p = 0; p < parts; p++) s += sums[p][k][j];
    const uint at = (r0 + k) * a.y_stride + o;
    y[at] = s + (a.has_b ? b[o] : 0) + (a.has_res ? res[at] : 0);
}
