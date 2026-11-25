// Shared M5 tensor-unit op and ordered reduction for Inkling 2 Mini's BF16 projections.
using namespace metal;

struct Ink2GroupedArgs { uint k, n, topk, cap; };
struct Ink2GroupArgs { uint pairs, experts, cap; };

inline uint ink2_word(device const uchar* p) {
  return as_type<uint>(uchar4(*(device const packed_uchar4*)p));
}
inline float ink2_e2m1_twice(uint code) {
  const float v = float((0xC8643210u >> (4u * (code & 7u))) & 15u);
  return (code & 8u) ? -v : v;
}
inline float ink2_e4m3_bits(uint code) {
  const uint exp = (code >> 3u) & 15u, mant = code & 7u;
  const float v = exp ? as_type<float>(((exp + 120u) << 23u) | (mant << 20u)) : float(mant) * (1.0f / 512.0f);
  const float s = (code & 0x80u) ? -v : v;
  return (exp == 15u && mant == 7u) ? NAN : s;
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_group_units(
    device const uint* ids [[buffer(0)]], device uint* counts [[buffer(1)]],
    device uint* members [[buffer(2)]], constant Ink2GroupArgs& a [[buffer(3)]],
    uint e [[thread_position_in_grid]]) {
  if (e >= a.experts) return;
  uint n = 0u;
  for (uint p = 0u; p < a.pairs; ++p) if (ids[p] == e) {
    if (n < a.cap) members[e * a.cap + n] = p;
    ++n;
  }
  counts[e] = min(n, a.cap);
}

inline void ink2_mma(thread tfp::frag<float>& lo, thread tfp::frag<float>& hi, thread const tfp::frag<bfloat>& a,
                     thread const tfp::frag<bfloat>& b0, thread const tfp::frag<bfloat>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto right = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, float>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}
inline void ink2_part_put(threadgroup float* part, thread const tfp::frag<float>& lo,
                          thread const tfp::frag<float>& hi, uint sg, uint lane) {
  for (uint e = 0u; e < 8u; ++e) {
    part[(sg * 16u + e) * 32u + lane] = lo[e];
    part[(sg * 16u + 8u + e) * 32u + lane] = hi[e];
  }
}
inline void ink2_part_sum(threadgroup float* part, thread tfp::frag<float>& lo, thread tfp::frag<float>& hi, uint lane) {
  for (uint e = 0u; e < 8u; ++e) for (uint o = 1u; o < 8u; ++o) {
    lo[e] += part[(o * 16u + e) * 32u + lane];
    hi[e] += part[(o * 16u + 8u + e) * 32u + lane];
  }
}
