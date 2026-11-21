// H3's products at 64 rows x 128 columns a threadgroup, each warp 32 x 64: every loaded fragment feeds two MMAs
// instead of one. Each output's K chain is MLX's, 32-deep steps in order. The int8 form takes both operands
// quantized by h3_quantize_rows a scale per 256 values of K: each 256-deep block's int32 sums fold into fp32 at
// both scales for that block.
namespace tfp {

// acc += a b^T over one 32-deep step: each a fragment feeds two MMAs.
template <typename T, typename C>
inline void h3_mma(thread frag<C> (&acc)[2][4], thread const frag<T> (&a)[2][2], thread const frag<T> (&b)[2][4]) {
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short k = 0; k < 2; k++) {
      mma_16x32<false, true>(acc[m][0], acc[m][1], a[m][k], b[k][0], b[k][1]);
      mma_16x32<false, true>(acc[m][2], acc[m][3], a[m][k], b[k][2], b[k][3]);
    }
  }
}

template <bool BOUND, typename T, typename C>
inline void h3_step(thread frag<C> (&acc)[2][4], const device T* A, const device T* B, int lda, int ldb, int kk, int kn,
                    int rows, int cols, short2 home) {
  frag<T> a[2][2], b[2][4];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short k = 0; k < 2; k++) {
      if (BOUND) frag_get_in(a[m][k], A, lda, 16 * m, kk + 16 * k, home, rows, kk + kn);
      else frag_get(a[m][k], A, lda, 16 * m, kk + 16 * k, home);
    }
  }
  TF_UNROLL
  for (short k = 0; k < 2; k++) {
    TF_UNROLL
    for (short n = 0; n < 4; n++) {
      if (BOUND) frag_get_in(b[k][n], B, ldb, 16 * n, kk + 16 * k, home, cols, kk + kn);
      else frag_get(b[k][n], B, ldb, 16 * n, kk + 16 * k, home);
    }
  }
  h3_mma(acc, a, b);
}

// D [P[0], P[1]] = A [P[0], P[2]] * B^T [P[1], P[2]], N a multiple of 128; AM false bounds the rows. Int8 operands
// come with SA [P[0], P[9]] and SB [P[1], P[9]], a scale per 256-deep block of K, K a multiple of 256.
template <bool AM, typename T, typename C>
inline void h3_gemm(const device T* A, const device T* B, device bfloat* D, const device int* P, const device float* SA,
                    const device float* SB, uint3 tg, uint sg, uint lane) {
  constexpr bool I8 = is_same_v<T, int8_t>;
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1)), tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) return;
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * 64 + 32 * int(sg / 2), col = tx * 128 + 64 * int(sg % 2);
  A += long(P[13]) + long(P[10]) * tg.z + long(row) * lda;
  B += long(P[14]) + long(P[11]) * tg.z + long(col) * ldb;
  D += long(P[12]) * tg.z + long(row) * ldd + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = 64;
  const bool full = AM || rows == 32;
  const bool live = rows > 0;
  const short2 home = frag_home(ushort(lane));
  frag<C> acc[2][4];
  frag<float> total[2][4];
  TF_UNROLL
  for (short m = 0; m < 2; m++) TF_UNROLL for (short n = 0; n < 4; n++) acc[m][n] = frag<C>(0), total[m][n] = frag<float>(0);
  const int kblocks = P[9];
#pragma clang loop unroll(disable)
  for (int kb = 0; kb < kblocks; kb++) {
    threadgroup_barrier(mem_flags::mem_none);
    if (full) {
#pragma clang loop unroll(disable)
      for (int kk = 0; kk < 256; kk += 32) h3_step<false>(acc, A, B, lda, ldb, kk, 32, rows, cols, home);
    } else if (live) {
#pragma clang loop unroll(disable)
      for (int kk = 0; kk < 256; kk += 32) h3_step<true>(acc, A, B, lda, ldb, kk, 32, rows, cols, home);
    }
    if (I8 && live) {
      // lane columns col + 16 n + home.x + (e & 3), each with its scale for this block, and rows row + 16 m +
      // home.y or 8 below; a ragged tile's dead rows read the last row's scale
      TF_UNROLL
      for (short n = 0; n < 4; n++) {
        const device float* s = SB + long(col + 16 * n + home.x) * kblocks + kb;
        TF_UNROLL
        for (short m = 0; m < 2; m++) {
          const int r = row + 16 * m + home.y, last = P[0] - 1;
          const float r0 = SA[long(min(r, last)) * kblocks + kb], r1 = SA[long(min(r + 8, last)) * kblocks + kb];
          TF_UNROLL
          for (short e = 0; e < 8; e++) total[m][n][e] += float(acc[m][n][e]) * s[(e & 3) * kblocks] * (e < 4 ? r0 : r1);
          acc[m][n] = frag<C>(0);
        }
      }
    }
    A += 256;
    B += 256;
  }
  const int rest = P[2] - kblocks * 256;
  if (rest > 0) {
    simdgroup_barrier(mem_flags::mem_none);
    if (live) {
#pragma clang loop unroll(disable)
      for (int kk = 0; kk < rest; kk += 32) h3_step<true>(acc, A, B, lda, ldb, kk, max(0, rest - kk), rows, cols, home);
    }
  }
  if (!live) return;
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short n = 0; n < 4; n++) {
      const frag<float> f = I8 ? total[m][n] : frag<float>(acc[m][n]);
      if (full) frag_put(f, D, ldd, 16 * m, 16 * n, home);
      else frag_put_in(f, D, ldd, 16 * m, 16 * n, home, rows, cols);
    }
  }
}

// The int8 product stages each K step's operands in threadgroup memory: the threadgroup's 64 rows of A and 128 of B,
// read 16 contiguous bytes a thread. A step multiplies from one stage while the next step's device reads are in
// flight, held in registers until the stage they overwrite is free. Rows sit kI8Depth + 16 bytes apart so the lanes
// of a fragment read distinct banks.
constant constexpr int kI8Depth = 32;
constant constexpr int kI8Pitch = kI8Depth + 16;
constant constexpr int kI8Stage = 192 * kI8Pitch;
// the rows a pass of the threadgroup's 128 threads fills, and its passes over A's 64 rows and B's 128
constant constexpr int kI8Rows = 128 * 16 / kI8Depth;
constant constexpr int kI8PassA = 64 / kI8Rows, kI8PassB = 128 / kI8Rows;

// The 16x16 block at (r, c) of a staged tile: a lane's 4 bytes of a row and of the row 8 below, one word each.
inline void frag_get_i8(thread frag<int8_t>& f, const threadgroup int8_t* tile, int r, int c, short2 home) {
  const threadgroup vec<int8_t, 4>* q = (const threadgroup vec<int8_t, 4>*)(tile + (r + home.y) * kI8Pitch + c + home.x);
  f = frag<int8_t>(q[0], q[2 * kI8Pitch]);
}

// h3_gemm's int8 form (P[0..9] alike) over a staged tile pair; dead rows of a ragged tile read A's last row.
inline void h3_gemm_i8(const device int8_t* A, const device int8_t* B, device bfloat* D, const device int* P,
                       const device float* SA, const device float* SB, threadgroup int8_t* tile, uint3 tg, uint sg,
                       uint lane) {
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1)), tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) return;
  const int lda = P[3], ldb = P[4], ldd = P[5], kblocks = P[9], last = P[0] - 1;
  // this thread's pieces: bytes at .. at + 15 of a step in rows h + kI8Rows p of the threadgroup's A, then of its B
  const int t = int(sg) * 32 + int(lane), h = t / (kI8Depth / 16), at = 16 * (t % (kI8Depth / 16));
  const device int8_t* src[kI8PassA + 1];
  TF_UNROLL
  for (short p = 0; p < kI8PassA; p++) src[p] = A + long(min(ty * 64 + h + kI8Rows * p, last)) * lda + at;
  src[kI8PassA] = B + long(tx * 128 + h) * ldb + at;
  threadgroup int8_t* dst = tile + h * kI8Pitch + at;
  const int row = ty * 64 + 32 * int(sg / 2), col = tx * 128 + 64 * int(sg % 2);
  const int rows = min(32, P[0] - row);
  const bool live = rows > 0;
  const short2 home = frag_home(ushort(lane));
  const int wa = 32 * int(sg / 2) * kI8Pitch, wb = (64 + 64 * int(sg % 2)) * kI8Pitch;
  frag<int> acc[2][4];
  frag<float> total[2][4];
  TF_UNROLL
  for (short m = 0; m < 2; m++) TF_UNROLL for (short n = 0; n < 4; n++) acc[m][n] = frag<int>(0), total[m][n] = frag<float>(0);
  int4 next[kI8PassA + kI8PassB];
  const auto fetch = [&](int k) {
    TF_UNROLL
    for (short p = 0; p < kI8PassA; p++) next[p] = *(const device int4*)(src[p] + k);
    TF_UNROLL
    for (short p = 0; p < kI8PassB; p++) next[kI8PassA + p] = *(const device int4*)(src[kI8PassA] + long(kI8Rows * p) * ldb + k);
  };
  const auto stash = [&](int s) {
    TF_UNROLL
    for (short p = 0; p < kI8PassA + kI8PassB; p++) *(threadgroup int4*)(dst + (s & 1) * kI8Stage + kI8Rows * p * kI8Pitch) = next[p];
  };
  fetch(0);
  stash(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int per = 256 / kI8Depth;
  const int steps = per * kblocks;
#pragma clang loop unroll(disable)
  for (int s = 0; s < steps; s++) {
    const bool more = s + 1 < steps;
    if (more) fetch((s + 1) * kI8Depth);
    if (live) {
      const threadgroup int8_t* stage = tile + (s & 1) * kI8Stage;
      TF_UNROLL
      for (short kk = 0; kk < kI8Depth; kk += 32) {
        frag<int8_t> a[2][2], b[2][4];
        TF_UNROLL
        for (short k = 0; k < 2; k++) {
          TF_UNROLL
          for (short m = 0; m < 2; m++) frag_get_i8(a[m][k], stage + wa, 16 * m, kk + 16 * k, home);
          TF_UNROLL
          for (short n = 0; n < 4; n++) frag_get_i8(b[k][n], stage + wb, 16 * n, kk + 16 * k, home);
        }
        h3_mma(acc, a, b);
      }
    }
    if (live && s % per == per - 1) {
      // lane columns col + 16 n + home.x + (e & 3), each with its scale for this block, and rows row + 16 m +
      // home.y or 8 below; a ragged tile's dead rows read the last row's scale
      const int kb = s / per;
      TF_UNROLL
      for (short n = 0; n < 4; n++) {
        const device float* sb = SB + long(col + 16 * n + home.x) * kblocks + kb;
        TF_UNROLL
        for (short m = 0; m < 2; m++) {
          const int r = row + 16 * m + home.y;
          const float r0 = SA[long(min(r, last)) * kblocks + kb], r1 = SA[long(min(r + 8, last)) * kblocks + kb];
          TF_UNROLL
          for (short e = 0; e < 8; e++) total[m][n][e] += float(acc[m][n][e]) * sb[(e & 3) * kblocks] * (e < 4 ? r0 : r1);
          acc[m][n] = frag<int>(0);
        }
      }
    }
    if (more) stash(s + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!live) return;
  D += long(row) * ldd + col;
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short n = 0; n < 4; n++) {
      if (rows == 32) frag_put(total[m][n], D, ldd, 16 * m, 16 * n, home);
      else frag_put_in(total[m][n], D, ldd, 16 * m, 16 * n, home, rows, 64);
    }
  }
}

}  // namespace tfp

// Each of P[0] rows of X (P[1] values) to int8 in groups of P[2], each at its absolute max over 127: Q = rint(X / S),
// S [row, group] its scale (non-finite when the group holds NaN or an infinity). One simdgroup a row, its groups in turn.
[[kernel]] void h3_quantize_rows(
  const device bfloat16_t* X [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device int8_t* Q [[buffer(2)]],
  device float* S [[buffer(3)]],
  uint at [[threadgroup_position_in_grid]],
  uint lane [[thread_index_in_simdgroup]]) {
  const int k = P[1], g = P[2];
  X += long(at) * k;
  Q += long(at) * k;
  S += long(at) * (k / g);
  for (int from = 0; from < k; from += g) {
    float most = 0;
    bool nan = false;
    for (int i = from + int(lane); i < from + g; i += 32) {
      const float v = abs(float(X[i]));
      most = max(most, v);
      nan = nan || isnan(v);
    }
    // max drops NaN: a group holding one takes a NaN scale, so its products come out NaN
    most = simd_any(nan) ? NAN : simd_max(most);
    const float scale = most > 0 || isnan(most) ? most / 127 : 1;
    for (int i = from + int(lane); i < from + g; i += 32) Q[i] = int8_t(rint(float(X[i]) / scale));
    if (lane == 0) S[from / g] = scale;
  }
}

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void h3_gemm_32x64(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::h3_gemm<true, bfloat, float>(A, B, D, P, nullptr, nullptr, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void h3_gemm_32x64_ragged(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::h3_gemm<false, bfloat, float>(A, B, D, P, nullptr, nullptr, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void h3_gemm_32x64_i8(
  const device int8_t* A [[buffer(0)]],
  const device int8_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  const device float* SA [[buffer(4)]],
  const device float* SB [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup int8_t tile[2 * tfp::kI8Stage];
  tfp::h3_gemm_i8(A, B, D, P, SA, SB, tile, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}
