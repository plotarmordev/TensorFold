// MiniMax H3 / FastH3 on Metal: tile-routed attention in bf16 on the tensor units, fp32 sums and an online softmax.

// softmax(q k^T) v over a query tile's key tiles in bf16, fp32 sums and an online softmax. [query tiles, H] of 256.
[[kernel]] void h3_attention(
  const device bfloat* Q [[buffer(0)]],
  const device bfloat* K [[buffer(1)]],
  const device bfloat* V [[buffer(2)]],
  const device int32_t* IDX [[buffer(3)]],
  const device int32_t* SIZES [[buffer(4)]],
  const device int32_t* ROWOF [[buffer(5)]],
  const constant int32_t* P [[buffer(6)]],
  const constant float* SC [[buffer(7)]],
  device bfloat* Y [[buffer(8)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int D = 128, TQ = 64, KEYS = 64, TS = H3_SLOTS, H = H3_HEADS;
  constexpr int TPR = 256 / TQ;
  constexpr int C1 = TQ * KEYS / 256, C2 = TQ * D / 256;
  const int NQ = P[0], KSEL = P[1];
  const int head = int(tg.y), r0 = (P[2] + int(tg.x)) * TQ;
  const long hb = long(head) * TS;
#ifdef H3_FAR
  // a far-field binding's routed lists are its own: one a query tile, shared by every head
  const device int32_t* list = IDX + (P[3] ? long(tg.x) * KSEL : 0);
#else
  const device int32_t* list = IDX + (P[3] ? (long(head) * NQ + tg.x) * KSEL : 0);
#endif
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> q((device bfloat*)Q + hb * D, dextents<int32_t, 2>(D, TS));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> k((device bfloat*)K + hb * D, dextents<int32_t, 2>(D, TS));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> v((device bfloat*)V + hb * D, dextents<int32_t, 2>(D, TS));
  threadgroup float st[TQ * KEYS];
  threadgroup bfloat pt[TQ * KEYS];
  threadgroup float tops[256];
  threadgroup float fac[TQ];
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> p((threadgroup bfloat*)pt, dextents<int32_t, 2>(KEYS, TQ));
  constexpr auto d1 = matmul2d_descriptor(TQ, KEYS, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto d2 = matmul2d_descriptor(TQ, D, KEYS, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d1, execution_simdgroups<8>> op1;
  matmul2d<d2, execution_simdgroups<8>> op2;
  auto a = q.slice<D, TQ>(0, r0);
  auto b0 = k.slice<D, KEYS>(0, 0);
  auto v0 = v.slice<D, KEYS>(0, 0);
  auto acc = op1.template get_destination_cooperative_tensor<decltype(a), decltype(b0), float>();
  auto out = op2.template get_destination_cooperative_tensor<decltype(p), decltype(v0), float>();
  short srow[C1], scol[C1], orow[C2], ocol[C2];
  H3_UNROLL
  for (ushort i = 0; i < C1; i++) {
    auto ids = acc.get_multidimensional_index(i);
    scol[i] = ids[0];
    srow[i] = ids[1];
  }
  H3_UNROLL
  for (ushort i = 0; i < C2; i++) {
    auto ids = out.get_multidimensional_index(i);
    ocol[i] = ids[0];
    orow[i] = ids[1];
    out[i] = 0.0f;
  }
  const float scale = SC[0];
  const int row = int(tid) / TPR, first = (int(tid) % TPR) * C1;
  const int mine = row * KEYS + first;
  float top = -INFINITY, total = 0.0f;
  float s[C1];
  for (int sel = 0; sel < KSEL; sel++) {
    const int tile = list[sel];
#ifdef H3_FAR
    // a pooled tile (past H3_TILES) is 64 whole keys, each standing for four: ln 4 on its logits
    const bool pooled = tile >= H3_TILES;
    const int k0 = tile * KEYS, size = pooled ? KEYS : SIZES[tile];
    const float lift = pooled ? 2.0f * M_LN2_F : 0.0f;
#else
    const int k0 = tile * KEYS, size = SIZES[tile];
#endif
    H3_UNROLL
    for (ushort i = 0; i < C1; i++) acc[i] = 0.0f;
    auto b = k.slice<D, KEYS>(0, k0);
    op1.run(a, b, acc);
#ifdef H3_FAR
    H3_UNROLL
    for (ushort i = 0; i < C1; i++) st[srow[i] * KEYS + scol[i]] = scol[i] < size ? acc[i] * scale + lift : -INFINITY;
#else
    H3_UNROLL
    for (ushort i = 0; i < C1; i++) st[srow[i] * KEYS + scol[i]] = scol[i] < size ? acc[i] * scale : -INFINITY;
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local = -INFINITY;
    H3_UNROLL
    for (ushort j = 0; j < C1; j++) {
      s[j] = st[mine + j];
      local = max(local, s[j]);
    }
    tops[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float next = top;
    H3_UNROLL
    for (ushort j = 0; j < TPR; j++) next = max(next, tops[row * TPR + j]);
    // a tile whose keys are all padding leaves next at -inf; its weights are zero and nothing shrinks
    const float shrink = next == -INFINITY ? 1.0f : exp(top - next);
    if (tid % TPR == 0) fac[row] = shrink;
    total *= shrink;
    top = next;
    H3_UNROLL
    for (ushort j = 0; j < C1; j++) {
      const float weight = next == -INFINITY ? 0.0f : exp(s[j] - next);
      total += weight;
      pt[mine + j] = bfloat(weight);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    H3_UNROLL
    for (ushort i = 0; i < C2; i++) out[i] *= fac[orow[i]];
    auto vb = v.slice<D, KEYS>(0, k0);
    op2.run(p, vb, out);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  tops[tid] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid % TPR == 0) {
    float sum = 0.0f;
    H3_UNROLL
    for (ushort j = 0; j < TPR; j++) sum += tops[tid + j];
    fac[row] = 1.0f / sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  H3_UNROLL
  for (ushort i = 0; i < C2; i++) {
    const int to = ROWOF[r0 + orow[i]];
    if (to >= 0) Y[(long(to) * H + head) * D + ocol[i]] = bfloat(out[i] * fac[orow[i]]);
  }
}

// --far-pool: the target's keys and values 2 x 2 mean-pooled (fp32 sums) into the tiles after H3_TILES, one
// pooled row of 64 a tile. FIRST: the first target tile; picture rows are 128 wide. [pooled rows, H] of 64.
[[kernel]] void h3_far_pool(
  device bfloat* K [[buffer(0)]],
  device bfloat* V [[buffer(1)]],
  const constant int32_t* FIRST [[buffer(2)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const long hb = long(tg.y) * H3_SLOTS;
  const long top = hb + (FIRST[0] + 4 * long(tg.x)) * 64 + 2 * tid;
  const long to = (hb + (H3_TILES + long(tg.x)) * 64 + tid) * 128;
  for (int d = 0; d < 128; d++) {
    const long at = top * 128 + d;
    K[to + d] = bfloat((float(K[at]) + float(K[at + 128]) + float(K[at + 128 * 128]) + float(K[at + 129 * 128])) * 0.25f);
    V[to + d] = bfloat((float(V[at]) + float(V[at + 128]) + float(V[at + 128 * 128]) + float(V[at + 129 * 128])) * 0.25f);
  }
}
