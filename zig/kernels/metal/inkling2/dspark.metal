#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

constant uint DSP_H = 2304;
constant uint DSP_Q = 3072;
constant uint DSP_KV = 1024;
constant uint DSP_HEAD = 128;
constant uint DSP_RANK = 256;
constant uint DSP_RNN = 768;
constant float DSP_EPS = 1e-6f;
constant float DSP_SCALE = 0x1.6a09e6p-4f;

struct DsparkPositionArgs { uint base; };
struct DsparkContextArgs { uint base, rows, ring; };
struct DsparkAttentionArgs { uint committed, ring, local, from; };
struct DsparkTokenArgs { uint token; };
struct DsparkResidualArgs { uint init; };
struct DsparkCountArgs { uint count; };
struct DsparkBaseArgs { uint rows, vocab; };
struct DsparkCohortArgs { uint streams; uint row; };
struct DsparkPackArgs { uint active[7]; uint first, count; };
constant constexpr uint DSP_TILE = 28;

// Threadgroup (row, stream): row zero copies the stream's normalized anchor, the other seven the raw mask embedding.
kernel void tf_ink2_dspark_init_rows(const device bfloat* anchors [[buffer(0)]], const device bfloat* mask [[buffer(1)]], device bfloat* blocks [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint2 t [[thread_position_in_threadgroup]]) {
  const uint row = tg.x;
  const device bfloat* anchor = anchors + size_t(tg.y) * DSP_H;
  device bfloat* rows = blocks + size_t(tg.y) * 8 * DSP_H;
  for (uint d = t.x; d < DSP_H; d += 256) {
    const device bfloat* src = row == 0 ? anchor : mask;
    rows[row * DSP_H + d] = src[d];
  }
}

kernel void tf_ink2_dspark_fuse_taps(const device bfloat* taps [[buffer(0)]], const device bfloat* fuse_weight [[buffer(1)]], const device bfloat* norm_weight [[buffer(2)]], device bfloat* c [[buffer(3)]], device bfloat* newest_f [[buffer(4)]], constant DsparkCountArgs& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  const uint fusion = group / a.count, row = group % a.count;
  threadgroup float alpha[4];
  threadgroup float scale;
  if (t == 0) {
    float scores[4];
    float peak = -INFINITY;
    for (uint j = 0; j < 4; ++j) { scores[j] = float(fuse_weight[fusion * 4 + j]); peak = max(peak, scores[j]); }
    float sum = 0.0f;
    for (uint j = 0; j < 4; ++j) { scores[j] = exp(scores[j] - peak); sum += scores[j]; }
    for (uint j = 0; j < 4; ++j) alpha[j] = float(bfloat(scores[j] / sum));
    float square = 0.0f;
    for (uint d = 0; d < DSP_H; ++d) {
      float mixed = 0.0f;
      for (uint j = 0; j < 4; ++j) mixed += alpha[j] * float(taps[(row * 4 + j) * DSP_H + d]);
      square += mixed * mixed;
    }
    scale = rsqrt(square / float(DSP_H) + DSP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < DSP_H; d += 256) {
    float mixed = 0.0f;
    for (uint j = 0; j < 4; ++j) mixed += alpha[j] * float(taps[(row * 4 + j) * DSP_H + d]);
    const bfloat out = bfloat(mixed * scale * float(norm_weight[fusion * DSP_H + d]));
    c[(fusion * a.count + row) * DSP_H + d] = out;
    if (fusion == 3 && row + 1 == a.count) newest_f[d] = out;
  }
}

static inline float dspark_rope_angle(uint d, uint position) {
  const uint pair = d % 64;
  const float frequency = pow(1e6f, -float(pair) / 64.0f);
  return float(position) * frequency;
}

// One block row at `position`: work 0..31 a query, then a key head; 32..39 a value head's unchanged 128 threads.
static inline void dspark_block_qkv(const device bfloat* q_raw, const device bfloat* k_raw, const device bfloat* v_raw, const device bfloat* q_weight, const device bfloat* k_weight, device bfloat* q, device bfloat* k, device bfloat* v, threadgroup float* scale, uint position, uint work, uint d) {
  if (work < 32 && d == 0) {
    float sum = 0.0f;
    if (work < 24) {
      const uint h = work;
      for (uint i = 0; i < DSP_HEAD; ++i) { const float x = float(q_raw[h * DSP_HEAD + i]); sum += x * x; }
    } else {
      const uint h = work - 24;
      for (uint i = 0; i < DSP_HEAD; ++i) { const float x = float(k_raw[h * DSP_HEAD + i]); sum += x * x; }
    }
    scale[0] = rsqrt(sum / float(DSP_HEAD) + DSP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (work >= 32) {
    const uint h = work - 32;
    v[h * DSP_HEAD + d] = v_raw[h * DSP_HEAD + d];
    return;
  }
  const bool query = work < 24;
  const uint h = query ? work : work - 24;
  const uint at = h * DSP_HEAD + d;
  const float x = float(bfloat(float(query ? q_raw[at] : k_raw[at]) * scale[0] * float(query ? q_weight[d] : k_weight[d])));
  const uint partner = d < 64 ? d + 64 : d - 64;
  const float other = float(bfloat(float(query ? q_raw[h * DSP_HEAD + partner] : k_raw[h * DSP_HEAD + partner]) * scale[0] * float(query ? q_weight[partner] : k_weight[partner])));
  const float angle = dspark_rope_angle(d, position);
  const float cs = cos(angle), sn = sin(angle);
  const float rotated = d < 64 ? x * cs - other * sn : other * sn + x * cs;
  if (query) q[at] = bfloat(rotated); else k[at] = bfloat(rotated);
}

kernel void tf_ink2_dspark_block_qkv(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]], const device bfloat* v_raw [[buffer(2)]], const device bfloat* q_weight [[buffer(3)]], const device bfloat* k_weight [[buffer(4)]], device bfloat* q [[buffer(5)]], device bfloat* k [[buffer(6)]], device bfloat* v [[buffer(7)]], constant DsparkPositionArgs& a [[buffer(8)]], uint group [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]]) {
  threadgroup float scale;
  const uint row = group / 40, work = group % 40;
  dspark_block_qkv(q_raw + size_t(row) * DSP_Q, k_raw + size_t(row) * DSP_KV, v_raw + size_t(row) * DSP_KV, q_weight, k_weight, q + size_t(row) * DSP_Q, k + size_t(row) * DSP_KV, v + size_t(row) * DSP_KV, &scale, a.base + row, work, d);
}

// The cohort's: threadgroup (work, row) over blocks of eight rows, each block at its own entry of `anchors`.
kernel void tf_ink2_dspark_cohort_qkv(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]], const device bfloat* v_raw [[buffer(2)]], const device bfloat* q_weight [[buffer(3)]], const device bfloat* k_weight [[buffer(4)]], device bfloat* q [[buffer(5)]], device bfloat* k [[buffer(6)]], device bfloat* v [[buffer(7)]], const device uint* anchors [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], uint2 d [[thread_position_in_threadgroup]]) {
  threadgroup float scale;
  const uint work = tg.x, row = tg.y;
  dspark_block_qkv(q_raw + size_t(row) * DSP_Q, k_raw + size_t(row) * DSP_KV, v_raw + size_t(row) * DSP_KV, q_weight, k_weight, q + size_t(row) * DSP_Q, k + size_t(row) * DSP_KV, v + size_t(row) * DSP_KV, &scale, anchors[row / 8] + row % 8, work, d.x);
}

kernel void tf_ink2_dspark_context_kv(const device bfloat* k_raw [[buffer(0)]], const device bfloat* v_raw [[buffer(1)]], const device bfloat* weight [[buffer(2)]], device bfloat* keys [[buffer(3)]], device bfloat* values [[buffer(4)]], constant DsparkContextArgs& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]]) {
  const uint row = group / 8, h = group % 8;
  threadgroup float scale;
  if (d == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < DSP_HEAD; ++i) { const float x = float(k_raw[row * DSP_KV + h * DSP_HEAD + i]); sum += x * x; }
    scale = rsqrt(sum / float(DSP_HEAD) + DSP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint position = a.base + row;
  const uint at = (position % a.ring) * DSP_KV + h * DSP_HEAD + d;
  const uint src = row * DSP_KV + h * DSP_HEAD + d;
  const uint partner = d < 64 ? d + 64 : d - 64;
  const float x = float(bfloat(float(k_raw[src]) * scale * float(weight[d])));
  const float other = float(bfloat(float(k_raw[row * DSP_KV + h * DSP_HEAD + partner]) * scale * float(weight[partner])));
  const float angle = dspark_rope_angle(d, position);
  const float cs = cos(angle), sn = sin(angle);
  keys[at] = bfloat(d < 64 ? x * cs - other * sn : other * sn + x * cs);
  values[at] = v_raw[src];
}

static inline float dspark_attention_score(const device bfloat* q, const device bfloat* k, uint q_at, uint k_at) {
  float dot = 0.0f;
  for (uint d = 0; d < DSP_HEAD; ++d) dot += float(q[q_at + d]) * float(k[k_at + d]);
  return dot * DSP_SCALE;
}

// A threadgroup a (row, head): thread t scores keys t, t + 128, ...; the denominator and value sums stay in key order.
kernel void tf_ink2_dspark_attention(const device bfloat* q [[buffer(0)]], const device bfloat* block_k [[buffer(1)]], const device bfloat* block_v [[buffer(2)]], const device bfloat* context_k [[buffer(3)]], const device bfloat* context_v [[buffer(4)]], device float* scores [[buffer(5)]], device bfloat* out [[buffer(6)]], constant DsparkAttentionArgs& a [[buffer(7)]], uint group [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]]) {
  const uint row = group / 24, h = group % 24, kv = h / 3;
  const uint q_at = row * DSP_Q + h * DSP_HEAD;
  const uint out_at = q_at + d;
  const uint first = max(a.from, a.local && a.committed > 1024 ? a.committed - 1024 : 0);
  const uint retained = a.committed - first;
  const uint count = retained + 8;
  const uint score_at = group * count;
  threadgroup float peaks[128], peak, denominator;
  float m = -INFINITY;
  for (uint j = d; j < count; j += 128) {
    const float s = j < retained
        ? dspark_attention_score(q, context_k, q_at, ((first + j) % a.ring) * DSP_KV + kv * DSP_HEAD)
        : dspark_attention_score(q, block_k, q_at, (j - retained) * DSP_KV + kv * DSP_HEAD);
    scores[score_at + j] = s;
    m = max(m, s);
  }
  peaks[d] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (d == 0) {
    float top = -INFINITY;
    for (uint t = 0; t < 128; ++t) top = max(top, peaks[t]);
    peak = top;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = d; j < count; j += 128) scores[score_at + j] = exp(scores[score_at + j] - peak);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (d == 0) {
    float sum = 0.0f;
    for (uint j = 0; j < count; ++j) sum += scores[score_at + j];
    denominator = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = d; j < count; j += 128) scores[score_at + j] = scores[score_at + j] / denominator;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  float value = 0.0f;
  for (uint j = 0; j < retained; ++j) {
    const uint position = first + j;
    const uint v_at = (position % a.ring) * DSP_KV + kv * DSP_HEAD + d;
    value += scores[score_at + j] * float(context_v[v_at]);
  }
  for (uint j = 0; j < 8; ++j) {
    const uint v_at = (j * DSP_KV + kv * DSP_HEAD) + d;
    value += scores[score_at + retained + j] * float(block_v[v_at]);
  }
  out[out_at] = bfloat(value);
}

kernel void tf_ink2_dspark_residual_norm(const device bfloat* x [[buffer(0)]], const device bfloat* residual [[buffer(1)]], const device bfloat* weight [[buffer(2)]], device bfloat* new_residual [[buffer(3)]], device bfloat* normalized [[buffer(4)]], constant DsparkResidualArgs& a [[buffer(5)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup float scale;
  if (t == 0) {
    float sum = 0.0f;
    for (uint d = 0; d < DSP_H; ++d) {
      const uint at = row * DSP_H + d;
      const float v = a.init ? float(x[at]) : float(x[at]) + float(residual[at]);
      sum += v * v;
    }
    scale = rsqrt(sum / float(DSP_H) + DSP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint n = 0; n < 9; ++n) {
    const uint d = n * 256 + t;
    const uint at = row * DSP_H + d;
    const float sigma = a.init ? float(x[at]) : float(x[at]) + float(residual[at]);
    new_residual[at] = bfloat(sigma);
    normalized[at] = bfloat(sigma * scale * float(weight[d]));
  }
}

kernel void tf_ink2_dspark_swiglu(const device bfloat* g [[buffer(0)]], const device bfloat* u [[buffer(1)]], device bfloat* out [[buffer(2)]], constant DsparkCountArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const float gate = float(g[i]);
  out[i] = bfloat((gate / (1.0f + exp(-gate))) * float(u[i]));
}

kernel void tf_ink2_dspark_kappa(const device bfloat* y [[buffer(0)]], const device bfloat* f [[buffer(1)]], const device bfloat* weight [[buffer(2)]], const device bfloat* bias [[buffer(3)]], device float* out [[buffer(4)]], constant DsparkCountArgs& a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  const uint row = i / DSP_RNN, channel = i % DSP_RNN;
  if (row >= a.count) return;
  float sum = 0.0f;
  for (uint d = 0; d < DSP_H; ++d) sum += float(y[row * DSP_H + d]) * float(weight[channel * 5120 + 512 + d]);
  for (uint d = 0; d < DSP_H; ++d) sum += float(f[d]) * float(weight[channel * 5120 + 2816 + d]);
  sum += float(bias[channel]);
  out[i] = sum;
}

static inline void dspark_recurrent_step(const device bfloat* state, const device float* kappa, const device bfloat* rnn_weight, const device bfloat* markov, device bfloat* z, device bfloat* next_state, device bfloat* lambda, uint token, uint i) {
  if (i >= DSP_RANK) return;
  float values[3];
  for (uint part = 0; part < 3; ++part) {
    const uint channel = part * DSP_RANK + i;
    float sum = 0.0f;
    for (uint d = 0; d < DSP_RANK; ++d) sum += float(state[d]) * float(rnn_weight[channel * 5120 + d]);
    for (uint d = 0; d < DSP_RANK; ++d) sum += float(markov[uint64_t(token) * DSP_RANK + d]) * float(rnn_weight[channel * 5120 + 256 + d]);
    sum += float(kappa[channel]);
    const bfloat rounded = bfloat(sum);
    z[channel] = rounded;
    values[part] = float(rounded);
  }
  const bfloat gamma = bfloat(1.0f / (1.0f + exp(-values[0])));
  const bfloat candidate = bfloat(tanh(values[1]));
  const bfloat keep = bfloat(1.0f - float(gamma));
  const bfloat left = bfloat(float(gamma) * float(state[i]));
  const bfloat right = bfloat(float(keep) * float(candidate));
  next_state[i] = bfloat(float(left) + float(right));
  lambda[i] = bfloat(tanh(values[2]));
}

kernel void tf_ink2_dspark_recurrent(const device bfloat* state [[buffer(0)]], const device float* kappa [[buffer(1)]], const device bfloat* rnn_weight [[buffer(2)]], const device bfloat* markov [[buffer(3)]], device bfloat* z [[buffer(4)]], device bfloat* next_state [[buffer(5)]], device bfloat* lambda [[buffer(6)]], constant DsparkTokenArgs& a [[buffer(7)]], uint i [[thread_position_in_grid]]) {
  dspark_recurrent_step(state, kappa, rnn_weight, markov, z, next_state, lambda, a.token, i);
}

// The recurrent step after held[a.token]: the chain's previous token read on the GPU.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_chain_step(const device bfloat* state [[buffer(0)]], const device float* kappa [[buffer(1)]], const device bfloat* rnn_weight [[buffer(2)]], const device bfloat* markov [[buffer(3)]], device bfloat* z [[buffer(4)]], device bfloat* next_state [[buffer(5)]], device bfloat* lambda [[buffer(6)]], constant DsparkTokenArgs& a [[buffer(7)]], const device uint* held [[buffer(8)]], uint i [[thread_position_in_grid]]) {
  dspark_recurrent_step(state, kappa, rnn_weight, markov, z, next_state, lambda, held[a.token], i);
}

// A depth's recurrent step, a threadgroup a stream: stream s takes kappa row 7s + a.row after the token held[8s + a.row].
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_cohort_step(const device bfloat* state [[buffer(0)]], const device float* kappa [[buffer(1)]], const device bfloat* rnn_weight [[buffer(2)]], const device bfloat* markov [[buffer(3)]], device bfloat* z [[buffer(4)]], device bfloat* next_state [[buffer(5)]], device bfloat* lambda [[buffer(6)]], constant DsparkCohortArgs& a [[buffer(7)]], const device uint* held [[buffer(8)]], uint s [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]]) {
  dspark_recurrent_step(state + s * DSP_RANK, kappa + (s * 7 + a.row) * DSP_RNN, rnn_weight, markov, z + s * 3 * DSP_RANK, next_state + s * DSP_RANK, lambda + s * DSP_RANK, held[s * 8 + a.row], i);
}

// The first greatest of one row of 200,058 logits for a 1,024-thread group: the lower id of equals, a NaN never.
static inline uint dspark_first_greatest(const device bfloat* logits, uint t, uint lane, uint sg, threadgroup float* values, threadgroup uint* places) {
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < 200058u; i += 1024u) {
    const float raw = float(logits[i]);
    const float v = isnan(raw) ? -INFINITY : raw;
    if (v > best || (v == best && i < at)) { best = v; at = i; }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(best, o);
    const uint oi = simd_shuffle_xor(at, o);
    if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
  }
  if (lane == 0) { values[sg] = best; places[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return 0xffffffffu;
  for (uint g = 0; g < 32; ++g)
    if (values[g] > best || (values[g] == best && places[g] < at)) { best = values[g]; at = places[g]; }
  return at;
}

// held[a.token + 1] = the first greatest of the 200,058 logits; a NaN is never chosen unless logit 0 is one.
[[max_total_threads_per_threadgroup(1024)]]
kernel void tf_ink2_dspark_chain_pick(const device bfloat* logits [[buffer(0)]], device uint* held [[buffer(1)]], constant DsparkTokenArgs& a [[buffer(2)]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float values[32];
  threadgroup uint places[32];
  const uint at = dspark_first_greatest(logits, t, lane, sg, values, places);
  if (t == 0) held[a.token + 1] = isnan(float(logits[0])) ? 0u : at;
}

// A depth's picks, a threadgroup a stream: held[8s + a.row + 1] is the first greatest of logits row s.
[[max_total_threads_per_threadgroup(1024)]]
kernel void tf_ink2_dspark_cohort_pick(const device bfloat* logits [[buffer(0)]], device uint* held [[buffer(1)]], constant DsparkCohortArgs& a [[buffer(2)]], uint s [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float values[32];
  threadgroup uint places[32];
  const device bfloat* row = logits + size_t(s) * 200058u;
  const uint at = dspark_first_greatest(row, t, lane, sg, values, places);
  if (t == 0) held[s * 8 + a.row + 1] = isnan(float(row[0])) ? 0u : at;
}

// A simdgroup an id: eight columns a lane a row; the rows share the read of unembed[id] and round once to BF16.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_base(const device bfloat* unembed [[buffer(0)]], const device bfloat* y [[buffer(1)]], device bfloat* out [[buffer(2)]], constant DsparkBaseArgs& a [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint id = tg * 8 + sg;
  if (id >= a.vocab) return;
  const device bfloat* w = unembed + size_t(id) * DSP_H;
  float acc[DSP_TILE];
  #pragma unroll
  for (uint row = 0; row < DSP_TILE; ++row) acc[row] = 0.0f;
  for (uint i = 8 * lane; i < DSP_H; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    #pragma unroll
    for (uint row = 0; row < DSP_TILE; ++row) {
      if (row < a.rows) {
        const device bfloat* x = y + size_t(row) * DSP_H + i;
        acc[row] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));
      }
    }
  }
  #pragma unroll
  for (uint row = 0; row < DSP_TILE; ++row) {
    const float sum = simd_sum(acc[row]);
    if (row < a.rows && lane == 0) out[size_t(row) * a.vocab + id] = bfloat(sum);
  }
}

kernel void tf_ink2_dspark_logits(const device bfloat* base [[buffer(0)]], const device bfloat* phi [[buffer(1)]], const device bfloat* beta [[buffer(2)]], const device bfloat* lambda [[buffer(3)]], device bfloat* out [[buffer(4)]], constant DsparkCountArgs& a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  float dot = 0.0f;
  for (uint d = 0; d < DSP_RANK; ++d) dot += float(phi[i * DSP_RANK + d]) * float(lambda[d]);
  const bfloat bias = bfloat(float(beta[i]) + dot);
  const bfloat logit = bfloat(float(base[i]) + float(bias));
  out[i] = bfloat(30.0f * tanh(float(logit) / 30.0f));
}

// A vocabulary id's logits for `streams` rows: phi's row is read once, each row keeping its own serial 256-sum.
template <uint N>
inline void dspark_cohort_logits(const device bfloat* base, const device bfloat* phi, const device bfloat* beta, const device bfloat* lambda, device bfloat* out, uint streams, uint i) {
  float dot[N];
  #pragma unroll
  for (uint s = 0; s < N; ++s) dot[s] = 0.0f;
  for (uint d = 0; d < DSP_RANK; ++d) {
    const float p = float(phi[size_t(i) * DSP_RANK + d]);
    #pragma unroll
    for (uint s = 0; s < N; ++s) {
      if (s < streams) dot[s] += p * float(lambda[size_t(s) * DSP_RANK + d]);
    }
  }
  #pragma unroll
  for (uint s = 0; s < N; ++s) {
    if (s < streams) {
      const bfloat bias = bfloat(float(beta[i]) + dot[s]);
      const bfloat logit = bfloat(float(base[size_t(s) * 200058u + i]) + float(bias));
      out[size_t(s) * 200058u + i] = bfloat(30.0f * tanh(float(logit) / 30.0f));
    }
  }
}

// A depth's final logits for streams 0 .. a.streams: base rows a.row .., lambda row s, out row s.
kernel void tf_ink2_dspark_cohort_logits(const device bfloat* base [[buffer(0)]], const device bfloat* phi [[buffer(1)]], const device bfloat* beta [[buffer(2)]], const device bfloat* lambda [[buffer(3)]], device bfloat* out [[buffer(4)]], constant DsparkCohortArgs& a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i >= 200058u) return;
  const device bfloat* rows = base + size_t(a.row) * 200058u;
  if (a.streams <= 4) dspark_cohort_logits<4>(rows, phi, beta, lambda, out, a.streams, i);
  else if (a.streams <= 8) dspark_cohort_logits<8>(rows, phi, beta, lambda, out, a.streams, i);
  else dspark_cohort_logits<16>(rows, phi, beta, lambda, out, a.streams, i);
}

// The packed unembed rows a.first .. a.first + a.count: draft row tg.y + 1 of the first active[tg.y] streams, a row a stream.
kernel void tf_ink2_dspark_cohort_pack(const device bfloat* x [[buffer(0)]], device bfloat* out [[buffer(1)]], constant DsparkPackArgs& a [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint2 t [[thread_position_in_threadgroup]]) {
  if (tg.x >= a.active[tg.y]) return;
  uint row = tg.x;
  for (uint j = 0; j < tg.y; ++j) row += a.active[j];
  if (row < a.first || row - a.first >= a.count) return;
  const device bfloat4* src = (const device bfloat4*)(x + size_t(tg.x * 8 + tg.y + 1) * DSP_H);
  device bfloat4* dst = (device bfloat4*)(out + size_t(row - a.first) * DSP_H);
  for (uint d = t.x; d < DSP_H / 4; d += 256) dst[d] = src[d];
}
