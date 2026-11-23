//! Inkling 2 Mini's DSpark draft head in float64 on the host, read straight from the bf16 checkpoint: the reference
//! the CUDA head's logits are checked against (teacher-forced by the device's tokens). No intermediate rounding, and
//! unfolded: hidden_out is applied before the unembedding, where the engine folds the two once at load.
const std = @import("std");
const core = @import("core");

const Allocator = std.mem.Allocator;
const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;
const threads = core.contract.reference_threads;

const hidden = 2304;
const q_heads = 24;
const kv_heads = 8;
const head = 128;
const q_width = q_heads * head;
const kv_width = kv_heads * head;
const rank = 256;
const inter = 9216;
const block = 8;
const taps_n = 4;
const window = 1024;
const eps_d = 1e-6;
/// Block rows 1..7 each draft one token.
pub const max_depth = block - 1;

fn value(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

fn bf(bytes: []const u8, i: usize) f64 {
    return value(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
}

/// `t` as bf16 of `dims`; a dim of 0 takes any size (the vocabulary's padded rows).
fn shaped(t: Tensor, dims: []const usize) !Tensor {
    if (t.dtype != .bf16 or t.rank != dims.len) return error.DSparkTensorShape;
    for (dims, t.shape[0..t.rank]) |want, got| if (want != 0 and got != want) return error.DSparkTensorShape;
    if (t.bytes.len != 2 * t.numel()) return error.DSparkTensorShape;
    return t;
}

const Rows = struct {
    w: []const u8,
    x: []const f64,
    y: []f64,
    first: usize,
    last: usize,

    pub fn run(j: *Rows) void {
        const k = j.x.len;
        for (j.first..j.last) |row| {
            var s: f64 = 0;
            for (j.x, 0..) |v, i| s += bf(j.w, row * k + i) * v;
            j.y[row] = s;
        }
    }
};

/// y = W x over W's first y.len rows (bf16, row-major [out][x.len]), the rows spread over threads.
fn linear(w: Tensor, x: []const f64, y: []f64) void {
    var jobs: [threads.maximum]Rows = undefined;
    const n = threads.count(threads.maximum, y.len);
    for (jobs[0..n], 0..) |*j, i| j.* = .{ .w = w.bytes, .x = x, .y = y, .first = y.len * i / n, .last = y.len * (i + 1) / n };
    threads.run(Rows, jobs[0..n], .normal);
}

/// x * w / sqrt(mean(x^2) + eps), in place.
fn rms(x: []f64, w: Tensor, eps: f64) void {
    var s: f64 = 0;
    for (x) |v| s += v * v;
    const inv = 1.0 / @sqrt(s / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, 0..) |*v, i| v.* *= inv * bf(w.bytes, i);
}

fn softmax(x: []f64) void {
    var peak = -std.math.inf(f64);
    for (x) |v| peak = @max(peak, v);
    var sum: f64 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - peak);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

/// Rotate-half RoPE of one head at `pos`: element d turns with d + 64, both by pos * 1e6^(-d / 64).
fn rope(x: *[head]f64, pos: u64) void {
    const half = head / 2;
    for (0..half) |d| {
        const a = @as(f64, @floatFromInt(pos)) * std.math.pow(f64, 1e6, -@as(f64, @floatFromInt(d)) / half);
        const lo = x[d];
        const hi = x[d + half];
        x[d] = lo * @cos(a) - hi * @sin(a);
        x[d + half] = lo * @sin(a) + hi * @cos(a);
    }
}

/// Each head of x normed by `w`, then rotated at `pos`.
fn normRope(x: []f64, w: Tensor, pos: u64) void {
    var at: usize = 0;
    while (at < x.len) : (at += head) {
        const h = x[at..][0..head];
        rms(h, w, eps_d);
        rope(h, pos);
    }
}

fn normed(resid: []const []f64, w: Tensor, x: []const []f64) void {
    for (resid, x) |src, dst| {
        @memcpy(dst, src);
        rms(dst, w, eps_d);
    }
}

fn mat(a: Allocator, n: usize, w: usize) ![][]f64 {
    const x = try a.alloc([]f64, n);
    for (x) |*row| row.* = try a.alloc(f64, w);
    return x;
}

/// One row's attention over every row of k and v, unmasked; q head h reads kv head h / 3. `w` holds a head's weights.
fn attend(q: []const f64, k: []const []f64, v: []const []f64, w: []f64, out: []f64) void {
    const scale = 1.0 / @sqrt(@as(f64, head));
    for (0..q_heads) |h| {
        const kh = h / (q_heads / kv_heads) * head;
        const qh = q[h * head ..][0..head];
        for (k, w) |kr, *s| {
            var dot: f64 = 0;
            for (qh, kr[kh..][0..head]) |x, y| dot += x * y;
            s.* = dot * scale;
        }
        softmax(w);
        const o = out[h * head ..][0..head];
        @memset(o, 0);
        for (v, w) |vr, p| for (o, vr[kh..][0..head]) |*x, y| {
            x.* += p * y;
        };
    }
}

pub const Ref = struct {
    gpa: Allocator,
    ck: *Checkpoint, // the target model with mtp.safetensors added
    eps: f64, // the target's rms eps, for the pending token's embed norm
    vocab: usize, // the target's unpadded vocabulary
    cap: f64, // the target's logit cap
    /// planted control: attention drops the context rows; the GPU check must go red
    plant: bool = false,

    fn get(r: *const Ref, comptime fmt: []const u8, args: anytype, dims: []const usize) !Tensor {
        var buf: [96]u8 = undefined;
        return shaped(try r.ck.get(try std.fmt.bufPrint(&buf, fmt, args)), dims);
    }

    /// taps: rows * 4 * hidden bf16 ([row][tap][hidden]), the committed context rows at positions start.. (their
    /// count from its length); pending: the token at the anchor, start + rows (block row 0). tokens[s] is the
    /// device's pick at step s, so step s + 1 is forced by it and step 0 by pending. out: tokens.len * vocab
    /// soft-capped logits, step-major.
    pub fn draft(r: *const Ref, taps: []const u16, start: u64, pending: u32, tokens: []const u32, out: []f64) !void {
        const rows = taps.len / (taps_n * hidden);
        if (rows == 0 or taps.len != rows * taps_n * hidden or tokens.len == 0 or tokens.len > max_depth or
            out.len != tokens.len * r.vocab) return error.DSparkRefArgs;
        var arena = std.heap.ArenaAllocator.init(r.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const c = try r.fuse(a, taps, rows);
        const x = try r.blockRows(a, &c, start, pending);
        try r.markov(a, x, c[3][rows - 1], pending, tokens, out);
    }

    /// c[f][row]: the row's taps mixed by softmax(fuse_weight[f]), normed by fusion f's dflare (the three layers',
    /// then the Markov head's).
    fn fuse(r: *const Ref, a: Allocator, taps: []const u16, rows: usize) ![4][][]f64 {
        const fw = try r.get("fuse_weight", .{}, &.{ 4, taps_n });
        var c: [4][][]f64 = undefined;
        for (&c, 0..) |*cf, f| {
            const dflare = if (f < 3) try r.get("layers.{d}.dflare_norm.weight", .{f}, &.{hidden}) else try r.get("markov_head.dflare_norm.weight", .{}, &.{hidden});
            var alpha: [taps_n]f64 = undefined;
            for (&alpha, 0..) |*x, j| x.* = bf(fw.bytes, f * taps_n + j);
            softmax(&alpha);
            cf.* = try mat(a, rows, hidden);
            for (cf.*, 0..) |m, row| {
                @memset(m, 0);
                for (alpha, 0..) |al, j| for (m, taps[(row * taps_n + j) * hidden ..][0..hidden]) |*o, t| {
                    o.* += al * value(t);
                };
                rms(m, dflare, eps_d);
            }
        }
        return c;
    }

    /// The 8 block rows after the three layers and the final norm; rows 1..7 are the draft steps' outputs.
    fn blockRows(r: *const Ref, a: Allocator, c: *const [4][][]f64, start: u64, pending: u32) ![][]f64 {
        const rows = c[0].len;
        const anchor = start + rows;
        const embed = try r.get("model.llm.embed.weight", .{}, &.{ 0, hidden });
        if (pending >= embed.dim(0)) return error.DSparkRefArgs;
        const mask = try r.get("mask_embed", .{}, &.{hidden});
        const resid = try mat(a, block, hidden);
        const x = try mat(a, block, hidden);
        for (resid[0], 0..) |*v, d| v.* = bf(embed.bytes, @as(usize, pending) * hidden + d);
        rms(resid[0], try r.get("model.llm.embed_norm.weight", .{}, &.{hidden}), r.eps);
        for (resid[1..]) |row| for (row, 0..) |*v, d| {
            v.* = bf(mask.bytes, d);
        };
        normed(resid, try r.get("layers.0.input_layernorm.weight", .{}, &.{hidden}), x);
        const q = try mat(a, block, q_width);
        const att = try a.alloc(f64, q_width);
        const y = try a.alloc(f64, hidden);
        const gate = try a.alloc(f64, inter);
        const up = try a.alloc(f64, inter);
        for (0..3) |i| {
            const L = .{i};
            // layers 0 and 2 are local: the last `window` context positions
            const first = if (r.plant) rows else if (i != 1) rows -| window else 0;
            const kept = rows - first;
            const k = try mat(a, kept + block, kv_width);
            const v = try mat(a, kept + block, kv_width);
            const k_ctx = try r.get("layers.{d}.self_attn.k_proj_ctx.weight", L, &.{ kv_width, hidden });
            const v_ctx = try r.get("layers.{d}.self_attn.v_proj_ctx.weight", L, &.{ kv_width, hidden });
            const k_norm_ctx = try r.get("layers.{d}.self_attn.k_norm_ctx.weight", L, &.{head});
            for (c[i][first..], k[0..kept], v[0..kept], first..) |ci, kr, vr, row| {
                linear(k_ctx, ci, kr);
                normRope(kr, k_norm_ctx, start + row);
                linear(v_ctx, ci, vr);
            }
            const q_proj = try r.get("layers.{d}.self_attn.q_proj.weight", L, &.{ q_width, hidden });
            const k_proj = try r.get("layers.{d}.self_attn.k_proj.weight", L, &.{ kv_width, hidden });
            const v_proj = try r.get("layers.{d}.self_attn.v_proj.weight", L, &.{ kv_width, hidden });
            const q_norm = try r.get("layers.{d}.self_attn.q_norm.weight", L, &.{head});
            const k_norm = try r.get("layers.{d}.self_attn.k_norm.weight", L, &.{head});
            for (x, q, k[kept..], v[kept..], 0..) |xr, qr, kr, vr, row| {
                linear(q_proj, xr, qr);
                normRope(qr, q_norm, anchor + row);
                linear(k_proj, xr, kr);
                normRope(kr, k_norm, anchor + row);
                linear(v_proj, xr, vr);
            }
            const o_proj = try r.get("layers.{d}.self_attn.o_proj.weight", L, &.{ hidden, q_width });
            const scores = try a.alloc(f64, kept + block);
            for (q, resid) |qr, res| {
                attend(qr, k, v, scores, att);
                linear(o_proj, att, y);
                for (res, y) |*o, d| o.* += d;
            }
            normed(resid, try r.get("layers.{d}.post_attention_layernorm.weight", L, &.{hidden}), x);
            const gate_proj = try r.get("layers.{d}.mlp.gate_proj.weight", L, &.{ inter, hidden });
            const up_proj = try r.get("layers.{d}.mlp.up_proj.weight", L, &.{ inter, hidden });
            const down_proj = try r.get("layers.{d}.mlp.down_proj.weight", L, &.{ hidden, inter });
            for (x, resid) |xr, res| {
                linear(gate_proj, xr, gate);
                linear(up_proj, xr, up);
                for (gate, up) |*g, u| g.* = g.* / (1.0 + @exp(-g.*)) * u;
                linear(down_proj, gate, y);
                for (res, y) |*o, d| o.* += d;
            }
            const next = if (i < 2) try r.get("layers.{d}.input_layernorm.weight", .{i + 1}, &.{hidden}) else try r.get("norm.weight", .{}, &.{hidden});
            normed(resid, next, x);
        }
        return x;
    }

    /// The Markov head over the block's rows: step s reads row s + 1, the recurrent state and the previous token.
    fn markov(r: *const Ref, a: Allocator, x: []const []f64, newest: []const f64, pending: u32, tokens: []const u32, out: []f64) !void {
        const rnn = try r.get("markov_head.rnn_proj.weight", .{}, &.{ 3 * rank, 2 * rank + 2 * hidden });
        const bias = try r.get("markov_head.rnn_proj.bias", .{}, &.{3 * rank});
        const w1 = try r.get("markov_head.markov_w1.weight", .{}, &.{ 0, rank });
        const hidden_out = try r.get("markov_head.hidden_out.weight", .{}, &.{ hidden, rank });
        const hidden_bias = try r.get("markov_head.hidden_out.bias", .{}, &.{hidden});
        const unembed = try r.get("model.llm.unembed.weight", .{}, &.{ 0, hidden });
        if (unembed.dim(0) < r.vocab) return error.DSparkTensorShape;
        // rnn_proj's columns: the state, markov_w1[previous], the step's row, the newest fused context row
        const in = try a.alloc(f64, 2 * rank + 2 * hidden);
        const z = try a.alloc(f64, 3 * rank);
        const h = try a.alloc(f64, hidden);
        const state = in[0..rank];
        @memset(state, 0);
        @memcpy(in[2 * rank + hidden ..], newest);
        var prev = pending;
        for (tokens, 0..) |token, s| {
            if (prev >= w1.dim(0)) return error.DSparkRefArgs;
            const y = x[s + 1];
            for (in[rank..][0..rank], 0..) |*v, d| v.* = bf(w1.bytes, @as(usize, prev) * rank + d);
            @memcpy(in[2 * rank ..][0..hidden], y);
            linear(rnn, in, z);
            for (z, 0..) |*v, c| v.* += bf(bias.bytes, c);
            const lambda = z[2 * rank ..];
            for (state, z[0..rank], z[rank .. 2 * rank], lambda) |*st, g, cand, *l| {
                const gamma = 1.0 / (1.0 + @exp(-g));
                st.* = gamma * st.* + (1.0 - gamma) * std.math.tanh(cand);
                l.* = std.math.tanh(l.*);
            }
            linear(hidden_out, lambda, h);
            for (h, y, 0..) |*v, yv, d| v.* += yv + bf(hidden_bias.bytes, d);
            const logits = out[s * r.vocab ..][0..r.vocab];
            linear(unembed, h, logits);
            for (logits) |*l| l.* = r.cap * std.math.tanh(l.* / r.cap);
            prev = token;
        }
    }
};

test "RoPE turns element d with d + 64: the low half takes -sin of its partner, the high half +sin" {
    var x: [head]f64 = @splat(0);
    x[0] = 1; // pair 0 turns by pos
    x[64 + 32] = 1; // pair 32 by pos * 1e6^(-1/2)
    rope(&x, 500);
    try std.testing.expectApproxEqAbs(@cos(@as(f64, 500)), x[0], 1e-15);
    try std.testing.expectApproxEqAbs(@sin(@as(f64, 500)), x[64], 1e-15);
    try std.testing.expectApproxEqAbs(-@sin(@as(f64, 0.5)), x[32], 1e-12);
    try std.testing.expectApproxEqAbs(@cos(@as(f64, 0.5)), x[96], 1e-12);
}

test "a misshapen tensor and a draft the block cannot hold are refused by name" {
    const bytes: [12]u8 = @splat(0);
    var t: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 2, 3, 1, 1, 1 }, .bytes = &bytes };
    _ = try shaped(t, &.{ 2, 3 });
    _ = try shaped(t, &.{ 0, 3 });
    try std.testing.expectError(error.DSparkTensorShape, shaped(t, &.{ 3, 2 }));
    try std.testing.expectError(error.DSparkTensorShape, shaped(t, &.{6}));
    t.dtype = .f16;
    try std.testing.expectError(error.DSparkTensorShape, shaped(t, &.{ 2, 3 }));

    // refused before any tensor is read
    const r: Ref = .{ .gpa = std.testing.allocator, .ck = undefined, .eps = 1e-6, .vocab = 2, .cap = 30 };
    const taps: [taps_n * hidden]u16 = @splat(0);
    var out: [16]f64 = undefined;
    const tokens: [8]u32 = @splat(0);
    try std.testing.expectError(error.DSparkRefArgs, r.draft(taps[1..], 0, 0, tokens[0..1], out[0..2]));
    try std.testing.expectError(error.DSparkRefArgs, r.draft(&taps, 0, 0, tokens[0..0], out[0..0]));
    try std.testing.expectError(error.DSparkRefArgs, r.draft(&taps, 0, 0, &tokens, &out));
    try std.testing.expectError(error.DSparkRefArgs, r.draft(&taps, 0, 0, tokens[0..2], out[0..3]));
}
