//! CUDA DSpark numerical contracts against independent host arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const dspark = @import("inkling2").dspark;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const eps: f32 = 1e-6;
const soft_why = "online softmax over the ring and block keys: no input makes its sums exact in fp32; gate 3's derived bound holds it";

/// The block's Q and K heads normed and RoPE'd at anchor + row in place: 24 Q heads + 8 K heads over 8 rows.
pub const Qk = struct {
    kern: *const dspark.Kernels,
    /// Control: RoPE's partner read one step early (j vs (j+2)%4 stays), the angle doubled.
    twice: bool = false,
    qk_in: []u16 = &.{},
    norms: []u16 = &.{},
    ptr: [4]u64 = @splat(0),

    const q_width = 3072;
    const kv_width = 1024;
    const hd = 128;
    pub fn spec(self: *const Qk) c.Spec {
        const served = &[_]c.Shape{.{ .rows = 8, .n = q_width + kv_width, .k = hd }};
        const name = if (self.twice) "control: inkling2 dspark tf_ink2_dsp_qk rope at twice the angle" else "inkling2 dspark tf_ink2_dsp_qk (32 heads normed + rope in place)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .in_place = true, .served = served };
    }
    pub fn control(self: *const Qk) ?g.Gate {
        return if (self.twice) .shapes else null;
    }
    pub fn free(self: *Qk, a: std.mem.Allocator) void {
        a.free(self.qk_in);
        a.free(self.norms);
        self.qk_in = &.{};
        self.norms = &.{};
    }
    /// 8 rows of q [8][3072] and k [8][1024] bf16; RoPE position anchor + row.
    pub fn prepare(self: *Qk, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows != 8 or shape.n != q_width + kv_width or shape.k != hd) return false;
        self.free(a);
        self.qk_in = try ref.bfs(a, rng, 8 * (q_width + kv_width), -1, 1);
        self.norms = try ref.bfs(a, rng, 2 * hd, 0.5, 1.5);
        if (fill == .nonfinite) {
            self.qk_in[0] = c.toBf16(ref.nan);
            self.qk_in[hd] = c.toBf16(ref.inf);
            self.qk_in[8 * q_width] = c.toBf16(-ref.inf);
        }
        self.ptr = .{ try rig.output(8 * q_width * 2), try rig.output(8 * kv_width * 2), try ref.input(rig, self.norms[0..hd]), try ref.input(rig, self.norms[hd..]) };
        return true;
    }
    pub fn run(self: *Qk, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.ptr[0], std.mem.sliceAsBytes(self.qk_in[0 .. 8 * q_width]));
        try ref.restore(rig, self.ptr[1], std.mem.sliceAsBytes(self.qk_in[8 * q_width ..]));
        try self.kern.launchQk(rig.s, self.ptr[0], self.ptr[1], self.ptr[2], self.ptr[3], 41);
    }
    pub fn count(_: *const Qk) usize {
        return 8 * (q_width + kv_width);
    }
    /// q rows' first 24 heads normed+roped, k's 8 heads normed+roped, the rest of q's row beyond head 24 untouched.
    pub fn check(self: *const Qk, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        _ = a;
        for (0..8) |row| {
            for (0..32) |h| {
                const is_q = h < 24;
                const at = if (is_q) row * q_width + h * hd else 8 * q_width + row * kv_width + (h - 24) * hd;
                const w = if (is_q) self.norms[0..hd] else self.norms[hd..];
                var ss: f64 = 0;
                for (0..hd) |d| {
                    const x = ref.bf(self.qk_in[at + d]);
                    ss += x * x;
                }
                const scale = 1 / @sqrt(ss / hd + eps);
                const pos = 41 + row;
                for (0..hd) |d| {
                    const j = d / 32;
                    const lane = d % 32;
                    const dim = lane + 32 * j;
                    const od = lane + 32 * ((j + 2) % 4);
                    const angle = @as(f64, @floatFromInt(pos)) * (if (self.twice) @as(f64, 2) else 1) * std.math.pow(f64, 1e6, -@as(f64, @floatFromInt(dim % 64)) / 64);
                    const x0 = ref.bf(self.qk_in[at + dim]) * scale * ref.bf(w[dim]);
                    const other = ref.bf(self.qk_in[at + od]) * scale * ref.bf(w[od]);
                    const want = if (dim < 64) x0 * @cos(angle) - other * @sin(angle) else other * @sin(angle) + x0 * @cos(angle);
                    const byte_at = if (is_q) 2 * (row * q_width + h * hd + dim) else 8 * q_width * 2 + 2 * (row * kv_width + (h - 24) * hd + dim);
                    const gotv = ref.bf(@bitCast(@as(u16, std.mem.readInt(u16, got[byte_at..][0..2], .little))));
                    if (!(std.math.isNan(want) and std.math.isNan(gotv)) and @abs(gotv - want) > 0.02 * @abs(want) + 0.01)
                        return try std.fmt.bufPrint(msg, "row {d} head {d} dim {d}: {e} where {e} is due", .{ row, h, dim, gotv, want });
                }
            }
        }
        return null;
    }
    pub fn solve(_: *const Qk, _: std.mem.Allocator, _: ref.Sums, _: []Iv) !void {}
};

/// Attention of each (row, q head) over the context ring plus the block's eight K/V rows, online softmax.
pub const Attn = struct {
    kern: *const dspark.Kernels,
    /// Control: the softmax denominator is the count, not the sum of weights.
    mean: bool = false,
    q: []u16 = &.{},
    bk: []u16 = &.{},
    bv: []u16 = &.{},
    ring: []u16 = &.{},
    ring_v: []u16 = &.{},
    ptr: [6]u64 = @splat(0),
    committed: u32 = 0,
    from: u32 = 0,
    shape: c.Shape = undefined,

    const q_width = 3072;
    const kv_width = 1024;
    const hd = 128;
    const ring_rows = 2048;
    pub fn spec(self: *const Attn) c.Spec {
        const served = &[_]c.Shape{.{ .rows = 8, .n = q_width, .k = 1050 }};
        const name = if (self.mean) "control: inkling2 dspark tf_ink2_dsp_attn output over key count" else "inkling2 dspark tf_ink2_dsp_attn (ring + block kv, online softmax)";
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 5, .out = .bf16, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why } };
    }
    fn depth(s: c.Shape) usize {
        return s.k + hd + 2;
    }
    pub const sums = {};
    pub fn control(self: *const Attn) ?g.Gate {
        return if (self.mean) .shapes else null;
    }
    pub fn free(self: *Attn, a: std.mem.Allocator) void {
        a.free(self.q);
        a.free(self.bk);
        a.free(self.bv);
        a.free(self.ring);
        a.free(self.ring_v);
        self.q = &.{};
        self.bk = &.{};
        self.bv = &.{};
        self.ring = &.{};
        self.ring_v = &.{};
    }
    /// A wrapped local ring contributes 1050 committed rows and eight block rows.
    pub fn prepare(self: *Attn, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.rows > 8 or shape.n != q_width or shape.k != 1050) return false;
        self.free(a);
        self.shape = shape;
        self.q = try ref.bfs(a, rng, 8 * q_width, -1, 1);
        self.bk = try ref.bfs(a, rng, 8 * kv_width, -1, 1);
        self.bv = try ref.bfs(a, rng, 8 * kv_width, -1, 1);
        self.ring = try ref.bfs(a, rng, ring_rows * kv_width, -1, 1);
        self.ring_v = try ref.bfs(a, rng, ring_rows * kv_width, -1, 1);
        self.from = 700;
        self.committed = 1750;
        if (fill == .stress) for ([_][]u16{ self.q, self.bk, self.bv, self.ring, self.ring_v }) |x| for (x) |*v| {
            v.* = c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 17))) - 8)) / 8);
        };
        if (fill == .nonfinite) {
            self.q[0] = c.toBf16(ref.nan);
            self.ring[0] = c.toBf16(ref.inf);
            self.ring_v[kv_width] = c.toBf16(-ref.inf);
            self.bk[0] = c.toBf16(ref.inf);
        }
        self.ptr = .{ try ref.input(rig, self.q), try ref.input(rig, self.bk), try ref.input(rig, self.bv), try ref.input(rig, self.ring), try ref.input(rig, self.ring_v), try rig.output(8 * q_width * 2) };
        return true;
    }
    pub fn run(self: *Attn, rig: *rig_.Rig) !void {
        try self.kern.launchAttn(rig.s, self.ptr[0], self.ptr[1], self.ptr[2], self.ptr[3], self.ptr[4], self.ptr[5], self.from, self.committed, ring_rows, true);
    }
    pub fn count(self: *const Attn) usize {
        return self.shape.rows * q_width;
    }
    /// Head-slice value of key t at dim d: the ring's held rows, then the block's.
    fn value(self: *const Attn, first: u32, held: u32, kv: usize, t: usize, d: usize) f64 {
        const at = if (t < held) (@as(usize, first + t) % ring_rows) * kv_width else (t - held) * kv_width;
        return ref.bf((if (t < held) self.ring_v else self.bv)[at + kv * hd + d]);
    }
    /// Bound the reference softmax over the local ring and block rows, including expf and rescaling errors.
    pub fn solve(self: *const Attn, a: std.mem.Allocator, _: ref.Sums, out: []Iv) !void {
        const u = ref.iv.unit;
        const scale = 1 / @sqrt(@as(f64, hd));
        const first = self.committed - 1024;
        const held = self.committed - first;
        const keys = held + 8;
        const nk: f64 = @floatFromInt(keys);
        const scores = try a.alloc(f64, keys);
        defer a.free(scores);
        const probs = try a.alloc(f64, keys);
        defer a.free(probs);
        for (0..self.shape.rows) |row| {
            for (0..24) |h| {
                const kv = h / 3;
                var m: f64 = -ref.inf;
                var low: f64 = ref.inf;
                var err: f64 = 0;
                for (0..keys) |t| {
                    const kbase = if (t < held) (@as(usize, first + t) % ring_rows) * kv_width + kv * hd else 0;
                    const ksrc = if (t < held) self.ring[kbase..][0..hd] else self.bk[(t - held) * kv_width + kv * hd ..][0..hd];
                    var dot: f64 = 0;
                    var abs: f64 = 0;
                    for (0..hd) |d| {
                        const p = ref.bf(self.q[row * q_width + h * hd + d]) * ref.bf(ksrc[d]);
                        dot += p;
                        abs += @abs(p);
                    }
                    scores[t] = dot * scale;
                    // the dot's 128-term chain, then the scale (rsqrtf's ulps) and its product
                    err = @max(err, c.gamma(hd, u) * abs * scale + (ref.iv.libm_ulps + 2) * u * @abs(scores[t]));
                    m = @max(m, scores[t]);
                    low = @min(low, scores[t]);
                }
                const eps_w = 2 * err + 3 * u * (m - low) + (2 * ref.iv.libm_ulps + 2) * u * (nk + 1);
                const grow = eps_w / (1 - eps_w);
                var l: f64 = 0;
                for (0..keys) |t| {
                    probs[t] = @exp(scores[t] - m);
                    l += probs[t];
                }
                const denom = if (self.mean) nk else l;
                for (0..hd) |d| {
                    var acc: f64 = 0;
                    var acc_abs: f64 = 0;
                    for (0..keys) |t| {
                        const v = self.value(first, held, kv, t, d);
                        acc += probs[t] * v;
                        acc_abs += probs[t] * @abs(v);
                    }
                    const y = acc / denom;
                    var spread: f64 = 0;
                    for (0..keys) |t| spread += probs[t] * @abs(self.value(first, held, kv, t, d) - y);
                    const tol = (grow * spread + c.gamma(3 * nk + 2, u) * (acc_abs + @abs(acc))) / l * (1 + grow) + 2 * u * @abs(y);
                    out[row * q_width + h * hd + d] = (Iv{ .m = y, .lo = y - tol, .hi = y + tol }).bf();
                }
            }
        }
    }
};
