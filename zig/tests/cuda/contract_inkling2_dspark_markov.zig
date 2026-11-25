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
const hidden = 2304;
const vocab = 200_058;

/// The Markov chain's step-independent z part for each (step, channel): y rows 1..7 + newest against rnn's columns.
pub const Kappa = struct {
    kern: *const dspark.Kernels,
    /// Control: the bias added twice.
    bias2: bool = false,
    y: []u16 = &.{},
    newest: []u16 = &.{},
    rnn: []u16 = &.{},
    bias: []u16 = &.{},
    ptr: [5]u64 = @splat(0),
    shape: c.Shape = undefined,

    const rnn_w = 5120;
    const rank = 256;
    pub fn spec(self: *const Kappa) c.Spec {
        const served = &[_]c.Shape{.{ .rows = 7, .n = 3 * rank, .k = 2 * hidden }};
        const name = if (self.bias2) "control: inkling2 dspark tf_ink2_dsp_kappa bias added twice" else "inkling2 dspark tf_ink2_dsp_kappa (y_s and newest columns of rnn)";
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 2, .out = .f32, .served = served };
    }
    fn depth(s: c.Shape) usize {
        return s.k / 32 + 5 + 1;
    }
    pub const sums = {};
    pub fn control(self: *const Kappa) ?g.Gate {
        return if (self.bias2) .shapes else null;
    }
    pub fn free(self: *Kappa, a: std.mem.Allocator) void {
        a.free(self.y);
        a.free(self.newest);
        a.free(self.rnn);
        a.free(self.bias);
        self.y = &.{};
        self.newest = &.{};
        self.rnn = &.{};
        self.bias = &.{};
    }
    /// Seven step-dependent rows and the newest feature produce seven Kappa rows.
    pub fn prepare(self: *Kappa, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.rows > 7 or shape.n != 3 * rank or shape.k != 2 * hidden) return false;
        self.free(a);
        self.shape = shape;
        self.y = try ref.bfs(a, rng, 8 * hidden, -1, 1);
        self.newest = try ref.bfs(a, rng, hidden, -1, 1);
        self.rnn = try ref.bfs(a, rng, 3 * rank * rnn_w, -0.05, 0.05);
        self.bias = try ref.bfs(a, rng, 3 * rank, -1, 1);
        if (fill == .stress) {
            // Integer products 1..21 over 4,608 terms: exact in fp32 in any order; bf16 sums round past 256.
            for (self.y, 0..) |*v, i| v.* = c.toBf16(@floatFromInt(1 + i % 7));
            for (self.newest, 0..) |*v, i| v.* = c.toBf16(@floatFromInt(1 + i % 5));
            for (self.rnn, 0..) |*v, i| v.* = c.toBf16(@floatFromInt(1 + i % 3));
        }
        if (fill == .nonfinite) {
            // Plant NaN and both infinities while leaving other rows finite.
            self.y[hidden] = c.toBf16(ref.nan);
            self.y[2 * hidden] = c.toBf16(ref.inf);
            self.rnn[2816] = c.toBf16(-ref.inf);
        }
        self.ptr = .{ try ref.input(rig, self.y), try ref.input(rig, self.newest), try ref.input(rig, self.rnn), try ref.input(rig, self.bias), try rig.output(7 * 3 * rank * 4) };
        return true;
    }
    pub fn run(self: *Kappa, rig: *rig_.Rig) !void {
        try self.kern.launchKappa(rig.s, self.ptr[0], self.ptr[1], self.ptr[2], self.ptr[3], self.ptr[4]);
    }
    pub fn count(self: *const Kappa) usize {
        return self.shape.rows * 3 * rank;
    }
    /// kappa[step][c] = sum_d y[step+1][d]*w[512+d] + newest[d]*w[2816+d] + bias[c], fp32.
    pub fn solve(self: *const Kappa, a: std.mem.Allocator, sum: ref.Sums, out: []Iv) !void {
        const terms = try a.alloc(Iv, 2 * hidden);
        defer a.free(terms);
        const scratch = try a.alloc(f64, 2 * hidden);
        defer a.free(scratch);
        for (0..self.shape.rows) |step| {
            for (0..3 * rank) |ch| {
                const w = self.rnn[ch * rnn_w ..];
                for (0..hidden) |d| {
                    terms[2 * d] = Iv.of(ref.bf(self.y[(step + 1) * hidden + d])).mul(Iv.of(ref.bf(w[512 + d])));
                    terms[2 * d + 1] = Iv.of(ref.bf(self.newest[d])).mul(Iv.of(ref.bf(w[2816 + d])));
                }
                const total = sum.sum(terms, depth(self.shape), scratch);
                const b = Iv.of(ref.bf(self.bias[ch]));
                out[step * 3 * rank + ch] = if (self.bias2) total.add(b).add(b) else total.add(b);
            }
        }
    }
};

/// One Markov step: z from state and prev's markov row, the gated update, then h bf16 [H] for the unembed.
pub const Step = struct {
    kern: *const dspark.Kernels,
    /// Control: the gate's sigmoid read as its complement.
    flipped: bool = false,
    prev: u32 = 0,
    markov: []u16 = &.{},
    rnn: []u16 = &.{},
    kappa_in: []f32 = &.{},
    hidden_w: []u16 = &.{},
    hidden_b: []u16 = &.{},
    y: []u16 = &.{},
    state_in: []f32 = &.{},
    ptr: [9]u64 = @splat(0),

    const rnn_w = 5120;
    const rank = 256;
    pub fn spec(self: *const Step) c.Spec {
        const served = &[_]c.Shape{.{ .rows = 1, .n = hidden, .k = rank }};
        const name = if (self.flipped) "control: inkling2 dspark tf_ink2_dsp_step gate as 1 - sigmoid" else "inkling2 dspark tf_ink2_dsp_step (gated state + h for the unembed)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .served = served };
    }
    pub fn control(self: *const Step) ?g.Gate {
        return if (self.flipped) .shapes else null;
    }
    pub fn free(self: *Step, a: std.mem.Allocator) void {
        a.free(self.markov);
        a.free(self.rnn);
        a.free(self.kappa_in);
        a.free(self.hidden_w);
        a.free(self.hidden_b);
        a.free(self.y);
        a.free(self.state_in);
        self.markov = &.{};
        self.rnn = &.{};
        self.kappa_in = &.{};
        self.hidden_w = &.{};
        self.hidden_b = &.{};
        self.y = &.{};
        self.state_in = &.{};
    }
    /// markov a fixture vocab of 4096 rows [4096][rank]; prev picks one.
    pub fn prepare(self: *Step, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n != hidden or shape.k != rank) return false;
        self.free(a);
        const vocab_rows = 4096;
        self.prev = rng.uintLessThan(u32, vocab_rows);
        self.markov = try ref.bfs(a, rng, vocab_rows * rank, -1, 1);
        self.rnn = try ref.bfs(a, rng, 3 * rank * rnn_w, -0.05, 0.05);
        self.kappa_in = try ref.f32s(a, rng, 3 * rank, -4, 4);
        self.hidden_w = try ref.bfs(a, rng, hidden * rank, -0.05, 0.05);
        self.hidden_b = try ref.bfs(a, rng, hidden, -1, 1);
        self.y = try ref.bfs(a, rng, hidden, -1, 1);
        self.state_in = try ref.f32s(a, rng, rank, -1, 1);
        if (fill == .nonfinite) {
            self.state_in[0] = ref.nan;
            self.kappa_in[rank] = ref.inf;
            self.markov[@as(usize, self.prev) * rank] = c.toBf16(-ref.inf);
        }
        self.ptr = .{ try ref.input(rig, std.mem.sliceAsBytes(&[_]u32{self.prev})), try ref.input(rig, self.markov), try ref.input(rig, self.rnn), try ref.input(rig, self.kappa_in), try ref.input(rig, self.hidden_w), try ref.input(rig, self.hidden_b), try ref.input(rig, self.y), try rig.output(rank * 4 + hidden * 2), 0 };
        self.ptr[8] = self.ptr[7] + rank * 4;
        return true;
    }
    pub fn run(self: *Step, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.ptr[7], std.mem.sliceAsBytes(self.state_in));
        try self.kern.launchStep(rig.s, self.ptr[0], self.ptr[1], self.ptr[2], self.ptr[3], self.ptr[4], self.ptr[5], self.ptr[6], self.ptr[7], self.ptr[8]);
    }
    pub fn count(_: *const Step) usize {
        return rank + hidden;
    }
    /// state and h as one checked output: state fp32 [rank] then h bf16 [hidden].
    pub fn check(self: *const Step, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        _ = a;
        var z: [3 * rank]f64 = undefined;
        var s_in: [2 * rank]f64 = undefined;
        for (0..rank) |r| {
            s_in[r] = self.state_in[r];
            s_in[rank + r] = ref.bf(self.markov[@as(usize, self.prev) * rank + r]);
        }
        for (0..3 * rank) |ch| {
            const w = self.rnn[ch * rnn_w ..];
            var total: f64 = self.kappa_in[ch];
            for (0..2 * rank) |d| total += s_in[d] * ref.bf(w[d]);
            z[ch] = total;
        }
        var lam: [rank]f64 = undefined;
        for (0..rank) |r| {
            const gamma = if (self.flipped) 1 - 1 / (1 + @exp(-z[r])) else 1 / (1 + @exp(-z[r]));
            const want_s = gamma * s_in[r] + (1 - gamma) * std.math.tanh(z[rank + r]);
            lam[r] = std.math.tanh(z[2 * rank + r]);
            const got_s = ref.readF32(got, r);
            if (!(std.math.isNan(want_s) and std.math.isNan(got_s)) and @abs(got_s - want_s) > 1e-4)
                return try std.fmt.bufPrint(msg, "state {d}: {e} where {e} is due", .{ r, got_s, want_s });
        }
        for (0..hidden) |d| {
            var v = ref.bf(self.y[d]) + ref.bf(self.hidden_b[d]);
            for (0..rank) |r| v += ref.bf(self.hidden_w[d * rank + r]) * lam[r];
            const goth = ref.bf(@bitCast(@as(u16, std.mem.readInt(u16, got[rank * 4 + 2 * d ..][0..2], .little))));
            if (!(std.math.isNan(v) and std.math.isNan(goth)) and @abs(goth - v) > 0.02 * @abs(v) + 0.02)
                return try std.fmt.bufPrint(msg, "h {d}: {e} where {e} is due", .{ d, goth, v });
        }
        return null;
    }
    pub fn solve(_: *const Step, _: std.mem.Allocator, _: ref.Sums, _: []Iv) !void {}
};
