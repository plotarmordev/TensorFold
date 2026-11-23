//! Inkling 2 Mini's served message and gate launches under FP64 references and planted controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const loop = @import("inkling2").loop;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const hidden = 2304;
const rank = 64;
const eps = ref.eps;
const mix_shapes = [_]c.Shape{ .{ .rows = 1, .n = hidden, .k = 3 }, .{ .rows = 3, .n = hidden, .k = 2 } };
const gate_shapes = [_]c.Shape{ .{ .rows = 1, .n = hidden, .k = rank }, .{ .rows = 3, .n = hidden, .k = rank } };

/// Raw outputs use the old loop contract's FP32 tolerance; BF16 references include their final rounding.
pub fn Checked(comptime T: type) type {
    return struct {
        const Self = @This();
        inner: ref.Gated(T),
        rig: *rig_.Rig = undefined,
        msg: [200]u8 = undefined,

        pub fn deinit(self: *Self) void {
            self.inner.deinit();
        }
        pub fn kernel(self: *Self) g.Kernel {
            self.rig = &self.inner.rig;
            return .{ .spec = self.inner.k.spec(), .ctx = self, .vt = &vt, .control = self.inner.k.control() };
        }
        const vt = g.hooks(Self);
        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            return self.inner.prepare(shape, fill, seed);
        }
        pub fn run(self: *Self) !void {
            try self.inner.run();
        }
        pub fn count(self: *Self) usize {
            return self.inner.count();
        }
        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            try self.inner.reference(want, mag);
        }
        pub fn bounds(self: *Self, want: []const f64, mag: []const f64, allow: []f64) !void {
            try self.inner.bounds(want, mag, allow);
        }
        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            try self.inner.simulate(acc, order, out);
        }
        pub fn check(self: *Self, _: g.Gate) !?[]const u8 {
            const a = self.rig.gpu.gpa;
            const buf = try a.alloc(f64, 2 * self.count());
            defer a.free(buf);
            const want = buf[0..self.count()];
            try self.reference(want, buf[self.count()..]);
            const got = try self.rig.outputBytes();
            for (want, 0..) |w, i| {
                const x = c.read(self.inner.k.spec().out, got, i);
                const ok = if (std.math.isNan(w)) std.math.isNan(x) else if (!std.math.isFinite(w)) x == w else std.math.isFinite(x) and @abs(x - w) <= 1e-4 * (1 + @abs(w));
                if (!ok) return try std.fmt.bufPrint(&self.msg, "output {d}: {e} where {e} is due", .{ i, x, w });
            }
            return null;
        }
    };
}

fn stressBank(i: usize) f32 {
    return if ((i * 37 + i / 7) % 5 == 0) 384 else 256;
}

/// The four role queries over completed blocks and a partial, with Mini's post-dot 1/48 scale.
pub const Mix = struct {
    kern: *const loop.Kernels,
    f32_partial: bool = false,
    aliased: bool = false,
    shape: c.Shape = undefined,
    blocks: []u16 = &.{},
    partial: []f32 = &.{},
    queries: []f32 = &.{},
    args: loop.MixArgs = undefined,

    pub const sums = {};
    pub fn spec(self: *const Mix) c.Spec {
        return .{ .name = if (self.aliased) "control: inkling2 loop mix with block one replaced by block zero" else "inkling2 loop tf_ink2_loop_mix (four role queries, normalized sources)", .accum = .f32, .depth = depth, .out = .f32, .served = if (self.aliased) mix_shapes[0..1] else &mix_shapes };
    }
    fn depth(s: c.Shape) usize {
        return 2 * ref.blockDepth(s.n, 256) + 16;
    }
    pub fn control(self: *const Mix) ?g.Gate {
        return if (self.aliased) .shapes else null;
    }
    pub fn free(self: *Mix, a: std.mem.Allocator) void {
        a.free(self.blocks);
        a.free(self.partial);
        a.free(self.queries);
        self.blocks = &.{};
        self.partial = &.{};
        self.queries = &.{};
    }
    pub fn prepare(self: *Mix, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != hidden or shape.k < 1 or shape.k > loop.max_blocks + 1) return false;
        self.free(a);
        self.shape = shape;
        const nb = shape.k - 1;
        const plane = shape.rows * hidden;
        self.blocks = try ref.bfs(a, rng, nb * plane, -1, 1);
        self.partial = try ref.f32s(a, rng, plane, -1, 1);
        self.queries = try ref.f32s(a, rng, 4 * hidden, -0.25, 0.25);
        if (fill == .stress) {
            for (self.blocks, 0..) |*v, i| v.* = c.toBf16(stressBank(i % hidden));
            for (self.partial, 0..) |*v, i| v.* = stressBank(i % hidden);
            @memset(self.queries, 0);
        }
        if (!self.f32_partial) {
            for (self.partial) |*v| v.* = c.fromBf16(c.toBf16(v.*));
        }
        if (fill == .nonfinite) {
            if (nb != 0) ref.plant(u16, self.blocks[0..plane], shape.rows, hidden, ref.asBf);
            ref.plant(f32, self.partial, shape.rows, hidden, ref.asF32);
        }
        var args: loop.MixArgs = .{ .partial = 0, .out = 0, .out_stride = @intCast(plane), .n_blocks = @intCast(nb), .partial_f32 = @intFromBool(self.f32_partial), .nq = 4, .rows = @intCast(shape.rows), .h = hidden, .eps = eps, .scale = 1.0 / 48.0 };
        for (0..nb) |b| args.blocks[b] = try ref.input(rig, self.blocks[b * plane ..][0..plane]);
        if (self.f32_partial) {
            args.partial = try ref.input(rig, self.partial);
        } else {
            const bits = try a.alloc(u16, plane);
            defer a.free(bits);
            for (bits, self.partial) |*d, v| d.* = c.toBf16(v);
            args.partial = try ref.input(rig, bits);
        }
        for (0..4) |q| args.query[q] = try ref.input(rig, self.queries[q * hidden ..][0..hidden]);
        args.out = try rig.output(4 * plane * 4);
        self.args = args;
        return true;
    }
    pub fn run(self: *Mix, rig: *rig_.Rig) !void {
        var args = self.args;
        if (self.aliased) args.blocks[1] = args.blocks[0];
        try loop.mix(self.kern, rig.s, args);
    }
    pub fn count(self: *const Mix) usize {
        return 4 * self.shape.rows * hidden;
    }
    fn source(self: *const Mix, s: usize, row: usize, col: usize) f64 {
        const plane = self.shape.rows * hidden;
        return if (s < self.shape.k - 1) ref.bf(self.blocks[s * plane + row * hidden + col]) else self.partial[row * hidden + col];
    }
    pub fn solve(self: *const Mix, a: std.mem.Allocator, sum: ref.Sums, out: []ref.Iv) !void {
        const terms = try a.alloc(ref.Iv, hidden);
        defer a.free(terms);
        const scratch = try a.alloc(f64, hidden);
        defer a.free(scratch);
        const d = depth(self.shape);
        for (0..self.shape.rows) |row| {
            var inv: [loop.max_blocks + 1]ref.Iv = undefined;
            for (0..self.shape.k) |s| {
                for (terms, 0..) |*t, col| {
                    const x = ref.Iv.of(self.source(s, row, col));
                    t.* = x.mul(x);
                }
                inv[s] = ref.rms(sum.sum(terms, d, scratch), hidden, eps);
            }
            for (0..4) |q| {
                var logits: [loop.max_blocks + 1]ref.Iv = undefined;
                for (0..self.shape.k) |s| {
                    for (terms, 0..) |*t, col| t.* = ref.Iv.of(self.queries[q * hidden + col]).mul(ref.Iv.of(self.source(s, row, col))).mul(inv[s]);
                    logits[s] = sum.sum(terms, d, scratch).mul(ref.Iv.of(1.0 / 48.0));
                }
                softmax(logits[0..self.shape.k]);
                for (0..hidden) |col| {
                    var value = ref.Iv.of(0);
                    for (0..self.shape.k) |s| value = value.add(logits[s].mul(ref.Iv.of(self.source(s, row, col)).mul(inv[s])));
                    out[(q * self.shape.rows + row) * hidden + col] = value;
                }
            }
        }
    }
};

fn softmax(logits: []ref.Iv) void {
    var peak = logits[0];
    for (logits[1..]) |v| peak = .{ .m = @max(peak.m, v.m), .lo = @max(peak.lo, v.lo), .hi = @max(peak.hi, v.hi) };
    var den = ref.Iv.of(0);
    for (logits) |*v| {
        v.* = v.sub(peak).exp();
        den = den.add(v.*);
    }
    for (logits) |*v| v.* = v.div(den);
}

const Stage = enum { down, lin, mix };

/// Three gates read message role planes 0, 1 and 2; each registration checks one launch's result.
pub const Gate = struct {
    kern: *const loop.Kernels,
    stage: Stage,
    wrong_slot: bool = false,
    shape: c.Shape = undefined,
    h: []f32 = &.{},
    msg: []f32 = &.{},
    down_w: []u16 = &.{},
    forget: []u16 = &.{},
    gain: []u16 = &.{},
    norm: []u16 = &.{},
    args: loop.GateArgs = undefined,

    pub const sums = {};
    pub fn spec(self: *const Gate) c.Spec {
        const names = [_][]const u8{ "inkling2 loop tf_ink2_loop_gate_down (BF16 rank-64 projections)", "inkling2 loop tf_ink2_loop_gate_lin (forget/gain blend)", "inkling2 loop tf_ink2_loop_gate_mix (BF16 normed output)" };
        return .{ .name = if (self.wrong_slot) "control: inkling2 loop gate zero reads message role plane one" else names[@backingInt(self.stage)], .accum = .f32, .depth = depth, .out = if (self.stage == .lin) .f32 else .bf16, .served = if (self.wrong_slot) gate_shapes[0..1] else &gate_shapes };
    }
    fn depth(s: c.Shape) usize {
        return ref.blockDepth(s.n, 256) + (2 * s.n + 31) / 32 + 5 + 2 * rank;
    }
    pub fn control(self: *const Gate) ?g.Gate {
        return if (self.wrong_slot) .shapes else null;
    }
    pub fn free(self: *Gate, a: std.mem.Allocator) void {
        a.free(self.h);
        a.free(self.msg);
        a.free(self.down_w);
        a.free(self.forget);
        a.free(self.gain);
        a.free(self.norm);
        self.h = &.{};
        self.msg = &.{};
        self.down_w = &.{};
        self.forget = &.{};
        self.gain = &.{};
        self.norm = &.{};
    }
    pub fn prepare(self: *Gate, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != hidden or shape.k != rank) return false;
        self.free(a);
        self.shape = shape;
        const plane = shape.rows * hidden;
        self.h = try ref.f32s(a, rng, 3 * plane, -1, 1);
        self.msg = try ref.f32s(a, rng, 4 * plane, -1, 1);
        self.down_w = try ref.bfs(a, rng, 3 * rank * 2 * hidden, -0.1, 0.1);
        self.forget = try ref.bfs(a, rng, 3 * hidden * rank, -0.2, 0.2);
        self.gain = try ref.bfs(a, rng, 3 * hidden * rank, -0.2, 0.2);
        self.norm = try ref.bfs(a, rng, 3 * hidden, 0.5, 1.5);
        if (fill == .stress) {
            for (self.h, 0..) |*v, i| v.* = stressBank(i % hidden);
            for (self.msg, 0..) |*v, i| v.* = stressBank(i % hidden);
            for (self.down_w) |*v| v.* = c.toBf16(ref.int(rng, -8, 8));
            for (self.forget) |*v| v.* = c.toBf16(ref.int(rng, -8, 8));
            for (self.gain) |*v| v.* = c.toBf16(ref.int(rng, -8, 8));
        }
        if (fill == .nonfinite) {
            ref.plant(f32, self.h, 3 * shape.rows, hidden, ref.asF32);
            ref.plant(f32, self.msg, 4 * shape.rows, hidden, ref.asF32);
        }
        var args: loop.GateArgs = .{ .msg = try ref.input(rig, self.msg), .msg_stride = @intCast(plane), .slot = .{ 0, 1, 2 }, .down = 0, .out = 0, .out_stride = @intCast(plane), .rows = @intCast(shape.rows), .h_dim = hidden, .rank = rank, .eps_unit = eps, .eps = eps, .ys = 0 };
        for (0..3) |gate| {
            args.h[gate] = try ref.input(rig, self.h[gate * plane ..][0..plane]);
            args.down_w[gate] = try ref.input(rig, self.down_w[gate * rank * 2 * hidden ..][0 .. rank * 2 * hidden]);
            args.forget[gate] = try ref.input(rig, self.forget[gate * hidden * rank ..][0 .. hidden * rank]);
            args.gain[gate] = try ref.input(rig, self.gain[gate * hidden * rank ..][0 .. hidden * rank]);
            args.norm[gate] = try ref.input(rig, self.norm[gate * hidden ..][0..hidden]);
        }
        const dn = 3 * shape.rows * rank * 2;
        const yn = 3 * plane * 4;
        const on = 3 * plane * 2;
        args.down = if (self.stage == .down) try rig.output(dn) else try rig.scratch(dn);
        args.ys = if (self.stage == .lin) try rig.output(yn) else try rig.scratch(yn);
        args.out = if (self.stage == .mix) try rig.output(on) else try rig.scratch(on);
        self.args = args;
        return true;
    }
    pub fn run(self: *Gate, rig: *rig_.Rig) !void {
        var args = self.args;
        if (self.wrong_slot) args.slot[0] = 1;
        try loop.gates(self.kern, rig.s, args, 3);
    }
    pub fn count(self: *const Gate) usize {
        return switch (self.stage) {
            .down => 3 * self.shape.rows * rank,
            .lin, .mix => 3 * self.shape.rows * hidden,
        };
    }
    pub fn solve(self: *const Gate, a: std.mem.Allocator, sum: ref.Sums, out: []ref.Iv) !void {
        const terms = try a.alloc(ref.Iv, 2 * hidden);
        defer a.free(terms);
        const scratch = try a.alloc(f64, 2 * hidden);
        defer a.free(scratch);
        const normalized = try a.alloc(ref.Iv, 2 * hidden);
        defer a.free(normalized);
        const zeta = try a.alloc(ref.Iv, rank);
        defer a.free(zeta);
        const ys = try a.alloc(ref.Iv, 3 * self.shape.rows * hidden);
        defer a.free(ys);
        const later = ref.later(sum);
        const d = depth(self.shape);
        for (0..3) |gate| for (0..self.shape.rows) |row| {
            const h = self.h[(gate * self.shape.rows + row) * hidden ..][0..hidden];
            const m = self.msg[(gate * self.shape.rows + row) * hidden ..][0..hidden];
            var inv: [2]ref.Iv = undefined;
            for (0..2) |side| {
                const x = if (side == 0) h else m;
                for (terms[0..hidden], x) |*t, v| {
                    const e = ref.Iv.of(v);
                    t.* = e.mul(e);
                }
                inv[side] = ref.rms(sum.sum(terms[0..hidden], d, scratch), hidden, eps);
            }
            for (normalized[0..hidden], h) |*v, x| v.* = ref.Iv.of(x).mul(inv[0]).bf();
            for (normalized[hidden..], m) |*v, x| v.* = ref.Iv.of(x).mul(inv[1]).bf();
            for (0..rank) |r| {
                for (terms, normalized, self.down_w[(gate * rank + r) * 2 * hidden ..][0 .. 2 * hidden]) |*t, x, w| t.* = ref.Iv.of(ref.bf(w)).mul(x);
                zeta[r] = later.sum(terms, (2 * hidden + 31) / 32 + 5, scratch).bf();
                if (self.stage == .down) out[(gate * self.shape.rows + row) * rank + r] = zeta[r];
            }
            for (0..hidden) |col| {
                for (terms[0..rank], zeta, self.forget[(gate * hidden + col) * rank ..][0..rank]) |*t, z, w| t.* = ref.Iv.of(ref.bf(w)).mul(z);
                const f = ref.tanh(later.sum(terms[0..rank], rank, scratch));
                for (terms[0..rank], zeta, self.gain[(gate * hidden + col) * rank ..][0..rank]) |*t, z, w| t.* = ref.Iv.of(ref.bf(w)).mul(z);
                const gain = ref.tanh(later.sum(terms[0..rank], rank, scratch));
                const y = ref.Iv.of(1).sub(f).mul(ref.Iv.of(h[col])).add(gain.mul(ref.Iv.of(m[col])));
                const at = (gate * self.shape.rows + row) * hidden + col;
                ys[at] = y;
                if (self.stage == .lin) out[at] = y;
            }
        };
        if (self.stage != .mix) return;
        for (0..3) |gate| for (0..self.shape.rows) |row| {
            const at = (gate * self.shape.rows + row) * hidden;
            for (terms[0..hidden], ys[at..][0..hidden]) |*t, y| t.* = ref.sq(y);
            const inv = ref.rms(later.sum(terms[0..hidden], d, scratch), hidden, eps);
            for (out[at..][0..hidden], ys[at..][0..hidden], self.norm[gate * hidden ..][0..hidden]) |*o, y, n| o.* = ref.Iv.of(ref.bf(n)).mul(y.mul(inv)).bf();
        };
    }
};

const groups = .{ "mixes", "gates" };
pub const Set = struct {
    k: loop.Kernels = undefined,
    loaded: bool = false,
    mixes: [3]Checked(Mix) = undefined,
    gates: [4]Checked(Gate) = undefined,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try loop.Kernels.load(gpu.d);
        set.loaded = true;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&set.mixes, 0..) |*x, i| x.* = .{ .inner = .{ .rig = rig, .k = .{ .kern = &set.k, .f32_partial = i == 1, .aliased = i == 2 } } };
        for (&set.gates, 0..) |*x, i| x.* = .{ .inner = .{ .rig = rig, .k = .{ .kern = &set.k, .stage = @fromBackingInt(@intCast(i % 3)), .wrong_slot = i == 3 } } };
        inline for (groups) |field| for (&@field(set, field)) |*x| try list.append(a, x.kernel());
    }
    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        inline for (groups) |field| for (&@field(set, field)) |*x| x.deinit();
        set.k.unload();
        set.loaded = false;
    }
};
