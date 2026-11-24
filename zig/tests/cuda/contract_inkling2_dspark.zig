//! Inkling 2 Mini's DSpark head launches under FP64 references and planted controls: the residual norm, SwiGLU and the
//! capped greedy pick. Its attention, ring and Markov launches are checked whole by `tf-cuda-test inkling2-dspark`.
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
const inter = 9216;
const vocab = 200_058;
const cap: f32 = 30;
const eps: f32 = 1e-6;

/// resid += y, then x = bf16(rms-normed resid * norm): the output is x [rows][H]; resid is the launch's own state.
pub const AddNorm = struct {
    kern: *const dspark.Kernels,
    /// Control: the norm reads the residual before y lands.
    stale: bool = false,
    shape: c.Shape = undefined,
    y: []u16 = &.{},
    norm: []u16 = &.{},
    resid: []f32 = &.{},
    ptr: [4]u64 = @splat(0),

    pub const sums = {};
    pub fn spec(self: *const AddNorm) c.Spec {
        const served = &[_]c.Shape{ .{ .rows = 8, .n = hidden, .k = 1 }, .{ .rows = 1, .n = hidden, .k = 1 } };
        const name = if (self.stale) "control: inkling2 dspark tf_ink2_dsp_add_norm normed before y lands" else "inkling2 dspark tf_ink2_dsp_add_norm (fp32 residual, bf16 rms norm)";
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 3, .out = .bf16, .served = if (self.stale) served[0..1] else served };
    }
    fn depth(s: c.Shape) usize {
        return ref.blockDepth(s.n, 256) + 1;
    }
    pub fn control(self: *const AddNorm) ?g.Gate {
        return if (self.stale) .shapes else null;
    }
    pub fn free(self: *AddNorm, a: std.mem.Allocator) void {
        a.free(self.y);
        a.free(self.norm);
        a.free(self.resid);
        self.y = &.{};
        self.norm = &.{};
        self.resid = &.{};
    }
    pub fn prepare(self: *AddNorm, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != hidden) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * hidden;
        self.y = try ref.bfs(a, rng, n, -1, 1);
        self.norm = try ref.bfs(a, rng, hidden, 0.5, 1.5);
        self.resid = try ref.f32s(a, rng, n, -4, 4);
        if (fill == .stress) for (self.resid, self.y, 0..) |*r, *y, i| {
            r.* = if (i % 3 == 0) 1e4 else -1e4;
            y.* = c.toBf16(0.5);
        };
        self.ptr = .{ try ref.input(rig, self.y), try ref.input(rig, self.norm), try rig.scratch(n * 4), try rig.output(n * 2) };
        return true;
    }
    pub fn run(self: *AddNorm, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.ptr[2], std.mem.sliceAsBytes(self.resid));
        try self.kern.launchAddNorm(rig.s, if (self.stale) self.ptr[3] else self.ptr[0], self.ptr[1], self.ptr[2], self.ptr[3], self.shape.rows);
    }
    pub fn count(self: *const AddNorm) usize {
        return self.shape.rows * hidden;
    }
    pub fn solve(self: *const AddNorm, a: std.mem.Allocator, sum: ref.Sums, out: []Iv) !void {
        const r = try a.alloc(Iv, 2 * hidden);
        defer a.free(r);
        const scratch = try a.alloc(f64, hidden);
        defer a.free(scratch);
        for (0..self.shape.rows) |row| {
            const at = row * hidden;
            for (r[0..hidden], r[hidden..], self.resid[at..][0..hidden], self.y[at..][0..hidden]) |*v, *t, x, y| {
                v.* = Iv.of(x).add(Iv.of(ref.bf(y)));
                t.* = ref.sq(v.*);
            }
            const inv = ref.rms(sum.sum(r[hidden..], depth(self.shape), scratch), hidden, eps);
            for (out[at..][0..hidden], r[0..hidden], self.norm) |*o, v, w| o.* = v.mul(inv).mul(Iv.of(ref.bf(w))).bf();
        }
    }
};

/// act = bf16(silu(gate) * up) over the block's rows x the intermediate width.
pub const Swiglu = struct {
    kern: *const dspark.Kernels,
    /// Control: gate and up swapped.
    swapped: bool = false,
    shape: c.Shape = undefined,
    gate: []u16 = &.{},
    up: []u16 = &.{},
    ptr: [3]u64 = @splat(0),

    pub fn spec(self: *const Swiglu) c.Spec {
        const served = &[_]c.Shape{ .{ .rows = 8, .n = inter, .k = 1 }, .{ .rows = 1, .n = inter, .k = 1 } };
        const name = if (self.swapped) "control: inkling2 dspark tf_ink2_dsp_swiglu with gate and up swapped" else "inkling2 dspark tf_ink2_dsp_swiglu (silu(gate) * up, bf16)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .ops = 4, .out = .bf16, .served = if (self.swapped) served[0..1] else served };
    }
    pub fn control(self: *const Swiglu) ?g.Gate {
        return if (self.swapped) .shapes else null;
    }
    pub fn free(self: *Swiglu, a: std.mem.Allocator) void {
        a.free(self.gate);
        a.free(self.up);
        self.gate = &.{};
        self.up = &.{};
    }
    pub fn prepare(self: *Swiglu, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != inter) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * inter;
        const span: f32 = if (fill == .stress) 80 else 6;
        self.gate = try ref.bfs(a, rng, n, -span, span);
        self.up = try ref.bfs(a, rng, n, -span, span);
        if (fill == .nonfinite) {
            self.gate[0] = c.toBf16(ref.nan);
            self.gate[1] = c.toBf16(-ref.inf);
            self.up[2] = c.toBf16(ref.inf);
        }
        self.ptr = .{ try ref.input(rig, self.gate), try ref.input(rig, self.up), try rig.output(n * 2) };
        return true;
    }
    pub fn run(self: *Swiglu, rig: *rig_.Rig) !void {
        const gate, const up = if (self.swapped) .{ self.ptr[1], self.ptr[0] } else .{ self.ptr[0], self.ptr[1] };
        try self.kern.launchSwiglu(rig.s, gate, up, self.ptr[2], self.shape.rows * inter);
    }
    pub fn count(self: *const Swiglu) usize {
        return self.shape.rows * inter;
    }
    pub fn solve(self: *const Swiglu, _: std.mem.Allocator, _: ref.Sums, out: []Iv) !void {
        for (out, self.gate, self.up) |*o, gw, uw| {
            const x = Iv.of(ref.bf(gw));
            o.* = x.mul(ref.sigmoid(x)).mul(Iv.of(ref.bf(uw))).bf();
        }
    }
};

/// One row of fp32 logits capped in place (30 tanh(x / 30)), then the first id of the capped maximum.
pub const Pick = struct {
    kern: *const dspark.Kernels,
    /// Control: one id short of the vocabulary.
    short: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    ptr: [2]u64 = @splat(0),

    pub fn spec(self: *const Pick) c.Spec {
        const served = &[_]c.Shape{ .{ .rows = 1, .n = vocab, .k = 1 }, .{ .rows = 1, .n = 1000, .k = 1 } };
        const name = if (self.short) "control: inkling2 dspark tf_ink2_dsp_pick one id short" else "inkling2 dspark tf_ink2_dsp_pick (capped fp32 logits, first id of the maximum)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .in_place = true, .nan = .{ .stated = "finite logits only: the head's logits are a bf16 GEMV of normed rows" }, .served = if (self.short) served[0..1] else served };
    }
    pub fn control(self: *const Pick) ?g.Gate {
        return if (self.short) .shapes else null;
    }
    pub fn free(self: *Pick, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }
    /// Values on 1/8 steps of [-64, 64): ties everywhere, and the unique maximum at the last id, where a short scan misses.
    pub fn prepare(self: *Pick, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows != 1 or shape.n < 2 or shape.n > vocab or fill == .nonfinite) return false;
        self.free(a);
        self.shape = shape;
        self.x = try a.alloc(f32, shape.n);
        for (self.x) |*v| v.* = ref.int(rng, -512, 511) / 8;
        self.x[shape.n - 1] = 100;
        self.ptr = .{ try rig.output(shape.n * 4), try rig.output(4) };
        return true;
    }
    pub fn run(self: *Pick, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.ptr[0], std.mem.sliceAsBytes(self.x));
        try self.kern.launchPick(rig.s, self.ptr[0], self.shape.n - @intFromBool(self.short), cap, self.ptr[1]);
    }
    pub fn count(self: *const Pick) usize {
        return 4 * (self.shape.n + 1);
    }
    /// The capped logits within tanhf's error of float64's, and the pick exactly the host's first maximum.
    pub fn check(self: *const Pick, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        var best: usize = 0;
        for (self.x, 0..) |x, i| {
            const want = cap * std.math.tanh(@as(f64, x) / cap);
            const v = ref.readF32(got, i);
            if (@abs(v - want) > 1e-5 * cap) return try std.fmt.bufPrint(msg, "id {d}: capped {e} where {e} is due", .{ i, v, want });
            if (ref.readF32(got, i) > ref.readF32(got, best)) best = i;
        }
        const pick = ref.readI32(got, self.shape.n);
        if (pick != @as(i32, @intCast(best))) return try std.fmt.bufPrint(msg, "picked {d} where {d} is due", .{ pick, best });
        return null;
    }
};

const groups = .{ "norms", "swiglus", "picks" };
pub const Set = struct {
    k: dspark.Kernels = undefined,
    loaded: bool = false,
    norms: [2]ref.Gated(AddNorm) = undefined,
    swiglus: [2]ref.Gated(Swiglu) = undefined,
    picks: [2]ref.Checked(Pick) = undefined,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try dspark.Kernels.load(gpu.d);
        set.loaded = true;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&set.norms, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .stale = i == 1 } };
        for (&set.swiglus, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .swapped = i == 1 } };
        for (&set.picks, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .short = i == 1 } };
        inline for (groups) |field| for (&@field(set, field)) |*x| try list.append(a, x.kernel());
    }
    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        inline for (groups) |field| for (&@field(set, field)) |*x| x.deinit();
        set.k.unload();
        set.loaded = false;
    }
};
