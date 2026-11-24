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
const soft_why = "online softmax over the ring and block keys: no input makes its sums exact in fp32; gate 3's derived bound holds it";

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
            r.* = if ((i * 37 + i / 7) % 5 == 0) 384 else 256; // whole-number squares bf16 sums lose and fp32 keeps
            y.* = 0;
        };
        if (fill == .nonfinite) {
            self.resid[3] = ref.nan;
            self.y[5] = c.toBf16(ref.inf);
        }
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

/// Rows of fp32 logits, one launch each as the head's seven steps: each capped in place (30 tanh(x / 30)), then the
/// first id of its capped maximum. Outputs: the capped rows, then the picks.
pub const Pick = struct {
    kern: *const dspark.Kernels,
    /// Control: one id short of the vocabulary.
    short: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    ptr: [2]u64 = @splat(0),

    pub fn spec(self: *const Pick) c.Spec {
        const served = &[_]c.Shape{ .{ .rows = 1, .n = vocab, .k = 1 }, .{ .rows = 7, .n = vocab, .k = 1 }, .{ .rows = 1, .n = 1000, .k = 1 } };
        const name = if (self.short) "control: inkling2 dspark tf_ink2_dsp_pick one id short" else "inkling2 dspark tf_ink2_dsp_pick (capped fp32 logits, first id of the maximum)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .in_place = true, .nan = .{ .stated = "NaN stays NaN and is never picked; +-Inf caps to +-30" }, .served = if (self.short) served[0..1] else served };
    }
    pub fn control(self: *const Pick) ?g.Gate {
        return if (self.short) .shapes else null;
    }
    pub fn free(self: *Pick, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }
    /// Values on 1/8 steps of [-64, 64): ties everywhere, and each row's unique maximum at its last id, where a short
    /// scan misses; nonfinite: NaN, -Inf and +Inf at row 0's ids 0-2.
    pub fn prepare(self: *Pick, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n < 4 or shape.n > vocab) return false;
        self.free(a);
        self.shape = shape;
        self.x = try a.alloc(f32, shape.rows * shape.n);
        for (self.x) |*v| v.* = ref.int(rng, -512, 511) / 8;
        for (0..shape.rows) |r| self.x[r * shape.n + shape.n - 1] = 100;
        if (fill == .nonfinite) self.x[0..3].* = .{ ref.nan, -ref.inf, ref.inf };
        self.ptr = .{ try rig.output(self.x.len * 4), try rig.output(shape.rows * 4) };
        return true;
    }
    pub fn run(self: *Pick, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.ptr[0], std.mem.sliceAsBytes(self.x));
        const n = self.shape.n;
        for (0..self.shape.rows) |r| try self.kern.launchPick(rig.s, self.ptr[0] + r * n * 4, n - @intFromBool(self.short), cap, self.ptr[1] + r * 4);
    }
    pub fn count(self: *const Pick) usize {
        return 4 * self.shape.rows * (self.shape.n + 1);
    }
    /// The capped logits within tanhf's error of float64's (NaN for NaN), and each pick its row's first device maximum.
    pub fn check(self: *const Pick, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| {
            var best: usize = 0;
            for (self.x[r * n ..][0..n], 0..) |x, i| {
                const want = cap * std.math.tanh(@as(f64, x) / cap);
                const v = ref.readF32(got, r * n + i);
                if (std.math.isNan(want) != std.math.isNan(v) or @abs(v - want) > 1e-5 * cap) return try std.fmt.bufPrint(msg, "row {d} id {d}: capped {e} where {e} is due", .{ r, i, v, want });
                const top = ref.readF32(got, r * n + best);
                if (v > top or std.math.isNan(top)) best = i;
            }
            const pick = ref.readI32(got, self.x.len + r);
            if (pick != @as(i32, @intCast(best))) return try std.fmt.bufPrint(msg, "row {d}: picked {d} where {d} is due", .{ r, pick, best });
        }
        return null;
    }
};

/// The four taps mixed by each fusion's softmax weights, then normed by its dflare row: c [4][n][H], plus fusion
/// 3's last row into newest [H].
pub const Fuse = struct {
    kern: *const dspark.Kernels,
    /// Control: the mix weights normalized by their max, not their sum.
    bymax: bool = false,
    shape: c.Shape = undefined,
    taps: []u16 = &.{},
    fuse: []u16 = &.{},
    dflare: []u16 = &.{},
    ptr: [5]u64 = @splat(0),

    pub const sums = {};
    pub fn spec(self: *const Fuse) c.Spec {
        const served = &[_]c.Shape{ .{ .rows = 8, .n = 4 * hidden, .k = 4 }, .{ .rows = 1, .n = 4 * hidden, .k = 4 } };
        const name = if (self.bymax) "control: inkling2 dspark tf_ink2_dsp_fuse weights over their max" else "inkling2 dspark tf_ink2_dsp_fuse (4-way softmax tap mix, rms norm)";
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 5, .out = .bf16, .served = if (self.bymax) served[0..1] else served };
    }
    fn depth(s: c.Shape) usize {
        return s.k + ref.blockDepth(s.n / 4, 256) + 1;
    }
    pub fn control(self: *const Fuse) ?g.Gate {
        return if (self.bymax) .shapes else null;
    }
    pub fn free(self: *Fuse, a: std.mem.Allocator) void {
        a.free(self.taps);
        a.free(self.fuse);
        a.free(self.dflare);
        self.taps = &.{};
        self.fuse = &.{};
        self.dflare = &.{};
    }
    /// `n` rows of four taps [n][4][H]; served n is the block's eight rows or one.
    pub fn prepare(self: *Fuse, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.n != 4 * hidden or shape.k != 4 or shape.rows == 0 or shape.rows > 64) return false;
        self.free(a);
        self.shape = shape;
        self.taps = try ref.bfs(a, rng, shape.rows * 4 * hidden, -1, 1);
        self.fuse = try ref.bfs(a, rng, 16, -2, 2);
        self.dflare = try ref.bfs(a, rng, 4 * hidden, 0.5, 1.5);
        if (fill == .stress) {
            // Equal weights (a quarter each) of integer taps 1..9: every m and m^2 is exact and their sum stays under
            // 2^24 in any order, while bf16 terms and partials round and fp16's sum overflows.
            @memset(self.fuse, c.toBf16(0));
            for (self.taps, 0..) |*t, i| t.* = c.toBf16(@floatFromInt(1 + (i * 37 + i / 7) % 9));
        }
        if (fill == .nonfinite) {
            self.taps[0] = c.toBf16(ref.nan);
            self.taps[hidden] = c.toBf16(ref.inf);
            self.taps[2 * hidden] = c.toBf16(-ref.inf);
        }
        self.ptr = .{ try ref.input(rig, self.taps), try ref.input(rig, self.fuse), try ref.input(rig, self.dflare), try rig.output(4 * shape.rows * hidden * 2), try rig.output(hidden * 2) };
        return true;
    }
    pub fn run(self: *Fuse, rig: *rig_.Rig) !void {
        try self.kern.launchFuse(rig.s, self.ptr[0], self.ptr[1], self.ptr[2], self.ptr[3], self.ptr[4], self.shape.rows);
    }
    pub fn count(self: *const Fuse) usize {
        return 4 * self.shape.rows * hidden + hidden;
    }
    /// c[f][row][d] = bf16(mix * rsqrt(sum sq / H + eps) * dflare[f][d]); newest[d] is c[3][rows-1][d] again.
    pub fn solve(self: *const Fuse, a: std.mem.Allocator, sum: ref.Sums, out: []Iv) !void {
        const n = self.shape.rows;
        const scratch = try a.alloc(f64, hidden);
        defer a.free(scratch);
        const m = try a.alloc(Iv, hidden);
        defer a.free(m);
        const sqs = try a.alloc(Iv, hidden);
        defer a.free(sqs);
        for (0..4) |f| {
            var alpha: [4]Iv = undefined;
            var peak: f64 = -ref.inf;
            for (0..4) |j| peak = @max(peak, ref.bf(self.fuse[f * 4 + j]));
            var denom: f64 = 0;
            for (0..4) |j| {
                alpha[j] = Iv.of(@exp(ref.bf(self.fuse[f * 4 + j]) - peak));
                denom += if (self.bymax) alpha[j].m / peak else alpha[j].m;
            }
            for (0..n) |row| {
                for (m, sqs, 0..) |*v, *q, d| {
                    v.* = Iv.of(0);
                    for (0..4) |j| v.* = v.*.add(alpha[j].mul(Iv.of(ref.bf(self.taps[(row * 4 + j) * hidden + d]))).div(Iv.of(denom)));
                    q.* = ref.sq(v.*);
                }
                const inv = ref.rms(sum.sum(sqs, depth(self.shape), scratch), hidden, eps);
                for (out[(f * n + row) * hidden ..][0..hidden], m, self.dflare[f * hidden ..][0..hidden]) |*o, v, w| o.* = v.mul(inv).mul(Iv.of(ref.bf(w))).bf();
            }
        }
        for (out[4 * n * hidden ..][0..hidden], out[(3 * n + n - 1) * hidden ..][0..hidden]) |*o, v| o.* = v;
    }
};

/// Context K (normed, RoPE'd) and V of rows at positions base.. into a layer's rings.
pub const Ctx = struct {
    kern: *const dspark.Kernels,
    /// Control: K stored unnormalized (k_norm skipped).
    raw: bool = false,
    shape: c.Shape = undefined,
    kv_in: []u16 = &.{},
    knorm: []u16 = &.{},
    ptr: [5]u64 = @splat(0),

    const kv = 1024;
    const hd = 128;
    const ring_rows = 2048;
    /// The fixture's ring base: rows land at positions 37...
    const base = 37;
    pub fn spec(self: *const Ctx) c.Spec {
        const served = &[_]c.Shape{ .{ .rows = 4, .n = kv, .k = hd }, .{ .rows = 64, .n = kv, .k = hd } };
        const name = if (self.raw) "control: inkling2 dspark tf_ink2_dsp_ctx K stored raw" else "inkling2 dspark tf_ink2_dsp_ctx (ctx k/v into the rings, k normed + rope)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .served = if (self.raw) served[0..1] else served };
    }
    pub fn control(self: *const Ctx) ?g.Gate {
        return if (self.raw) .shapes else null;
    }
    pub fn free(self: *Ctx, a: std.mem.Allocator) void {
        a.free(self.kv_in);
        a.free(self.knorm);
        self.kv_in = &.{};
        self.knorm = &.{};
    }
    /// `rows` rows of K and V at kv (8 heads of 128); ring at most rows past base, base under u32.
    pub fn prepare(self: *Ctx, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.n != kv or shape.k != hd or shape.rows == 0 or shape.rows > 64) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * kv;
        self.kv_in = try ref.bfs(a, rng, 2 * n, -1, 1);
        self.knorm = try ref.bfs(a, rng, hd, 0.5, 1.5);
        if (fill == .nonfinite) {
            self.kv_in[0] = c.toBf16(ref.nan);
            self.kv_in[kv] = c.toBf16(ref.inf);
            self.kv_in[n] = c.toBf16(-ref.inf);
        }
        self.ptr = .{ try ref.input(rig, self.kv_in[0..n]), try ref.input(rig, self.kv_in[n..]), try ref.input(rig, self.knorm), try rig.output(ring_rows * kv * 2), try rig.output(ring_rows * kv * 2) };
        return true;
    }
    pub fn run(self: *Ctx, rig: *rig_.Rig) !void {
        try self.kern.launchCtx(rig.s, self.ptr[0], self.ptr[1], self.ptr[2], self.ptr[3], self.ptr[4], base, self.shape.rows, ring_rows);
    }
    pub fn count(self: *const Ctx) usize {
        return 2 * self.shape.rows * kv;
    }
    /// The touched ring rows hold K normed+roped and V as read; the rest is whatever the output was.
    pub fn check(self: *const Ctx, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        _ = a;
        const n = self.shape.rows;
        var sq: [hd]f64 = undefined;
        for (0..n) |row| {
            for (0..8) |h| {
                var ss: f64 = 0;
                for (0..hd) |d| {
                    const x = ref.bf(self.kv_in[row * kv + h * hd + d]);
                    sq[d] = x * x;
                    ss += sq[d];
                }
                const scale = 1 / @sqrt(ss / hd + eps);
                const pos = base + row;
                const at = (@as(usize, @intCast(pos)) % ring_rows) * kv + h * hd;
                for (0..hd) |d| {
                    const j = d / 32;
                    const lane = d % 32;
                    const dim = lane + 32 * j;
                    const angle = @as(f64, @floatFromInt(pos)) * std.math.pow(f64, 1e6, -@as(f64, @floatFromInt(dim % 64)) / 64);
                    const x0 = ref.bf(self.kv_in[row * kv + h * hd + dim]) * scale * ref.bf(self.knorm[dim]);
                    const od = lane + 32 * ((j + 2) % 4);
                    const other = ref.bf(self.kv_in[row * kv + h * hd + od]) * scale * ref.bf(self.knorm[od]);
                    const want = if (self.raw) x0 else if (dim < 64) x0 * @cos(angle) - other * @sin(angle) else other * @sin(angle) + x0 * @cos(angle);
                    const gotk = ref.bf(@bitCast(@as(u16, std.mem.readInt(u16, got[2 * (at + dim) ..][0..2], .little))));
                    if (!(std.math.isNan(want) and std.math.isNan(gotk)) and @abs(gotk - want) > 0.02 * @abs(want) + 0.01)
                        return try std.fmt.bufPrint(msg, "row {d} head {d} dim {d}: ring k {e} where {e} is due", .{ row, h, dim, gotk, want });
                    const gotv = ref.bf(@bitCast(@as(u16, std.mem.readInt(u16, got[ring_rows * kv * 2 + 2 * (at + dim) ..][0..2], .little))));
                    const wantv = ref.bf(self.kv_in[n * kv + row * kv + h * hd + dim]);
                    if (!(std.math.isNan(wantv) and std.math.isNan(gotv)) and @abs(gotv - wantv) > 1e-3)
                        return try std.fmt.bufPrint(msg, "row {d} head {d} dim {d}: ring v {e} where {e} is due", .{ row, h, dim, gotv, wantv });
                }
            }
        }
        return null;
    }
    pub fn solve(_: *const Ctx, _: std.mem.Allocator, _: ref.Sums, _: []Iv) !void {}
};

/// Block row 0: the pending token's normed embedding; rows 1..7 the mask row. resid fp32 [8][H], x bf16 [8][H].
pub const Init = struct {
    kern: *const dspark.Kernels,
    /// Control: the embed norm's rsqrt read as sqrt.
    sqrt: bool = false,
    pending: u32 = 0,
    embed: []u16 = &.{},
    embed_norm: []u16 = &.{},
    mask: []u16 = &.{},
    norm: []u16 = &.{},
    ptr: [6]u64 = @splat(0),

    const rows_n = 8;
    pub fn spec(self: *const Init) c.Spec {
        const served = &[_]c.Shape{ .{ .rows = rows_n, .n = hidden, .k = 1 } };
        const name = if (self.sqrt) "control: inkling2 dspark tf_ink2_dsp_init embed norm squared" else "inkling2 dspark tf_ink2_dsp_init (pending's normed embed, masks, resid + x)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .served = served };
    }
    pub fn control(self: *const Init) ?g.Gate {
        return if (self.sqrt) .shapes else null;
    }
    pub fn free(self: *Init, a: std.mem.Allocator) void {
        a.free(self.embed);
        a.free(self.embed_norm);
        a.free(self.mask);
        a.free(self.norm);
        self.embed = &.{};
        self.embed_norm = &.{};
        self.mask = &.{};
        self.norm = &.{};
    }
    /// The fixture's own vocab slice; the pending id reads a row of it.
    pub fn prepare(self: *Init, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows != rows_n or shape.n != hidden or shape.k != 1) return false;
        self.free(a);
        const vocab_rows = 4096;
        self.pending = rng.uintLessThan(u32, vocab_rows);
        self.embed = try ref.bfs(a, rng, vocab_rows * hidden, -1, 1);
        self.embed_norm = try ref.bfs(a, rng, hidden, 0.5, 1.5);
        self.mask = try ref.bfs(a, rng, hidden, -1, 1);
        self.norm = try ref.bfs(a, rng, hidden, 0.5, 1.5);
        if (fill == .nonfinite) {
            self.embed[@as(usize, self.pending) * hidden] = c.toBf16(ref.nan);
            self.embed[@as(usize, self.pending) * hidden + 1] = c.toBf16(ref.inf);
            self.mask[2] = c.toBf16(-ref.inf);
        }
        self.ptr = .{ try ref.input(rig, std.mem.sliceAsBytes(&[_]u32{self.pending})), try ref.input(rig, self.embed), try ref.input(rig, self.embed_norm), try ref.input(rig, self.mask), try ref.input(rig, self.norm), try rig.output(rows_n * hidden * 4 + rows_n * hidden * 2) };
        return true;
    }
    pub fn run(self: *Init, rig: *rig_.Rig) !void {
        const resid = self.ptr[5];
        const x = self.ptr[5] + rows_n * hidden * 4;
        try self.kern.launchInit(rig.s, self.ptr[0], self.ptr[1], self.ptr[2], eps, self.ptr[3], self.ptr[4], resid, x);
    }
    pub fn count(_: *const Init) usize {
        return rows_n * hidden * 2;
    }
    /// resid keeps fp32 (raw); x holds the normed rows. The output layout is resid then x; the check splits it.
    pub fn check(self: *const Init, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const residb = got[0 .. rows_n * hidden * 4];
        const xb = got[rows_n * hidden * 4 ..];
        _ = a;
        var e: [hidden]f64 = undefined;
        for (0..rows_n) |row| {
            const src = if (row == 0) self.embed[@as(usize, self.pending) * hidden ..][0..hidden] else self.mask;
            var ss: f64 = 0;
            for (src, 0..) |v, d| {
                e[d] = ref.bf(v);
                ss += e[d] * e[d];
            }
            if (row == 0) {
                const scale = if (self.sqrt) @sqrt(ss / hidden + eps) else 1 / @sqrt(ss / hidden + eps);
                ss = 0;
                for (0..hidden) |d| {
                    e[d] = ref.bf(c.toBf16(@floatCast(e[d] * scale * ref.bf(self.embed_norm[d]))));
                    ss += e[d] * e[d];
                }
            }
            const scale = 1 / @sqrt(ss / hidden + eps);
            for (0..hidden) |d| {
                const resid = ref.readF32(residb, row * hidden + d);
                if (!ref.sameF32(resid, @floatCast(e[d])))
                    return try std.fmt.bufPrint(msg, "row {d} dim {d}: resid {e} where {e} is due", .{ row, d, resid, e[d] });
                const want = e[d] * scale * ref.bf(self.norm[d]);
                const gotx = ref.bf(@bitCast(@as(u16, std.mem.readInt(u16, xb[2 * (row * hidden + d) ..][0..2], .little))));
                if (!(std.math.isNan(want) and std.math.isNan(gotx)) and @abs(gotx - want) > 0.02 * @abs(want) + 0.01)
                    return try std.fmt.bufPrint(msg, "row {d} dim {d}: x {e} where {e} is due", .{ row, d, gotx, want });
            }
        }
        return null;
    }
    pub fn solve(_: *const Init, _: std.mem.Allocator, _: ref.Sums, _: []Iv) !void {}
};

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
        const served = &[_]c.Shape{ .{ .rows = 8, .n = q_width + kv_width, .k = hd } };
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
        const served = &[_]c.Shape{ .{ .rows = 8, .n = q_width, .k = 1050 } };
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
    /// A ring of 2048 rows with committed - from = 1050 valid rows (a local layer's window wraps once). The launch
    /// attends all eight block rows; the first `rows` are checked.
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
    /// out[row][h][d] = softmax(q·k/sqrt(128))·v over the last 1024 ring rows + 8 block rows; local layer. The device's
    /// weights are each within a relative `eps_w` of exp(s - max): twice the worst score error, the exps' fp32
    /// arguments (up to the score range) and an expf with a rescale per key. y = sum p v / sum p then moves at most
    /// grow = eps_w / (1 - eps_w) of sum p |v - y| / sum p, plus the fp32 chains of both sums and the division.
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
        const served = &[_]c.Shape{ .{ .rows = 7, .n = 3 * rank, .k = 2 * hidden } };
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
    /// y [8][H] (rows 1..7 feed the steps), newest [H], rnn [768][5120] bf16, bias [768]; the launch fills all seven
    /// steps, the first `rows` are checked.
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
            // NaN in step 0's y, +Inf in step 1's, -Inf in channel 0's newest weight: later steps' other channels stay
            // finite.
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
        const served = &[_]c.Shape{ .{ .rows = 1, .n = hidden, .k = rank } };
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

const groups = .{ "norms", "swiglus", "picks", "fuses", "ctxs", "inits", "qks", "attns", "kappas", "steps" };
pub const Set = struct {
    k: dspark.Kernels = undefined,
    loaded: bool = false,
    norms: [2]ref.Gated(AddNorm) = undefined,
    swiglus: [2]ref.Gated(Swiglu) = undefined,
    picks: [2]ref.Checked(Pick) = undefined,
    fuses: [2]ref.Gated(Fuse) = undefined,
    ctxs: [2]ref.Checked(Ctx) = undefined,
    inits: [2]ref.Checked(Init) = undefined,
    qks: [2]ref.Checked(Qk) = undefined,
    attns: [2]ref.Gated(Attn) = undefined,
    kappas: [2]ref.Gated(Kappa) = undefined,
    steps: [2]ref.Checked(Step) = undefined,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try dspark.Kernels.load(gpu.d);
        set.loaded = true;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&set.norms, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .stale = i == 1 } };
        for (&set.swiglus, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .swapped = i == 1 } };
        for (&set.picks, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .short = i == 1 } };
        for (&set.fuses, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .bymax = i == 1 } };
        for (&set.ctxs, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .raw = i == 1 } };
        for (&set.inits, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .sqrt = i == 1 } };
        for (&set.qks, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .twice = i == 1 } };
        for (&set.attns, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .mean = i == 1 } };
        for (&set.kappas, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .bias2 = i == 1 } };
        for (&set.steps, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .flipped = i == 1 } };
        inline for (groups) |field| for (&@field(set, field)) |*x| try list.append(a, x.kernel());
    }
    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        inline for (groups) |field| for (&@field(set, field)) |*x| x.deinit();
        set.k.unload();
        set.loaded = false;
    }
};
