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

/// Each draft step caps FP32 logits in place before selecting the first-id maximum.
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
    /// Tied and unique maxima across draft steps expose a truncated vocabulary scan.
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

/// Fuse four taps by softmax, normalize with dflare, and expose the newest fused row.
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
            // Integer taps stress an exactly representable fp32 reduction.
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
        const served = &[_]c.Shape{.{ .rows = rows_n, .n = hidden, .k = 1 }};
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
