//! h3d_gemm under the contract: y = bf16(x w^T + b), its LayerScale form y += bf16(that s), and w1's SwiGLU form
//! y = bf16(bf16(SiLU(gate)) * value) over [gate n | value n] weight rows, against float64.
const std = @import("std");
const mtl = @import("metal");
const parts = @import("metal_h3_image_decoder_parts.zig");

const c = parts.c;
const Range = parts.Range;
const Ends = parts.Ends;
const Sum = parts.Sum;
const bf = parts.bf;
const value = parts.value;
const W = parts.W;
const F = parts.F;

/// Rows a pass multiplies: one tile of 16 x 16 cells and its 5 tokens after, the batch of 16, one small tile of 65.
const passes = [_]usize{ 261, 16 * 261, 65 };

fn served(comptime nk: []const [2]usize) [passes.len * nk.len]c.Shape {
    var out: [passes.len * nk.len]c.Shape = undefined;
    for (passes, 0..) |r, i| for (nk, 0..) |p, j| {
        out[i * nk.len + j] = .{ .rows = r, .n = p[0], .k = p[1] };
    };
    return out;
}

/// to_qkv (3,072 x 1,024), w1 without tensor units (8,192 x 1,024) and proj_out (768 x 1,024).
const plain = served(&.{ .{ 3 * W, W }, .{ 2 * F, W }, .{ 768, W } });
/// to_out (1,024 x 1,024) and w2 (1,024 x 4,096), each added to the stream through its LayerScale.
const scaled = served(&.{ .{ W, W }, .{ W, F } });
/// w1 on tensor units: 4,096 outputs from 8,192 weight rows over 1,024 inputs.
const gated = served(&.{.{ F, W }});

/// The k-deep chain of the products' fp32 sums (exact bf16 products either way), then the bias.
fn chain(s: c.Shape) usize {
    return s.k + 1;
}

/// float64 x . w and sum |x_i w_i| over fp32 copies of bf16 values.
fn dot(x: []const f32, w: []const f32) [2]f64 {
    var s: @Vector(8, f64) = @splat(0);
    var m: @Vector(8, f64) = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const a: @Vector(8, f64) = @floatCast(@as(@Vector(8, f32), x[i..][0..8].*));
        const b: @Vector(8, f64) = @floatCast(@as(@Vector(8, f32), w[i..][0..8].*));
        s += a * b;
        m += @abs(a * b);
    }
    var sv = @reduce(.Add, s);
    var mv = @reduce(.Add, m);
    while (i < x.len) : (i += 1) {
        const p = @as(f64, x[i]) * w[i];
        sv += p;
        mv += @abs(p);
    }
    return .{ sv, mv };
}

fn widen(a: std.mem.Allocator, b: []const u16) ![]f32 {
    const out = try a.alloc(f32, b.len);
    for (out, b) |*o, v| o.* = c.fromBf16(v);
    return out;
}

/// The kernel's epilogue modes, in order.
pub const Form = enum { plain, layer_scale, swiglu };

/// Where SiLU turns: its minimum, about -0.2785 at x = -1.2785.
const silu_turn = -1.2784645427610738;

/// SiLU over every gate in `g` (the kernel's `gv` when it holds one value or a non-finite end).
fn siluOver(g: Range, gv: f64) Range {
    if (g.lo == g.hi or !std.math.isFinite(g.lo) or !std.math.isFinite(g.hi)) return parts.silu(gv);
    const a = parts.silu(g.lo);
    const z = parts.silu(g.hi);
    var r: Range = .{ .lo = @min(a.lo, z.lo), .hi = @max(a.hi, z.hi) };
    if (g.lo < silu_turn and silu_turn < g.hi) r.lo = @min(r.lo, parts.silu(silu_turn).lo);
    return r;
}

pub const Gemm = struct {
    /// The epilogue: kernel mode 0, 1 (LayerScale: y = bf16(y + bf16(bf16(x w^T + b) s)), in place) or 2 (SwiGLU).
    form: Form,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []u16 = &.{},
    s: []u16 = &.{},
    y0: []u16 = &.{},

    /// Rows: rows; n: the projection's outputs (64 a tile); k: its inputs (32 a step).
    pub fn spec(name: []const u8, k: Gemm) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = chain,
            .ops = if (k.form == .swiglu) 2 * (parts.exp_ulps + parts.div_ulps) + 2 else 0,
            .mids = switch (k.form) {
                .plain => 0,
                .layer_scale, .swiglu => 2,
            },
            .out = .bf16,
            .in_place = k.form == .layer_scale,
            .tiles = .{ .{ .size = 64 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 32, .step = 32, .min = 32 } },
            .served = switch (k.form) {
                .plain => &plain,
                .layer_scale => &scaled,
                .swiglu => &gated,
            },
        };
    }

    /// Gate 4's rows: integers, k's second half repeating the first, its weights negated plus 1 on 4: every sum is 4.
    fn mirror(k: *Gemm, rng: std.Random, xb: []u16, wb: []u16, s: c.Shape) void {
        const h = s.k / 2;
        for (0..s.rows) |r| for (0..h) |i| {
            const v = if (i < 4) 1 else rng.intRangeAtMost(i32, 1, 48);
            xb[r * s.k + i] = parts.integer(v);
            xb[r * s.k + h + i] = parts.integer(v);
        };
        for (0..s.n) |o| for (0..h) |i| {
            const v = rng.intRangeAtMost(i32, 0, 48);
            wb[o * s.k + i] = parts.integer(v);
            wb[o * s.k + h + i] = parts.integer(-v + @as(i32, if (i < 4) 1 else 0));
        };
        @memset(k.b, 0);
        @memset(k.s, parts.integer(2));
        parts.integers(rng, k.y0, -8, 8);
    }

    pub fn prepare(k: *Gemm, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        if (s.rows == 0 or s.n == 0 or s.n % 64 != 0 or s.k == 0 or s.k % 32 != 0) return false;
        const al = t.host();
        const wrows = k.weightRows(s);
        const xb = try al.alloc(u16, s.rows * s.k);
        const wb = try al.alloc(u16, wrows * s.k);
        k.b = try al.alloc(u16, wrows);
        k.s = try al.alloc(u16, s.n);
        k.y0 = try al.alloc(u16, s.rows * s.n);
        if (t.fill == .stress) k.mirror(rng, xb, wb, .{ .rows = s.rows, .n = wrows, .k = s.k }) else {
            parts.uniform(rng, xb, -2, 2);
            const a: f32 = @floatCast(@sqrt(3.0 / @as(f64, @floatFromInt(s.k))));
            parts.uniform(rng, wb, -a, a);
            parts.uniform(rng, k.b, -0.1, 0.1);
            parts.uniform(rng, k.s, 0.1, 0.5);
            parts.uniform(rng, k.y0, -2, 2);
        }
        if (t.fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, s.k / 3 }, .{ 1, s.k / 2 }, .{ 2, s.k - 1 }, .{ 3, 0 }, .{ 3, s.k - 1 } };
            for (at, parts.plants) |p, v| if (p[0] < s.rows) {
                xb[p[0] * s.k + p[1]] = v;
            };
            // the LayerScale form's old stream value and scale too
            if (k.form == .layer_scale and s.rows > 4) k.y0[4 * s.n + 3] = parts.nan16;
            if (k.form == .layer_scale) k.s[5] = parts.inf16;
        }
        k.x = try widen(al, xb);
        k.w = try widen(al, wb);
        try t.input(xb);
        try t.input(wb);
        try t.input(k.b);
        if (k.form == .layer_scale) try t.input(k.s);
        try t.output(s.rows * s.n * 2);
        return true;
    }

    /// Weight rows the product reads: n, or the SwiGLU form's [gate n | value n].
    fn weightRows(k: *const Gemm, s: c.Shape) usize {
        return if (k.form == .swiglu) 2 * s.n else s.n;
    }

    /// The launcher's gemm: 64 x 64 tiles a group of 128; modes 0 and 2 bind the bias again where the scale goes.
    pub fn launch(k: *Gemm, t: anytype, e: mtl.ComputeEncoder) void {
        const s = t.shape;
        const scaled_form = k.form == .layer_scale;
        const out: usize = if (scaled_form) 4 else 3;
        // the stream the LayerScale form adds into, fresh before every run
        if (scaled_form) @memcpy(t.rig.bytes(t.bufs[out], k.y0.len * 2), std.mem.sliceAsBytes(k.y0));
        for (0..3) |i| t.bind(e, i, i);
        t.bind(e, if (scaled_form) 3 else 2, 3);
        t.bind(e, out, 4);
        e.setValue(parts.Shape{ .m = @intCast(s.rows), .n = @intCast(s.n), .k = @intCast(s.k), .mode = @backingInt(k.form) }, 5);
        e.dispatchGroups(mtl.Size.of(s.n / 64, (s.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    }

    pub fn count(_: *Gemm, s: c.Shape) usize {
        return s.rows * s.n;
    }

    /// Output (r, o) from its sums (the SwiGLU form's value row second): the bias and one rounding, then for
    /// LayerScale the scale and the old value, for SwiGLU the rounded SiLU of the gate times the value.
    fn finish(k: *const Gemm, s: c.Shape, sums: [2]Sum.Out, r: usize, o: usize, ends: *Ends, want: []f64, mag: []f64) void {
        const at = r * s.n + o;
        const b = value(k.b[o]);
        const v = bf(sums[0].v + b);
        const range = sums[0].r.add(Range.of(b)).fp32(0.5).bf16();
        switch (k.form) {
            .plain => ends.put(want, mag, at, v, range),
            .layer_scale => {
                const sc = value(k.s[o]);
                const old = value(k.y0[at]);
                const y = range.mul(Range.of(sc)).fp32(0.5).bf16().add(Range.of(old)).fp32(0.5).bf16();
                ends.put(want, mag, at, bf(old + bf(v * sc)), y);
            },
            .swiglu => {
                const vb = value(k.b[s.n + o]);
                const vv = bf(sums[1].v + vb);
                const vr = sums[1].r.add(Range.of(vb)).fp32(0.5).bf16();
                const y = siluOver(range, v).bf16().mul(vr).fp32(0.5).bf16();
                ends.put(want, mag, at, bf(bf(v / (1 + @exp(-v))) * vv), y);
            },
        }
    }

    const Job = struct {
        k: *const Gemm,
        s: c.Shape,
        sum: Sum,
        ends: *Ends,
        want: []f64,
        mag: []f64,

        fn rows(j: Job, from: usize, to: usize) void {
            const kk = j.s.k;
            const x = j.k.x;
            const w = j.k.w;
            for (from..to) |r| for (0..j.s.n) |o| {
                var sums: [2]Sum.Out = undefined;
                for (0..@divExact(j.k.weightRows(j.s), j.s.n)) |h| {
                    const d = dot(x[r * kk ..][0..kk], w[(h * j.s.n + o) * kk ..][0..kk]);
                    sums[h] = .{ .v = d[0], .r = j.sum.spread(d[0], d[1], kk) };
                }
                j.k.finish(j.s, sums, r, o, j.ends, j.want, j.mag);
            };
        }
    };

    pub fn outputs(k: *Gemm, t: anytype, sum: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        const s = t.shape;
        if (sum.acc == null) return parts.parallel(s.rows, Job{ .k = k, .s = s, .sum = sum, .ends = ends, .want = want, .mag = mag }, Job.rows);
        // gate 4's replays: each output's terms in the accumulator and order asked
        const terms = try t.a.alloc(f64, s.k);
        defer t.a.free(terms);
        for (0..s.rows) |r| for (0..s.n) |o| {
            var sums: [2]Sum.Out = undefined;
            for (0..@divExact(k.weightRows(s), s.n)) |h| {
                for (terms, k.x[r * s.k ..][0..s.k], k.w[(h * s.n + o) * s.k ..][0..s.k]) |*p, a, b| p.* = @as(f64, a) * b;
                sums[h] = sum.of(terms, s.k);
            }
            k.finish(s, sums, r, o, ends, want, mag);
        };
    }
};
