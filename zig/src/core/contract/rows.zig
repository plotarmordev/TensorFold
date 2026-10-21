//! Row operations for the contract (RMS norm, softmax, sums, dots, maxima): inputs, float64 references, simulated sums.
const std = @import("std");
const c = @import("contract.zig");
const norms_ref = @import("../norms_ref.zig");

/// What a gate asks the inputs to hold: ordinary values, NaN and infinities at known places, or sums that test fp32.
pub const Fill = enum { random, nonfinite, stress };

/// The order a simulated accumulator adds terms in: one chain, contiguous halves, or 32 lanes then their halves.
pub const Order = enum { sequential, pairwise, strided };

/// add_rms: rms of the residual bf16(x + z) with gamma 1 + w, after the residual itself; gated: rms(x) w silu(z).
pub const Op = enum { rms, softmax, sum, dot, max, add_rms, gated };

/// One row's terms summed in `acc` (each add rounded to it) in `order`.
pub fn sumIn(acc: c.Accum, order: Order, t: []const f64) f64 {
    return switch (order) {
        .sequential => chain(acc, t, 1, 0),
        .pairwise => halves(acc, t),
        .strided => blk: {
            var lanes: [32]f64 = @splat(0);
            for (&lanes, 0..) |*l, i| l.* = chain(acc, t, 32, i);
            break :blk halves(acc, &lanes);
        },
    };
}

fn round(acc: c.Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
        .none => v,
    };
}

fn chain(acc: c.Accum, t: []const f64, stride: usize, first: usize) f64 {
    var s: f64 = 0;
    var i = first;
    while (i < t.len) : (i += stride) s = round(acc, s + round(acc, t[i]));
    return s;
}

fn halves(acc: c.Accum, t: []const f64) f64 {
    if (t.len == 0) return 0;
    if (t.len == 1) return round(acc, t[0]);
    const mid = t.len / 2;
    return round(acc, halves(acc, t[0..mid]) + halves(acc, t[mid..]));
}

/// A row op's host inputs: x [rows, k], w [k] and (add_rms, gated) z [rows, k] as bf16 bits; outputs per `outputs`.
pub const Rows = struct {
    a: std.mem.Allocator,
    op: Op,
    shape: c.Shape,
    x: []u16,
    w: []u16,
    z: []u16,
    eps: f32 = 1e-6,

    pub fn init(a: std.mem.Allocator, op: Op, shape: c.Shape) !Rows {
        const x = try a.alloc(u16, shape.rows * shape.k);
        errdefer a.free(x);
        const w = try a.alloc(u16, shape.k);
        errdefer a.free(w);
        const z = try a.alloc(u16, if (op == .add_rms or op == .gated) shape.rows * shape.k else 0);
        return .{ .a = a, .op = op, .shape = shape, .x = x, .w = w, .z = z };
    }

    pub fn deinit(r: Rows) void {
        r.a.free(r.x);
        r.a.free(r.w);
        r.a.free(r.z);
    }

    /// rms, gated and softmax write k outputs a row, add_rms the residual rows then the norm rows, the rest one.
    pub fn outputs(r: Rows) usize {
        return r.shape.rows * switch (r.op) {
            .rms, .gated, .softmax => r.shape.k,
            .add_rms => 2 * r.shape.k,
            .sum, .dot, .max => 1,
        };
    }

    pub fn fill(r: Rows, mode: Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = r.shape.k;
        for (r.w) |*v| v.* = c.toBf16(switch (r.op) {
            .dot => rng.float(f32) * 2 - 1,
            .add_rms => rng.float(f32) - 0.5,
            else => 0.5 + rng.float(f32),
        });
        for (r.x, 0..) |*v, i| v.* = c.toBf16(switch (r.op) {
            .softmax => rng.float(f32) * 16 - 8,
            else => (rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1),
        });
        for (r.z) |*v| v.* = c.toBf16(if (r.op == .gated) rng.float(f32) * 8 - 4 else rng.float(f32) * 2 - 1);
        if (mode == .stress) return r.stress(rng);
        if (mode != .nonfinite) return;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
        const put = [_]u16{ nan, inf, ninf, inf, ninf };
        for (at, put) |p, v| if (p[0] < r.shape.rows) {
            r.x[p[0] * k + p[1]] = v;
        };
    }

    /// Gate 4's rows: integer terms, the second half mirroring the first (8 nudged by one): big partials, small result.
    fn stress(r: Rows, rng: std.Random) void {
        const k = r.shape.k;
        const h = k / 2;
        const norm = r.op == .rms or r.op == .add_rms;
        if (r.op == .gated) {
            // squares of 128 to 255: four of them pass fp16's range; a 128-wide head's sum stays below 2^24
            for (r.x) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(128 + rng.uintLessThan(u32, 128))) * @as(f32, if (rng.boolean()) 1 else -1));
            return;
        }
        @memset(r.z, 0);
        for (0..r.shape.rows) |row| {
            const x = r.x[row * k ..][0..k];
            for (0..h) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                const nudge: f32 = if (i < 8) 1 else 0;
                x[i] = c.toBf16(if (norm) v - 32 else v);
                x[h + i] = c.toBf16(switch (r.op) {
                    .sum => nudge - v,
                    .rms, .add_rms => @as(f32, @floatFromInt(rng.uintLessThan(u32, 65))) - 32,
                    else => v + nudge,
                });
            }
            if (k % 2 == 1) x[k - 1] = 0;
        }
        if (r.op != .dot) return;
        for (0..h) |i| {
            const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            r.w[i] = c.toBf16(w);
            r.w[h + i] = c.toBf16(-w);
        }
        if (k % 2 == 1) r.w[k - 1] = 0;
    }

    /// x, or the residual bf16(x + z) add_rms normalizes.
    fn xv(r: Rows, row: usize, i: usize) f64 {
        const x = c.fromBf16(r.x[row * r.shape.k + i]);
        if (r.op != .add_rms) return x;
        return c.fromBf16(c.toBf16(@floatCast(x + c.fromBf16(r.z[row * r.shape.k + i]))));
    }

    /// w, or add_rms's zero-centred gamma 1 + w.
    fn wv(r: Rows, i: usize) f64 {
        const w = c.fromBf16(r.w[i]);
        return if (r.op == .add_rms) 1 + w else w;
    }

    /// gated's silu(z) for output i of row `row`.
    fn gate(r: Rows, row: usize, i: usize) f64 {
        const z = c.fromBf16(r.z[row * r.shape.k + i]);
        return z / (1 + @exp(-z));
    }

    /// Where row `row`'s normed outputs start: add_rms writes all the residual rows first.
    fn normAt(r: Rows, row: usize) usize {
        return (if (r.op == .add_rms) r.shape.rows + row else row) * r.shape.k;
    }

    /// Row `row`'s summed terms (rms: squares, softmax: exp(x - max), dot: x w, sum: x) into `t`; max has none.
    fn terms(r: Rows, row: usize, t: []f64) f64 {
        var top: f64 = -std.math.floatMax(f64);
        if (r.op == .softmax) for (0..r.shape.k) |i| {
            const v = r.xv(row, i);
            if (std.math.isNan(v) or v > top) top = v;
        };
        for (t, 0..) |*o, i| {
            const v = r.xv(row, i);
            o.* = switch (r.op) {
                .rms, .add_rms, .gated => v * v,
                .softmax => @exp(v - top),
                .dot => v * r.wv(i),
                .sum => v,
                .max => 0,
            };
        }
        return top;
    }

    /// Float64 outputs and, per output, the magnitude whose gamma(depth) bounds the error the sums carry into it.
    pub fn reference(r: Rows, want: []f64, mag: []f64) !void {
        const k = r.shape.k;
        const t = try r.a.alloc(f64, k);
        defer r.a.free(t);
        for (0..r.shape.rows) |row| {
            const top = r.terms(row, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            switch (r.op) {
                .rms, .add_rms, .gated => {
                    const xs = try r.a.alloc(f64, k);
                    defer r.a.free(xs);
                    const ws = try r.a.alloc(f64, k);
                    defer r.a.free(ws);
                    for (xs, ws, 0..) |*x, *w, i| {
                        x.* = r.xv(row, i);
                        w.* = r.wv(i);
                    }
                    const y = want[r.normAt(row)..][0..k];
                    norms_ref.rms(xs, ws, r.eps, y);
                    if (r.op == .gated) for (y, 0..) |*v, i| {
                        v.* *= r.gate(row, i);
                    };
                    for (mag[r.normAt(row)..][0..k], y) |*m, v| m.* = @abs(v) / 2; // y ~ ss^-1/2
                    if (r.op == .add_rms) {
                        @memcpy(want[row * k ..][0..k], xs);
                        @memset(mag[row * k ..][0..k], 0);
                    }
                },
                .softmax => for (0..k) |i| {
                    want[row * k + i] = @exp(r.xv(row, i) - top) / sum;
                    mag[row * k + i] = @abs(want[row * k + i]);
                },
                .sum, .dot => {
                    want[row] = sum;
                    mag[row] = abs;
                },
                .max => {
                    var m: f64 = -std.math.inf(f64);
                    for (0..k) |i| {
                        const v = r.xv(row, i);
                        if (std.math.isNan(v) or std.math.isNan(m)) m = std.math.nan(f64) else m = @max(m, v);
                    }
                    want[row] = m;
                    mag[row] = 0;
                },
            }
        }
    }

    /// The outputs if every row's sums ran in `acc` in `order`, the rest of the arithmetic exact.
    pub fn simulate(r: Rows, acc: c.Accum, order: Order, out: []f64) !void {
        const k = r.shape.k;
        const t = try r.a.alloc(f64, k);
        defer r.a.free(t);
        for (0..r.shape.rows) |row| {
            const top = r.terms(row, t);
            const s = sumIn(acc, order, t);
            switch (r.op) {
                .rms, .add_rms, .gated => {
                    const inv = 1 / @sqrt(s / @as(f64, @floatFromInt(k)) + r.eps);
                    for (0..k) |i| out[r.normAt(row) + i] = r.xv(row, i) * inv * r.wv(i) * (if (r.op == .gated) r.gate(row, i) else 1);
                    if (r.op == .add_rms) for (0..k) |i| {
                        out[row * k + i] = r.xv(row, i);
                    };
                },
                .softmax => for (0..k) |i| {
                    out[row * k + i] = @exp(r.xv(row, i) - top) / s;
                },
                .sum, .dot => out[row] = s,
                .max => return error.NoSums,
            }
        }
    }
};

test "the stress rows: fp32 sums exact in every order, fp16 and bf16 sums far off a small result" {
    const a = std.testing.allocator;
    for ([_]Op{ .dot, .sum }) |op| {
        const r = try Rows.init(a, op, .{ .rows = 3, .n = 1, .k = 4096 });
        defer r.deinit();
        r.fill(.stress, 3);
        var want: [3]f64 = undefined;
        var mag: [3]f64 = undefined;
        try r.reference(&want, &mag);
        var got: [3]f64 = undefined;
        for ([_]Order{ .sequential, .pairwise, .strided }) |o| {
            try r.simulate(.f32, o, &got);
            try std.testing.expectEqualSlices(f64, &want, &got);
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                try r.simulate(acc, o, &got);
                var most: f64 = 0; // the worst row's miss in half bf16 ulps of its result
                for (got, want) |g, w| most = @max(most, if (std.math.isFinite(g)) @abs(g - w) / (0x1p-8 * @abs(w) + 0x1p-133) else std.math.inf(f64));
                try std.testing.expect(most >= 100);
            }
        }
        for (want) |w| try std.testing.expect(@abs(w) <= 8 * 7 and w != 0 or op == .sum);
        try std.testing.expect(mag[0] > 0x1p17 and mag[0] < 0x1p23);
    }
}

test "planted non-finites reach the reference: NaN, both infinities, and an infinity cancelling to NaN" {
    const a = std.testing.allocator;
    const r = try Rows.init(a, .max, .{ .rows = 5, .n = 1, .k = 96 });
    defer r.deinit();
    r.fill(.nonfinite, 9);
    var want: [5]f64 = undefined;
    var mag: [5]f64 = undefined;
    try r.reference(&want, &mag);
    try std.testing.expectEqual(c.Class.nan, c.classOf(want[0]));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[1]));
    try std.testing.expectEqual(c.Class.finite, c.classOf(want[2]));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[3]));
    try std.testing.expectEqual(c.Class.finite, c.classOf(want[4]));
}

test "the norm stress rows: fp32 sums exact in every order, fp16 sums past their range" {
    const a = std.testing.allocator;
    for ([_]Op{ .rms, .add_rms, .gated }, [_]usize{ 2048, 2048, 128 }) |op, k| {
        const r = try Rows.init(a, op, .{ .rows = 4, .n = k, .k = k });
        defer r.deinit();
        r.fill(.stress, 5);
        const n = r.outputs();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        try r.reference(buf[0..n], buf[n .. 2 * n]);
        for ([_]Order{ .sequential, .pairwise, .strided }) |o| {
            try r.simulate(.f32, o, buf[2 * n ..]);
            try std.testing.expectEqualSlices(f64, buf[0..n], buf[2 * n ..]);
            try r.simulate(.f16, o, buf[2 * n ..]);
            var most: f64 = 0; // the worst output's miss in half bf16 ulps of its value
            for (buf[2 * n ..], buf[0..n]) |g, w| most = @max(most, @abs(g - w) / (0x1p-8 * @abs(w) + 0x1p-133));
            try std.testing.expect(most >= 100);
        }
    }
}

test "the norm rows' non-finites: a NaN row, NaN where an infinity stands and zeros beside it" {
    const a = std.testing.allocator;
    const k = 128;
    for ([_]Op{ .add_rms, .gated }) |op| {
        const r = try Rows.init(a, op, .{ .rows = 5, .n = k, .k = k });
        defer r.deinit();
        r.fill(.nonfinite, 9);
        const want = try a.alloc(f64, 2 * r.outputs());
        defer a.free(want);
        try r.reference(want[0..r.outputs()], want[r.outputs()..]);
        const norm = struct {
            fn at(x: Rows, w: []const f64, row: usize, i: usize) c.Class {
                return c.classOf(w[x.normAt(row) + i]);
            }
        }.at;
        for (0..k) |i| try std.testing.expectEqual(c.Class.nan, norm(r, want, 0, i));
        try std.testing.expectEqual(c.Class.nan, norm(r, want, 1, k / 2));
        try std.testing.expectEqual(@as(f64, 0), @abs(want[r.normAt(1)]));
        for (0..k) |i| try std.testing.expectEqual(c.Class.finite, norm(r, want, 4, i));
        if (op == .add_rms) try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[k + k / 2]));
    }
}
