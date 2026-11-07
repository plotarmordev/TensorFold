//! Woof's dense CPU arithmetic: weights in panels of 16 outputs, four rows a panel or one row four panels.
const std = @import("std");

const lanes = 16;
const V = @Vector(lanes, f32);

/// y = W x (+ b) for W (out, in), stored as ceil(out / 16) panels of (in, 16), the last zero-padded.
pub const Linear = struct {
    w: []const f32,
    b: ?[]const f32 = null,
    out: usize,
    in: usize,

    /// Packs W[o][i] = src.get(o, i).
    pub fn pack(a: std.mem.Allocator, out: usize, in: usize, src: anytype, b: ?[]const f32) !Linear {
        const w = try a.alloc(f32, panels(out) * in * lanes);
        for (0..panels(out)) |p| for (0..in) |i| for (0..lanes) |j| {
            const o = p * lanes + j;
            w[(p * in + i) * lanes + j] = if (o < out) src.get(o, i) else 0;
        };
        return .{ .w = w, .b = b, .out = out, .in = in };
    }
};

/// A pack source: W[o][i] = m[o * so + i * si].
pub const Strided = struct {
    m: []const f32,
    so: usize,
    si: usize,

    pub fn get(s: Strided, o: usize, i: usize) f32 {
        return s.m[o * s.so + i * s.si];
    }
};

fn panels(out: usize) usize {
    return std.math.divCeil(usize, out, lanes) catch unreachable;
}

/// y[r][o] = x[r * x_stride ..][0..in] . W[o] (+ b[o]) (+ res[r][o]) for r < n, rows y_stride apart; res may be y.
pub fn rows(l: Linear, x: []const f32, x_stride: usize, n: usize, y: []f32, y_stride: usize, res: ?[]const f32) void {
    var r: usize = 0;
    while (r + 4 <= n) : (r += 4) for (0..panels(l.out)) |p| tile(4, 1, l, x, x_stride, r, p, y, y_stride, res);
    while (r < n) : (r += 1) {
        var p: usize = 0;
        while (p + 4 <= panels(l.out)) : (p += 4) tile(1, 4, l, x, x_stride, r, p, y, y_stride, res);
        while (p < panels(l.out)) : (p += 1) tile(1, 1, l, x, x_stride, r, p, y, y_stride, res);
    }
}

/// Rows r0.. (R of them) times panels p0.. (P of them).
inline fn tile(comptime R: usize, comptime P: usize, l: Linear, x: []const f32, x_stride: usize, r0: usize, p0: usize, y: []f32, y_stride: usize, res: ?[]const f32) void {
    const zero: V = @splat(0);
    var acc: [R][P]V = @splat(@splat(zero));
    for (0..l.in) |i| inline for (0..P) |q| {
        const w: V = l.w[((p0 + q) * l.in + i) * lanes ..][0..lanes].*;
        inline for (0..R) |k| acc[k][q] += @as(V, @splat(x[(r0 + k) * x_stride + i])) * w;
    };
    inline for (0..R) |k| inline for (0..P) |q| {
        const v: [lanes]f32 = acc[k][q];
        for (v, (p0 + q) * lanes..) |s, o| {
            if (o >= l.out) break;
            const at = (r0 + k) * y_stride + o;
            y[at] = s + (if (l.b) |b| b[o] else 0) + if (res) |e| e[at] else 0;
        }
    };
}

/// y = W x (+ b).
pub fn vec(l: Linear, x: []const f32, y: []f32) void {
    rows(l, x, l.in, 1, y, l.out, null);
}

pub fn dot(a: []const f32, b: []const f32) f32 {
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= a.len) : (i += lanes) acc += @as(V, a[i..][0..lanes].*) * @as(V, b[i..][0..lanes].*);
    var s = @reduce(.Add, acc);
    for (i..a.len) |j| s += a[j] * b[j];
    return s;
}

/// 1 / sqrt(mean(x^2) + eps).
pub fn rmsInv(x: []const f32, eps: f32) f32 {
    return 1 / @sqrt(dot(x, x) / @as(f32, @floatFromInt(x.len)) + eps);
}

/// (x - mean) / sqrt(var + eps), in place.
pub fn layerNorm(x: []f32, eps: f32) void {
    const n: f32 = @floatFromInt(x.len);
    var mu: f32 = 0;
    for (x) |v| mu += v;
    mu /= n;
    for (x) |*v| v.* -= mu;
    const inv = 1 / @sqrt(dot(x, x) / n + eps);
    for (x) |*v| v.* *= inv;
}

pub fn silu(x: f32) f32 {
    return x / (1 + @exp(-x));
}

/// Softmax over s in place.
pub fn softmax(s: []f32) void {
    var m: f32 = -std.math.inf(f32);
    for (s) |v| m = @max(m, v);
    var sum: f32 = 0;
    for (s) |*v| {
        v.* = @exp(v.* - m);
        sum += v.*;
    }
    for (s) |*v| v.* /= sum;
}

/// Rotate-half rope on one head: x[:h] = x[:h] cos - x[h:] sin, x[h:] = x[h:] cos + x[:h] sin.
pub fn rope(x: []f32, cos: []const f32, sin: []const f32) void {
    const h = x.len / 2;
    for (0..h) |j| {
        const a = x[j];
        const b = x[j + h];
        x[j] = a * cos[j] - b * sin[j];
        x[j + h] = b * cos[j] + a * sin[j];
    }
}

test "rows: packed panels equal plain dots over four-row blocks, tail rows and a padded panel" {
    const out = 37;
    const in = 5;
    var w: [out * in]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) - 3;
    var bias: [out]f32 = undefined;
    for (&bias, 0..) |*v, i| v.* = @floatFromInt(i);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const l = try Linear.pack(arena.allocator(), out, in, Strided{ .m = &w, .so = in, .si = 1 }, &bias);
    var x: [6 * 8]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.5;
    var y: [6 * 40]f32 = undefined;
    for (&y, 0..) |*v, i| v.* = @floatFromInt(i);
    rows(l, &x, 8, 6, &y, 40, &y);
    for (0..6) |r| for (0..40) |o| {
        const at: f32 = @floatFromInt(r * 40 + o);
        if (o >= out) {
            try std.testing.expectEqual(at, y[r * 40 + o]);
            continue;
        }
        var s: f32 = 0;
        for (0..in) |i| s += x[r * 8 + i] * w[o * in + i];
        try std.testing.expectApproxEqAbs(s + bias[o] + at, y[r * 40 + o], 1e-4);
    };
}
