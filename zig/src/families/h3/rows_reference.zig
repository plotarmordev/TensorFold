//! The Metal H3 row kernels as host functions: the stream's rows in and out, the final norm and the
//! AdaLN projections, for the backends that run the blocks another way (CUDA) or a check without a GPU.
//! Accumulation follows each kernel's float32 order: sequential for h3_rows_in, the 32 lane sums then
//! combined for the strided h3_rows_out and h3_modulation.
const std = @import("std");
const bf16 = @import("bf16_round");
/// The AdaLN shapes the projections share; `pub` for the tools that fetch the embedder's width.
pub const shape = @import("modulation_shape.zig");

/// A modulated value, rounded to bf16 where the reference's bf16 arithmetic rounds (h3_norm's h3_mod).
fn modulated(x: f32, inv: f32, w: f32, scale: f32, shift: f32) f32 {
    return bf16.round32(bf16.round32(bf16.round32(x * inv * w) * bf16.round32(1.0 + bf16.round32(scale))) + bf16.round32(shift));
}

/// Each output value's partial sums: one lane's sequential share over strided elements, then combined.
fn sum32(at: anytype, w: []align(1) const f32, w0: usize, k: usize) f32 {
    var partials: [32]f32 = @splat(0);
    for (0..k) |j| partials[j % 32] += at.f(j) * w[w0 + j];
    var sum: f32 = 0;
    for (partials) |p| sum += p;
    return sum;
}

/// h3_rows_in: y[row0 + r] = bf16(sum over k of x[r, k] w[n, k] + b[n]), summed one k at a time in f32.
/// `x` and `w` are f32, `y` bf16 bits; w and b are `out_w` rows of `width` (n, k).
pub fn rowsIn(x: []align(1) const f32, w: []align(1) const f32, b: []align(1) const f32, y: []u16, row0: usize, count: usize, width: usize, out_w: usize) void {
    for (0..count) |r| for (0..out_w) |n| {
        var sum: f32 = 0;
        for (0..width) |k| sum += x[r * width + k] * w[n * width + k];
        y[(row0 + r) * out_w + n] = bf16.bits(sum + b[n]);
    };
}

/// h3_rows_out: y[r] = b[n] + sum over k of bf16(x[row0 + r, k]) w[n, k], lane-strided like the kernel.
/// `x` is bf16 bits of `hidden` a row, `w` and `b` f32 of `width` rows.
pub fn rowsOut(x: []align(1) const u16, w: []align(1) const f32, b: []align(1) const f32, y: []f32, row0: usize, count: usize, hidden: usize, width: usize) void {
    for (0..count) |r| for (0..width) |n| {
        const at = struct {
            x: []align(1) const u16,
            base: usize,
            fn f(self: @This(), j: usize) f32 {
                return bf16.value(self.x[self.base + j]);
            }
        }{ .x = x, .base = (row0 + r) * hidden };
        y[r * width + n] = sum32(at, w, n * hidden, hidden) + b[n];
    };
}

/// h3_final_norm: each row's RMS norm, modulated by its table line's shift then scale, to bf16 bits.
/// `table` is the (times, 2 * hidden) final table in bf16, `line[r]` its row; w is final_norm.weight.
pub fn finalNorm(x: []align(1) const u16, w: []align(1) const u16, table: []align(1) const u16, line: []align(1) const i32, y: []u16, hidden: usize, eps: f32) void {
    for (line, 0..) |t, r| {
        const xr = x[r * hidden ..][0..hidden];
        const tab = table[@as(usize, @intCast(t)) * 2 * hidden ..][0 .. 2 * hidden];
        var sq: f32 = 0;
        for (xr) |v| {
            const f = bf16.value(v);
            sq += f * f;
        }
        const inv = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(hidden)) + eps);
        for (0..hidden) |c| {
            y[r * hidden + c] = bf16.bits(modulated(bf16.value(xr[c]), inv, bf16.value(w[c]), bf16.value(tab[hidden + c]), bf16.value(tab[c])));
        }
    }
}

/// h3_modulation: the AdaLN projection for the CUDA tables' layout — Y[(line, part, c)] as the kernel's
/// value (bf16 bits when `Out` is u16, else the same value as f32); line = timestep * groups + group.
/// The Metal kernel writes (part, line, c). `x` is the embedder's bf16 output (t rows of k), `w` and `b`
/// the projection's values as f32 (`floats` of bf16 entries is exact).
pub fn modulation(comptime Out: type, x: []align(1) const u16, w: []align(1) const f32, b: []align(1) const f32, out: []Out, t: usize, k: usize, groups: usize, parts: usize, width: usize) !void {
    const n = try shape.projection(t, k, groups, parts, width);
    if (x.len != t * k or w.len != n * k or b.len != n or out.len != t * n) return error.H3BadProjectionShape;
    // every output value is the kernel's n in (group, part, c) order; the CUDA layout reorders it
    for (0..n) |o| {
        const group = o / (parts * width);
        const part = o / width % parts;
        const c = o % width;
        for (0..t) |ti| {
            const at = struct {
                x: []align(1) const u16,
                base: usize,
                fn f(self: @This(), j: usize) f32 {
                    return bf16.value(self.x[self.base + j]);
                }
            }{ .x = x, .base = ti * k };
            const value = bf16.round32(sum32(at, w, o * k, k) + b[o]);
            out[((ti * groups + group) * parts + part) * width + c] = if (Out == u16) bf16.bits(value) else value;
        }
    }
}

fn silu(v: f32) f32 {
    return v / (1.0 + @exp(-v));
}

/// y (rows, n) = x (rows, k) w^T + b on the host, summed in float64 and rounded once to float32.
fn linear(a: std.mem.Allocator, x: []const f32, w: []const f32, b: []const f32, rows: usize, k: usize) ![]f32 {
    const n = b.len;
    const y = try a.alloc(f32, try shape.linear(x.len, w.len, n, rows, k));
    for (0..rows) |r| for (0..n) |o| {
        var s: f64 = 0;
        for (x[r * k ..][0..k], w[o * k ..][0..k]) |p, q| s += @as(f64, p) * q;
        y[r * n + o] = @floatCast(s + b[o]);
    };
    return y;
}

/// silu(time_embedder(sinusoid(t))) in bf16 for every timestep of `table`: the AdaLN projections' input.
/// The weights are the embedder's values as f32 (`floats` of bf16 entries is exact); `dim` is its width.
pub fn embed(a: std.mem.Allocator, table: []const f32, dim: usize, w_in: []const f32, b_in: []const f32, w_out: []const f32, b_out: []const f32) ![]u16 {
    const half = dim / 2;
    // cosine half first, angles and frequencies in float32 as MLX computes them
    const sinusoid = try a.alloc(f32, table.len * dim);
    for (table, 0..) |t, r| for (0..half) |i| {
        const freq = @exp(@as(f32, @floatCast(-@log(10000.0))) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        sinusoid[r * dim + i] = @cos(t * freq);
        sinusoid[r * dim + half + i] = @sin(t * freq);
    };
    const hidden = try linear(a, sinusoid, w_in, b_in, table.len, dim);
    for (hidden) |*v| v.* = silu(v.*);
    const temb = try linear(a, hidden, w_out, b_out, table.len, b_in.len);
    const out = try a.alloc(u16, temb.len);
    for (out, temb) |*o, v| o.* = bf16.bits(silu(v));
    return out;
}

test "rows in: the latent rows projected to bf16 stream rows, hand-checked" {
    // x = [1, 2], w = [[0.5, 0.5], [1, -1]], b = [0.25, -0.25]: bf16(1.75) and bf16(-1.25) at row 1
    var y: [6]u16 = @splat(0);
    rowsIn(&.{ 1.0, 2.0 }, &.{ 0.5, 0.5, 1.0, -1.0 }, &.{ 0.25, -0.25 }, &y, 1, 1, 2, 2);
    try std.testing.expectEqualSlices(u16, &.{ 0, 0, bf16.bits(1.75), bf16.bits(-1.25), 0, 0 }, &y);
    try std.testing.expectEqual(@as(f32, 1.75), bf16.value(y[2]));
    try std.testing.expectEqual(@as(f32, -1.25), bf16.value(y[3]));
}

test "final norm and the step head: concrete bf16 values at the kernel's rounding points" {
    // x = [2, 2, 2, 2], w all 1: sq = 16, inv = 1/sqrt(4 + 1e-5) ~= 0.49999875; bf16(2*inv) = 1.0
    const x = [_]u16{ bf16.bits(2.0), bf16.bits(2.0), bf16.bits(2.0), bf16.bits(2.0) };
    const w = [_]u16{ bf16.bits(1.0), bf16.bits(1.0), bf16.bits(1.0), bf16.bits(1.0) };
    var table: [8]u16 = @splat(0);
    // shift [0.25, 0, -1, 0], scale [0.5, -1, 0, 2]: values 1.75, 0.25? bf16(1*0)=0 then +0; -1+... see below
    for ([_]f32{ 0.25, 0, -1.0, 0 }, 0..) |v, i| table[i] = bf16.bits(v);
    for ([_]f32{ 0.5, -1.0, 0, 2.0 }, 0..) |v, i| table[4 + i] = bf16.bits(v);
    var y: [4]u16 = undefined;
    finalNorm(&x, &w, &table, &.{0}, &y, 4, 1e-5);
    // channel 0: bf16(1.0 * bf16(1+0.5)) = 1.5, + 0.25 -> 1.75; channel 1: bf16(1*(1-1)) = 0 -> 0.0
    // channel 2: bf16(1*1) = 1, + (-1) -> 0.0; channel 3: bf16(1*3) = 3.0
    try std.testing.expectEqualSlices(u16, &.{ bf16.bits(1.75), bf16.bits(0.0), bf16.bits(0.0), bf16.bits(3.0) }, &y);
    // the head: normed rows to f32 logits, lane-strided sums like h3_rows_out, identity weights
    var logits: [4]f32 = undefined;
    const eye = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    rowsOut(&y, &eye, &.{ 0.5, 0, 0, -0.5 }, &logits, 0, 1, 4, 4);
    try std.testing.expectEqualSlices(f32, &.{ 2.25, 0.0, 0.0, 2.5 }, &logits);
}

test "modulation: the projection's values in the CUDA tables' (line, part, c) layout" {
    // t = 2 timesteps, k = 1, groups = 3, parts = 6, width = 4: out n = 72, tables 144 f32
    const t = 2;
    const k = 1;
    const groups = 3;
    const parts = 6;
    const width = 4;
    var w: [groups * parts * width * k]f32 = undefined;
    var b: [groups * parts * width]f32 = undefined;
    for (0..groups * parts * width) |n| {
        w[n * k] = 0;
        b[n] = @floatFromInt(n);
    }
    // only the (group 1, part 2, c 3) weight row sees the input
    const on = (1 * parts + 2) * width + 3;
    w[on * k] = 2.0;
    const x = [_]u16{ bf16.bits(1.5), bf16.bits(-1.5) };
    var out: [t * groups * parts * width]f32 = undefined;
    try modulation(f32, &x, &w, &b, &out, t, k, groups, parts, width);
    for (0..t) |ti| for (0..groups) |g| for (0..parts) |p| for (0..width) |c| {
        const n = (g * parts + p) * width + c;
        const want = bf16.round32(@as(f32, @floatFromInt(n)) + if (n == on) 2.0 * bf16.value(x[ti]) else 0);
        try std.testing.expectEqual(want, out[((ti * groups + g) * parts + p) * width + c]);
    };
    // bf16 out: the same values as bits
    var bits_out: [t * groups * parts * width]u16 = undefined;
    try modulation(u16, &x, &w, &b, &bits_out, t, k, groups, parts, width);
    try std.testing.expectEqual(bits_out[((1 * groups + 1) * parts + 2) * width + 3], bf16.bits(out[((1 * groups + 1) * parts + 2) * width + 3]));
    var bad_out: [1]f32 = undefined;
    try std.testing.expectError(error.H3BadProjectionShape, modulation(f32, &.{0}, &.{ 0, 0 }, &.{ 0, 0 }, &bad_out, 1, 1, 1, 1, 2));
}

test "embed: the sinusoid through the two projections to bf16, hand-checked at one timestep" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // dim 4: sinusoid [cos(t), cos(0.01 t), sin(t), sin(0.01 t)]; w_in picks one column per row
    var w_in: [16]f32 = @splat(0);
    w_in[0] = 1; // row 0: cos(t)
    w_in[2 * 4 + 2] = 1; // row 2: sin(t)
    var w_out: [16]f32 = @splat(0);
    w_out[0] = 1;
    const out = try embed(a, &.{0.5}, 4, &w_in, &.{ 0, 0, 0, 0 }, &w_out, &.{ 0, 0, 0, 0 });
    // out[0] = bf16(silu(silu(cos(0.5)))), the silu's in f32 between the projections' f64 sums
    const silu_of = struct {
        fn f(v: f32) f32 {
            return v / (1.0 + @exp(-v));
        }
    };
    const want = bf16.bits(silu_of.f(silu_of.f(@cos(@as(f32, 0.5)))));
    try std.testing.expectEqual(want, out[0]);
    try std.testing.expectEqualSlices(u16, &.{ bf16.bits(0.0), bf16.bits(0.0), bf16.bits(0.0) }, out[1..4]);
}
