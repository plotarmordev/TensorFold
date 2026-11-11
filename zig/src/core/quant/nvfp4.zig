//! ModelOpt NVFP4 weights (e2m1 codes, an e4m3 scale every 16 inputs, a global scale) on the host: fp64 references.
const std = @import("std");
const types = @import("types.zig");
const fp8 = @import("fp8.zig");

const Tensor = types.Tensor;

/// Inputs one e4m3 scale covers.
pub const group = 16;

/// An e2m1 nibble's value.
pub fn e2m1(n: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return (if (n & 8 != 0) @as(f64, -1) else 1) * mag[n & 7];
}

/// As stored: `weight` u8 [..., N, K / 2] (even columns low nibbles), `scale` e4m3 [..., N, K / 16], `global` a matrix.
pub const Host = struct { weight: Tensor, scale: Tensor, global: []const f32 };

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    return h.weight.numel() * 2;
}

/// fp64 dequantization of every row of `h` (a stack's leading dimensions folded), code * scale * global, into `out`.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len or h.scale.rank != rank or h.weight.dtype != .u8 or h.scale.dtype != .f8_e4m3) return error.UnexpectedTensor;
    const n = h.weight.shape[rank - 2];
    const k = h.weight.shape[rank - 1] * 2;
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    for (0..rank - 2) |d| if (h.weight.shape[d] == 0 or h.weight.shape[d] != h.scale.shape[d]) return error.UnexpectedTensor;
    if (h.scale.shape[rank - 2] != n or h.scale.shape[rank - 1] * group != k or out.len != elements(h)) return error.UnexpectedTensor;
    const stack = elements(h) / (n * k);
    if (h.global.len != stack) return error.UnexpectedTensor;
    for (0..stack) |e| for (0..n) |r| for (0..k) |c| {
        const row = e * n + r;
        const byte = h.weight.bytes[row * (k / 2) + c / 2];
        const nib: u8 = if (c & 1 == 0) byte & 0xF else byte >> 4;
        out[row * k + c] = e2m1(nib) * fp8.e4m3(h.scale.bytes[row * (k / group) + c / group]) * h.global[e];
    };
}

/// ModelOpt NVFP4 round trip of `w` [N,K]: global amax/(6*448), e4m3 scale per 16 inputs, e2m1 ties to even.
pub fn quantize(w: []const f64, k: usize, out: []f64) void {
    std.debug.assert(k % group == 0 and w.len % k == 0 and out.len == w.len);
    var amax: f64 = 0;
    for (w) |x| amax = @max(amax, @abs(x));
    const global = amax / (6 * 448);
    for (0..w.len / group) |b| {
        const in = w[b * group ..][0..group];
        var block_max: f64 = 0;
        for (in) |x| block_max = @max(block_max, @abs(x));
        const step = if (global > 0) fp8.toE4m3(block_max / (6 * global)) * global else 0;
        for (in, out[b * group ..][0..group]) |x, *q| q.* = if (step > 0) std.math.copysign(nearestE2m1(@abs(x) / step), x) * step else 0;
    }
}

/// The e2m1 magnitude nearest m (at most 6 after saturation); a midpoint takes the even code.
fn nearestE2m1(m: f64) f64 {
    var best: u8 = 0;
    for (1..8) |i| {
        const d = @abs(e2m1(@intCast(i)) - m);
        const bd = @abs(e2m1(best) - m);
        if (d < bd or (d == bd and i % 2 == 0)) best = @intCast(i);
    }
    return e2m1(best);
}

test "an NVFP4 stack against hand-decoded values" {
    // two experts of [1, 32]: two scale groups a row; byte 0 holds columns 0 (low) and 1 (high)
    var codes: [2 * 16]u8 = @splat(0);
    codes[0] = 0xF2; // column 0: e2m1 2 -> 1.0; column 1: e2m1 0xF -> -6
    codes[8] = 0x07; // column 16: 6
    codes[16 + 15] = 0x30; // expert 1, column 31: e2m1 3 -> 1.5
    const scales = [_]u8{ 0x38, 0x40, 0x38, 0x30 }; // 1, 2 | 1, 0.5
    const global = [_]f32{ 0.25, 2 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 16, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 2, 1, 2, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [64]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 0.25), out[0]);
    try std.testing.expectEqual(@as(f64, -1.5), out[1]);
    try std.testing.expectEqual(@as(f64, 3), out[16]);
    try std.testing.expectEqual(@as(f64, 1.5), out[32 + 31]);
    try std.testing.expectEqual(@as(f64, 0), out[2]);
}

test "a reference refuses mismatched expert stacks" {
    const codes: [16]u8 = @splat(0x22);
    const scales: [2]u8 = @splat(0x38);
    const global = [_]f32{ 1, 1 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 8, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [32]f64 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &out));
}

test "quantize rounds midpoints to the even e2m1 code" {
    // block 0 holds the amax, so global = 1 and block 1's scale is exactly 1
    var w: [32]f64 = @splat(0);
    w[0] = 2688;
    const ties = [_]f64{ 6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.75, -5 };
    const want = [_]f64{ 6, 0, 1, 1, 2, 2, 4, 4, -1, -4 };
    @memcpy(w[16..][0..ties.len], &ties);
    var out: [32]f64 = undefined;
    quantize(&w, 32, &out);
    try std.testing.expectEqualSlices(f64, &want, out[16..][0..want.len]);
}

test "quantize keeps a small block's precision beside a large tensor amax" {
    // global = 1: block 1's scale is exact at 2^-5; amax / 6 as global would flush it below e4m3's smallest subnormal
    var w: [32]f64 = undefined;
    @memset(w[0..16], 2688);
    @memset(w[16..], 0.1875);
    var out: [32]f64 = undefined;
    quantize(&w, 32, &out);
    try std.testing.expectEqualSlices(f64, &w, &out);
}
