//! The cascade's pool and lift: exact small cases, the noise-and-data identities, and refusals.
const std = @import("std");
const cascade = @import("cascade.zig");
const testing = std.testing;

const full: cascade.Grid = .{ .rows = 4, .columns = 8 };
const low: cascade.Grid = .{ .rows = 2, .columns = 4 };

fn draw(seed: u64, out: []f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (out) |*o| o.* = prng.random().floatNorm(f32);
}

test "cascade: pool sums each 2 x 2 block over 2, per plane" {
    var noise: [2 * 16]f32 = undefined;
    for (&noise, 0..) |*o, i| o.* = @floatFromInt(i % 16 + 1);
    var out: [8]f32 = undefined;
    try cascade.pool(&noise, .{ .rows = 4, .columns = 4 }, &out);
    try testing.expectEqualSlices(f32, &.{ 7, 11, 23, 27, 7, 11, 23, 27 }, &out);
}

test "cascade: pooled unit noise keeps unit variance" {
    const big: cascade.Grid = .{ .rows = 256, .columns = 256 };
    const noise = try testing.allocator.alloc(f32, big.rows * big.columns);
    defer testing.allocator.free(noise);
    draw(7, noise);
    var out: [128 * 128]f32 = undefined;
    try cascade.pool(noise, big, &out);
    var sq: f64 = 0;
    for (out) |o| sq += o * o;
    try testing.expectApproxEqAbs(1.0, sq / out.len, 0.03);
}

test "cascade: a clean lift upsamples bilinearly with clamped edges" {
    // x = x0 at sigma 0.5 with no velocity; to = 0 keeps the upsampled clean estimate alone
    const x: [8]f32 = .{ 0, 1, 2, 3, 0, 1, 2, 3 };
    const v: [8]f32 = @splat(0);
    const noise: [32]f32 = @splat(0);
    var out: [32]f32 = undefined;
    try cascade.lift(&x, &v, 0.5, 0, &noise, full, &out);
    const row = [_]f32{ 0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3 };
    for (0..4) |r| try testing.expectEqualSlices(f32, &row, out[r * 8 ..][0..8]);
}

test "cascade: when the step's noise is the pooled seed, the lift is the seed's own noise over the clean data" {
    // two planes with different clean values, so a plane's offset cannot borrow the other's
    var noise: [64]f32 = undefined;
    draw(3, &noise);
    var eps: [16]f32 = undefined;
    try cascade.pool(&noise, full, &eps);
    const clean = [2]f32{ 0.3, -0.6 };
    const from: f32 = 0.75;
    const to: f32 = 0.4;
    var x: [16]f32 = undefined;
    var v: [16]f32 = undefined;
    for (&x, &v, eps, 0..) |*xo, *vo, e, i| {
        xo.* = (1 - from) * clean[i / 8] + from * e;
        vo.* = clean[i / 8] - e;
    }
    var out: [64]f32 = undefined;
    try cascade.lift(&x, &v, from, to, &noise, full, &out);
    for (out, noise, 0..) |o, n, i| try testing.expectApproxEqAbs((1 - to) * clean[i / 32] + to * n, o, 1e-5);
}

test "cascade: the lifted noise pools to the step's noise estimate and keeps the seed's detail" {
    var noise: [32]f32 = undefined;
    draw(5, &noise);
    var eps: [8]f32 = undefined;
    draw(6, &eps);
    const from: f32 = 0.75;
    const to: f32 = 0.5;
    var x: [8]f32 = undefined;
    var v: [8]f32 = undefined;
    for (&x, &v, eps) |*xo, *vo, e| {
        xo.* = (1 - from) * -0.2 + from * e;
        vo.* = -0.2 - e;
    }
    var out: [32]f32 = undefined;
    try cascade.lift(&x, &v, from, to, &noise, full, &out);
    var lifted: [32]f32 = undefined;
    for (&lifted, out) |*l, o| l.* = (o - (1 - to) * -0.2) / to;
    var pooled: [8]f32 = undefined;
    try cascade.pool(&lifted, full, &pooled);
    for (pooled, eps) |p, e| try testing.expectApproxEqAbs(e, p, 1e-5);
    // within a block the seed's differences survive
    try testing.expectApproxEqAbs(noise[1] - noise[0], lifted[1] - lifted[0], 1e-5);
}

test "cascade: refusals" {
    try testing.expectEqual(low, try cascade.half(full));
    try testing.expectError(error.H3BadCascadeGrid, cascade.half(.{ .rows = 6, .columns = 8 }));
    try testing.expectError(error.H3BadCascadeGrid, cascade.half(.{ .rows = 0, .columns = 8 }));
    const x: [8]f32 = @splat(0);
    const noise: [32]f32 = @splat(0);
    var out: [32]f32 = undefined;
    try testing.expectError(error.H3BadLatentLength, cascade.pool(noise[0..30], full, out[0..8]));
    try testing.expectError(error.H3BadLatentLength, cascade.pool(&noise, full, out[0..7]));
    try testing.expectError(error.H3BadLatentLength, cascade.lift(x[0..7], &x, 0.5, 0, &noise, full, &out));
    try testing.expectError(error.H3BadLatentLength, cascade.lift(&x, &x, 0.5, 0, &noise, full, out[0..31]));
    try testing.expectError(error.H3BadCascadeSigma, cascade.lift(&x, &x, 0.5, 0.5, &noise, full, &out));
    try testing.expectError(error.H3BadCascadeSigma, cascade.lift(&x, &x, 0, 0, &noise, full, &out));
    try testing.expectError(error.H3BadCascadeSigma, cascade.lift(&x, &x, 1.5, 0, &noise, full, &out));
    try testing.expectError(error.H3BadCascadeSigma, cascade.lift(&x, &x, std.math.nan(f32), 0, &noise, full, &out));
}
