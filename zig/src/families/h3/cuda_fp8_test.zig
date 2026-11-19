//! Device check of the block-scaled FP8 projection (`Fp8.weight`, `Fp8.run`) against a float64 product of the same
//! bf16 operands: rows that fill one 128-row scale tile and part of the next, k over two scale tiles, and one input
//! column 64 times the rest, the outlier a scale a row would spread over every value. A wrong tiled scale offset,
//! rotation or operand order lands near relative error 1; MXFP8's own rounding stays near e4m3's 2^-4 step.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const bf = @import("bf16_round");

/// Relative Frobenius error the product must stay within: e4m3 keeps 3 mantissa bits on each operand.
const bound = 0.06;

pub fn run(a: std.mem.Allocator) !void {
    const rows = 200;
    const n = 256;
    const k = 384;
    const model = try h3.Model.setup(a);
    defer model.deinit();
    model.hidden = k;
    model.inner = k;
    model.mlp = k;

    var prng = std.Random.DefaultPrng.init(89);
    const random = prng.random();
    const x = try a.alloc(u16, rows * k);
    defer a.free(x);
    const w = try a.alloc(u16, n * k);
    defer a.free(w);
    for (x, 0..) |*v, i| v.* = bf.bits(random.floatNorm(f32) * @as(f32, if (i % k == 37) 64 else 1));
    for (w) |*v| v.* = bf.bits(random.floatNorm(f32) / 16);

    var xd = try cuda.DeviceBuffer.alloc(&model.driver, rows * k * 2);
    defer xd.free();
    var wd = try cuda.DeviceBuffer.alloc(&model.driver, n * k * 2);
    var yd = try cuda.DeviceBuffer.alloc(&model.driver, rows * n * 2);
    defer yd.free();
    try xd.upload(0, std.mem.sliceAsBytes(x));
    try wd.upload(0, std.mem.sliceAsBytes(w));
    const t = try h3.Fp8.init(model);
    var weight = try t.weight(model, &wd, n, k);
    defer weight.q.free();
    try t.run(model, xd.ptr, weight, yd.ptr, rows);
    try model.stream.synchronize();
    const y = try a.alloc(u16, rows * n);
    defer a.free(y);
    try yd.download(0, std.mem.sliceAsBytes(y));

    var err: f64 = 0;
    var norm: f64 = 0;
    for (0..rows) |r| for (0..n) |c| {
        var want: f64 = 0;
        for (0..k) |i| want += @as(f64, bf.value(x[r * k + i])) * bf.value(w[c * k + i]);
        const got: f64 = bf.value(y[r * n + c]);
        err += (got - want) * (got - want);
        norm += want * want;
    };
    const relative = @sqrt(err / norm);
    if (!(relative <= bound)) {
        std.debug.print("H3 CUDA FP8 product: relative error {e:.3} past {d}\n", .{ relative, bound });
        return error.H3Fp8Mismatch;
    }
    std.debug.print("H3 CUDA FP8 product: {d} x {d} x {d} within relative error {e:.3} of float64 (bound {d})\n", .{ rows, n, k, relative, bound });
}
