//! Device regression of the fused int8 launch against the kernels it folds: h3_norm_quant's rows must be
//! byte-identical to h3_norm then h3_quantize. Every rounding and sum order is shared, so the check is
//! equality, not a bound.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const bf = @import("bf16_round");
const image align(16) = @embedFile("fatbin_h3").*;

fn download(a: std.mem.Allocator, b: cuda.DeviceBuffer, bytes: usize) ![]u8 {
    const got = try a.alloc(u8, bytes);
    try b.download(0, got);
    return got;
}

pub fn run(a: std.mem.Allocator) !void {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.shared(&driver);
    defer ctx.deinit();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var module = try cuda.Module.load(&driver, &image);
    defer module.unload();

    const C = 384; // three 128-wide chunks, wider than one warp's stride
    const rows = 9;
    const lines = 2;

    // bf16 stream rows and the block's branch rows, norm weights and AdaLN tables
    var x: [rows * C]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = bf.bits(@as(f32, @floatFromInt(@as(i32, @intCast(i * 2654435761 % 97)) - 48)) / 16);
    var y: [rows * C]u16 = undefined;
    for (&y, 0..) |*v, i| v.* = bf.bits(@as(f32, @floatFromInt(@as(i32, @intCast(i * 40503 % 61)) - 30)) / 32);
    var nw: [C]u16 = undefined;
    for (&nw, 0..) |*v, i| v.* = bf.bits(1 + @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 16);
    var tab: [lines * 6 * C]f32 = undefined;
    // the AdaLN parts h3_norm reads: gated add gate_m, then scale_a/shift_a modulation
    const gate_m: c_int = 5;
    const scale_a: c_int = 1;
    const shift_a: c_int = 0;

    for (&tab, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 97 % 41)) - 20)) / 32;
    var line: [rows]i32 = undefined;
    for (&line, 0..) |*v, i| v.* = @intCast(i % lines);

    var x_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&x));
    defer x_dev.free();
    var x2_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&x));
    defer x2_dev.free();
    var y_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&y));
    defer y_dev.free();
    var nw_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&nw));
    defer nw_dev.free();
    var tab_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&tab));
    defer tab_dev.free();
    var line_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&line));
    defer line_dev.free();

    const eps: f32 = h3.eps;

    // ---- norm + quantize: the unfused chain on x_dev into o_dev/q1/s1, the fused launch on x2_dev into q2/s2
    var o_dev = try cuda.DeviceBuffer.alloc(&driver, rows * C * 2);
    defer o_dev.free();
    var q1 = try cuda.DeviceBuffer.alloc(&driver, rows * C);
    defer q1.free();
    var q2 = try cuda.DeviceBuffer.alloc(&driver, rows * C);
    defer q2.free();
    var s1 = try cuda.DeviceBuffer.alloc(&driver, rows * 4);
    defer s1.free();
    var s2 = try cuda.DeviceBuffer.alloc(&driver, rows * 4);
    defer s2.free();
    {
        var args: cuda.Args = .{};
        for ([_]u64{ x_dev.ptr, y_dev.ptr, nw_dev.ptr, tab_dev.ptr, tab_dev.ptr, line_dev.ptr, o_dev.ptr }) |p| args.add(p);
        for ([_]c_int{ C, gate_m, scale_a, shift_a }) |v| args.add(v);
        args.add(eps);
        try cuda.launch.launch(try module.function("h3_norm"), .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args);
        args = .{};
        for ([_]u64{ o_dev.ptr, q1.ptr, s1.ptr }) |p| args.add(p);
        args.add(@as(c_int, C));
        try cuda.launch.launch(try module.function("h3_quantize"), .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args);
    }
    {
        var args: cuda.Args = .{};
        for ([_]u64{ x2_dev.ptr, y_dev.ptr, nw_dev.ptr, tab_dev.ptr, tab_dev.ptr, line_dev.ptr, q2.ptr, s2.ptr }) |p| args.add(p);
        for ([_]c_int{ C, gate_m, scale_a, shift_a }) |v| args.add(v);
        args.add(eps);
        try cuda.launch.launch(try module.function("h3_norm_quant"), .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args);
    }
    try stream.synchronize();

    const checks = .{
        .{ q1, q2, rows * C, "quantized rows" },
        .{ s1, s2, rows * 4, "row scales" },
        .{ x_dev, x2_dev, rows * C * 2, "gated stream rows" },
    };
    inline for (checks) |chk| {
        const first = try download(a, chk[0], chk[2]);
        defer a.free(first);
        const second = try download(a, chk[1], chk[2]);
        defer a.free(second);
        if (!std.mem.eql(u8, first, second)) {
            std.debug.print("fused mismatch in {s}\n", .{chk[3]});
            return error.H3FusedMismatch;
        }
    }
    std.debug.print("H3 CUDA fused: h3_norm_quant matches h3_norm + h3_quantize byte for byte\n", .{});
}
