//! Device regression of the fused launches against the kernels they fold: h3_norm_quant's rows must be
//! byte-identical to h3_norm then h3_quantize, h3_norm_quant_fp4's to h3_norm then h3_quantize_fp4, h3_heads_i8's
//! and h3_swiglu_int8's to h3_dequantize then h3_heads / h3_swiglu. Every rounding and sum order is shared, so the
//! check is equality, not a bound.
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
    const heads = 3;
    const rot = 48;
    const slots = 64; // one tile of slots over the rows
    const width = 3 * heads * 128;

    // bf16 stream rows and the block's branch rows, norm weights, AdaLN tables and rotary tables
    var x: [rows * C]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = bf.bits(@as(f32, @floatFromInt(@as(i32, @intCast(i * 2654435761 % 97)) - 48)) / 16);
    var y: [rows * C]u16 = undefined;
    for (&y, 0..) |*v, i| v.* = bf.bits(@as(f32, @floatFromInt(@as(i32, @intCast(i * 40503 % 61)) - 30)) / 32);
    var nw: [C]u16 = undefined;
    for (&nw, 0..) |*v, i| v.* = bf.bits(1 + @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 16);
    var nq: [128]u16 = undefined;
    for (&nq, 0..) |*v, i| v.* = bf.bits(1 + @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) / 16);
    var nk: [128]u16 = undefined;
    for (&nk, 0..) |*v, i| v.* = bf.bits(1 + @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) / 16);
    var tab: [lines * 6 * C]f32 = undefined;
    // the AdaLN parts h3_norm reads: gated add gate_m, then scale_a/shift_a modulation
    const gate_m: c_int = 5;
    const scale_a: c_int = 1;
    const shift_a: c_int = 0;

    for (&tab, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 97 % 41)) - 20)) / 32;
    var line: [rows]i32 = undefined;
    for (&line, 0..) |*v, i| v.* = @intCast(i % lines);
    var cos: [rows * rot]f32 = undefined;
    var sin: [rows * rot]f32 = undefined;
    for (0..rows) |r| for (0..rot) |j| {
        const ang = @as(f32, @floatFromInt(r * rot + j)) * 0.01;
        cos[r * rot + j] = @cos(ang);
        sin[r * rot + j] = @sin(ang);
    };
    // slot order: every row once, the rest padding
    var rowof: [slots]i32 = @splat(-1);
    for (0..rows) |r| rowof[r] = @intCast(r);

    var x_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&x));
    defer x_dev.free();
    var x2_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&x));
    defer x2_dev.free();
    var x3_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&x));
    defer x3_dev.free();
    var y_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&y));
    defer y_dev.free();
    var nw_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&nw));
    defer nw_dev.free();
    var nq_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&nq));
    defer nq_dev.free();
    var nk_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&nk));
    defer nk_dev.free();
    var tab_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&tab));
    defer tab_dev.free();
    var line_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&line));
    defer line_dev.free();
    var cos_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&cos));
    defer cos_dev.free();
    var sin_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&sin));
    defer sin_dev.free();
    var rowof_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&rowof));
    defer rowof_dev.free();

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

    // ---- norm + fp4: h3_quantize_fp4 on the unfused chain's normed rows, the fused launch on x3_dev; the scales'
    // tile padding is never written, so both buffers start alike
    const fp4_scales = cuda.cublaslt.scaleBytes(rows, C);
    var f1 = try cuda.DeviceBuffer.alloc(&driver, rows * C / 2 + fp4_scales);
    defer f1.free();
    try f1.fill8(0xAB, stream.handle);
    var f2 = try cuda.DeviceBuffer.alloc(&driver, rows * C / 2 + fp4_scales);
    defer f2.free();
    try f2.fill8(0xAB, stream.handle);
    {
        var args: cuda.Args = .{};
        for ([_]u64{ o_dev.ptr, f1.ptr, f1.ptr + rows * C / 2 }) |p| args.add(p);
        args.add(@as(c_int, C));
        try cuda.launch.launch(try module.function("h3_quantize_fp4"), .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args);
        args = .{};
        for ([_]u64{ x3_dev.ptr, y_dev.ptr, nw_dev.ptr, tab_dev.ptr, tab_dev.ptr, line_dev.ptr, f2.ptr, f2.ptr + rows * C / 2 }) |p| args.add(p);
        for ([_]c_int{ C, gate_m, scale_a, shift_a }) |v| args.add(v);
        args.add(eps);
        try cuda.launch.launch(try module.function("h3_norm_quant_fp4"), .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args);
    }
    try stream.synchronize();

    // ---- dequantize + heads: int32 sums dequantize as h3_dequantize, then h3_heads vs the fused h3_heads_i8
    var acc: [slots * width]i32 = undefined;
    var sx: [slots]f32 = undefined;
    var sw: [width]f32 = undefined;
    for (&acc, 0..) |*v, i| v.* = @as(i32, @intCast(i * 2654435761 % 4000)) - 2000;
    for (&sx, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7 + 1)) * 0.01;
    for (&sw, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11 + 1)) * 0.001;
    var acc_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&acc));
    defer acc_dev.free();
    var sx_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&sx));
    defer sx_dev.free();
    var sw_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&sw));
    defer sw_dev.free();
    var qkv_dev = try cuda.DeviceBuffer.alloc(&driver, slots * width * 2);
    defer qkv_dev.free();
    // dead slots (rowof -1) are never written: a fill before the launches makes the full-tile compare deterministic
    var tq1 = try cuda.DeviceBuffer.alloc(&driver, heads * slots * 128 * 2);
    defer tq1.free();
    try tq1.fill8(0xAB, stream.handle);
    var tk1 = try cuda.DeviceBuffer.alloc(&driver, heads * slots * 128 * 2);
    defer tk1.free();
    try tk1.fill8(0xAB, stream.handle);
    var tv1 = try cuda.DeviceBuffer.alloc(&driver, heads * slots * 128 * 2);
    defer tv1.free();
    try tv1.fill8(0xAB, stream.handle);
    var tq2 = try cuda.DeviceBuffer.alloc(&driver, heads * slots * 128 * 2);
    defer tq2.free();
    try tq2.fill8(0xAB, stream.handle);
    var tk2 = try cuda.DeviceBuffer.alloc(&driver, heads * slots * 128 * 2);
    defer tk2.free();
    try tk2.fill8(0xAB, stream.handle);
    var tv2 = try cuda.DeviceBuffer.alloc(&driver, heads * slots * 128 * 2);
    defer tv2.free();
    try tv2.fill8(0xAB, stream.handle);
    {
        var args: cuda.Args = .{};
        for ([_]u64{ acc_dev.ptr, sx_dev.ptr, sw_dev.ptr, qkv_dev.ptr }) |p| args.add(p);
        for ([_]c_int{ slots, width }) |v| args.add(v);
        try cuda.launch.launch(try module.function("h3_dequantize"), .{ .grid = .{ .x = 1024 }, .block = .{ .x = 256 } }, stream, &args);
        args = .{};
        for ([_]u64{ qkv_dev.ptr, nq_dev.ptr, nk_dev.ptr, cos_dev.ptr, sin_dev.ptr, rowof_dev.ptr, tq1.ptr, tk1.ptr, tv1.ptr }) |p| args.add(p);
        for ([_]c_int{ heads, width, rot, slots }) |v| args.add(v);
        args.add(eps);
        try cuda.launch.launch(try module.function("h3_heads"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &args);
    }
    {
        var args: cuda.Args = .{};
        for ([_]u64{ acc_dev.ptr, sx_dev.ptr, sw_dev.ptr, nq_dev.ptr, nk_dev.ptr, cos_dev.ptr, sin_dev.ptr, rowof_dev.ptr, tq2.ptr, tk2.ptr, tv2.ptr }) |p| args.add(p);
        for ([_]c_int{ heads, width, rot, slots }) |v| args.add(v);
        args.add(eps);
        try cuda.launch.launch(try module.function("h3_heads_i8"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &args);
    }
    try stream.synchronize();

    // ---- swiglu: the unfused chain runs h3_dequantize then h3_swiglu; the fused kernel does both on int32 sums
    const n = C / 2; // the act row's width
    var wide = try cuda.DeviceBuffer.alloc(&driver, rows * 2 * n * 2);
    defer wide.free();
    var act1 = try cuda.DeviceBuffer.alloc(&driver, rows * n * 2);
    defer act1.free();
    var act2 = try cuda.DeviceBuffer.alloc(&driver, rows * n * 2);
    defer act2.free();
    {
        var args: cuda.Args = .{};
        for ([_]u64{ acc_dev.ptr, sx_dev.ptr, sw_dev.ptr, wide.ptr }) |p| args.add(p);
        for ([_]c_int{ rows, 2 * n }) |v| args.add(v);
        try cuda.launch.launch(try module.function("h3_dequantize"), .{ .grid = .{ .x = 1024 }, .block = .{ .x = 256 } }, stream, &args);
        args = .{};
        for ([_]u64{ wide.ptr, act1.ptr }) |p| args.add(p);
        for ([_]c_int{ rows, n }) |v| args.add(v);
        try cuda.launch.launch(try module.function("h3_swiglu"), .{ .grid = .{ .x = 1024 }, .block = .{ .x = 256 } }, stream, &args);
        args = .{};
        for ([_]u64{ acc_dev.ptr, sx_dev.ptr, sw_dev.ptr, act2.ptr }) |p| args.add(p);
        args.add(@as(c_int, n));
        try cuda.launch.launch(try module.function("h3_swiglu_int8"), .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args);
    }
    try stream.synchronize();

    const checks = .{
        .{ q1, q2, rows * C, "quantized rows" },
        .{ s1, s2, rows * 4, "row scales" },
        .{ f1, f2, rows * C / 2 + fp4_scales, "fp4 codes and block scales" },
        .{ x_dev, x3_dev, rows * C * 2, "fp4 gated stream rows" },
        .{ x_dev, x2_dev, rows * C * 2, "gated stream rows" },
        .{ tq1, tq2, heads * slots * 128 * 2, "q tiles" },
        .{ tk1, tk2, heads * slots * 128 * 2, "k tiles" },
        .{ tv1, tv2, heads * slots * 128 * 2, "v tiles" },
        .{ act1, act2, rows * n * 2, "act rows" },
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
    std.debug.print("H3 CUDA fused: h3_norm_quant, h3_norm_quant_fp4, h3_heads_i8 and h3_swiglu_int8 match the unfused kernels byte for byte\n", .{});
}
