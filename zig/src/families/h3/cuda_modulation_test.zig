//! Device regression of the actual AdaLN kernel against the host's ordered sums, including ragged K and N.
const std = @import("std");
const cuda = @import("cuda");
const rows = @import("rows_reference.zig");
const bf = @import("bf16_round");
const image align(16) = @embedFile("fatbin_h3_modulation").*;

pub fn run(a: std.mem.Allocator) !void {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.shared(&driver);
    defer ctx.deinit();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var module = try cuda.Module.load(&driver, &image);
    defer module.unload();
    const t = 2;
    const k = 65;
    const groups = 3;
    const parts = 6;
    const width = 3;
    const n = groups * parts * width;
    var x: [t * k]f32 = undefined;
    var xb: [t * k]u16 = undefined;
    var w: [n * k]f32 = undefined;
    var b: [n]f32 = undefined;
    for (&x, &xb, 0..) |*f, *bits, i| {
        f.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16;
        bits.* = bf.bits(f.*);
    }
    for (&w, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 32;
    for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) / 8;
    var expected: [t * n]f32 = undefined;
    try rows.modulation(f32, &xb, &w, &b, &expected, t, k, groups, parts, width);
    var x_dev = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&x));
    defer x_dev.free();
    var output = try cuda.DeviceBuffer.alloc(&driver, @sizeOf(@TypeOf(expected)));
    defer output.free();
    var actual: [t * n]f32 = undefined;
    var wb: [n * k]u16 = undefined;
    var bb: [n]u16 = undefined;
    for (&wb, w) |*v, f| v.* = bf.bits(f);
    for (&bb, b) |*v, f| v.* = bf.bits(f);
    for (0..2) |wide| {
        var weights = try cuda.DeviceBuffer.fromHost(&driver, if (wide == 1) std.mem.sliceAsBytes(&w) else std.mem.sliceAsBytes(&wb));
        defer weights.free();
        var bias = try cuda.DeviceBuffer.fromHost(&driver, if (wide == 1) std.mem.sliceAsBytes(&b) else std.mem.sliceAsBytes(&bb));
        defer bias.free();
        var args: cuda.Args = .{};
        for ([_]u64{ x_dev.ptr, weights.ptr, bias.ptr, output.ptr }) |p| args.add(p);
        for ([_]c_int{ t, n, k, @intCast(wide), @intCast(wide) }) |v| args.add(v);
        for (0..5) |_| {
            errdefer cuda.completion.drain(&driver, stream, "H3 modulation test did not drain");
            try cuda.launch.launch(try module.function("h3m_project"), .{ .grid = .{ .x = (n + 7) / 8, .y = t }, .block = .{ .x = 256 } }, stream, &args);
            try stream.synchronize();
            try output.download(0, std.mem.sliceAsBytes(&actual));
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(&actual), std.mem.sliceAsBytes(&expected))) return error.H3ModulationMismatch;
        }
    }
    try embedCheck(a, &driver, module, stream);
    std.debug.print("H3 CUDA modulation: BF16/F32 weights, ragged K/N, two times, three modalities, five repeats match ordered host sums\n", .{});
}

fn embedCheck(a: std.mem.Allocator, d: *const cuda.Driver, module: cuda.Module, stream: cuda.Stream) !void {
    const times = [_]f32{ 0, 1000, 650 };
    const dim = 4;
    const w = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    const b = [_]f32{ 0.125, -0.25, 0.5, -1 };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const expected = try rows.embed(arena.allocator(), &times, dim, &w, &b, &w, &b);
    var td = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&times));
    defer td.free();
    var wd = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&w));
    defer wd.free();
    var bd = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&b));
    defer bd.free();
    var x = try cuda.DeviceBuffer.alloc(d, times.len * dim * 4);
    defer x.free();
    var mid = try cuda.DeviceBuffer.alloc(d, x.len);
    defer mid.free();
    var out = try cuda.DeviceBuffer.alloc(d, x.len);
    defer out.free();
    errdefer cuda.completion.drain(d, stream, "H3 embed test did not drain");
    var args: cuda.Args = .{};
    args.add(td.ptr);
    args.add(x.ptr);
    args.add(@as(c_int, dim));
    args.add(@as(c_int, times.len));
    try cuda.launch.launch(try module.function("h3m_sinusoid"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &args);
    for ([_][2]u64{ .{ x.ptr, mid.ptr }, .{ mid.ptr, out.ptr } }) |pair| {
        args = .{};
        for ([_]u64{ pair[0], wd.ptr, bd.ptr, pair[1] }) |p| args.add(p);
        for ([_]c_int{ times.len, dim, dim, 1, 1 }) |v| args.add(v);
        try cuda.launch.launch(try module.function("h3m_embed"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &args);
    }
    try stream.synchronize();
    var actual: [times.len * dim]f32 = undefined;
    try out.download(0, std.mem.sliceAsBytes(&actual));
    for (actual, expected) |got, want| if (bf.bits(got) != want) return error.H3EmbeddingMismatch;
}
