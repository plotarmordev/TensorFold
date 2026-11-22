//! GPU timestep embedding and AdaLN preparation into bf16 tables; only the small final-norm table returns to the host.
const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("checkpoint.zig");
const shape = @import("modulation_shape.zig");
const geometry = @import("geometry.zig");
const image align(16) = @embedFile("fatbin_h3_modulation").*;

pub const Tables = struct {
    blocks: cuda.DeviceBuffer,
    final: []u16,
    allocator: std.mem.Allocator,

    pub fn deinit(t: *Tables) void {
        t.blocks.free();
        t.allocator.free(t.final);
    }
};

const staging_bytes = 64 << 20;

const Tensor = struct {
    buffer: cuda.DeviceBuffer,
    wide: c_int,

    /// The stored tensor `name` uploaded through the pinned stages, alternating so one window's copy
    /// runs while the next is filled; a pageable fromHost copy is several times slower.
    fn load(d: *const cuda.Driver, staging: *[2]cuda.copy_source.Stage, stream: cuda.Stream, ck: *const checkpoint.Shards, name: []const u8, count: usize) !Tensor {
        const e = ck.entry(name) orelse return error.MissingTensor;
        if (e.dtype != .bf16 and e.dtype != .f32) return error.H3BadProjectionShape;
        const bytes = try ck.at(name, e.dtype, count);
        var buffer = try cuda.DeviceBuffer.alloc(d, bytes.len);
        errdefer {
            // a staged copy can still run: drain before the destination leaves the device
            cuda.completion.drain(d, stream, "H3 modulation upload did not drain");
            buffer.free();
        }
        var done: usize = 0;
        var turn: usize = 0;
        while (done < bytes.len) {
            const n = @min(bytes.len - done, staging_bytes);
            const window = try staging[turn].take(d, n);
            @memcpy(window, bytes[done..][0..n]);
            try staging[turn].copyRangeTo(buffer, done, 0, n, stream);
            done += n;
            turn ^= 1;
        }
        return .{ .buffer = buffer, .wide = @intFromBool(e.dtype == .f32) };
    }
};

fn launch(module: cuda.Module, stream: cuda.Stream, name: [:0]const u8, grid: cuda.Dim3, args: *cuda.Args) !void {
    errdefer cuda.completion.drain(stream.d, stream, "H3 modulation launch did not drain");
    try cuda.launch.launch(try module.function(name), .{ .grid = grid, .block = .{ .x = 256 } }, stream, args);
    // Temporary uploads belong to this projection, and must not be released while its stream still reads them.
    try stream.synchronize();
}

fn project(module: cuda.Module, staging: *[2]cuda.copy_source.Stage, stream: cuda.Stream, ck: *const checkpoint.Shards, prefix: []const u8, x: u64, out: u64, rows: usize, n: usize, k: usize, embed: bool) !void {
    var name: [128]u8 = undefined;
    var w = try Tensor.load(stream.d, staging, stream, ck, try std.fmt.bufPrint(&name, "{s}weight", .{prefix}), try geometry.product(&.{ n, k }));
    defer w.buffer.free();
    var b = try Tensor.load(stream.d, staging, stream, ck, try std.fmt.bufPrint(&name, "{s}bias", .{prefix}), n);
    defer b.buffer.free();
    var args: cuda.Args = .{};
    for ([_]u64{ x, w.buffer.ptr, b.buffer.ptr, out }) |p| args.add(p);
    for ([_]usize{ rows, n, k }) |v| args.add(@as(c_int, @intCast(v)));
    args.add(w.wide);
    args.add(b.wide);
    try launch(module, stream, if (embed) "h3m_embed" else "h3m_project", if (embed) .{ .x = @intCast((rows * n + 255) / 256) } else .{ .x = @intCast((n + 7) / 8), .y = @intCast(rows) }, &args);
}

pub fn prepare(a: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, ck: *const checkpoint.Shards, times: []const f32, layers: usize, hidden: usize) !Tables {
    try geometry.dimension(times.len);
    try geometry.dimension(layers);
    if (times.len > 65535) return error.H3BadProjectionShape;
    const iw = ck.entry("time_embedder.proj_in.weight") orelse return error.MissingTensor;
    const ib = ck.entry("time_embedder.proj_in.bias") orelse return error.MissingTensor;
    const ow = ck.entry("time_embedder.proj_out.weight") orelse return error.MissingTensor;
    const ob = ck.entry("time_embedder.proj_out.bias") orelse return error.MissingTensor;
    const dim = try shape.embedding(iw, ib, ow, ob, times.len);
    const k = ob.shape[0];
    const n = try shape.projection(times.len, k, 3, 6, hidden);
    const count = try geometry.product(&.{ layers, times.len, n });
    const final_count = try geometry.product(&.{ times.len, 2, hidden });
    for ([_]usize{ times.len * dim, times.len * ib.shape[0], times.len * k, times.len * n, final_count }) |v|
        try geometry.dimension(v);
    var module = try cuda.Module.load(d, &image);
    defer module.unload();
    var staging: [2]cuda.copy_source.Stage = .{ .{}, .{} };
    defer for (&staging) |*stage| stage.deinit();
    var time = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(times));
    defer time.free();
    var sinusoid = try cuda.DeviceBuffer.alloc(d, try geometry.product(&.{ times.len, dim, 4 }));
    defer sinusoid.free();
    var middle = try cuda.DeviceBuffer.alloc(d, try geometry.product(&.{ times.len, ib.shape[0], 4 }));
    defer middle.free();
    var embedded = try cuda.DeviceBuffer.alloc(d, try geometry.product(&.{ times.len, k, 4 }));
    defer embedded.free();
    var blocks = try cuda.DeviceBuffer.alloc(d, try geometry.product(&.{ count, 2 }));
    errdefer blocks.free();
    var last = try cuda.DeviceBuffer.alloc(d, try geometry.product(&.{ final_count, 2 }));
    defer last.free();
    var args: cuda.Args = .{};
    args.add(time.ptr);
    args.add(sinusoid.ptr);
    args.add(@as(c_int, @intCast(dim)));
    args.add(@as(c_int, @intCast(times.len)));
    try launch(module, stream, "h3m_sinusoid", .{ .x = @intCast((times.len * dim + 255) / 256) }, &args);
    try project(module, &staging, stream, ck, "time_embedder.proj_in.", sinusoid.ptr, middle.ptr, times.len, ib.shape[0], dim, true);
    try project(module, &staging, stream, ck, "time_embedder.proj_out.", middle.ptr, embedded.ptr, times.len, k, ib.shape[0], true);
    var name: [128]u8 = undefined;
    for (0..layers) |i| try project(module, &staging, stream, ck, try std.fmt.bufPrint(&name, "blocks.{d}.adaln_proj.linear.", .{i}), embedded.ptr, blocks.ptr + i * times.len * n * 2, times.len, n, k, false);
    try project(module, &staging, stream, ck, "final_layer.adaln_proj.linear.", embedded.ptr, last.ptr, times.len, 2 * hidden, k, false);
    const final = try a.alloc(u16, final_count);
    errdefer a.free(final);
    try last.download(0, std.mem.sliceAsBytes(final));
    return .{ .blocks = blocks, .final = final, .allocator = a };
}
