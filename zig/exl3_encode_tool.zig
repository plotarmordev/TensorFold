//! tf-exl3-encode WEIGHT.npy HESSIAN.npy BITS OUT_DIR [--unweighted]: one calibrated EXL3 group, with its receipt.
const std = @import("std");
const encoder = @import("src/core/exl3_encode.zig");
const format = @import("src/core/exl3_format.zig");
const npy = @import("src/core/npy.zig");
const shard = @import("src/core/shard_edit.zig");
const loader = @import("src/core/exl3_load.zig");
const nvfp4 = @import("src/core/quant/nvfp4.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const Array = struct { bytes: []u8, values: []f64, shape: [2]usize };
fn readArray(gpa: Allocator, io: Io, path: []const u8) !Array {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    errdefer gpa.free(bytes);
    const a = try npy.parse(bytes);
    if (a.rank != 2 or (!std.mem.eql(u8, a.descr, "<f4") and !std.mem.eql(u8, a.descr, "<f8"))) return error.ExpectedF32OrF64Matrix;
    const values = try gpa.alloc(f64, a.count());
    errdefer gpa.free(values);
    if (std.mem.eql(u8, a.descr, "<f4")) {
        for (values, std.mem.bytesAsSlice(f32, a.data)) |*v, x| v.* = x;
    } else for (values, std.mem.bytesAsSlice(f64, a.data)) |*v, x| v.* = x;
    return .{ .bytes = bytes, .values = values, .shape = .{ a.shape[0], a.shape[1] } };
}
fn freeArray(gpa: Allocator, a: Array) void {
    gpa.free(a.values);
    gpa.free(a.bytes);
}

/// [rows, cols] row-major to [cols, rows].
fn transpose(gpa: Allocator, m: []const f64, rows: usize, cols: usize) ![]f64 {
    const t = try gpa.alloc(f64, m.len);
    for (0..rows) |r| for (0..cols) |c| {
        t[c * rows + r] = m[r * cols + c];
    };
    return t;
}

fn seconds(io: Io, start: Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(start.durationTo(Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
}

fn verify(gpa: Allocator, io: Io, out: []const u8, expected: encoder.Result) !void {
    var scan = try loader.scan(gpa, io, out);
    defer scan.deinit();
    if (scan.groups.len != 1 or scan.bad != 0 or scan.summary.mul1 != 1 or scan.groups[0].halves != expected.width.halves) return error.ReaderRefused;
    const group = scan.groups[0];
    if (group.k != expected.k or group.n != expected.n) return error.ReaderShape;
    const parts = try scan.parts(gpa, 0);
    const actual = try gpa.alloc(f64, expected.k * expected.n);
    defer gpa.free(actual);
    const scratch = try gpa.alloc(f64, 2 * expected.k * expected.n);
    defer gpa.free(scratch);
    format.dequantize(expected.width, .mul1, @alignCast(std.mem.bytesAsSlice(i16, parts.trellis.bytes)), group.k / 16, group.n / 16, @alignCast(std.mem.bytesAsSlice(f16, parts.suh.bytes)), @alignCast(std.mem.bytesAsSlice(f16, parts.svh.bytes)), actual, scratch);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(actual), std.mem.sliceAsBytes(expected.reconstructed))) return error.ReconstructionDiffers;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var out_buf: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writerStreaming(init.io, &out_buf);
    defer stdout.interface.flush() catch {};
    const out = &stdout.interface;
    const unweighted = args.len == 6 and std.mem.eql(u8, args[5], "--unweighted");
    if (args.len != 5 and !unweighted) {
        try out.writeAll("usage: tf-exl3-encode WEIGHT.npy HESSIAN.npy {2.0|2.5|3.0|4.0} OUT_DIR [--unweighted]\nweight is HF [N,K] f32/f64; Hessian is [K,K] f32/f64\n--unweighted also reports the same search with H = I, scored under HESSIAN\n");
        return 2;
    }
    const gpa = init.gpa;
    const bits = std.fmt.parseFloat(f64, args[3]) catch return 2;
    const halves: u4 = if (bits == 2.0) 4 else if (bits == 2.5) 5 else if (bits == 3.0) 6 else if (bits == 4.0) 8 else return 2;
    const source = try readArray(gpa, init.io, args[1]);
    defer freeArray(gpa, source);
    const h = try readArray(gpa, init.io, args[2]);
    defer freeArray(gpa, h);
    const n = source.shape[0];
    const k = source.shape[1];
    if (h.shape[0] != k or h.shape[1] != k) return error.HessianShape;
    const weight = try transpose(gpa, source.values, n, k);
    defer gpa.free(weight);

    const started = Io.Clock.awake.now(init.io);
    var result = try encoder.encode(gpa, weight, h.values, k, n, .{ .halves = halves });
    defer result.deinit(gpa);
    const elapsed = seconds(init.io, started);
    try Io.Dir.cwd().createDirPath(init.io, args[4]);
    const model = try std.fs.path.join(init.arena.allocator(), &.{ args[4], "model.safetensors" });
    const marker: i32 = @bitCast(format.Mul1Mul);
    const tensors = [_]shard.New{
        .{ .name = "weight.trellis", .dtype = .i16, .shape = &.{ k / 16, n / 16, result.width.tileWords() }, .bytes = std.mem.sliceAsBytes(result.trellis) },
        .{ .name = "weight.suh", .dtype = .f16, .shape = &.{k}, .bytes = std.mem.sliceAsBytes(result.suh) },
        .{ .name = "weight.svh", .dtype = .f16, .shape = &.{n}, .bytes = std.mem.sliceAsBytes(result.svh) },
        .{ .name = "weight.mul1", .dtype = .i32, .shape = &.{1}, .bytes = std.mem.asBytes(&marker) },
    };
    try shard.create(gpa, model, &.{.{ "format", "exl3" }}, &tensors);
    try verify(gpa, init.io, args[4], result);
    const exl3_error = encoder.hessianError(weight, result.reconstructed, h.values, k, n);
    const nv_rows = try gpa.alloc(f64, n * k);
    defer gpa.free(nv_rows);
    nvfp4.quantize(source.values, k, nv_rows);
    const nv = try transpose(gpa, nv_rows, n, k);
    defer gpa.free(nv);
    const nv_error = encoder.hessianError(weight, nv, h.values, k, n);
    const bytes = std.mem.sliceAsBytes(result.trellis).len + 2 * (k + n) + 4;
    const ratio = exl3_error / nv_error;
    try out.print("bits_per_weight={d:.1} shape=[{d},{d}] encode_seconds={d:.3} encode_hours_for_5900_matrices={d:.3}\n", .{ bits, n, k, elapsed, elapsed * 5900 / 3600 });
    try out.print("hessian_error exl3={e:.9} nvfp4={e:.9} exl3_over_nvfp4={d:.6} bytes={d} nvfp4_bytes_approx={d}\n", .{ exl3_error, nv_error, ratio, bytes, k * n / 2 + k * n / 16 + 4 });
    if (unweighted) {
        const identity = try gpa.alloc(f64, k * k);
        defer gpa.free(identity);
        for (identity, 0..) |*v, i| v.* = if (i / k == i % k) 1 else 0;
        var baseline = try encoder.encode(gpa, weight, identity, k, n, .{ .halves = halves });
        defer baseline.deinit(gpa);
        const baseline_error = encoder.hessianError(weight, baseline.reconstructed, h.values, k, n);
        try out.print("unweighted_hessian_error={e:.9} exl3_over_unweighted={d:.6}\n", .{ baseline_error, exl3_error / baseline_error });
    }
    try out.writeAll("reader=ok reconstruction=bit_exact\n");
    return 0;
}
