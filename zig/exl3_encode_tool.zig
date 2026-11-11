//! tf-exl3-encode WEIGHT.npy HESSIAN.npy BITS OUT_DIR: one calibrated EXL3 group, with its receipt.
const std = @import("std");
const encoder = @import("src/core/exl3_encode.zig");
const format = @import("src/core/exl3_format.zig");
const npy = @import("src/core/npy.zig");
const shard = @import("src/core/shard_edit.zig");
const loader = @import("src/core/exl3_load.zig");
const nvfp4 = @import("src/core/quant/nvfp4.zig");
const max_threads = @import("src/core/exl3_threads.zig").max_threads;
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

const Command = struct { threads: usize, weight: []const u8, hessian: []const u8, bits: []const u8, out_dir: []const u8 };

/// [--threads N] WEIGHT HESSIAN BITS OUT_DIR; N defaults to every core and must be 1 to max_threads.
fn parseArgs(args: []const []const u8, cores: usize) ?Command {
    var rest = args;
    var threads: usize = @min(cores, max_threads);
    if (rest.len > 0 and std.mem.eql(u8, rest[0], "--threads")) {
        if (rest.len < 2) return null;
        threads = std.fmt.parseInt(usize, rest[1], 10) catch return null;
        if (threads == 0 or threads > max_threads) return null;
        rest = rest[2..];
    }
    if (rest.len != 4) return null;
    return .{ .threads = threads, .weight = rest[0], .hessian = rest[1], .bits = rest[2], .out_dir = rest[3] };
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var out_buf: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writerStreaming(init.io, &out_buf);
    defer stdout.interface.flush() catch {};
    const out = &stdout.interface;
    const cmd = parseArgs(args[1..], std.Thread.getCpuCount() catch 1) orelse {
        try out.writeAll("usage: tf-exl3-encode [--threads N] WEIGHT.npy HESSIAN.npy {2.0|2.5|3.0|4.0} OUT_DIR\nweight is HF [N,K] f32/f64; Hessian is [K,K] f32/f64; N is 1 to 64, every core by default\n");
        return 2;
    };
    const gpa = init.gpa;
    const bits = std.fmt.parseFloat(f64, cmd.bits) catch return 2;
    const halves: u4 = if (bits == 2.0) 4 else if (bits == 2.5) 5 else if (bits == 3.0) 6 else if (bits == 4.0) 8 else return 2;
    const source = try readArray(gpa, init.io, cmd.weight);
    defer freeArray(gpa, source);
    const h = try readArray(gpa, init.io, cmd.hessian);
    defer freeArray(gpa, h);
    const n = source.shape[0];
    const k = source.shape[1];
    if (h.shape[0] != k or h.shape[1] != k) return error.HessianShape;
    const weight = try transpose(gpa, source.values, n, k);
    defer gpa.free(weight);

    const started = Io.Clock.awake.now(init.io);
    var result = try encoder.encode(gpa, weight, h.values, k, n, .{ .halves = halves }, cmd.threads);
    defer result.deinit(gpa);
    const elapsed = seconds(init.io, started);
    try Io.Dir.cwd().createDirPath(init.io, cmd.out_dir);
    const model = try std.fs.path.join(init.arena.allocator(), &.{ cmd.out_dir, "model.safetensors" });
    const marker: i32 = @bitCast(format.Mul1Mul);
    const tensors = [_]shard.New{
        .{ .name = "weight.trellis", .dtype = .i16, .shape = &.{ k / 16, n / 16, result.width.tileWords() }, .bytes = std.mem.sliceAsBytes(result.trellis) },
        .{ .name = "weight.suh", .dtype = .f16, .shape = &.{k}, .bytes = std.mem.sliceAsBytes(result.suh) },
        .{ .name = "weight.svh", .dtype = .f16, .shape = &.{n}, .bytes = std.mem.sliceAsBytes(result.svh) },
        .{ .name = "weight.mul1", .dtype = .i32, .shape = &.{1}, .bytes = std.mem.asBytes(&marker) },
    };
    try shard.create(gpa, model, &.{.{ "format", "exl3" }}, &tensors);
    try verify(gpa, init.io, cmd.out_dir, result);
    const nv_rows = try gpa.alloc(f64, n * k);
    defer gpa.free(nv_rows);
    nvfp4.quantize(source.values, k, nv_rows);
    const nv = try transpose(gpa, nv_rows, n, k);
    defer gpa.free(nv);
    const nv_error = try encoder.hessianError(gpa, cmd.threads, weight, nv, h.values, k, n);
    const bytes = std.mem.sliceAsBytes(result.trellis).len + 2 * (k + n) + 4;
    const ratio = result.error_exl3 / nv_error;
    try out.print("bits_per_weight={d:.1} shape=[{d},{d}] threads={d} encode_seconds={d:.3} encode_hours_for_5900_matrices={d:.3}\n", .{ bits, n, k, cmd.threads, elapsed, elapsed * 5900 / 3600 });
    try out.print("hessian_error exl3={e:.9} plain={e:.9} nvfp4={e:.9} exl3_over_nvfp4={d:.6} bytes={d} nvfp4_bytes_approx={d}\n", .{ result.error_exl3, result.error_plain, nv_error, ratio, bytes, k * n / 2 + k * n / 16 + 4 });
    try out.writeAll("reader=ok reconstruction=bit_exact\n");
    return 0;
}

test "--threads sets the count, every core is the default, and 0 or over the cap is refused" {
    const paths = [_][]const u8{ "w.npy", "h.npy", "2.0", "out" };
    try std.testing.expectEqual(@as(usize, 36), parseArgs(&paths, 36).?.threads);
    try std.testing.expectEqual(@as(usize, max_threads), parseArgs(&paths, 256).?.threads);
    const nine = parseArgs(&(.{ "--threads", "9" } ++ paths), 36).?;
    try std.testing.expectEqual(@as(usize, 9), nine.threads);
    try std.testing.expectEqualStrings("out", nine.out_dir);
    for ([_][]const u8{ "0", "65", "x", "-1" }) |bad| try std.testing.expectEqual(null, parseArgs(&(.{ "--threads", bad } ++ paths), 36));
    try std.testing.expectEqual(null, parseArgs(&.{ "--threads", "9", "w.npy" }, 36));
    try std.testing.expectEqual(null, parseArgs(paths[0..3], 36));
}
