//! condition_proj's fused bias at every prompt length, on a cuBLASLt stub that answers as GB10's did (card 154):
//! the canonical algorithm never adds a bias, and the heuristic's plain algorithm adds one only at some row counts.
const std = @import("std");
const cuda = @import("cuda");
const bf = @import("bf16_round");

const cublaslt = cuda.cublaslt;
const abi = cuda.abi;
const S = cublaslt.Status;
const not_supported: S = 15;
const bias_tag: u32 = @backingInt(cublaslt.Epilogue.bias);
/// An algorithm's tag bit: picked with the bias epilogue on the descriptor.
const picked_for_bias: u64 = 1 << 32;

const Desc = struct { epilogue: u32, bias: u64 };
const Lay = struct { rows: u64, cols: u64 };

var descs: [8]Desc = undefined;
var ndescs: usize = 0;
var lays: [24]Lay = undefined;
var nlays: usize = 0;
var obj: u8 = 0;

/// GB10's probe: the plain algorithm for (T, 5376, 5120) took the bias at T <= 32 and 65 to 90, at no length tried else.
fn plainTakesBias(rows: u64) bool {
    return rows <= 32 or (rows >= 65 and rows <= 90);
}

fn descOf(d: cublaslt.MatmulDesc) *Desc {
    return @ptrCast(@alignCast(d.?));
}

fn layOf(l: cublaslt.Layout) *const Lay {
    return @ptrCast(@alignCast(l.?));
}

fn create(comptime T: type) fn (*T) callconv(.c) S {
    return struct {
        fn f(out: *T) callconv(.c) S {
            out.* = @ptrCast(&obj);
            return 0;
        }
    }.f;
}

fn destroy(comptime T: type) fn (T) callconv(.c) S {
    return struct {
        fn f(_: T) callconv(.c) S {
            return 0;
        }
    }.f;
}

fn statusName(_: S) callconv(.c) ?[*:0]const u8 {
    return null;
}

fn descCreate(out: *cublaslt.MatmulDesc, _: c_int, _: cublaslt.DataType) callconv(.c) S {
    descs[ndescs] = .{ .epilogue = @backingInt(cublaslt.Epilogue.none), .bias = 0 };
    out.* = @ptrCast(&descs[ndescs]);
    ndescs += 1;
    return 0;
}

fn descSet(desc: cublaslt.MatmulDesc, attr: c_int, value: ?*const anyopaque, _: usize) callconv(.c) S {
    const d = descOf(desc);
    switch (attr) {
        cublaslt.desc_epilogue => d.epilogue = @as(*const u32, @ptrCast(@alignCast(value.?))).*,
        cublaslt.desc_bias_pointer => d.bias = @as(*const u64, @ptrCast(@alignCast(value.?))).*,
        else => {},
    }
    return 0;
}

fn layoutCreate(out: *cublaslt.Layout, _: cublaslt.DataType, rows: u64, cols: u64, _: i64) callconv(.c) S {
    lays[nlays] = .{ .rows = rows, .cols = cols };
    out.* = @ptrCast(&lays[nlays]);
    nlays += 1;
    return 0;
}

fn prefSet(_: cublaslt.Preference, _: c_int, _: ?*const anyopaque, _: usize) callconv(.c) S {
    return 0;
}

/// An algorithm names the rows it was picked at, and whether the bias epilogue was asked for.
fn getHeuristic(_: cublaslt.Handle, desc: cublaslt.MatmulDesc, _: cublaslt.Layout, _: cublaslt.Layout, _: cublaslt.Layout, d: cublaslt.Layout, _: cublaslt.Preference, _: c_int, out: [*]cublaslt.Heuristic, count: *c_int) callconv(.c) S {
    out[0] = std.mem.zeroes(cublaslt.Heuristic);
    out[0].algo.data[0] = layOf(d).cols | if (descOf(desc).epilogue == bias_tag) picked_for_bias else 0;
    count.* = 1;
    return 0;
}

fn takes(desc: cublaslt.MatmulDesc, d: cublaslt.Layout, algo: *const cublaslt.Algo) bool {
    if (descOf(desc).epilogue != bias_tag) return true;
    const rows = algo.data[0] & (picked_for_bias - 1);
    return rows == layOf(d).cols and (algo.data[0] & picked_for_bias != 0 or (rows != 16384 and plainTakesBias(rows)));
}

fn algoCheck(_: cublaslt.Handle, desc: cublaslt.MatmulDesc, _: cublaslt.Layout, _: cublaslt.Layout, _: cublaslt.Layout, d: cublaslt.Layout, algo: *const cublaslt.Algo, out: *cublaslt.Heuristic) callconv(.c) S {
    out.* = std.mem.zeroes(cublaslt.Heuristic);
    if (!takes(desc, d, algo)) {
        out.state = not_supported;
        return not_supported;
    }
    out.algo = algo.*;
    return 0;
}

/// The product plus the bias on host buffers, refused for an algorithm that cannot run the descriptor.
fn matmul(_: cublaslt.Handle, desc: cublaslt.MatmulDesc, _: ?*const anyopaque, w_dev: abi.DevicePtr, w_lay: cublaslt.Layout, x_dev: abi.DevicePtr, _: cublaslt.Layout, _: ?*const anyopaque, _: abi.DevicePtr, _: cublaslt.Layout, d_dev: abi.DevicePtr, d_lay: cublaslt.Layout, algo: ?*const cublaslt.Algo, _: abi.DevicePtr, _: usize, _: abi.Stream) callconv(.c) S {
    if (!takes(desc, d_lay, algo.?)) return not_supported;
    const e = descOf(desc);
    const k = layOf(w_lay).rows;
    const n = layOf(d_lay).rows;
    const w: [*]const u16 = @ptrFromInt(w_dev);
    const x: [*]const u16 = @ptrFromInt(x_dev);
    const y: [*]f32 = @ptrFromInt(d_dev);
    for (0..layOf(d_lay).cols) |r| for (0..n) |j| {
        var sum: f32 = if (e.epilogue == bias_tag) @as([*]const f32, @ptrFromInt(e.bias))[j] else 0;
        for (0..k) |l| sum += bf.value(x[r * k + l]) * bf.value(w[j * k + l]);
        y[r * n + j] = sum;
    };
    return 0;
}

fn api() cublaslt.Api {
    var a: cublaslt.Api = undefined;
    a.cublasLtCreate = create(cublaslt.Handle);
    a.cublasLtDestroy = destroy(cublaslt.Handle);
    a.cublasLtGetStatusName = statusName;
    a.cublasLtMatmulDescCreate = descCreate;
    a.cublasLtMatmulDescDestroy = destroy(cublaslt.MatmulDesc);
    a.cublasLtMatmulDescSetAttribute = descSet;
    a.cublasLtMatrixLayoutCreate = layoutCreate;
    a.cublasLtMatrixLayoutDestroy = destroy(cublaslt.Layout);
    a.cublasLtMatmulPreferenceCreate = create(cublaslt.Preference);
    a.cublasLtMatmulPreferenceDestroy = destroy(cublaslt.Preference);
    a.cublasLtMatmulPreferenceSetAttribute = prefSet;
    a.cublasLtMatmulAlgoGetHeuristic = getHeuristic;
    a.cublasLtMatmulAlgoCheck = algoCheck;
    a.cublasLtMatmul = matmul;
    return a;
}

fn dev(v: anytype) u64 {
    return @intCast(@intFromPtr(v));
}

/// Each length draws: 1 and 77 on the plain algorithm, 55 and 301 (and the encoder's 4,096) on one picked for the bias.
pub fn check(comptime Products: type, comptime run: anytype) !void {
    const a = std.testing.allocator;
    const stream: cuda.Stream = .{ .d = undefined, .handle = null };
    var ws: [64]u8 align(256) = undefined;
    const row = [_]u16{ bf.bits(1), bf.bits(2), bf.bits(-1), bf.bits(0.5) };
    const w = [_]u16{ bf.bits(1), 0, 0, 0, bf.bits(0.5), bf.bits(1), bf.bits(-1), bf.bits(2), bf.bits(-1), bf.bits(0.5), bf.bits(0.5), bf.bits(1) };
    const bias = [_]f32{ 10, -0.5, 0.25 };
    const want = [_]f32{ 11, 4, 0.25 };
    const max_rows = 4096;
    const x = try a.alloc(u16, max_rows * row.len);
    defer a.free(x);
    for (0..max_rows) |r| @memcpy(x[r * row.len ..][0..row.len], &row);
    const y = try a.alloc(f32, max_rows * w.len / row.len);
    defer a.free(y);
    for ([_]usize{ 1, 55, 77, 301, max_rows }) |rows| {
        ndescs = 0;
        nlays = 0;
        var p: Products = .{ .lt = .{ .lib = undefined, .api = api() }, .gpa = a, .workspace = .{ .d = undefined, .ptr = dev(&ws), .len = ws.len, .borrowed = true } };
        defer {
            var it = p.plans.valueIterator();
            while (it.next()) |l| l.deinit();
            p.plans.deinit(a);
        }
        @memset(y, 0);
        try run(&p, stream, dev(x.ptr), dev(&w), dev(&bias), dev(y.ptr), rows, 3, 4);
        for (0..rows) |r| try std.testing.expectEqualSlices(f32, &want, y[r * 3 ..][0..3]);
    }
}
