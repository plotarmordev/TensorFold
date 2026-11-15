//! Exercise canonical-algorithm bias selection when the actual-row heuristic declines fusion.
const std = @import("std");
const cuda = @import("cuda");
const bf = @import("bf16_round");

const cublaslt = cuda.cublaslt;
const abi = cuda.abi;
const S = cublaslt.Status;
const canonical: u64 = 16384; // Products.canonical_rows
const bias_tag: u32 = @backingInt(cublaslt.Epilogue.bias);

const Desc = struct { epilogue: u32, bias: u64 };
const Lay = struct { rows: u64, cols: u64 };

var bias_capable = false;
var matmul_calls: usize = 0;
var heuristic_calls: usize = 0;
var last_epilogue: u32 = 0;
var last_bias: u64 = 0;
var descs: [16]Desc = undefined;
var ndescs: usize = 0;
var lays: [48]Lay = undefined;
var nlays: usize = 0;
var lt_obj: u8 = 0;
var pref_obj: u8 = 0;

fn reset(capable: bool) void {
    bias_capable = capable;
    matmul_calls = 0;
    heuristic_calls = 0;
    last_epilogue = 0;
    last_bias = 0;
    ndescs = 0;
    nlays = 0;
}

fn descOf(d: cublaslt.MatmulDesc) *Desc {
    return @ptrCast(@alignCast(d.?));
}

fn layOf(l: cublaslt.Layout) *const Lay {
    return @ptrCast(@alignCast(l.?));
}

fn ltCreate(out: *cublaslt.Handle) callconv(.c) S {
    out.* = @ptrCast(&lt_obj);
    return 0;
}

fn ltDestroy(_: cublaslt.Handle) callconv(.c) S {
    return 0;
}

fn ltStatusName(_: S) callconv(.c) ?[*:0]const u8 {
    return null;
}

fn descCreate(out: *cublaslt.MatmulDesc, _: c_int, _: cublaslt.DataType) callconv(.c) S {
    descs[ndescs] = .{ .epilogue = @backingInt(cublaslt.Epilogue.none), .bias = 0 };
    out.* = @ptrCast(&descs[ndescs]);
    ndescs += 1;
    return 0;
}

fn descDestroy(_: cublaslt.MatmulDesc) callconv(.c) S {
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

fn layoutDestroy(_: cublaslt.Layout) callconv(.c) S {
    return 0;
}

fn prefCreate(out: *cublaslt.Preference) callconv(.c) S {
    out.* = @ptrCast(&pref_obj);
    return 0;
}

fn prefDestroy(_: cublaslt.Preference) callconv(.c) S {
    return 0;
}

fn prefSet(_: cublaslt.Preference, _: c_int, _: ?*const anyopaque, _: usize) callconv(.c) S {
    return 0;
}

/// Tag algorithms by row count so only the canonical shape can support bias.
fn getHeuristic(_: cublaslt.Handle, _: cublaslt.MatmulDesc, _: cublaslt.Layout, _: cublaslt.Layout, _: cublaslt.Layout, d: cublaslt.Layout, _: cublaslt.Preference, _: c_int, out: [*]cublaslt.Heuristic, count: *c_int) callconv(.c) S {
    heuristic_calls += 1;
    out[0] = std.mem.zeroes(cublaslt.Heuristic);
    out[0].algo.data[0] = layOf(d).cols;
    count.* = 1;
    return 0;
}

fn algoCheck(_: cublaslt.Handle, desc: cublaslt.MatmulDesc, _: cublaslt.Layout, _: cublaslt.Layout, _: cublaslt.Layout, _: cublaslt.Layout, algo: *const cublaslt.Algo, out: *cublaslt.Heuristic) callconv(.c) S {
    if (descOf(desc).epilogue == bias_tag and !(bias_capable and algo.data[0] == canonical)) return 7;
    out.* = std.mem.zeroes(cublaslt.Heuristic);
    out.*.algo = algo.*;
    return 0;
}

/// Evaluate the selected descriptor's product and optional bias on host buffers.
fn matmul(_: cublaslt.Handle, desc: cublaslt.MatmulDesc, alpha: ?*const anyopaque, a_dev: abi.DevicePtr, a_lay: cublaslt.Layout, x_dev: abi.DevicePtr, _: cublaslt.Layout, beta: ?*const anyopaque, _: abi.DevicePtr, _: cublaslt.Layout, d_dev: abi.DevicePtr, d_lay: cublaslt.Layout, _: ?*const cublaslt.Algo, _: abi.DevicePtr, _: usize, _: abi.Stream) callconv(.c) S {
    const d = descOf(desc);
    matmul_calls += 1;
    last_epilogue = d.epilogue;
    last_bias = d.bias;
    const k = layOf(a_lay).rows;
    const n = layOf(d_lay).rows;
    const m = layOf(d_lay).cols;
    const al = @as(*const f32, @ptrCast(@alignCast(alpha.?))).*;
    const be = @as(*const f32, @ptrCast(@alignCast(beta.?))).*;
    const w = @as([*]const u16, @ptrFromInt(a_dev));
    const x = @as([*]const u16, @ptrFromInt(x_dev));
    const y = @as([*]f32, @ptrFromInt(d_dev));
    for (0..m) |i| for (0..n) |j| {
        var sum: f32 = 0;
        for (0..k) |l| sum += bf.value(x[i * k + l]) * bf.value(w[j * k + l]);
        if (d.epilogue == bias_tag) sum += @as([*]const f32, @ptrFromInt(d.bias))[j];
        y[i * n + j] = al * sum + be * y[i * n + j];
    };
    return 0;
}

fn api() cublaslt.Api {
    var a: cublaslt.Api = undefined;
    a.cublasLtCreate = ltCreate;
    a.cublasLtDestroy = ltDestroy;
    a.cublasLtGetStatusName = ltStatusName;
    a.cublasLtMatmulDescCreate = descCreate;
    a.cublasLtMatmulDescDestroy = descDestroy;
    a.cublasLtMatmulDescSetAttribute = descSet;
    a.cublasLtMatrixLayoutCreate = layoutCreate;
    a.cublasLtMatrixLayoutDestroy = layoutDestroy;
    a.cublasLtMatmulPreferenceCreate = prefCreate;
    a.cublasLtMatmulPreferenceDestroy = prefDestroy;
    a.cublasLtMatmulPreferenceSetAttribute = prefSet;
    a.cublasLtMatmulAlgoGetHeuristic = getHeuristic;
    a.cublasLtMatmulAlgoCheck = algoCheck;
    a.cublasLtMatmul = matmul;
    return a;
}

fn dev(v: anytype) u64 {
    return @intCast(@intFromPtr(v));
}

fn drain(comptime Products: type, p: *Products) void {
    var it = p.plans.valueIterator();
    while (it.next()) |l| l.deinit();
    p.plans.deinit(p.gpa);
}

pub fn check(comptime Products: type, comptime run: anytype) !void {
    const a = std.testing.allocator;
    const stream: cuda.Stream = .{ .d = undefined, .handle = null };
    var ws: [64]u8 align(256) = undefined;
    const x = [_]u16{ bf.bits(1), bf.bits(2), bf.bits(-1), bf.bits(0.5), bf.bits(0), bf.bits(1), bf.bits(0.25), bf.bits(-2) };
    const w = [_]u16{ bf.bits(1), 0, 0, 0, bf.bits(0.5), bf.bits(1), bf.bits(-1), bf.bits(2), bf.bits(-1), bf.bits(0.5), bf.bits(0.5), bf.bits(1) };
    const bias = [_]f32{ 10, -0.5, 0.25 };
    var y: [6]f32 = @splat(0);
    const xp = dev(&x);
    const wp = dev(&w);
    const bp = dev(&bias);
    const yp = dev(&y);
    // The actual-row algorithm declines bias; the canonical algorithm accepts it only on a bias descriptor.
    reset(true);
    var p: Products = .{ .lt = .{ .lib = undefined, .api = api() }, .gpa = a, .workspace = .{ .d = undefined, .ptr = dev(&ws), .len = ws.len, .borrowed = true } };
    defer drain(Products, &p);
    try run(&p, stream, xp, wp, bp, yp, 2, 3, 4);
    try std.testing.expectEqual(@as(usize, 1), matmul_calls);
    try std.testing.expectEqual(bias_tag, last_epilogue);
    try std.testing.expectEqual(bp, last_bias);
    try std.testing.expectEqualSlices(f32, &.{ 11, 4, 0.25, 10, -3.75, -1.125 }, &y);
    const calls = heuristic_calls;
    y = @splat(0);
    try run(&p, stream, xp, wp, bp, yp, 2, 3, 4);
    try std.testing.expectEqual(calls, heuristic_calls); // the cached plan, still fused
    try std.testing.expectEqual(bias_tag, last_epilogue);
    try std.testing.expectEqualSlices(f32, &.{ 11, 4, 0.25, 10, -3.75, -1.125 }, &y);
    try run(&p, stream, xp, wp, 0, yp, 2, 3, 4); // no bias: the same shape unfused
    try std.testing.expectEqual(@as(u32, @backingInt(cublaslt.Epilogue.none)), last_epilogue);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4.5, 0, 0, -3.25, -1.375 }, &y);
    // the canonical algorithm not supporting .bias either: refused, nothing cached, nothing run
    reset(false);
    var q: Products = .{ .lt = .{ .lib = undefined, .api = api() }, .gpa = a, .workspace = .{ .d = undefined, .ptr = dev(&ws), .len = ws.len, .borrowed = true } };
    defer drain(Products, &q);
    try std.testing.expectError(error.NoAlgorithm, run(&q, stream, xp, wp, bp, yp, 2, 3, 4));
    try std.testing.expectEqual(@as(u32, 0), q.plans.count());
    try std.testing.expectEqual(@as(usize, 0), matmul_calls);
}
