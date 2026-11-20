//! GB10's native block-scaled NVFP4 products against the int8 path on the picture's four projection shapes:
//! e2m1 operands at one e4m3 scale every 16 values of k (cuBLASLt VEC16_UE4M3, fp32 sums, bf16 out) against
//! the heuristic-first int8 TN product (int32 sums) the transformer ships. Card 147's stop rule: if the FP4
//! product is not faster, no quantized path is built. `dtype` and `scale_mode` are CUDA_R_4F_E2M1 and
//! CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3's header values, passed by the job so a wrong enum cannot fake a
//! result; the plan refuses to run when cuBLASLt offers no algorithm for the shape.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const lt_ = cuda.cublaslt;

const workspace_bytes = 32 << 20;

/// The four projections a 2048² picture steps through, (name, rows M, n N, k K), as card 129's bench had them.
const cases = [_]struct { name: []const u8, rows: usize, n: usize, k: usize }{
    .{ .name = "qkvc", .rows = 4120, .n = 21504, .k = 5376 },
    .{ .name = "out ", .rows = 4120, .n = 5376, .k = 7168 },
    .{ .name = "fc1 ", .rows = 4120, .n = 28672, .k = 5376 },
    .{ .name = "fc2 ", .rows = 4120, .n = 5376, .k = 14336 },
};

/// One planned block-scaled FP4 product: D^T (n, rows) bf16 = W (k, n)^T . X^T (k, rows), scales in the desc.
const Fp4 = struct {
    lt: *const lt_.Library,
    handle: lt_.Handle,
    desc: lt_.MatmulDesc,
    a: lt_.Layout,
    b: lt_.Layout,
    c: lt_.Layout,
    algo: lt_.Algo,

    fn init(lt: *const lt_.Library, handle: lt_.Handle, dtype: c_int, scale_mode: c_int, rows: usize, n: usize, k: usize, scales_a: u64, scales_b: u64) !Fp4 {
        const api = lt.api;
        var f: Fp4 = .{ .lt = lt, .handle = handle, .desc = null, .a = null, .b = null, .c = null, .algo = undefined };
        try lt.check(api.cublasLtMatmulDescCreate(&f.desc, lt_.compute_32f, .f32), "cublasLtMatmulDescCreate");
        errdefer _ = api.cublasLtMatmulDescDestroy(f.desc);
        const ta: c_int = @backingInt(lt_.Op.t);
        const tb: c_int = @backingInt(lt_.Op.n);
        try lt.check(api.cublasLtMatmulDescSetAttribute(f.desc, lt_.desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(api.cublasLtMatmulDescSetAttribute(f.desc, lt_.desc_transb, &tb, @sizeOf(c_int)), "set transb");
        // the heuristic refuses a block-scaled plan without its scales
        for ([_]c_int{ lt_.desc_a_scale_mode, lt_.desc_b_scale_mode }) |m|
            try lt.check(api.cublasLtMatmulDescSetAttribute(f.desc, m, &scale_mode, @sizeOf(c_int)), "set scale mode");
        for ([_]struct { c_int, u64 }{ .{ lt_.desc_a_scale_pointer, scales_a }, .{ lt_.desc_b_scale_pointer, scales_b } }) |p|
            try lt.check(api.cublasLtMatmulDescSetAttribute(f.desc, p[0], &p[1], @sizeOf(u64)), "set scale pointer");
        const t: lt_.DataType = @fromBackingInt(@intCast(dtype));
        try lt.check(api.cublasLtMatrixLayoutCreate(&f.a, t, k, n, @intCast(k)), "layout W");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(f.a);
        try lt.check(api.cublasLtMatrixLayoutCreate(&f.b, t, k, rows, @intCast(k)), "layout X");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(f.b);
        try lt.check(api.cublasLtMatrixLayoutCreate(&f.c, .bf16, n, rows, @intCast(n)), "layout D");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(f.c);
        var pref: lt_.Preference = null;
        try lt.check(api.cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        defer _ = api.cublasLtMatmulPreferenceDestroy(pref);
        const limit: u64 = workspace_bytes;
        try lt.check(api.cublasLtMatmulPreferenceSetAttribute(pref, lt_.pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        var found: [1]lt_.Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(api.cublasLtMatmulAlgoGetHeuristic(handle, f.desc, f.a, f.b, f.c, f.c, pref, 1, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        f.algo = found[0].algo;
        return f;
    }

    fn deinit(f: *Fp4) void {
        const api = f.lt.api;
        _ = api.cublasLtMatrixLayoutDestroy(f.c);
        _ = api.cublasLtMatrixLayoutDestroy(f.b);
        _ = api.cublasLtMatrixLayoutDestroy(f.a);
        _ = api.cublasLtMatmulDescDestroy(f.desc);
    }

    fn run(f: *const Fp4, w: u64, x: u64, d: u64, ws: u64, stream: cuda.Stream) !void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try f.lt.check(f.lt.api.cublasLtMatmul(f.handle, f.desc, &one, w, f.a, x, f.b, &zero, d, f.c, d, f.c, &f.algo, ws, workspace_bytes, stream.handle), "cublasLtMatmul");
    }
};

fn timeIt(comptime f: anytype, args: anytype, s: cuda.Stream, reps: usize, io: std.Io) !i96 {
    const start = check.now(io);
    for (0..reps) |_| try @call(.auto, f, args);
    try s.synchronize();
    return @divTrunc(check.now(io) - start, @as(i96, @intCast(reps)));
}

pub fn run(gpu: check.Gpu, dtype: c_int, scale_mode: c_int) !void {
    const io = gpu.io;
    const cap = try gpu.ctx.capability();
    try check.expect(cap == 121, "h3 fp4 benchmark needs sm_121, got sm_{d}", .{cap});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const a = gpu.gpa;

    var lt = try lt_.Library.open();
    defer lt.close();
    const handle = try lt_.createHandle(&lt);
    defer _ = lt.api.cublasLtDestroy(handle);
    var ws = try cuda.DeviceBuffer.alloc(gpu.d, workspace_bytes);
    defer ws.free();
    std.debug.print("BENCH h3_fp4 cublasLt {d} dtype {d} scale_mode {d} sm_{d}\n", .{ lt.api.cublasLtGetVersion(), dtype, scale_mode, cap });

    const reps = 5;
    for (cases, 0..) |sh, i| {
        var prng = std.Random.DefaultPrng.init(0x147 + i);
        const rng = prng.random();
        // int8 operands and sums, the shipping path's buffers
        const xq = try a.alloc(i8, sh.rows * sh.k);
        defer a.free(xq);
        const wq = try a.alloc(i8, sh.n * sh.k);
        defer a.free(wq);
        for (xq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
        for (wq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
        // e2m1 pairs and e4m3 scales: bytes only, the timing is the gate
        const w4 = try a.alloc(u8, sh.n * sh.k / 2);
        defer a.free(w4);
        const x4 = try a.alloc(u8, sh.rows * sh.k / 2);
        defer a.free(x4);
        const wsc = try a.alloc(u8, lt_.scaleBytes(sh.n, sh.k));
        defer a.free(wsc);
        const xsc = try a.alloc(u8, lt_.scaleBytes(sh.rows, sh.k));
        defer a.free(xsc);
        for (w4) |*v| v.* = rng.int(u8);
        for (x4) |*v| v.* = rng.int(u8);
        for (wsc) |*v| v.* = rng.int(u8) & 0x7e; // any finite e4m3
        for (xsc) |*v| v.* = rng.int(u8) & 0x7e;

        var d_xq = try cuda.DeviceBuffer.alloc(gpu.d, xq.len);
        defer d_xq.free();
        var d_wq = try cuda.DeviceBuffer.alloc(gpu.d, wq.len);
        defer d_wq.free();
        var d_w4 = try cuda.DeviceBuffer.alloc(gpu.d, w4.len);
        defer d_w4.free();
        var d_x4 = try cuda.DeviceBuffer.alloc(gpu.d, x4.len);
        defer d_x4.free();
        var d_ws = try cuda.DeviceBuffer.alloc(gpu.d, wsc.len);
        defer d_ws.free();
        var d_xs = try cuda.DeviceBuffer.alloc(gpu.d, xsc.len);
        defer d_xs.free();
        var d_acc = try cuda.DeviceBuffer.alloc(gpu.d, sh.rows * sh.n * 4);
        defer d_acc.free();
        var d_out = try cuda.DeviceBuffer.alloc(gpu.d, sh.rows * sh.n * 2);
        defer d_out.free();
        try d_xq.upload(0, std.mem.sliceAsBytes(xq));
        try d_wq.upload(0, std.mem.sliceAsBytes(wq));
        try d_w4.upload(0, w4);
        try d_x4.upload(0, x4);
        try d_ws.upload(0, wsc);
        try d_xs.upload(0, xsc);

        var fp4 = Fp4.init(&lt, handle, dtype, scale_mode, sh.rows, sh.n, sh.k, d_ws.ptr, d_xs.ptr) catch |err| {
            std.debug.print("BENCH h3_fp4 {s} M={d} N={d} K={d}: NO ALGORITHM ({s})\n", .{ sh.name, sh.rows, sh.n, sh.k, @errorName(err) });
            return err;
        };
        defer fp4.deinit();

        var int8 = try lt_.Gemm.init(&lt, handle, .{ .ta = .t, .m = sh.n, .n = sh.rows, .k = sh.k, .a = .i8, .b = .i8, .c = .i32, .lda = sh.k, .ldb = sh.k, .ldc = sh.n }, workspace_bytes);
        defer int8.deinit();

        try int8.runInt(d_wq.ptr, d_xq.ptr, d_acc.ptr, ws.ptr, workspace_bytes, s.handle);
        try fp4.run(d_w4.ptr, d_x4.ptr, d_out.ptr, ws.ptr, s);
        try s.synchronize();
        const i8ns = try timeIt(lt_.Gemm.runInt, .{ &int8, d_wq.ptr, d_xq.ptr, d_acc.ptr, ws.ptr, workspace_bytes, s.handle }, s, reps, io);
        const f4ns = try timeIt(Fp4.run, .{ &fp4, d_w4.ptr, d_x4.ptr, d_out.ptr, ws.ptr, s }, s, reps, io);

        const flops: f64 = 2.0 * @as(f64, @floatFromInt(sh.rows)) * @as(f64, @floatFromInt(sh.n)) * @as(f64, @floatFromInt(sh.k));
        const i8ms = @as(f64, @floatFromInt(i8ns)) / std.time.ns_per_ms;
        const f4ms = @as(f64, @floatFromInt(f4ns)) / std.time.ns_per_ms;
        std.debug.print("BENCH h3_fp4 {s} M={d} N={d} K={d}: int8={d:.2} ms ({d:.1} TOPS) fp4={d:.2} ms ({d:.1} TOPS) int8/fp4={d:.2}x\n", .{
            sh.name, sh.rows, sh.n, sh.k, i8ms, flops / (i8ms * 1e9), f4ms, flops / (f4ms * 1e9), i8ms / f4ms,
        });
    }
}
