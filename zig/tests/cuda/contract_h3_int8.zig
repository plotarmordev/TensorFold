//! MiniMax H3's int8 projection kernels (h3_quantize, h3_gemm_int8_tma) under the contract, against float64 formulas.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Order = g.Order;
const sumIn = core.contract.rows.sumIn;

/// The rows' widths the blocks quantize: hidden (q k v gate, mlp in), attention out (56 heads of 128), mlp out.
const hidden = 5376;
const inner = 7168;
const mlp = 14336;

/// A rotated value: the lane's 4-point transform, five shuffle butterflies, then the 1 / sqrt(128) product.
pub fn rotationChain(_: c.Shape) usize {
    return 2 + 5 + 1;
}
fn none(_: c.Shape) usize {
    return 0;
}

const quantize_rule = "a row with an infinity scales by +inf and quantizes to zeros; a NaN's chunk quantizes to zeros and leaves the row's absmax";

/// One scale a row; rows and k (the 128-wide chunks) are its dimensions, n repeats k.
pub fn quantizeSpec() c.Spec {
    return .{
        .name = "cuda h3 int8 rows h3_quantize",
        .accum = .f32,
        .depth = rotationChain,
        .ops = 2,
        .out = .f32,
        .nan = .{ .stated = quantize_rule },
        .tiles = .{ .{}, .{}, .{ .size = 1024, .step = 128, .min = 128 } },
        .served = &.{ .{ .rows = 16, .n = hidden, .k = hidden }, .{ .rows = 16, .n = inner, .k = inner }, .{ .rows = 16, .n = mlp, .k = mlp } },
        .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "an absmax over 128-term rotations: one fp32 scale a row, its codes checked against it" },
    };
}

/// The Sylvester Hadamard sign of row `i`, column `j`: h3.cu's lane transform then its shuffle butterflies.
pub fn sign(i: usize, j: usize) f64 {
    return if (@popCount(i & j) % 2 == 0) 1 else -1;
}

pub const Quantize = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    xd: u64 = 0,
    q: u64 = 0,
    s: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(t: *Quantize) void {
        t.rig.deinit();
        t.arena.deinit();
    }

    pub fn kernel(t: *Quantize) g.Kernel {
        return .{ .spec = quantizeSpec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Quantize);

    pub fn prepare(t: *Quantize, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 128 != 0 or shape.k == 0) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        t.shape = shape;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.arena.allocator().alloc(f64, shape.rows * k);
        const h = try t.arena.allocator().alloc(u16, t.x.len);
        for (t.x, 0..) |*v, i| {
            const r = rng.float(f64) * 4 - 2;
            v.* = if (fill == .stress) @round(r * 8) else r * @as(f64, if (i % 97 == 0) 16 else 1);
        }
        // a zero row: the scale's `most > 0` arm
        if (shape.rows > 3) @memset(t.x[3 * k ..][0..k], 0);
        if (fill == .nonfinite) for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, row| {
            t.x[row * k + @min(k - 1, 130)] = v;
        };
        for (h, t.x) |*o, *v| {
            o.* = c.toBf16(@floatCast(v.*));
            v.* = c.fromBf16(o.*);
        }
        t.xd = try t.rig.input(std.mem.sliceAsBytes(h));
        t.s = try t.rig.output(shape.rows * 4);
        t.q = try t.rig.output(shape.rows * k);
        return true;
    }

    pub fn run(t: *Quantize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ t.xd, t.q, t.s }) |p| args.add(p);
        args.add(@as(c_int, @intCast(t.shape.k)));
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(t.shape.rows) }, .block = .{ .x = 256 } }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Quantize) usize {
        return t.shape.rows;
    }

    /// Rotated value `j` of chunk `ch` in row `row`, summed as `acc` in `order` (float64 when .none), and its magnitude.
    fn rotated(t: *Quantize, terms: []f64, row: usize, ch: usize, j: usize, acc: c.Accum, order: Order) [2]f64 {
        const x = t.x[row * t.shape.k + ch * 128 ..][0..128];
        var mag: f64 = 0;
        for (terms, x, 0..) |*o, v, i| {
            o.* = sign(i, j) * v;
            mag += @abs(v);
        }
        var sum: f64 = 0;
        if (acc == .none) {
            for (terms) |v| sum += v;
        } else sum = sumIn(acc, order, terms);
        return .{ sum / @sqrt(128.0), mag / @sqrt(128.0) };
    }

    /// Each row's absmax / 127 over its finite chunks, and the largest chunk magnitude / 127 as the bound's mag.
    fn scales(t: *Quantize, acc: c.Accum, order: Order, want: []f64, mag: []f64) void {
        var terms: [128]f64 = undefined;
        for (want, mag, 0..) |*w, *m, row| {
            var most: f64 = 0;
            var big: f64 = 0;
            for (0..t.shape.k / 128) |ch| for (0..128) |j| {
                const v = t.rotated(&terms, row, ch, j, acc, order);
                if (!std.math.isNan(v[0])) most = @max(most, @abs(v[0]));
                if (std.math.isFinite(v[1])) big = @max(big, v[1]);
            };
            w.* = if (most > 0) most / 127 else 1;
            m.* = big / 127;
        }
    }

    pub fn reference(t: *Quantize, want: []f64, mag: []f64) !void {
        t.scales(.none, .sequential, want, mag);
    }

    pub fn simulate(t: *Quantize, acc: c.Accum, order: Order, out: []f64) !void {
        const mag = try t.arena.allocator().alloc(f64, out.len);
        t.scales(acc, order, out, mag);
    }

    /// The scales within their bound, then every code the nearest integer to its rotated value over the row's own scale,
    /// within a code where the rotation's fp32 error reaches a rounding tie; the stated rule at planted rows.
    pub fn check(t: *Quantize, at: g.Gate) !?[]const u8 {
        const s = quantizeSpec();
        const rows = t.shape.rows;
        const k = t.shape.k;
        const a = t.arena.allocator();
        const want = try a.alloc(f64, rows);
        const mag = try a.alloc(f64, rows);
        try t.reference(want, mag);
        const got_s = try a.dupe(u8, try t.rig.bytes(t.s, rows * 4));
        const got_q = std.mem.bytesAsSlice(i8, try t.rig.bytes(t.q, rows * k));
        var terms: [128]f64 = undefined;
        var planted: usize = 0;
        for (0..rows) |row| {
            const scale = c.read(.f32, got_s, row);
            var inf = false;
            for (t.x[row * k ..][0..k]) |v| inf = inf or std.math.isInf(v);
            if (inf) {
                planted += 1;
                if (scale != std.math.inf(f64)) return try std.fmt.bufPrint(&t.msg, "row {d} holds an infinity but scales by {e}", .{ row, scale });
                for (got_q[row * k ..][0..k]) |q| if (q != 0) return try std.fmt.bufPrint(&t.msg, "row {d} holds an infinity but a code is {d}", .{ row, q });
                continue;
            }
            if (!(@abs(scale - want[row]) <= c.bound(s, t.shape, want[row], mag[row]))) return try std.fmt.bufPrint(&t.msg, "row {d} scales by {e} where float64 gives {e}", .{ row, scale, want[row] });
            for (0..k / 128) |ch| for (0..128) |j| {
                const q: f64 = @floatFromInt(got_q[row * k + ch * 128 + j]);
                const v = t.rotated(&terms, row, ch, j, .none, .sequential);
                if (std.math.isNan(v[0])) {
                    planted += 1;
                    if (q != 0) return try std.fmt.bufPrint(&t.msg, "row {d} chunk {d} holds a NaN but code {d} is {d}", .{ row, ch, j, q });
                    continue;
                }
                const exact = v[0] / scale;
                const slack = (c.gamma(@floatFromInt(rotationChain(t.shape)), 0x1p-24) * v[1] + 0x1p-22 * @abs(v[0])) / scale;
                if (@abs(q) > 127 or @abs(q - exact) > 0.5 + slack) return try std.fmt.bufPrint(&t.msg, "row {d} code {d} is {d} where float64 gives {d:.4}", .{ row, ch * 128 + j, q, exact });
            };
        }
        if (at == .nonfinite and planted == 0) return "no planted value reached a launched row: no proof";
        return null;
    }
};

// ---------------------------------------------------------------------------
// The TMA product (h3_int8_tma.cuh): 256x128 tiles walked by one CTA an SM, tensor maps encoded
// per launch. Its contract gates the exact int32 sums plus the fp32 scale fold against float64.

const gpu_check = @import("check.zig");

/// The driver encodes a [rows, columns] matrix into a CUtensorMap (cuda_int8.zig's TensorMap).
const TensorMap = extern struct { bytes: [128]u8 align(64) };
const Encode = *const fn (*TensorMap, c_uint, c_uint, ?*anyopaque, *const [2]u64, *const u64, *const [2]u32, *const [2]u32, c_uint, c_uint, c_uint, c_uint) callconv(.c) cuda.abi.Result;

/// cuTensorMapEncodeTiled, bound once: a driver without it cannot run the product at all.
pub fn encoder(d: *const cuda.Driver) !Encode {
    var lib = d.lib;
    return lib.lookup(Encode, "cuTensorMapEncodeTiled") orelse error.MissingSymbol;
}

/// UINT8, 2D, swizzle 128B, L2 256B promotion, zero fill: cuda_int8.zig's encode, here for the
/// contract's own buffers.
fn encodeTiled(enc: Encode, map: *TensorMap, ptr: u64, rows: usize, k: usize, box_rows: u32) !void {
    const dims: [2]u64 = .{ k, rows };
    const strides: u64 = k;
    const box: [2]u32 = .{ 128, box_rows };
    const es: [2]u32 = .{ 1, 1 };
    const res = enc(map, 0, 2, @ptrFromInt(@as(usize, @intCast(ptr))), &dims, &strides, &box, &es, 0, 3, 3, 0);
    if (res != cuda.abi.success) return error.CudaFailed;
}

/// TallTile's ring (the only tile the picture uses): 2 stages of (256 + 128) x 128 bytes.
const tma_shared_bytes = 2 * (256 + 128) * 128 + 1024 + 2 * 16;

/// A TMA product kernel with its ring's dynamic shared memory allowed (past the 48 KiB default).
pub fn tmaFunction(mod: cuda.Module, name: [:0]const u8) !cuda.Function {
    const f = try mod.function(name);
    try f.allowDynamicShared(tma_shared_bytes);
    return f;
}

/// What a GemmTma registration launches: the plain product, or fc1's SwiGLU epilogue on interleaved rows.
pub const Mode = enum { plain, swiglu };

/// One output a product element: bf16(acc * xs[row] * ws[col]), the int32 sums exact; SwiGLU's
/// is h3_swiglu's on the plain product's [gate | up] halves, bit for bit.
pub fn gemmTmaSpec(mode: Mode) c.Spec {
    const served: []const c.Shape = switch (mode) {
        .plain => &.{
            .{ .rows = 16, .n = 128, .k = 128 }, // one tile, one stage
            .{ .rows = 300, .n = 128, .k = 256 }, // a partial M tile and two K blocks
            .{ .rows = 64, .n = 384, .k = 384 }, // three N tiles past a partial M tile
            .{ .rows = 16, .n = 392, .k = 128 }, // an 8-column edge tile
            .{ .rows = 300, .n = 1352, .k = 256 }, // a 72-column edge tile past partial M and two K blocks
            .{ .rows = 16, .n = 21504, .k = 5376 }, // qkvc's real fan-out and depth
        },
        .swiglu => &.{
            .{ .rows = 16, .n = 256, .k = 128 }, // one feature tile
            .{ .rows = 300, .n = 512, .k = 256 }, // a partial M tile, two tiles and K blocks
            .{ .rows = 300, .n = 776, .k = 256 }, // an 8-row (4-feature) edge tile past partial M and two K blocks
            .{ .rows = 256, .n = 2 * mlp, .k = hidden }, // fc1's real fan-out
        },
    };
    return .{
        .name = switch (mode) {
            .plain => "cuda h3 int8 tma product h3_gemm_int8_tma",
            .swiglu => "cuda h3 int8 tma swiglu h3_swiglu_rows + h3_swiglu_int8_tma",
        },
        .accum = .none, // int32 sums, exact: no fp accumulator for gate 4 to stress
        .depth = none,
        .ops = 3,
        .out = .bf16,
        .nan = .{ .stated = "a nonfinite row or column scale reaches every output it multiplies, as h3_dequantize's did" },
        // n % 8: a lane's eight output columns are one 16-byte store (SwiGLU's four features, one 8-byte store)
        .tiles = switch (mode) {
            .plain => .{ .{}, .{ .size = 128, .step = 8, .min = 8 }, .{ .size = 128, .step = 128, .min = 128 } },
            .swiglu => .{ .{}, .{ .size = 256, .step = 8, .min = 8 }, .{ .size = 128, .step = 128, .min = 128 } },
        },
        .served = served,
        .replaces = if (mode == .swiglu) "h3_gemm_int8_tma then h3_swiglu" else null,
        .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "mma.sync sums the codes' products in int32, exact to a bit; fp32 carries only the acc * xs * ws fold and the bf16 store" },
    };
}

/// C (rows, n) bf16 = (X int8 rows,k . W int8 n,k)^T * xs[rows] * ws[n], as the TMA kernel writes
/// it; in swiglu mode W's rows are [gate | up], interleaved on the device as fc1's are, and C is
/// silu(gate) * up, (rows, n / 2).
pub const GemmTma = struct {
    rig: rig_.Rig,
    mode: Mode,
    mod: cuda.Module,
    enc: Encode,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    xq: []i8 = &.{},
    wq: []i8 = &.{},
    xs: []f32 = &.{},
    ws: []f32 = &.{},
    /// xq, xs, wq, ws, out; swiglu adds the interleaved wq, ws and the baseline's two buffers.
    dev: [9]u64 = @splat(0),

    pub fn deinit(t: *GemmTma) void {
        t.rig.deinit();
        t.arena.deinit();
    }

    pub fn kernel(t: *GemmTma) g.Kernel {
        return .{ .spec = gemmTmaSpec(t.mode), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(GemmTma);

    fn outColumns(t: *const GemmTma) usize {
        return if (t.mode == .swiglu) t.shape.n / 2 else t.shape.n;
    }

    pub fn prepare(t: *GemmTma, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 128 != 0 or shape.n % 8 != 0 or shape.k == 0 or shape.n == 0 or shape.rows == 0) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        t.shape = shape;
        const a = t.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.xq = try a.alloc(i8, shape.rows * shape.k);
        t.wq = try a.alloc(i8, shape.n * shape.k);
        t.xs = try a.alloc(f32, shape.rows);
        t.ws = try a.alloc(f32, shape.n);
        for (t.xq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
        for (t.wq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
        for (t.xs) |*v| v.* = 0.001 + 0.004 * rng.float(f32);
        for (t.ws) |*v| v.* = 0.002 + 0.006 * rng.float(f32);
        if (fill == .stress) {
            for (t.xq) |*v| v.* = if (v.* >= 0) 127 else -127;
        }
        if (fill == .nonfinite) {
            if (shape.rows > 2) t.xs[2] = std.math.nan(f32);
            if (shape.n > 3) t.ws[3] = std.math.inf(f32);
        }
        t.dev[0] = try t.rig.input(std.mem.sliceAsBytes(t.xq));
        t.dev[1] = try t.rig.input(std.mem.sliceAsBytes(t.xs));
        t.dev[2] = try t.rig.input(std.mem.sliceAsBytes(t.wq));
        t.dev[3] = try t.rig.input(std.mem.sliceAsBytes(t.ws));
        t.dev[4] = try t.rig.output(shape.rows * t.outColumns() * 2);
        if (t.mode == .swiglu) {
            t.dev[5] = try t.rig.scratch(shape.n * shape.k);
            t.dev[6] = try t.rig.scratch(shape.n * 4);
            t.dev[7] = try t.rig.scratch(shape.rows * shape.n * 2);
            t.dev[8] = try t.rig.scratch(shape.rows * shape.n);
        }
        return true;
    }

    /// One TMA product launch: `wq` and `ws` the weight rows and scales it reads, into `out`.
    fn product(t: *GemmTma, name: [:0]const u8, wq: u64, ws: u64, out: u64) !void {
        var maps: [2]TensorMap = undefined;
        try encodeTiled(t.enc, &maps[0], t.dev[0], t.shape.rows, t.shape.k, 256);
        try encodeTiled(t.enc, &maps[1], wq, t.shape.n, t.shape.k, 128);
        var args: cuda.Args = .{};
        args.add(maps[0]);
        args.add(maps[1]);
        for ([_]u64{ t.dev[1], ws, out }) |p| args.add(p);
        for ([_]usize{ t.shape.rows, t.shape.n, t.shape.k }) |v| args.add(@as(c_int, @intCast(v)));
        const sm: usize = @intCast(try t.rig.gpu.ctx.attribute(.multiprocessor_count));
        const tiles = (t.shape.rows + 255) / 256 * ((t.shape.n + 127) / 128);
        try cuda.launch.launch(try tmaFunction(t.mod, name), .{ .grid = .{ .x = @intCast(@min(sm, tiles)) }, .block = .{ .x = 256 }, .shared = tma_shared_bytes }, t.rig.s, &args);
    }

    pub fn run(t: *GemmTma) !void {
        if (t.mode == .swiglu) {
            var args: cuda.Args = .{};
            for ([_]u64{ t.dev[2], t.dev[3], t.dev[5], t.dev[6] }) |p| args.add(p);
            args.add(@as(c_int, @intCast(t.shape.k)));
            try cuda.launch.launch(try t.mod.function("h3_swiglu_rows"), .{ .grid = .{ .x = @intCast(t.shape.n / 2) }, .block = .{ .x = 256 } }, t.rig.s, &args);
            try t.product("h3_swiglu_int8_tma", t.dev[5], t.dev[6], t.dev[4]);
        } else try t.product("h3_gemm_int8_tma", t.dev[2], t.dev[3], t.dev[4]);
        try t.rig.s.synchronize();
    }

    /// SwiGLU's baseline: the plain product on the original [gate | up] rows, then h3_swiglu.
    pub fn replaced(t: *GemmTma) ![]const u8 {
        try t.product("h3_gemm_int8_tma", t.dev[2], t.dev[3], t.dev[7]);
        var args: cuda.Args = .{};
        for ([_]u64{ t.dev[7], t.dev[8] }) |p| args.add(p);
        for ([_]usize{ t.shape.rows, t.shape.n / 2 }) |v| args.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(try t.mod.function("h3_swiglu"), .{ .grid = .{ .x = 256 }, .block = .{ .x = 256 } }, t.rig.s, &args);
        return t.rig.bytes(t.dev[8], t.shape.rows * t.shape.n);
    }

    pub fn count(t: *GemmTma) usize {
        return t.shape.rows * t.outColumns();
    }

    /// The float64 product: exact int32 dots over the whole row (the mma order is irrelevant to
    /// them), then acc * xs * ws in fp32 rounded to bf16 as the store does; SwiGLU then rounds
    /// silu(gate) and silu * up to bf16 as h3_swiglu does.
    pub fn reference(t: *GemmTma, want: []f64, mag: []f64) !void {
        const k = t.shape.k;
        const n = t.shape.n;
        const cols = t.outColumns();
        const half = n / 2;
        for (0..t.shape.rows) |row| for (0..cols) |col| {
            const x = t.xq[row * k ..][0..k];
            const at = row * cols + col;
            if (t.mode != .swiglu) {
                const dot = dots(x, t.wq[col * k ..][0..k]);
                want[at] = c.fromBf16(c.toBf16(@as(f32, @floatFromInt(dot[0])) * t.xs[row] * t.ws[col]));
                mag[at] = @as(f64, @floatFromInt(dot[1])) * t.xs[row] * t.ws[col];
                continue;
            }
            const gd = dots(x, t.wq[col * k ..][0..k]);
            const ud = dots(x, t.wq[(col + half) * k ..][0..k]);
            const gate = c.fromBf16(c.toBf16(@as(f32, @floatFromInt(gd[0])) * t.xs[row] * t.ws[col]));
            const up = c.fromBf16(c.toBf16(@as(f32, @floatFromInt(ud[0])) * t.xs[row] * t.ws[col + half]));
            const g32: f32 = @floatCast(gate);
            const silu = c.fromBf16(c.toBf16(g32 / (1 + @exp(-g32))));
            want[at] = c.fromBf16(c.toBf16(@floatCast(silu * up)));
            mag[at] = @abs(want[at]);
        };
    }
};

/// A row's int32 dot with a weight row and the sum of its terms' magnitudes.
fn dots(x: []const i8, w: []const i8) [2]i64 {
    var dot: i64 = 0;
    var sum: i64 = 0;
    for (x, w) |a, b| {
        const p = @as(i64, a) * b;
        dot += p;
        sum += @intCast(@abs(p));
    }
    return .{ dot, sum };
}

// ---------------------------------------------------------------------------
// Microbench: the TMA tile against cuBLASLt's int8 product on the real picture
// shapes — the gate the card sets before any picture run.

const lt_ = cuda.cublaslt;
const workspace_bytes = 32 << 20; // cuda_gemm's own; duplicated so the test needs no family import

const image_h3 align(16) = @embedFile("fatbin_h3").*;

/// Per-shape wall-clock of `reps` back-to-back launches of `f`, ns each.
fn timeKernel(f: cuda.Function, s: cuda.Stream, shared: u32, args: *cuda.Args, grid_x: u32, reps: usize, io: std.Io) !i96 {
    const start = gpu_check.now(io);
    for (0..reps) |_| try cuda.launch.launch(f, .{ .grid = .{ .x = grid_x }, .block = .{ .x = 256 }, .shared = shared }, s, args);
    try s.synchronize();
    return @divTrunc(gpu_check.now(io) - start, @as(i96, @intCast(reps)));
}

/// Per-shape wall-clock of `reps` cuBLASLt int8 products (C i32), ns each.
fn timeLt(gm: *const lt_.Gemm, a: u64, b: u64, cc: u64, w: u64, s: cuda.Stream, reps: usize, io: std.Io) !i96 {
    const start = gpu_check.now(io);
    for (0..reps) |_| try gm.runInt(a, b, cc, w, workspace_bytes, s.handle);
    try s.synchronize();
    return @divTrunc(gpu_check.now(io) - start, @as(i96, @intCast(reps)));
}

/// A/B the picture projections on the same device buffers: `h3_gemm_int8_tma` writes bf16 from its
/// own persistent tiles; cuBLASLt sums into int32 (what the old path paid h3_dequantize to read).
pub fn bench(gpu: gpu_check.Gpu) !void {
    const io = gpu.io;
    const cap = try gpu.ctx.capability();
    try gpu_check.expect(cap == 121, "h3 int8 tma benchmark needs sm_121, got sm_{d}", .{cap});
    const enc = try encoder(gpu.d);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const a = gpu.gpa;

    var mod = try cuda.Module.load(gpu.d, &image_h3);
    defer mod.unload();
    const f = try tmaFunction(mod, "h3_gemm_int8_tma");

    var lt = try lt_.Library.open();
    defer lt.close();
    const handle = try lt_.createHandle(&lt);
    var ws = try cuda.DeviceBuffer.alloc(gpu.d, workspace_bytes);
    defer ws.free();

    // the four projections, (name, rows M, n N, k K): qkvc, out, fc1, fc2
    const cases = [_]struct { name: []const u8, rows: usize, n: usize, k: usize }{
        .{ .name = "qkvc", .rows = 4120, .n = 21504, .k = 5376 },
        .{ .name = "out ", .rows = 4120, .n = 5376, .k = 7168 },
        .{ .name = "fc1 ", .rows = 4120, .n = 28672, .k = 5376 },
        .{ .name = "fc2 ", .rows = 4120, .n = 5376, .k = 14336 },
    };
    const reps = 5;
    for (cases, 0..) |sh, i| {
        var prng = std.Random.DefaultPrng.init(0x1398 + i);
        const rng = prng.random();
        const xq = try a.alloc(i8, sh.rows * sh.k);
        defer a.free(xq);
        const wq = try a.alloc(i8, sh.n * sh.k);
        defer a.free(wq);
        const xs = try a.alloc(f32, sh.rows);
        defer a.free(xs);
        const wsv = try a.alloc(f32, sh.n);
        defer a.free(wsv);
        for (xq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
        for (wq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
        for (xs) |*v| v.* = 0.001 + 0.004 * rng.float(f32);
        for (wsv) |*v| v.* = 0.002 + 0.006 * rng.float(f32);

        var d_xq = try cuda.DeviceBuffer.alloc(gpu.d, xq.len);
        defer d_xq.free();
        var d_xs = try cuda.DeviceBuffer.alloc(gpu.d, xs.len * 4);
        defer d_xs.free();
        var d_wq = try cuda.DeviceBuffer.alloc(gpu.d, wq.len);
        defer d_wq.free();
        var d_ws = try cuda.DeviceBuffer.alloc(gpu.d, wsv.len * 4);
        defer d_ws.free();
        var d_out = try cuda.DeviceBuffer.alloc(gpu.d, sh.rows * sh.n * 2);
        defer d_out.free();
        var d_acc = try cuda.DeviceBuffer.alloc(gpu.d, sh.rows * sh.n * 4);
        defer d_acc.free();
        try d_xq.upload(0, std.mem.sliceAsBytes(xq));
        try d_xs.upload(0, std.mem.sliceAsBytes(xs));
        try d_wq.upload(0, std.mem.sliceAsBytes(wq));
        try d_ws.upload(0, std.mem.sliceAsBytes(wsv));

        // TallTile, persistent grid of min(SMs, tiles) CTAs
        const sm: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        var maps: [2]TensorMap = undefined;
        try encodeTiled(enc, &maps[0], d_xq.ptr, sh.rows, sh.k, 256);
        try encodeTiled(enc, &maps[1], d_wq.ptr, sh.n, sh.k, 128);
        var args: cuda.Args = .{};
        args.add(maps[0]);
        args.add(maps[1]);
        for ([_]u64{ d_xs.ptr, d_ws.ptr, d_out.ptr }) |p| args.add(p);
        for ([_]usize{ sh.rows, sh.n, sh.k }) |v| args.add(@as(c_int, @intCast(v)));
        const grid_x: u32 = @intCast(@min(sm, (sh.rows + 255) / 256 * (sh.n / 128)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid_x }, .block = .{ .x = 256 }, .shared = tma_shared_bytes }, s, &args);
        try s.synchronize();
        const ours = try timeKernel(f, s, tma_shared_bytes, &args, grid_x, reps, io);

        // cuBLASLt int8 on the same inputs, C int32
        var ltg = try lt_.Gemm.init(&lt, handle, .{ .ta = .t, .m = sh.n, .n = sh.rows, .k = sh.k, .a = .i8, .b = .i8, .c = .i32, .lda = sh.k, .ldb = sh.k, .ldc = sh.n }, workspace_bytes);
        defer ltg.deinit();
        try ltg.runInt(d_wq.ptr, d_xq.ptr, d_acc.ptr, ws.ptr, workspace_bytes, s.handle);
        try s.synchronize();
        const ltns = try timeLt(&ltg, d_wq.ptr, d_xq.ptr, d_acc.ptr, ws.ptr, s, reps, io);

        const flops: f64 = 2.0 * @as(f64, @floatFromInt(sh.rows)) * @as(f64, @floatFromInt(sh.n)) * @as(f64, @floatFromInt(sh.k));
        const lt_ms = @as(f64, @floatFromInt(ltns)) / std.time.ns_per_ms;
        const ours_ms = @as(f64, @floatFromInt(ours)) / std.time.ns_per_ms;
        std.debug.print("BENCH h3_int8_tma {s} M={d} N={d} K={d}: tile={d:.2} ms ({d:.1} TOPS) cublaslt={d:.2} ms ({d:.1} TOPS) ratio={d:.3}x\n", .{
            sh.name, sh.rows, sh.n, sh.k, ours_ms, flops / (ours_ms * 1e9), lt_ms, flops / (lt_ms * 1e9), lt_ms / ours_ms,
        });
    }
}
