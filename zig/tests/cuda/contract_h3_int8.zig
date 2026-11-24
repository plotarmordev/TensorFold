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

const HeadEpilogue = extern struct {
    nq: u64,
    nk: u64,
    cos: u64,
    sin: u64,
    slot: u64,
    tq: u64,
    tk: u64,
    tv: u64,
    heads: c_int,
    slots: c_int,
    eps: f32,
};

/// A TMA product kernel with its ring's dynamic shared memory allowed (past the 48 KiB default).
pub fn tmaFunction(mod: cuda.Module, name: [:0]const u8) !cuda.Function {
    const f = try mod.function(name);
    try f.allowDynamicShared(tma_shared_bytes);
    return f;
}

/// What a GemmTma registration launches: the plain product, fc1's SwiGLU epilogue on interleaved
/// rows, or the control - the plain product at a width cuBLASLt took (n % 4) but the TMA grid,
/// whole 128-column tiles only, leaves partly unwritten.
pub const Mode = enum { plain, swiglu, heads, control };

/// One output a product element: bf16(acc * xs[row] * ws[col]), the int32 sums exact; SwiGLU's
/// is h3_swiglu's on the plain product's [gate | up] halves, bit for bit.
pub fn gemmTmaSpec(mode: Mode) c.Spec {
    const served: []const c.Shape = switch (mode) {
        .plain => &.{
            .{ .rows = 16, .n = 128, .k = 128 }, // one tile, one stage
            .{ .rows = 300, .n = 128, .k = 256 }, // a partial M tile and two K blocks
            .{ .rows = 64, .n = 384, .k = 384 }, // three N tiles past a partial M tile
            .{ .rows = 16, .n = 21504, .k = 5376 }, // qkvc's real fan-out and depth
        },
        .swiglu => &.{
            .{ .rows = 16, .n = 256, .k = 128 }, // one feature tile
            .{ .rows = 300, .n = 512, .k = 256 }, // a partial M tile, two tiles and K blocks
            .{ .rows = 256, .n = 2 * mlp, .k = hidden }, // fc1's real fan-out
        },
        .heads => &.{
            .{ .rows = 16, .n = 384, .k = 128 },
            .{ .rows = 300, .n = 384, .k = 256 },
            .{ .rows = 64, .n = 768, .k = 384 },
            .{ .rows = 16, .n = 21504, .k = 5376 },
        },
        .control => &.{.{ .rows = 16, .n = 392, .k = 128 }}, // n % 8 keeps the epilogue's 16-byte stores aligned
    };
    return .{
        .name = switch (mode) {
            .plain => "cuda h3 int8 tma product h3_gemm_int8_tma",
            .swiglu => "cuda h3 int8 tma swiglu h3_swiglu_rows + h3_swiglu_int8_tma",
            .heads => "cuda h3 int8 tma heads h3_heads_int8_tma",
            .control => "control: h3_gemm_int8_tma at a width off whole 128-column tiles",
        },
        .accum = .none, // int32 sums, exact: no fp accumulator for gate 4 to stress
        .depth = none,
        .ops = if (mode == .heads) 8 else 3,
        .mids = if (mode == .heads) 5 else 0, // dequantization, norm, cosine, sine and their two products
        .out = .bf16,
        .nan = .{ .stated = "a nonfinite row or column scale reaches every output it multiplies, as h3_dequantize's did" },
        .tiles = switch (mode) {
            .plain => .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 128, .step = 128, .min = 128 } },
            .swiglu => .{ .{}, .{ .size = 256, .step = 256, .min = 256 }, .{ .size = 128, .step = 128, .min = 128 } },
            .heads => .{ .{}, .{}, .{} },
            .control => .{ .{}, .{}, .{} },
        },
        .served = served,
        .replaces = switch (mode) {
            .swiglu => "h3_gemm_int8_tma then h3_swiglu",
            .heads => "h3_gemm_int8_tma then h3_heads",
            else => null,
        },
        .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = if (mode == .heads)
            "int32 dots are exact; dequantization, normalized values, bf16 cosine/sine and each rotary product round to bf16 before the final bf16 pack; norm sums are fp32"
        else
            "mma.sync sums the codes' products in int32, exact to a bit; fp32 carries only the acc * xs * ws fold and the bf16 store" },
    };
}

/// C (rows, n) bf16 = (X int8 rows,k . W int8 n,k)^T * xs[rows] * ws[n], as the TMA kernel writes
/// it; in swiglu mode W's rows are [gate | up], interleaved on the device as fc1's are, and C is
/// silu(gate) * up, (rows, n / 2).
pub const GemmTma = struct {
    rig: rig_.Rig,
    mode: Mode,
    mod: cuda.Module,
    /// The heads control: one fused output word flipped after the launch, which gate 1 must catch.
    planted: bool = false,
    enc: Encode,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    xq: []i8 = &.{},
    wq: []i8 = &.{},
    xs: []f32 = &.{},
    ws: []f32 = &.{},
    /// xq, xs, wq, ws, out; epilogue baselines and head inputs follow.
    dev: [13]u64 = @splat(0),

    pub fn deinit(t: *GemmTma) void {
        t.rig.deinit();
        t.arena.deinit();
    }

    pub fn kernel(t: *GemmTma) g.Kernel {
        var spec = gemmTmaSpec(t.mode);
        if (t.planted) spec.name = "control: h3_heads_int8_tma with one output word flipped";
        return .{ .spec = spec, .ctx = t, .vt = if (t.mode == .heads) &vt_heads else &vt, .control = if (t.mode == .control) .shapes else if (t.planted) .nonfinite else null };
    }

    const vt = g.hooks(GemmTma);
    const vt_heads = blk: {
        var v = vt;
        v.check = checkHeads;
        break :blk v;
    };

    fn headCount(t: *const GemmTma) usize {
        return t.shape.n / 384;
    }

    fn headSlots(t: *const GemmTma) usize {
        return std.mem.alignForward(usize, t.shape.rows, 64);
    }

    fn headBytes(t: *const GemmTma) usize {
        return 3 * t.headCount() * t.headSlots() * 128 * 2;
    }

    fn checkHeads(ctx: *anyopaque, gate: g.Gate) !?[]const u8 {
        const t: *GemmTma = @ptrCast(@alignCast(ctx));
        // the rig downloads into one host buffer: keep the baseline's bytes before the fused output overwrites them
        const want = try t.arena.allocator().dupe(u8, try t.replaced());
        const got = try t.rig.bytes(t.dev[4], t.headBytes());
        if (!std.mem.eql(u8, got, want)) return "fused norm or rotary differs from h3_gemm_int8_tma + h3_heads";
        if (gate != .nonfinite) return null;
        for (0..got.len / 2) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return "no output is non-finite: the planted values prove nothing";
    }

    fn outColumns(t: *const GemmTma) usize {
        return if (t.mode == .swiglu) t.shape.n / 2 else t.shape.n;
    }

    pub fn prepare(t: *GemmTma, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const width: usize = switch (t.mode) {
            .plain => 128,
            .swiglu => 256,
            .heads => 384,
            .control => 8,
        };
        if (shape.k % 128 != 0 or shape.n % width != 0 or shape.k == 0 or shape.n == 0 or shape.rows == 0) return false;
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
        t.dev[4] = try t.rig.output(if (t.mode == .heads) t.headBytes() else shape.rows * t.outColumns() * 2);
        if (t.mode == .swiglu) {
            t.dev[5] = try t.rig.scratch(shape.n * shape.k);
            t.dev[6] = try t.rig.scratch(shape.n * 4);
            t.dev[7] = try t.rig.scratch(shape.rows * shape.n * 2);
            t.dev[8] = try t.rig.scratch(shape.rows * shape.n);
        } else if (t.mode == .heads) {
            t.dev[5] = try t.rig.scratch(shape.rows * shape.n * 2); // plain product's qkvc rows
            t.dev[12] = try t.rig.scratch(t.headBytes()); // baseline's tile-order q, k and v
            for ([_]usize{ 6, 7 }, 0..) |which, which_qk| {
                // Distinct q/k weights: an nq/nk swap in the kernel must change the outputs.
                const nw = try a.alloc(u16, 128);
                for (nw, 0..) |*v, i| v.* = c.toBf16(0.5 + @as(f32, @floatFromInt((i + which_qk * 5) % 17)) / 16);
                t.dev[which] = try t.rig.input(std.mem.sliceAsBytes(nw));
            }
            for ([_]usize{ 8, 9 }, 0..) |which, table| {
                const v = try a.alloc(f32, shape.rows * 48);
                for (v) |*x| x.* = if (table == 0) 0.9 + 0.1 * rng.float(f32) else 0.002 * rng.float(f32);
                t.dev[which] = try t.rig.input(std.mem.sliceAsBytes(v));
            }
            const slot = try a.alloc(i32, shape.rows);
            for (slot, 0..) |*v, i| v.* = @intCast(i);
            rng.shuffle(i32, slot);
            t.dev[10] = try t.rig.input(std.mem.sliceAsBytes(slot));
            const row_of = try a.alloc(i32, t.headSlots());
            @memset(row_of, -1);
            for (slot, 0..) |s, row| row_of[@intCast(s)] = @intCast(row);
            t.dev[11] = try t.rig.input(std.mem.sliceAsBytes(row_of));
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
        const tiles = (t.shape.rows + 255) / 256 * (t.shape.n / 128);
        try cuda.launch.launch(try tmaFunction(t.mod, name), .{ .grid = .{ .x = @intCast(@min(sm, tiles)) }, .block = .{ .x = 256 }, .shared = tma_shared_bytes }, t.rig.s, &args);
    }

    fn fillOutput(t: *GemmTma, ptr: u64, len: usize) !void {
        const b: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = ptr, .len = len };
        try b.fill8(0xAB, t.rig.s.handle);
    }

    fn productHeads(t: *GemmTma, out: u64) !void {
        var maps: [2]TensorMap = undefined;
        try encodeTiled(t.enc, &maps[0], t.dev[0], t.shape.rows, t.shape.k, 256);
        try encodeTiled(t.enc, &maps[1], t.dev[2], t.shape.n, t.shape.k, 128);
        const block = t.headBytes() / 3;
        const heads: HeadEpilogue = .{
            .nq = t.dev[6],
            .nk = t.dev[7],
            .cos = t.dev[8],
            .sin = t.dev[9],
            .slot = t.dev[10],
            .tq = out,
            .tk = out + block,
            .tv = out + 2 * block,
            .heads = @intCast(t.headCount()),
            .slots = @intCast(t.headSlots()),
            .eps = 1e-5,
        };
        var args: cuda.Args = .{};
        args.add(maps[0]);
        args.add(maps[1]);
        for ([_]u64{ t.dev[1], t.dev[3] }) |p| args.add(p);
        for ([_]usize{ t.shape.rows, t.shape.n, t.shape.k }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(heads);
        const sm: usize = @intCast(try t.rig.gpu.ctx.attribute(.multiprocessor_count));
        const tiles = (t.shape.rows + 255) / 256 * (t.shape.n / 128);
        try cuda.launch.launch(try tmaFunction(t.mod, "h3_heads_int8_tma"), .{ .grid = .{ .x = @intCast(@min(sm, tiles)) }, .block = .{ .x = 256 }, .shared = tma_shared_bytes }, t.rig.s, &args);
    }

    pub fn run(t: *GemmTma) !void {
        if (t.mode == .swiglu) {
            var args: cuda.Args = .{};
            for ([_]u64{ t.dev[2], t.dev[3], t.dev[5], t.dev[6] }) |p| args.add(p);
            args.add(@as(c_int, @intCast(t.shape.k)));
            try cuda.launch.launch(try t.mod.function("h3_swiglu_rows"), .{ .grid = .{ .x = @intCast(t.shape.n / 2) }, .block = .{ .x = 256 } }, t.rig.s, &args);
            try t.product("h3_swiglu_int8_tma", t.dev[5], t.dev[6], t.dev[4]);
        } else if (t.mode == .heads) {
            try t.fillOutput(t.dev[4], t.headBytes());
            try t.productHeads(t.dev[4]);
            if (t.planted) {
                const b: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = t.dev[4], .len = 2 };
                try b.fill8(0x3f, t.rig.s.handle);
            }
        } else try t.product("h3_gemm_int8_tma", t.dev[2], t.dev[3], t.dev[4]);
        try t.rig.s.synchronize();
    }

    /// SwiGLU's baseline: the plain product on the original [gate | up] rows, then h3_swiglu.
    pub fn replaced(t: *GemmTma) ![]const u8 {
        if (t.mode == .heads) {
            try t.fillOutput(t.dev[12], t.headBytes());
            try t.product("h3_gemm_int8_tma", t.dev[2], t.dev[3], t.dev[5]);
            const block = t.headBytes() / 3;
            var args: cuda.Args = .{};
            for ([_]u64{ t.dev[5], t.dev[6], t.dev[7], t.dev[8], t.dev[9], t.dev[11], t.dev[12], t.dev[12] + block, t.dev[12] + 2 * block }) |p| args.add(p);
            for ([_]usize{ t.headCount(), t.shape.n, 48, t.headSlots() }) |v| args.add(@as(c_int, @intCast(v)));
            args.add(@as(f32, 1e-5));
            try cuda.launch.launch(try t.mod.function("h3_heads"), .{ .grid = .{ .x = @intCast(t.headSlots() / 64) }, .block = .{ .x = 256 } }, t.rig.s, &args);
            return t.rig.bytes(t.dev[12], t.headBytes());
        }
        try t.product("h3_gemm_int8_tma", t.dev[2], t.dev[3], t.dev[7]);
        var args: cuda.Args = .{};
        for ([_]u64{ t.dev[7], t.dev[8] }) |p| args.add(p);
        for ([_]usize{ t.shape.rows, t.shape.n / 2 }) |v| args.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(try t.mod.function("h3_swiglu"), .{ .grid = .{ .x = 256 }, .block = .{ .x = 256 } }, t.rig.s, &args);
        return t.rig.bytes(t.dev[8], t.shape.rows * t.shape.n);
    }

    pub fn count(t: *GemmTma) usize {
        return if (t.mode == .heads) t.headBytes() / 2 else t.shape.rows * t.outColumns();
    }

    /// The float64 product: exact int32 dots over the whole row (the mma order is irrelevant to
    /// them), then acc * xs * ws in fp32 rounded to bf16 as the store does; SwiGLU then rounds
    /// silu(gate) and silu * up to bf16 as h3_swiglu does.
    pub fn reference(t: *GemmTma, want: []f64, mag: []f64) !void {
        if (t.mode == .heads) {
            const bytes = try t.replaced();
            for (want, mag, 0..) |*v, *m, i| {
                v.* = c.fromBf16(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
                m.* = @abs(v.*);
            }
            return;
        }
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
