//! MiniMax H3's int8 projection kernels (h3_quantize, h3_quantize_chunks, h3_gemm_int8) under the contract, against float64 formulas.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Order = g.Order;
const sumIn = core.contract.rows.sumIn;
const gpu_check = @import("check.zig");

/// The rows' widths the blocks quantize: hidden (q k v gate, mlp in), attention out (56 heads of 128), mlp out.
const hidden = 5376;
const inner = 7168;
const mlp = 14336;

/// A rotated value: the lane's 4-point transform, five shuffle butterflies, then the 1 / sqrt(128) product.
fn rotationChain(_: c.Shape) usize {
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
fn sign(i: usize, j: usize) f64 {
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
/// One output a product sum: rows by n, int32 sums times the row's and the column's fp32 scales, then bf16.
const quantize_chunks_rule = "a chunk holding an infinity scales by +inf and quantizes to zeros; a NaN's chunk quantizes to zeros and leaves its scale";

/// One scale a 128-wide chunk; rows and k (the chunks) are its dimensions, n repeats k.
pub fn quantizeChunksSpec() c.Spec {
    return .{
        .name = "cuda h3 int8 rows h3_quantize_chunks",
        .accum = .f32,
        .depth = rotationChain,
        .ops = 2,
        .out = .f32,
        .nan = .{ .stated = quantize_chunks_rule },
        .tiles = .{ .{}, .{}, .{ .size = 1024, .step = 128, .min = 128 } },
        .served = &.{ .{ .rows = 16, .n = hidden, .k = hidden }, .{ .rows = 16, .n = inner, .k = inner }, .{ .rows = 16, .n = mlp, .k = mlp } },
        .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "an absmax over a 128-term rotation: one fp32 scale a chunk, its codes checked against it" },
    };
}

pub const QuantizeChunks = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    xd: u64 = 0,
    q: u64 = 0,
    s: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(t: *QuantizeChunks) void {
        t.rig.deinit();
        t.arena.deinit();
    }

    pub fn kernel(t: *QuantizeChunks) g.Kernel {
        return .{ .spec = quantizeChunksSpec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(QuantizeChunks);

    pub fn prepare(t: *QuantizeChunks, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
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
        if (shape.rows > 3) @memset(t.x[3 * k ..][0..k], 0);
        if (fill == .nonfinite) for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, row| {
            t.x[row * k + @min(k - 1, 130)] = v;
        };
        for (h, t.x) |*o, *v| {
            o.* = c.toBf16(@floatCast(v.*));
            v.* = c.fromBf16(o.*);
        }
        t.xd = try t.rig.input(std.mem.sliceAsBytes(h));
        t.s = try t.rig.output(shape.rows * (k / 128) * 4);
        t.q = try t.rig.output(shape.rows * k);
        return true;
    }

    pub fn run(t: *QuantizeChunks) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ t.xd, t.q, t.s }) |p| args.add(p);
        args.add(@as(c_int, @intCast(t.shape.k)));
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(t.shape.rows) }, .block = .{ .x = 256 } }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *QuantizeChunks) usize {
        return t.shape.rows * (t.shape.k / 128);
    }

    /// Rotated value `j` of chunk `ch` in row `row`, summed as `acc` in `order` (float64 when .none), and its magnitude.
    fn rotated(t: *QuantizeChunks, terms: []f64, row: usize, ch: usize, j: usize, acc: c.Accum, order: Order) [2]f64 {
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

    /// Each chunk's absmax / 127 over its finite values, and that chunk's magnitude / 127 as the bound's mag.
    fn scales(t: *QuantizeChunks, acc: c.Accum, order: Order, want: []f64, mag: []f64) void {
        var terms: [128]f64 = undefined;
        const chunks = t.shape.k / 128;
        for (0..t.shape.rows) |row| for (0..chunks) |ch| {
            var most: f64 = 0;
            var big: f64 = 0;
            for (0..128) |j| {
                const v = t.rotated(&terms, row, ch, j, acc, order);
                if (!std.math.isNan(v[0])) most = @max(most, @abs(v[0]));
                if (std.math.isFinite(v[1])) big = @max(big, v[1]);
            }
            want[row * chunks + ch] = if (most > 0) most / 127 else 1;
            mag[row * chunks + ch] = big / 127;
        };
    }

    pub fn reference(t: *QuantizeChunks, want: []f64, mag: []f64) !void {
        t.scales(.none, .sequential, want, mag);
    }

    pub fn simulate(t: *QuantizeChunks, acc: c.Accum, order: Order, out: []f64) !void {
        const mag = try t.arena.allocator().alloc(f64, out.len);
        t.scales(acc, order, out, mag);
    }
};

const gemm_rule = "a scale's NaN or infinity reaches its whole row or column of the product; the int8 codes are always finite";

/// h3_gemm_int8's dynamic shared bytes: its 3 stages fill sm_121's per-block cap (cuda_int8's own
/// `gemm_shared_bytes`; duplicated so the contract needs no family import).
pub const gemm_shared_bytes = 3 * (128 * 128 + 128 * 128 + 128 * 4);

/// The product's chains: two fp32 roundings a chunk (the activation-scale fold's fma and the accumulator's
/// add), one more for the row-scale epilogue, plus the bf16 store's own rounding which `out` carries.
fn gemmDepth(shape: c.Shape) usize {
    return 2 * (shape.k / 128) + 2;
}

pub fn gemmSpec() c.Spec {
    return .{
        .name = "cuda h3 int8 product h3_gemm_int8",
        .accum = .f32,
        .depth = gemmDepth,
        .ops = 0,
        .out = .bf16,
        .nan = .{ .stated = gemm_rule },
        .tiles = .{ .{}, .{ .size = 128, .step = 8, .min = 8 }, .{ .size = 128, .step = 128, .min = 128 } },
        .served = &.{
            .{ .rows = 17, .n = 136, .k = 256 }, // partial M and N tiles, two chunks
            .{ .rows = 129, .n = 264, .k = 384 }, // a second M tile, n past two n8 tiles
            .{ .rows = 16, .n = 56, .k = 14336 }, // the widest projection input, n under one tile
            .{ .rows = 130, .n = 128, .k = 128 }, // one chunk, M just over the tile
        },
        .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "a chunk's 128 products are an exact int32 tensor-core sum; fp32 carries only the per-chunk activation-scale fold and the row-scale epilogue" },
    };
}

/// C (rows, N) bf16 = X int8 (rows, K) . W int8 (N, K)^T at per-chunk activation scales XS (rows, K/128)
/// and per-row weight scales WS (N).
pub const Gemm = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    xq: []i8 = &.{},
    wq: []i8 = &.{},
    xs: []f64 = &.{},
    ws: []f64 = &.{},
    dev: [5]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(t: *Gemm) void {
        t.rig.deinit();
        t.arena.deinit();
    }

    pub fn kernel(t: *Gemm) g.Kernel {
        return .{ .spec = gemmSpec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Gemm);

    pub fn prepare(t: *Gemm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 128 != 0 or shape.k == 0 or shape.n % 8 != 0 or shape.n == 0 or shape.rows == 0) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        t.shape = shape;
        const a = t.arena.allocator();
        const chunks = shape.k / 128;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.xq = try a.alloc(i8, shape.rows * shape.k);
        t.wq = try a.alloc(i8, shape.n * shape.k);
        t.xs = try a.alloc(f64, shape.rows * chunks);
        t.ws = try a.alloc(f64, shape.n);
        for (t.xq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
        for (t.wq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
        // one loud chunk a row: its scale carries it, the rest stay small — the per-chunk claim's proof
        for (t.xs, 0..) |*v, i| v.* = 0.001 + 0.004 * rng.float(f64) * if (i % chunks == 1) 100 else 1;
        for (t.ws) |*v| v.* = 0.002 + 0.006 * rng.float(f64);
        if (fill == .stress) {
            for (t.xq) |*v| v.* = if (v.* >= 0) 127 else -127;
        }
        if (fill == .nonfinite) {
            if (shape.rows * chunks > 2) t.xs[2] = std.math.nan(f64);
            if (shape.n > 3) t.ws[3] = std.math.inf(f64);
        }
        const xs32 = try a.alloc(f32, t.xs.len);
        const ws32 = try a.alloc(f32, t.ws.len);
        for (xs32, t.xs) |*o, v| o.* = @floatCast(v);
        for (ws32, t.ws) |*o, v| o.* = @floatCast(v);
        t.dev[0] = try t.rig.input(std.mem.sliceAsBytes(t.xq));
        t.dev[1] = try t.rig.input(std.mem.sliceAsBytes(xs32));
        t.dev[2] = try t.rig.input(std.mem.sliceAsBytes(t.wq));
        t.dev[3] = try t.rig.input(std.mem.sliceAsBytes(ws32));
        t.dev[4] = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(t: *Gemm) !void {
        var args: cuda.Args = .{};
        for (t.dev) |p| args.add(p);
        for ([_]usize{ t.shape.rows, t.shape.n, t.shape.k }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(@as(c_int, 8)); // tile bands' L2 swizzle group, as cuda_int8's gemm_group
        const tiles_m = (t.shape.rows + 127) / 128;
        const tiles_n = (t.shape.n + 127) / 128;
        const shared: u32 = @intCast(gemm_shared_bytes);
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(tiles_m * tiles_n) }, .block = .{ .x = 256 }, .shared = shared }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Gemm) usize {
        return t.shape.rows * t.shape.n;
    }

    /// The kernel's formula in float64: per chunk an exact int32 dot of the codes times the chunk's xs,
    /// summed in fp32 order `acc`/`order` when set (the fold chain), times the row's ws, then rounded to
    /// bf16 as the store does. `mag` is each output's sum of |chunk dot * xs * ws| — the bound's term magnitude.
    fn product(t: *Gemm, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        const a = t.arena.allocator();
        const chunks = t.shape.k / 128;
        const terms = try a.alloc(f64, chunks);
        const mags = try a.alloc(f64, chunks);
        for (0..t.shape.rows) |row| for (0..t.shape.n) |col| {
            var big: f64 = 0;
            for (0..chunks) |ch| {
                var dot: i64 = 0;
                for (0..128) |j| {
                    dot += @as(i64, t.xq[row * t.shape.k + ch * 128 + j]) * t.wq[col * t.shape.k + ch * 128 + j];
                }
                const xs = t.xs[row * chunks + ch];
                const ws = t.ws[col];
                terms[ch] = @as(f64, @floatFromInt(dot)) * xs * ws;
                mags[ch] = @abs(@as(f64, @floatFromInt(dot)) * xs * ws);
            }
            const v = if (acc == .none) blk: {
                var s: f64 = 0;
                for (terms) |x| s += x;
                break :blk s;
            } else sumIn(acc, order, terms);
            for (mags) |m| big += m;
            const i = row * t.shape.n + col;
            out[i] = c.fromBf16(c.toBf16(@floatCast(v)));
            if (mag) |mm| mm[i] = big;
        };
    }

    pub fn reference(t: *Gemm, want: []f64, mag: []f64) !void {
        try t.product(.none, .sequential, want, mag);
    }

    pub fn simulate(t: *Gemm, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.product(acc, order, out, null);
    }

    /// Beyond the float64 bound: at a nonfinite fill the stated rule must reach the output — a NaN scale's row
    /// all NaN, an infinite scale's column all infinite or NaN where its codes cancel.
    pub fn check(t: *Gemm, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const s = t.shape;
        const chunks = s.k / 128;
        const got = try t.rig.bytes(t.dev[4], s.rows * s.n * 2);
        var planted: usize = 0;
        for (0..s.rows) |row| for (0..chunks) |ch| {
            if (!std.math.isNan(t.xs[row * chunks + ch]) and !std.math.isInf(t.xs[row * chunks + ch])) continue;
            planted += 1;
            for (0..s.n) |col| {
                const v = c.read(.bf16, got, row * s.n + col);
                if (std.math.isFinite(v)) return try std.fmt.bufPrint(&t.msg, "row {d} chunk {d}'s nonfinite scale left output {d} finite: {e}", .{ row, ch, col, v });
            }
        };
        for (0..s.n) |col| {
            if (!std.math.isNan(t.ws[col]) and !std.math.isInf(t.ws[col])) continue;
            planted += 1;
            for (0..s.rows) |row| {
                const v = c.read(.bf16, got, row * s.n + col);
                if (std.math.isFinite(v)) return try std.fmt.bufPrint(&t.msg, "column {d}'s nonfinite scale left output {d} finite: {e}", .{ col, row, v });
            }
        }
        if (planted == 0) return "no planted scale reached a launched row: no proof";
        return null;
    }
};

// ---------------------------------------------------------------------------
// Microbench: the own tile against cuBLASLt's int8 product on the real picture
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

/// A/B the picture projections on the same device buffers: `h3_gemm_int8` writes bf16 from its own tile;
/// cuBLASLt sums into int32 (`h3_dequantize`'s old input). Prints both ns/launch and the implied TOPS.
pub fn bench(gpu: gpu_check.Gpu) !void {
    const io = gpu.io;
    const cap = try gpu.ctx.capability();
    try gpu_check.expect(cap == 121, "h3 int8 benchmark needs sm_121, got sm_{d}", .{cap});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const a = gpu.gpa;

    var mod = try cuda.Module.load(gpu.d, &image_h3);
    defer mod.unload();
    const f = try mod.function("h3_gemm_int8");
    try f.allowDynamicShared(gemm_shared_bytes);

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
        const chunks = sh.k / 128;
        var prng = std.Random.DefaultPrng.init(0x1398 + i);
        const rng = prng.random();
        const xq = try a.alloc(i8, sh.rows * sh.k);
        defer a.free(xq);
        const wq = try a.alloc(i8, sh.n * sh.k);
        defer a.free(wq);
        const xs = try a.alloc(f32, sh.rows * chunks);
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

        // our tile
        var args: cuda.Args = .{};
        for ([_]u64{ d_xq.ptr, d_xs.ptr, d_wq.ptr, d_ws.ptr, d_out.ptr }) |p| args.add(p);
        for ([_]usize{ sh.rows, sh.n, sh.k, 8 }) |v| args.add(@as(c_int, @intCast(v)));
        const grid_x: u32 = @intCast(((sh.rows + 127) / 128) * ((sh.n + 127) / 128));
        // warm + throw the first timed sample's cold-start out via one untimed launch
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid_x }, .block = .{ .x = 256 }, .shared = @intCast(gemm_shared_bytes) }, s, &args);
        try s.synchronize();
        const ours = try timeKernel(f, s, @intCast(gemm_shared_bytes), &args, grid_x, reps, io);

        // cuBLASLt int8 on the same inputs, C int32 (what the product used to write)
        var ltg = try lt_.Gemm.init(&lt, handle, .{ .ta = .t, .m = sh.n, .n = sh.rows, .k = sh.k, .a = .i8, .b = .i8, .c = .i32, .lda = sh.k, .ldb = sh.k, .ldc = sh.n }, workspace_bytes);
        defer ltg.deinit();
        try ltg.runInt(d_wq.ptr, d_xq.ptr, d_acc.ptr, ws.ptr, workspace_bytes, s.handle);
        try s.synchronize();
        const ltns = try timeLt(&ltg, d_wq.ptr, d_xq.ptr, d_acc.ptr, ws.ptr, s, reps, io);

        const flops: f64 = 2.0 * @as(f64, @floatFromInt(sh.rows)) * @as(f64, @floatFromInt(sh.n)) * @as(f64, @floatFromInt(sh.k));
        const ours_ms = @as(f64, @floatFromInt(ours)) / std.time.ns_per_ms;
        const lt_ms = @as(f64, @floatFromInt(ltns)) / std.time.ns_per_ms;
        std.debug.print("BENCH h3_int8 {s} M={d} N={d} K={d}: tile={d:.2} ms ({d:.1} TOPS) cublaslt={d:.2} ms ({d:.1} TOPS) ratio={d:.3}x\n", .{
            sh.name, sh.rows, sh.n, sh.k, ours_ms, flops / (ours_ms * 1e9), lt_ms, flops / (lt_ms * 1e9), lt_ms / ours_ms,
        });
    }
}
