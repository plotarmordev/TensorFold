//! MiniMax H3's int8 projection kernels (h3_quantize, h3_dequantize) under the contract, against float64 formulas.
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
pub fn dequantizeSpec() c.Spec {
    return .{
        .name = "cuda h3 int8 sums h3_dequantize",
        .accum = .none,
        .depth = none,
        .ops = 3,
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = 1024, .step = 4, .min = 4 }, .{} },
        // 1024 blocks of 256 threads cover 1M quads: 64 rows of mlp out's 2 x 14336 run the grid-stride loop
        .served = &.{ .{ .rows = 16, .n = hidden, .k = 1 }, .{ .rows = 16, .n = 3 * inner, .k = 1 }, .{ .rows = 64, .n = 2 * mlp, .k = 1 } },
    };
}

pub const Dequantize = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    acc: []i32 = &.{},
    sx: []f32 = &.{},
    sw: []f32 = &.{},
    bufs: [4]u64 = undefined,

    pub fn deinit(t: *Dequantize) void {
        t.rig.deinit();
        t.arena.deinit();
    }

    pub fn kernel(t: *Dequantize) g.Kernel {
        return .{ .spec = dequantizeSpec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Dequantize);

    pub fn prepare(t: *Dequantize, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n % 4 != 0 or shape.n == 0) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        t.shape = shape;
        const al = t.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.acc = try al.alloc(i32, shape.rows * shape.n);
        t.sx = try al.alloc(f32, shape.rows);
        t.sw = try al.alloc(f32, shape.n);
        // |sums| up to 127 * 127 * 14336, as a full-width product's can reach
        for (t.acc) |*v| v.* = rng.intRangeAtMost(i32, -231_000_000, 231_000_000);
        for (t.sx) |*v| v.* = @floatCast(0x1p-10 * (0.5 + rng.float(f64)));
        for (t.sw) |*v| v.* = @floatCast(0x1p-12 * (0.5 + rng.float(f64)));
        if (fill == .nonfinite) {
            // a NaN row scale, both infinities on columns, and 0 * inf where a sum is zero
            t.sx[shape.rows - 1] = std.math.nan(f32);
            t.sw[1] = std.math.inf(f32);
            t.sw[shape.n - 2] = -std.math.inf(f32);
            t.acc[1] = 0;
        }
        t.bufs[0] = try t.rig.input(std.mem.sliceAsBytes(t.acc));
        t.bufs[1] = try t.rig.input(std.mem.sliceAsBytes(t.sx));
        t.bufs[2] = try t.rig.input(std.mem.sliceAsBytes(t.sw));
        t.bufs[3] = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(t: *Dequantize) !void {
        var args: cuda.Args = .{};
        for (t.bufs) |p| args.add(p);
        args.add(@as(c_int, @intCast(t.shape.rows)));
        args.add(@as(c_int, @intCast(t.shape.n)));
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = 1024 }, .block = .{ .x = 256 } }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Dequantize) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Dequantize, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, t.acc, 0..) |*w, *m, a, i| {
            w.* = @as(f64, @floatFromInt(a)) * t.sx[i / n] * t.sw[i % n];
            m.* = 0;
        }
    }
};
