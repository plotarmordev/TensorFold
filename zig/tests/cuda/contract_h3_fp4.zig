//! MiniMax H3's NVFP4 row quantizer (h3_quantize_fp4) under the contract: every e4m3 block scale and e2m1 code
//! against the float64 rotation, at the byte cuBLASLt's VEC16_UE4M3 128 x 4 tiled layout reads it from.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const int8 = @import("contract_h3_int8.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const fp8 = core.quant.fp8;
const nvfp4 = core.quant.nvfp4;

const rule = "a block whose rotation holds a NaN scales by e4m3 NaN (0x7f) with every code 7; one holding an infinity saturates its scale to 448 and its codes to 6 with the value's sign";

/// One 16-value block a scale; 130 rows cross the scales' 128-row tile into its padding.
pub fn spec() c.Spec {
    return .{
        .name = "cuda h3 fp4 rows h3_quantize_fp4",
        .accum = .none,
        .depth = int8.rotationChain,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{}, .{ .size = 1024, .step = 128, .min = 128 } },
        .served = &.{ .{ .rows = 130, .n = 5376, .k = 5376 }, .{ .rows = 16, .n = 7168, .k = 7168 }, .{ .rows = 16, .n = 14336, .k = 14336 } },
    };
}

/// The e2m1 values in order: the nearest code is monotone in its input.
const grid = [_]f64{ -6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// The e2m1 value nearest `t`, a tie going down or up.
fn nearest(t: f64, up: bool) f64 {
    var best = grid[0];
    for (grid[1..]) |v| {
        const d = @abs(t - v) - @abs(t - best);
        if (d < 0 or (d == 0 and up)) best = v;
    }
    return best;
}

/// Byte offset of row `row`'s block `b` scale in cuBLASLt's tiled layout: 128 rows by 4 blocks a 512-byte tile.
fn scaleAt(row: usize, b: usize, k: usize) usize {
    const tiles_k = (k / 16 + 3) / 4;
    return ((row / 128) * tiles_k + b / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + b % 4;
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
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
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
        // a zero row: every block's zero scale
        if (shape.rows > 3) @memset(t.x[3 * k ..][0..k], 0);
        if (fill == .nonfinite) for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, row| {
            t.x[row * k + @min(k - 1, 130)] = v;
        };
        for (h, t.x) |*o, *v| {
            o.* = c.toBf16(@floatCast(v.*));
            v.* = c.fromBf16(o.*);
        }
        t.xd = try t.rig.input(std.mem.sliceAsBytes(h));
        t.q = try t.rig.output(shape.rows * k / 2);
        t.s = try t.rig.output(cuda.cublaslt.scaleBytes(shape.rows, k));
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
        return t.shape.rows * t.shape.k / 2 + cuda.cublaslt.scaleBytes(t.shape.rows, t.shape.k);
    }

    /// Unused: `check` judges every byte.
    pub fn reference(_: *Quantize, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Every block's scale the e4m3 rounding of its rotated absmax / 6, within the rotation's fp32 error; every code
    /// the nearest e2m1 value to its rotated value over that scale; the stated rule at planted blocks; every
    /// padding byte of the scale tiles left unwritten.
    pub fn check(t: *Quantize, at: g.Gate) !?[]const u8 {
        const rows = t.shape.rows;
        const k = t.shape.k;
        const a = t.arena.allocator();
        const q = try a.dupe(u8, try t.rig.bytes(t.q, rows * k / 2));
        const s = try a.dupe(u8, try t.rig.bytes(t.s, cuda.cublaslt.scaleBytes(rows, k)));
        const written = try a.alloc(bool, s.len);
        @memset(written, false);
        const rot = c.gamma(@floatFromInt(int8.rotationChain(t.shape)), 0x1p-24);
        var planted: usize = 0;
        var v: [16]f64 = undefined;
        var m: [16]f64 = undefined;
        for (0..rows) |row| for (0..k / 16) |b| {
            const x = t.x[row * k + b / 8 * 128 ..][0..128];
            var nan = false;
            var inf = false;
            var most: f64 = 0;
            var err: f64 = 0;
            for (&v, &m, 0..) |*o, *mo, i| {
                var sum: f64 = 0;
                var mag: f64 = 0;
                for (x, 0..) |xv, l| {
                    sum += int8.sign(l, b % 8 * 16 + i) * xv;
                    mag += @abs(xv);
                }
                o.* = sum / @sqrt(128.0);
                mo.* = mag / @sqrt(128.0);
                nan = nan or std.math.isNan(o.*);
                inf = inf or std.math.isInf(o.*);
                most = @max(most, @abs(o.*));
                err = @max(err, rot * mo.*);
            }
            const at_s = scaleAt(row, b, k);
            written[at_s] = true;
            const sb = s[at_s];
            const codes = q[(row * k + b * 16) / 2 ..][0..8];
            if (nan or inf) {
                planted += 1;
                const want_s: u8 = if (nan) 0x7f else 0x7e;
                if (sb != want_s) return try std.fmt.bufPrint(&t.msg, "row {d} block {d} holds a {s} but scales by byte 0x{x:0>2}", .{ row, b, if (nan) "NaN" else "infinity", sb });
                for (0..16) |i| {
                    const code: u8 = (codes[i / 2] >> @intCast(4 * (i % 2))) & 15;
                    const want: u8 = if (nan) 7 else if (v[i] < 0) 15 else 7;
                    if (code != want) return try std.fmt.bufPrint(&t.msg, "row {d} block {d} holds a {s} but code {d} is {d}", .{ row, b, if (nan) "NaN" else "infinity", i, code });
                }
                continue;
            }
            const scale = fp8.e4m3(sb);
            const lo = fp8.toE4m3(@max(most - err, 0) / 6 * (1 - 0x1p-22));
            const hi = fp8.toE4m3((most + err) / 6 * (1 + 0x1p-22));
            if (!(scale >= lo and scale <= hi)) return try std.fmt.bufPrint(&t.msg, "row {d} block {d} scales by {e} where float64 gives {e} to {e}", .{ row, b, scale, lo, hi });
            for (0..16) |i| {
                const got = nvfp4.e2m1((codes[i / 2] >> @intCast(4 * (i % 2))) & 15);
                const e = rot * m[i];
                const low = if (scale == 0) 0 else nearest((v[i] - e) / scale * (1 - std.math.copysign(@as(f64, 0x1p-22), v[i] - e)), false);
                const high = if (scale == 0) 0 else nearest((v[i] + e) / scale * (1 + std.math.copysign(@as(f64, 0x1p-22), v[i] + e)), true);
                if (!(got >= low and got <= high)) return try std.fmt.bufPrint(&t.msg, "row {d} block {d} code {d} is {d} where float64 gives {d} to {d}", .{ row, b, i, got, low, high });
            }
        };
        for (s, written, 0..) |byte, w, i| if (!w and byte != 0xff) return try std.fmt.bufPrint(&t.msg, "padding scale byte {d} written as 0x{x:0>2}", .{ i, byte });
        if (at == .nonfinite and planted == 0) return "no planted value reached a launched row: no proof";
        return null;
    }
};
