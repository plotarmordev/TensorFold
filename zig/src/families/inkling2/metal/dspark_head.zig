//! DSpark's private full-vocabulary products: the BF16 originals, or FP8/FP4 copies packed once at load.
const std = @import("std");
const mtl = @import("metal");
const bf = @import("bf16_round");
const k = @import("dspark_kernels.zig");

/// The draft head's precision, from `INKLING2_DRAFT_HEAD` (bf16, fp8 or fp4; fp4 when unset).
pub const Precision = enum(u5) {
    bf16 = 16,
    fp8 = 8,
    fp4 = 4,

    pub fn fromEnv() !Precision {
        const text = @import("sys").getenv("INKLING2_DRAFT_HEAD") orelse return .fp4;
        return std.meta.stringToEnum(Precision, text) orelse error.DSparkHeadPrecision;
    }
    pub fn bits(p: Precision) usize {
        return @intFromEnum(p);
    }
};

/// Columns sharing one BF16 scale.
pub const block = 32;

/// A [rows, cols] matrix in one shared buffer: BITS-wide codes a row, then (packed only) a BF16 scale a block.
pub const Packed = struct {
    buffer: mtl.Buffer,
    precision: Precision,
    rows: usize,
    cols: usize,

    pub fn codeBytes(precision: Precision, rows: usize, cols: usize) usize {
        return rows * cols * precision.bits() / 8;
    }
    pub fn bytes(precision: Precision, rows: usize, cols: usize) usize {
        const scale_bytes: usize = if (precision == .bf16) 0 else rows * cols / block * 2;
        return codeBytes(precision, rows, cols) + scale_bytes;
    }
    pub fn init(device: mtl.Device, precision: Precision, rows: usize, cols: usize) !Packed {
        std.debug.assert(cols % block == 0);
        return .{ .buffer = try device.buffer(bytes(precision, rows, cols), mtl.ResourceOptions.shared), .precision = precision, .rows = rows, .cols = cols };
    }
    pub fn deinit(p: *Packed) void {
        p.buffer.deinit();
        p.* = undefined;
    }
    pub fn codes(p: Packed) k.Ref {
        return .{ .buf = p.buffer, .off = 0 };
    }
    pub fn scales(p: Packed) k.Ref {
        return .{ .buf = p.buffer, .off = codeBytes(p.precision, p.rows, p.cols) };
    }
};

const e2m1 = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// A code's value before its block scale.
pub fn decode(precision: Precision, code: u8) f32 {
    return switch (precision) {
        .fp4 => if (code & 8 != 0) -e2m1[code & 7] else e2m1[code & 7],
        .fp8 => {
            const e: u5 = @intCast((code >> 3) & 15);
            const m: f32 = @floatFromInt(code & 7);
            const v = if (e == 0) m * 0x1p-9 else std.math.ldexp(1 + m / 8, @as(i32, e) - 7);
            return if (code & 0x80 != 0) -v else v;
        },
        .bf16 => unreachable,
    };
}

/// The nearest code to `x` (already divided by its scale), ties to the even code, saturating at the largest.
pub fn encode(precision: Precision, x: f32) u8 {
    const a = @min(@abs(x), top(precision));
    if (precision == .fp4) {
        var best: u8 = 0;
        for (1..8) |c| {
            const code: u8 = @intCast(c);
            const d = @abs(e2m1[code] - a);
            const best_d = @abs(e2m1[best] - a);
            if (d < best_d or (d == best_d and code & 1 == 0)) best = code;
        }
        return best | @as(u8, if (x < 0) 8 else 0);
    }
    const sign: u8 = if (x < 0) 0x80 else 0;
    // Below 2^-6 the codes are subnormal steps of 2^-9; rounding up to 8 lands on the first normal, code 8.
    if (a < 0x1p-6) return sign | @as(u8, @intFromFloat(evenRound(a * 512)));
    const f = std.math.frexp(a); // a = f * 2^e, f in [0.5, 1)
    var m = evenRound(f.significand * 16) - 8; // three mantissa bits
    var e = f.exponent + 6;
    if (m == 8) {
        m = 0;
        e += 1;
    }
    return sign | @as(u8, @intCast(e)) << 3 | @as(u8, @intFromFloat(m));
}

fn evenRound(v: f32) f32 {
    const r = @round(v);
    return if (@abs(v - @trunc(v)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(v) else r;
}

/// The largest code magnitude: 6 for E2M1, 448 for E4M3.
pub fn top(precision: Precision) f32 {
    return if (precision == .fp4) 6 else 448;
}

/// The BF16 scale a block's largest magnitude maps to the top code with, never below it.
fn scaleOf(precision: Precision, amax: f32) u16 {
    const want = amax / top(precision);
    const s = bf.bits(want);
    return if (bf.value(s) < want) s + 1 else s;
}

fn packRows(precision: Precision, source: []const u16, cols: usize, codes: []u8, scales: []u16, begin: usize, end: usize) void {
    var x: [block]f32 = undefined;
    for (begin..end) |row| {
        for (0..cols / block) |b| {
            const at = row * cols + b * block;
            var amax: f32 = 0;
            for (&x, source[at..][0..block]) |*v, h| {
                v.* = bf.value(h);
                amax = @max(amax, @abs(v.*));
            }
            const s = scaleOf(precision, amax);
            scales[at / block] = s;
            const inv: f32 = if (amax == 0) 0 else 1 / bf.value(s);
            for (x, 0..) |v, j| {
                const code = encode(precision, v * inv);
                if (precision == .fp8) {
                    codes[at + j] = code;
                } else if (j & 1 == 0) {
                    codes[(at + j) / 2] = code;
                } else {
                    codes[(at + j) / 2] |= code << 4;
                }
            }
        }
    }
}

/// Fills packed `p` from the BF16 rows in `source`, on up to eight threads.
pub fn pack(p: Packed, source: []const u16) !void {
    std.debug.assert(p.precision != .bf16);
    if (source.len != p.rows * p.cols) return error.DSparkHeadShape;
    const all = p.buffer.slice(u8, Packed.bytes(p.precision, p.rows, p.cols));
    const code_bytes = Packed.codeBytes(p.precision, p.rows, p.cols);
    const codes = all[0..code_bytes];
    const scales: []u16 = @alignCast(std.mem.bytesAsSlice(u16, all[code_bytes..]));
    const count = @min(@as(usize, 8), p.rows);
    const chunk = (p.rows + count - 1) / count;
    var threads: [8]?std.Thread = @splat(null);
    for (0..count) |i| {
        const begin = @min(i * chunk, p.rows);
        const end = @min(begin + chunk, p.rows);
        const args = .{ p.precision, source, p.cols, codes, scales, begin, end };
        threads[i] = std.Thread.spawn(.{}, packRows, args) catch null;
        if (threads[i] == null) @call(.auto, packRows, args);
    }
    for (threads) |thread| if (thread) |t| t.join();
}

/// The packed head kernels of one precision.
pub const Pipelines = struct {
    base: mtl.Pipeline,
    logits: mtl.Pipeline,

    pub fn load(device: mtl.Device, precision: Precision) !Pipelines {
        std.debug.assert(precision != .bf16);
        const lib = try mtl.Library.fromSource(device, @import("inkling2_metal").inkling2_dspark, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const base_name = if (precision == .fp4) "tf_ink2_dspark_head_base_fp4" else "tf_ink2_dspark_head_base_fp8";
        const logits_name = if (precision == .fp4) "tf_ink2_dspark_head_logits_fp4" else "tf_ink2_dspark_head_logits_fp8";
        const base_pipe = try mtl.Pipeline.init(device, lib, base_name, false);
        errdefer base_pipe.deinit();
        return .{ .base = base_pipe, .logits = try mtl.Pipeline.init(device, lib, logits_name, false) };
    }
    pub fn deinit(p: *Pipelines) void {
        p.logits.deinit();
        p.base.deinit();
        p.* = undefined;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const k.Ref) void {
    for (refs, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
}

/// `k.base` on a packed unembed copy.
pub fn base(p: *const Pipelines, e: mtl.ComputeEncoder, unembed: Packed, y: k.Ref, out: k.Ref, rows: u32, vocabulary_count: u32) void {
    std.debug.assert(rows > 0 and rows <= 7 and vocabulary_count <= unembed.rows and unembed.cols == k.hidden);
    e.setPipeline(p.base);
    bind(e, &.{ unembed.codes(), unembed.scales(), y, out });
    e.setValue(k.BaseArgs{ .rows = rows, .vocab = vocabulary_count }, 4);
    e.dispatchGroups(mtl.Size.of((@as(usize, vocabulary_count) + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
}

/// `k.finalLogits` on a packed Phi copy, for its first `count` ids.
pub fn logits(p: *const Pipelines, e: mtl.ComputeEncoder, base_values: k.Ref, phi: Packed, beta: k.Ref, lambda: k.Ref, out: k.Ref, count: u32) void {
    std.debug.assert(count <= phi.rows and phi.cols == k.rank);
    e.setPipeline(p.logits);
    bind(e, &.{ base_values, phi.codes(), phi.scales(), beta, lambda, out });
    e.setValue(k.CountArgs{ .count = count }, 6);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}

test "every code decodes to the value it encodes from, and the top code saturates" {
    for ([_]Precision{ .fp4, .fp8 }) |p| {
        const codes: u8 = if (p == .fp4) 16 else 255;
        for (0..codes) |c| {
            const code: u8 = @intCast(c);
            if (p == .fp8 and code & 0x7f == 0x7f) continue; // E4M3's NaN
            const v = decode(p, code);
            const back = encode(p, v);
            try std.testing.expectEqual(v, decode(p, back));
        }
        try std.testing.expectEqual(top(p), decode(p, encode(p, 1e9)));
        try std.testing.expectEqual(-top(p), decode(p, encode(p, -1e9)));
    }
    try std.testing.expectEqual(@as(f32, 448), decode(.fp8, 0x7e));
    try std.testing.expectEqual(@as(f32, 0x1p-9), decode(.fp8, 1));
    // 2.5 lies between 2 (code 4) and 3 (code 5): the tie goes to the even code.
    try std.testing.expectEqual(@as(u8, 4), encode(.fp4, 2.5));
}

test "packing keeps every weight within half a step of its block's grid and the largest exact" {
    const cols = 64;
    var source: [2 * cols]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&source) |*h| h.* = bf.bits((prng.random().float(f32) - 0.5) * 0.08);
    source[cols + 5] = bf.bits(-0.75);
    for (source[0..block]) |*h| h.* = 0;
    for ([_]Precision{ .fp4, .fp8 }) |p| {
        var bytes: [Packed.bytes(.fp8, 2, cols)]u8 align(16) = @splat(0xff);
        const code_bytes = Packed.codeBytes(p, 2, cols);
        const scales = std.mem.bytesAsSlice(u16, bytes[code_bytes..Packed.bytes(p, 2, cols)]);
        packRows(p, &source, cols, bytes[0..code_bytes], @alignCast(scales), 0, 2);
        for (source, 0..) |h, i| {
            const code = if (p == .fp8) bytes[i] else (bytes[i / 2] >> @intCast(4 * (i & 1))) & 15;
            const scale = bf.value(scales[i / block]);
            const got = decode(p, code) * scale;
            const want = bf.value(h);
            // E2M1's widest step is 2 (between 4 and 6); E4M3's is an eighth of a binade.
            const step: f32 = if (p == .fp4) 2 * scale else @max(@abs(want) / 8, 0x1p-9 * scale);
            try std.testing.expect(@abs(got - want) <= step / 2 + 1e-12);
        }
        try std.testing.expectEqual(@as(u16, 0), scales[0]);
        const big = cols + 5;
        const big_code = if (p == .fp8) bytes[big] else (bytes[big / 2] >> 4) & 15;
        try std.testing.expect(@abs(decode(p, big_code) * bf.value(scales[big / block]) + 0.75) <= 0.75 / 64.0);
    }
}

test "the packed kernels match their codes decoded on the host" {
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const ids = 24;
    const rows = 3;
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const unembed = try gpa.alloc(u16, ids * k.hidden);
    defer gpa.free(unembed);
    for (unembed) |*h| h.* = bf.bits((rng.float(f32) - 0.5) * 0.1);
    var phi_src: [ids * k.rank]u16 = undefined;
    for (&phi_src) |*h| h.* = bf.bits((rng.float(f32) - 0.5) * 0.5);
    // y, beta, lambda, base, logits
    const sizes = [5]usize{ rows * k.hidden, ids, k.rank, rows * ids, ids };
    var b: [5]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (b[0..made]) |buf| buf.deinit();
    for (&b, sizes) |*buf, n| {
        buf.* = try device.buffer(n * 2, mtl.ResourceOptions.shared);
        made += 1;
    }
    const xs = b[0].slice(u16, sizes[0]);
    const beta = b[1].slice(u16, ids);
    const lambda = b[2].slice(u16, k.rank);
    for (xs) |*h| h.* = bf.bits(rng.float(f32) - 0.5);
    for (beta) |*h| h.* = bf.bits(rng.float(f32) - 0.5);
    for (lambda) |*h| h.* = bf.bits(rng.float(f32) * 2 - 1);
    const at = struct {
        fn ref(buf: mtl.Buffer) k.Ref {
            return .{ .buf = buf, .off = 0 };
        }
    }.ref;
    for ([_]Precision{ .fp4, .fp8 }) |precision| {
        var u = try Packed.init(device, precision, ids, k.hidden);
        defer u.deinit();
        try pack(u, unembed);
        var phi = try Packed.init(device, precision, ids, k.rank);
        defer phi.deinit();
        try pack(phi, &phi_src);
        var pipes = try Pipelines.load(device, precision);
        defer pipes.deinit();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        base(&pipes, e, u, at(b[0]), at(b[3]), rows, ids);
        logits(&pipes, e, at(b[3]), phi, at(b[1]), at(b[2]), at(b[4]), ids);
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        const got_base = b[3].slice(u16, sizes[3]);
        const got_logits = b[4].slice(u16, ids);
        for (0..rows) |row| for (0..ids) |id| {
            var want: f64 = 0;
            for (0..k.hidden) |c| want += weight(u, id, c) * bf.value(xs[row * k.hidden + c]);
            try std.testing.expectApproxEqAbs(want, bf.value(got_base[row * ids + id]), @abs(want) / 128 + 1e-4);
        };
        for (0..ids) |id| {
            var dot: f64 = 0;
            for (0..k.rank) |c| dot += weight(phi, id, c) * bf.value(lambda[c]);
            const want = k.foldedLogit(bf.value(got_base[id]), bf.value(beta[id]), @floatCast(dot));
            try std.testing.expectApproxEqAbs(want, bf.value(got_logits[id]), @abs(want) / 64 + 1e-3);
        }
    }
}

fn weight(p: Packed, row: usize, col: usize) f64 {
    const i = row * p.cols + col;
    const all = p.buffer.slice(u8, Packed.bytes(p.precision, p.rows, p.cols));
    const code = if (p.precision == .fp8) all[i] else (all[i / 2] >> @intCast(4 * (i & 1))) & 15;
    const scales: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, all[Packed.codeBytes(p.precision, p.rows, p.cols)..]));
    return decode(p.precision, code) * bf.value(scales[i / block]);
}
