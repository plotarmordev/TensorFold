//! One FastH3 block in bf16: products on H3's 32x64-warp GEMM (fp32 sums). A picture's (the released layout's) four
//! linears run on int8 operands (int32 sums), a scale per 256 values of K on both; a clip's stay bf16.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3.zig");

const kernels = @import("h3_kernels");

const Model = h3.Model;
const Ref = h3.Ref;

/// Threadgroups walk 2^swizzle row tiles under one weight tile before the next, so 64 row tiles (4,096 rows) read a
/// weight tile while it is still cached; MLX's 2 read every weight 16 times a product at 2048 x 2048. Sums unchanged.
const swizzle = 6;

/// d (rows, n) = x (rows, k) w^T for w (n, k) in bf16.
fn gemm(m: *const Model, enc: mtl.ComputeEncoder, x: mtl.Buffer, w: Ref, d: mtl.Buffer, n: usize, k: usize) void {
    const pipe = if (m.rows % 64 == 0) m.pipeline(kernels.gemm_32x64) else m.pipeline(kernels.gemm_32x64_ragged);
    product(enc, pipe, swizzle, x, w, d, m.rows, n, k, 128);
}

fn product(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, s: usize, x: mtl.Buffer, w: Ref, d: mtl.Buffer, rows: usize, n: usize, k: usize, comptime threads: usize) void {
    const band = @as(usize, 1) << @intCast(s);
    const tn = n / 128;
    const tm = (rows + 63) / 64;
    enc.setPipeline(pipe);
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(w.buffer, w.offset, 1);
    var p: [16]i32 = @splat(0);
    const values = [_]usize{ rows, n, k, k, k, n, tn, tm, s, k / 256 };
    for (values, 0..) |v, i| p[i] = @intCast(v);
    enc.setBytes(std.mem.asBytes(&p), 2);
    enc.setBuffer(d, 0, 3);
    enc.dispatchGroups(mtl.Size.of(tn * band, (tm + band - 1) / band, 1), mtl.Size.of(32, 4, threads / 128));
}

/// The int8 scale group: one scale a row for each 256 values of K, the product's K block.
pub const group = 256;

/// `rows` rows of x (k values each) to int8 in q, in groups of `g` values, each group's scale in s.
pub fn quantize(m: *const Model, enc: mtl.ComputeEncoder, x: Ref, rows: usize, k: usize, g: usize, q: Ref, s: Ref) void {
    enc.setPipeline(m.pipeline("h3_quantize_rows"));
    enc.setBuffer(x.buffer, x.offset, 0);
    enc.setValue([3]i32{ @intCast(rows), @intCast(k), @intCast(g) }, 1);
    enc.setBuffer(q.buffer, q.offset, 2);
    enc.setBuffer(s.buffer, s.offset, 3);
    enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
}

/// `d` = x w^T for a linear of n x k. A picture's: x to int8 a scale per 256 values of K, the weights' int8 rows
/// bound per block (`qi`); a clip's bf16. `quantized` when xq/xs already hold x's int8 rows (h3_norm_quant's).
fn linear(m: *Model, pass: *h3.Pass, x: mtl.Buffer, w: Ref, d: mtl.Buffer, n: usize, k: usize, qi: h3.Int8, quantized: bool) !void {
    if (!m.base) return gemm(m, pass.enc, x, w, d, n, k);
    if (!quantized) {
        quantize(m, pass.enc, .{ .buffer = x }, m.rows, k, group, .{ .buffer = m.xq }, .{ .buffer = m.xs });
        try m.lap(pass, "quantize");
    }
    const enc = pass.enc;
    // the scales' slots are past product's own; an encoder keeps them across its pipeline change
    enc.setBuffer(m.xs, 0, 4);
    enc.setBuffer(qi.s.buffer, qi.s.offset, 5);
    product(enc, m.pipeline(kernels.gemm_32x64_i8), swizzle, m.xq, qi.w, d, m.rows, n, k, 128);
}

/// The stream's modulated norm to bf16 rows in xn; with `gates` it first adds the branch in `y` gated by that part.
fn norm(m: *const Model, enc: mtl.ComputeEncoder, w: Ref, gates: ?Ref, gate: i32, tables: Ref, scale: i32, shift: i32) void {
    const g = gates orelse tables;
    enc.setPipeline(m.pipeline("h3_norm"));
    enc.setBuffer(m.x, 0, 0);
    enc.setBuffer(m.y, 0, 1);
    enc.setBuffer(w.buffer, w.offset, 2);
    enc.setBuffer(g.buffer, g.offset, 3);
    enc.setBuffer(tables.buffer, tables.offset, 4);
    enc.setBuffer(m.line, 0, 5);
    enc.setValue([6]i32{ @intCast(m.rows), @intCast(m.hidden), @intCast(m.lines), if (gates == null) -1 else gate, scale, shift }, 6);
    enc.setValue([1]f32{h3.eps}, 7);
    enc.setBuffer(m.xn, 0, 8);
    enc.dispatchGroups(mtl.Size.of(m.rows, 1, 1), mtl.Size.of(32, 1, 1));
}

/// norm's int8 form for the picture's second norm: the gated add and the modulated norm round exactly as norm
/// rounds them, then each row straight to int8 in xq with its per-256-value block scales in xs — the bf16 normed
/// rows are never written.
fn normQuant(m: *const Model, enc: mtl.ComputeEncoder, w: Ref, gates: ?Ref, gate: i32, tables: Ref, scale: i32, shift: i32) void {
    const g = gates orelse tables;
    enc.setPipeline(m.pipeline("h3_norm_quant"));
    enc.setBuffer(m.x, 0, 0);
    enc.setBuffer(m.y, 0, 1);
    enc.setBuffer(w.buffer, w.offset, 2);
    enc.setBuffer(g.buffer, g.offset, 3);
    enc.setBuffer(tables.buffer, tables.offset, 4);
    enc.setBuffer(m.line, 0, 5);
    enc.setValue([6]i32{ @intCast(m.rows), @intCast(m.hidden), @intCast(m.lines), if (gates == null) -1 else gate, scale, shift }, 6);
    enc.setValue([1]f32{h3.eps}, 7);
    enc.setBuffer(m.xq, 0, 8);
    enc.setBuffer(m.xs, 0, 9);
    enc.dispatchGroups(mtl.Size.of(m.rows, 1, 1), mtl.Size.of(32, 1, 1));
}

fn attend(m: *const Model, enc: mtl.ComputeEncoder, list: mtl.Buffer, queries: usize, keys: usize, first: usize, per_query: bool) void {
    enc.setPipeline(m.pipeline("h3_attention"));
    enc.setBuffer(m.tq, 0, 0);
    enc.setBuffer(m.tk, 0, 1);
    enc.setBuffer(m.tv, 0, 2);
    enc.setBuffer(list, 0, 3);
    enc.setBuffer(m.sizes.buffer, m.sizes.offset, 4);
    enc.setBuffer(m.row_of, 0, 5);
    enc.setValue([4]i32{ @intCast(queries), @intCast(keys), @intCast(first), @intFromBool(per_query) }, 6);
    enc.setValue([1]f32{1.0 / @sqrt(@as(f32, h3.head_dim))}, 7);
    enc.setBuffer(m.qr, 0, 8);
    enc.dispatchGroups(mtl.Size.of(queries, m.heads, 1), mtl.Size.of(256, 1, 1));
}

/// --far-pool: the target's pooled tiles, then the prefix's query tiles over every tile and each target tile over its
/// far-field list (shared by every head).
fn attendFar(m: *const Model, enc: mtl.ComputeEncoder) void {
    enc.setPipeline(m.pipeline("h3_far_pool"));
    enc.setBuffer(m.tk, 0, 0);
    enc.setBuffer(m.tv, 0, 1);
    enc.setValue([1]i32{@intCast(m.prefix_tiles)}, 2);
    enc.dispatchGroups(mtl.Size.of(m.far_pooled, m.heads, 1), mtl.Size.of(64, 1, 1));
    enc.barrier();
    attend(m, enc, m.all_tiles, m.prefix_tiles, m.tiles, 0, false);
    attend(m, enc, m.far_list, m.tiles - m.prefix_tiles, m.far_per_query, m.prefix_tiles, true);
}

/// Block `index` of a forward; the block before left its MLP's rows in y, added here with this block's first norm.
pub fn run(m: *Model, pass: *h3.Pass, index: usize) !void {
    const b = &m.blocks[index];
    const enc = &pass.enc;
    const heads = m.heads;
    const gates: ?Ref = if (index == 0) null else m.blocks[index - 1].tables;
    norm(m, enc.*, b.norm1, gates, h3.gate_m, b.tables, h3.scale_a, h3.shift_a);
    try m.lap(pass, "norm");
    if (m.base) try linear(m, pass, m.xn, b.q, m.qkv, 3 * m.inner, m.hidden, b.int8.q, false) else {
        gemm(m, enc.*, m.xn, b.q, m.qr, m.inner, m.hidden);
        gemm(m, enc.*, m.xn, b.k, m.kr, m.inner, m.hidden);
        gemm(m, enc.*, m.xn, b.v, m.vr, m.inner, m.hidden);
        if (!m.dense) gemm(m, enc.*, m.xn, b.compress, m.cr, m.inner, m.hidden);
    }
    try m.lap(pass, "q k v gate");
    enc.setPipeline(m.pipeline("h3_heads"));
    // the released layout's rows hold each head's q, k and v together: 384 values a row and head
    enc.setBuffer(if (m.base) m.qkv else m.qr, 0, 0);
    enc.setBuffer(if (m.base) m.qkv else m.kr, if (m.base) 2 * h3.head_dim else 0, 1);
    enc.setBuffer(if (m.base) m.qkv else m.vr, if (m.base) 4 * h3.head_dim else 0, 2);
    enc.setBuffer(b.norm_q.buffer, b.norm_q.offset, 3);
    enc.setBuffer(b.norm_k.buffer, b.norm_k.offset, 4);
    enc.setBuffer(m.cos.buffer, m.cos.offset, 5);
    enc.setBuffer(m.sin.buffer, m.sin.offset, 6);
    enc.setBuffer(m.slot.buffer, m.slot.offset, 7);
    enc.setValue([5]i32{ @intCast(m.rows), @intCast(heads), @intCast(m.rot), 0, if (m.base) 3 * h3.head_dim else h3.head_dim }, 8);
    enc.setValue([1]f32{h3.eps}, 9);
    enc.setBuffer(m.tq, 0, 10);
    enc.setBuffer(m.tk, 0, 11);
    enc.setBuffer(m.tv, 0, 12);
    enc.dispatchGroups(mtl.Size.of(m.rows, heads, 1), mtl.Size.of(32, 1, 1));
    try m.lap(pass, "heads");
    if (m.dense) {
        if (m.far_pooled > 0) attendFar(m, enc.*) else attend(m, enc.*, m.all_tiles, m.tiles, m.tiles, 0, false);
        try m.lap(pass, "attention");
    } else {
        enc.setPipeline(m.pipeline("h3_pool"));
        enc.setBuffer(m.tq, 0, 0);
        enc.setBuffer(m.tk, 0, 1);
        enc.setBuffer(m.tv, 0, 2);
        enc.setBuffer(m.sizes.buffer, m.sizes.offset, 3);
        enc.setBuffer(m.qp, 0, 4);
        enc.setBuffer(m.kp, 0, 5);
        enc.setBuffer(m.vp, 0, 6);
        enc.dispatchGroups(mtl.Size.of(m.tiles, heads, 1), mtl.Size.of(32, 1, 1));
        m.route(enc.*);
        try m.lap(pass, "routing");
        // the prefix's query tiles see every tile; a video tile sees the prefix and its chosen tiles
        attend(m, enc.*, m.all_tiles, m.prefix_tiles, m.tiles, 0, false);
        attend(m, enc.*, m.chosen, m.tiles - m.prefix_tiles, m.prefix_tiles + m.keep, m.prefix_tiles, true);
        try m.lap(pass, "attention");
        // the pooled branch gated into the attention rows in place
        enc.setPipeline(m.pipeline("h3_gate_mix"));
        enc.setBuffer(m.qr, 0, 0);
        enc.setBuffer(m.cr, 0, 1);
        enc.setBuffer(m.coarse, 0, 2);
        enc.setBuffer(m.slot.buffer, m.slot.offset, 3);
        enc.setValue([3]i32{ @intCast(m.rows), @intCast(heads), @intCast(m.tiles) }, 4);
        enc.dispatchThreads(mtl.Size.of(m.rows, heads, 1), mtl.Size.of(64, 4, 1));
        try m.lap(pass, "gate mix");
    }
    try linear(m, pass, m.qr, b.out, m.y, m.hidden, m.inner, b.int8.out, false);
    try m.lap(pass, "attention out");
    if (m.base) {
        normQuant(m, enc.*, b.norm2, b.tables, h3.gate_a, b.tables, h3.scale_m, h3.shift_m);
        try m.lap(pass, "norm quant");
    } else {
        norm(m, enc.*, b.norm2, b.tables, h3.gate_a, b.tables, h3.scale_m, h3.shift_m);
        try m.lap(pass, "norm");
    }
    // the first projection's rows are [value | gate] ([gate | value] in the released layout), one product
    try linear(m, pass, m.xn, b.fc1, m.wide, 2 * m.mlp, m.hidden, b.int8.fc1, true);
    try m.lap(pass, "mlp in");
    enc.setPipeline(m.pipeline("h3_swiglu"));
    enc.setBuffer(m.wide, 0, 0);
    enc.setValue([3]i32{ @intCast(m.rows), @intCast(m.mlp), @intFromBool(m.base) }, 1);
    enc.setBuffer(m.act, 0, 2);
    enc.dispatchThreads(mtl.Size.of(m.mlp, m.rows, 1), mtl.Size.of(256, 1, 1));
    try m.lap(pass, "swiglu");
    try linear(m, pass, m.act, b.fc2, m.y, m.hidden, m.mlp, b.int8.fc2, false);
    try m.lap(pass, "mlp out");
}

test "h3 gemm: 32x64-warp products write every tile as MLX's 32x32 band of 4 does, across a band edge and a ragged tile" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const text = try @import("source.zig").library(a, 56, 18432, 288, "");
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const whole_pipe = try mtl.Pipeline.init(device, lib, kernels.gemm_whole, false);
    defer whole_pipe.deinit();
    const ragged_pipe = try mtl.Pipeline.init(device, lib, kernels.gemm_ragged, false);
    defer ragged_pipe.deinit();
    const wide_pipe = try mtl.Pipeline.init(device, lib, kernels.gemm_32x64, false);
    defer wide_pipe.deinit();
    const wide_ragged_pipe = try mtl.Pipeline.init(device, lib, kernels.gemm_32x64_ragged, false);
    defer wide_ragged_pipe.deinit();
    const n = 256;
    const k = 512;
    const guard = 64 * n; // one more row tile than the product writes, which must stay untouched
    var prng = std.Random.DefaultPrng.init(81);
    // a whole band of 64 tiles, one tile past it, and 65 tiles plus a ragged tail
    for ([_]usize{ 64 * 64 + 24, 64 * 65, 64 * 65 + 24 }) |rows| {
        const whole = rows % 64 == 0;
        const shared = mtl.ResourceOptions.shared;
        const x = try device.buffer(rows * k * 2, shared);
        defer x.deinit();
        const w = try device.buffer(n * k * 2, shared);
        defer w.deinit();
        // bf16 values of magnitude [1, 2): seven random mantissa bits and a random sign
        for ([_][]u16{ x.slice(u16, rows * k), w.slice(u16, n * k) }) |values| {
            for (values) |*v| v.* = 0x3f80 | (prng.random().int(u16) & 0x807f);
        }
        const old = try device.buffer((rows * n + guard) * 2, shared);
        defer old.deinit();
        const wide = try device.buffer((rows * n + guard) * 2, shared);
        defer wide.deinit();
        for ([_]mtl.Buffer{ old, wide }) |d| @memset(d.slice(u16, rows * n + guard), 0xffff); // a NaN no product of these inputs makes
        // MLX's geometry: 32 x 32 a warp, 256 threads; ours: 32 x 64, 128 threads
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.concurrent);
            product(enc, if (whole) whole_pipe else ragged_pipe, 2, x, .{ .buffer = w }, old, rows, n, k, 256);
            enc.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
        }
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.concurrent);
            product(enc, if (whole) wide_pipe else wide_ragged_pipe, swizzle, x, .{ .buffer = w }, wide, rows, n, k, 128);
            enc.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
        }
        const after = wide.slice(u16, rows * n + guard);
        try std.testing.expectEqualSlices(u16, old.slice(u16, rows * n + guard), after);
        try std.testing.expect(std.mem.indexOfScalar(u16, after[0 .. rows * n], 0xffff) == null);
        for (after[rows * n ..]) |v| try std.testing.expectEqual(@as(u16, 0xffff), v);
    }
}

test "h3 gemm: int8 operands give their exact dequantized product to bf16 rounding, across a ragged tile" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const text = try @import("source.zig").library(a, 56, 18432, 288, "");
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var m: Model = undefined;
    inline for (kernels.names, 0..) |name, i| m.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
    defer for (m.pipelines) |p| p.deinit();
    const rows = 64 * 3 + 24;
    const n = 256;
    const k = 2 * group;
    const groups = k / group;
    const guard = 64 * n; // past the last row, which must stay untouched
    const shared = mtl.ResourceOptions.shared;
    const x = try device.buffer(rows * k * 2, shared);
    defer x.deinit();
    const w = try device.buffer(n * k * 2, shared);
    defer w.deinit();
    var prng = std.Random.DefaultPrng.init(84);
    // bf16 values of magnitude [1, 2) times a power of two that differs by row, group and weight group
    for (x.slice(u16, rows * k), 0..) |*v, i| v.* = (0x3f80 + @as(u16, @intCast(i / group % 8)) * 0x80) | (prng.random().int(u16) & 0x807f);
    for (w.slice(u16, n * k), 0..) |*v, i| v.* = (0x3c00 + @as(u16, @intCast(i / group % 16)) * 0x80) | (prng.random().int(u16) & 0x807f);
    const xq = try device.buffer(rows * k, shared);
    defer xq.deinit();
    const xs = try device.buffer(rows * groups * 4, shared);
    defer xs.deinit();
    const wq = try device.buffer(n * k, shared);
    defer wq.deinit();
    const ws = try device.buffer(n * groups * 4, shared);
    defer ws.deinit();
    const d = try device.buffer((rows * n + guard) * 2, shared);
    defer d.deinit();
    @memset(d.slice(u16, rows * n + guard), 0xffff); // a NaN no product of these inputs makes
    const cb = queue.commandBuffer();
    const enc = cb.compute(.concurrent);
    quantize(&m, enc, .{ .buffer = x }, rows, k, group, .{ .buffer = xq }, .{ .buffer = xs });
    quantize(&m, enc, .{ .buffer = w }, n, k, group, .{ .buffer = wq }, .{ .buffer = ws });
    enc.barrier();
    enc.setBuffer(xs, 0, 4);
    enc.setBuffer(ws, 0, 5);
    product(enc, m.pipeline(kernels.gemm_32x64_i8), swizzle, xq, .{ .buffer = wq }, d, rows, n, k, 128);
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);

    const bf = struct {
        fn f(v: u16) f32 {
            return @bitCast(@as(u32, v) << 16);
        }
    }.f;
    // each group's largest magnitude maps to 127 and every value to its nearest step of the group's scale
    const sx = xs.slice(f32, rows * groups);
    const sw = ws.slice(f32, n * groups);
    for ([_][]const u16{ x.slice(u16, rows * k), w.slice(u16, n * k) }, [_][]const i8{ xq.slice(i8, rows * k), wq.slice(i8, n * k) }, [_][]const f32{ sx, sw }, [_]usize{ group, group }) |src, q, s, g| {
        for (s, 0..) |scale, gi| {
            var most: f32 = 0;
            for (src[gi * g ..][0..g]) |v| most = @max(most, @abs(bf(v)));
            try std.testing.expectApproxEqRel(most / 127, scale, 1e-6);
            for (src[gi * g ..][0..g], q[gi * g ..][0..g]) |v, qi| try std.testing.expect(@abs(bf(v) / scale - @as(f32, @floatFromInt(qi))) <= 0.5 + 1e-3);
        }
    }
    const xi = xq.slice(i8, rows * k);
    const wi = wq.slice(i8, n * k);
    const out = d.slice(u16, rows * n + guard);
    for (0..rows) |r| for (0..n) |c| {
        var exact: f64 = 0;
        var size: f64 = 0;
        for (0..k) |j| {
            const term = @as(f64, @floatFromInt(@as(i32, xi[r * k + j]) * wi[c * k + j])) * sx[r * groups + j / group] * sw[c * groups + j / group];
            exact += term;
            size += @abs(term);
        }
        // one bf16 rounding of the result, and fp32 folds of the groups' sums
        try std.testing.expect(@abs(bf(out[r * n + c]) - exact) <= @abs(exact) * 0x1p-8 + size * 1e-6);
    };
    for (out[rows * n ..]) |v| try std.testing.expectEqual(@as(u16, 0xffff), v);
}

test "h3 fused: h3_norm_quant's rows are bit-identical to h3_norm then h3_quantize_rows" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const text = try @import("source.zig").library(a, 56, 18432, 288, "");
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var m: Model = undefined;
    inline for (kernels.names, 0..) |name, i| m.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
    defer for (m.pipelines) |p| p.deinit();
    const rows = 9;
    const C = 512;
    const lines = 3;
    const shared = mtl.ResourceOptions.shared;
    var prng = std.Random.DefaultPrng.init(91);
    const mk = struct {
        fn f(device_: mtl.Device, n_: usize, opts: usize, prng_: *std.Random.DefaultPrng) !mtl.Buffer {
            const b = try device_.buffer(n_ * 2, opts);
            for (b.slice(u16, n_)) |*v| v.* = 0x3c00 | (prng_.random().int(u16) & 0x83ff);
            return b;
        }
    }.f;
    const x = try mk(device, rows * C, shared, &prng);
    defer x.deinit();
    const x2 = try device.buffer(rows * C * 2, shared);
    defer x2.deinit();
    @memcpy(x2.slice(u16, rows * C), x.slice(u16, rows * C));
    const y = try mk(device, rows * C, shared, &prng);
    defer y.deinit();
    const nw = try mk(device, C, shared, &prng);
    defer nw.deinit();
    const tab = try mk(device, 6 * lines * C, shared, &prng);
    defer tab.deinit();
    const line = try device.buffer(rows * 4, shared);
    defer line.deinit();
    for (line.slice(i32, rows), 0..) |*v, i| v.* = @intCast(i % lines);
    const xn = try device.buffer(rows * C * 2, shared);
    defer xn.deinit();
    const xq = try device.buffer(rows * C, shared);
    defer xq.deinit();
    const xs = try device.buffer(rows * (C / group) * 4, shared);
    defer xs.deinit();
    const xq2 = try device.buffer(rows * C, shared);
    defer xq2.deinit();
    const xs2 = try device.buffer(rows * (C / group) * 4, shared);
    defer xs2.deinit();
    m.x = x;
    m.y = y;
    m.xn = xn;
    m.xq = xq;
    m.xs = xs;
    m.line = line;
    m.rows = rows;
    m.hidden = C;
    m.lines = lines;
    const gate: i32 = 2;
    const scale: i32 = 1;
    const shift: i32 = 0;
    const parts = [6]i32{ @intCast(rows), @intCast(C), @intCast(lines), gate, scale, shift };
    {
        // the two launches the fused kernel folds: h3_norm (gated add + modulated norm), then h3_quantize_rows
        // at the gemm's 256-value block grouping
        const cb = queue.commandBuffer();
        const enc = cb.compute(.concurrent);
        enc.setPipeline(m.pipeline("h3_norm"));
        enc.setBuffer(x, 0, 0);
        enc.setBuffer(y, 0, 1);
        enc.setBuffer(nw, 0, 2);
        enc.setBuffer(tab, 0, 3);
        enc.setBuffer(tab, 0, 4);
        enc.setBuffer(line, 0, 5);
        enc.setBytes(std.mem.asBytes(&parts), 6);
        enc.setValue([1]f32{h3.eps}, 7);
        enc.setBuffer(xn, 0, 8);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
        enc.barrier();
        quantize(&m, enc, .{ .buffer = xn }, rows, C, group, .{ .buffer = xq }, .{ .buffer = xs });
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
    }
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.concurrent);
        enc.setPipeline(m.pipeline("h3_norm_quant"));
        enc.setBuffer(x2, 0, 0);
        enc.setBuffer(y, 0, 1);
        enc.setBuffer(nw, 0, 2);
        enc.setBuffer(tab, 0, 3);
        enc.setBuffer(tab, 0, 4);
        enc.setBuffer(line, 0, 5);
        enc.setBytes(std.mem.asBytes(&parts), 6);
        enc.setValue([1]f32{h3.eps}, 7);
        enc.setBuffer(xq2, 0, 8);
        enc.setBuffer(xs2, 0, 9);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
    }
    // the fused kernel's in-place add must match h3_norm's, its quantized rows and block scales h3_quantize_rows',
    // bit for bit
    try std.testing.expectEqualSlices(u16, x.slice(u16, rows * C), x2.slice(u16, rows * C));
    try std.testing.expectEqualSlices(i8, xq.slice(i8, rows * C), xq2.slice(i8, rows * C));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(xs.slice(f32, rows * (C / group))), std.mem.sliceAsBytes(xs2.slice(f32, rows * (C / group))));
}

test "h3 gemm: a clip's linears stay bf16, bit for bit main's MLX product, whole and ragged" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const text = try @import("source.zig").library(a, 56, 18432, 288, "");
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    // a clip's model: no int8 weights bound (wq, ws, xq and xs stay unset), so a picture's path would fault
    var m: Model = undefined;
    inline for (kernels.names, 0..) |name, i| m.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
    defer for (m.pipelines) |p| p.deinit();
    m.base = false;
    m.mlp = 128;
    m.hidden = 2 * group;
    const n = 2 * m.mlp;
    const k = m.hidden;
    var prng = std.Random.DefaultPrng.init(87);
    for ([_]usize{ 64 * 3, 64 * 3 + 24 }) |rows| {
        m.rows = rows;
        const shared = mtl.ResourceOptions.shared;
        m.xn = try device.buffer(rows * k * 2, shared);
        defer m.xn.deinit();
        const w = try device.buffer(n * k * 2, shared);
        defer w.deinit();
        for ([_][]u16{ m.xn.slice(u16, rows * k), w.slice(u16, n * k) }) |values| {
            for (values) |*v| v.* = 0x3f80 | (prng.random().int(u16) & 0x807f);
        }
        const old = try device.buffer(rows * n * 2, shared);
        defer old.deinit();
        m.wide = try device.buffer(rows * n * 2, shared);
        defer m.wide.deinit();
        const mlx = try mtl.Pipeline.init(device, lib, if (rows % 64 == 0) kernels.gemm_whole else kernels.gemm_ragged, false);
        defer mlx.deinit();
        var pass: h3.Pass = .{ .cb = queue.commandBuffer(), .enc = undefined };
        pass.enc = pass.cb.compute(.concurrent);
        product(pass.enc, mlx, 2, m.xn, .{ .buffer = w }, old, rows, n, k, 256);
        try linear(&m, &pass, m.xn, .{ .buffer = w }, m.wide, n, k, .{ .w = .{ .buffer = w }, .s = .{ .buffer = w } }, false);
        pass.enc.end();
        pass.cb.commit();
        pass.cb.wait();
        try std.testing.expect(pass.cb.failure() == null);
        try std.testing.expectEqualSlices(u16, old.slice(u16, rows * n), m.wide.slice(u16, rows * n));
    }
}

/// One head's attention over `tiles` tiles (and `pooled` far-field tiles past them) with the case's buffers in `m`.
fn attendOnce(device: mtl.Device, queue: mtl.Queue, m: *Model, tiles: usize, pooled: usize) !void {
    const a = std.testing.allocator;
    const text = try @import("source.zig").library(a, 1, (tiles + pooled) * 64, tiles, if (pooled > 0) "#define H3_HIDDEN 128\n#define H3_FAR 1\n" else "#define H3_HIDDEN 128\n");
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    inline for (kernels.names, 0..) |name, i| m.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
    defer for (m.pipelines) |p| p.deinit();
    m.tiles = tiles;
    m.far_pooled = pooled;
    m.dense = true;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.concurrent);
    if (pooled > 0) attendFar(m, enc) else attend(m, enc, m.all_tiles, tiles, tiles, 0, false);
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
}

test "h3 far field: where each 2 x 2 block's keys and values are equal, pooled far keys give dense attention" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    // one head; a prefix tile of 24 text rows, then a 4 x 128 target: 8 tiles, 2 pooled rows of one tile each
    const prefix = 24;
    const tiles = 9;
    const slots = (tiles + 2) * 64;
    const rows = prefix + 4 * 128;
    const shared = mtl.ResourceOptions.shared;
    var m: Model = undefined;
    m.heads = 1;
    m.prefix_tiles = 1;
    inline for (.{ "tq", "tk", "tv" }) |f| @field(m, f) = try device.buffer(slots * 128 * 2, shared);
    defer inline for (.{ "tq", "tk", "tv" }) |f| @field(m, f).deinit();
    m.row_of = try device.buffer(slots * 4, shared);
    defer m.row_of.deinit();
    m.all_tiles = try device.buffer(tiles * 4, shared);
    defer m.all_tiles.deinit();
    const sizes = try device.buffer(tiles * 4, shared);
    defer sizes.deinit();
    m.sizes = .{ .buffer = sizes };
    m.far_list = try device.buffer(8 * 6 * 4, shared);
    defer m.far_list.deinit();
    m.far_per_query = 6;
    const dense = try device.buffer(rows * 128 * 2, shared);
    defer dense.deinit();
    const far = try device.buffer(rows * 128 * 2, shared);
    defer far.deinit();
    var prng = std.Random.DefaultPrng.init(88);
    const r = prng.random();
    for ([_]mtl.Buffer{ m.tq, m.tk, m.tv }) |b| @memset(b.slice(u16, slots * 128), 0);
    @memset(m.row_of.slice(i32, slots), -1);
    for (0..prefix) |row| m.row_of.slice(i32, slots)[row] = @intCast(row);
    for (0..rows - prefix) |row| m.row_of.slice(i32, slots)[64 + row] = @intCast(prefix + row);
    for (m.all_tiles.slice(i32, tiles), 0..) |*t, i| t.* = @intCast(i);
    for (sizes.slice(i32, tiles), 0..) |*s, i| s.* = if (i == 0) prefix else 64;
    // the prefix's and target's rows; a target slot's key and value are its 2 x 2 block's top-left one's
    const q = m.tq.slice(u16, slots * 128);
    const k = m.tk.slice(u16, slots * 128);
    const v = m.tv.slice(u16, slots * 128);
    for (0..64 * tiles) |s| if (s < prefix or s >= 64) {
        for (0..128) |d| q[s * 128 + d] = 0x3e00 | (r.int(u16) & 0x807f); // magnitude [0.125, 0.25)
    };
    for (0..64 * tiles) |s| if (s < prefix or s >= 64) {
        const t = s -| 64;
        const top = if (s < 64) s else 64 + (t / 128 / 2 * 2) * 128 + t % 128 / 2 * 2;
        for (0..128) |d| {
            if (top == s) {
                k[s * 128 + d] = 0x3f80 | (r.int(u16) & 0x807f);
                v[s * 128 + d] = 0x3f80 | (r.int(u16) & 0x807f);
            } else {
                k[s * 128 + d] = k[top * 128 + d];
                v[s * 128 + d] = v[top * 128 + d];
            }
        }
    };
    // a query tile in pooled row p: the prefix, its own row's four tiles, the other row's pooled tile
    for (0..8) |t| {
        const p = t / 4;
        const base: i32 = @intCast(1 + 4 * p);
        @memcpy(m.far_list.slice(i32, 48)[t * 6 ..][0..6], &[_]i32{ 0, base, base + 1, base + 2, base + 3, @intCast(tiles + 1 - p) });
    }
    m.qr = dense;
    try attendOnce(device, queue, &m, tiles, 0);
    m.qr = far;
    try attendOnce(device, queue, &m, tiles, 2);
    // relative to the largest output: one bf16 step, the sums in another order
    var worst: f32 = 0;
    var largest: f32 = 0;
    for (dense.slice(u16, rows * 128), far.slice(u16, rows * 128)) |want, got| {
        const x: f32 = @bitCast(@as(u32, want) << 16);
        const y: f32 = @bitCast(@as(u32, got) << 16);
        worst = @max(worst, @abs(x - y));
        largest = @max(largest, @abs(x));
    }
    errdefer std.debug.print("worst difference {e} of largest {e}\n", .{ worst, largest });
    try std.testing.expect(largest > 0 and worst <= largest / 128.0);
}
