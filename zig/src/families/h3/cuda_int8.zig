//! mmh3's INT8 linear layers on the H3 CUDA blocks: each projection's weight rows Hadamard-rotated and
//! quantized to int8 at the row's absmax (one fp32 scale a row, stored behind the codes); each activation
//! row rotated in 128-wide chunks and quantized at its own chunk's absmax (one fp32 scale a chunk), so a
//! loud chunk cannot shrink its neighbours. `h3_gemm_int8` sums the rotated codes exactly in int32 on
//! `m16n8k32`, folds each chunk's sum in at the activation scale and writes bf16 from its own tile —
//! no int32 round trip through a dequantize pass. The bf16 weights are freed once quantized.
//!
//! All four projections run this tile: `qkvc`, `out`, `fc1` and `fc2`.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const Model = h3.Model;

/// A projection (N, K): its int8 rows, then N fp32 row scales.
pub const Weight = struct {
    q: cuda.DeviceBuffer,
    n: usize,
    k: usize,

    fn scales(w: Weight) u64 {
        return w.q.ptr + w.n * w.k;
    }
};

/// One block's quantized projections.
pub const Block = struct { qkvc: Weight, out: Weight, fc1: Weight, fc2: Weight };

/// h3_gemm_int8's dynamic shared bytes: 2 stages of a 128-deep chunk — 256 rows of X, 128 rows of W and
/// the 256 activation scales of that chunk (cuda_int8's own; the contract repeats it so it needs no import).
pub const gemm_shared_bytes = 2 * (256 * 128 + 128 * 128 + 256 * 4);
/// The tile bands' L2 swizzle width, as tile_of's `group`.
const gemm_group = 8;

pub const Int8 = struct {
    blocks: []Block,
    /// int8 activation rows (rows, widest k), then the rows' per-chunk fp32 scales (rows, widest k / 128).
    scratch: cuda.DeviceBuffer,
    rows: usize = 0,

    /// Replaces every block's bf16 projections with int8 ones, freeing each bf16 buffer once it is quantized.
    pub fn quantize(self: *Model) !void {
        if (self.gated) return error.H3NotPictureModel; // the dense forward's three of q, k, v only
        try self.ctx.makeCurrent();
        const t = try self.gpa.create(Int8);
        t.* = .{ .blocks = &.{}, .scratch = .{ .d = &self.driver, .ptr = 0, .len = 0 } };
        self.int8 = t; // the model's deinit frees what a failed quantize already made
        const none: Weight = .{ .q = .{ .d = &self.driver, .ptr = 0, .len = 0 }, .n = 0, .k = 0 };
        t.blocks = try self.gpa.alloc(Block, self.blocks.len);
        @memset(t.blocks, .{ .qkvc = none, .out = none, .fc1 = none, .fc2 = none });
        for (self.blocks, t.blocks) |*b, *q| {
            q.qkvc = try weight(self, &b.qkvc, 3 * self.inner, self.hidden);
            q.out = try weight(self, &b.out, self.hidden, self.inner);
            q.fc1 = try weight(self, &b.fc1, 2 * self.mlp, self.hidden);
            q.fc2 = try weight(self, &b.fc2, self.hidden, self.mlp);
        }
    }

    fn weight(self: *Model, bf16: *cuda.DeviceBuffer, n: usize, k: usize) !Weight {
        if (k % 128 != 0 or k > 14336 or n % 8 != 0) return error.BadGeometry; // h3_quantize's quantize_chunks
        const w: Weight = .{ .q = try cuda.DeviceBuffer.alloc(&self.driver, n * k + n * 4), .n = n, .k = k };
        var args: cuda.Args = .{};
        for ([_]u64{ bf16.ptr, w.q.ptr, w.scales() }) |a| args.add(a);
        args.add(@as(c_int, @intCast(k)));
        try self.launch("h3_quantize", .{ .x = @intCast(n) }, 256, 0, &args);
        try self.stream.synchronize();
        bf16.free();
        bf16.* = .{ .d = &self.driver, .ptr = 0, .len = 0 };
        return w;
    }

    pub fn deinit(t: *Int8, self: *Model) void {
        for (t.blocks) |*b| for ([_]*Weight{ &b.qkvc, &b.out, &b.fc1, &b.fc2 }) |w| w.q.free();
        self.gpa.free(t.blocks);
        t.scratch.free();
        self.gpa.destroy(t);
    }

    /// y (rows, w.n) bf16 = x (rows, w.k) bf16 . w^T through the int8 tile.
    pub fn run(t: *Int8, self: *Model, x: u64, w: Weight, y: u64, rows: usize) !void {
        const p = try t.product(self, x, w, rows);
        var args: cuda.Args = .{};
        for ([_]u64{ p.xq, p.xs, w.q.ptr, w.scales(), y }) |a| args.add(a);
        for ([_]usize{ rows, w.n, w.k, gemm_group }) |n| args.add(@as(c_int, @intCast(n)));
        try self.launch("h3_gemm_int8", .{ .x = @intCast(tiles(rows, w.n)) }, 256, gemm_shared_bytes, &args);
    }

    /// act (rows, w.n / 2) = swiglu of x . w^T [gate | value]: the product writes bf16 `wide`,
    /// then the same bf16 `h3_swiglu` the unquantized path runs.
    pub fn mlpIn(t: *Int8, self: *Model, x: u64, w: Weight, wide: u64, act: u64, rows: usize) !void {
        try t.run(self, x, w, wide, rows);
        var args: cuda.Args = .{};
        for ([_]u64{ wide, act }) |a| args.add(a);
        args.add(@as(c_int, @intCast(rows)));
        args.add(@as(c_int, @intCast(w.n / 2)));
        try self.launch("h3_swiglu", .{ .x = 1024 }, 256, 0, &args);
    }

    /// Room for `rows` before a step is captured: `run` and `mlpIn` then only launch.
    pub fn warm(t: *Int8, self: *Model, rows: usize) !void {
        try t.reserve(self, rows);
    }

    /// x's rows quantized in scratch: their int8 codes `xq` and per-128-chunk scales `xs`.
    fn product(t: *Int8, self: *Model, x: u64, w: Weight, rows: usize) !struct { xq: u64, xs: u64 } {
        try t.reserve(self, rows);
        const xq = t.scratch.ptr;
        const xs = xq + rows * t.widest(self);
        var args: cuda.Args = .{};
        for ([_]u64{ x, xq, xs }) |a| args.add(a);
        args.add(@as(c_int, @intCast(w.k)));
        try self.launch("h3_quantize_chunks", .{ .x = @intCast(rows) }, 256, 0, &args);
        return .{ .xq = xq, .xs = xs };
    }

    fn widest(_: *const Int8, self: *const Model) usize {
        return @max(3 * self.inner, 2 * self.mlp, self.hidden);
    }

    fn reserve(t: *Int8, self: *Model, rows: usize) !void {
        if (rows <= t.rows) return;
        try self.stream.synchronize();
        t.scratch.free();
        t.scratch = try cuda.DeviceBuffer.alloc(&self.driver, rows * (t.widest(self) + t.widest(self) / 32));
        t.rows = rows;
    }
};

fn tiles(rows: usize, n: usize) usize {
    return ((rows + 255) / 256) * ((n + 127) / 128); // 256x128 CTAs
}
