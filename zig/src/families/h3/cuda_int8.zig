//! mmh3's INT8 linear layers on the H3 CUDA blocks: each projection's weight rows and each activation row
//! Hadamard-rotated in 128-wide chunks (ConvRot's outlier spreading), quantized to int8 at each chunk's absmax,
//! summed exactly in int32 by h3_gemm_int8's tensor-core tile, and scaled back to bf16 from the fp32 accumulator.
//! All four projections run int8: a per-128-chunk scale carries the picture where a per-row one could not, and the
//! product writing bf16 from its own tile leaves no int32 sums to round-trip. The bf16 weights are freed once
//! quantized.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const Model = h3.Model;

/// A projection (N, K): its int8 rows, then N * K / 128 fp32 chunk scales.
pub const Weight = struct {
    q: cuda.DeviceBuffer,
    n: usize,
    k: usize,

    fn scales(w: Weight) u64 {
        return w.q.ptr + w.n * w.k;
    }

    fn chunks(w: Weight) usize {
        return w.k / 128;
    }
};

/// One block's quantized projections.
pub const Block = struct { qkvc: Weight, out: Weight, fc1: Weight, fc2: Weight };

pub const Int8 = struct {
    blocks: []Block,
    /// int8 rows (rows, widest k), then the rows' fp32 chunk scales (rows, widest k / 128).
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
        if (k % 128 != 0 or k > 14336 or n % 8 != 0) return error.BadGeometry; // h3_quantize's chunks, the gemm's tiles
        const w: Weight = .{ .q = try cuda.DeviceBuffer.alloc(&self.driver, n * k + n * (k / 128) * 4), .n = n, .k = k };
        try launchQuantize(self, bf16.ptr, w.q.ptr, w.scales(), n, k);
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

    /// y (rows, w.n) bf16 = x (rows, w.k) bf16 . w^T through the int8 product.
    pub fn run(t: *Int8, self: *Model, x: u64, w: Weight, y: u64, rows: usize) !void {
        const p = try t.product(self, x, w, rows);
        var args: cuda.Args = .{};
        for ([_]u64{ p.xq, p.xs, w.q.ptr, w.scales(), y }) |a| args.add(a);
        for ([_]usize{ rows, w.n, w.k, gemm_group }) |n| args.add(@as(c_int, @intCast(n)));
        try self.launch("h3_gemm_int8", .{ .x = @intCast(tiles(rows, w.n)) }, 256, gemm_shared_bytes, &args);
    }

    /// act (rows, w.n / 2) = swiglu of x . w^T [gate | value]: the product into `wide`, then the bf16 swiglu.
    pub fn mlpIn(t: *Int8, self: *Model, x: u64, w: Weight, wide: u64, act: u64, rows: usize) !void {
        try t.run(self, x, w, wide, rows);
        var args: cuda.Args = .{};
        for ([_]u64{ wide, act }) |a| args.add(a);
        args.add(@as(c_int, @intCast(rows)));
        args.add(@as(c_int, @intCast(w.n / 2)));
        try self.launch("h3_swiglu", .{ .x = 1024 }, 256, 0, &args);
    }

    /// x's rows quantized in scratch: their int8 codes `xq` and per-128-chunk scales `xs`.
    fn product(t: *Int8, self: *Model, x: u64, w: Weight, rows: usize) !struct { xq: u64, xs: u64 } {
        try t.reserve(self, rows);
        const xq = t.scratch.ptr;
        const xs = xq + rows * t.widest(self);
        try launchQuantize(self, x, xq, xs, rows, w.k);
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

/// h3_gemm_int8's dynamic shared bytes: its 3 stages of a 128-deep chunk of BM rows, BN columns and their
/// scales fill sm_121's 101376-byte per-block cap exactly.
pub const gemm_shared_bytes = 3 * (128 * 128 + 128 * 128 + (128 + 128) * 4);
/// The tile bands' L2 swizzle width, as tile_of's `group`.
const gemm_group = 8;

fn tiles(rows: usize, n: usize) usize {
    return ((rows + 127) / 128) * ((n + 127) / 128);
}

fn launchQuantize(self: *Model, x: u64, q: u64, s: u64, rows: usize, k: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ x, q, s }) |a| args.add(a);
    args.add(@as(c_int, @intCast(k)));
    try self.launch("h3_quantize", .{ .x = @intCast(rows) }, 256, 0, &args);
}
