//! mmh3's INT8 linear layers on the H3 CUDA blocks: each projection's weight rows and each activation row Hadamard-rotated
//! in 128-wide chunks (ConvRot's outlier spreading), quantized to int8 at its absmax and summed exactly in int32 by
//! cuBLASLt. The sums' consumers (`h3_heads_int8`, `h3_swiglu_int8`) scale them to bf16 as they read them, so no bf16
//! rows of the product are written. The bf16 weights are freed once quantized.
//!
//! Only the two projections whose products int8 makes faster are quantized: `qkvc` and `fc1`. `out` and `fc2` measured
//! no faster (they are bound by their weight reads, not their arithmetic) and their quantization was most of the
//! picture's distance from episode 1's Mac picture, so they keep their bf16 weights.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;
const h3 = @import("cuda.zig");
const workspace_bytes = @import("cuda_gemm.zig").workspace_bytes;
const Model = h3.Model;

/// A projection (N, K): its int8 rows, then N fp32 row scales.
pub const Weight = struct {
    q: cuda.DeviceBuffer,
    n: usize,
    k: usize,

    pub fn scales(w: Weight) u64 {
        return w.q.ptr + w.n * w.k;
    }
};

/// One block's quantized projections.
pub const Block = struct { qkvc: Weight, fc1: Weight };

/// An int8 product in scratch: int32 sums (rows, n) and the rows' fp32 scales.
pub const Product = struct { acc: u64, xs: u64 };

pub const Int8 = struct {
    blocks: []Block,
    handle: lt_.Handle,
    plans: std.AutoHashMapUnmanaged([3]u64, lt_.Gemm) = .empty,
    /// int32 sums (rows, widest n), then int8 rows (rows, widest k), then the rows' fp32 scales.
    scratch: cuda.DeviceBuffer,
    rows: usize = 0,

    /// Replaces every block's bf16 projections with int8 ones, freeing each bf16 buffer once it is quantized.
    pub fn quantize(self: *Model) !void {
        if (self.gated) return error.H3NotPictureModel; // the dense forward's three of q, k, v only
        try self.ctx.makeCurrent();
        var handle: lt_.Handle = null;
        try self.gemms.lt.check(self.gemms.lt.api.cublasLtCreate(&handle), "cublasLtCreate");
        const t = try self.gpa.create(Int8);
        t.* = .{ .blocks = &.{}, .handle = handle, .scratch = .{ .d = &self.driver, .ptr = 0, .len = 0 } };
        self.int8 = t; // the model's deinit frees what a failed quantize already made
        const none: Weight = .{ .q = .{ .d = &self.driver, .ptr = 0, .len = 0 }, .n = 0, .k = 0 };
        t.blocks = try self.gpa.alloc(Block, self.blocks.len);
        @memset(t.blocks, .{ .qkvc = none, .fc1 = none });
        for (self.blocks, t.blocks) |*b, *q| {
            q.qkvc = try weight(self, &b.qkvc, 3 * self.inner, self.hidden);
            q.fc1 = try weight(self, &b.fc1, 2 * self.mlp, self.hidden);
        }
    }

    fn weight(self: *Model, bf16: *cuda.DeviceBuffer, n: usize, k: usize) !Weight {
        if (k % 128 != 0 or k > 14336 or n % 4 != 0) return error.BadGeometry; // h3_quantize's quantize_chunks
        const w: Weight = .{ .q = try cuda.DeviceBuffer.alloc(&self.driver, n * k + n * 4), .n = n, .k = k };
        try launchQuantize(self, bf16.ptr, w.q.ptr, w.scales(), n, k);
        try self.stream.synchronize();
        bf16.free();
        bf16.* = .{ .d = &self.driver, .ptr = 0, .len = 0 };
        return w;
    }

    pub fn deinit(t: *Int8, self: *Model) void {
        var it = t.plans.valueIterator();
        while (it.next()) |g| g.deinit();
        t.plans.deinit(self.gpa);
        for (t.blocks) |*b| for ([_]*Weight{ &b.qkvc, &b.fc1 }) |w| w.q.free();
        self.gpa.free(t.blocks);
        t.scratch.free();
        _ = self.gemms.lt.api.cublasLtDestroy(t.handle);
        self.gpa.destroy(t);
    }

    /// act (rows, w.n / 2) = swiglu of x . w^T [gate | value], the int32 sums scaled in h3_swiglu_int8 itself.
    pub fn mlpIn(t: *Int8, self: *Model, x: u64, w: Weight, act: u64, rows: usize) !void {
        const p = try t.product(self, x, w, rows);
        var args: cuda.Args = .{};
        for ([_]u64{ p.acc, p.xs, w.scales(), act }) |a| args.add(a);
        args.add(@as(c_int, @intCast(rows)));
        args.add(@as(c_int, @intCast(w.n / 2)));
        try self.launch("h3_swiglu_int8", .{ .x = 1024 }, 256, 0, &args);
    }

    /// x's rows quantized and their int32 sums against w in scratch: `acc` (rows, w.n), the rows' scales `xs`.
    pub fn product(t: *Int8, self: *Model, x: u64, w: Weight, rows: usize) !Product {
        try t.reserve(self, rows);
        const acc = t.scratch.ptr;
        const xq = acc + rows * t.widest(self) * 4;
        const xs = xq + rows * t.widest(self);
        try launchQuantize(self, x, xq, xs, rows, w.k);
        const p = try t.plan(self, rows, w.n, w.k);
        try p.runInt(w.q.ptr, xq, acc, self.gemms.workspace.ptr, workspace_bytes, self.stream.handle);
        return .{ .acc = acc, .xs = xs };
    }

    fn widest(_: *const Int8, self: *const Model) usize {
        return @max(3 * self.inner, 2 * self.mlp, self.hidden);
    }

    fn reserve(t: *Int8, self: *Model, rows: usize) !void {
        if (rows <= t.rows) return;
        try self.stream.synchronize();
        t.scratch.free();
        t.scratch = try cuda.DeviceBuffer.alloc(&self.driver, rows * (t.widest(self) * 5 + 4));
        t.rows = rows;
    }

    /// Column-major D^T (n, rows) = W (k, n stored)^T . X^T (k, rows), int32 sums: cuBLASLt's int8 TN layout.
    fn plan(t: *Int8, self: *Model, rows: usize, n: usize, k: usize) !*const lt_.Gemm {
        const got = try t.plans.getOrPut(self.gpa, .{ rows, n, k });
        if (got.found_existing) return got.value_ptr;
        errdefer t.plans.removeByPtr(got.key_ptr);
        got.value_ptr.* = try lt_.Gemm.init(&self.gemms.lt, t.handle, .{ .ta = .t, .m = n, .n = rows, .k = k, .a = .i8, .b = .i8, .c = .i32, .lda = k, .ldb = k, .ldc = n }, workspace_bytes);
        return got.value_ptr;
    }
};

fn launchQuantize(self: *Model, x: u64, q: u64, s: u64, rows: usize, k: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ x, q, s }) |a| args.add(a);
    args.add(@as(c_int, @intCast(k)));
    try self.launch("h3_quantize", .{ .x = @intCast(rows) }, 256, 0, &args);
}
