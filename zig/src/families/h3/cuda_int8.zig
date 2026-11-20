//! mmh3's INT8 linear layers on the H3 CUDA blocks: each projection's weight rows and each activation row Hadamard-rotated
//! in 128-wide chunks (ConvRot's outlier spreading), quantized to int8 at absmax scales, summed exactly in int32 by
//! cuBLASLt, and scaled back to bf16 rows. The bf16 weights are freed once quantized.
//!
//! `qkvc` and `fc1` quantize each row at one scale; `out` and `fc2` scale every 256-wide group of K, the fold card 136's
//! picture needed (per-row scales there measured 28.15 dB and 4.23 against the 28.2 dB and 4.1 gates). A grouped product
//! is one strided-batched cuBLASLt call, a batch a 256-deep slice of K writing its own (rows, n) int32 plane, whose
//! partials h3_dequantize_grouped folds at both operands' per-group scales.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;
const h3 = @import("cuda.zig");
const workspace_bytes = @import("cuda_gemm.zig").workspace_bytes;
const Model = h3.Model;

/// K values one scale covers in a grouped projection.
pub const group_k = 256;

/// A projection (N, K): its int8 rows, then N * `groups` fp32 scales, one per `group_k`-wide group when `groups` > 1.
pub const Weight = struct {
    q: cuda.DeviceBuffer,
    n: usize,
    k: usize,
    groups: usize = 1,

    fn scales(w: Weight) u64 {
        return w.q.ptr + w.n * w.k;
    }
};

/// One block's quantized projections.
pub const Block = struct { qkvc: Weight, out: Weight, fc1: Weight, fc2: Weight };

pub const Int8 = struct {
    blocks: []Block,
    handle: lt_.Handle,
    plans: std.AutoHashMapUnmanaged([4]u64, lt_.Gemm) = .empty,
    /// int32 sums (rows x `accWide`: a product's n x groups), then int8 rows (rows, widest k), then their fp32 scales.
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
        @memset(t.blocks, .{ .qkvc = none, .out = none, .fc1 = none, .fc2 = none });
        for (self.blocks, t.blocks) |*b, *q| {
            q.qkvc = try weight(self, &b.qkvc, 3 * self.inner, self.hidden, 1);
            q.out = try weight(self, &b.out, self.hidden, self.inner, self.inner / group_k);
            q.fc1 = try weight(self, &b.fc1, 2 * self.mlp, self.hidden, 1);
            q.fc2 = try weight(self, &b.fc2, self.hidden, self.mlp, self.mlp / group_k);
        }
    }

    fn weight(self: *Model, bf16: *cuda.DeviceBuffer, n: usize, k: usize, groups: usize) !Weight {
        if (k % group_k != 0 or k > 14336 or n % 4 != 0) return error.BadGeometry; // h3_quantize's quantize_chunks, the batch slice
        const w: Weight = .{ .q = try cuda.DeviceBuffer.alloc(&self.driver, n * k + n * groups * 4), .n = n, .k = k, .groups = groups };
        try launchQuantize(self, bf16.ptr, w.q.ptr, w.scales(), n, k, groups);
        try self.stream.synchronize();
        bf16.free();
        bf16.* = .{ .d = &self.driver, .ptr = 0, .len = 0 };
        return w;
    }

    pub fn deinit(t: *Int8, self: *Model) void {
        var it = t.plans.valueIterator();
        while (it.next()) |g| g.deinit();
        t.plans.deinit(self.gpa);
        for (t.blocks) |*b| for ([_]*Weight{ &b.qkvc, &b.out, &b.fc1, &b.fc2 }) |w| w.q.free();
        self.gpa.free(t.blocks);
        t.scratch.free();
        _ = self.gemms.lt.api.cublasLtDestroy(t.handle);
        self.gpa.destroy(t);
    }

    /// y (rows, w.n) bf16 = x (rows, w.k) bf16 . w^T through the int8 product.
    pub fn run(t: *Int8, self: *Model, x: u64, w: Weight, y: u64, rows: usize) !void {
        const p = try t.product(self, x, w, rows);
        var args: cuda.Args = .{};
        for ([_]u64{ p.acc, p.xs, w.scales(), y }) |a| args.add(a);
        args.add(@as(c_int, @intCast(rows)));
        args.add(@as(c_int, @intCast(w.n)));
        if (w.groups == 1) return self.launch("h3_dequantize", .{ .x = 1024 }, 256, 0, &args);
        args.add(@as(c_int, @intCast(w.groups)));
        try self.launch("h3_dequantize_grouped", .{ .x = 1024 }, 256, 0, &args);
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

    /// Room and plans for `rows` before a step is captured: `run` and `mlpIn` then only launch.
    pub fn warm(t: *Int8, self: *Model, rows: usize) !void {
        try t.reserve(self, rows);
        _ = try t.plan(self, rows, 3 * self.inner, self.hidden, 1);
        _ = try t.plan(self, rows, self.hidden, self.inner, self.inner / group_k);
        _ = try t.plan(self, rows, 2 * self.mlp, self.hidden, 1);
        _ = try t.plan(self, rows, self.hidden, self.mlp, self.mlp / group_k);
    }

    /// x's rows quantized and their int32 sums against w in scratch: `acc` (`w.groups` planes of rows x w.n), `xs`.
    fn product(t: *Int8, self: *Model, x: u64, w: Weight, rows: usize) !struct { acc: u64, xs: u64 } {
        try t.reserve(self, rows);
        const acc = t.scratch.ptr;
        const xq = acc + rows * t.accWide(self) * 4;
        const xs = xq + rows * t.widest(self);
        try launchQuantize(self, x, xq, xs, rows, w.k, w.groups);
        const p = try t.plan(self, rows, w.n, w.k, w.groups);
        try p.runInt(w.q.ptr, xq, acc, self.gemms.workspace.ptr, workspace_bytes, self.stream.handle);
        return .{ .acc = acc, .xs = xs };
    }

    /// A product's widest int32 count a row: the biggest `n` x `groups` over the four projections — fc2's
    /// `hidden` x `mlp` / 256 — not the product of the widest n and the most groups (no projection has both).
    fn accWide(_: *const Int8, self: *const Model) usize {
        return @max(3 * self.inner, 2 * self.mlp, self.hidden * (self.inner / group_k), self.hidden * (self.mlp / group_k));
    }

    fn widest(_: *const Int8, self: *const Model) usize {
        return @max(3 * self.inner, 2 * self.mlp, self.hidden);
    }

    fn reserve(t: *Int8, self: *Model, rows: usize) !void {
        if (rows <= t.rows) return;
        try self.stream.synchronize();
        t.scratch.free();
        const wide = t.widest(self);
        t.scratch = try cuda.DeviceBuffer.alloc(&self.driver, rows * (t.accWide(self) * 4 + wide + (self.mlp / group_k) * 4));
        t.rows = rows;
    }

    /// Column-major D^T (n, rows) = W (k, n stored)^T . X^T (k, rows), int32 sums: cuBLASLt's int8 TN layout. A
    /// grouped weight runs `groups` batches of the k / groups-deep slice, each writing its own (n, rows) plane.
    fn plan(t: *Int8, self: *Model, rows: usize, n: usize, k: usize, groups: usize) !*const lt_.Gemm {
        const got = try t.plans.getOrPut(self.gpa, .{ rows, n, k, groups });
        if (got.found_existing) return got.value_ptr;
        errdefer t.plans.removeByPtr(got.key_ptr);
        const batch: u64 = if (groups > 1) k / groups else k;
        const spec: lt_.Spec = .{ .ta = .t, .m = n, .n = rows, .k = batch, .a = .i8, .b = .i8, .c = .i32, .lda = k, .ldb = k, .ldc = n, .batch = @intCast(groups), .sa = batch, .sb = batch, .sc = n * rows };
        got.value_ptr.* = try lt_.Gemm.init(&self.gemms.lt, t.handle, spec, workspace_bytes);
        return got.value_ptr;
    }
};

fn launchQuantize(self: *Model, x: u64, q: u64, s: u64, rows: usize, k: usize, groups: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ x, q, s }) |a| args.add(a);
    args.add(@as(c_int, @intCast(k)));
    args.add(@as(c_int, @intCast(groups)));
    try self.launch("h3_quantize", .{ .x = @intCast(rows) }, 256, 0, &args);
}
