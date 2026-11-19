//! Block-scaled FP8 linear layers on the H3 CUDA blocks (GB10's MXFP8): each projection's weight rows and each
//! activation row Hadamard-rotated in 128-wide chunks as `cuda_int8.zig` rotates them, then e4m3 at one UE8M0 scale
//! every 32 values of k, and cuBLASLt's block-scaled product writes the bf16 rows from its own tile: no int32 round
//! trip. With a scale every 32 values instead of one a row, all four projections (`qkvc`, `out`, `fc1`, `fc2`) are
//! quantized. The bf16 weights are freed once quantized.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;
const h3 = @import("cuda.zig");
const workspace_bytes = @import("cuda_gemm.zig").workspace_bytes;
const Model = h3.Model;

/// Bytes of the UE8M0 scales of `rows` rows of k values in cuBLASLt's 128 x 4 tiles, padding included.
pub fn scaleBytes(rows: usize, k: usize) usize {
    return std.mem.alignForward(usize, rows, 128) * std.mem.alignForward(usize, k / 32, 4);
}

/// A projection (N, K): its e4m3 rows, then their tiled scales.
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

pub const Fp8 = struct {
    blocks: []Block,
    handle: lt_.Handle,
    plans: std.AutoHashMapUnmanaged([3]u64, lt_.Gemm) = .empty,
    /// e4m3 rows (rows, widest k), then their tiled scales.
    scratch: cuda.DeviceBuffer,
    rows: usize = 0,

    /// Replaces every block's bf16 projections with block-scaled FP8 ones, freeing each bf16 buffer once quantized.
    pub fn quantize(self: *Model) !void {
        if (self.gated) return error.H3NotPictureModel; // the dense forward's three of q, k, v only
        const t = try init(self);
        t.blocks = try self.gpa.alloc(Block, self.blocks.len);
        const none: Weight = .{ .q = .{ .d = &self.driver, .ptr = 0, .len = 0 }, .n = 0, .k = 0 };
        @memset(t.blocks, .{ .qkvc = none, .out = none, .fc1 = none, .fc2 = none });
        for (self.blocks, t.blocks) |*b, *q| {
            q.qkvc = try t.weight(self, &b.qkvc, 3 * self.inner, self.hidden);
            q.out = try t.weight(self, &b.out, self.hidden, self.inner);
            q.fc1 = try t.weight(self, &b.fc1, 2 * self.mlp, self.hidden);
            q.fc2 = try t.weight(self, &b.fc2, self.hidden, self.mlp);
        }
    }

    /// The handle and an empty set of blocks on `self`, whose deinit frees what a failed quantize already made.
    pub fn init(self: *Model) !*Fp8 {
        try self.ctx.makeCurrent();
        var handle: lt_.Handle = null;
        try self.gemms.lt.check(self.gemms.lt.api.cublasLtCreate(&handle), "cublasLtCreate");
        const t = try self.gpa.create(Fp8);
        t.* = .{ .blocks = &.{}, .handle = handle, .scratch = .{ .d = &self.driver, .ptr = 0, .len = 0 } };
        self.fp8 = t;
        return t;
    }

    /// A bf16 (n, k) projection to e4m3 and its scales, the bf16 buffer freed.
    pub fn weight(_: *Fp8, self: *Model, bf16: *cuda.DeviceBuffer, n: usize, k: usize) !Weight {
        if (k % 128 != 0 or n % 16 != 0) return error.BadGeometry; // the rotation's chunks; 16-byte aligned scales
        const w: Weight = .{ .q = try cuda.DeviceBuffer.alloc(&self.driver, n * k + scaleBytes(n, k)), .n = n, .k = k };
        try quantizeRows(self, bf16.ptr, w.q.ptr, w.scales(), n, k);
        try self.stream.synchronize();
        bf16.free();
        bf16.* = .{ .d = &self.driver, .ptr = 0, .len = 0 };
        return w;
    }

    pub fn deinit(t: *Fp8, self: *Model) void {
        var it = t.plans.valueIterator();
        while (it.next()) |g| g.deinit();
        t.plans.deinit(self.gpa);
        for (t.blocks) |*b| for ([_]*Weight{ &b.qkvc, &b.out, &b.fc1, &b.fc2 }) |w| w.q.free();
        self.gpa.free(t.blocks);
        t.scratch.free();
        _ = self.gemms.lt.api.cublasLtDestroy(t.handle);
        self.gpa.destroy(t);
    }

    /// y (rows, w.n) bf16 = x (rows, w.k) bf16 . w^T: x's rows quantized into scratch, then the block-scaled product.
    pub fn run(t: *Fp8, self: *Model, x: u64, w: Weight, y: u64, rows: usize) !void {
        try t.reserve(self, rows);
        const xq = t.scratch.ptr;
        const xs = xq + std.mem.alignForward(usize, rows * t.widest(self), 16);
        try quantizeRows(self, x, xq, xs, rows, w.k);
        const p = try t.plan(self, rows, w, xs);
        try p.runScaled(w.q.ptr, w.scales(), xq, xs, y, self.gemms.workspace.ptr, workspace_bytes, self.stream.handle);
    }

    fn widest(_: *const Fp8, self: *const Model) usize {
        return @max(self.hidden, self.inner, self.mlp);
    }

    /// Room for `rows` rows of the widest k. The scales' padding rows past a product's own only reach output columns
    /// past its rows, which are never stored, so the scratch is cleared once and not between products.
    fn reserve(t: *Fp8, self: *Model, rows: usize) !void {
        if (rows <= t.rows) return;
        try self.stream.synchronize();
        t.scratch.free();
        const bytes = std.mem.alignForward(usize, rows * t.widest(self), 16) + scaleBytes(rows, t.widest(self));
        t.scratch = try cuda.DeviceBuffer.alloc(&self.driver, bytes);
        try t.scratch.fill8(0, null);
        t.rows = rows;
    }

    /// Column-major D^T (n, rows) = W (k, n stored)^T . X^T (k, rows) in bf16: cuBLASLt's FP8 TN layout, planned with
    /// this product's scales.
    fn plan(t: *Fp8, self: *Model, rows: usize, w: Weight, xs: u64) !*const lt_.Gemm {
        const got = try t.plans.getOrPut(self.gpa, .{ rows, w.n, w.k });
        if (got.found_existing) return got.value_ptr;
        errdefer t.plans.removeByPtr(got.key_ptr);
        got.value_ptr.* = try lt_.Gemm.init(&self.gemms.lt, t.handle, .{ .ta = .t, .m = w.n, .n = rows, .k = w.k, .a = .e4m3, .b = .e4m3, .c = .bf16, .lda = w.k, .ldb = w.k, .ldc = w.n, .scale_a = w.scales(), .scale_b = xs }, workspace_bytes);
        return got.value_ptr;
    }
};

fn quantizeRows(self: *Model, x: u64, q: u64, s: u64, rows: usize, k: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ x, q, s }) |a| args.add(a);
    args.add(@as(c_int, @intCast(k)));
    try self.launch("h3_quantize_fp8", .{ .x = @intCast(rows) }, 256, 0, &args);
}
