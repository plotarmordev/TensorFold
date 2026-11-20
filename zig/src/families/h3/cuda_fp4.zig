//! mmh3's NVFP4 linear layers as an opt-in (`--fp4`): every projection's weight rows and each activation row
//! Hadamard-rotated in 128-wide chunks as `cuda_int8.zig` rotates them, then e2m1 at one e4m3 scale every 16
//! values of k (`h3_quantize_fp4`), and cuBLASLt's block-scaled FP4 product writes the bf16 rows from its own
//! tile: no int32 round trip, no `h3_dequantize`. The bf16 weights are freed once quantized. The finer scales do
//! not rescue e2m1's one mantissa bit: the picture this draws is a different picture (the recipe's numbers), so
//! it never runs unless asked.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;
const h3 = @import("cuda.zig");
const workspace_bytes = @import("cuda_gemm.zig").workspace_bytes;
const Model = h3.Model;

/// Bytes of the e4m3 scales of `rows` rows of k values in cuBLASLt's 128 x 4 tiles, padding included.
pub fn scaleBytes(rows: usize, k: usize) usize {
    return std.mem.alignForward(usize, rows, 128) * std.mem.alignForward(usize, k / 16, 4);
}

/// A projection (N, K): its e2m1 codes, then N rows of tiled scales.
pub const Weight = struct {
    q: cuda.DeviceBuffer,
    n: usize,
    k: usize,

    fn scales(w: Weight) u64 {
        return w.q.ptr + w.n * w.k / 2;
    }
};

/// One block's quantized projections.
pub const Block = struct { qkvc: Weight, out: Weight, fc1: Weight, fc2: Weight };

/// The buffers a step's quantized rows use: `norm` fills q/s before the product reads them.
pub const Scratch = struct { q: u64, s: u64 };

pub const Fp4 = struct {
    blocks: []Block,
    handle: lt_.Handle,
    plans: std.AutoHashMapUnmanaged([3]u64, lt_.Gemm) = .empty,
    /// e2m1 codes (rows, widest k / 2), then their tiled scales; products run one at a time, so each
    /// quantize overwrites the last product's rows once that product has read them.
    scratch: cuda.DeviceBuffer,
    rows: usize = 0,

    /// Replaces every block's bf16 projections with block-scaled FP4 ones, freeing each bf16 buffer once quantized.
    pub fn quantize(self: *Model) !void {
        if (self.gated) return error.H3NotPictureModel; // the dense forward's three of q, k and v only
        if (self.int8 != null) return error.Quantized;
        try self.ctx.makeCurrent();
        var handle: lt_.Handle = null;
        try self.gemms.lt.check(self.gemms.lt.api.cublasLtCreate(&handle), "cublasLtCreate");
        const t = try self.gpa.create(Fp4);
        t.* = .{ .blocks = &.{}, .handle = handle, .scratch = .{ .d = &self.driver, .ptr = 0, .len = 0 } };
        self.fp4 = t; // the model's deinit frees what a failed quantize already made
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

    /// A bf16 (n, k) projection to e2m1 codes and tiled scales, the bf16 buffer freed.
    fn weight(self: *Model, bf16: *cuda.DeviceBuffer, n: usize, k: usize) !Weight {
        if (k % 128 != 0) return error.BadGeometry; // the rotation's chunks; every k is a multiple of 32 otherwise
        const w: Weight = .{ .q = try cuda.DeviceBuffer.alloc(&self.driver, n * k / 2 + scaleBytes(n, k)), .n = n, .k = k };
        var args = quantizeArgs(bf16.ptr, w.q.ptr, w.scales(), k);
        try self.launch("h3_quantize_fp4", .{ .x = @intCast(n) }, 256, 0, &args);
        bf16.free();
        bf16.* = .{ .d = &self.driver, .ptr = 0, .len = 0 };
        return w;
    }

    pub fn deinit(t: *Fp4, self: *Model) void {
        var it = t.plans.valueIterator();
        while (it.next()) |g| g.deinit();
        t.plans.deinit(self.gpa);
        for (t.blocks) |*b| for ([_]*Weight{ &b.qkvc, &b.out, &b.fc1, &b.fc2 }) |w| w.q.free();
        self.gpa.free(t.blocks);
        t.scratch.free();
        _ = self.gemms.lt.api.cublasLtDestroy(t.handle);
        self.gpa.destroy(t);
    }

    /// The row buffers a step shares, as Scratch.
    pub fn layout(t: *Fp4, self: *Model, rows: usize) !Scratch {
        try t.reserve(self, rows);
        const q = t.scratch.ptr;
        return .{ .q = q, .s = q + std.mem.alignForward(usize, rows * t.widest(self) / 2, 16) };
    }

    /// The product over rows `norm` already quantized into `lay`: the bf16 rows land at `y`.
    pub fn gemm(t: *Fp4, self: *Model, lay: Scratch, w: Weight, y: u64, rows: usize) !void {
        const p = try t.plan(self, rows, w.n, w.k, w.scales(), lay.s);
        try p.runNvfp4(w.q.ptr, w.scales(), lay.q, lay.s, y, self.gemms.workspace.ptr, workspace_bytes, self.stream.handle);
    }

    /// y (rows, w.n) bf16 = x (rows, w.k) bf16 . w^T: x's rows quantized into the scratch, then the product.
    pub fn linear(t: *Fp4, self: *Model, x: u64, w: Weight, y: u64, rows: usize) !void {
        const lay = try t.layout(self, rows);
        var args = quantizeArgs(x, lay.q, lay.s, w.k);
        try self.launch("h3_quantize_fp4", .{ .x = @intCast(rows) }, 256, 0, &args);
        try t.gemm(self, lay, w, y, rows);
    }

    /// Room and plans for `rows` before a step is captured: `layout` then only hands out buffers. The
    /// plans' scale pointers are placeholders — any buffers of the right layout, the first block's —
    /// for the heuristic; each run points itself at the live ones.
    pub fn warm(t: *Fp4, self: *Model, rows: usize) !void {
        try t.reserve(self, rows);
        const lay = try t.layout(self, rows);
        const w = t.blocks[0].qkvc.scales();
        _ = try t.plan(self, rows, 3 * self.inner, self.hidden, w, lay.s);
        _ = try t.plan(self, rows, self.hidden, self.inner, t.blocks[0].out.scales(), lay.s);
        _ = try t.plan(self, rows, 2 * self.mlp, self.hidden, t.blocks[0].fc1.scales(), lay.s);
        _ = try t.plan(self, rows, self.hidden, self.mlp, t.blocks[0].fc2.scales(), lay.s);
    }

    fn widest(_: *const Fp4, self: *const Model) usize {
        return @max(self.hidden, self.inner, self.mlp);
    }

    fn reserve(t: *Fp4, self: *Model, rows: usize) !void {
        if (rows <= t.rows) return;
        try self.stream.synchronize();
        t.scratch.free();
        t.scratch = try cuda.DeviceBuffer.alloc(&self.driver, std.mem.alignForward(usize, rows * t.widest(self) / 2, 16) + scaleBytes(rows, t.widest(self)));
        try t.scratch.fill8(0, null); // the padding rows of the scale tiles never change
        t.rows = rows;
    }

    /// Column-major D^T (n, rows) = W (k, n stored)^T . X^T (k, rows) in bf16: cuBLASLt's FP4 TN layout. The
    /// scale pointers in the plan are placeholders for the heuristic; each run sets the live buffers'.
    fn plan(t: *Fp4, self: *Model, rows: usize, n: usize, k: usize, scale_a: u64, scale_b: u64) !*const lt_.Gemm {
        const got = try t.plans.getOrPut(self.gpa, .{ rows, n, k });
        if (got.found_existing) return got.value_ptr;
        errdefer t.plans.removeByPtr(got.key_ptr);
        got.value_ptr.* = try lt_.Gemm.init(&self.gemms.lt, t.handle, .{ .ta = .t, .m = n, .n = rows, .k = k, .a = .f4, .b = .f4, .c = .bf16, .lda = k, .ldb = k, .ldc = n, .scale_a = scale_a, .scale_b = scale_b }, workspace_bytes);
        return got.value_ptr;
    }
};

fn quantizeArgs(x: u64, q: u64, s: u64, k: usize) cuda.Args {
    var args: cuda.Args = .{};
    for ([_]u64{ x, q, s }) |a| args.add(a);
    args.add(@as(c_int, @intCast(k)));
    return args;
}
