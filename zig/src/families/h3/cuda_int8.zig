//! mmh3's INT8 linear layers on the H3 CUDA blocks, on its own TMA product (h3_int8_tma.cuh): each
//! projection's weight rows and each activation row Hadamard-rotated in 128-wide chunks (ConvRot's
//! outlier spreading), quantized to int8 at its absmax, summed exactly in int32 by mma.sync
//! m16n8k32.s8, and scaled in registers to bf16 rows from the tile — no int32 round trip, and the
//! same products and roundings h3_dequantize and h3_swiglu_int8 computed, so the path's latents stay
//! bit-identical to the cuBLASLt + dequantize path it replaces. The bf16 weights are freed once
//! quantized.
//!
//! Only `qkvc` and `fc1` run it: `out` and `fc2` keep their bf16 weights. Their int8 products
//! measured faster (286 TFLOP a picture saved 0.6 s warm at 2048²) but fail the episode-1 picture
//! gates (28.15 dB, mean 4.23 against 28.2 dB and 4.1) at per-row scales; the finer scales need
//! per-256-of-K groups this product does not fold (card 139).
const std = @import("std");
const cuda = @import("cuda");
const report = @import("report.zig");
const h3 = @import("cuda.zig");
const Model = h3.Model;

/// The driver encodes a [rows, columns] matrix with 128-byte rows into a CUtensorMap whose TMA box
/// is `box_rows` by 128 bytes: one K block of one operand tile.
const TensorMap = extern struct { bytes: [128]u8 align(64) };
const Encode = *const fn (*TensorMap, c_uint, c_uint, ?*anyopaque, *const [2]u64, *const u64, *const [2]u32, *const [2]u32, c_uint, c_uint, c_uint, c_uint) callconv(.c) cuda.abi.Result;

fn encoder(d: *cuda.Driver) !Encode {
    return d.lib.lookup(Encode, "cuTensorMapEncodeTiled") orelse error.MissingSymbol;
}

/// TallTile's Config::shared_bytes: 2 stages of (256 + 128) * 128 bytes, the 1 KiB alignment slack
/// and the barriers.
pub const gemm_shared = 2 * (256 + 128) * 128 + 1024 + 2 * 16;
const block_m = 256;
const block_n = 128;

/// A projection (N, K): its int8 rows, then N fp32 row scales; `swiglu` marks fc1's
/// pairwise-interleaved rows.
pub const Weight = struct {
    q: cuda.DeviceBuffer,
    n: usize,
    k: usize,
    swiglu: bool = false,

    pub fn scales(w: Weight) u64 {
        return w.q.ptr + w.n * w.k;
    }
};

/// h3_quantize's K (quantize_chunks) and the product's whole N tiles: the grid has no edge tile, so
/// an N off 128 would leave its last columns unwritten.
pub fn checkGeometry(n: usize, k: usize) !void {
    if (k % 128 != 0 or k > 14336 or n % block_n != 0) return error.BadGeometry;
}

/// One block's quantized projections; fc1's rows and scales interleave gate/up pairs for the
/// swiglu epilogue (h3_swiglu_rows).
pub const Block = struct { qkvc: Weight, fc1: Weight };

/// The buffers a product's input rows use: `norm` writes q/s before `gemm` reads them.
pub const Scratch = struct { xq: u64, xs: u64 };

pub const Int8 = struct {
    blocks: []Block,
    enc: Encode,
    /// int8 rows (rows, hidden), then the rows' fp32 scales; no int32 sums any more.
    scratch: cuda.DeviceBuffer,
    rows: usize = 0,

    /// Replaces every block's bf16 projections with int8 ones, freeing each bf16 buffer once it is
    /// quantized.
    pub fn quantize(self: *Model) !void {
        if (self.gated) return error.H3NotPictureModel; // the dense forward's three of q, k, v only
        if (self.fp4 != null) return error.Quantized;
        try self.ctx.makeCurrent();
        // the TMA images are sm_90+; older targets carry a trap stub in their place
        if (try self.ctx.capability() < 90) return error.H3Int8RequiresSm90;
        const enc = try encoder(&self.driver); // refuse early on a driver without TMA
        const t = try self.gpa.create(Int8);
        t.* = .{ .blocks = &.{}, .enc = enc, .scratch = .{ .d = &self.driver, .ptr = 0, .len = 0 } };
        self.int8 = t; // the model's deinit frees what a failed quantize already made
        const none: Weight = .{ .q = .{ .d = &self.driver, .ptr = 0, .len = 0 }, .n = 0, .k = 0 };
        t.blocks = try self.gpa.alloc(Block, self.blocks.len);
        @memset(t.blocks, .{ .qkvc = none, .fc1 = none });
        for (self.blocks, t.blocks) |*b, *q| {
            q.qkvc = try weight(self, &b.qkvc, 3 * self.inner, self.hidden, false);
            q.fc1 = try weight(self, &b.fc1, 2 * self.mlp, self.hidden, true);
        }
    }

    /// The bf16 (n, k) rows to int8 at h3_quantize's per-row scales; `swiglu` interleaves the
    /// result pairwise (gate rows' first half against the up rows' second) for the fc1 epilogue.
    fn weight(self: *Model, bf16: *cuda.DeviceBuffer, n: usize, k: usize, swiglu: bool) !Weight {
        try checkGeometry(n, k);
        var w: Weight = .{ .q = try cuda.DeviceBuffer.alloc(&self.driver, n * k + n * 4), .n = n, .k = k, .swiglu = swiglu };
        errdefer release(self, &w.q);
        var tmp: cuda.DeviceBuffer = .{ .d = &self.driver, .ptr = 0, .len = 0 };
        defer release(self, &tmp);
        if (swiglu) {
            tmp = try cuda.DeviceBuffer.alloc(&self.driver, w.q.len);
            try launchQuantize(self, bf16.ptr, tmp.ptr, tmp.ptr + n * k, n, k);
            var args: cuda.Args = .{};
            for ([_]u64{ tmp.ptr, tmp.ptr + n * k, w.q.ptr, w.scales() }) |a| args.add(a);
            args.add(@as(c_int, @intCast(k)));
            try self.launch("h3_swiglu_rows", .{ .x = @intCast(n / 2) }, 256, 0, &args);
        } else {
            try launchQuantize(self, bf16.ptr, w.q.ptr, w.scales(), n, k);
        }
        try self.stream.synchronize();
        bf16.free();
        bf16.* = .{ .d = &self.driver, .ptr = 0, .len = 0 };
        return w;
    }

    /// Frees `b` once the queued launches that may still use it have finished; a failed drain
    /// retains it rather than free memory a kernel may still touch.
    fn release(self: *Model, b: *cuda.DeviceBuffer) void {
        if (b.ptr == 0) return;
        self.drain() catch |e| {
            report.say("H3 int8 weight buffer retained: {t}", .{e});
            return;
        };
        b.free();
    }

    pub fn deinit(t: *Int8, self: *Model) void {
        for (t.blocks) |*b| for ([_]*Weight{ &b.qkvc, &b.fc1 }) |w| w.q.free();
        self.gpa.free(t.blocks);
        t.scratch.free();
        self.gpa.destroy(t);
    }

    /// The row buffers the fused launches share, as Scratch.
    pub fn layout(t: *Int8, self: *Model, rows: usize) !Scratch {
        try t.reserve(self, rows);
        const xq = t.scratch.ptr;
        const xs = xq + rows * self.hidden;
        return .{ .xq = xq, .xs = xs };
    }

    /// x's rows (already quantized at lay.xq by h3_norm_quant) against w: bf16 rows at `out`, or
    /// silu(gate) * up at (rows, w.n / 2) when w.swiglu interleaved the weights.
    pub fn gemm(t: *Int8, self: *Model, lay: Scratch, w: Weight, out: u64, rows: usize) !void {
        var maps: [2]TensorMap = undefined;
        try encode(t.enc, &maps[0], lay.xq, rows, w.k, block_m);
        try encode(t.enc, &maps[1], w.q.ptr, w.n, w.k, block_n);
        var args: cuda.Args = .{};
        args.add(maps[0]);
        args.add(maps[1]);
        for ([_]u64{ lay.xs, w.scales(), out }) |a| args.add(a);
        for ([_]usize{ rows, w.n, w.k }) |v| args.add(@as(c_int, @intCast(v)));
        const sm: usize = @intCast(try self.ctx.attribute(.multiprocessor_count));
        const tiles = (rows + block_m - 1) / block_m * (w.n / block_n);
        const blocks: u32 = @intCast(@min(sm, tiles));
        const name = if (w.swiglu) Model.functionIndex("h3_swiglu_int8_tma") else Model.functionIndex("h3_gemm_int8_tma");
        try self.launchShared(name, blocks, 256, gemm_shared, &args);
    }

    /// Room for `rows` before a step is captured: `layout` then only hands out buffers.
    pub fn warm(t: *Int8, self: *Model, rows: usize) !void {
        try t.reserve(self, rows);
    }

    fn reserve(t: *Int8, self: *Model, rows: usize) !void {
        if (rows <= t.rows) return;
        try self.stream.synchronize();
        t.scratch.free();
        t.scratch = try cuda.DeviceBuffer.alloc(&self.driver, rows * (self.hidden + 4));
        t.rows = rows;
    }
};

/// TMA's UINT8 map of a [rows, k] int8 matrix: boxes of `box_rows` by 128 bytes, the K block, L2
/// 256B promotion and zero fill past the end.
fn encode(enc: Encode, map: *TensorMap, ptr: u64, rows: usize, k: usize, box_rows: u32) !void {
    const dims: [2]u64 = .{ k, rows };
    const strides: u64 = k; // bytes between rows; dimension 0's stride is the contiguous row
    const box: [2]u32 = .{ 128, box_rows };
    const element_strides: [2]u32 = .{ 1, 1 };
    // UINT8 = 0, INTERLEAVE_NONE = 0, SWIZZLE_128B = 3, L2_PROMOTION_256B = 3, OOB_FILL_NONE = 0
    const res = enc(map, 0, 2, @ptrFromInt(@as(usize, @intCast(ptr))), &dims, &strides, &box, &element_strides, 0, 3, 3, 0);
    if (res != cuda.abi.success) return error.CudaFailed;
}

fn launchQuantize(self: *Model, x: u64, q: u64, s: u64, rows: usize, k: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ x, q, s }) |a| args.add(a);
    args.add(@as(c_int, @intCast(k)));
    try self.launch("h3_quantize", .{ .x = @intCast(rows) }, 256, 0, &args);
}
