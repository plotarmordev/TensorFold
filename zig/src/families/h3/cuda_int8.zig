//! mmh3's INT8 linear layers on the H3 CUDA blocks, on its own TMA product (h3_int8_tma.cuh): each
//! projection's weight rows and each activation row Hadamard-rotated in 128-wide chunks (ConvRot's
//! outlier spreading), quantized to int8, summed exactly in int32 by mma.sync m16n8k32.s8, and
//! scaled in registers to bf16 rows from the tile — no int32 round trip. The bf16 weights are freed
//! once quantized.
//!
//! `qkvc` and `fc1` scale each row by its absmax: the same products and roundings h3_dequantize and
//! h3_swiglu_int8 computed, so their latents stay bit-identical to the cuBLASLt path they replaced.
//! `out` and `fc2` failed the episode-1 picture gates at a row's scale (card 129: 28.15 dB, mean 4.23
//! against 28.2 dB and 4.1). Their opt-in path (`--int8-blocks`) scales each 128-wide chunk of both
//! operands by its own absmax and folds every 128-deep K block's sums into fp32 at that block's
//! scales; it is 0.77 s faster warm at 2048², but its fox still misses the gates (card 176: 26.26 dB,
//! mean 5.17), so the default keeps them bf16.
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
/// and the barriers; SquareTile's for the per-block product.
pub const gemm_shared = 2 * (256 + 128) * 128 + 1024 + 2 * 16;
pub const blocks_shared = 2 * (128 + 128) * 128 + 1024 + 2 * 16;
const block_n = 128;

/// How a projection's int8 rows scale: one scale a row, fc1's rows interleaved for the swiglu
/// epilogue, or a scale per 128-wide chunk of K (h3_quantize_blocks' block-major layout).
pub const Epilogue = enum { rows, swiglu, blocks };

/// A projection (N, K): its int8 rows, then its fp32 scales.
pub const Weight = struct {
    q: cuda.DeviceBuffer,
    n: usize,
    k: usize,
    epilogue: Epilogue = .rows,

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
/// swiglu epilogue (h3_swiglu_rows). `out` and `fc2` are int8 only under `quantize(.., true)`.
pub const Block = struct { qkvc: Weight, out: ?Weight = null, fc1: Weight, fc2: ?Weight = null };

/// The buffers a product's input rows use: `norm` or `linear` writes q/s before `gemm` reads them.
pub const Scratch = struct { xq: u64, xs: u64 };

pub const Int8 = struct {
    blocks: []Block,
    enc: Encode,
    /// int8 rows (rows, widest input), then their fp32 scales: a row's, or a row's per 128-wide chunk.
    scratch: cuda.DeviceBuffer,
    rows: usize = 0,

    /// Replaces every block's bf16 `qkvc` and `fc1` with int8 ones, and `out` and `fc2` too when
    /// `blocks` (a scale per 128-wide chunk; opt-in, as the fox misses episode 1's gates at it),
    /// freeing each bf16 buffer once it is quantized.
    pub fn quantize(self: *Model, blocks: bool) !void {
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
            q.qkvc = try weight(self, &b.qkvc, 3 * self.inner, self.hidden, .rows);
            q.fc1 = try weight(self, &b.fc1, 2 * self.mlp, self.hidden, .swiglu);
            if (!blocks) continue;
            q.out = try weight(self, &b.out, self.hidden, self.inner, .blocks);
            q.fc2 = try weight(self, &b.fc2, self.hidden, self.mlp, .blocks);
        }
    }

    /// The bf16 (n, k) rows to int8 at `epilogue`'s scales; `.swiglu` interleaves the result
    /// pairwise (gate rows' first half against the up rows' second) for the fc1 epilogue.
    fn weight(self: *Model, bf16: *cuda.DeviceBuffer, n: usize, k: usize, epilogue: Epilogue) !Weight {
        try checkGeometry(n, k);
        const scale_count = if (epilogue == .blocks) n * (k / 128) else n;
        var w: Weight = .{ .q = try cuda.DeviceBuffer.alloc(&self.driver, n * k + scale_count * 4), .n = n, .k = k, .epilogue = epilogue };
        errdefer release(self, &w.q);
        var tmp: cuda.DeviceBuffer = .{ .d = &self.driver, .ptr = 0, .len = 0 };
        defer release(self, &tmp);
        switch (epilogue) {
            .rows => try launchQuantize(self, "h3_quantize", bf16.ptr, w.q.ptr, w.scales(), n, k),
            .blocks => try launchQuantize(self, "h3_quantize_blocks", bf16.ptr, w.q.ptr, w.scales(), n, k),
            .swiglu => {
                tmp = try cuda.DeviceBuffer.alloc(&self.driver, w.q.len);
                try launchQuantize(self, "h3_quantize", bf16.ptr, tmp.ptr, tmp.ptr + n * k, n, k);
                var args: cuda.Args = .{};
                for ([_]u64{ tmp.ptr, tmp.ptr + n * k, w.q.ptr, w.scales() }) |a| args.add(a);
                args.add(@as(c_int, @intCast(k)));
                try self.launch("h3_swiglu_rows", .{ .x = @intCast(n / 2) }, 256, 0, &args);
            },
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
        for (t.blocks) |*b| {
            b.qkvc.q.free();
            b.fc1.q.free();
            if (b.out) |*w| w.q.free();
            if (b.fc2) |*w| w.q.free();
        }
        self.gpa.free(t.blocks);
        t.scratch.free();
        self.gpa.destroy(t);
    }

    /// The row buffers the launches share, as Scratch: the widest input's int8 rows, then a scale
    /// for each of its 128-wide chunks (a row's scale fits in its first).
    pub fn layout(t: *Int8, self: *Model, rows: usize) !Scratch {
        try t.reserve(self, rows);
        const xq = t.scratch.ptr;
        return .{ .xq = xq, .xs = xq + rows * widest(self) };
    }

    /// x's rows (already quantized at lay.xq) against w: bf16 rows at `out`, or silu(gate) * up at
    /// (rows, w.n / 2) when w's epilogue is `.swiglu`.
    pub fn gemm(t: *Int8, self: *Model, lay: Scratch, w: Weight, out: u64, rows: usize) !void {
        const block_m: usize = if (w.epilogue == .blocks) 128 else 256;
        var maps: [2]TensorMap = undefined;
        try encode(t.enc, &maps[0], lay.xq, rows, w.k, @intCast(block_m));
        try encode(t.enc, &maps[1], w.q.ptr, w.n, w.k, block_n);
        var args: cuda.Args = .{};
        args.add(maps[0]);
        args.add(maps[1]);
        for ([_]u64{ lay.xs, w.scales(), out }) |a| args.add(a);
        for ([_]usize{ rows, w.n, w.k }) |v| args.add(@as(c_int, @intCast(v)));
        const sm: usize = @intCast(try self.ctx.attribute(.multiprocessor_count));
        const tiles = (rows + block_m - 1) / block_m * (w.n / block_n);
        const blocks: u32 = @intCast(@min(sm, tiles));
        const name, const shared: u32 = switch (w.epilogue) {
            .rows => .{ Model.functionIndex("h3_gemm_int8_tma"), gemm_shared },
            .swiglu => .{ Model.functionIndex("h3_swiglu_int8_tma"), gemm_shared },
            .blocks => .{ Model.functionIndex("h3_gemm_int8_tma_blocks"), blocks_shared },
        };
        try self.launchShared(name, blocks, 256, shared, &args);
    }

    /// y (rows, w.n) bf16 = x (rows, w.k) bf16 . w^T for a `.blocks` weight: x's rows quantized a
    /// scale per 128-wide chunk into the scratch, then the product.
    pub fn linear(t: *Int8, self: *Model, x: u64, w: Weight, y: u64, rows: usize) !void {
        const lay = try t.layout(self, rows);
        try launchQuantize(self, "h3_quantize_blocks", x, lay.xq, lay.xs, rows, w.k);
        try t.gemm(self, lay, w, y, rows);
    }

    /// Room for `rows` before a step is captured: `layout` then only hands out buffers.
    pub fn warm(t: *Int8, self: *Model, rows: usize) !void {
        try t.reserve(self, rows);
    }

    fn reserve(t: *Int8, self: *Model, rows: usize) !void {
        if (rows <= t.rows) return;
        try self.stream.synchronize();
        t.scratch.free();
        t.scratch = try cuda.DeviceBuffer.alloc(&self.driver, rows * (widest(self) + widest(self) / 128 * 4));
        t.rows = rows;
    }
};

/// Every projection's input width: h3_norm_quant's `hidden`, `out`'s `inner`, `fc2`'s `mlp`.
fn widest(self: *const Model) usize {
    return @max(self.hidden, self.inner, self.mlp);
}

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

/// `kernel` (h3_quantize or h3_quantize_blocks) over the rows of x (rows, k) into q and s.
fn launchQuantize(self: *Model, comptime kernel: []const u8, x: u64, q: u64, s: u64, rows: usize, k: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ x, q, s }) |a| args.add(a);
    args.add(@as(c_int, @intCast(k)));
    try self.launch(kernel, .{ .x = @intCast(rows) }, 256, 0, &args);
}
