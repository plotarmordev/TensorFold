//! MiniMax-H3-Image's prompt stage on CUDA: Qwen3-VL's text model to layer 50, then condition_proj and
//! the refiner, all on the GPU at the Metal kernels' and the host reference's rounding. Text-encoder products sum K in reference
//! order; refiner products use cuBLASLt with FP32 sums. Every value the reference stores in bf16 is rounded to bf16
//! by a kernel before the next step reads it.
//! The text model's layer weights and the refiner's tensors live in one resident device arena, loaded
//! once by `loadResident`; a warm prompt moves no weights at all.
const std = @import("std");
pub const seconds = @import("clock.zig").seconds;
const cuda = @import("cuda");
pub const rules = @import("picture_prompt.zig");
pub const reference = @import("prompt_reference.zig");
pub const case = @import("prompt_case.zig");
pub const picture = @import("picture.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const cache = @import("prompt_cache.zig");
const report = @import("report.zig");
const bf16_round = @import("bf16_round");

const D = rules.head_dim;
const mode_qwen: c_int = 0;
const mode_h3: c_int = 1;

const image align(16) = @embedFile("fatbin_h3_image_prompt").*;
const kernel_names = [_][:0]const u8{ "h3p_norm", "h3p_heads", "h3p_attention", "h3p_glu", "h3p_add", "h3p_scatter_add", "h3p_embed", "h3p_splice", "h3p_linear", "h3p_round" };

/// A presentation's vision rows, as prompt.Vision: the tower's merged rows at its image pads in bf16,
/// and DeepStack's three blocks added after encoder layers 0 to 2.
pub const Vision = struct {
    pads: []const u32,
    rows: []const u16,
    deepstack: [3][]const u16,

    fn check(v: Vision, T: usize, H: usize) !usize {
        const n = v.pads.len;
        for (v.pads, 0..) |at, i| if (at >= T or (i > 0 and at <= v.pads[i - 1])) return error.H3BadVisionRows;
        if (v.rows.len != n * H) return error.H3BadVisionRows;
        for (v.deepstack) |d| if (d.len != n * H) return error.H3BadVisionRows;
        return n;
    }
};

/// A checked weight tensor's bf16 bytes, read in place from a mapped shard.
pub const Tensor = []const u8;
/// A text layer's tensors in rules.Layer field order.
pub const LayerSet = rules.Layer(Tensor);
/// A refiner block's tensors in rules.Block field order.
pub const BlockSet = rules.Block(Tensor);

/// The parts of the one device scratch a pass draws from, each part aligned to 256 bytes. Products
/// all write the shared fp32 `p` (they run one at a time on one stream), then `h3p_round` moves the
/// value to its bf16 buffer — out of place, so a round never writes what another thread still reads.
const EncodePart = enum { x, xn, p, qb, kb, vb, att, gb, ub, act, cos, sin, idx, vision, deep };
const RefinePart = enum { x, xn, p, qkv, att, wide, act };

fn Scratch(comptime Part: type) type {
    return struct {
        buf: cuda.DeviceBuffer,
        offs: [std.enums.values(Part).len]usize,

        fn make(d: *const cuda.Driver, sizes: std.enums.EnumArray(Part, usize)) !@This() {
            var s: @This() = .{ .buf = undefined, .offs = undefined };
            var total: usize = 0;
            for (std.enums.values(Part), 0..) |p, i| {
                s.offs[i] = total;
                total += std.mem.alignForward(usize, @max(sizes.get(p), 1), 256);
            }
            s.buf = try cuda.DeviceBuffer.alloc(d, total);
            return s;
        }

        fn at(s: @This(), p: Part) u64 {
            return s.buf.ptr + s.offs[@backingInt(p)];
        }
    };
}

/// The bf16 `bytes` as fp32 values, on the allocator's lifetime until the call returns.
fn asF32(a: std.mem.Allocator, bytes: []const u8) ![]f32 {
    if (bytes.len % 2 != 0) return error.H3UnexpectedTensor;
    const bits = std.mem.bytesAsSlice(u16, bytes);
    const f = try a.alloc(f32, bits.len);
    for (f, bits) |*o, b| o.* = bf16_round.value(b);
    return f;
}

/// Every weight the prompt stage reads, on the device in one arena: the embedding table, each text
/// layer's tensors, condition_proj's pair and the final norm, and each refiner block's tensors.
/// Vectors (norms, cond_b) sit as fp32 values; matrices as their stored bf16 bytes. `deinit` frees
/// the arena; the layers and blocks slices live on the allocator passed to `loadResident`.
pub const Resident = struct {
    arena: cuda.DeviceBuffer,
    embed: u64,
    layers: []rules.Layer(u64),
    cond_w: u64,
    cond_b: u64,
    final_norm: u64,
    blocks: []rules.Block(u64),

    pub fn deinit(res: *Resident) void {
        res.arena.free();
    }
};

/// One 256-byte-aligned slot per tensor: `n` bf16 elements on the device are `n * 2` bytes, `n`
/// fp32 (a converted vector) `n * 4`. The slot sizes come from the geometry alone, so a stored
/// tensor of another length is `H3UnexpectedTensor` before any copy.
fn slotOf(shape: rules.Shape) usize {
    const n = shape[0] * @max(shape[1], 1);
    return std.mem.alignForward(usize, n * if (shape[1] == 0) @as(usize, 4) else 2, 256);
}

/// `bytes` into the next slot of `at` on the runner's stream, vectors converted to fp32; the
/// stored length must equal the geometry's bf16 length.
fn fill(r: *Runner, a: std.mem.Allocator, at: *usize, arena: u64, shape: rules.Shape, bytes: Tensor) !u64 {
    const n = shape[0] * @max(shape[1], 1);
    if (bytes.len != n * 2) return error.H3UnexpectedTensor;
    const ptr = arena + at.*;
    if (shape[1] == 0) {
        const f = try asF32(a, bytes);
        defer a.free(f);
        try r.copy(std.mem.sliceAsBytes(f), ptr);
    } else try r.copy(bytes, ptr);
    at.* += slotOf(shape);
    return ptr;
}

/// Every tensor of the prompt stage onto the device once: `embed`, each layer's set from `layer`,
/// condition_proj's weight and bias, the final norm and each block's set from `block`, in one
/// arena whose slots the returned Resident names. The copies queue on the runner's stream ahead of
/// any encode or refine; the caller's mapped bytes may be dropped once the call returns.
pub fn loadResident(
    r: *Runner,
    a: std.mem.Allocator,
    g: rules.Text,
    front: rules.Refiner,
    embed: Tensor,
    layer: *const fn (ctx: *anyopaque, li: usize) anyerror!LayerSet,
    ctx: *anyopaque,
    block: *const fn (ctx: *anyopaque, bi: usize) anyerror!BlockSet,
    front_ctx: *anyopaque,
    cond_w: Tensor,
    cond_b: Tensor,
    final_norm: Tensor,
) !Resident {
    const layer_specs = rules.layerTensors(g);
    const front_specs = rules.frontTensors(front);
    const block_specs = rules.blockTensors(front);
    var total = slotOf(.{ g.vocab, g.hidden });
    for (layer_specs) |spec| total += slotOf(spec[2]) * g.layers;
    for (front_specs) |spec| total += slotOf(spec[2]);
    for (block_specs) |spec| total += slotOf(spec[2]) * front.blocks;
    var arena = try cuda.DeviceBuffer.alloc(&r.driver, total);
    errdefer {
        cuda.completion.drain(&r.driver, r.stream, "h3 prompt cuda: a resident copy did not drain");
        arena.free();
    }
    var res: Resident = .{
        .arena = arena,
        .embed = undefined,
        .layers = try a.alloc(rules.Layer(u64), g.layers),
        .cond_w = undefined,
        .cond_b = undefined,
        .final_norm = undefined,
        .blocks = try a.alloc(rules.Block(u64), front.blocks),
    };
    var at: usize = 0;
    res.embed = try fill(r, a, &at, res.arena.ptr, .{ g.vocab, g.hidden }, embed);
    for (0..g.layers) |li| {
        const set = try layer(ctx, li);
        inline for (comptime std.meta.fieldNames(LayerSet), layer_specs) |field, spec| {
            std.debug.assert(std.mem.eql(u8, field, spec[0]));
            @field(res.layers[li], field) = try fill(r, a, &at, res.arena.ptr, spec[2], @field(set, field));
        }
    }
    const front_tensors: [3]Tensor = .{ cond_w, cond_b, final_norm };
    inline for (.{ "cond_w", "cond_b", "final_norm" }, front_specs, front_tensors) |field, spec, t| {
        @field(res, field) = try fill(r, a, &at, res.arena.ptr, spec[2], t);
    }
    for (0..front.blocks) |bi| {
        const set = try block(front_ctx, bi);
        inline for (comptime std.meta.fieldNames(BlockSet), block_specs) |field, spec| {
            std.debug.assert(std.mem.eql(u8, field, spec[0]));
            @field(res.blocks[bi], field) = try fill(r, a, &at, res.arena.ptr, spec[2], @field(set, field));
        }
    }
    return res;
}

/// One product's plan: cuBLASLt bf16 in, fp32 sums, fp32 out. The bias variant's plan is separate so a
/// fused fp32 epilogue is picked only where a bias exists; bf16 bias values convert exactly.
const Products = struct {
    lt: cuda.cublaslt.Library,
    workspace: cuda.DeviceBuffer,
    gpa: std.mem.Allocator,
    /// (rows, n, k, bias) to its plan.
    plans: std.AutoHashMapUnmanaged([4]u64, cuda.cublaslt.Linear) = .empty,

    const workspace_bytes = 32 << 20;
    const canonical_rows = 16384;

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Products {
        var lt = try cuda.cublaslt.Library.open();
        errdefer lt.close();
        return .{ .lt = lt, .gpa = gpa, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    fn deinit(p: *Products) void {
        var it = p.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        p.plans.deinit(p.gpa);
        p.workspace.free();
        p.lt.close();
    }

    /// y (rows, n) fp32 = x (rows, k) bf16 . w (n, k) bf16^T; bias [n] fp32 fused when nonzero.
    fn run(p: *Products, stream: cuda.Stream, x: u64, w: u64, bias: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        const key = [4]u64{ rows, n, k, @intFromBool(bias != 0) };
        const got = try p.plans.getOrPut(p.gpa, key);
        if (!got.found_existing) {
            errdefer p.plans.removeByPtr(got.key_ptr);
            // Prefer the canonical algorithm, but short prompts must use an algorithm valid at their own shape.
            var probe = try cuda.cublaslt.Linear.initEpilogue(&p.lt, canonical_rows, n, k, .f32, workspace_bytes, if (bias != 0) .bias else .none);
            defer probe.deinit();
            var l = try cuda.cublaslt.Linear.initEpilogue(&p.lt, rows, n, k, .f32, workspace_bytes, if (bias != 0) .bias else .none);
            errdefer l.deinit();
            if (bias != 0) {
                const epilogue: u32 = @backingInt(cuda.cublaslt.Epilogue.bias);
                try p.lt.check(p.lt.api.cublasLtMatmulDescSetAttribute(l.desc, cuda.cublaslt.desc_epilogue, &epilogue, @sizeOf(u32)), "set H3 condition bias");
            }
            var checked: cuda.cublaslt.Heuristic = undefined;
            if (p.fits(&l, &probe.algo, &checked)) {
                l.algo = probe.algo;
            } else if (!p.fits(&l, &l.algo, &checked)) {
                // the plain product's algorithm cannot add the bias at this shape: ask for one that can
                if (bias != 0) try l.pick();
                if (bias == 0 or !p.fits(&l, &l.algo, &checked)) {
                    std.log.warn("h3 prompt cuda: no cuBLASLt algorithm for ({d}, {d}, {d}) bias {}", .{ rows, n, k, bias != 0 });
                    return error.NoAlgorithm;
                }
            }
            l.workspace_need = checked.workspace_size;
            if (bias != 0) l.epilogue = .bias;
            got.value_ptr.* = l;
        }
        if (bias != 0) return got.value_ptr.runBias(x, w, bias, y, p.workspace.ptr, workspace_bytes, stream.handle);
        try got.value_ptr.run(x, w, y, p.workspace.ptr, workspace_bytes, stream.handle);
    }

    /// Whether `algo` runs `l`'s descriptor (its epilogue included) within the workspace; `checked` gets its need.
    fn fits(p: *const Products, l: *const cuda.cublaslt.Linear, algo: *const cuda.cublaslt.Algo, checked: *cuda.cublaslt.Heuristic) bool {
        const s = p.lt.api.cublasLtMatmulAlgoCheck(l.handle, l.desc, l.a, l.b, l.c, l.c, algo, checked);
        return s == 0 and checked.state == 0 and checked.workspace_size <= workspace_bytes;
    }
};

pub const Runner = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    module: cuda.Module,
    functions: [kernel_names.len]cuda.Function,
    stream: cuda.Stream,
    products: Products,
    /// Two pinned windows: a chunk's copy runs while the next is being filled.
    stages: [2]cuda.copy_source.Stage = .{ .{}, .{} },
    /// GPU seconds of the last encode or refine.
    gpu: f64 = 0,

    fn function(r: *const Runner, comptime name: []const u8) cuda.Function {
        inline for (kernel_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return r.functions[i];
        @compileError("no kernel " ++ name);
    }

    /// A launch whose failure leaves the device drained before buffers owned by the caller are freed.
    fn launch(r: *const Runner, comptime name: []const u8, grid: cuda.Dim3, threads: u32, args: *cuda.Args) !void {
        errdefer cuda.completion.drain(&r.driver, r.stream, "h3 prompt cuda: a launch did not drain");
        try cuda.launch.launch(r.function(name), .{ .grid = grid, .block = .{ .x = threads } }, r.stream, args);
    }

    /// The driver, its primary context, the prompt kernels and the cuBLASLt plans; all freed by deinit.
    /// Heap-made: the stream, events and buffers each hold `&r.driver`, which a returned copy would move.
    pub fn init(gpa: std.mem.Allocator) !*Runner {
        const r = try gpa.create(Runner);
        errdefer gpa.destroy(r);
        r.gpa = gpa;
        r.stages = .{ .{}, .{} };
        r.gpu = 0;
        r.driver = try cuda.Driver.open();
        errdefer r.driver.close();
        r.ctx = try cuda.Context.shared(&r.driver);
        errdefer r.ctx.deinit();
        r.module = try cuda.Module.load(&r.driver, &image);
        errdefer r.module.unload();
        inline for (kernel_names, 0..) |name, i| r.functions[i] = try r.module.function(name);
        r.stream = try cuda.Stream.init(&r.driver, true);
        errdefer r.stream.deinit();
        r.products = try Products.init(gpa, &r.driver);
        return r;
    }

    pub fn deinit(r: *Runner) void {
        r.stream.synchronize() catch |e| report.say("h3 prompt cuda: work still queued at close: {t}", .{e});
        for (&r.stages) |*stage| stage.deinit();
        r.products.deinit();
        r.stream.deinit();
        r.module.unload();
        _ = r.driver.api.cuDevicePrimaryCtxRelease_v2(r.ctx.device);
        r.driver.close();
        r.gpa.destroy(r);
    }

    /// `bytes` through the two pinned stages onto `dst` in 64 MiB windows, each stage waited only when
    /// its turn comes again, so a copy runs while the other window is filled; the uploads order ahead
    /// of later work on the runner's stream.
    fn copy(r: *Runner, bytes: []const u8, dst: u64) !void {
        var done: usize = 0;
        var turn: usize = 0;
        while (done < bytes.len) {
            const n = @min(bytes.len - done, 64 << 20);
            const window = try r.stages[turn].take(&r.driver, n);
            @memcpy(window, bytes[done..][0..n]);
            try r.stages[turn].copyRangeTo(.{ .d = &r.driver, .ptr = dst + done, .len = n }, 0, 0, n, r.stream);
            done += n;
            turn ^= 1;
        }
    }

    fn linear(r: *const Runner, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, w, y }) |p| args.add(p);
        for ([_]usize{ rows, n, k }) |v| args.add(@as(c_int, @intCast(v)));
        try r.launch("h3p_linear", .{ .x = @intCast((n + 7) / 8) }, 256, &args);
    }

    fn norm(r: *const Runner, x: u64, w: u64, y: u64, rows: usize, width: usize, eps: f32, mode: c_int) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, w, y }) |p| args.add(p);
        for ([_]usize{ width, @intCast(mode) }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(eps);
        try r.launch("h3p_norm", .{ .x = @intCast(rows) }, 256, &args);
    }

    /// The text model's heads: q and k bf16 in place, normed and turned by the [rows, 64] rotary
    /// tables `rot` (cos, sin device pointers) when set.
    fn heads(r: *const Runner, q: u64, k: u64, wq: u64, wk: u64, rot: ?[2]u64, count: usize, kv_count: usize, q_row: usize, k_row: usize, head: usize, rows: usize, eps: f32, mode: c_int) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ q, k, wq, wk, if (rot) |t| t[0] else 0, if (rot) |t| t[1] else 0 }) |p| args.add(p);
        for ([_]usize{ count, kv_count, q_row, k_row, head, @intCast(mode), @intFromBool(rot != null) }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(eps);
        try r.launch("h3p_heads", .{ .x = @intCast(rows), .y = @intCast((count + kv_count + 7) / 8) }, 256, &args);
    }

    fn attention(r: *const Runner, q: u64, k: u64, v: u64, o: u64, rows: usize, count: usize, kv_count: usize, q_row: usize, k_row: usize, v_row: usize, head: usize, causal: bool) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ q, k, v, o }) |p| args.add(p);
        for ([_]usize{ rows, count, kv_count, q_row, k_row, v_row, head, @intFromBool(causal) }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(@as(f32, 1.0 / @sqrt(@as(f32, D))));
        try r.launch("h3p_attention", .{ .x = @intCast(rows), .y = @intCast((count + 7) / 8) }, 256, &args);
    }

    fn glu(r: *const Runner, g: u64, v: u64, o: u64, rows: usize, width: usize, g_row: usize, v_row: usize, mode: c_int) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ g, v, o }) |p| args.add(p);
        for ([_]usize{ width, g_row, v_row, @intCast(mode) }) |n| args.add(@as(c_int, @intCast(n)));
        try r.launch("h3p_glu", .{ .x = @intCast((width + 255) / 256), .y = @intCast(rows) }, 256, &args);
    }

    fn add(r: *const Runner, x: u64, y: u64, n: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, y }) |p| args.add(p);
        args.add(@as(c_int, @intCast(n)));
        try r.launch("h3p_add", .{ .x = @intCast((n + 255) / 256) }, 256, &args);
    }

    fn round(r: *const Runner, x: u64, y: u64, n: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, y }) |p| args.add(p);
        args.add(@as(c_int, @intCast(n)));
        try r.launch("h3p_round", .{ .x = @intCast((n + 255) / 256) }, 256, &args);
    }

    fn indexed(r: *const Runner, comptime name: []const u8, a_ptr: u64, b_ptr: u64, idx: u64, rows: usize, width: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ a_ptr, b_ptr, idx }) |p| args.add(p);
        args.add(@as(c_int, @intCast(width)));
        try r.launch(name, .{ .x = @intCast((width + 255) / 256), .y = @intCast(rows) }, 256, &args);
    }

    /// The text model's rows leaving its last run layer [T, hidden] bf16 for `ids` at M-RoPE positions
    /// `pos`, vision rows spliced at the pads when given. `res` holds the weights on the device, so
    /// a warm call moves none — only the rotary tables, ids and vision rows cross.
    pub fn encode(r: *Runner, a: std.mem.Allocator, g: rules.Text, res: *const Resident, ids: []const u32, pos: []const [3]i32, vision: ?Vision, out: []u16) !void {
        const T = ids.len;
        const H = g.hidden;
        const qh = g.heads * D;
        const kvh = g.kv_heads * D;
        if (T == 0 or pos.len != T or out.len != T * H or g.layers == 0 or res.layers.len < g.layers) return error.H3PromptRows;
        const n = if (vision) |v| try v.check(T, H) else 0;
        var sizes = std.enums.EnumArray(EncodePart, usize).initFill(T * H * 2);
        // `p` holds one product's fp32 rows at a time; the rest are bf16 buffers at their width
        sizes.set(.p, T * @max(H, @max(qh, g.ffn)) * 4);
        sizes.set(.qb, T * qh * 2);
        sizes.set(.kb, T * kvh * 2);
        sizes.set(.vb, T * kvh * 2);
        sizes.set(.att, T * qh * 2);
        sizes.set(.gb, T * g.ffn * 2);
        sizes.set(.ub, T * g.ffn * 2);
        sizes.set(.act, T * g.ffn * 2);
        sizes.set(.cos, T * 64 * 4);
        sizes.set(.sin, T * 64 * 4);
        sizes.set(.idx, (T + n) * 4);
        sizes.set(.vision, n * H * 2);
        sizes.set(.deep, 3 * n * H * 2);
        var s = try Scratch(EncodePart).make(&r.driver, sizes);
        defer s.buf.free();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 prompt encode did not drain");

        // the rotary tables on the host, as picture_prompt.rotary writes them
        const cs = try a.alloc(f32, T * 64);
        defer a.free(cs);
        const sn = try a.alloc(f32, T * 64);
        defer a.free(sn);
        rules.rotary(cs, sn, pos, g.theta, g.sections);
        try r.copy(std.mem.sliceAsBytes(cs), s.at(.cos));
        try r.copy(std.mem.sliceAsBytes(sn), s.at(.sin));
        {
            const idx = try a.alloc(u32, T + n);
            defer a.free(idx);
            @memcpy(idx[0..T], ids);
            if (vision) |v| @memcpy(idx[T..], v.pads);
            try r.copy(std.mem.sliceAsBytes(idx), s.at(.idx));
            if (vision) |v| {
                try r.copy(std.mem.sliceAsBytes(v.rows), s.at(.vision));
                for (v.deepstack, 0..) |d, i| try r.copy(std.mem.sliceAsBytes(d), s.at(.deep) + i * n * H * 2);
            }
        }
        // the resident table gathers ids' rows; pads' vision rows splice over them
        try r.indexed("h3p_embed", res.embed, s.at(.idx), s.at(.x), T, H);
        if (vision) |_| try r.indexed("h3p_splice", s.at(.x), s.at(.vision), s.at(.idx) + T * 4, n, H);

        var begun = try cuda.Event.init(&r.driver, true);
        defer begun.deinit();
        var ended = try cuda.Event.init(&r.driver, true);
        defer ended.deinit();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 prompt timing did not drain");
        try begun.record(r.stream);
        for (0..g.layers) |li| {
            const w = res.layers[li];
            try r.norm(s.at(.x), w.norm1, s.at(.xn), T, H, g.eps, mode_qwen);
            try r.linear(s.at(.xn), w.q, s.at(.p), T, qh, H);
            try r.round(s.at(.p), s.at(.qb), T * qh);
            try r.linear(s.at(.xn), w.k, s.at(.p), T, kvh, H);
            try r.round(s.at(.p), s.at(.kb), T * kvh);
            try r.linear(s.at(.xn), w.v, s.at(.p), T, kvh, H);
            try r.round(s.at(.p), s.at(.vb), T * kvh);
            try r.heads(s.at(.qb), s.at(.kb), w.q_norm, w.k_norm, .{ s.at(.cos), s.at(.sin) }, g.heads, g.kv_heads, qh, kvh, D, T, g.eps, mode_qwen);
            try r.attention(s.at(.qb), s.at(.kb), s.at(.vb), s.at(.att), T, g.heads, g.kv_heads, qh, kvh, kvh, D, true);
            try r.linear(s.at(.att), w.o, s.at(.p), T, H, qh);
            try r.add(s.at(.x), s.at(.p), T * H);
            try r.norm(s.at(.x), w.norm2, s.at(.xn), T, H, g.eps, mode_qwen);
            try r.linear(s.at(.xn), w.gate, s.at(.p), T, g.ffn, H);
            try r.round(s.at(.p), s.at(.gb), T * g.ffn);
            try r.linear(s.at(.xn), w.up, s.at(.p), T, g.ffn, H);
            try r.round(s.at(.p), s.at(.ub), T * g.ffn);
            try r.glu(s.at(.gb), s.at(.ub), s.at(.act), T, g.ffn, g.ffn, g.ffn, mode_qwen);
            try r.linear(s.at(.act), w.down, s.at(.p), T, H, g.ffn);
            try r.add(s.at(.x), s.at(.p), T * H);
            if (n > 0 and li < 3) try r.indexed("h3p_scatter_add", s.at(.x), s.at(.deep) + li * n * H * 2, s.at(.idx) + T * 4, n, H);
        }
        try ended.record(r.stream);
        try ended.synchronize();
        r.gpu = @as(f64, try cuda.Event.elapsedMs(begun, ended)) / 1000;
        var dst: cuda.DeviceBuffer = .{ .d = &r.driver, .ptr = s.at(.x), .len = out.len * 2 };
        try dst.download(0, std.mem.sliceAsBytes(out));
    }

    /// condition_proj, the refiner's blocks over every row as one document, its final norm:
    /// rows [T, text] bf16 to out [T, hidden] bf16, its weights `res`'s resident ones.
    pub fn refine(r: *Runner, g: rules.Refiner, res: *const Resident, rows: []const u16, out: []u16) !void {
        const H = g.hidden;
        const inner = g.heads * D;
        const T = rows.len / g.text;
        if (T == 0 or rows.len != T * g.text or out.len != T * H or g.blocks == 0 or res.blocks.len < g.blocks) return error.H3PromptRows;
        var sizes = std.enums.EnumArray(RefinePart, usize).initFill(T * @max(g.text, H) * 2);
        sizes.set(.p, T * @max(H, @max(3 * inner, 2 * g.mlp)) * 4);
        sizes.set(.qkv, T * 3 * inner * 2);
        sizes.set(.att, T * inner * 2);
        sizes.set(.wide, T * 2 * g.mlp * 2);
        sizes.set(.act, T * g.mlp * 2);
        var s = try Scratch(RefinePart).make(&r.driver, sizes);
        defer s.buf.free();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 refiner did not drain");
        try r.copy(std.mem.sliceAsBytes(rows), s.at(.x));

        var begun = try cuda.Event.init(&r.driver, true);
        defer begun.deinit();
        var ended = try cuda.Event.init(&r.driver, true);
        defer ended.deinit();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 refiner timing did not drain");
        try begun.record(r.stream);
        // condition_proj: its bias fused in fp32 into the product, then the reference's one rounding
        try r.products.run(r.stream, s.at(.x), res.cond_w, res.cond_b, s.at(.p), T, H, g.text);
        try r.round(s.at(.p), s.at(.x), T * H);
        const fused_row = 3 * inner;
        for (0..g.blocks) |bi| {
            const b = res.blocks[bi];
            try r.norm(s.at(.x), b.norm1, s.at(.xn), T, H, g.eps, mode_h3);
            try r.products.run(r.stream, s.at(.xn), b.qkv, 0, s.at(.p), T, 3 * inner, H);
            try r.round(s.at(.p), s.at(.qkv), T * 3 * inner);
            // the fused rows' q and k heads normed in place, no rotary; v sits `head` = 384 apart
            try r.heads(s.at(.qkv), s.at(.qkv) + D * 2, b.q_norm, b.k_norm, null, g.heads, g.heads, fused_row, fused_row, 3 * D, T, g.qk_eps, mode_h3);
            try r.attention(s.at(.qkv), s.at(.qkv) + D * 2, s.at(.qkv) + 2 * D * 2, s.at(.att), T, g.heads, g.heads, fused_row, fused_row, fused_row, 3 * D, false);
            try r.products.run(r.stream, s.at(.att), b.out, 0, s.at(.p), T, H, inner);
            try r.add(s.at(.x), s.at(.p), T * H);
            try r.norm(s.at(.x), b.norm2, s.at(.xn), T, H, g.eps, mode_h3);
            try r.products.run(r.stream, s.at(.xn), b.fc1, 0, s.at(.p), T, 2 * g.mlp, H);
            try r.round(s.at(.p), s.at(.wide), T * 2 * g.mlp);
            try r.glu(s.at(.wide), s.at(.wide) + g.mlp * 2, s.at(.act), T, g.mlp, 2 * g.mlp, 2 * g.mlp, mode_h3);
            try r.products.run(r.stream, s.at(.act), b.fc2, 0, s.at(.p), T, H, g.mlp);
            try r.add(s.at(.x), s.at(.p), T * H);
        }
        try r.norm(s.at(.x), res.final_norm, s.at(.xn), T, H, g.final_eps, mode_h3);
        try ended.record(r.stream);
        try ended.synchronize();
        r.gpu = @as(f64, try cuda.Event.elapsedMs(begun, ended)) / 1000;
        // the final norm wrote bf16 rows straight to `out`'s shape
        var dst: cuda.DeviceBuffer = .{ .d = &r.driver, .ptr = s.at(.xn), .len = out.len * 2 };
        try dst.download(0, std.mem.sliceAsBytes(out));
    }
};

test "condition projection keeps bias when the actual-row heuristic cannot fuse it" {
    try @import("prompt_cuda_products_test.zig").check(Products, Products.run);
}

test "condition projection draws at every prompt length, its bias fused" {
    try @import("prompt_cuda_lengths_test.zig").check(Products, Products.run);
}

test "the prompt cache's states, budget and LRU order" {
    _ = @import("prompt_cache_test.zig");
}
