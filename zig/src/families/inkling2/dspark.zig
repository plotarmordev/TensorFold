//! CUDA DSpark: BF16 three-layer draft block over committed visit taps, then seven device-side greedy Markov steps.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wts = @import("loop_weights.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Lane = cuda.qmmf.Weight;

pub const hidden = 2304;
pub const query = 3072;
pub const kv = 1024;
pub const inter = 9216;
pub const rank = 256;
pub const markov_vocab = 201024;
/// The block's rows: the pending token, then seven masks; each mask row drafts one step.
pub const rows = 8;
pub const depth = rows - 1;
/// Context rows one absorb launch set takes.
pub const absorb_rows = 64;
pub const taps = [_]u32{ 5, 23, 41, 48 };
/// Local layers (0 and 2) attend to the last 1024 context rows; their rings hold that much.
pub const local_ring = 1024;

const Proj = struct { q: Lane, k: Lane, v: Lane, k_ctx: Lane, v_ctx: Lane, o: Lane, gate: Lane, up: Lane, down: Lane };
const Norms = struct { input: u64, post: u64, q: u64, k: u64, k_ctx: u64 };

pub const Weights = struct {
    proj: [3]Proj,
    norms: [3]Norms,
    fuse: u64, // bf16 [4][4]
    dflare: u64, // bf16 [4][H]: the layers' then the Markov head's
    mask: u64,
    norm: u64,
    hidden_w: u64, // bf16 [H][R]
    hidden_b: u64,
    markov: u64, // bf16 [201024][R]
    rnn: u64, // bf16 [768][5120]
    rnn_b: u64,
    unembed: Lane, // packed BF16 copy of the target's output table
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ck: *core.checkpoint.Checkpoint,
    w: *Weights,

    fn tensor(l: Loader, name: []const u8, dims: []const usize) !core.checkpoint.Tensor {
        const t = l.ck.get(name) catch |e| {
            std.log.err("inkling2 dspark: mtp.safetensors has no {s}", .{name});
            return e;
        };
        if (t.dtype != .bf16 or t.rank != dims.len) return error.DSparkTensorShape;
        for (dims, 0..) |n, i| if (t.dim(i) != n) return error.DSparkTensorShape;
        return t;
    }

    fn upload(l: Loader, bytes: []const u8) !u64 {
        var b = try DeviceBuffer.fromHost(l.d, bytes);
        errdefer b.free();
        try l.w.buffers.append(l.gpa, b);
        return b.ptr;
    }

    fn raw(l: Loader, name: []const u8, dims: []const usize) !u64 {
        return l.upload((try l.tensor(name, dims)).bytes);
    }

    /// A [n][k] projection packed for the lane GEMM.
    fn lane(l: Loader, name: []const u8, n: usize, k: usize) !Lane {
        const t = try l.tensor(name, &.{ n, k });
        const src = try l.gpa.alloc(u16, n * k); // mapped bytes need not be u16-aligned
        defer l.gpa.free(src);
        @memcpy(std.mem.sliceAsBytes(src), t.bytes);
        const out = try l.gpa.alloc(u16, cuda.modelopt.padded(n) * k);
        defer l.gpa.free(out);
        cuda.qmmf.packBf16(out, src, n, k);
        return cuda.modelopt.bf16Weight(try l.upload(std.mem.sliceAsBytes(out)), n, k);
    }
};

/// Load mtp.safetensors and a packed target unembedding for the shared BF16 matmul lane.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, target_dir: []const u8, dir: []const u8, vocab: usize) !Weights {
    var ck: core.checkpoint.Checkpoint = .{ .gpa = gpa, .io = io };
    defer ck.close();
    ck.add(dir, "mtp.safetensors") catch |e| {
        std.log.err("inkling2 dspark: no mtp.safetensors in {s}", .{dir});
        return e;
    };
    var w: Weights = undefined;
    w.buffers = .empty;
    errdefer w.deinit(gpa);
    const l: Loader = .{ .gpa = gpa, .d = d, .ck = &ck, .w = &w };
    var dflare: [4 * hidden]u16 = undefined;
    for (0..3) |i| {
        var b: [96]u8 = undefined;
        const n = struct {
            fn at(buf: []u8, layer: usize, s: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ layer, s });
            }
        }.at;
        w.proj[i] = .{
            .q = try l.lane(try n(&b, i, "self_attn.q_proj.weight"), query, hidden),
            .k = try l.lane(try n(&b, i, "self_attn.k_proj.weight"), kv, hidden),
            .v = try l.lane(try n(&b, i, "self_attn.v_proj.weight"), kv, hidden),
            .k_ctx = try l.lane(try n(&b, i, "self_attn.k_proj_ctx.weight"), kv, hidden),
            .v_ctx = try l.lane(try n(&b, i, "self_attn.v_proj_ctx.weight"), kv, hidden),
            .o = try l.lane(try n(&b, i, "self_attn.o_proj.weight"), hidden, query),
            .gate = try l.lane(try n(&b, i, "mlp.gate_proj.weight"), inter, hidden),
            .up = try l.lane(try n(&b, i, "mlp.up_proj.weight"), inter, hidden),
            .down = try l.lane(try n(&b, i, "mlp.down_proj.weight"), hidden, inter),
        };
        w.norms[i] = .{
            .input = try l.raw(try n(&b, i, "input_layernorm.weight"), &.{hidden}),
            .post = try l.raw(try n(&b, i, "post_attention_layernorm.weight"), &.{hidden}),
            .q = try l.raw(try n(&b, i, "self_attn.q_norm.weight"), &.{128}),
            .k = try l.raw(try n(&b, i, "self_attn.k_norm.weight"), &.{128}),
            .k_ctx = try l.raw(try n(&b, i, "self_attn.k_norm_ctx.weight"), &.{128}),
        };
        @memcpy(std.mem.sliceAsBytes(dflare[i * hidden ..][0..hidden]), (try l.tensor(try n(&b, i, "dflare_norm.weight"), &.{hidden})).bytes);
    }
    @memcpy(std.mem.sliceAsBytes(dflare[3 * hidden ..]), (try l.tensor("markov_head.dflare_norm.weight", &.{hidden})).bytes);
    w.dflare = try l.upload(std.mem.sliceAsBytes(&dflare));
    w.fuse = try l.raw("fuse_weight", &.{ 4, 4 });
    w.mask = try l.raw("mask_embed", &.{hidden});
    w.norm = try l.raw("norm.weight", &.{hidden});
    w.hidden_w = try l.raw("markov_head.hidden_out.weight", &.{ hidden, rank });
    w.hidden_b = try l.raw("markov_head.hidden_out.bias", &.{hidden});
    w.markov = try l.raw("markov_head.markov_w1.weight", &.{ markov_vocab, rank });
    w.rnn = try l.raw("markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * hidden + 2 * rank });
    w.rnn_b = try l.raw("markov_head.rnn_proj.bias", &.{3 * rank});
    var target = try core.Checkpoint.openModelPrefixes(gpa, io, target_dir, &.{"model.llm.unembed.weight"});
    defer target.close();
    const source = try target.expect("model.llm.unembed.weight", .bf16, &.{ vocab, hidden });
    var aligned: ?[]u16 = null;
    defer if (aligned) |buf| gpa.free(buf);
    const src: []const u16 = if (@intFromPtr(source.bytes.ptr) % @alignOf(u16) == 0)
        @as([*]const u16, @ptrCast(@alignCast(source.bytes.ptr)))[0 .. vocab * hidden]
    else blk: {
        const buf = try gpa.alloc(u16, vocab * hidden);
        @memcpy(std.mem.sliceAsBytes(buf), source.bytes);
        aligned = buf;
        break :blk buf;
    };
    const table = try gpa.alloc(u16, cuda.modelopt.padded(vocab) * hidden);
    defer gpa.free(table);
    cuda.qmmf.packBf16(table, src, vocab, hidden);
    w.unembed = cuda.modelopt.bf16Weight(try l.upload(std.mem.sliceAsBytes(table)), vocab, hidden);
    return w;
}

pub const Kernels = struct {
    mod: cuda.Module,
    fuse: cuda.Function,
    ctx: cuda.Function,
    init: cuda.Function,
    add_norm: cuda.Function,
    qk: cuda.Function,
    attn: cuda.Function,
    swiglu: cuda.Function,
    kappa: cuda.Function,
    step: cuda.Function,
    pick: cuda.Function,
    tap: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.mod = try cuda.Module.load(d, cuda.kernels.inkling2_dspark);
        errdefer k.mod.unload();
        inline for (@typeInfo(Kernels).@"struct".field_names[1..]) |name| @field(k, name) = try k.mod.function("tf_ink2_dsp_" ++ name);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
    /// Store one completed visit's BF16 residual in its interleaved tap column.
    pub fn launchTap(k: *const Kernels, s: cuda.Stream, partial: u64, taps_out: u64, n: usize, index: usize) !void {
        var a: cuda.Args = .{};
        a.add(partial);
        a.add(taps_out);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(index)));
        try go(k.tap, s, .{ .x = (hidden + 255) / 256, .y = @intCast(n) }, 256, &a);
    }

    /// Fuse [n][4][H] taps into [4][n][H]; fusion three's final row also becomes newest.
    pub fn launchFuse(k: *const Kernels, s: cuda.Stream, taps_in: u64, fuse: u64, dflare: u64, c: u64, newest: u64, n: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ taps_in, fuse, dflare, c, newest }) |p| a.add(p);
        a.add(@as(c_int, @intCast(n)));
        try go(k.fuse, s, .{ .x = @intCast(n), .y = 4 }, 256, &a);
    }

    /// Rows k/v [n][KV] normed (k) and RoPE'd at positions base.. into a layer's rings, one warp a (row, kv head).
    pub fn launchCtx(k: *const Kernels, s: cuda.Stream, k_in: u64, v_in: u64, knorm: u64, ring_k: u64, ring_v: u64, base: u64, n: usize, ring: u32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ k_in, v_in, knorm, ring_k, ring_v }) |p| a.add(p);
        a.add(@as(u32, @intCast(base)));
        a.add(@as(c_int, @intCast(ring)));
        try go(k.ctx, s, .{ .x = @intCast(n), .y = 8 }, 32, &a);
    }

    /// Row zero uses the pending token's normalized embedding; rows one through seven use the mask.
    pub fn launchInit(k: *const Kernels, s: cuda.Stream, pending: u64, embed: u64, embed_norm: u64, embed_eps: f32, mask: u64, norm: u64, resid: u64, x: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ pending, embed, embed_norm }) |p| a.add(p);
        a.add(embed_eps);
        for ([_]u64{ mask, norm, resid, x }) |p| a.add(p);
        try go(k.init, s, .{ .x = rows }, 256, &a);
    }

    /// The block's Q and K heads normed and RoPE'd at anchor + row in place; one warp a (row, head).
    pub fn launchQk(k: *const Kernels, s: cuda.Stream, q: u64, kb: u64, qnorm: u64, knorm: u64, anchor: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ q, kb, qnorm, knorm }) |p| a.add(p);
        a.add(@as(u32, @intCast(anchor)));
        try go(k.qk, s, .{ .x = rows, .y = 32 }, 32, &a);
    }

    /// Attend over the context ring and eight block rows, one warp per (row, head).
    pub fn launchAttn(k: *const Kernels, s: cuda.Stream, q: u64, bk: u64, bv: u64, ring_k: u64, ring_v: u64, out: u64, from: u64, committed: u64, ring: u32, local: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ q, bk, bv, ring_k, ring_v, out }) |p| a.add(p);
        a.add(@as(u32, @intCast(from)));
        a.add(@as(u32, @intCast(committed)));
        a.add(@as(c_int, @intCast(ring)));
        a.add(@as(c_int, @intFromBool(local)));
        try go(k.attn, s, .{ .x = rows, .y = 24 }, 32, &a);
    }

    /// The Markov chain's step-independent z part per (step, channel): one warp each of 768 over 7 steps.
    pub fn launchKappa(k: *const Kernels, s: cuda.Stream, y: u64, newest: u64, rnn: u64, bias: u64, kappa: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ y, newest, rnn, bias, kappa }) |p| a.add(p);
        try go(k.kappa, s, .{ .x = 3 * rank / 8, .y = depth }, 256, &a);
    }

    /// One Markov step: z from the state and prev's markov row, the gated update, then h bf16 [H] for the unembed.
    pub fn launchStep(k: *const Kernels, s: cuda.Stream, prev: u64, markov: u64, rnn: u64, kappa: u64, hidden_w: u64, hidden_b: u64, y: u64, state: u64, h: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ prev, markov, rnn, kappa, hidden_w, hidden_b, y, state, h }) |p| a.add(p);
        try go(k.step, s, .{ .x = 1 }, 3 * rank, &a);
    }

    /// act = silu(gate) * up over `count` bf16 values (a multiple of 256).
    pub fn launchSwiglu(k: *const Kernels, s: cuda.Stream, gate: u64, up: u64, act: u64, count: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ gate, up, act }) |p| a.add(p);
        a.add(@as(c_int, @intCast(count)));
        try go(k.swiglu, s, .{ .x = @intCast(count / 256) }, 256, &a);
    }

    /// resid += y, then x = bf16(rmsnorm(resid) * norm), `n` rows of the hidden width.
    pub fn launchAddNorm(k: *const Kernels, s: cuda.Stream, y: u64, norm: u64, resid: u64, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ y, norm, resid, x }) |p| a.add(p);
        try go(k.add_norm, s, .{ .x = @intCast(n) }, 256, &a);
    }

    /// Caps fp32 logits in place (cap * tanh(x / cap)) and writes the first id of their maximum to `pick`.
    pub fn launchPick(k: *const Kernels, s: cuda.Stream, logits: u64, n: usize, cap: f32, pick: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(c_int, @intCast(n)));
        a.add(cap);
        a.add(pick);
        try go(k.pick, s, .{ .x = 1 }, 1024, &a);
    }
};

/// One stream's committed context: each layer's K/V ring, the newest fused Markov feature, and its span.
pub const Context = struct {
    k: [3]DeviceBuffer,
    v: [3]DeviceBuffer,
    rings: [3]u32,
    newest: DeviceBuffer,
    from: u64 = 0,
    end: u64 = 0,

    /// `capacity`: the global layer's longest context.
    pub fn init(d: *const cuda.Driver, capacity: u32) !Context {
        var c: Context = .{ .k = undefined, .v = undefined, .rings = .{ local_ring, capacity, local_ring }, .newest = undefined };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            c.k[i].free();
            c.v[i].free();
        };
        for (c.rings, 0..) |ring, i| {
            c.k[i] = try DeviceBuffer.alloc(d, @as(usize, ring) * kv * 2);
            c.v[i] = DeviceBuffer.alloc(d, @as(usize, ring) * kv * 2) catch |e| {
                c.k[i].free();
                return e;
            };
            made += 1;
        }
        c.newest = try DeviceBuffer.alloc(d, hidden * 2);
        return c;
    }

    pub fn deinit(c: *Context) void {
        c.newest.free();
        for (0..3) |i| {
            c.k[i].free();
            c.v[i].free();
        }
    }

    const Span = struct { from: u64, end: u64 };

    /// Reject overlap and overflow before device work; a gap restarts the committed span.
    fn admit(c: *const Context, start: u64, n: usize) !Span {
        if (start < c.end) return error.DraftContextGap;
        const from = if (start > c.end) start else c.from;
        if (start + n - from > c.rings[1]) return error.DSparkContextFull;
        return .{ .from = from, .end = start + n };
    }
};

test "a rejected absorb leaves the context's span whole and the next one lands" {
    var c: Context = .{ .k = undefined, .v = undefined, .rings = .{ local_ring, 64, local_ring }, .newest = undefined, .from = 10, .end = 20 };
    try std.testing.expectError(error.DSparkContextFull, c.admit(100, 65));
    try std.testing.expectError(error.DraftContextGap, c.admit(19, 1));
    try std.testing.expectEqual(@as(u64, 10), c.from);
    try std.testing.expectEqual(@as(u64, 20), c.end);
    try std.testing.expectEqual(Context.Span{ .from = 10, .end = 74 }, try c.admit(20, 54));
    const s = try c.admit(100, 64);
    try std.testing.expectEqual(@as(u64, 100), s.from);
    try std.testing.expectEqual(@as(u64, 164), s.end);
    c.from, c.end = .{ s.from, s.end };
    try std.testing.expectError(error.DraftContextGap, c.admit(163, 1));
    try std.testing.expectEqual(@as(u64, 164), c.end);
}

/// The target pieces the head reads: its embedding, embed norm and unembedding, and its lane GEMM.
pub const Target = struct { w: *const wts.Weights, lane: *const cuda.qmmf.Lane, unembed: Lane };

/// The head's device scratch: the block's rows, the absorb rows' features, the chain's state and every step's logits.
pub const Head = struct {
    w: *const Weights,
    k: *const Kernels,
    t: Target,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    resid: u64 = 0, // fp32 [8][H]
    x: u64 = 0, // bf16 [8][H]
    y: u64 = 0,
    q: u64 = 0, // bf16 [8][Q]
    kb: u64 = 0, // bf16 [8][KV]
    vb: u64 = 0,
    att: u64 = 0, // bf16 [8][Q]
    gate: u64 = 0, // bf16 [8][I]
    up: u64 = 0,
    act: u64 = 0,
    fused: u64 = 0, // bf16 [4][64][H]
    ck: u64 = 0, // bf16 [64][KV]
    cv: u64 = 0,
    kappa: u64 = 0, // fp32 [7][768]
    state: u64 = 0, // fp32 [R]
    h: u64 = 0, // bf16 [H]
    /// fp32 [7][vocab]: every step's soft-capped logits; u32 picks [1 + 7]: the pending token, then each step's draft
    logits: u64 = 0,
    picks: u64 = 0,
    part: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const Weights, k: *const Kernels, t: Target) !Head {
        var h: Head = .{ .w = w, .k = k, .t = t };
        errdefer h.deinit(gpa);
        const v = vocab(t);
        var part: usize = t.lane.partBytes(1, t.unembed);
        for (w.proj) |p| for ([_]Lane{ p.q, p.k, p.v, p.o, p.gate, p.up, p.down }) |l| {
            part = @max(part, t.lane.partBytes(rows, l));
        };
        for (w.proj) |p| for (1..absorb_rows + 1) |n| for ([_]Lane{ p.k_ctx, p.v_ctx }) |l| {
            part = @max(part, t.lane.partBytes(n, l));
        };
        const sizes = .{
            .resid = rows * hidden * 4,
            .x = rows * hidden * 2,
            .y = rows * hidden * 2,
            .q = rows * query * 2,
            .kb = rows * kv * 2,
            .vb = rows * kv * 2,
            .att = rows * query * 2,
            .gate = rows * inter * 2,
            .up = rows * inter * 2,
            .act = rows * inter * 2,
            .fused = 4 * absorb_rows * hidden * 2,
            .ck = absorb_rows * kv * 2,
            .cv = absorb_rows * kv * 2,
            .kappa = depth * 3 * rank * 4,
            .state = rank * 4,
            .h = hidden * 2,
            .logits = depth * v * 4,
            .picks = (1 + depth) * 4,
            .part = @max(part, 16),
        };
        inline for (@typeInfo(@TypeOf(sizes)).@"struct".field_names) |name| {
            var b = try DeviceBuffer.alloc(d, @field(sizes, name));
            h.buffers.append(gpa, b) catch |err| {
                b.free();
                return err;
            };
            @field(h, name) = b.ptr;
        }
        return h;
    }

    pub fn deinit(h: *Head, gpa: std.mem.Allocator) void {
        for (h.buffers.items) |*b| b.free();
        h.buffers.deinit(gpa);
    }

    fn vocab(t: Target) usize {
        return t.w.c.text.unpadded_vocab;
    }

    fn gemm(h: *const Head, s: cuda.Stream, x: u64, n: usize, w: Lane, y: u64) !void {
        try h.t.lane.matmulOut(s, x, w.k, n, w, y, .bf16, h.part);
    }

    /// Committed rows at positions start.. (taps bf16 [n][4][H] on the device) into the context's rings.
    pub fn absorb(h: *const Head, s: cuda.Stream, c: *Context, tap_rows: u64, start: u64, n: usize) !void {
        const span = try c.admit(start, n);
        var at: usize = 0;
        while (at < n) : (at += absorb_rows) {
            const m: usize = @min(absorb_rows, n - at);
            try h.k.launchFuse(s, tap_rows + at * 4 * hidden * 2, h.w.fuse, h.w.dflare, h.fused, c.newest.ptr, m);
            for (h.w.proj, h.w.norms, 0..) |p, nm, i| {
                const fused = h.fused + i * m * hidden * 2;
                try h.gemm(s, fused, m, p.k_ctx, h.ck);
                try h.gemm(s, fused, m, p.v_ctx, h.cv);
                try h.k.launchCtx(s, h.ck, h.cv, nm.k_ctx, c.k[i].ptr, c.v[i].ptr, start + at, m, c.rings[i]);
            }
        }
        c.from, c.end = .{ span.from, span.end };
    }

    /// Queue seven device-side greedy picks without advancing the committed context span.
    pub fn draft(h: *const Head, s: cuda.Stream, c: *const Context) !void {
        if (c.end <= c.from) return error.DSparkNoContext;
        const tw = h.t.w;
        try h.k.launchInit(s, h.picks, tw.embed, tw.embed_norm, tw.c.text.eps, h.w.mask, h.w.norms[0].input, h.resid, h.x);
        for (h.w.proj, h.w.norms, 0..) |p, nm, i| {
            try h.gemm(s, h.x, rows, p.q, h.q);
            try h.gemm(s, h.x, rows, p.k, h.kb);
            try h.gemm(s, h.x, rows, p.v, h.vb);
            try h.k.launchQk(s, h.q, h.kb, nm.q, nm.k, c.end);
            try h.k.launchAttn(s, h.q, h.kb, h.vb, c.k[i].ptr, c.v[i].ptr, h.att, c.from, c.end, c.rings[i], i != 1);
            try h.gemm(s, h.att, rows, p.o, h.y);
            try h.addNorm(s, nm.post);
            try h.gemm(s, h.x, rows, p.gate, h.gate);
            try h.gemm(s, h.x, rows, p.up, h.up);
            try h.k.launchSwiglu(s, h.gate, h.up, h.act, rows * inter);
            try h.gemm(s, h.act, rows, p.down, h.y);
            try h.addNorm(s, if (i < 2) h.w.norms[i + 1].input else h.w.norm);
        }
        try h.k.launchKappa(s, h.x, c.newest.ptr, h.w.rnn, h.w.rnn_b, h.kappa);
        try (DeviceBuffer{ .d = s.d, .ptr = h.state, .len = rank * 4, .borrowed = true }).fill8(0, s.handle);
        const v = vocab(h.t);
        for (0..depth) |step| {
            try h.k.launchStep(s, h.picks + step * 4, h.w.markov, h.w.rnn, h.kappa + step * 3 * rank * 4, h.w.hidden_w, h.w.hidden_b, h.x + (step + 1) * hidden * 2, h.state, h.h);
            const logits = h.logits + step * v * 4;
            try h.t.lane.matmulOut(s, h.h, hidden, 1, h.t.unembed, logits, .f32, h.part);
            try h.k.launchPick(s, logits, v, tw.c.text.logit_cap, h.picks + (step + 1) * 4);
        }
    }

    fn addNorm(h: *const Head, s: cuda.Stream, norm: u64) !void {
        try h.k.launchAddNorm(s, h.y, norm, h.resid, h.x, rows);
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}
