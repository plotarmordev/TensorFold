//! Inkling 2 Mini's DSpark draft head on CUDA: mtp.safetensors' three-layer block over the target's committed taps
//! (layers 5, 23, 41, 48), then the Markov chain's greedy drafts, every step's token picked and fed back on the device.
//! The projections run on the target's bf16 lane GEMM (qmmf); its embedding and unembedding are the target's.
//! Unlike the Mac's folded Phi head, a step's correction hidden_out·lambda + bias joins y before the one unembed GEMM.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wts = @import("mini_weights.zig");

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

/// mtp.safetensors in `dir`, every tensor checked by name and shape, to the GPU.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
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
};

/// The target pieces the head reads: its embedding, embed norm and unembedding, and its lane GEMM.
pub const Target = struct { w: *const wts.Weights, lane: *const cuda.qmmf.Lane };

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
        var part: usize = t.lane.partBytes(1, t.w.unembed);
        for (w.proj) |p| for ([_]Lane{ p.q, p.k, p.v, p.o, p.gate, p.up, p.down }) |l| {
            part = @max(part, t.lane.partBytes(rows, l));
        };
        for (w.proj) |p| for (1..absorb_rows + 1) |n| for ([_]Lane{ p.k_ctx, p.v_ctx }) |l| {
            part = @max(part, t.lane.partBytes(n, l));
        };
        const sizes = .{
            .resid = rows * hidden * 4, .x = rows * hidden * 2,   .y = rows * hidden * 2,     .q = rows * query * 2,
            .kb = rows * kv * 2,        .vb = rows * kv * 2,      .att = rows * query * 2,    .gate = rows * inter * 2,
            .up = rows * inter * 2,     .act = rows * inter * 2,  .fused = 4 * absorb_rows * hidden * 2,
            .ck = absorb_rows * kv * 2, .cv = absorb_rows * kv * 2, .kappa = depth * 3 * rank * 4, .state = rank * 4,
            .h = hidden * 2,            .logits = depth * v * 4, .picks = (1 + depth) * 4,   .part = @max(part, 16),
        };
        inline for (@typeInfo(@TypeOf(sizes)).@"struct".field_names) |name| {
            var b = try DeviceBuffer.alloc(d, @field(sizes, name));
            errdefer b.free();
            try h.buffers.append(gpa, b);
            @field(h, name) = b.ptr;
        }
        return h;
    }

    pub fn deinit(h: *Head, gpa: std.mem.Allocator) void {
        for (h.buffers.items) |*b| b.free();
        h.buffers.deinit(gpa);
    }

    fn vocab(t: Target) usize {
        return t.w.config.text.unpadded_vocab;
    }

    fn gemm(h: *const Head, s: cuda.Stream, x: u64, n: usize, w: Lane, y: u64) !void {
        try h.t.lane.matmulOut(s, x, w.k, n, w, y, .bf16, h.part);
    }

    /// Committed rows at positions start.. (taps bf16 [n][4][H] on the device) into the context's rings.
    pub fn absorb(h: *const Head, s: cuda.Stream, c: *Context, tap_rows: u64, start: u64, n: usize) !void {
        if (start < c.end) return error.DraftContextGap;
        if (start > c.end) c.from = start;
        if (start + n - c.from > c.rings[1]) return error.DSparkContextFull;
        var at: usize = 0;
        while (at < n) : (at += absorb_rows) {
            const m: u32 = @intCast(@min(absorb_rows, n - at));
            var f: cuda.Args = .{};
            for ([_]u64{ tap_rows + at * 4 * hidden * 2, h.w.fuse, h.w.dflare, h.fused, c.newest.ptr }) |p| f.add(p);
            f.add(@as(c_int, @intCast(m)));
            try go(h.k.fuse, s, .{ .x = m, .y = 4 }, 256, &f);
            for (h.w.proj, h.w.norms, 0..) |p, nm, i| {
                const fused = h.fused + i * m * hidden * 2;
                try h.gemm(s, fused, m, p.k_ctx, h.ck);
                try h.gemm(s, fused, m, p.v_ctx, h.cv);
                var a: cuda.Args = .{};
                for ([_]u64{ h.ck, h.cv, nm.k_ctx, c.k[i].ptr, c.v[i].ptr }) |ptr| a.add(ptr);
                a.add(@as(u32, @intCast(start + at)));
                a.add(@as(c_int, @intCast(c.rings[i])));
                try go(h.k.ctx, s, .{ .x = m, .y = 8 }, 32, &a);
            }
        }
        c.end = start + n;
    }

    /// Seven greedy drafts after the pending token (picks[0] on the device, at position c.end): queued on `s` with no
    /// host step between them; picks[1..8] and the logits hold the drafts and their soft-capped logits.
    pub fn draft(h: *const Head, s: cuda.Stream, c: *const Context) !void {
        if (c.end == c.from) return error.DSparkNoContext;
        const tw = h.t.w;
        var a: cuda.Args = .{};
        for ([_]u64{ h.picks, tw.embed, tw.embed_norm }) |p| a.add(p);
        a.add(tw.config.text.eps);
        for ([_]u64{ h.w.mask, h.w.norms[0].input, h.resid, h.x }) |p| a.add(p);
        try go(h.k.init, s, .{ .x = rows }, 256, &a);
        for (h.w.proj, h.w.norms, 0..) |p, nm, i| {
            try h.gemm(s, h.x, rows, p.q, h.q);
            try h.gemm(s, h.x, rows, p.k, h.kb);
            try h.gemm(s, h.x, rows, p.v, h.vb);
            var qk: cuda.Args = .{};
            for ([_]u64{ h.q, h.kb, nm.q, nm.k }) |ptr| qk.add(ptr);
            qk.add(@as(u32, @intCast(c.end)));
            try go(h.k.qk, s, .{ .x = rows, .y = 32 }, 32, &qk);
            var at: cuda.Args = .{};
            for ([_]u64{ h.q, h.kb, h.vb, c.k[i].ptr, c.v[i].ptr, h.att }) |ptr| at.add(ptr);
            at.add(@as(u32, @intCast(c.from)));
            at.add(@as(u32, @intCast(c.end)));
            at.add(@as(c_int, @intCast(c.rings[i])));
            at.add(@as(c_int, @intFromBool(i != 1)));
            try go(h.k.attn, s, .{ .x = rows, .y = 24 }, 32, &at);
            try h.gemm(s, h.att, rows, p.o, h.y);
            try h.addNorm(s, nm.post);
            try h.gemm(s, h.x, rows, p.gate, h.gate);
            try h.gemm(s, h.x, rows, p.up, h.up);
            var sw: cuda.Args = .{};
            for ([_]u64{ h.gate, h.up, h.act }) |ptr| sw.add(ptr);
            sw.add(@as(c_int, rows * inter));
            try go(h.k.swiglu, s, .{ .x = rows * inter / 256 }, 256, &sw);
            try h.gemm(s, h.act, rows, p.down, h.y);
            try h.addNorm(s, if (i < 2) h.w.norms[i + 1].input else h.w.norm);
        }
        var kp: cuda.Args = .{};
        for ([_]u64{ h.x, c.newest.ptr, h.w.rnn, h.w.rnn_b, h.kappa }) |ptr| kp.add(ptr);
        try go(h.k.kappa, s, .{ .x = 3 * rank / 8, .y = depth }, 256, &kp);
        try (DeviceBuffer{ .d = s.d, .ptr = h.state, .len = rank * 4, .borrowed = true }).fill8(0, s.handle);
        const v = vocab(h.t);
        for (0..depth) |step| {
            var st: cuda.Args = .{};
            for ([_]u64{ h.picks + step * 4, h.w.markov, h.w.rnn, h.kappa + step * 3 * rank * 4, h.w.hidden_w, h.w.hidden_b, h.x + (step + 1) * hidden * 2, h.state, h.h }) |ptr| st.add(ptr);
            try go(h.k.step, s, .{ .x = 1 }, 3 * rank, &st);
            const logits = h.logits + step * v * 4;
            try h.t.lane.matmulOut(s, h.h, hidden, 1, tw.unembed, logits, .f32, h.part);
            var pk: cuda.Args = .{};
            pk.add(logits);
            pk.add(@as(c_int, @intCast(v)));
            pk.add(tw.config.text.logit_cap);
            pk.add(h.picks + (step + 1) * 4);
            try go(h.k.pick, s, .{ .x = 1 }, 1024, &pk);
        }
    }

    fn addNorm(h: *const Head, s: cuda.Stream, norm: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ h.y, norm, h.resid, h.x }) |p| a.add(p);
        try go(h.k.add_norm, s, .{ .x = rows }, 256, &a);
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}
