//! One-rank, text-only Inkling 2 Mini: the complete 48-visit CUDA forward.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const loop = @import("loop.zig");
const weights_ = @import("loop_weights.zig");
const ops = @import("ops.zig");
const attention = @import("kernels.zig");
const modelopt = @import("modelopt.zig");
const loop_math = @import("loop_math_cuda.zig");
const snapshot = @import("loop_snapshot.zig");

pub const prompt_rows: usize = 2048;
pub const window_rows: usize = 16;
const tiles_cap = 128; // a layer's tiles at most: 2048 rows of 4 heads a key head, 64 pairs a tile

pub const Kernels = struct {
    glue: ops.Kernels,
    attn: attention.Kernels,
    moe: modelopt.Kernels,
    loop: loop_math.Kernels,
    sample_mod: cuda.Module,
    draws: cuda.sample.Draws,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        var glue = try ops.Kernels.load(d);
        errdefer glue.unload();
        var attn = try attention.Kernels.load(d);
        errdefer attn.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        var moe = try modelopt.Kernels.load(d, sms);
        errdefer moe.unload();
        var lm = try loop_math.Kernels.load(d);
        errdefer lm.unload();
        var sample_mod = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer sample_mod.unload();
        return .{
            .glue = glue,
            .attn = attn,
            .moe = moe,
            .loop = lm,
            .sample_mod = sample_mod,
            .draws = try cuda.sample.Draws.resolve(sample_mod),
        };
    }

    pub fn deinit(k: *Kernels) void {
        k.sample_mod.unload();
        k.loop.unload();
        k.moe.unload();
        k.attn.unload();
        k.glue.unload();
        k.* = undefined;
    }
};

const Scratch = struct {
    ids: u64,
    pos: u64,
    seg_of: u64,
    segs: u64,
    tiles: u64,
    blocks: [ops.max_blocks]u64 = @splat(0),
    partial_a: u64,
    partial_b: u64,
    prelude: u64,
    bank0: u64,
    bank1: u64,
    raw: u64, // three FP32 AR/gate rows per token
    messages: u64, // four FP32 message rows per token
    normalized: u64, // three BF16 rows per token
    gate_zeta: u64, // BF16 [rows][64] for decode's parallel gate
    t: u64,
    q_raw: u64,
    rel_raw: u64,
    sink: u64,
    gate: u64,
    k_raw: u64,
    v_raw: u64,
    k_conv: u64,
    v_conv: u64,
    k_dynamic: u64,
    v_dynamic: u64,
    q_prepared: u64,
    rel_prepared: u64,
    rel_table: u64,
    part_o: u64,
    part_ml: u64,
    attn_heads: u64,
    attn_out: u64,
    route_scores: u64,
    route_pick: u64,
    route_weights: u64,
    moe_out: u64,
    moe_scratch: u64,
    hidden: u64,
    logits: u64,
    backup: u64,
    hist: u64,
};

/// A stream's state (every visit's K/V rows and conv states, its rewind sites); at most `max_slots`, one in use.
const Slot = struct { cache: cuda.DeviceBuffer, sites: cuda.DeviceBuffer };
pub const max_slots = 16;

/// The bytes a slot's allocations ask for: the cache layout and four rewind sites a visit.
pub fn slotAllocs(layout: *const loop.CacheLayout, n_visits: u32) [2]usize {
    return .{ @intCast(layout.bytes), @as(usize, n_visits) * 4 * @sizeOf(ops.RewindSite) };
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const weights_.Weights,
    context: usize,
    layout: loop.CacheLayout,
    slots: [max_slots]Slot = undefined,
    n_slots: usize = 0,
    slot: usize = 0, // the slot forwards, snapshots and rewinds use; forwards share the scratch
    recorded: ?usize = null, // the slot whose window `backup` and `hist` hold
    s: Scratch = undefined,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    graphs: [max_slots]?cuda.graph.Exec = @splat(null), // each slot's one-token decode

    pub fn slotBase(m: *const Model) u64 {
        return m.slots[m.slot].cache.ptr;
    }

    /// Allocates slots until `n` exist, each with zeroed conv states and its own rewind sites.
    pub fn addSlots(m: *Model, s: cuda.Stream, n: usize) !void {
        if (n > max_slots) return error.TooManySlots;
        const keep = m.slot;
        defer m.slot = keep;
        const kv: usize = @as(usize, m.w.c.text.local.kv) * m.w.c.text.local.dim;
        var sites: [config.max_visits * 4]ops.RewindSite = undefined;
        while (m.n_slots < n) {
            var slot = try m.newSlot();
            errdefer for ([2]*cuda.DeviceBuffer{ &slot.sites, &slot.cache }) |b| b.free();
            m.slots[m.n_slots] = slot;
            m.slot = m.n_slots;
            for (0..m.w.p.n_visits) |v| for (0..4) |site| {
                const c = if (site < 2) kv else m.w.c.text.hidden;
                const hist = m.histOf(@intCast(v), site) - m.s.hist;
                sites[v * 4 + site] = .{ .state = m.convOf(@intCast(v), site), .hist = @intCast(hist), .backup = @intCast(m.backupOf(@intCast(v), site)), .c = @intCast(c) };
            };
            try m.upload(s, slot.sites.ptr, std.mem.sliceAsBytes(sites[0 .. m.w.p.n_visits * 4]));
            try m.reset(s);
            try s.synchronize();
            m.n_slots += 1;
        }
    }

    /// A slot's two allocations, sized by slotAllocs: its cache and its rewind-site table.
    pub fn newSlot(m: *const Model) !Slot {
        const sizes = slotAllocs(&m.layout, m.w.p.n_visits);
        var cache = try cuda.DeviceBuffer.alloc(m.d, sizes[0]);
        errdefer cache.free();
        return .{ .cache = cache, .sites = try cuda.DeviceBuffer.alloc(m.d, sizes[1]) };
    }

    fn alloc(m: *Model, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer b.free();
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(m.d, dst, bytes, s);
    }

    pub fn copy(m: *const Model, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    fn zeros(m: *const Model, s: cuda.Stream, ptr: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemsetD8Async(ptr, 0, bytes, s.handle), "cuMemsetD8Async");
    }

    pub fn layerGeometry(m: *const Model, layer: u32) attention.Layer {
        return if (config.isGlobal(layer))
            attention.Layer.global(@intCast(m.context))
        else
            attention.Layer.sliding(m.w.c.text.window, @intCast(@min(prompt_rows, m.context)));
    }

    pub fn cacheOf(m: *const Model, visit: u32) attention.Cache {
        const slot = m.layout.at(visit);
        const base = m.slotBase() + slot.kv_offset;
        return .{ .k = base, .v = base + slot.kv_bytes / 2 };
    }

    fn convOf(m: *const Model, visit: u32, site: usize) u64 {
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        const widths = [4]usize{ kv, kv, h, h };
        var at = m.slotBase() + m.layout.at(visit).conv_offset;
        for (0..site) |i| at += 3 * widths[i] * 4;
        return at;
    }

    fn histOf(m: *const Model, visit: u32, site: usize) u64 {
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        const widths = [4]usize{ kv, kv, h, h };
        var at = m.s.hist + visit * window_rows * (2 * kv + 2 * h) * 4;
        for (0..site) |i| at += window_rows * widths[i] * 4;
        return at;
    }

    fn backupOf(m: *const Model, visit: u32, site: usize) usize {
        return m.convOf(visit, site) - (m.slotBase() + m.layout.at(0).conv_offset);
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const weights_.Weights, context: usize) !Model {
        const layout = try loop.CacheLayout.init(&w.c.text, &w.p, @intCast(context));
        var m: Model = .{
            .gpa = gpa,
            .d = d,
            .w = w,
            .context = context,
            .layout = layout,
        };
        errdefer m.deinitBuffers();
        const c = w.c.text;
        const h: usize = c.hidden;
        const q: usize = @as(usize, c.global.q) * c.global.dim;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const rows = prompt_rows;
        const rel_rows = rows * c.global.q * attention.extent;
        const local = attention.Layer.sliding(c.window, @intCast(@min(prompt_rows, context)));
        const global = attention.Layer.global(@intCast(context));
        const local_slots = attention.slotsFor(local, context);
        const global_slots = attention.slotsFor(global, context);
        const part_slots = @max(rows * c.local.q * local_slots, rows * c.global.q * global_slots);
        const hist_values = @as(usize, w.p.n_visits) * (2 * kv + 2 * h) * window_rows;
        m.s = .{
            .ids = try m.alloc(rows * 4),
            .pos = try m.alloc(rows * 4),
            .seg_of = try m.alloc(rows * 4),
            .segs = try m.alloc(@sizeOf(attention.Seg)),
            .tiles = try m.alloc(2 * tiles_cap * @sizeOf(attention.Tile)),
            .partial_a = try m.alloc(rows * h * 2),
            .partial_b = try m.alloc(rows * h * 2),
            .prelude = try m.alloc(rows * h * 2),
            .bank0 = try m.alloc(rows * h * 2),
            .bank1 = try m.alloc(rows * h * 2),
            .raw = try m.alloc(3 * rows * h * 4),
            .messages = try m.alloc(4 * rows * h * 4),
            .normalized = try m.alloc(3 * rows * h * 2),
            .gate_zeta = try m.alloc(window_rows * loop_math.rank * 2),
            .t = try m.alloc(rows * h * 4),
            .q_raw = try m.alloc(rows * q * 2),
            .rel_raw = try m.alloc(rows * c.global.q * c.d_rel * 2),
            .sink = try m.alloc(rows * c.global.q * 2),
            .gate = try m.alloc(rows * q * 2),
            .k_raw = try m.alloc(rows * kv * 2),
            .v_raw = try m.alloc(rows * kv * 2),
            .k_conv = try m.alloc(rows * kv * 2),
            .v_conv = try m.alloc(rows * kv * 2),
            .k_dynamic = try m.alloc(rows * c.global.kv * config.conv_taps * 4),
            .v_dynamic = try m.alloc(rows * c.global.kv * config.conv_taps * 4),
            .q_prepared = try m.alloc(rows * q * 2),
            .rel_prepared = try m.alloc(rows * c.global.q * c.d_rel * 2),
            .rel_table = try m.alloc(rel_rows * 2),
            .part_o = try m.alloc(part_slots * attention.head_dim * 4),
            .part_ml = try m.alloc(part_slots * 8),
            .attn_heads = try m.alloc(rows * q * 2),
            .attn_out = try m.alloc(rows * h * 2),
            .route_scores = try m.alloc(rows * c.experts * 4),
            .route_pick = try m.alloc(rows * c.experts * 4),
            .route_weights = try m.alloc(rows * c.experts * 4),
            .moe_out = try m.alloc(rows * h * 2),
            .moe_scratch = try m.alloc(try modelopt.scratchBytes(rows, c.topk, h, c.expert_inter)),
            .hidden = try m.alloc(rows * h * 2),
            .logits = try m.alloc(window_rows * c.unpadded_vocab * 4),
            .backup = try m.alloc(layout.bytes - layout.at(0).conv_offset),
            .hist = try m.alloc(hist_values * 4),
        };
        for (0..ops.max_blocks) |i| m.s.blocks[i] = try m.alloc(rows * h * 2);
        try m.addSlots(s, 1);
        return m;
    }

    fn deinitBuffers(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
    }

    pub fn deinit(m: *Model) void {
        for (&m.graphs) |*g| if (g.*) |*exec| exec.deinit();
        m.deinitBuffers();
        for (m.slots[0..m.n_slots]) |*slot| for ([2]*cuda.DeviceBuffer{ &slot.sites, &slot.cache }) |b| b.free();
        m.* = undefined;
    }

    pub fn reset(m: *const Model, s: cuda.Stream) !void {
        try m.zeros(s, m.slotBase() + m.layout.at(0).conv_offset, m.layout.bytes - m.layout.at(0).conv_offset);
    }

    pub fn rewind(m: *const Model, k: *const Kernels, s: cuda.Stream, kept: usize) !void {
        if (kept > window_rows) return error.WindowTooWide;
        if (m.recorded != m.slot) return error.RewindOfAnotherSlot;
        try ops.rewind(&k.glue, s, m.slots[m.slot].sites.ptr, m.w.p.n_visits * 4, m.w.c.text.hidden, m.s.backup, m.s.hist, 0, 0, kept);
    }

    fn arBlocks(m: *const Model, blocks: []const u64, partial: u64, query: u64, out: u64, nq: usize, partial_bf16: bool, rows: usize) loop_math.ArArgs {
        var a: loop_math.ArArgs = .{
            .partial = partial,
            .query = query,
            .out = out,
            .n_blocks = @intCast(blocks.len),
            .partial_bf16 = @intFromBool(partial_bf16),
            .nq = @intCast(nq),
            .rows = @intCast(rows),
            .c = @intCast(m.w.c.text.hidden),
            .eps = m.w.c.text.eps,
        };
        for (blocks, 0..) |b, i| a.blocks[i] = b;
        return a;
    }

    fn mixArgs(m: *const Model, blocks: []const u64, partial: u64, query: u64, norms: [3]u64, out: u64, nq: usize, partial_bf16: bool, rows: usize) ops.MixArgs {
        var a: ops.MixArgs = .{
            .partial = partial,
            .query = query,
            .out = out,
            .n_blocks = @intCast(blocks.len),
            .partial_bf16 = @intFromBool(partial_bf16),
            .nq = @intCast(nq),
            .rows = @intCast(rows),
            .c = @intCast(m.w.c.text.hidden),
            .eps = m.w.c.text.eps,
        };
        for (blocks, 0..) |b, i| a.blocks[i] = b;
        a.norm = norms;
        return a;
    }

    /// One prompt chunk or one decode window. `record` retains every convolution input for exact prefix rollback.
    /// A one-token decode replays the slot's CUDA graph of `body`; the inputs it reads are staged beforehand.
    pub fn forward(m: *Model, k: *const Kernels, s: cuda.Stream, tokens: []const u32, p0: usize, head0: usize, heads: usize, record: bool) !void {
        const c = m.w.c.text;
        const rows = tokens.len;
        if (rows == 0 or rows > prompt_rows or p0 + rows > m.context or (!record and heads > window_rows) or head0 + heads > rows)
            return error.BadLoopRows;
        if (record and rows > window_rows) return error.WindowTooWide;
        var ids: [prompt_rows]i32 = undefined;
        var pos: [prompt_rows]i32 = undefined;
        var seg_of: [prompt_rows]i32 = @splat(0);
        for (tokens, 0..) |token, i| {
            ids[i] = @intCast(token);
            pos[i] = @intCast(p0 + i);
        }
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(ids[0..rows]));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(pos[0..rows]));
        try m.upload(s, m.s.seg_of, std.mem.sliceAsBytes(seg_of[0..rows]));
        const seg: attention.Seg = .{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(p0), .off = 0 };
        try m.upload(s, m.s.segs, std.mem.asBytes(&seg));
        // Local then global layers' tiles; a call's tiles depend only on its rows and the layer's head group.
        var n_tiles: [2]usize = undefined;
        for ([2]config.Heads{ c.local, c.global }, 0..) |hd, i| {
            var tiles: [tiles_cap]attention.Tile = undefined;
            n_tiles[i] = try attention.tiles(&.{seg}, hd.q / hd.kv, &tiles);
            try m.upload(s, m.tileList(i), std.mem.sliceAsBytes(tiles[0..n_tiles[i]]));
        }
        m.recorded = if (record) m.slot else null;
        if (record) try m.copy(s, m.s.backup, m.slotBase() + m.layout.at(0).conv_offset, m.layout.bytes - m.layout.at(0).conv_offset);
        if (rows != 1 or heads != 1 or record) return m.body(k, s, rows, p0, head0, heads, record, n_tiles);
        const exec = &m.graphs[m.slot];
        if (exec.* == null) exec.* = try m.capture(k, s, p0, n_tiles);
        try exec.*.?.launchOn(s);
    }

    /// One-token decode's body as a graph: the slot's cache addresses are fixed, the rest is staged scratch.
    fn capture(m: *Model, k: *const Kernels, s: cuda.Stream, p0: usize, n_tiles: [2]usize) !cuda.graph.Exec {
        try cuda.graph.beginCapture(s, .thread_local);
        m.body(k, s, 1, p0, 0, 1, false, n_tiles) catch |err| {
            var g = cuda.graph.endCapture(s) catch return err;
            g.deinit();
            return err;
        };
        var g = try cuda.graph.endCapture(s);
        defer g.deinit();
        return g.instantiate();
    }

    fn tileList(m: *const Model, global: usize) u64 {
        return m.s.tiles + global * tiles_cap * @sizeOf(attention.Tile);
    }

    /// The forward's kernels over staged ids, positions, segments and tiles: no host copies, so a graph can hold it.
    fn body(m: *Model, k: *const Kernels, s: cuda.Stream, rows: usize, p0: usize, head0: usize, heads: usize, record: bool, n_tiles: [2]usize) !void {
        const c = m.w.c.text;
        const h: usize = c.hidden;
        try ops.embed(&k.glue, s, m.s.ids, rows, m.w.embed, m.w.embed_norm, m.s.partial_a, h, c.eps);
        var partial = m.s.partial_a;
        var next = m.s.partial_b;
        var n_blocks: usize = 0;
        const lp = c.loop.?;

        for (m.w.p.visits[0..m.w.p.n_visits]) |visit| {
            const layer = m.w.p.at(visit.layer);
            const lw = &m.w.layers[visit.layer];
            if (visit.pass >= 2 and visit.layer == lp.core_start) {
                try m.copy(s, partial, m.s.prelude, rows * h * 2);
                n_blocks = 1; // block zero is the embedding; later passes never inherit completed blocks
            }
            const messages = visit.hasMessages();
            if (messages) {
                const core_j = visit.layer - lp.core_start;
                try loop_math.messages(&k.loop, s, m.s.bank0, m.s.bank1, m.w.cross_qm + 2 * core_j * h * 4, m.w.cross_kv + 2 * core_j * h * 4, m.s.messages, rows, 1, h, c.eps);
            }
            const nq: usize = if (layer.kind == .global) 3 else 1;
            const query = m.w.attn_queries[visit.layer];
            const blocks = m.s.blocks[0..n_blocks];
            const raw_stride = rows * h * 4;
            const norm_stride = rows * h * 2;
            if (messages) {
                if (nq == 3) {
                    try loop_math.ar(&k.loop, s, m.arBlocks(blocks, partial, query, m.s.raw, 3, true, rows));
                } else {
                    // A sliding layer stores one query; each gated role gets its own AR result from that query.
                    for (0..3) |role| try loop_math.ar(&k.loop, s, m.arBlocks(blocks, partial, query, m.s.raw + role * raw_stride, 1, true, rows));
                }
                for (0..3) |role| {
                    try m.gate(k, s, lw.controllers[role], m.s.raw + role * raw_stride, m.s.messages + role * raw_stride, rows);
                    const norms = [3]u64{ lw.attn.norms[role], lw.attn.norms[role], lw.attn.norms[role] };
                    try ops.mix(&k.glue, s, m.mixArgs(&.{}, m.s.raw + role * raw_stride, 0, norms, m.s.normalized + role * norm_stride, 1, false, rows));
                }
            } else {
                try ops.mix(&k.glue, s, m.mixArgs(blocks, partial, query, lw.attn.norms, m.s.normalized, nq, true, rows));
            }
            const u_q = m.s.normalized;
            const u_k = if (layer.kind == .global or messages) m.s.normalized + norm_stride else u_q;
            const u_v = if (layer.kind == .global or messages) m.s.normalized + 2 * norm_stride else u_q;
            const q_width: usize = @as(usize, layer.heads.q) * layer.heads.dim;
            const kv_width: usize = @as(usize, layer.heads.kv) * layer.heads.dim;
            try loop_math.projectQkv(&k.loop, s, u_q, u_k, u_v, lw.attn.wq, lw.attn.wk, lw.attn.wv, m.s.q_raw, m.s.k_raw, m.s.v_raw, rows, h, q_width, kv_width);
            try ops.rowdot3(&k.glue, s, u_q, lw.attn.wr, lw.attn.gate, lw.attn.sink, m.s.rel_raw, m.s.gate, m.s.sink, rows, layer.heads.q * c.d_rel, q_width, layer.heads.q, h);

            const geometry = m.layerGeometry(visit.layer);
            const cache = m.cacheOf(visit.cache_slot);
            const dynamic = layer.kind == .global;
            for (0..2) |kind| {
                const input = if (kind == 0) u_k else u_v;
                const dyn_out = if (kind == 0) m.s.k_dynamic else m.s.v_dynamic;
                if (dynamic) {
                    try ops.rowdot(&k.glue, s, input, if (kind == 0) lw.attn.k_dynamic else lw.attn.v_dynamic, dyn_out, rows, layer.heads.kv * config.conv_taps, h, false, false);
                    try loop_math.roundBf16(&k.loop, s, dyn_out, dyn_out, rows * layer.heads.kv * config.conv_taps);
                }
                try ops.conv(&k.glue, s, .{
                    .x = if (kind == 0) m.s.k_raw else m.s.v_raw,
                    .state = m.convOf(visit.cache_slot, kind),
                    .w = if (kind == 0) lw.attn.k_conv else lw.attn.v_conv,
                    .dyn = if (dynamic) dyn_out else 0,
                    .out = if (kind == 0) m.s.k_conv else m.s.v_conv,
                    .hist = if (record) m.histOf(visit.cache_slot, kind) else 0,
                    .x_bf16 = 1,
                    .heads = @intCast(layer.heads.kv),
                    .c = @intCast(kv_width),
                    .rows = @intCast(rows),
                    .mode = 0,
                }, 1);
            }
            const global = @intFromBool(layer.kind == .global);
            const stride = attention.slotsFor(geometry, m.context);
            const part_capacity = prompt_rows * layer.heads.q * stride;
            const r: attention.Rows = .{
                .rows = @intCast(rows),
                .heads = layer.heads.q,
                .kv_heads = layer.heads.kv,
                .pos = m.s.pos,
                .seg_of = m.s.seg_of,
                .segs = m.s.segs,
                .tiles = m.tileList(global),
                .n_tiles = @intCast(n_tiles[global]),
                .last_pos = @intCast(p0 + rows - 1),
            };
            const prep: attention.Prep = .{ .qk_clip = c.qk_clip, .rel_clip = c.rel_clip, .floor = @floatFromInt(c.log_floor), .alpha = c.log_alpha, .eps = c.eps };
            try attention.prep(&k.attn, s, r, geometry, prep, .{ m.s.q_raw, m.s.k_conv, m.s.v_conv, m.s.rel_raw }, .{ lw.attn.q_scale, lw.attn.rel_scale }, m.s.q_prepared, m.s.rel_prepared, cache);
            try attention.relTable(&k.attn, s, r, m.s.rel_prepared, lw.attn.rel_proj, m.s.rel_table);
            try attention.attend(&k.attn, s, r, geometry, stride, part_capacity, m.s.q_prepared, cache, m.s.rel_table, m.s.part_o, m.s.part_ml, m.s.sink, m.s.gate, m.s.attn_heads);
            try ops.rowdot(&k.glue, s, m.s.attn_heads, lw.attn.wo, m.s.attn_out, rows, h, q_width, true, false);

            if (layer.boundary) {
                try m.copy(s, m.s.blocks[n_blocks], partial, rows * h * 2);
                n_blocks += 1;
            }
            try ops.conv(&k.glue, s, .{
                .x = m.s.attn_out,
                .state = m.convOf(visit.cache_slot, 2),
                .w = lw.attn_conv,
                .base = if (layer.boundary) 0 else partial,
                .resid = m.s.t,
                .hist = if (record) m.histOf(visit.cache_slot, 2) else 0,
                .x_bf16 = 1,
                .c = @intCast(h),
                .rows = @intCast(rows),
                .mode = 1,
            }, 1);

            const mlp_query = m.w.mlp_queries[visit.layer];
            const mlp_blocks = m.s.blocks[0..n_blocks];
            if (messages) {
                try loop_math.ar(&k.loop, s, m.arBlocks(mlp_blocks, m.s.t, mlp_query, m.s.raw, 1, false, rows));
                try m.gate(k, s, lw.controllers[3], m.s.raw, m.s.messages + 3 * raw_stride, rows);
                try ops.mix(&k.glue, s, m.mixArgs(&.{}, m.s.raw, 0, .{ lw.mlp_norm, lw.mlp_norm, lw.mlp_norm }, m.s.normalized, 1, false, rows));
            } else {
                try ops.mix(&k.glue, s, m.mixArgs(mlp_blocks, m.s.t, mlp_query, .{ lw.mlp_norm, lw.mlp_norm, lw.mlp_norm }, m.s.normalized, 1, false, rows));
            }
            try ops.rowdot(&k.glue, s, m.s.normalized, lw.router, m.s.route_scores, rows, c.experts, h, false, true);
            try ops.topk(&k.glue, s, m.s.route_scores, lw.router_bias, m.s.route_pick, m.s.route_weights, rows, c.experts, c.topk, c.route_scale);
            try modelopt.run(&k.moe, s, lw.experts, .{ .gate = c.gate_cap, .up = c.up_cap }, m.s.normalized, h, m.s.route_pick, m.s.route_weights, rows, c.topk, m.s.moe_out, m.s.moe_scratch);
            try ops.conv(&k.glue, s, .{
                .x = m.s.moe_out,
                .state = m.convOf(visit.cache_slot, 3),
                .w = lw.mlp_conv,
                .resid = m.s.t,
                .out = next,
                .hist = if (record) m.histOf(visit.cache_slot, 3) else 0,
                .x_bf16 = 1,
                .c = @intCast(h),
                .rows = @intCast(rows),
                .mode = 2,
            }, 1);
            std.mem.swap(u64, &partial, &next);

            if (visit.layer + 1 == lp.core_start) try m.copy(s, m.s.prelude, partial, rows * h * 2);
            if (visit.pass > 0 and visit.pass < lp.passes and visit.layer + 1 == lp.core_end) {
                try m.copy(s, m.s.bank0, m.s.blocks[n_blocks - 1], rows * h * 2);
                try m.copy(s, m.s.bank1, partial, rows * h * 2);
            }
        }

        if (heads == 0) return;
        try ops.mix(&k.glue, s, m.mixArgs(m.s.blocks[0..n_blocks], partial, m.w.output_query, .{ m.w.norm, m.w.norm, m.w.norm }, m.s.hidden, 1, true, rows));
        try ops.rowdot(&k.glue, s, m.s.hidden + head0 * h * 2, m.w.unembed, m.s.logits, heads, c.unpadded_vocab, h, false, false);
        try loop_math.roundBf16(&k.loop, s, m.s.logits, m.s.logits, heads * c.unpadded_vocab);
        try ops.softcap(&k.glue, s, m.s.logits, heads * c.unpadded_vocab, c.logit_cap);
    }

    /// A controller's gate in place on `x`; decode rows spread it over rank and output blocks (the same bits).
    fn gate(m: *const Model, k: *const Kernels, s: cuda.Stream, ctl: weights_.Controller, x: u64, msg: u64, rows: usize) !void {
        const c = m.w.c.text;
        if (rows <= window_rows)
            try loop_math.gateParallel(&k.loop, s, x, msg, ctl.down, ctl.forget, ctl.gain, x, m.s.gate_zeta, rows, c.hidden, c.eps)
        else
            try loop_math.gate(&k.loop, s, x, msg, ctl.down, ctl.forget, ctl.gain, x, rows, c.hidden, c.eps);
    }

    pub const Saved = snapshot.Saved;
    pub const snapshotBytes = snapshot.snapshotBytes;
    pub const save = snapshot.save;
    pub const restore = snapshot.restore;
    pub const drop = snapshot.drop;
};
