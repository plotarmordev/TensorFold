//! Inkling 2 Mini DFlash2: the shared block-diffusion graph over this target's BF16 head and visit taps.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const runtime = @import("../../qwen27/dflash/runtime_model.zig");
const selector = @import("../../qwen27/dflash/selector.zig");
const df_config = @import("../../qwen27/dflash/config.zig");
const target_binding = @import("../../qwen27/dflash/runtime_target.zig");
const qwen_glue = @import("../../qwen27/glue_gpu.zig");
const inkling_glue = @import("glue.zig");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const Stream = lanes.Stream;
const dr = lanes.drafter;
const draft_prior = [_]f64{ 0.65, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15 };

const Lane = struct {
    context: *runtime.Context,
    held: [7]u32 = undefined,
    held_len: usize = 0,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    glue: qwen_glue.Kernels,
    model: *runtime.Model,
    lanes: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    step_ms: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *Model, dir: []const u8) !*Drafter {
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.gpa = gpa;
        d.lanes = .empty;
        d.step_ms = 0;
        d.glue = try qwen_glue.Kernels.init(target.device);
        errdefer d.glue.deinit();
        const binding: target_binding.Target = .{
            .device = target.device, .queue = target.queue, .checkpoint = &target.checkpoint,
            .config = .{ .hidden = st.hidden, .vocab = target.config.text.vocab, .layers = 48 },
            .weights = undefined, .kernels = undefined, .glue = &d.glue, .dims = undefined,
            .dense = .{ .embed = target.weights.embed, .head = target.weights.unembed,
                .embedding = .{ .ptr = target, .run = embedNorm } },
        };
        d.model = try runtime.Model.loadBound(gpa, io, binding, dir, .bf16_reference);
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        var it = d.lanes.valueIterator();
        while (it.next()) |lane| lane.context.deinit(d.gpa);
        d.lanes.deinit(d.gpa);
        d.model.deinit();
        d.glue.deinit();
        d.gpa.destroy(d);
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        const c = d.model.graph.config;
        return d.model.checkpoint.residentBytes() +
            @as(u64, c.span()) * c.ringWidth() * 2 + contextCacheBytes(c);
    }

    pub fn streamBytes(d: *const Drafter) u64 {
        const c = d.model.graph.config;
        return @as(u64, c.span()) * c.ringWidth() * 2 + contextCacheBytes(c);
    }

    fn contextCacheBytes(c: df_config.Config) u64 {
        var total: u64 = 0;
        for (0..c.layers) |i| total += @as(u64, c.layerWindow(i) + 16) * c.kvWidth() * 2 * 2;
        return total;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = d, .vtable = &.{
            .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn,
            .hold = holdFn, .held = heldFn, .release = releaseFn, .clear_proposals = clearFn,
        } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }
    fn embedNorm(ptr: *const anyopaque, e: mtl.ComputeEncoder, ids: mtl.Buffer, ids_off: usize, out: mtl.Buffer, out_off: usize, rows: u32) void {
        const target: *const Model = @ptrCast(@alignCast(ptr));
        inkling_glue.embedNorm(&target.glue, e,
            .{ .buf = ids, .off = ids_off },
            .{ .buf = target.weights.embed.buffer, .off = target.weights.embed.offset },
            .{ .buf = target.weights.embed_norm.buffer, .off = target.weights.embed_norm.offset },
            .{ .buf = out, .off = out_off }, rows);
    }
    fn tapsFn(ptr: *anyopaque) []const u32 {
        return cast(ptr).model.graph.config.tapIds();
    }
    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = 7, .prior = &draft_prior, .fixed_block = true, .plain_guard = false,
            .batched = true, .step_ms = d.step_ms, .stream_bytes = @intCast(d.streamBytes()) };
    }
    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(stream)) return error.DFlashStreamOpen;
        const context = try runtime.Context.init(d.gpa, d.model, true);
        errdefer context.deinit(d.gpa);
        try d.lanes.put(d.gpa, stream, .{ .context = context });
    }
    fn releaseFn(ptr: *anyopaque, stream: *Stream) void {
        const d = cast(ptr);
        if (d.lanes.fetchRemove(stream)) |entry| entry.value.context.deinit(d.gpa);
    }
    fn clearFn(ptr: *anyopaque, stream: *Stream) void {
        if (cast(ptr).lanes.getPtr(stream)) |lane| lane.held_len = 0;
    }
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const c = d.model.graph.config;
        for (items) |item| {
            const lane = d.lanes.getPtr(item.stream) orelse return error.DFlashStreamUnknown;
            const f = item.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != st.feature_row_bytes or
                f.rows == 0 or item.rows != null or item.follow.len != f.rows) return error.DFlashFeaturesUnsupported;
            if (item.start != lane.context.committed_end) return error.DFlashContextGap;
            const source = mtl.Buffer{ .id = @ptrFromInt(f.buffer) };
            try lane.context.ring(c).absorb(source, f.offset, f.rows);
            lane.held_len = 0;
        }
    }
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        var at: usize = 0;
        while (at < items.len) {
            const n = @min(items.len - at, 8);
            const batch = items[at .. at + n];
            var requests: [8]runtime.Model.Proposal = undefined;
            var trees: [8]selector.Tree = undefined;
            for (batch, requests[0..n]) |item, *req| {
                const lane = d.lanes.getPtr(item.stream) orelse return error.DFlashStreamUnknown;
                if (item.depth == 0 or item.depth > 7 or item.position == 0 or
                    item.position - 1 != lane.context.committed_end or item.lanes != null)
                    return error.DFlashHoldGeometry;
                req.* = .{ .context = lane.context, .pending = item.pending, .nodes = item.depth, .chain = true };
            }
            if (n > 1) {
                try d.model.proposeBatch(requests[0..n], trees[0..n]);
            } else {
                const context = requests[0].context.?;
                try d.model.swap(context);
                const result = d.model.proposeOptions(batch[0].pending, runtime.Model.selection(requests[0])) catch |err| {
                    try d.model.swap(context);
                    return err;
                };
                d.model.swap(context) catch |err| {
                    var lost = result;
                    lost.deinit();
                    return err;
                };
                trees[0] = result;
            }
            for (batch, trees[0..n]) |item, tree| {
                defer {
                    var owned = tree;
                    owned.deinit();
                }
                const lane = d.lanes.getPtr(item.stream).?;
                if (tree.tokens.len < item.depth) return error.DFlashIncompleteTree;
                @memcpy(lane.held[0..item.depth], tree.tokens[0..item.depth]);
                lane.held_len = item.depth;
            }
            at += n;
        }
    }
    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        if (streams.len != out.len) return error.DFlashHeldShape;
        for (streams, out) |stream, tokens| {
            const lane = d.lanes.getPtr(stream) orelse return error.DFlashStreamUnknown;
            if (tokens.len > lane.held_len) return error.DFlashHeldShape;
            @memcpy(tokens, lane.held[0..tokens.len]);
        }
    }
};
