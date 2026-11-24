//! Inkling 2 Mini DSpark: committed target features, three-layer proposal blocks, folded Markov logits.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const gpu_rule = lanes.gpu_rule;
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;
const fwd = @import("forward.zig");
const glue = @import("glue.zig");
const k = @import("dspark_kernels.zig");
const weight_loader = @import("dspark_weights.zig");
const state = @import("dspark_state.zig");
const st = @import("state.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const tap_list = [_]u32{ 5, 23, 41, 48 };
/// Drafts a hold offers at most: the block after its anchor. The depth rule prices how many a round takes.
const hold_depth: u32 = state.block - 1;

fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
fn failed(cb: mtl.CommandBuffer, what: []const u8) !void {
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop DSpark {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *const Model,
    weights: weight_loader.Loaded,
    pipelines: k.Pipelines,
    scratch: state.Scratch,
    /// Scratch for the second and later streams of a sampled hold; the first uses `scratch`, as absorb does.
    more: std.ArrayList(state.Scratch) = .empty,
    /// The solo timing kernels and eight held tokens per cohort slot.
    chain: k.Chain = undefined,
    chain_tokens: mtl.Buffer = undefined,
    /// Shared greedy block rows, packed unembed inputs and base rows.
    cohort: k.Cohort = undefined,
    cohort_state: state.Cohort = undefined,
    scores: ?mtl.Buffer = null,
    /// The block attention's tensor-unit kernels, on the chips that have the units.
    units: ?k.Units = null,
    capacity: usize = 1,
    /// A draft's share of a full hold, timed by `measure`.
    step_ms: f64 = 0,
    contexts: std.AutoHashMapUnmanaged(*Stream, state.Lane) = .empty,
    /// The previous verify's keep and the following absorb and hold share a submission.
    pending: std.ArrayListUnmanaged(dr.Absorb) = .empty,
    keeps: [st.batch_rows]*st.Cache = undefined,
    keep_streams: [st.batch_rows]*Stream = undefined,
    accepted: [st.batch_rows]usize = undefined,
    keep_len: usize = 0,
    holds: [k.cohort_streams]PendHold = undefined,
    depths: [k.cohort_streams]u32 = undefined,
    slots: [k.cohort_streams]u32 = undefined,
    hold_order: k.Order = undefined,
    held_len: usize = 0,

    /// A greedy hold's inputs, encoded after its keeps and absorbs.
    const PendHold = struct { stream: *Stream, pending: u32, position: u64, depth: u32 };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, target: *const Model, drafter_dir: []const u8) !*Drafter {
        _ = io; // Native factory convention; Checkpoint.addFile uses platform file calls.
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{
            .gpa = gpa,
            .target = target,
            .weights = weight_loader.load(gpa, target, drafter_dir) catch |err| return err,
            .pipelines = undefined,
            .scratch = undefined,
        };
        errdefer d.weights.deinit();
        d.pipelines = try k.Pipelines.load(device);
        errdefer d.pipelines.deinit();
        d.scratch = try state.Scratch.init(device);
        errdefer d.scratch.deinit();
        const lib = try mtl.Library.fromSource(device, @import("inkling2_metal").inkling2_dspark, mtl.CompileOptions.mlx());
        defer lib.deinit();
        d.chain = try k.Chain.fromLibrary(device, lib);
        errdefer d.chain.deinit();
        d.cohort = try k.Cohort.fromLibrary(device, lib);
        errdefer d.cohort.deinit();
        d.cohort_state = try state.Cohort.init(device);
        errdefer d.cohort_state.deinit();
        d.chain_tokens = try device.buffer(k.cohort_streams * 8 * @sizeOf(u32), mtl.ResourceOptions.shared);
        errdefer d.chain_tokens.deinit();
        if (device.tensorUnits()) d.units = try k.Units.load(device, @import("inkling2_metal").inkling2_dspark_units);
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        d.pending.deinit(d.gpa);
        if (d.scores) |scores| scores.deinit();
        if (d.units) |*u| u.deinit();
        d.cohort_state.deinit();
        d.cohort.deinit();
        d.chain_tokens.deinit();
        d.chain.deinit();
        for (d.more.items) |*extra| extra.deinit();
        d.more.deinit(d.gpa);
        d.scratch.deinit();
        d.pipelines.deinit();
        d.weights.deinit();
        d.gpa.destroy(d);
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        return d.weights.fixedBytes() + state.scratchBytes() + state.cohortBytes();
    }

    pub fn scoreBytes(capacity: usize) u64 {
        return state.scoreBytes(capacity);
    }

    pub fn streamBytes(capacity: usize) u64 {
        return state.laneBytes(capacity);
    }

    /// A stream's bytes a token of its window; its global rings round the window up by at most `ring_slack` tokens.
    pub fn tokenBytes() u64 {
        return state.laneTokenBytes();
    }
    pub const ring_slack = state.global_step - 1;

    /// The selected target window fixes every new lane's global ring and the shared attention score allocation.
    pub fn setContext(d: *Drafter, window: u32) !void {
        const capacity: usize = @intCast(window);
        if (capacity == 0 or capacity > 1048576) return error.DSparkLaneGeometry;
        if (d.contexts.count() != 0) return error.DSparkContextActive;
        const scores = try d.target.device.buffer(@intCast(state.scoreBytes(capacity)), mtl.ResourceOptions.private);
        if (d.scores) |old| old.deinit();
        d.scores = scores;
        d.capacity = capacity;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    pub fn saveSnap(d: *Drafter, stream: *Stream, at_pos: u32, last_tap: ?mtl.Buffer, last_tap_start: u64, last_tap_rows: usize) !?state.Snap {
        const lane = d.contexts.getPtr(stream) orelse return null;
        if (lane.end == at_pos) return try state.Snap.save(d.target.device, d.target.queue, lane, null);
        if (lane.end + 1 == at_pos) {
            const tap = last_tap orelse return error.SnapshotOutOfStep;
            if (last_tap_rows != 1 or last_tap_start != lane.end) return error.SnapshotOutOfStep;
            return try state.Snap.save(d.target.device, d.target.queue, lane, tap);
        }
        return error.SnapshotOutOfStep;
    }

    pub fn restoreSnap(d: *Drafter, stream: *Stream, snap: ?*const state.Snap, at_pos: usize) !void {
        const lane = d.contexts.getPtr(stream) orelse return;
        const s = snap orelse return error.DSparkSnapshotMissing;
        const expected_end = s.end + @as(u64, if (s.tail_off != null) 1 else 0);
        if (expected_end != at_pos) return error.DSparkSnapshotGeometry;
        try state.Snap.restore(lane, d.target.queue, s);
        if (s.tail_off) |tail_off| {
            errdefer {
                lane.from = 0;
                lane.end = 0;
                lane.held_len = 0;
            }
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.absorbRows(e, lane, at(s.buf, tail_off), s.end, 1);
            e.end();
            cb.commit();
            try failed(cb, "restore tail");
            lane.end = s.end + 1;
            lane.held_len = 0;
        }
    }

    pub fn resetLane(d: *Drafter, stream: *Stream) void {
        d.dropPending(stream);
        if (d.contexts.getPtr(stream)) |lane| {
            lane.from = 0;
            lane.end = 0;
            lane.held_len = 0;
        }
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &tap_list;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        // No plain rounds: one draft pays from about a third landing, and the guard's estimate near that line is noise.
        return .{ .depth = hold_depth, .step_ms = d.step_ms, .plain_guard = false, .batched = true, .stream_bytes = @intCast(state.laneBytes(d.capacity)) };
    }

    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.contexts.contains(stream)) return error.DraftStreamOpen;
        const lane = try state.Lane.init(d.target.device, d.capacity);
        errdefer {
            var owned = lane;
            owned.deinit();
        }
        try d.contexts.put(d.gpa, stream, lane);
    }

    fn releaseFn(ptr: *anyopaque, stream: *Stream) void {
        const d = cast(ptr);
        d.dropPending(stream);
        if (d.contexts.fetchRemove(stream)) |entry| {
            var lane = entry.value;
            lane.deinit();
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        try d.flushPending();
        for (streams, out) |stream, tokens| {
            const lane = d.contexts.getPtr(stream) orelse return error.DraftStreamUnknown;
            if (tokens.len > lane.held_len) return error.DraftHeldFewer;
            @memcpy(tokens, lane.held[0..tokens.len]);
        }
    }

    fn absorbRows(d: *Drafter, e: mtl.ComputeEncoder, lane: *const state.Lane, taps: Ref, start: u64, rows: usize) void {
        const n: u32 = @intCast(rows);
        k.fuseTaps(&d.pipelines, e, taps, r(d.weights.weights.fuse), at(d.weights.dflare, 0), at(d.scratch.fused, 0), at(lane.newest, 0), n);
        for (d.weights.weights.layers, 0..) |layer, i| {
            const c = at(d.scratch.fused, i * rows * state.hidden * 2);
            fwd.dense(d.target, e, layer.k_ctx, c, at(d.scratch.ck, 0), rows);
            fwd.dense(d.target, e, layer.v_ctx, c, at(d.scratch.cv, 0), rows);
            k.commitContext(&d.pipelines, e, at(d.scratch.ck, 0), at(d.scratch.cv, 0), r(layer.k_norm_ctx), at(lane.k[i], 0), at(lane.v[i], 0), @intCast(start), n, @intCast(lane.rings[i]));
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        if (items.len == 0) return;
        if (d.pending.items.len > 0 or d.held_len > 0) try d.flushPending();
        const row_bytes = 4 * state.hidden * 2;
        var prompt = false;
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const features = item.features;
            if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (features.rows == 0 or features.rows > 1048576) return error.DraftFeaturesUnsupported;
            if (item.start < lane.end) return error.DraftContextGap;
            if (item.start > lane.end) lane.from = item.start;
            const end = item.start + features.rows;
            if (end > 1048576) return error.DraftContextGap;
            lane.end = end;
            lane.held_len = 0;
            prompt = prompt or item.prompt;
        }
        // A prompt's rows are absorbed before the round drafting from them; a round's live rows ride its keep.
        if (!prompt) return d.pending.appendSlice(d.gpa, items);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (items) |item| d.encodeAbsorb(e, item);
        e.end();
        cb.commit();
        try failed(cb, "absorb");
    }

    fn encodeAbsorb(d: *Drafter, e: mtl.ComputeEncoder, item: dr.Absorb) void {
        const lane = d.contexts.getPtr(item.stream).?;
        const source = mtl.Buffer{ .id = @ptrFromInt(item.features.buffer) };
        const row_bytes = 4 * state.hidden * 2;
        const end = item.start + item.features.rows;
        var position = item.start;
        while (position < end) {
            const rows: usize = @intCast(@min(state.absorb_rows, end - position));
            const offset = item.features.offset + @as(usize, @intCast(position - item.start)) * row_bytes;
            d.absorbRows(e, lane, at(source, offset), position, rows);
            position += rows;
        }
    }

    fn flushPending(d: *Drafter) !void {
        if (d.keep_len == 0 and d.pending.items.len == 0 and d.held_len == 0) return;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        if (d.keep_len > 0) {
            const b = cb.blit();
            st.encodeKeep(b, d.keeps[0..d.keep_len], d.accepted[0..d.keep_len]);
            b.end();
        }
        if (d.pending.items.len > 0 or d.held_len > 0) {
            const e = cb.compute(.serial);
            for (d.pending.items) |item| d.encodeAbsorb(e, item);
            if (d.held_len > 0) d.encodeHolds(e);
            e.end();
        }
        cb.commit();
        try failed(cb, "draft");
        d.keep_len = 0;
        d.pending.clearRetainingCapacity();
        try d.finishHolds();
    }

    /// Keep bookkeeping is visible to the draft's features; the GPU copies precede its absorb and hold.
    pub fn stageKeep(d: *Drafter, streams: []const *Stream, caches: []const *st.Cache, accepted: []const usize) !void {
        try d.flushPending();
        @memcpy(d.keeps[0..caches.len], caches);
        @memcpy(d.keep_streams[0..streams.len], streams);
        @memcpy(d.accepted[0..accepted.len], accepted);
        d.keep_len = caches.len;
        st.settleKeep(caches, accepted);
    }

    /// Prompt forwards reuse the verify's tap buffer, so they finish its absorb first.
    pub fn beforeForward(d: *Drafter) !void {
        try d.flushPending();
    }

    /// The drafts a stashed hold wrote, read back once its submission has run.
    fn finishHolds(d: *Drafter) !void {
        const tokens = d.chain_tokens.slice(u32, k.cohort_streams * 8);
        for (d.holds[0..d.held_len], d.depths[0..d.held_len], d.slots[0..d.held_len]) |item, depth, at_slot| {
            const lane = d.contexts.getPtr(item.stream) orelse continue;
            for (0..depth) |step| {
                const token = tokens[at_slot * 8 + step + 1];
                if (token >= state.vocabulary) return error.DraftNonfinite;
                lane.held[step] = token;
            }
            lane.held_len = depth;
        }
        d.held_len = 0;
    }

    /// A stream leaving the rounds takes its stashed work with it.
    fn dropPending(d: *Drafter, stream: *Stream) void {
        var kept: usize = 0;
        for (d.pending.items) |item| {
            if (item.stream == stream) continue;
            d.pending.items[kept] = item;
            kept += 1;
        }
        d.pending.shrinkRetainingCapacity(kept);
        var n: usize = 0;
        for (d.holds[0..d.held_len], 0..) |item, i| {
            if (item.stream == stream) continue;
            d.holds[n] = item;
            d.depths[n] = d.depths[i];
            n += 1;
        }
        d.held_len = n;
        if (n == 0) return;
        d.hold_order = k.Order.sort(d.depths[0..n]);
        for (d.hold_order.slots[0..n], 0..) |item, s| d.slots[item] = @intCast(s);
    }

    /// The target frees a stream's cache: its staged keep goes first, whether or not the stream drafted.
    pub fn dropKeep(d: *Drafter, stream: *Stream) void {
        var keep: usize = 0;
        for (d.keep_streams[0..d.keep_len], d.keeps[0..d.keep_len], d.accepted[0..d.keep_len]) |s, cache, accepted| {
            if (s == stream) continue;
            d.keep_streams[keep] = s;
            d.keeps[keep] = cache;
            d.accepted[keep] = accepted;
            keep += 1;
        }
        d.keep_len = keep;
    }

    fn encodeBlock(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, lane: *const state.Lane, pending: u32, anchor: u64, depth: u32) void {
        s.ids.slice(u32, 1)[0] = pending;
        glue.embedNorm(&d.target.glue, e, at(s.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(s.x, 0), 1);
        k.initializeRows(&d.pipelines, e, at(s.x, 0), r(d.weights.weights.mask), at(s.x, 0));
        k.residualNorm(&d.pipelines, e, at(s.x, 0), at(s.resid, 0), r(d.weights.weights.layers[0].input_norm), at(s.resid, 0), at(s.x, 0), 8, true);
        for (d.weights.weights.layers, 0..) |layer, i| {
            fwd.dense(d.target, e, layer.q, at(s.x, 0), at(s.q_raw, 0), 8);
            fwd.dense(d.target, e, layer.k, at(s.x, 0), at(s.k_raw, 0), 8);
            fwd.dense(d.target, e, layer.v, at(s.x, 0), at(s.v_raw, 0), 8);
            k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), @intCast(anchor));
            if (d.units) |*u| k.attendUnits(u, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(d.scores.?, 0), at(s.attn, 0), @intCast(lane.end), @intCast(lane.rings[i]), i != 1, @intCast(lane.from)) else k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(d.scores.?, 0), at(s.attn, 0), @intCast(lane.end), @intCast(lane.rings[i]), i != 1, @intCast(lane.from));
            fwd.dense(d.target, e, layer.o, at(s.attn, 0), at(s.y, 0), 8);
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.resid, 0), r(layer.post_norm), at(s.next_resid, 0), at(s.x, 0), 8, false);
            fwd.dense(d.target, e, layer.gate, at(s.x, 0), at(s.gate, 0), 8);
            fwd.dense(d.target, e, layer.up, at(s.x, 0), at(s.up, 0), 8);
            k.swiglu(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), 8 * state.intermediate);
            fwd.dense(d.target, e, layer.down, at(s.act, 0), at(s.y, 0), 8);
            const next_norm = if (i + 1 < d.weights.weights.layers.len) d.weights.weights.layers[i + 1].input_norm else d.weights.weights.norm;
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.next_resid, 0), r(next_norm), at(s.resid, 0), at(s.x, 0), 8, false);
        }
        k.base(&d.pipelines, e, r(d.target.weights.unembed), at(s.x, state.hidden * 2), at(s.base, 0), depth, state.vocabulary);
        k.kappa(&d.pipelines, e, at(s.x, state.hidden * 2), at(lane.newest, 0), r(d.weights.weights.rnn), r(d.weights.weights.rnn_bias), at(s.kappa, 0), depth);
    }

    fn recurrent(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, step: usize, previous: u32, second_bank: bool) void {
        const current = if (second_bank) s.next_state else s.state;
        const next = if (second_bank) s.state else s.next_state;
        k.recurrentStep(&d.pipelines, e, at(current, 0), at(s.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(s.z, 0), at(next, 0), at(s.lambda, 0), previous);
        k.finalLogits(&d.pipelines, e, at(s.base, step * state.vocabulary * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0));
    }

    fn slot(d: *Drafter, index: usize) !*const state.Scratch {
        if (index == 0) return &d.scratch;
        while (d.more.items.len < index) {
            var extra = try state.Scratch.init(d.target.device);
            errdefer extra.deinit();
            try d.more.append(d.gpa, extra);
        }
        return &d.more.items[index - 1];
    }

    /// One stream's greedy hold: its block, then each step's recurrent pass, logits and pick; `held` takes the drafts.
    fn encodeChain(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, lane: *const state.Lane, held: Ref, pending: u32, anchor: u64, depth: usize) void {
        d.encodeBlock(e, s, lane, pending, anchor, @intCast(depth));
        var second_bank = false;
        for (0..depth) |step| {
            const current = if (second_bank) s.next_state else s.state;
            const next = if (second_bank) s.state else s.next_state;
            k.chainStep(&d.chain, e, at(current, 0), at(s.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(s.z, 0), at(next, 0), at(s.lambda, 0), held, @intCast(step));
            k.finalLogits(&d.pipelines, e, at(s.base, step * state.vocabulary * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0));
            k.chainPick(&d.chain, e, at(s.logits, 0), held, @intCast(step));
            second_bank = !second_bank;
        }
    }

    /// A full hold on an empty context, the third of three runs timed: the depth rule prices a draft at its share.
    pub fn measure(d: *Drafter) !void {
        var lane = try state.Lane.init(d.target.device, d.capacity);
        defer lane.deinit();
        const held = Ref{ .buf = d.chain_tokens, .off = 0 };
        var ms: f64 = 0;
        for (0..3) |attempt| {
            var t0: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t0);
            @memset(d.scratch.state.slice(u16, state.rank), 0);
            d.chain_tokens.slice(u32, 8)[0] = 2;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.encodeChain(e, &d.scratch, &lane, held, 2, 0, hold_depth);
            e.end();
            cb.commit();
            try failed(cb, "timed hold");
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t1);
            if (attempt == 2) ms = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(hold_depth));
    }

    /// Every stream's block in the cohort's launches, then its kappa and the drafts' packed unembed rows.
    fn encodeCohort(d: *Drafter, e: mtl.ComputeEncoder, items: []const PendHold, order: *const k.Order) void {
        const c = &d.cohort_state;
        const p = &d.pipelines;
        const streams: u32 = @intCast(order.active[0]);
        const slots = order.slots[0..streams];
        const rows = streams * state.block;
        const anchors = c.anchors.slice(u32, k.cohort_streams);
        const tokens = d.chain_tokens.slice(u32, k.cohort_streams * 8);
        for (slots, 0..) |item, s| {
            anchors[s] = @intCast(items[item].position - 1);
            c.ids.slice(u32, k.cohort_streams)[s] = items[item].pending;
            tokens[s * 8] = items[item].pending;
        }
        @memset(c.state.slice(u16, k.cohort_streams * state.rank), 0);
        glue.embedNorm(&d.target.glue, e, at(c.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(c.tile, 0), streams);
        k.cohortInitializeRows(p, e, at(c.tile, 0), r(d.weights.weights.mask), at(c.x, 0), streams);
        k.residualNorm(p, e, at(c.x, 0), at(c.resid, 0), r(d.weights.weights.layers[0].input_norm), at(c.resid, 0), at(c.x, 0), rows, true);
        for (d.weights.weights.layers, 0..) |layer, i| {
            fwd.dense(d.target, e, layer.q, at(c.x, 0), at(c.q_raw, 0), rows);
            fwd.dense(d.target, e, layer.k, at(c.x, 0), at(c.k_raw, 0), rows);
            fwd.dense(d.target, e, layer.v, at(c.x, 0), at(c.v_raw, 0), rows);
            k.cohortQkv(&d.cohort, e, at(c.q_raw, 0), at(c.k_raw, 0), at(c.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(c.q, 0), at(c.kb, 0), at(c.vb, 0), at(c.anchors, 0), streams);
            for (slots, 0..) |item, s| {
                const lane = d.contexts.getPtr(items[item].stream).?;
                const q = at(c.q, s * state.block * state.query * 2);
                const kb = at(c.kb, s * state.block * state.kv * 2);
                const vb = at(c.vb, s * state.block * state.kv * 2);
                const attn = at(c.attn, s * state.block * state.query * 2);
                if (d.units) |*u| k.attendUnits(u, e, q, kb, vb, at(lane.k[i], 0), at(lane.v[i], 0), at(d.scores.?, 0), attn, @intCast(lane.end), @intCast(lane.rings[i]), i != 1, @intCast(lane.from)) else k.attend(p, e, q, kb, vb, at(lane.k[i], 0), at(lane.v[i], 0), at(d.scores.?, 0), attn, @intCast(lane.end), @intCast(lane.rings[i]), i != 1, @intCast(lane.from));
            }
            fwd.dense(d.target, e, layer.o, at(c.attn, 0), at(c.y, 0), rows);
            k.residualNorm(p, e, at(c.y, 0), at(c.resid, 0), r(layer.post_norm), at(c.next_resid, 0), at(c.x, 0), rows, false);
            fwd.dense(d.target, e, layer.gate, at(c.x, 0), at(c.gate, 0), rows);
            fwd.dense(d.target, e, layer.up, at(c.x, 0), at(c.up, 0), rows);
            k.swiglu(p, e, at(c.gate, 0), at(c.up, 0), at(c.act, 0), rows * state.intermediate);
            fwd.dense(d.target, e, layer.down, at(c.act, 0), at(c.y, 0), rows);
            const next_norm = if (i + 1 < d.weights.weights.layers.len) d.weights.weights.layers[i + 1].input_norm else d.weights.weights.norm;
            k.residualNorm(p, e, at(c.y, 0), at(c.next_resid, 0), r(next_norm), at(c.resid, 0), at(c.x, 0), rows, false);
        }
        for (slots, 0..) |item, s| {
            const lane = d.contexts.getPtr(items[item].stream).?;
            const y = at(c.x, (s * state.block + 1) * state.hidden * 2);
            const out = at(c.kappa, s * (state.block - 1) * 3 * state.rank * 4);
            k.kappa(p, e, y, at(lane.newest, 0), r(d.weights.weights.rnn), r(d.weights.weights.rnn_bias), out, items[item].depth);
        }
        var first: usize = 0;
        while (first < order.rows) {
            const count: u32 = @intCast(@min(@as(usize, k.tile_rows), order.rows - first));
            const args = k.PackArgs{ .active = order.active, .first = @intCast(first), .count = count };
            k.cohortPack(&d.cohort, e, at(c.x, 0), at(c.tile, 0), args);
            k.base(p, e, r(d.target.weights.unembed), at(c.tile, 0), at(c.base, first * state.vocabulary * 2), count, state.vocabulary);
            first += count;
        }
    }

    /// Share block and unembed launches across greedy streams. A lone stream skips the cohort's packing launches;
    /// both forms read the absorbed rows after the staged keep's blits in their shared submission.
    fn stashHolds(d: *Drafter, items: []const dr.Hold) !void {
        if (d.held_len > 0) try d.flushPending();
        d.held_len = items.len;
        for (items, 0..) |item, i| {
            const depth: u32 = @min(item.depth, hold_depth);
            d.depths[i] = depth;
            d.holds[i] = .{ .stream = item.stream, .pending = item.pending, .position = item.position, .depth = depth };
        }
        d.hold_order = k.Order.sort(d.depths[0..items.len]);
        for (d.hold_order.slots[0..items.len], 0..) |item, s| d.slots[item] = @intCast(s);
    }

    fn encodeHolds(d: *Drafter, e: mtl.ComputeEncoder) void {
        const items = d.holds[0..d.held_len];
        if (items.len == 1) {
            @memset(d.scratch.state.slice(u16, state.rank), 0);
            d.chain_tokens.slice(u32, 8)[0] = items[0].pending;
            const lane = d.contexts.getPtr(items[0].stream).?;
            d.encodeChain(e, &d.scratch, lane, .{ .buf = d.chain_tokens, .off = 0 }, items[0].pending, items[0].position - 1, items[0].depth);
        } else d.encodeSteps(e, items, &d.hold_order);
    }

    /// The cohort's shared block and unembed launches, then one step, logits and pick launch set per recurrent depth.
    fn encodeSteps(d: *Drafter, e: mtl.ComputeEncoder, items: []const PendHold, order: *const k.Order) void {
        d.encodeCohort(e, items, order);
        const c = &d.cohort_state;
        const held = Ref{ .buf = d.chain_tokens, .off = 0 };
        for (order.active, 0..) |streams, step| {
            if (streams == 0) break;
            const current = if (step % 2 == 0) c.state else c.next_state;
            const next = if (step % 2 == 0) c.next_state else c.state;
            k.cohortStep(&d.cohort, e, at(current, 0), at(c.kappa, 0), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(c.z, 0), at(next, 0), at(c.lambda, 0), held, @intCast(step), streams);
            k.cohortLogits(&d.cohort, e, at(c.base, 0), at(d.weights.phi, 0), at(d.weights.beta, 0), at(c.lambda, 0), at(c.logits, 0), @intCast(order.first[step]), streams);
            k.cohortPick(&d.cohort, e, at(c.logits, 0), held, @intCast(step), streams);
        }
    }

    /// A greedy hold of up to `cohort_streams` streams runs step-major, all depths in one command buffer. A sampled or
    /// larger hold encodes one command buffer per depth for all its streams, with each block in the first.
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        if (items.len == 0) return;
        var deepest: usize = 0;
        var greedy = items.len <= k.cohort_streams;
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            if (lane.end != item.position - 1) return error.DraftContextGap;
            deepest = @max(deepest, @min(item.depth, 7));
            greedy = greedy and item.sampling == null;
        }
        if (greedy) return d.stashHolds(items);
        try d.flushPending();
        for (items, 0..) |_, i| @memset((try d.slot(i)).state.slice(u16, state.rank), 0);
        const previous = try d.gpa.alloc(u32, items.len);
        defer d.gpa.free(previous);
        const values = try d.gpa.alloc(f32, state.vocabulary);
        defer d.gpa.free(values);
        var second_bank = false;
        for (0..deepest) |step| {
            {
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = d.target.queue.commandBuffer();
                const e = cb.compute(.serial);
                for (items, 0..) |item, i| {
                    const depth: usize = @min(item.depth, 7);
                    if (step >= depth) continue;
                    const s = try d.slot(i);
                    if (step == 0) {
                        const lane = d.contexts.getPtr(item.stream).?;
                        d.encodeBlock(e, s, lane, item.pending, item.position - 1, @intCast(depth));
                        previous[i] = item.pending;
                    }
                    d.recurrent(e, s, step, previous[i], second_bank);
                }
                e.end();
                cb.commit();
                try failed(cb, if (step == 0) "block" else "recurrent step");
            }
            for (items, 0..) |item, i| {
                const depth: usize = @min(item.depth, 7);
                if (step >= depth) continue;
                const lane = d.contexts.getPtr(item.stream).?;
                const raw = (try d.slot(i)).logits.slice(u16, state.vocabulary);
                for (raw, values) |bits, *value| value.* = @bitCast(@as(u32, bits) << 16);
                const base_position = std.math.cast(u32, item.position) orelse return error.DraftDepth;
                const position = std.math.add(u32, base_position, @intCast(step)) catch return error.DraftDepth;
                previous[i] = try gpu_rule.sample(d.gpa, values, item.sampling, position, null);
                if (previous[i] >= state.vocabulary) return error.DraftNonfinite;
                lane.held[step] = previous[i];
                if (step + 1 == depth) lane.held_len = @intCast(depth);
            }
            second_bank = !second_bank;
        }
    }
};

test "Mini taps and block depth match checkpoint section eight" {
    try std.testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, &tap_list);
    try std.testing.expectEqual(@as(u32, 48), tap_list[3]);
    try std.testing.expectEqual(@as(u64, 768 * 1032), Drafter.scoreBytes(1024));
}
