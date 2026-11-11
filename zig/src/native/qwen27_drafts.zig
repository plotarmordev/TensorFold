//! The 27B lane backend's drafts: each drafting slot's tree, up to eight in one drafter pass, or the MTP layer's one
//! draft a drafting stream's round, lone or shared; and the held trees.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const be = lanes.backend;
const lb = @import("qwen27_lanes.zig");
const Metal = lb.Metal;
const Tree = q.dflash.selector.Tree;
const Proposal = q.dflash.runtime_model.Model.Proposal;

/// Only slots that currently draft compete for rows in the next shared allocation.
fn live(b: *const Metal) usize {
    var n: usize = 0;
    for (b.owners[0..b.runner.slots]) |owner| n += @intFromBool(owner != null);
    return n;
}

pub fn capacityFn(ptr: *anyopaque, s: *lanes.Stream) u32 {
    const b = Metal.self(ptr);
    const slot = b.slotOf(s) orelse return 0;
    return if (b.draftsOwn(slot)) @import("qwen27_window.zig").proposal(b.window, b.copies, b.nodes, b.window + 1, live(b) == 1) else 0;
}

/// Each drafting slot proposes its next tree, up to eight in one drafter pass; a shared round's kept taps first.
pub fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const b = Metal.self(ptr);
    errdefer mtl.pass_metrics.end();
    if (b.mtp) |d| return mtpDrafts(b, d, requests);
    @import("qwen27_slots.zig").promoteDue(b); // a stream moves into a freed slot 0 before it drafts
    var batch: [8]Proposal = undefined;
    var owners: [8]u32 = undefined;
    var n: usize = 0;
    const lone = live(b) == 1;
    var costs: [lb.max_rows]f64 = undefined;
    if (lone) {
        if (b.timed < b.window) return error.UntimedDraftWindow;
        for (costs[0..b.window], b.costs[0..b.window]) |*ms, cost| ms.* = cost.ms;
    }
    for (requests) |r| {
        const slot = try b.slotFor(r.stream);
        if (!b.draftsOwn(slot)) continue;
        if (r.depth == 0) {
            b.dropTreeOf(slot);
            continue;
        }
        const g = &b.draft.?.generation;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (b.count > 1) try absorbEarly(b, slot, r.rows orelse return error.NoKeptRows) else try b.settleRound();
        if (requests.len == 1) mtl.pass_metrics.begin(b.pass_profile, "lanes", b.window, mtl.clock.seconds);
        b.dropTreeOf(slot);
        const stands = b.runner.offsets[slot] + b.early[slot];
        if ((if (slot == 0) g.draft.committed_end else b.contexts[slot].?.committed_end) != stands) return error.BadDraftPosition;
        const room = b.runner.capacity - stands;
        if (r.depth == 0 or room < 2) continue;
        const pending = if (r.rows != null) r.follow[r.follow.len - 1] else try b.value(r.first orelse return error.NoFirstToken);
        const nodes = @import("qwen27_window.zig").proposal(b.window, b.copies, r.depth, @intCast(room), lone);
        if (slot == 0) if (g.proposer) |p| {
            b.trees[0] = try p.call(p.ptr, pending, nodes);
            continue;
        };
        batch[n] = .{ .context = if (slot == 0) null else b.contexts[slot].?, .pending = pending, .nodes = nodes, .costs = if (lone) costs[0..b.window] else &.{}, .overhead_ms = if (lone) @max(0, r.stream.tree_over orelse r.stream.round_over orelse 0) else 0, .defer_price = lone };
        owners[n] = slot;
        n += 1;
        if (n == batch.len) {
            try proposeAll(b, batch[0..n], owners[0..n]);
            n = 0;
        }
    }
    if (n > 0) try proposeAll(b, batch[0..n], owners[0..n]);
    for (requests) |r| {
        const slot = b.slotOf(r.stream) orelse continue;
        if (b.trees[slot]) |*t| {
            const opening = if (r.rows == null) try b.value(r.first orelse return error.NoFirstToken) else null;
            try @import("qwen27_masks.zig").prune(b.gpa, r.stream.grammar, opening, t);
        }
    }
}

/// How far each slot's MTP K/V follows its target rows from the prompt's start, and the layer's draft at that end.
pub const MtpCache = struct {
    ends: [lb.max_streams]u32 = @splat(0),
    drafts: [lb.max_streams]u32 = undefined,
    /// The slot whose prompt's last chunk the runner's taps hold, from `ends` on, until another target pass.
    tapped: ?u32 = null,
};

/// The layer steps over `slot`'s last `next.len` cache rows (the taps from round row `at`) only when its own cache
/// reaches their start: rows it missed (a resumed prompt, a skipped tail) stop its drafts for the stream.
fn mtpStep(b: *Metal, d: *q.mtp.Drafter, slot: u32, at: u32, next: []const u32) !void {
    const end = b.runner.offsets[slot];
    if (b.mtp_cache.ends[slot] + next.len != end) return;
    b.mtp_cache.drafts[slot] = try d.step(slot, at, next);
    b.mtp_cache.ends[slot] = end;
}

/// A prompt pass opens: the layer's cache holds none of it (a resumed or restored state's rows never reached it).
pub fn mtpOpen(b: *Metal, slot: u32) void {
    b.mtp_cache.ends[slot] = 0;
}

/// A prompt chunk `ids[at..end]` of `s` just passed: the layer steps over every chunk but the last, whose last row's
/// next token the first draft request brings.
pub fn mtpChunk(b: *Metal, s: *const lanes.Stream, slot: u32, at: usize, end: usize) !void {
    const d = b.mtp orelse return;
    b.mtp_cache.tapped = null;
    if (!s.drafts or b.mtp_cache.ends[slot] != at) return;
    const ids = s.prompt();
    if (end < ids.len) return mtpStep(b, d, slot, 0, ids[at + 1 .. end + 1]);
    b.mtp_cache.tapped = slot;
}

/// Every window's kept chain steps the layer, lone or shared, so a stream left alone drafts from a whole cache. Each
/// kept row's next token is the reply's own (committed before the keep), forced or drawn.
pub fn mtpKeep(b: *Metal, paths: []const []const u32) !void {
    const d = b.mtp orelse return;
    windows: for (paths, b.round.?.firsts, b.windows[0..b.count]) |path, first, slot| {
        const s = b.owners[slot] orelse continue;
        // the layer reads the tapped rows from the window's first: its kept rows must be them, in order
        for (path, 0..) |row, i| if (row != i) continue :windows;
        const end = b.runner.offsets[slot];
        const context = s.context.items;
        if (!s.drafts or context.len <= end) continue;
        try mtpStep(b, d, slot, first, context[end - path.len + 1 ..][0..path.len]);
    }
}

/// A prompt's end steps its last chunk's taps with the first token. The round is kept here, ahead of the engine's
/// keep, so the layer steps every kept chain before each stream holds its draft from its cache's end.
fn mtpDrafts(b: *Metal, d: *q.mtp.Drafter, requests: []const be.DraftRequest) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    for (requests) |r| {
        const slot = try b.slotFor(r.stream);
        b.dropTreeOf(slot);
        if (r.rows == null) try promptEnd(b, d, slot, r);
    }
    try keepEarly(b, requests);
    if (b.round != null) return; // a window it could not keep: its cache would trail its rows
    for (requests) |r| {
        const slot = try b.slotFor(r.stream);
        const end = b.runner.offsets[slot];
        if (r.depth == 0 or b.mtp_cache.ends[slot] != end or b.runner.capacity - end < 2) continue;
        b.trees[slot] = try chain(b.gpa, b.mtp_cache.drafts[slot]);
    }
}

/// A lone round settles; a shared one keeps every window's rows once it knows them all (the engine asks for every
/// drafting stream's in one batch; a plain stream's one-row window keeps its row), and the engine's keep then finds it
/// done.
fn keepEarly(b: *Metal, requests: []const be.DraftRequest) !void {
    if (b.count < 2) return b.settleRound();
    var paths: [lb.max_streams][]const u32 = undefined;
    for (b.windows[0..b.count], b.widths[0..b.count], paths[0..b.count]) |slot, width, *path| {
        const r = for (requests) |r| {
            if (b.slotOf(r.stream) == slot) break r;
        } else {
            if (width != 1) return;
            path.* = &.{0};
            continue;
        };
        path.* = r.rows orelse return;
    }
    try b.keepPaths(paths[0..b.count]);
    b.kept_early = true;
}

/// The prompt's last chunk, still in the taps, steps the layer with the first token after it.
fn promptEnd(b: *Metal, d: *q.mtp.Drafter, slot: u32, r: be.DraftRequest) !void {
    if (b.mtp_cache.tapped != slot) return;
    b.mtp_cache.tapped = null;
    const ids = r.stream.prompt();
    const at = b.mtp_cache.ends[slot];
    const first = r.first orelse return;
    if (b.runner.offsets[slot] != ids.len or at >= ids.len) return;
    const next = try b.gpa.alloc(u32, ids.len - at);
    defer b.gpa.free(next);
    @memcpy(next[0 .. next.len - 1], ids[at + 1 ..]);
    next[next.len - 1] = try b.value(first);
    try mtpStep(b, d, slot, 0, next);
}

/// A one-node tree; its score 0 leaves its chance to the landing prior.
fn chain(gpa: std.mem.Allocator, token: u32) !Tree {
    const tokens = try gpa.dupe(u32, &.{token});
    errdefer gpa.free(tokens);
    const parents = try gpa.dupe(i32, &.{-1});
    errdefer gpa.free(parents);
    return .{ .gpa = gpa, .tokens = tokens, .parents = parents, .scores = try gpa.dupe(f64, &.{0}) };
}

/// The batch's trees: a stream alone takes its own pass (slot 0's fused absorb and block); several share one.
fn proposeAll(b: *Metal, batch: []const Proposal, owners: []const u32) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = b.draft.?.generation.draft;
    if (batch.len == 1) {
        const p = batch[0];
        const x = p.context orelse {
            b.trees[owners[0]] = try m.proposeOptions(p.pending, q.dflash.runtime_model.Model.selection(p));
            return;
        };
        // the shared drafter takes the slot's context for one proposal, then hands it back
        try m.swap(x);
        const tree = m.proposeOptions(p.pending, q.dflash.runtime_model.Model.selection(p));
        m.swap(x) catch {
            m.backend.failed = true; // slot 0's context is parked in this slot's: recovery resets both
            return error.DraftContextSwap;
        };
        b.trees[owners[0]] = try tree;
        return;
    }
    var trees: [8]Tree = undefined;
    try m.proposeBatch(batch, trees[0..batch.len]);
    for (owners, trees[0..batch.len]) |slot, t| b.trees[slot] = t;
}

/// A slot's kept rows of a shared round's verify, gathered from its taps for its drafter before the keep.
fn absorbEarly(b: *Metal, slot: u32, path: []const u32) !void {
    if (b.early[slot] > 0) return error.AbsorbedTwice;
    const r = b.runner;
    const w = std.mem.indexOfScalar(u32, b.windows[0..b.count], slot) orelse return error.UnknownStream;
    const first = b.round.?.firsts[w];
    const rows = b.kept.slice(u32, path.len);
    for (rows, path) |*x, row| x.* = first + row;
    const cb = r.model.queue.commandBuffer();
    const e = cb.compute(.serial);
    try r.taps.accepted(r.model.glue, e, rows, .{ .buffer = b.kept }, .{ .buffer = b.gathered_taps.? });
    e.end();
    cb.commit();
    cb.wait();
    if (q.decode_round.gpuFailed(cb, "a taps gather")) return error.DraftTapGpuFailure;
    try b.ringOf(slot).?.absorb(b.gathered_taps.?, 0, @intCast(path.len));
    b.early[slot] = path.len;
}

/// Each held node's chance of landing: its own given its parent (e^own as the prior) times its parent's.
pub fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
    const b = Metal.self(ptr);
    const slot = b.slotOf(s) orelse return false;
    const t = b.trees[slot] orelse return false;
    if (out.len > t.scores.len) return false;
    for (out, t.parents[0..out.len], 0..) |*p, parent, i| {
        p.* = b.landing.chanceFor(s.shared_round, s.sampling != null, Metal.local(t, i)) * (if (parent >= 0) out[@intCast(parent)] else 1);
    }
    return true;
}

/// A slot's held tree as host tokens; a slot that does not draft holds none (its windows verify copies or one row).
pub fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
    const b = Metal.self(ptr);
    const slot = b.slotOf(s) orelse return null;
    const t = b.trees[slot] orelse return .{ .count = 0, .tokens = try gpa.alloc(u32, 0) };
    const tokens = try gpa.dupe(u32, t.tokens);
    errdefer gpa.free(tokens);
    return .{ .count = @intCast(t.tokens.len), .tokens = tokens, .parents = try gpa.dupe(i32, t.parents) };
}
