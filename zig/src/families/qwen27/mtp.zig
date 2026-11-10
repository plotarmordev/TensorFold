//! The checkpoint's MTP layer: the target's hidden row and the next token in, one attention block, its norm and the
//! target's head out.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const forward = @import("forward.zig");
const wts = @import("gpu_weights.zig");
const Ref = @import("projection.zig").Ref;
const rounds = @import("decode_round.zig");
const Runner = rounds.Runner;
const Round = @import("round_plan.zig").Round;

/// The layer's own buffers: the normed planes `fc` reads side by side, the gather's identity rows, its logits row.
pub const Scratch = struct {
    planes: mtl.Buffer,
    rows: mtl.Buffer,
    logits: mtl.Buffer,
    capacity: u32,

    pub fn init(device: mtl.Device, capacity: u32, hidden: usize, vocab: usize) !Scratch {
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const planes = try device.buffer(2 * @as(usize, capacity) * hidden * 2, opts);
        errdefer planes.deinit();
        const rows = try device.buffer(@as(usize, capacity) * 4, opts);
        errdefer rows.deinit();
        const logits = try device.buffer(vocab * 2, opts);
        const ids: [*]u32 = @ptrCast(@alignCast(rows.contents()));
        for (0..capacity) |i| ids[i] = @intCast(i);
        return .{ .planes = planes, .rows = rows, .logits = logits, .capacity = capacity };
    }

    pub fn deinit(s: *Scratch) void {
        s.planes.deinit();
        s.rows.deinit();
        s.logits.deinit();
    }
};

/// One pass over `rows` frame rows (ids and positions set): `hidden` row i beside token i; returns the normed rows,
/// and the last row's logits in `sc.logits`.
fn encode(m: *Model, sc: *const Scratch, e: *mtl.ComputeEncoder, hooks: forward.Mixer, hidden: Ref, rows: u32) !Ref {
    const w = m.weights.mtp orelse return error.NoMtpLayer;
    if (rows == 0 or rows > sc.capacity or rows > m.frame.capacity) return error.BadFrameRows;
    const f = &m.frame;
    const h: u32 = @intCast(m.config.hidden);
    try m.glue.embedding(e.*, m.config, m.weights.embed, f.get(.ids), f.get(.output), rows);
    e.barrier();
    try m.glue.headNorm(e.*, f.get(.output), wts.ref(w.norm_embed), .{ .buffer = sc.planes }, h, rows, m.config.eps);
    try m.glue.headNorm(e.*, hidden, wts.ref(w.norm_hidden), .{ .buffer = sc.planes, .offset = @as(usize, sc.capacity) * h * 2 }, h, rows, m.config.eps);
    e.barrier();
    try m.glue.tapRows(e.*, .{ .buffer = sc.planes }, .{ .buffer = sc.rows }, f.get(.mixed), .{ .rows = rows, .width = h, .capacity = sc.capacity, .planes = 2 });
    e.barrier();
    try m.kernels.quant(e.*, w.fc, f.get(.mixed), null, f.get(.dims), f.get(.hidden), rows);
    e.barrier();
    var lane = @import("core").segments.Lane{ .k = 0, .cb = undefined, .enc = e.*, .fence = undefined };
    defer e.* = lane.enc;
    var s = forward.Stages{ .model = m, .rows = rows, .hooks = hooks, .head = .last, .norm = w.norm, .logits = .{ .buffer = sc.logits }, .hidden = f.get(.hidden), .residual = f.get(.residual) };
    const block = m.config.layers;
    try s.pre(&lane, block);
    try s.mixer(&lane, block);
    try s.post(&lane, block);
    try s.finish(&lane);
    return f.get(.input);
}

/// One MTP pass on `r` (`round`: its cache rows and positions); its K/V rows go to the layer's own cache.
/// `r` must be built after loadMtp, so it holds the layer's block; one Mac, outside a round left to keep.
pub fn pass(r: *Runner, round: *const Round, sc: *const Scratch, hidden: Ref) !Ref {
    if (r.failed or r.active != null or r.reduce != null or r.model.weights.mtp == null or r.attention_index.len <= r.model.config.layers or round.ids.len > r.model.frame.capacity) return error.RoundNotReady;
    for (round.slots) |slot| if (slot >= r.ready) return error.RoundNotReady;
    var attention_plan = try r.attentionPlan(round);
    defer attention_plan.deinit();
    try r.model.frame.setRows(round.ids, round.feed_positions);
    try r.scratch.upload(attention_plan);
    r.current_round = round;
    r.current_attention = &attention_plan;
    defer {
        r.current_round = null;
        r.current_attention = null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.model.queue.commandBuffer();
    var e = cb.compute(.serial);
    errdefer e.end(); // an encoder left open aborts the process when it is released
    const out = try encode(r.model, sc, &e, .{ .ptr = r, .gdn = Runner.gdnForward, .attention = Runner.attentionForward }, hidden, @intCast(round.ids.len));
    e.end();
    cb.commit();
    cb.wait();
    if (rounds.gpuFailed(cb, "the MTP layer")) {
        r.failed = true;
        return error.RoundGpuFailure;
    }
    return out;
}

/// The layer once a decode step: after each target pass over one slot's rows, it runs over the same cache rows (its K/V
/// in its own cache) on the target's last-layer rows before the final norm, which the runner taps for it (so not for
/// DFlash2's drafter, nor prompt-cache marks, which refuse tapped passes).
pub const Drafter = struct {
    runner: *Runner,
    scratch: Scratch,

    pub fn init(r: *Runner) !Drafter {
        const m = r.model;
        if (r.capture_taps) return error.TapsTaken;
        try r.taps.use(m.device, &.{@intCast(m.config.layers - 1)});
        r.capture_taps = true;
        return .{ .runner = r, .scratch = try Scratch.init(m.device, m.frame.capacity, m.config.hidden, m.config.vocab) };
    }

    pub fn deinit(d: *Drafter) void {
        d.scratch.deinit();
    }

    /// After a target pass over `slot`'s last `next.len` rows, `next` holds each row's following token (the pass's ids
    /// from the second on, then the target's pick); returns the layer's greedy draft of the token after `next`'s last.
    pub fn step(d: *Drafter, slot: u32, next: []const u32) !u32 {
        const r = d.runner;
        const rows: u32 = @intCast(next.len);
        if (rows == 0 or r.taps.rows != rows or rows > r.offsets[slot]) return error.TapsDiffer;
        defer r.taps.begin(); // the taps are this step's: a second step without a target pass is refused
        var round = try Round.init(r.allocator, &.{.{ .slot = slot, .start = r.offsets[slot] - rows, .capacity = r.capacity, .ids = next }}, @intCast(r.model.config.conv_kernel), r.slots, @intCast(r.model.shard.global.vocab));
        defer round.deinit();
        _ = try pass(r, &round, &d.scratch, try r.taps.plane(0));
        return @import("session.zig").argmax(d.scratch.logits.slice(u16, r.model.config.vocab));
    }
};
