//! Real Inkling 2 Mini CUDA serving: device draws against host selection, snapshots and admission.
const std = @import("std");
const lanes = @import("lanes");
const mini = @import("inkling2");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const native = mini.native;
const Mini = mini.mini_backend.Mini;
const Saved = mini.mini_snap.Saved;
const Gpu = check.Gpu;
const sampled: lanes.Sampling = .{ .seed = 37, .temperature = 0.8, .top_k = 64, .top_p = 0.95 };

const Trace = struct {
    ids: []u32,
    logits: []f32,
    cached: u32,
    failed: bool,

    fn deinit(t: *Trace, gpa: std.mem.Allocator) void {
        gpa.free(t.ids);
        gpa.free(t.logits);
    }
};

fn download(t: *Mini, src: u64, out: []f32) !void {
    try t.s.synchronize();
    try cuda.memory.download(t.d, src, std.mem.sliceAsBytes(out));
}

/// A real target pass with either CUDA's draws or host selection over its downloaded fp32 logits.
fn trace(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32, sampling: ?lanes.Sampling, n: usize, host: bool, reuse: lanes.stream.Reuse) !Trace {
    const t = loaded.target;
    const v: usize = t.m.c.unpadded_vocab;
    var s = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-trace", .prompt = prompt, .max_new = @intCast(n), .sampling = sampling, .reuse = reuse });
    defer s.deinit(gpu.gpa);
    const b = loaded.backend;
    defer b.release(&s);
    const ids = try gpu.gpa.alloc(u32, n);
    errdefer gpu.gpa.free(ids);
    const logits = try gpu.gpa.alloc(f32, n * v);
    errdefer gpu.gpa.free(logits);
    const draw_row: ?[]f32 = if (host and cuda.sample.ruleFor(sampling) != null) try gpu.gpa.alloc(f32, v) else null;
    defer if (draw_row) |rounded| gpu.gpa.free(rounded);
    try b.prefill(&s);
    try download(t, t.lanes.getPtr(&s).?.last, logits[0..v]); // the prompt's last row, kept for the stream's slot
    var handle: ?u64 = try b.first(&s, prompt.len);
    for (0..n) |i| {
        ids[i] = if (host) blk: {
            const row = logits[i * v ..][0..v];
            if (draw_row) |rounded| {
                for (rounded, row) |*dst, value| dst.* = core.bf16_round.round32(value);
                break :blk try lanes.gpu_full.draw(gpu.gpa, rounded, sampling, @intCast(prompt.len + i), null);
            }
            break :blk try lanes.gpu_full.draw(gpu.gpa, row, null, @intCast(prompt.len + i), null);
        } else try b.read(handle.?);
        if (i + 1 < n) {
            handle = try b.queue(&s, .{ .value = ids[i] }, prompt.len + i + 1);
            try download(t, t.m.s.logits, logits[(i + 1) * v ..][0..v]);
        }
    }
    return .{ .ids = ids, .logits = logits, .cached = s.cached, .failed = s.reuse_failed };
}

fn sameTrace(a: *const Trace, b: *const Trace) !void {
    if (!std.mem.eql(u32, a.ids, b.ids)) return error.MiniServingTokenMismatch;
    if (std.mem.indexOfDiff(u32, @ptrCast(a.logits), @ptrCast(b.logits))) |i| {
        const v = a.logits.len / a.ids.len;
        std.debug.print("RESULT mini-serving logits differ step={d} id={d} {e} vs {e}\n", .{ i / v, i % v, a.logits[i], b.logits[i] });
        return error.MiniServingLogitMismatch;
    }
}

fn hostReference(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32) !void {
    for ([_]?lanes.Sampling{ null, sampled }, 0..) |sampling, case| {
        var device = try trace(gpu, loaded, prompt, sampling, 12, false, .{});
        defer device.deinit(gpu.gpa);
        var host = try trace(gpu, loaded, prompt, sampling, 12, true, .{});
        defer host.deinit(gpu.gpa);
        sameTrace(&device, &host) catch |err| {
            std.debug.print("RESULT mini-serving host-reference case={d} mismatch={s}\n", .{ case, @errorName(err) });
            return err;
        };
    }
}

/// Save one complete state at `cut`, restore it through the genuine prompt-reuse hook and rerun the suffix.
fn resumedAt(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32, cut: u32, sampling: ?lanes.Sampling, fresh: *const Trace) !void {
    const Keeper = struct {
        b: *Mini,
        saved: ?*Saved = null,
        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ptr));
            k.saved = k.b.saveState(s, mark) catch null;
        }
    };
    const t = loaded.target;
    const saved = snapshot: {
        var s = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-snapshot", .prompt = prompt, .max_new = 1, .reuse = .{ .marks = &.{cut} } });
        defer s.deinit(gpu.gpa);
        const b = loaded.backend;
        defer b.release(&s);
        var keeper: Keeper = .{ .b = t };
        s.reuse.hook = .{ .ptr = &keeper, .at = Keeper.at };
        try b.prefill(&s);
        break :snapshot keeper.saved orelse return error.MiniSnapshotNotSaved;
    };
    defer t.dropState(saved);
    var resumed = try trace(gpu, loaded, prompt, sampling, fresh.ids.len, false, .{ .saved = saved, .at = cut });
    defer resumed.deinit(gpu.gpa);
    if (resumed.cached != cut or resumed.failed) return error.MiniSnapshotNotRestored;
    try sameTrace(fresh, &resumed);
}

fn freshAndResumed(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32) !void {
    for ([_]?lanes.Sampling{ null, sampled }, 0..) |sampling, case| {
        var fresh = try trace(gpu, loaded, prompt, sampling, 8, false, .{});
        defer fresh.deinit(gpu.gpa);
        const cuts: []const u32 = if (case == 0) &.{ 17, 2048 } else &.{ 2049, @intCast(prompt.len - 1) };
        for (cuts) |cut| resumedAt(gpu, loaded, prompt, cut, sampling, &fresh) catch |err| {
            std.debug.print("RESULT mini-serving resumed case={d} cut={d} mismatch={s}\n", .{ case, cut, @errorName(err) });
            return err;
        };
    }
}

/// Both slots decode at once in shared rounds; a third request finds no free slot until one ends.
fn admission(gpu: Gpu, loaded: *native.Loaded, a: []const u32, b: []const u32) !void {
    var plain_a = try trace(gpu, loaded, a, null, 10, false, .{});
    defer plain_a.deinit(gpu.gpa);
    var plain_b = try trace(gpu, loaded, b, sampled, 10, false, .{});
    defer plain_b.deinit(gpu.gpa);
    var first = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-first", .prompt = a, .max_new = 10 });
    defer first.deinit(gpu.gpa);
    var second = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-second", .prompt = b, .max_new = 10, .sampling = sampled });
    defer second.deinit(gpu.gpa);
    var third = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-third", .prompt = a, .max_new = 10 });
    defer third.deinit(gpu.gpa);
    var cfg = try lanes.Config.init(gpu.gpa, loaded.facts, loaded.rows, loaded.rows - 1); // the host's widths (native/cuda_startup_phase.zig)
    defer cfg.deinit(gpu.gpa);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpu.gpa, &cfg, loaded.backend, clock.clock());
    defer e.deinit();
    try e.addStream(&first);
    if (!native.fits(loaded.ctx, @intCast(b.len), second.max_new)) return error.MiniSecondLaneRefused;
    try e.addStream(&second);
    if (native.fits(loaded.ctx, @intCast(a.len), third.max_new)) return error.MiniBusyAdmissionRefused;
    try std.testing.expectError(error.NoFreeSlot, loaded.backend.prefill(&third));
    const rounds = e.shared_rounds;
    while (!first.finished or !second.finished) try e.step();
    if (e.shared_rounds == rounds) return error.MiniLanesNeverShared;
    try std.testing.expectEqual(lanes.stream.Reason.length, first.reason);
    try std.testing.expectEqual(lanes.stream.Reason.length, second.reason);
    if (!std.mem.eql(u32, first.emitted(), plain_a.ids)) return error.MiniAdmissionChangedFirst;
    if (!std.mem.eql(u32, second.emitted(), plain_b.ids)) return error.MiniAdmissionChangedSecond;
    if (!native.fits(loaded.ctx, @intCast(a.len), third.max_new)) return error.MiniSlotNotReleased;
}

fn readIds(gpu: Gpu, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 26));
    defer gpu.gpa.free(text);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpu.gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n\r\t[]");
    while (it.next()) |t| try out.append(gpu.gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpu.gpa);
}

pub fn run(gpu: Gpu, model: []const u8, ids_path: []const u8) !void {
    const ids = try readIds(gpu, ids_path);
    defer gpu.gpa.free(ids);
    if (ids.len < 4096) return error.MiniServingPromptTooShort;
    const context = ids.len + 64;
    var loaded = try native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = context, .drafts = false, .lanes = 2 });
    defer loaded.deinit(loaded.ctx);
    try hostReference(gpu, &loaded, ids[0..4096]);
    check.pass("Mini CUDA serving matches the host reference token for token", .{});
    try freshAndResumed(gpu, &loaded, ids[0..2309]);
    check.pass("Mini CUDA fresh and resumed prompts produce identical logits and tokens", .{});
    try admission(gpu, &loaded, ids[0..257], ids[17..1042]);
    check.pass("Mini CUDA two lanes decode together in shared rounds without changing tokens; a third waits", .{});
}

/// 64 streams of one prompt through one shared decode round, opened apart from `run` (its window at 64 streams is past what the
/// recipe serves): the plan the round's last MoE visit left on the device, beside the same picks routed in the 16-row form.
pub fn round64(gpu: Gpu, model: []const u8, ids_path: []const u8) !void {
    const ids = try readIds(gpu, ids_path);
    defer gpu.gpa.free(ids);
    const n = mini.mini_backend.round_rows;
    const len = 48; // not 64 rows: a prompt chunk's plan cannot pass for the round's
    if (ids.len < len) return error.MiniServingPromptTooShort;
    var loaded = try native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = len + 8, .drafts = false, .lanes = n });
    defer loaded.deinit(loaded.ctx);
    const t = loaded.target;
    const c = t.m.c;
    const pairs: usize = n * c.topk;
    const streams = try gpu.gpa.alloc(lanes.Stream, n);
    defer gpu.gpa.free(streams);
    var made: usize = 0;
    defer for (streams[0..made]) |*s| {
        loaded.backend.release(s);
        s.deinit(gpu.gpa);
    };
    for (streams) |*s| {
        s.* = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-64", .prompt = ids[0..len], .max_new = 4 });
        made += 1;
    }
    var cfg = try lanes.Config.init(gpu.gpa, loaded.facts, loaded.rows, loaded.rows - 1);
    defer cfg.deinit(gpu.gpa);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpu.gpa, &cfg, loaded.backend, clock.clock());
    defer e.deinit();
    for (streams) |*s| try e.addStream(s);
    try e.step();
    // the 16-row form of the same picks: its members, items and counts beside the round's (rank and hist are a route's own scratch)
    const all = [2]usize{ cuda.grouped.maxItems(pairs, c.experts, 64), cuda.grouped.maxItems(pairs, c.experts, 16) };
    var scratch = try cuda.DeviceBuffer.alloc(gpu.d, pairs * 4 + all[1] * 12 + 8);
    defer scratch.free();
    var plan16 = t.m.s.plan;
    plan16.members = scratch.ptr;
    plan16.items = scratch.ptr + pairs * 4;
    plan16.counts = plan16.items + all[1] * 12;
    try t.k.router.route(t.s, t.m.s.pick, pairs, c.experts, 16, plan16);
    try t.s.synchronize();
    var counts: [2][2]i32 = undefined; // tile 64, then 16: items the plan holds, experts with rows
    try cuda.memory.download(t.d, t.m.s.plan.counts, std.mem.sliceAsBytes(&counts[0]));
    try cuda.memory.download(t.d, plan16.counts, std.mem.sliceAsBytes(&counts[1]));
    try check.expect(e.shared_rounds == 1 and counts[0][0] > 0 and counts[0][0] <= all[0] and counts[0][1] > 0 and counts[1][0] > 0, "{d} streams: {d} shared rounds; the plan holds {d} items over {d} experts at tile 64, {d} at tile 16", .{ n, e.shared_rounds, counts[0][0], counts[0][1], counts[1][0] });
    const held: usize = @intCast(counts[0][0]);
    const used: usize = @intCast(counts[0][1]);
    const items = try gpu.gpa.alloc([3]i32, held); // expert, first, rows
    defer gpu.gpa.free(items);
    try cuda.memory.download(t.d, t.m.s.plan.items, std.mem.sliceAsBytes(items));
    var rows: i64 = 0;
    var least: i32 = std.math.maxInt(i32);
    var most: i32 = 0;
    for (items) |it| {
        rows += it[2];
        least = @min(least, it[2]);
        most = @max(most, it[2]);
    }
    // an item per expert with rows: each expert's weights are read once a round
    try check.expect(held == used and rows == pairs and counts[1][1] == counts[0][1], "{d} streams' round at tile 64: {d} items over {d} experts with rows ({d} routed at tile 16), {d} of {d} pairs in them", .{ n, held, used, counts[1][1], rows, pairs });
    check.pass("Mini CUDA {d} streams of one prompt, one shared round's last MoE visit: {d} rows, {d} pairs over {d} experts with rows; device items {d} at tile 64 ({d} to {d} rows an item, one weight read an expert), {d} at tile 16; maxItems over the {d} experts with rows {d} at tile 64, {d} at tile 16; launch bound (maxItems over all {d} experts) {d} at tile 64, {d} at tile 16", .{ n, n, pairs, used, held, least, most, counts[1][0], used, cuda.grouped.maxItems(pairs, used, 64), cuda.grouped.maxItems(pairs, used, 16), c.experts, all[0], all[1] });
}
