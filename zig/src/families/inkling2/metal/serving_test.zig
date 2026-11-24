//! Real-checkpoint Mini serving: serial target references, cache restore, DSpark and exact transactions.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const api = @import("engine_api");
const native = @import("../../../native/inkling2.zig");
const Model = @import("model.zig").Model;
const backend = @import("backend.zig");
const st = @import("state.zig");
const dspark = @import("dspark.zig");
const core = @import("core");
const mini_ref = @import("../mini_ref.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
/// A history the prompt cache can end a state on: the family's chunk grid.
const history = st.prompt_rows;
const context = 4 * history;
const made = 24;
const taps = [_]u32{ 5, 23, 41, 48 };

fn checkpoint() ![]const u8 {
    return @import("sys").getenv("TF_INKLING2_METAL_DIR") orelse error.SkipZigTest;
}

fn prompt(comptime n: usize, salt: u32) [n]u32 {
    var ids: [n]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1000 + (i * 37 + @as(usize, salt) * 101) % 20000);
    return ids;
}

const TapSink = struct {
    bytes: [st.feature_row_bytes]u8 = @splat(0),
    seen: usize = 0,

    fn take(ptr: *anyopaque, s: *lanes.Stream, f: lanes.backend.Features, start: u64) !void {
        const self: *TapSink = @ptrCast(@alignCast(ptr));
        try std.testing.expectEqual(lanes.backend.Features.Space.device, f.space);
        try std.testing.expectEqual(lanes.backend.Features.Dtype.bf16, f.dtype);
        try std.testing.expectEqual(@as(u32, st.feature_row_bytes), f.row_bytes);
        try std.testing.expect(f.rows > 0);
        if (start + f.rows != s.prompt_len) return;
        const buf: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
        const bytes = buf.slice(u8, @intCast(f.offset + f.rows * f.row_bytes));
        const last = bytes[@intCast(f.offset + (f.rows - 1) * f.row_bytes)..][0..st.feature_row_bytes];
        @memcpy(&self.bytes, last);
        self.seen += 1;
    }
};

fn featureRow(b: lanes.backend.Backend, s: *lanes.Stream, start: u64) ![st.feature_row_bytes]u8 {
    const f = try b.features(s, &taps, start, 1);
    try std.testing.expectEqual(lanes.backend.Features.Space.device, f.space);
    try std.testing.expectEqual(lanes.backend.Features.Dtype.bf16, f.dtype);
    try std.testing.expectEqual(@as(u32, 1), f.rows);
    try std.testing.expectEqual(@as(u32, st.feature_row_bytes), f.row_bytes);
    const buf: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
    const bytes = buf.slice(u8, @intCast(f.offset + f.row_bytes));
    var row: [st.feature_row_bytes]u8 = undefined;
    @memcpy(&row, bytes[@intCast(f.offset)..][0..st.feature_row_bytes]);
    return row;
}

const Reference = struct {
    tokens: [made]u32,
    first: lanes.LogprobRow,
    tap: [st.feature_row_bytes]u8,
    full_tap: [st.feature_row_bytes]u8 = undefined,
    partial_tap: [st.feature_row_bytes]u8 = undefined,
    continuation_tap: [st.feature_row_bytes]u8 = undefined,
};

fn serial(b: lanes.backend.Backend, ids: []const u32) !Reference {
    var sink: TapSink = .{};
    var s = try lanes.Stream.init(gpa, .{ .id = "serial-reference", .prompt = ids, .max_new = made, .drafts = false, .logprobs = 3 });
    defer s.deinit(gpa);
    try b.vtable.prefill_features.?(b.ptr, &s, &taps, .{ .ptr = &sink, .take = TapSink.take });
    defer b.release(&s);
    try std.testing.expectEqual(@as(usize, 1), sink.seen);
    var next = try b.read(try b.first(&s, ids.len));
    const first = try b.firstRow(&s);
    var out: Reference = .{ .tokens = undefined, .first = first, .tap = sink.bytes };
    out.tokens[0] = next;
    try std.testing.expectEqual(next, first.token);
    for (1..made) |i| {
        next = try b.read(try b.queue(&s, .{ .value = next }, ids.len + i));
        out.tokens[i] = next;
        switch (i) {
            8 => out.full_tap = try featureRow(b, &s, ids.len + 7),
            15 => out.partial_tap = try featureRow(b, &s, ids.len + 14),
            16 => out.continuation_tap = try featureRow(b, &s, ids.len + 15),
            else => {},
        }
    }
    return out;
}

fn withReference(dir: []const u8, ids: []const u32, extended: ?[]const u32) !struct { first: Reference, second: ?Reference } {
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 1, .budget = 2 * st.cacheBytes(context) });
    defer target.deinit();
    try target.warm();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps);
    const first = try serial(b, ids);
    const repeated = try serial(b, ids);
    try std.testing.expectEqualSlices(u8, &first.tap, &repeated.tap);
    try std.testing.expectEqualSlices(u32, &first.tokens, &repeated.tokens);
    try expectFirstRow(first.first, repeated.first);
    const second = if (extended) |p| try serial(b, p) else null;
    return .{ .first = first, .second = second };
}

fn expectFirstRow(want: lanes.LogprobRow, got: lanes.LogprobRow) !void {
    try std.testing.expectEqual(want.token, got.token);
    try std.testing.expectEqual(@as(u32, @bitCast(want.logprob)), @as(u32, @bitCast(got.logprob)));
    try std.testing.expectEqual(want.count, got.count);
    try std.testing.expectEqualSlices(u32, want.ids[0..want.count], got.ids[0..got.count]);
    try std.testing.expectEqualSlices(f32, want.logprobs[0..want.count], got.logprobs[0..got.count]);
}

const Reply = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    rows: std.ArrayList(lanes.LogprobRow) = .empty,
    cached: ?u32 = null,
    drafted: u64 = 0,
    accepted: u64 = 0,
    done: ?api.Reason = null,

    fn event(ptr: *anyopaque, _: api.Id, item: *const api.Event) void {
        const self: *Reply = @ptrCast(@alignCast(ptr));
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        switch (item.*) {
            .tokens => |ids| self.tokens.appendSlice(gpa, ids) catch {},
            .logprobs => |rows| self.rows.appendSlice(gpa, rows) catch {},
            .prefilled => |n| self.cached = n,
            .finished => |f| {
                self.done = f.reason;
                self.drafted = f.stats.drafted;
                self.accepted = f.stats.accepted;
            },
            else => {},
        }
    }

    fn wait(self: *Reply) !api.Reason {
        for (0..600_000) |_| {
            self.mutex.lockUncancelable(io);
            const done = self.done;
            self.mutex.unlock(io);
            if (done) |reason| return reason;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    fn deinit(self: *Reply) void {
        self.tokens.deinit(gpa);
        self.rows.deinit(gpa);
    }
};

fn serve(native_opened: api.Opened, id: api.Id, request: *const api.Request) !Reply {
    var reply: Reply = .{};
    errdefer reply.deinit();
    try native_opened.engine.submit(id, request, .{ .ctx = &reply, .event = Reply.event });
    try std.testing.expectEqual(api.Reason.length, try reply.wait());
    try std.testing.expectEqual(@as(usize, made), reply.tokens.items.len);
    return reply;
}

fn opened(dir: []const u8, drafter: ?[]const u8, cache_gib: ?f64, stream_count: u32) !api.Opened {
    var why: []const u8 = "";
    const result = try native.open(gpa, gpa, io, .{
        .dir = dir,
        .model_type = "inkling2_loop",
        .context = context,
        .lanes = stream_count,
        .drafts = drafter != null,
        .drafter = drafter,
        .prompt_cache_gib = cache_gib,
        .decode_share = 0.5,
    }, &why);
    if (result == null) {
        std.debug.print("Mini Metal open refused: {s}\n", .{why});
        return error.MiniMetalOpenRefused;
    }
    return result.?;
}

fn expectTokens(want: *const Reference, got: *const Reply) !void {
    try std.testing.expectEqualSlices(u32, &want.tokens, got.tokens.items);
    try std.testing.expect(got.rows.items.len >= 1);
    try expectFirstRow(want.first, got.rows.items[0]);
}

fn transactions(dir: []const u8, drafter_dir: []const u8, ids: []const u32, want: *const Reference) !void {
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 1, .budget = st.cacheBytes(context) });
    defer target.deinit();
    const draft_head = try dspark.Drafter.init(gpa, io, model.device, model, drafter_dir);
    defer draft_head.deinit();
    try draft_head.setContext(context);
    var wrapped = try lanes.drafted.Drafted.init(gpa, target.backend(), draft_head.drafter());
    defer wrapped.deinit();
    wrapped.first_rows = true;
    const b = wrapped.backend();
    const target_b = target.backend();
    var s = try lanes.Stream.init(gpa, .{ .id = "transactions", .prompt = ids, .max_new = made, .drafts = true, .logprobs = 3 });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    try std.testing.expectEqual(want.tokens[0], try b.read(try b.first(&s, ids.len)));
    try expectFirstRow(want.first, try b.firstRow(&s));
    try std.testing.expectEqualSlices(u8, &want.tap, &(try featureRow(target_b, &s, ids.len - 1)));

    const full_drafts = want.tokens[1..8];
    const full_positions = [_]u64{ ids.len + 1, ids.len + 2, ids.len + 3, ids.len + 4, ids.len + 5, ids.len + 6, ids.len + 7, ids.len + 8 };
    const full_path = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const full = [_]lanes.backend.Window{.{ .stream = &s, .pending = want.tokens[0], .held = 0, .tokens = full_drafts, .parents = null, .positions = &full_positions }};
    var sampled: [8]u32 = undefined;
    var echo: [7]u32 = undefined;
    var full_out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &echo }};
    try b.verify(&full, &full_out);
    try std.testing.expectEqualSlices(u32, want.tokens[1..9], &sampled);
    try std.testing.expectEqualSlices(u32, full_drafts, &echo);
    try b.keep(&full, &.{&full_path});
    try std.testing.expectEqualSlices(u8, &want.full_tap, &(try featureRow(target_b, &s, ids.len + 7)));
    try b.draft(&.{.{ .stream = &s, .follow = want.tokens[1..9], .rows = &full_path, .start = ids.len, .position = ids.len + 9, .depth = 7 }});

    var partial_drafts = want.tokens[9..16].*;
    partial_drafts[6] = (want.tokens[15] + 1) % model.config.text.unpadded_vocab;
    const partial_positions = [_]u64{ ids.len + 9, ids.len + 10, ids.len + 11, ids.len + 12, ids.len + 13, ids.len + 14, ids.len + 15, ids.len + 16 };
    const partial = [_]lanes.backend.Window{.{ .stream = &s, .pending = want.tokens[8], .held = 0, .tokens = &partial_drafts, .parents = null, .positions = &partial_positions }};
    const path = [_]u32{ 0, 1, 2, 3, 4, 5, 6 };
    var partial_out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &echo }};
    try b.verify(&partial, &partial_out);
    try std.testing.expectEqualSlices(u32, want.tokens[9..16], sampled[0..7]);
    try b.keep(&partial, &.{&path});
    try std.testing.expectEqualSlices(u8, &want.partial_tap, &(try featureRow(target_b, &s, ids.len + 14)));
    try b.draft(&.{.{ .stream = &s, .follow = want.tokens[9..16], .rows = &path, .start = ids.len + 8, .position = ids.len + 16, .depth = 7 }});
    const next = try b.read(try b.queue(&s, .{ .value = want.tokens[15] }, ids.len + 16));
    try std.testing.expectEqual(want.tokens[16], next);
    try std.testing.expectEqualSlices(u8, &want.continuation_tap, &(try featureRow(target_b, &s, ids.len + 15)));
}

/// A stream's next-row logits as the cache holds them, for a bitwise comparison.
fn nextLogits(target: *backend.Metal, s: *lanes.Stream) []const u8 {
    return target.caches.get(s).?.logits.slice(u8, st.vocabulary * 4);
}

/// A shared copy of a cache's convolution histories.
fn histories(model: *const Model, cache: *const st.Cache) !mtl.Buffer {
    const out = try model.device.buffer(st.conv_bytes, mtl.ResourceOptions.shared);
    errdefer out.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = model.queue.commandBuffer();
    const e = cb.blit();
    e.copy(cache.conv, 0, out, 0, st.conv_bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    return out;
}

test "Mini Metal: an open verify window leaves the histories untouched, and one that fails rolls back to a cache the next decode cannot tell apart" {
    const dir = try checkpoint();
    const ids = prompt(64, 9);
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 3, .budget = 3 * st.cacheBytes(context) });
    defer target.deinit();
    const b = target.backend();
    var streams: [3]lanes.Stream = undefined;
    for (&streams, [_][]const u8{ "clean", "failed", "stale" }) |*s, id| {
        s.* = try lanes.Stream.init(gpa, .{ .id = id, .prompt = &ids, .max_new = made, .drafts = true });
        try b.prefill(s);
    }
    defer for (&streams) |*s| {
        b.release(s);
        s.deinit(gpa);
    };
    const clean, const failed, const stale = .{ &streams[0], &streams[1], &streams[2] };
    const first = try b.read(try b.first(clean, ids.len));
    const wrong = (first + 1) % model.config.text.unpadded_vocab;
    const drafts = [_]u32{ wrong, wrong, wrong };
    const positions = [_]u64{ ids.len + 1, ids.len + 2, ids.len + 3, ids.len + 4 };
    var sampled: [4]u32 = undefined;
    var echo: [3]u32 = undefined;
    var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &echo }};

    // An open window has run its forward but staged its taps aside: the histories are those before it.
    const failed_cache = target.caches.get(failed).?;
    const before = try histories(model, failed_cache);
    defer before.deinit();
    try b.verify(&.{.{ .stream = failed, .pending = wrong, .held = 0, .tokens = &drafts, .parents = null, .positions = &positions, .masks = &.{} }}, &out);
    const open = try histories(model, failed_cache);
    defer open.deinit();
    try std.testing.expectEqualSlices(u8, before.slice(u8, st.conv_bytes), open.slice(u8, st.conv_bytes));
    failed_cache.rollback();

    // Every row's mask is empty: the draw fails after the forward has staged the window's taps on the GPU.
    const masks = try gpa.alloc(u32, positions.len * ((model.config.text.vocab + 31) / 32));
    defer gpa.free(masks);
    @memset(masks, 0);
    const window = [_]lanes.backend.Window{.{ .stream = failed, .pending = wrong, .held = 0, .tokens = &drafts, .parents = null, .positions = &positions, .masks = masks }};
    try std.testing.expectError(error.InklingEmptyTokenMask, b.verify(&window, &out));

    // The planted control: a forward outside any window writes its histories, and only the length is wound back.
    const stale_cache = target.caches.get(stale).?;
    _ = try b.queue(stale, .{ .value = wrong }, ids.len + 1);
    stale_cache.len -= 1;

    _ = try b.queue(clean, .{ .value = first }, ids.len + 1);
    _ = try b.queue(failed, .{ .value = first }, ids.len + 1);
    _ = try b.queue(stale, .{ .value = first }, ids.len + 1);
    try std.testing.expectEqualSlices(u8, nextLogits(target, clean), nextLogits(target, failed));
    try std.testing.expect(!std.mem.eql(u8, nextLogits(target, clean), nextLogits(target, stale)));
}

test "Mini Metal serving, fresh and restored, matches one serial stream on the backend token for token" {
    const dir = try checkpoint();
    const ids = prompt(history, 1);
    var turn: [history + 16]u32 = undefined;
    @memcpy(turn[0..history], &ids);
    @memcpy(turn[history..], &prompt(16, 2));
    const refs = try withReference(dir, &ids, &turn);
    var tap_nonzero = false;
    for (refs.first.tap) |byte| tap_nonzero = tap_nonzero or byte != 0;
    try std.testing.expect(tap_nonzero);

    const opened_once = try opened(dir, null, 1, 1);
    defer opened_once.close(opened_once.ctx);
    try std.testing.expectEqual(@as(?f64, 0.5), opened_once.engine.info().decode_share);
    const keepalive = opened_once.engine.keepaliveTarget() orelse return error.MissingMiniKeepalive;
    keepalive.tick(keepalive.ctx);
    var first = try serve(opened_once, 1, &.{ .prompt = &ids, .max_tokens = made, .drafts = false, .logprobs = 3, .history_len = ids.len });
    defer first.deinit();
    try expectTokens(&refs.first, &first);
    var fresh = try serve(opened_once, 2, &.{ .prompt = &turn, .max_tokens = made, .drafts = false, .logprobs = 3, .history_len = ids.len });
    defer fresh.deinit();
    try std.testing.expectEqual(@as(?u32, 0), fresh.cached);
    try expectTokens(&refs.second.?, &fresh);
    var restored = try serve(opened_once, 3, &.{ .prompt = &turn, .max_tokens = made, .drafts = false, .logprobs = 3, .history_len = ids.len });
    defer restored.deinit();
    try std.testing.expectEqual(@as(?u32, ids.len), restored.cached);
    try expectTokens(&refs.second.?, &restored);
}

test "Mini Metal opt-in DSpark drafting preserves target tokens and rolls back exactly" {
    const dir = try checkpoint();
    const drafter = @import("sys").getenv("TF_INKLING2_DSPARK_DIR") orelse return error.SkipZigTest;
    const ids = prompt(history, 3);
    var turn: [history + 16]u32 = undefined;
    @memcpy(turn[0..history], &ids);
    @memcpy(turn[history..], &prompt(16, 2));
    const refs = try withReference(dir, &ids, &turn);
    try transactions(dir, drafter, &ids, &refs.first);

    const opened_once = try opened(dir, drafter, 1, 1);
    defer opened_once.close(opened_once.ctx);
    var plain = try serve(opened_once, 10, &.{ .prompt = &ids, .max_tokens = made, .drafts = false, .logprobs = 3, .history_len = ids.len });
    defer plain.deinit();
    try expectTokens(&refs.first, &plain);
    try std.testing.expectEqual(@as(u64, 0), plain.drafted);
    var drafted = try serve(opened_once, 11, &.{ .prompt = &ids, .max_tokens = made, .drafts = true, .logprobs = 3, .history_len = ids.len });
    defer drafted.deinit();
    try expectTokens(&refs.first, &drafted);
    try std.testing.expect(drafted.drafted > 0);
    try std.testing.expect(drafted.accepted > 0);
    var fresh = try serve(opened_once, 12, &.{ .prompt = &turn, .max_tokens = made, .drafts = true, .logprobs = 3, .history_len = ids.len });
    defer fresh.deinit();
    try std.testing.expectEqual(@as(?u32, 0), fresh.cached);
    try expectTokens(&refs.second.?, &fresh);
    var restored = try serve(opened_once, 13, &.{ .prompt = &turn, .max_tokens = made, .drafts = true, .logprobs = 3, .history_len = ids.len });
    defer restored.deinit();
    try std.testing.expectEqual(@as(?u32, ids.len), restored.cached);
    try expectTokens(&refs.second.?, &restored);
}

const AbsorbSink = struct {
    d: lanes.drafter.Drafter,

    fn take(ptr: *anyopaque, s: *lanes.Stream, f: lanes.backend.Features, start: u64) !void {
        const self: *AbsorbSink = @ptrCast(@alignCast(ptr));
        const ids = s.prompt();
        const last = ids.len - 1;
        if (start >= last) return;
        var rows = f;
        rows.rows = @intCast(@min(f.rows, last - start));
        try self.d.absorb(&.{.{ .stream = s, .features = rows, .start = start, .follow = ids[@intCast(start + 1)..][0..rows.rows] }});
    }
};

fn prepareHold(b: lanes.backend.Backend, d: lanes.drafter.Drafter, s: *lanes.Stream) !u32 {
    try d.open(s);
    errdefer d.release(s);
    var sink: AbsorbSink = .{ .d = d };
    try b.vtable.prefill_features.?(b.ptr, s, d.taps(), .{ .ptr = &sink, .take = AbsorbSink.take });
    errdefer b.release(s);
    const ids = s.prompt();
    const pending = try b.read(try b.first(s, ids.len));
    const last = try b.features(s, d.taps(), ids.len - 1, 1);
    const follow = [1]u32{pending};
    try d.absorb(&.{.{ .stream = s, .features = last, .start = ids.len - 1, .follow = &follow }});
    return pending;
}

fn holdAlone(b: lanes.backend.Backend, d: lanes.drafter.Drafter, ids: []const u32, depth: u32, out: []u32) !void {
    var s = try lanes.Stream.init(gpa, .{ .id = "solo-hold", .prompt = ids, .max_new = made, .drafts = true });
    defer s.deinit(gpa);
    const pending = try prepareHold(b, d, &s);
    defer b.release(&s);
    defer d.release(&s);
    try d.hold(&.{.{ .stream = &s, .pending = pending, .position = ids.len + 1, .depth = depth }});
    try d.held(&.{&s}, &.{out});
}

test "Mini Metal DSpark sixteen-stream chain preserves solo holds at unequal depths" {
    const dir = try checkpoint();
    const drafter_dir = @import("sys").getenv("TF_INKLING2_DSPARK_DIR") orelse return error.SkipZigTest;
    const depths = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 3, 6 };
    var ids: [16][history]u32 = undefined;
    for (&ids, 0..) |*p, i| p.* = prompt(history, @intCast(40 + i));

    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 16, .budget = 16 * st.cacheBytes(context) });
    defer target.deinit();
    const draft_head = try dspark.Drafter.init(gpa, io, model.device, model, drafter_dir);
    defer draft_head.deinit();
    try draft_head.setContext(context);
    const b = target.backend();
    const d = draft_head.drafter();
    try b.vtable.prepare_features.?(b.ptr, d.taps());
    var want: [16][7]u32 = undefined;
    for (0..16) |i| try holdAlone(b, d, &ids[i], depths[i], want[i][0..depths[i]]);

    var streams: [16]lanes.Stream = undefined;
    var initialized: usize = 0;
    defer for (streams[0..initialized]) |*s| s.deinit(gpa);
    for (&streams, 0..) |*s, i| {
        s.* = try lanes.Stream.init(gpa, .{ .id = "chain-hold", .prompt = &ids[i], .max_new = made, .drafts = true });
        initialized += 1;
    }
    var prepared: usize = 0;
    defer for (streams[0..prepared]) |*s| b.release(s);
    defer for (streams[0..prepared]) |*s| d.release(s);
    var holds: [16]lanes.drafter.Hold = undefined;
    for (&streams, 0..) |*s, i| {
        const pending = try prepareHold(b, d, s);
        prepared += 1;
        holds[i] = .{ .stream = s, .pending = pending, .position = history + 1, .depth = depths[i] };
    }
    var got: [16][7]u32 = undefined;
    var ptrs: [16]*lanes.Stream = undefined;
    var outs: [16][]u32 = undefined;
    for (0..16) |i| {
        ptrs[i] = &streams[i];
        outs[i] = got[i][0..depths[i]];
    }
    try d.hold(holds[0..4]);
    try d.held(ptrs[0..4], outs[0..4]);
    for (0..4) |i| try std.testing.expectEqualSlices(u32, want[i][0..depths[i]], outs[i]);
    try d.hold(&holds);
    try d.held(&ptrs, &outs);
    for (0..16) |i| try std.testing.expectEqualSlices(u32, want[i][0..depths[i]], outs[i]);
}

test "Mini Metal DSpark four prompts equal solos through a leave and join" {
    const dir = try checkpoint();
    const drafter = @import("sys").getenv("TF_INKLING2_DSPARK_DIR") orelse return error.SkipZigTest;
    const opened_wide = try opened(dir, drafter, 0, 4);
    defer opened_wide.close(opened_wide.ctx);
    var ids: [4][history]u32 = undefined;
    var requests: [4]api.Request = undefined;
    for (0..4) |i| {
        ids[i] = prompt(history, @intCast(70 + i));
        // Null sampling is the native request's temperature-zero, greedy representation.
        requests[i] = .{ .prompt = &ids[i], .max_tokens = made, .sampling = null, .drafts = true };
    }
    var solo: [4]Reply = .{ .{}, .{}, .{}, .{} };
    defer for (&solo) |*r| r.deinit();
    for (0..4) |i| solo[i] = try serve(opened_wide, 200 + i, &requests[i]);

    var batched: [4]Reply = .{ .{}, .{}, .{}, .{} };
    defer for (&batched) |*r| r.deinit();
    var drafted: u64 = 0;
    var accepted: u64 = 0;
    for (0..4) |i| try opened_wide.engine.submit(210 + i, &requests[i], .{ .ctx = &batched[i], .event = Reply.event });
    for (0..4) |i| try std.testing.expectEqual(api.Reason.length, try batched[i].wait());
    for (0..4) |i| {
        try std.testing.expectEqual(@as(usize, made), batched[i].tokens.items.len);
        try std.testing.expectEqualSlices(u32, solo[i].tokens.items, batched[i].tokens.items);
        drafted += batched[i].drafted;
        accepted += batched[i].accepted;
    }
    for (0..4) |i| {
        var previous = batched[i];
        batched[i] = .{};
        previous.deinit();
    }
    const lengths = [_]u32{ 6, 12, made, made };
    for (0..3) |i| {
        requests[i].max_tokens = lengths[i];
        try opened_wide.engine.submit(220 + i, &requests[i], .{ .ctx = &batched[i], .event = Reply.event });
    }
    try std.testing.expectEqual(api.Reason.length, try batched[0].wait());
    try std.testing.expectEqual(@as(usize, 6), batched[0].tokens.items.len);
    try opened_wide.engine.submit(223, &requests[3], .{ .ctx = &batched[3], .event = Reply.event });
    try std.testing.expectEqual(api.Reason.length, try batched[1].wait());
    for (2..4) |i| try std.testing.expectEqual(api.Reason.length, try batched[i].wait());
    for (0..4) |i| {
        try std.testing.expectEqual(@as(usize, lengths[i]), batched[i].tokens.items.len);
        try std.testing.expectEqualSlices(u32, solo[i].tokens.items[0..lengths[i]], batched[i].tokens.items);
        drafted += batched[i].drafted;
        accepted += batched[i].accepted;
    }
    try std.testing.expect(drafted > 0);
    try std.testing.expect(accepted > 0);
}

test "Mini Metal fills a prompt a chunk a call to the whole pass's first row and features" {
    const dir = try checkpoint();
    const ids = prompt(64, 5);
    const cuts = [_]u32{ 20, 41 };
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 2, .budget = 2 * st.cacheBytes(context) });
    defer target.deinit();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps);
    var whole = try lanes.Stream.init(gpa, .{ .id = "whole", .prompt = &ids, .max_new = made, .drafts = false, .logprobs = 3, .chunks = &cuts });
    defer whole.deinit(gpa);
    try b.prefill(&whole);
    defer b.release(&whole);
    try std.testing.expectEqual(@as(u32, ids.len), whole.fill_at.load(.monotonic));
    var stepped = try lanes.Stream.init(gpa, .{ .id = "stepped", .prompt = &ids, .max_new = made, .drafts = false, .logprobs = 3, .chunks = &cuts });
    defer stepped.deinit(gpa);
    defer b.release(&stepped);
    var from: u32 = 0;
    for ([_]u32{ 20, 41, ids.len }) |end| {
        try std.testing.expectEqual(end == ids.len, try b.vtable.prefill_step.?(b.ptr, &stepped, 0));
        try std.testing.expectEqual(end, stepped.fill_at.load(.monotonic));
        // A chunk's rows are held until the next forward, and an earlier chunk's are not.
        try std.testing.expectEqual(end - from, (try b.features(&stepped, &taps, from, end - from)).rows);
        if (from > 0) try std.testing.expectError(error.Inkling2TapsNotHeld, b.features(&stepped, &taps, 0, from));
        from = end;
    }
    try std.testing.expectEqualSlices(u8, &(try featureRow(b, &whole, ids.len - 1)), &(try featureRow(b, &stepped, ids.len - 1)));
    const whole_first = try b.read(try b.first(&whole, ids.len));
    const whole_row = try b.firstRow(&whole);
    try std.testing.expectEqual(whole_first, try b.read(try b.first(&stepped, ids.len)));
    try expectFirstRow(whole_row, try b.firstRow(&stepped));
}

/// Greedy tokens after `ids` on the device into `out`, and the float64 reference over the same config and plan; the
/// model is released before the reference reads `weights` on the host.
fn greedy(dir: []const u8, ids: []const u32, out: []u32, weights: *core.checkpoint.Checkpoint) !mini_ref.Ref {
    const lock = try std.Io.Dir.cwd().createFile(io, "/tmp/tensorfold-model.lock", .{ .truncate = false });
    defer lock.close(io);
    try lock.lock(io, .exclusive);
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 1, .budget = st.cacheBytes(context) });
    defer target.deinit();
    const b = target.backend();
    var s = try lanes.Stream.init(gpa, .{ .id = "float64-reference", .prompt = ids, .max_new = @intCast(out.len), .drafts = false });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    out[0] = try b.read(try b.first(&s, ids.len));
    for (1..out.len) |i| out[i] = try b.read(try b.queue(&s, .{ .value = out[i - 1] }, ids.len + i));
    return .{ .gpa = gpa, .c = model.config, .p = model.weight_plan, .ck = weights };
}

/// Generated positions where the reference's pick is not the token the device drew.
fn differences(picks: []const mini_ref.Pick, all: []const u32, prompt_len: usize, report: bool) usize {
    var differ: usize = 0;
    for (prompt_len - 1..picks.len) |i| if (picks[i].id != all[i + 1]) {
        differ += 1;
        if (report) std.debug.print("DIFF Mini Metal position {d}: device {d}, float64 {d}, margin {d:.4}\n", .{
            i + 1, all[i + 1], picks[i].id, picks[i].margin,
        });
    };
    return differ;
}

test "Mini Metal greedy matches the float64 reference over 32 tokens, and a reference without messages does not" {
    // The reference walks every weight on the host in float64: minutes optimized, hours in a Debug build.
    if (@import("builtin").mode == .debug) return error.SkipZigTest;
    const dir = try checkpoint();
    const prompt_len = 16;
    var all: [prompt_len + 32]u32 = undefined;
    @memcpy(all[0..prompt_len], &prompt(prompt_len, 7));
    var weights = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer weights.close();
    var reference = try greedy(dir, all[0..prompt_len], all[prompt_len..], &weights);
    var picks: [all.len - 1]mini_ref.Pick = undefined;
    try reference.run(all[0 .. all.len - 1], &picks);
    try std.testing.expectEqual(@as(usize, 0), differences(&picks, &all, prompt_len, true));
    reference.plant = true;
    try reference.run(all[0 .. all.len - 1], &picks);
    try std.testing.expect(differences(&picks, &all, prompt_len, false) > 0);
}
