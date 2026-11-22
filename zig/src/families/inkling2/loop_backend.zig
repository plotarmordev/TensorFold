//! Inkling 2 Mini lane execution and snapshot adapter: one complete sequence state and CUDA stream a lane.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("loop_engine.zig");
const weights_ = @import("loop_weights.zig");
const ops = @import("ops.zig");

const Model = engine.Model;
const Kernels = engine.Kernels;
const prompt_rows = engine.prompt_rows;
const window_rows = engine.window_rows;
const token_ring = 1024;

/// A sequence's own cache, scratch and stream; shared rounds queue every lane's forward before any waits.
pub const Lane = struct {
    m: Model,
    s: cuda.Stream,
    logits16: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    last: cuda.DeviceBuffer,
    rule: cuda.DeviceBuffer,
    owner: ?*lanes.Stream = null,
    at: usize = 0, // the tokens buffer row this lane's verify draws start at
    len: usize = 0,
    fill: ?usize = null,
    base: usize = 0,
    rows: std.ArrayList(u32) = .empty,
    sampled: bool = false,

    fn init(l: *Lane, gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights_.Weights, context: usize) !void {
        const v = w.c.text.unpadded_vocab;
        var s = try cuda.Stream.init(d, true);
        errdefer s.deinit();
        var m = try Model.init(gpa, d, s, w, context);
        errdefer m.deinit();
        var logits16 = try cuda.DeviceBuffer.alloc(d, window_rows * v * 2);
        errdefer logits16.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        var last = try cuda.DeviceBuffer.alloc(d, v * 4);
        errdefer last.free();
        l.* = .{ .m = m, .s = s, .logits16 = logits16, .meta = meta, .last = last, .rule = try cuda.DeviceBuffer.alloc(d, @sizeOf(cuda.sample.Rule)) };
    }

    fn deinit(l: *Lane, gpa: std.mem.Allocator) void {
        l.s.synchronize() catch {};
        l.rows.deinit(gpa);
        l.rule.free();
        l.last.free();
        l.meta.free();
        l.logits16.free();
        l.m.deinit();
        l.s.deinit();
    }

    /// Device bytes the lane holds.
    fn heldBytes(l: *const Lane) u64 {
        return l.m.heldBytes() + l.logits16.len + l.meta.len + l.last.len + l.rule.len;
    }
};

pub const Backend = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const Kernels,
    lanes: []Lane,
    tokens: cuda.DeviceBuffer, // the token ring, then each lane's verify rows
    next: u64 = 0,

    const be = lanes.backend;

    /// Up to `count` lanes over shared weights and kernels, as many as `room` (device bytes left) holds, at least one.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights_.Weights, k: *const Kernels, context: usize, count: usize, room: ?u64) !Backend {
        var one: Lane = undefined;
        try one.init(gpa, d, w, context);
        var made: usize = 1;
        const fit = if (room) |r| r / one.heldBytes() else count;
        const all = gpa.alloc(Lane, std.math.clamp(fit, 1, @max(count, 1))) catch |err| {
            one.deinit(gpa);
            return err;
        };
        all[0] = one;
        errdefer gpa.free(all);
        errdefer for (all[0..made]) |*l| l.deinit(gpa);
        while (made < all.len) : (made += 1) try all[made].init(gpa, d, w, context);
        for (all, 0..) |*l, i| l.at = token_ring + i * window_rows;
        const tokens = try cuda.DeviceBuffer.alloc(d, (token_ring + all.len * window_rows) * 4);
        return .{ .gpa = gpa, .d = d, .k = k, .lanes = all, .tokens = tokens };
    }

    pub fn deinit(b: *Backend) void {
        for (b.lanes) |*l| l.deinit(b.gpa);
        b.gpa.free(b.lanes);
        b.tokens.free();
        b.* = undefined;
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable };
    }

    pub fn facts(b: *const Backend) lanes.Model {
        return .{ .exact_width = window_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = prompt_rows, .max_streams = @intCast(b.lanes.len), .streams_exact = true };
    }

    pub fn admits(b: *const Backend, prompt: usize, max_new: usize) bool {
        const m = &b.lanes[0].m;
        return b.free() != null and prompt <= m.context and max_new <= m.context - prompt;
    }

    fn free(b: *const Backend) ?usize {
        for (b.lanes, 0..) |l, i| if (l.owner == null) return i;
        return null;
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(b: *Backend, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(b.d, dst, bytes, s);
    }

    fn laneOf(b: *Backend, s: *const lanes.Stream) !*Lane {
        for (b.lanes) |*l| if (l.owner == s) return l;
        return error.UnknownStream;
    }

    fn resumeSaved(l: *Lane, s: *lanes.Stream) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const Model.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= s.prompt().len) {
            s.reuse_failed = true;
            return 0;
        }
        l.m.restore(l.s, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        s.cached = k.at;
        return k.at;
    }

    fn opening(b: *Backend, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        const context = b.lanes[0].m.context;
        if (prompt.len > context or s.max_new > context - prompt.len) return error.PastContext;
        const l = b.laneOf(s) catch &b.lanes[b.free() orelse return error.NoFreeSlot];
        l.owner = s;
        l.sampled = false;
        if (cuda.sample.ruleFor(s.sampling)) |r| {
            try b.upload(l.s, l.rule.ptr, std.mem.asBytes(&r));
            l.sampled = true;
        }
        l.len = 0;
        try l.m.reset(l.s);
        const at = resumeSaved(l, s);
        l.len = at;
        l.fill = at;
        s.fill_at.store(@intCast(at), .monotonic);
        return l;
    }

    fn chunk(b: *Backend, s: *lanes.Stream, l: *Lane) !bool {
        const prompt = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(prompt.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mark| {
            if (mark > at and mark < end) end = mark;
        }
        const done = end == prompt.len;
        try l.m.forward(b.k, l.s, prompt[at..end], at, end - at - 1, @intFromBool(done), false);
        if (done) try b.d.check(b.d.api.cuMemcpyDtoDAsync_v2(l.last.ptr, l.m.s.logits, l.m.w.c.text.unpadded_vocab * 4, l.s.handle), "cuMemcpyDtoDAsync");
        l.len = end;
        l.fill = if (done) null else end;
        s.fill_at.store(@intCast(end), .monotonic);
        if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null)
            hook.at(hook.ptr, s, @intCast(end));
        return done;
    }

    fn handle(b: *Backend) usize {
        const h = b.next;
        b.next += 1;
        return @intCast(h % token_ring);
    }

    fn drawRows(b: *Backend, l: *Lane, logits: u64, rows: usize, position0: u64, at: usize) !void {
        const v = l.m.w.c.text.unpadded_vocab;
        if (!l.sampled) return ops.top1(&b.k.glue, l.s, logits, rows, v, v, 0, l.m.s.t + at * 4, b.tokens.ptr + at * 4);
        try ops.tobf(&b.k.glue, l.s, logits, rows, v, v, l.logits16.ptr, v, 0, v);
        const metadata = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        try b.upload(l.s, l.meta.ptr, std.mem.asBytes(&metadata));
        try b.k.draws.draw(l.s, l.logits16.ptr, v, l.rule.ptr, l.meta.ptr, 0, b.tokens.ptr + at * 4, rows, null, null);
    }

    /// A ring token was drawn on its owner's stream; the ring does not record which, so every lane lands first.
    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const b = self(ptr);
        if (h >= b.next or b.next - h > token_ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        for (b.lanes) |l| if (l.owner != null) try l.s.synchronize();
        try b.tokens.download(@intCast((h % token_ring) * 4), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    const vtable: be.Backend.VTable = .{
        .prefill = prefill,
        .prefill_step = prefillStep,
        .first = first,
        .first_masked = firstMasked,
        .queue = queue,
        .read = read,
        .verify = verify,
        .keep = keep,
        .draft = draft,
        .release = release,
        .reports_fill = true,
    };

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        const l = try b.opening(s);
        errdefer release(ptr, s);
        while (!try b.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const b = self(ptr);
        const going = b.laneOf(s) catch null;
        const l = if (going != null and going.?.fill != null) going.? else try b.opening(s);
        errdefer release(ptr, s);
        return b.chunk(s, l);
    }

    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = b.next;
        try b.drawRows(l, l.last.ptr, 1, position, b.handle());
        return h;
    }

    fn firstMasked(_: *anyopaque, _: *lanes.Stream, _: u64, _: []const u32) anyerror!u64 {
        return error.StructuresUnsupported;
    }

    fn feed(b: *Backend, value: be.Feed) !u32 {
        return switch (value) {
            .value => |v| v,
            .handle => |h| read(b, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, value: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        const token = try b.feed(value);
        if (position != l.len + 1) return error.PositionMismatch;
        try l.m.forward(b.k, l.s, &.{token}, l.len, 0, 1, false);
        l.len += 1;
        const h = b.next;
        try b.drawRows(l, l.m.s.logits, 1, position, b.handle());
        return h;
    }

    /// Every window's forward and draw is queued on its own lane before the first wait.
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len != out.len) return error.WindowCount;
        for (windows) |w| {
            if (w.held != 0) return error.NoHeldDrafts;
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            const l = try b.laneOf(w.stream);
            const n = w.rows();
            if (n == 0 or n > window_rows) return error.WindowTooWide;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(b.gpa, w.pending);
            try l.rows.appendSlice(b.gpa, w.tokens);
            for (w.positions, 0..) |position, i| if (position != l.base + i + 1) return error.PositionMismatch;
            try l.m.forward(b.k, l.s, l.rows.items, l.base, 0, n, true);
            try b.drawRows(l, l.m.s.logits, n, w.positions[0], l.at);
        }
        for (windows, out) |w, o| {
            const l = try b.laneOf(w.stream);
            const n = w.rows();
            try l.s.synchronize();
            try b.tokens.download(l.at * 4, std.mem.sliceAsBytes(o.sampled[0..n]));
            @memcpy(o.drafts[0 .. n - 1], l.rows.items[1..n]);
            l.len = l.base + n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != paths.len) return error.WindowCount;
        for (windows, paths) |w, path| {
            const l = try b.laneOf(w.stream);
            if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
            for (path, 0..) |row, i| if (row != i) return error.ChainsOnly;
            try l.m.rewind(b.k, l.s, path.len);
            l.len = l.base + path.len;
        }
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    /// A snapshot lands before it returns: another lane may restore it on its own stream.
    pub fn saveState(b: *Backend, owner: ?*anyopaque, at: u32) !*Model.Saved {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        if ((l.fill orelse l.len) != at) return error.NotAtMark;
        const saved = try l.m.save(l.s, at);
        errdefer l.m.drop(l.s, saved);
        try l.s.synchronize();
        return saved;
    }

    pub fn restoreState(b: *Backend, owner: ?*anyopaque, saved: *const Model.Saved) !void {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        try l.m.restore(l.s, saved);
        l.len = saved.at;
    }

    /// Any lane may still be reading the snapshot: all of them land before its bytes go.
    pub fn dropState(b: *Backend, saved: *Model.Saved) void {
        for (b.lanes) |l| l.s.synchronize() catch {};
        b.lanes[0].m.drop(b.lanes[0].s, saved);
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        const l = b.laneOf(s) catch return;
        l.rows.clearRetainingCapacity();
        l.owner = null;
    }
};
