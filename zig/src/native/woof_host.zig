//! Underdog Woof on CPU: one worker thread speaks queued requests in submit order.
const std = @import("std");
const api = @import("engine_api");
const woof = @import("woof");
const Allocator = std.mem.Allocator;

/// Pending requests at most; a 17th submit is Busy rather than a longer line.
const max_jobs = 16;

const Job = struct {
    id: api.Id,
    request: *const api.SpeechRequest,
    sink: api.Sink,
    cancelled: std.atomic.Value(bool) = .init(false),
};

const Host = struct {
    gpa: Allocator,
    io: std.Io,
    model: *woof.Model,
    jobs: std.ArrayList(Job) = .empty,
    running: ?Job = null,
    info_: api.Info,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    worker: ?std.Thread = null,
    closing: bool = false,
    const vtable: api.Engine.VTable = .{
        .info = info,
        .submit = submitText,
        .submit_speech = submit,
        .cancel = cancel,
        .status = status,
        .memory = memory,
        .drain = drain,
    };
    fn self(p: *anyopaque) *Host {
        return @ptrCast(@alignCast(p));
    }
    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }
    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }
    fn info(p: *anyopaque) api.Info {
        return self(p).info_;
    }
    fn submitText(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn valid(h: *Host, r: *const api.SpeechRequest) bool {
        const sp = h.info_.speech.?;
        return r.prompt.len > 0 and r.voice < sp.voices.len and
            r.pcm.len >= @as(usize, r.max_frames) * sp.frame_length;
    }
    fn submit(p: *anyopaque, id: api.Id, r: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const h = self(p);
        if (!h.valid(r)) return error.InvalidRequest;
        h.lock();
        defer h.unlock();
        if (h.closing) return error.Closed;
        for (h.jobs.items) |j| if (j.id == id) return error.InvalidRequest;
        if (h.jobs.items.len == max_jobs) return error.Busy;
        r.result.* = .{};
        h.jobs.append(h.gpa, .{ .id = id, .request = r, .sink = sink }) catch return error.Busy;
        h.wake.signal(h.io);
    }
    fn cancel(p: *anyopaque, id: api.Id) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        if (h.running) |*j| if (j.id == id) j.cancelled.store(true, .release);
        for (h.jobs.items) |*j| if (j.id == id) j.cancelled.store(true, .release);
        h.wake.signal(h.io);
    }
    fn status(p: *anyopaque, out: *api.Status, _: []u32) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        out.* = .{};
        out.running = @intCast(h.jobs.items.len + @intFromBool(h.running != null));
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    /// The piece a speak call publishes: its samples so far, cancelled when the client left.
    fn piece(p: *anyopaque, samples: usize) bool {
        const j: *Job = @ptrCast(@alignCast(p));
        j.request.result.samples = samples;
        if (j.request.stream) j.sink.event(j.sink.ctx, j.id, &.{ .audio = samples });
        return !j.cancelled.load(.acquire);
    }
    fn run(h: *Host) void {
        while (true) {
            h.lock();
            if (h.jobs.items.len == 0) {
                if (h.closing) {
                    h.unlock();
                    return;
                }
                h.wake.waitUncancelable(h.io, &h.mutex);
                h.unlock();
                continue;
            }
            h.running = h.jobs.orderedRemove(0);
            const closing = h.closing;
            h.unlock();
            const j = &h.running.?;
            if (j.cancelled.load(.acquire) or closing) finish(j, .cancelled, "") else h.speak(j);
            h.lock();
            h.running = null;
            h.unlock();
        }
    }
    fn speak(h: *Host, j: *Job) void {
        const progress: woof.Progress = .{ .ctx = j, .piece = Host.piece };
        const spoken = h.model.speak(j.request.prompt, j.request.voice, j.request.seed, j.request.pcm, progress) catch |err|
            return finish(j, .failed, @errorName(err));
        j.request.result.samples = spoken.samples;
        finish(j, switch (spoken.reason) {
            .stop => .stop,
            .length => .length,
            .cancelled => .cancelled,
        }, "");
    }
    fn finish(j: *Job, reason: api.Reason, message: []const u8) void {
        j.request.result.reason = reason;
        j.sink.event(j.sink.ctx, j.id, &.{ .finished = .{ .reason = reason, .message = message } });
    }
    fn drain(p: *anyopaque) void {
        const h = self(p);
        h.lock();
        h.closing = true;
        for (h.jobs.items) |*j| j.cancelled.store(true, .release);
        h.wake.signal(h.io);
        h.unlock();
        if (h.worker) |t| t.join();
        h.worker = null;
    }
    fn close(p: *anyopaque) void {
        const h = self(p);
        drain(p);
        h.jobs.deinit(h.gpa);
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

/// The engine for `o.dir`, or null with `problem` set when a flag Woof does not read is given.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.tp != 1 or o.rank != 0 or o.context != null or o.max_frames != null or
        o.vision != null or o.precision != null or o.record != null or o.recall or o.learn != null or
        o.learn_conversations or o.drafter != null or o.mtp_drafts != null or o.mtp_confidence != null or
        o.speed_up != null or o.slide or (o.prompt_cache_gib != null and o.prompt_cache_gib.? != 0) or
        (o.keep != null and o.keep.? != 0) or o.policy.len != 0 or o.prompt_int8 or
        (o.device != null and o.device.? != 0) or (o.segments != null and o.segments.? != 1))
    {
        problem.* = "Woof serves speech on this machine's CPU alone: serve without text, vision, cache or device flags";
        return null;
    }
    const lanes = if (o.lanes_fixed) o.lanes else 4;
    if (o.lanes_fixed and (o.lanes == 0 or o.lanes > max_jobs)) {
        problem.* = "Woof queues 1 through 16 speech requests; use --parallel in that range";
        return null;
    }
    const model = woof.Model.load(gpa, io, o.dir, o.voices, 0) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native Woof engine could not load the checkpoint ({s})", .{@errorName(e)});
        return null;
    };
    const h = try gpa.create(Host);
    errdefer {
        model.deinit();
        gpa.destroy(h);
    }
    h.* = .{
        .gpa = gpa,
        .io = io,
        .model = model,
        .info_ = .{
            .name = "woof",
            .text_generation = false,
            .lanes = lanes,
            .plain_only = true,
            .speech = .{
                .sample_rate = woof.sample_rate,
                .frame_length = woof.frame_length,
                .max_frames = woof.max_frames,
                .max_prompt_tokens = 4096,
                .voices = model.voice_names,
            },
        },
    };
    try h.jobs.ensureTotalCapacity(gpa, @min(lanes, max_jobs));
    errdefer h.jobs.deinit(gpa);
    h.worker = try std.Thread.spawn(.{}, Host.run, .{h});
    return .{ .engine = .{ .ctx = h, .vtable = &Host.vtable }, .ctx = h, .close = Host.close };
}
