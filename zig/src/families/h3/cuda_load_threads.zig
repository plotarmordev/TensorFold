//! Threaded weight reads for the CUDA transformer: a checkpoint's tensors cut into read jobs that a
//! crew of reader threads pread straight into one pinned stage at a time, the two stages alternating
//! with the GPU upload, as mmh3's eight-thread loader does. Host-only: the device side stays with the
//! caller, so the plan and its failure cases are testable without a GPU.
const std = @import("std");
const Shards = @import("checkpoint.zig").Shards;

/// One pinned stage's room; the two stages alternate, each filled while the other's copy runs.
pub const stage_bytes = 64 << 20;
/// One reader's share of a window; jobs are cut no bigger, so `threads` readers share a fill.
pub const job_bytes = 8 << 20;
/// The most reader threads a load starts.
pub const max_readers = 16;

const c = struct {
    extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, offset: i64) isize;
};

const sync = struct {
    // Only the futex backend is used; no workers, as copy_source_sync does.
    var backend: std.Io.Threaded = .init_single_threaded;
    pub const Mutex = struct {
        inner: std.Io.Mutex = .init,
        pub fn lock(m: *Mutex) void {
            m.inner.lockUncancelable(backend.io());
        }
        pub fn unlock(m: *Mutex) void {
            m.inner.unlock(backend.io());
        }
    };
    pub const Condition = struct {
        inner: std.Io.Condition = .init,
        pub fn wait(cc: *Condition, m: *Mutex) void {
            cc.inner.waitUncancelable(backend.io(), &m.inner);
        }
        pub fn broadcast(cc: *Condition) void {
            cc.inner.broadcast(backend.io());
        }
    };
};

/// A tensor a load reads from its shard, tagged so the caller maps it onto its destination.
pub const Fill = struct {
    /// The caller's destination identity (in `cuda_weights`: block * 5 + buffer index).
    tag: usize,
    /// The tensor's byte offset inside that destination.
    at: usize,
    /// The shard's `pread` descriptor and the tensor's absolute offset in the file.
    fd: c_int,
    file: usize,
    len: usize,
    /// The gate-free release's fused qkv stacking, for the qkv fill only.
    fused: ?Fused = null,
};

/// A fused qkv product, (heads, [q, k, v], 128) rows, which the device holds stacked: ([q, k, v], heads, 128).
pub const Fused = struct {
    heads: usize,
    /// One row's bytes.
    row: usize,

    /// A head part's bytes: the unit the stacking moves.
    pub fn part(f: Fused) usize {
        return 128 * f.row;
    }

    /// The stored byte that lands at stacked byte `at`.
    pub fn source(f: Fused, at: usize) usize {
        const unit = f.part();
        const qkv = at / (f.heads * unit);
        const head = at / unit % f.heads;
        return (head * 3 + qkv) * unit + at % unit;
    }
};

/// A piece of one fill, read into `staged .. staged + len` of its window. `at` is the piece's offset
/// inside the fill's staged image (the destination order), `len` a whole number of fused parts.
pub const Job = struct {
    fill: usize,
    at: usize,
    staged: usize,
    len: usize,
};

/// One copy out of a stage: `len` staged bytes from `staged` to `to` inside `tag`'s destination.
pub const Copy = struct { tag: usize, to: usize, staged: usize, len: usize };

/// One window's work: the jobs that fill it and the copy ranges that upload it.
pub const Window = struct {
    jobs: []const Job,
    copies: []const Copy,
    bytes: usize,
};

/// The fills cut into per-window jobs and copy ranges; `deinit` releases them.
pub const Plan = struct {
    gpa: std.mem.Allocator,
    windows: []Window,

    pub fn deinit(p: *Plan) void {
        for (p.windows) |w| {
            p.gpa.free(w.jobs);
            p.gpa.free(w.copies);
        }
        p.gpa.free(p.windows);
    }
};

/// `fills` in staged order into windows of at most `stage` bytes, jobs no bigger than `job`.
/// A window's copy list coalesces adjacent jobs of one destination into one range.
pub fn plan(gpa: std.mem.Allocator, fills: []const Fill, stage: usize, job: usize) !Plan {
    var windows: std.ArrayList(Window) = .empty;
    var jobs: std.ArrayList(Job) = .empty;
    var copies: std.ArrayList(Copy) = .empty;
    errdefer {
        for (windows.items) |w| {
            gpa.free(w.jobs);
            gpa.free(w.copies);
        }
        windows.deinit(gpa);
        jobs.deinit(gpa);
        copies.deinit(gpa);
    }
    var staged: usize = 0; // the open window's room already taken
    for (fills, 0..) |f, i| {
        if (f.len == 0 or (f.fused != null and f.len % f.fused.?.part() != 0)) return error.BadGeometry;
        // fused jobs stay whole parts; a part bigger than `job` is one job of its own
        const piece = if (f.fused) |fu| fu.part() * @max(1, job / fu.part()) else job;
        var at: usize = 0;
        while (at < f.len) {
            const n = @min(piece, f.len - at);
            if (staged > 0 and staged + n > stage) {
                try windows.append(gpa, .{ .jobs = try jobs.toOwnedSlice(gpa), .copies = try copies.toOwnedSlice(gpa), .bytes = staged });
                staged = 0;
            }
            try jobs.append(gpa, .{ .fill = i, .at = at, .staged = staged, .len = n });
            if (copies.items.len > 0) {
                const tail = &copies.items[copies.items.len - 1];
                if (tail.tag == f.tag and tail.to + tail.len == f.at + at) {
                    tail.len += n;
                } else try copies.append(gpa, .{ .tag = f.tag, .to = f.at + at, .staged = staged, .len = n });
            } else try copies.append(gpa, .{ .tag = f.tag, .to = f.at + at, .staged = staged, .len = n });
            staged += n;
            at += n;
        }
    }
    if (staged > 0) try windows.append(gpa, .{ .jobs = try jobs.toOwnedSlice(gpa), .copies = try copies.toOwnedSlice(gpa), .bytes = staged });
    return .{ .gpa = gpa, .windows = try windows.toOwnedSlice(gpa) };
}

/// The job's `at .. at + len` staged bytes into `dst`, from the shard's file. A fused fill reads one
/// head part a call; a missing or unreadable range fails the load.
fn fill(f: *const Fill, dst: []u8, at: usize) !void {
    if (f.fused) |fu| {
        const unit = fu.part();
        var off: usize = 0;
        while (off < dst.len) : (off += unit) try into(f.fd, dst[off..][0..unit], f.file + fu.source(at + off));
    } else try into(f.fd, dst, f.file + at);
}

fn into(fd: c_int, dst: []u8, at: usize) !void {
    var off: i64 = @intCast(at);
    var left = dst;
    while (left.len > 0) {
        const got = c.pread(fd, left.ptr, left.len, off);
        if (got <= 0) return error.LoadRead;
        off += got;
        left = left[@intCast(got)..];
    }
}

/// `threads` readers filling one window a generation; with no readers `run` fills inline.
pub const Crew = struct {
    work: Work = .{},
    threads: [max_readers]?std.Thread = @splat(null),

    const Work = struct {
        mutex: sync.Mutex = .{},
        changed: sync.Condition = .{},
        /// The window being filled this generation.
        gen: usize = 0,
        window: []u8 = &.{},
        fills: []const Fill = &.{},
        jobs: []const Job = &.{},
        next: std.atomic.Value(usize) = .init(0),
        done: std.atomic.Value(usize) = .init(0),
        failed: std.atomic.Value(bool) = .init(false),
        err: anyerror = error.Unexpected,
        quit: bool = false,
    };

    /// Spawns the readers; they idle until `run` publishes the first window.
    pub fn start(crew: *Crew, threads: usize) !void {
        crew.* = .{};
        errdefer crew.stop();
        for (crew.threads[0..@min(threads, max_readers)]) |*t| t.* = try std.Thread.spawn(.{}, worker, .{crew});
    }

    /// Every reader fills `jobs`' staged bytes of `window`; an unreadable or short range fails.
    /// `fills` and `jobs` must stay alive until `run` returns.
    pub fn run(crew: *Crew, fills: []const Fill, jobs: []const Job, window: []u8) !void {
        const w = &crew.work;
        if (crew.threads[0] == null) {
            for (jobs) |j| try fill(&fills[j.fill], window[j.staged..][0..j.len], j.at);
            return;
        }
        w.mutex.lock();
        w.gen += 1;
        w.window = window;
        w.fills = fills;
        w.jobs = jobs;
        w.next.store(0, .monotonic);
        w.done.store(0, .monotonic);
        w.failed.store(false, .release);
        w.changed.broadcast();
        while (w.done.load(.monotonic) < jobs.len and !w.failed.load(.acquire)) w.changed.wait(&w.mutex);
        const err: ?anyerror = if (w.failed.load(.acquire)) w.err else null;
        w.mutex.unlock();
        if (err) |e| return e;
    }

    /// Joins the readers; a published window's fill has finished or failed before this returns.
    pub fn stop(crew: *Crew) void {
        const w = &crew.work;
        w.mutex.lock();
        w.quit = true;
        w.changed.broadcast();
        w.mutex.unlock();
        for (&crew.threads) |*t| if (t.*) |thread| {
            thread.join();
            t.* = null;
        };
    }

    fn worker(crew: *Crew) void {
        const w = &crew.work;
        var seen: usize = 0;
        while (true) {
            w.mutex.lock();
            while (!w.quit and w.gen == seen) w.changed.wait(&w.mutex);
            if (w.quit) {
                w.mutex.unlock();
                return;
            }
            seen = w.gen;
            const fills = w.fills;
            const jobs = w.jobs;
            const window = w.window;
            w.mutex.unlock();
            while (!w.failed.load(.acquire)) {
                const i = w.next.fetchAdd(1, .monotonic);
                if (i >= jobs.len) break;
                const j = jobs[i];
                fill(&fills[j.fill], window[j.staged..][0..j.len], j.at) catch |e| {
                    w.mutex.lock();
                    if (!w.failed.load(.monotonic)) w.err = e;
                    w.failed.store(true, .release);
                    w.done.store(jobs.len, .monotonic);
                    w.changed.broadcast();
                    w.mutex.unlock();
                    return;
                };
                if (w.done.fetchAdd(1, .monotonic) + 1 == jobs.len) {
                    w.mutex.lock();
                    w.changed.broadcast();
                    w.mutex.unlock();
                }
            }
        }
    }
};

test "h3 threaded load: a missing shard and a short shard both fail before any copy" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir_path);

    try std.testing.expectError(error.CannotOpen, Shards.open(gpa, dir_path, 1));

    // the header's data_offsets pass the file's tail, so opening the shard fails before a reader runs
    const header = try std.fmt.allocPrint(gpa, "{{\"blocks.0.attn.qkv_proj.weight\":{{\"dtype\":\"BF16\",\"shape\":[3,1],\"data_offsets\":[0,6]}},\"pad\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[6,8]}}}}", .{});
    defer gpa.free(header);
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    try bytes.appendNTimes(gpa, 0, 8);
    std.mem.writeInt(u64, bytes.items[0..8], header.len, .little);
    try bytes.appendSlice(gpa, header);
    try bytes.appendSlice(gpa, &[8]u8{ 9, 8, 7, 6, 5, 4, 0, 0 });
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = bytes.items });
    // the second shard one byte short of its header's promise: the open's header parse fails
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = bytes.items[0 .. bytes.items.len - 1] });
    var short = Shards.open(gpa, dir_path, 2);
    defer if (short) |*ck| ck.deinit(gpa) else |_| {};
    try std.testing.expectError(error.BadHeader, short);
}

test "h3 threaded load: readers stage a shard's bytes, fused qkv stacked per head" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir_path);
    var data: [768]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i *% 37);
    const header = try std.fmt.allocPrint(gpa, "{{\"t\":{{\"dtype\":\"BF16\",\"shape\":[{d},1],\"data_offsets\":[0,{d}]}}}}", .{ data.len / 2, data.len });
    defer gpa.free(header);
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    try bytes.appendNTimes(gpa, 0, 8);
    std.mem.writeInt(u64, bytes.items[0..8], header.len, .little);
    try bytes.appendSlice(gpa, header);
    try bytes.appendSlice(gpa, &data);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = bytes.items });
    var ck = try Shards.open(gpa, dir_path, 1);
    defer ck.deinit(gpa);
    const e = try ck.bf16("t");

    // plain: two windows of 512 staged bytes, three readers on 256-byte jobs
    const fills = [_]Fill{.{ .tag = 0, .at = 0, .fd = e.fd, .file = e.file, .len = e.bytes.len }};
    var p = try plan(gpa, &fills, 512, 256);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 2), p.windows.len);
    var staged: [data.len]u8 = @splat(0);
    var crew: Crew = .{};
    try crew.start(3);
    defer crew.stop();
    var off: usize = 0;
    for (p.windows) |w| {
        try crew.run(&fills, w.jobs, staged[off..][0..w.bytes]);
        off += w.bytes;
    }
    try std.testing.expectEqualSlices(u8, &data, &staged);
    // no readers fills the same jobs inline
    @memset(&staged, 0);
    off = 0;
    var single: Crew = .{};
    for (p.windows) |w| {
        try single.run(&fills, w.jobs, staged[off..][0..w.bytes]);
        off += w.bytes;
    }
    try std.testing.expectEqualSlices(u8, &data, &staged);
    // an unreadable descriptor fails the fill, before the window's copy would run
    const bad = [_]Fill{.{ .tag = 0, .at = 0, .fd = -1, .file = 0, .len = 64 }};
    try std.testing.expectError(error.LoadRead, single.run(&bad, &.{.{ .fill = 0, .at = 0, .staged = 0, .len = 64 }}, staged[0..64]));
    try std.testing.expectError(error.LoadRead, crew.run(&bad, &.{.{ .fill = 0, .at = 0, .staged = 0, .len = 64 }}, staged[0..64]));

    // fused: two heads of six 128-byte parts, staged in the stacked [q,k,v] per part order
    const fused = Fused{ .heads = 2, .row = 1 };
    const qkv = [_]Fill{.{ .tag = 0, .at = 0, .fd = e.fd, .file = e.file, .len = e.bytes.len, .fused = fused }};
    var fp = try plan(gpa, &qkv, data.len, fused.part() * 2);
    defer fp.deinit();
    var stacked: [data.len]u8 = @splat(0);
    try crew.run(&qkv, fp.windows[0].jobs, &stacked);
    var want: [data.len]u8 = undefined;
    for (&want, 0..) |*b, at| b.* = data[fused.source(at)];
    try std.testing.expectEqualSlices(u8, &want, &stacked);
}
