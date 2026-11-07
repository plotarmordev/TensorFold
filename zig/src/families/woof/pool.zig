//! Persistent workers for Woof's per-frame arithmetic: a job is split into parts, part 0 on the caller's thread.
//! Workers spin between the speech loop's back-to-back jobs and sleep on a futex when idle.
const std = @import("std");
const builtin = @import("builtin");

const max_threads = 32;
/// Spins (about a millisecond) before an idle worker sleeps.
const spins = 1 << 15;

pub const Pool = struct {
    io: std.Io,
    threads: []std.Thread,
    job: Job = undefined,
    /// Bumped once per job; workers wait for it to move.
    epoch: std.atomic.Value(u32) = .init(0),
    pending: std.atomic.Value(u32) = .init(0),
    sleepers: std.atomic.Value(u32) = .init(0),
    stop: bool = false,

    const Job = struct { ctx: *const anyopaque, run: *const fn (ctx: *const anyopaque, part: usize, parts: usize) void, parts: usize };

    /// `threads` in all, the caller's included: 0 picks the performance cores.
    pub fn init(p: *Pool, a: std.mem.Allocator, io: std.Io, threads: usize) !void {
        const n = @min(max_threads, if (threads == 0) performanceCores() else threads);
        p.* = .{ .io = io, .threads = try a.alloc(std.Thread, n - 1) };
        for (p.threads, 1..) |*t, part| t.* = try std.Thread.spawn(.{}, work, .{ p, part });
    }

    pub fn deinit(p: *Pool, a: std.mem.Allocator) void {
        p.stop = true;
        p.dispatch();
        for (p.threads) |t| t.join();
        a.free(p.threads);
    }

    pub fn size(p: *const Pool) usize {
        return p.threads.len + 1;
    }

    /// f(ctx, part, parts) for part < parts (at most size()), returning when all have run.
    pub fn each(p: *Pool, parts: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void {
        std.debug.assert(parts <= p.size());
        const C = @TypeOf(ctx);
        const run = struct {
            fn run(c: *const anyopaque, part: usize, n: usize) void {
                f(@as(*const C, @ptrCast(@alignCast(c))).*, part, n);
            }
        }.run;
        if (parts <= 1) return f(ctx, 0, parts);
        p.job = .{ .ctx = @ptrCast(&ctx), .run = run, .parts = parts };
        p.pending.store(@intCast(p.threads.len), .release);
        p.dispatch();
        f(ctx, 0, parts);
        while (p.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
    }

    fn dispatch(p: *Pool) void {
        _ = p.epoch.fetchAdd(1, .seq_cst);
        if (p.sleepers.load(.seq_cst) != 0) p.io.futexWake(u32, &p.epoch.raw, max_threads);
    }

    fn work(p: *Pool, part: usize) void {
        var seen: u32 = 0;
        while (true) {
            var spun: usize = 0;
            while (p.epoch.load(.acquire) == seen) : (spun += 1) {
                if (spun < spins) {
                    std.atomic.spinLoopHint();
                    continue;
                }
                _ = p.sleepers.fetchAdd(1, .seq_cst);
                if (p.epoch.load(.seq_cst) == seen) p.io.futexWaitUncancelable(u32, &p.epoch.raw, seen);
                _ = p.sleepers.fetchSub(1, .seq_cst);
            }
            seen = p.epoch.load(.acquire);
            if (p.stop) return;
            if (part < p.job.parts) p.job.run(p.job.ctx, part, p.job.parts);
            _ = p.pending.fetchSub(1, .release);
        }
    }
};

/// Performance cores on Apple silicon (efficiency cores would stall every job), else every core.
fn performanceCores() usize {
    if (builtin.os.tag == .macos) {
        var n: c_int = 0;
        var len: usize = @sizeOf(c_int);
        if (std.c.sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, null, 0) == 0 and n > 0) return @intCast(n);
    }
    return std.Thread.getCpuCount() catch 1;
}

test "pool: every part runs once per job, over back-to-back jobs and after the workers sleep" {
    var p: Pool = undefined;
    try p.init(std.testing.allocator, std.testing.io, 4);
    defer p.deinit(std.testing.allocator);
    var hits: [4]std.atomic.Value(u32) = @splat(.init(0));
    const Add = struct {
        fn add(h: *[4]std.atomic.Value(u32), part: usize, _: usize) void {
            _ = h[part].fetchAdd(1, .monotonic);
        }
    };
    for (0..1000) |_| p.each(4, &hits, Add.add);
    try std.testing.io.sleep(.fromMilliseconds(50), .awake);
    p.each(3, &hits, Add.add);
    for (hits[0..3]) |h| try std.testing.expectEqual(1001, h.load(.monotonic));
    try std.testing.expectEqual(1000, hits[3].load(.monotonic));
}
