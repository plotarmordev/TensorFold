//! The EXL3 encoder's work sharing: threads take the next index from a shared counter.
const std = @import("std");

pub const max_threads = 64;

/// Runs body(ctx, i) for every i < count on up to `threads` threads, each taking the next index until none is left.
pub fn forEach(threads: usize, count: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize) void) void {
    const Pool = struct {
        ctx: @TypeOf(ctx),
        count: usize,
        next: std.atomic.Value(usize) = .init(0),

        fn run(pool: *@This()) void {
            while (true) {
                const i = pool.next.fetchAdd(1, .monotonic);
                if (i >= pool.count) return;
                body(pool.ctx, i);
            }
        }
    };
    var pool: Pool = .{ .ctx = ctx, .count = count };
    var spawned: [max_threads]?std.Thread = @splat(null);
    const extra = @min(threads, count, max_threads) -| 1;
    // a thread that fails to start leaves its share to the others
    for (spawned[0..extra]) |*t| t.* = std.Thread.spawn(.{}, Pool.run, .{&pool}) catch null;
    pool.run();
    for (spawned[0..extra]) |t| if (t) |thread| thread.join();
}

test "forEach runs every index exactly once, whatever the thread count" {
    const Hits = struct {
        fn hit(counts: []std.atomic.Value(u32), i: usize) void {
            _ = counts[i].fetchAdd(1, .monotonic);
        }
    };
    for ([_][2]usize{ .{ 1, 10 }, .{ 3, 10 }, .{ 16, 5 }, .{ 7, 1 }, .{ 4, 0 }, .{ 200, 130 } }) |case| {
        var counts: [130]std.atomic.Value(u32) = @splat(.init(0));
        forEach(case[0], case[1], counts[0..case[1]], Hits.hit);
        for (counts[0..case[1]]) |c| try std.testing.expectEqual(@as(u32, 1), c.load(.monotonic));
    }
}

test "forEach keeps exactly `threads` bodies running at once" {
    const io = std.testing.io;
    const Gate = struct {
        threads: usize,
        deadline: std.Io.Timestamp,
        entered: std.atomic.Value(usize) = .init(0),
        inside: std.atomic.Value(usize) = .init(0),
        peak: std.atomic.Value(usize) = .init(0),

        // wait (bounded) until `threads` have entered, then linger so a surplus thread would raise the peak
        fn body(gate: *@This(), _: usize) void {
            _ = gate.peak.fetchMax(gate.inside.fetchAdd(1, .acq_rel) + 1, .acq_rel);
            _ = gate.entered.fetchAdd(1, .acq_rel);
            while (gate.entered.load(.acquire) < gate.threads and std.Io.Clock.awake.now(io).compare(.lt, gate.deadline))
                std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
            _ = gate.inside.fetchSub(1, .acq_rel);
        }
    };
    for ([_]usize{ 2, 7, 36, max_threads }) |threads| {
        var gate: Gate = .{ .threads = threads, .deadline = std.Io.Clock.awake.now(io).addDuration(.fromSeconds(30)) };
        forEach(threads, threads * 4, &gate, Gate.body);
        try std.testing.expectEqual(threads, gate.peak.load(.acquire));
    }
}
