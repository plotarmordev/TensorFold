//! The Mac's GPU turn between live speech and long batch work (a picture's render, a benchmark).
//! A speech request holds a shared flock on `path` while it synthesizes; batch work, between its own short
//! command buffers, waits until no speech holds it. Batch work never takes the lock itself: speech never waits.
const std = @import("std");

pub const path = "/tmp/tensorfold-speech.lock";

/// How often waiting batch work looks again; one poll is the most speech's end can be missed by.
const poll: std.Io.Duration = .fromMilliseconds(20);

/// One speech request's hold on the turn, or none if the file cannot be opened or something odd holds it
/// exclusively: speech runs either way, and never waits for the lock.
pub fn speak(io: std.Io, at: []const u8) ?std.Io.File {
    const file = open(io, at) catch return null;
    if (file.tryLock(io, .shared) catch false) return file;
    file.close(io);
    return null;
}

pub fn done(io: std.Io, held: ?std.Io.File) void {
    if (held) |file| file.close(io);
}

/// Waits while any speech request holds the turn; the seconds waited. An unusable lock file waits for nothing.
pub fn yield(io: std.Io, at: []const u8) f64 {
    const file = open(io, at) catch return 0;
    defer file.close(io);
    const started = std.Io.Clock.awake.now(io);
    while (true) {
        const free = file.tryLock(io, .exclusive) catch break;
        if (free) {
            file.unlock(io);
            break;
        }
        io.sleep(poll, .awake) catch break;
    }
    return @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
}

/// Read-only so the speech server's and a render's users can share one file; flock needs no write access.
fn open(io: std.Io, at: []const u8) !std.Io.File {
    const dir = std.Io.Dir.cwd();
    return dir.openFile(io, at, .{}) catch |e| switch (e) {
        error.FileNotFound => {
            const file = try dir.createFile(io, at, .{ .read = true, .truncate = false });
            file.setPermissions(io, .fromMode(0o644)) catch {}; // past the creator's umask, so the other user opens it
            return file;
        },
        else => e,
    };
}

test "batch work waits for every speech hold to end, and not at all without one" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const at = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(at);
    const lock = try std.fs.path.join(std.testing.allocator, &.{ at, "speech.lock" });
    defer std.testing.allocator.free(lock);

    try std.testing.expect(yield(io, lock) < 0.015);
    const first = speak(io, lock) orelse return error.NoHold;
    const second = speak(io, lock) orelse return error.NoHold; // two voices at once share the turn
    const Ender = struct {
        fn run(h: [2]std.Io.File) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(150), .awake) catch {};
            done(std.testing.io, h[0]);
            std.Io.sleep(std.testing.io, .fromMilliseconds(150), .awake) catch {};
            done(std.testing.io, h[1]);
        }
    };
    const ender = try std.Thread.spawn(.{}, Ender.run, .{[2]std.Io.File{ first, second }});
    const waited = yield(io, lock);
    ender.join();
    try std.testing.expect(waited >= 0.29);
    try std.testing.expect(yield(io, lock) < 0.015);
}

test "speech never waits on an exclusive holder, and the file it makes is open to every user" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const at = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(at);
    const lock = try std.fs.path.join(std.testing.allocator, &.{ at, "speech.lock" });
    defer std.testing.allocator.free(lock);

    const holder = try open(io, lock);
    defer holder.close(io);
    if (@import("builtin").os.tag != .windows) try std.testing.expectEqual(@as(u32, 0o644), (try holder.stat(io)).permissions.toMode() & 0o777);
    try holder.lock(io, .exclusive);
    try std.testing.expect(speak(io, lock) == null);
}
