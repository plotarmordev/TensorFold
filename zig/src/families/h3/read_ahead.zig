//! Weight bytes read into the page cache before the GPU or a loader touches their mapping.
//!
//! A cold mapping fills one fault at a time: 0.06 GB/s on an M5 Ultra's SSD, where 16 readers' 8 MiB preads take
//! 3-5 GB/s. Core's `checkpoint_prefetch` does the same for core checkpoints on Linux; the H3 tools map their own files.
const std = @import("std");

/// A byte range of one file.
pub const Span = struct { path: [*:0]const u8, lo: usize, hi: usize };

/// The bytes of one pread.
const piece = 8 << 20;
/// Readers at once, the calling thread among them.
const readers = 16;

const Work = struct {
    spans: []const Span,
    /// The next piece to read.
    next: std.atomic.Value(usize) = .init(0),
    read: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    /// Piece `n`, counted across every span in order, or null past the last.
    fn take(w: *Work, n: usize) ?Span {
        var left = n;
        for (w.spans) |s| {
            const count = std.math.divCeil(usize, s.hi - s.lo, piece) catch unreachable;
            if (left < count) {
                const lo = s.lo + left * piece;
                return .{ .path = s.path, .lo = lo, .hi = @min(lo + piece, s.hi) };
            }
            left -= count;
        }
        return null;
    }

    fn run(w: *Work) void {
        const buf = std.heap.page_allocator.alloc(u8, piece) catch return w.failed.store(true, .monotonic);
        defer std.heap.page_allocator.free(buf);
        var path: [*:0]const u8 = "";
        var fd: std.c.fd_t = -1;
        defer if (fd >= 0) close(fd);
        while (!w.failed.load(.monotonic)) {
            const s = w.take(w.next.fetchAdd(1, .monotonic)) orelse return;
            if (s.path != path) {
                if (fd >= 0) close(fd);
                fd = std.c.open(s.path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
                if (fd < 0) return w.failed.store(true, .monotonic);
                path = s.path;
            }
            var at = s.lo;
            while (at < s.hi) {
                const n = std.c.pread(fd, buf.ptr, s.hi - at, @intCast(at));
                if (n <= 0) return w.failed.store(true, .monotonic);
                at += @intCast(n);
            }
            _ = w.read.fetchAdd(s.hi - s.lo, .monotonic);
        }
    }
};

fn close(fd: std.c.fd_t) void {
    _ = std.c.close(fd);
}

/// Every byte of `spans`, by up to 16 readers in 8 MiB pieces: the bytes read, or an error when a file is missing or
/// shorter than a span.
pub fn read(spans: []const Span) !usize {
    if (spans.len == 0) return 0;
    var w: Work = .{ .spans = spans };
    var threads: [readers - 1]?std.Thread = @splat(null);
    for (&threads) |*t| t.* = std.Thread.spawn(.{}, Work.run, .{&w}) catch null;
    w.run();
    for (threads) |t| if (t) |thread| thread.join();
    if (w.failed.load(.monotonic)) return error.H3ReadAheadFailed;
    return w.read.load(.monotonic);
}

test "h3 read ahead: every span's bytes once across pieces and files, a short or missing file refused" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const big = try gpa.alloc(u8, piece + 5);
    defer gpa.free(big);
    @memset(big, 1);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a", .data = big });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b", .data = big[0..100] });
    const a = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/a", .{tmp.sub_path}, 0);
    defer gpa.free(a);
    const b = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/b", .{tmp.sub_path}, 0);
    defer gpa.free(b);
    // a's span crosses a piece boundary: two pieces; b's is one; an empty span reads nothing
    const spans = [_]Span{ .{ .path = a, .lo = 3, .hi = piece + 5 }, .{ .path = b, .lo = 0, .hi = 100 }, .{ .path = b, .lo = 7, .hi = 7 } };
    try std.testing.expectEqual(@as(usize, piece + 2 + 100), try read(&spans));
    try std.testing.expectEqual(@as(usize, 0), try read(&.{}));
    try std.testing.expectError(error.H3ReadAheadFailed, read(&.{.{ .path = b, .lo = 0, .hi = 101 }}));
    try std.testing.expectError(error.H3ReadAheadFailed, read(&.{.{ .path = ".zig-cache/tmp/none", .lo = 0, .hi = 1 }}));
}
