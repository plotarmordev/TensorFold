//! An 8-bit RGB picture as a PNG: one IDAT of zlib-compressed rows, each with filter 0 (none).
const std = @import("std");
const flate = std.compress.flate;

/// The PNG of `rgb` [height, width, 3], allocated with `a`.
pub fn encode(a: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) ![]u8 {
    if (rgb.len != width * height * 3) return error.PictureSize;
    const s = try a.create(Stream);
    defer a.destroy(s);
    try s.init(a, width, height);
    defer s.deinit();
    try s.rows(rgb);
    return s.finish(a);
}

/// A PNG fed its rows in order, a band at a time, through one zlib stream: the bytes are `encode`'s.
/// Pinned in place: the stream points into its own window and output.
pub const Stream = struct {
    width: usize,
    height: usize,
    written: usize,
    z: std.Io.Writer.Allocating,
    window: [flate.max_window_len]u8,
    c: flate.Compress,

    pub fn init(s: *Stream, a: std.mem.Allocator, width: usize, height: usize) !void {
        if (width == 0 or height == 0) return error.PictureSize;
        if (width > std.math.maxInt(u31) or height > std.math.maxInt(u31)) return error.PictureSize;
        s.width = width;
        s.height = height;
        s.written = 0;
        s.z = try .initCapacity(a, width * height * 3 / 2 + 64);
        errdefer s.z.deinit();
        s.c = try flate.Compress.init(&s.z.writer, &s.window, .zlib, .default);
    }

    pub fn deinit(s: *Stream) void {
        s.z.deinit();
    }

    /// The next whole rows of the picture [rows, width, 3].
    pub fn rows(s: *Stream, rgb: []const u8) !void {
        const row = s.width * 3;
        if (rgb.len % row != 0 or s.written + rgb.len / row > s.height) return error.PictureSize;
        for (0..rgb.len / row) |y| {
            try s.c.writer.writeByte(0);
            try s.c.writer.writeAll(rgb[y * row ..][0..row]);
        }
        s.written += rgb.len / row;
    }

    /// The PNG, allocated with `a`, once every row is in.
    pub fn finish(s: *Stream, a: std.mem.Allocator) ![]u8 {
        if (s.written != s.height) return error.PictureSize;
        try s.c.finish();
        var out: std.Io.Writer.Allocating = .init(a);
        errdefer out.deinit();
        try out.writer.writeAll("\x89PNG\r\n\x1a\n");
        var header: [13]u8 = undefined;
        std.mem.writeInt(u32, header[0..4], @intCast(s.width), .big);
        std.mem.writeInt(u32, header[4..8], @intCast(s.height), .big);
        header[8..13].* = .{ 8, 2, 0, 0, 0 };
        try chunk(&out.writer, "IHDR", &header);
        try chunk(&out.writer, "IDAT", s.z.written());
        try chunk(&out.writer, "IEND", "");
        return out.toOwnedSlice();
    }
};

fn chunk(w: *std.Io.Writer, kind: *const [4]u8, data: []const u8) !void {
    try w.writeInt(u32, @intCast(data.len), .big);
    try w.writeAll(kind);
    try w.writeAll(data);
    var crc = std.hash.Crc32.init();
    crc.update(kind);
    crc.update(data);
    try w.writeInt(u32, crc.final(), .big);
}

test "a PNG of a small picture decodes back to its pixels" {
    const a = std.testing.allocator;
    var rgb: [5 * 3 * 3]u8 = undefined;
    for (&rgb, 0..) |*v, i| v.* = @truncate(i * 37);
    const png = try encode(a, &rgb, 5, 3);
    defer a.free(png);
    const picture = @import("../../core/picture/png.zig");
    const s = try picture.decode(a, png);
    defer a.free(s.data);
    try std.testing.expectEqual(@as(usize, 5), s.width);
    try std.testing.expectEqual(@as(usize, 3), s.height);
    for (0..15) |i| try std.testing.expectEqualSlices(u8, rgb[i * 3 ..][0..3], s.data[i * s.channels ..][0..3]);
    try std.testing.expectError(error.PictureSize, encode(a, rgb[0..10], 5, 3));
}

test "a PNG streamed in uneven bands is encode's bytes" {
    const a = std.testing.allocator;
    const width = 301;
    const height = 211; // past the 64 KiB window, so the stream drains between bands
    const rgb = try a.alloc(u8, width * height * 3);
    defer a.free(rgb);
    var prng = std.Random.DefaultPrng.init(170);
    for (rgb, 0..) |*v, i| v.* = if (i % 7 == 0) prng.random().int(u8) else @truncate(i / 911);
    const whole = try encode(a, rgb, width, height);
    defer a.free(whole);
    const s = try a.create(Stream);
    defer a.destroy(s);
    try s.init(a, width, height);
    defer s.deinit();
    var y: usize = 0;
    for ([_]usize{ 1, 64, 0, 90 }) |band| {
        try s.rows(rgb[y * width * 3 ..][0 .. band * width * 3]);
        y += band;
    }
    try std.testing.expectError(error.PictureSize, s.finish(a));
    try std.testing.expectError(error.PictureSize, s.rows(rgb[0 .. width * 3 + 1]));
    try s.rows(rgb[y * width * 3 ..]);
    const streamed = try s.finish(a);
    defer a.free(streamed);
    try std.testing.expectEqualSlices(u8, whole, streamed);
}
