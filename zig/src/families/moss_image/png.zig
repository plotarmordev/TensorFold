//! An 8-bit RGB picture as a PNG: one IDAT of zlib-compressed rows, each with filter 0 (none).
const std = @import("std");
const flate = std.compress.flate;

/// The PNG of `rgb` [height, width, 3], allocated with `a`.
pub fn encode(a: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) ![]u8 {
    if (rgb.len != width * height * 3) return error.PictureSize;
    const s = try Stream.create(a, width, height);
    defer s.destroy();
    try s.rows(rgb);
    return s.finish();
}

/// `encode` a band of rows at a time: the bytes do not depend on how the rows are split.
pub const Stream = struct {
    a: std.mem.Allocator,
    width: usize,
    height: usize,
    written: usize = 0,
    z: std.Io.Writer.Allocating,
    window: [flate.max_window_len]u8 = undefined,
    c: flate.Compress = undefined,

    /// Heap-held: the compressor keeps pointers to its window and to `z`.
    pub fn create(a: std.mem.Allocator, width: usize, height: usize) !*Stream {
        if (width == 0 or height == 0) return error.PictureSize;
        if (width > std.math.maxInt(u31) or height > std.math.maxInt(u31)) return error.PictureSize;
        const s = try a.create(Stream);
        errdefer a.destroy(s);
        s.* = .{ .a = a, .width = width, .height = height, .z = try .initCapacity(a, width * height * 3 / 2 + 64) };
        errdefer s.z.deinit();
        s.c = try flate.Compress.init(&s.z.writer, &s.window, .zlib, .default);
        return s;
    }

    pub fn destroy(s: *Stream) void {
        s.z.deinit();
        s.a.destroy(s);
    }

    /// The next whole rows [n, width, 3].
    pub fn rows(s: *Stream, rgb: []const u8) !void {
        const row = s.width * 3;
        if (rgb.len % row != 0 or s.written + rgb.len / row > s.height) return error.PictureSize;
        var y: usize = 0;
        while (y < rgb.len) : (y += row) {
            try s.c.writer.writeByte(0);
            try s.c.writer.writeAll(rgb[y..][0..row]);
        }
        s.written += rgb.len / row;
    }

    /// The PNG once every row is in, allocated with the stream's allocator.
    pub fn finish(s: *Stream) ![]u8 {
        if (s.written != s.height) return error.PictureSize;
        try s.c.finish();
        var out: std.Io.Writer.Allocating = .init(s.a);
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

test "a picture streamed in uneven bands is the same PNG, and a stream short of its rows is refused" {
    const a = std.testing.allocator;
    const width = 37;
    const height = 41;
    const rgb = try a.alloc(u8, width * height * 3);
    defer a.free(rgb);
    var prng = std.Random.DefaultPrng.init(7);
    for (rgb, 0..) |*v, i| v.* = if (i % 5 == 0) prng.random().int(u8) else @truncate(i / 7);
    const whole = try encode(a, rgb, width, height);
    defer a.free(whole);
    const s = try Stream.create(a, width, height);
    defer s.destroy();
    const row = width * 3;
    try s.rows(rgb[0 .. 1 * row]);
    try std.testing.expectError(error.PictureSize, s.finish());
    try s.rows(rgb[1 * row .. 17 * row]);
    try s.rows(rgb[17 * row ..]);
    try std.testing.expectError(error.PictureSize, s.rows(rgb[0..row]));
    const banded = try s.finish();
    defer a.free(banded);
    try std.testing.expectEqualSlices(u8, whole, banded);
}
