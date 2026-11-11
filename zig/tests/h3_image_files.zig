//! The H3-Image tools' safetensors files: aligned images, a synthetic checkpoint of one shard and its index, and the
//! selftests' scratch folder.
const std = @import("std");
const fixture = @import("checkpoint_fixture");

/// A safetensors image whose data starts on 16 bytes: the header padded with spaces.
pub fn image(a: std.mem.Allocator, entries: []const fixture.Entry) ![]u8 {
    const plain = try fixture.image(a, entries);
    const header: usize = @intCast(std.mem.readInt(u64, plain[0..8], .little));
    const pad = (16 - (8 + header) % 16) % 16;
    const out = try a.alloc(u8, plain.len + pad);
    std.mem.writeInt(u64, out[0..8], header + pad, .little);
    @memcpy(out[8..][0..header], plain[8..][0..header]);
    @memset(out[8 + header ..][0..pad], ' ');
    @memcpy(out[8 + header + pad ..], plain[8 + header ..]);
    return out;
}

/// `entries` as `dir`'s one shard, with the index naming it for every tensor.
pub fn checkpoint(io: std.Io, a: std.mem.Allocator, dir: []const u8, entries: []const fixture.Entry) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const file = "model-00001-of-00001.safetensors";
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\"weight_map\": {");
    for (entries, 0..) |e, i| try index.print(a, "{s}\"{s}\": \"{s}\"", .{ if (i == 0) "" else ", ", e.name, file });
    try index.appendSlice(a, "}}");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), .data = index.items });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, file }), .data = try image(a, entries) });
}

/// A folder made new under `parent` for one selftest run: `deinit` removes it alone, never what `parent` already held.
pub const Scratch = struct {
    path: []const u8,

    pub fn init(io: std.Io, a: std.mem.Allocator, parent: []const u8) !Scratch {
        try std.Io.Dir.cwd().createDirPath(io, parent);
        var bytes: [9]u8 = undefined;
        io.random(&bytes);
        var name: [12]u8 = undefined;
        _ = std.base64.url_safe.Encoder.encode(&name, &bytes);
        const path = try std.fs.path.join(a, &.{ parent, &name });
        try std.Io.Dir.cwd().createDir(io, path, .default_dir); // fails on an existing name, so it is ours alone
        return .{ .path = path };
    }

    pub fn deinit(s: Scratch, io: std.Io) void {
        std.Io.Dir.cwd().deleteTree(io, s.path) catch {};
    }
};

/// A selftest's shape: a checkpoint written into its scratch, then a step that may fail.
fn run(io: std.Io, a: std.mem.Allocator, parent: []const u8, fail: bool) !void {
    const scratch = try Scratch.init(io, a, parent);
    defer scratch.deinit(io);
    try checkpoint(io, a, try std.fs.path.join(a, &.{ scratch.path, "text_encoder" }), &.{});
    if (fail) return error.GateFailed;
}

test "selftest scratch: a run, passing or failing, removes its own folder and keeps what WORK_DIR held" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const parent = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(io, .{ .sub_path = "sentinel", .data = "keep" });
    try tmp.dir.createDirPath(io, "text_encoder");
    try tmp.dir.writeFile(io, .{ .sub_path = "text_encoder/config.json", .data = "{}" });
    try run(io, a, parent, false);
    try std.testing.expectError(error.GateFailed, run(io, a, parent, true));
    try std.testing.expectEqualStrings("keep", try tmp.dir.readFileAlloc(io, "sentinel", a, .limited(16)));
    try std.testing.expectEqualStrings("{}", try tmp.dir.readFileAlloc(io, "text_encoder/config.json", a, .limited(16)));
    var it = tmp.dir.iterate();
    var left: usize = 0;
    while (try it.next(io)) |_| left += 1;
    try std.testing.expectEqual(2, left);
}
