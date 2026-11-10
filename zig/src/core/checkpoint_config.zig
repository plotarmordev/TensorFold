//! One bounded checkpoint-config reader for info, serve and benchmark metadata.
const std = @import("std");
pub const limit: usize = 64 << 20;

/// Missing is distinct from a failed read; diagnostics never replace an I/O error with a missing checkpoint.
pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) !?[]u8 {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const file = std.Io.Dir.cwd().openFile(io, path, .{ .allow_directory = true }) catch |e| {
        if (e == error.FileNotFound) return null;
        const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch null;
        return failed(a, problem, if (st) |s| s.size else null, e);
    };
    defer file.close(io);
    const stat = file.stat(io) catch |e| return failed(a, problem, null, e);
    if (stat.kind != .file) return failed(a, problem, stat.size, error.ConfigNotFile);
    if (stat.size > limit) return failed(a, problem, stat.size, error.ConfigTooLarge);
    var reader = file.reader(io, &.{});
    const bytes = reader.interface.allocRemaining(a, .limited(limit + 1)) catch |e| {
        const size = if (file.stat(io)) |s| s.size else |_| stat.size;
        return failed(a, problem, size, if (e == error.ReadFailed) reader.err.? else if (e == error.StreamTooLong) error.ConfigTooLarge else e);
    };
    if (bytes.len > limit) {
        defer a.free(bytes);
        return failed(a, problem, bytes.len, error.ConfigTooLarge);
    }
    return bytes;
}

/// Metadata callers that intentionally have an optional label still use the same file path and bound.
pub fn optional(a: std.mem.Allocator, io: std.Io, dir: []const u8) ?[]u8 {
    var problem: []const u8 = "";
    return read(a, io, dir, &problem) catch null;
}
fn failed(a: std.mem.Allocator, problem: *[]const u8, size: ?u64, cause: anyerror) anyerror {
    problem.* = if (cause == error.ConfigTooLarge and size != null)
        std.fmt.allocPrint(a, "config.json is {d} bytes, over the {d}-byte limit", .{ size.?, limit }) catch "config.json exceeds the read limit"
    else if (size) |n|
        std.fmt.allocPrint(a, "Cannot read config.json ({s}; size {d} bytes; limit {d} bytes)", .{ @errorName(cause), n, limit }) catch "Cannot read config.json"
    else
        std.fmt.allocPrint(a, "Cannot read config.json ({s}; size unknown; limit {d} bytes)", .{ @errorName(cause), limit }) catch "Cannot read config.json";
    return cause;
}

/// The engine family a config's `model_type` is served by. Qwen 4's expected Flash MoE tag (`qwen4_moe`, after
/// Qwen3.5's `qwen3_5_moe`) is the Flash Next architecture, so it shares `qwen4_exp`'s Metal engine, flags and memory plan.
pub fn servedType(model_type: []const u8) []const u8 {
    return if (std.mem.eql(u8, model_type, "qwen4_moe")) "qwen4_exp" else model_type;
}

test "issue 934: the shared config bound includes exactly 64 MiB" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const file = try tmp.dir.createFile(std.testing.io, "config.json", .{});
    try file.setLength(std.testing.io, limit);
    file.close(std.testing.io);
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    var problem: []const u8 = "";
    const bytes = (try read(a, std.testing.io, dir, &problem)).?;
    try std.testing.expectEqual(limit, bytes.len);
    try std.testing.expectEqualStrings("", problem);
}

test "Qwen 4's Flash MoE tier is served by the Flash Next family; other tags pass through" {
    try std.testing.expectEqualStrings("qwen4_exp", servedType("qwen4_moe"));
    for ([_][]const u8{ "qwen4_exp", "qwen4", "qwen4_text", "qwen3_5_moe" }) |t| try std.testing.expectEqualStrings(t, servedType(t));
}

/// `dir` ships `name` (Woof's int8 checkpoint has no config.json; its weights file is the marker).
pub fn hasFile(io: std.Io, dir: []const u8, name: []const u8) bool {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{}) catch return false;
    defer d.close(io);
    d.access(io, name, .{}) catch return false;
    return true;
}
