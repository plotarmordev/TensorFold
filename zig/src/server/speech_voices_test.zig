//! Speech voices through the HTTP route: the family CLI builder, engine catalogs and their refusals.
const std = @import("std");
const server = @import("root.zig");
const harness = @import("speech_stream_test.zig");
const gpa = std.testing.allocator;
const frame = harness.frame;
const pcm = harness.pcm;
const expectHeader = harness.expectHeader;
const Bytes = harness.Bytes;
const Script = harness.Script;
const Served = harness.Served;

test "voices: HTTP listing and reference rows use the family CLI builder" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const family = @import("drama_prompt");
    const codes = [_][10]i32{.{ 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 }};
    const refs = [_]family.Reference{.{ .transcript = "A small cloud passes.", .codes = &codes }};
    const voices = [_]@import("drama_voice.zig").Voice{.{ .name = "tone", .references = &refs }};
    const CliTokenizer = struct {
        pub fn specialTokenId(_: @This(), _: []const u8) ?u32 {
            return 1000;
        }
        pub fn encode(_: @This(), alloc: std.mem.Allocator, input: []const u8) ![]u32 {
            var unused: u8 = 0;
            return Bytes.encode(&unused, alloc, input, true);
        }
    };
    const expected = try family.build(a, CliTokenizer{}, "hi", &refs);
    var script: Script = .{ .steps = &.{frame}, .voices = &voices, .expected_rows = expected.frames };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"voice\":\"tone\",\"response_format\":\"pcm\"}", 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), " 200 ");
    s.join(&script);
    try std.testing.expect(script.rows_equal);
    try std.testing.expectEqual(expected.reference_rows, script.reference_rows[0]);
    try std.testing.expectEqualStrings(&pcm(1), try s.readRest());
    var listing_script: Script = .{ .steps = &.{}, .voices = &voices, .get_voices = true };
    var listing: Served = undefined;
    try listing.start(&listing_script, "HTTP/1.1", "", 2000);
    defer listing.deinit();
    try expectHeader(try listing.readHead(), " 200 ");
    listing.join(&listing_script);
    try std.testing.expectEqualStrings("{\"voices\": [\"default\", \"tone\"], \"details\": [{\"id\": \"tone\", \"description\": \"\"}]}", try listing.readRest());
}

test "voices: without configuration the default prompt rows stay unchanged" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var bytes: Bytes = .{};
    const tokens = try @import("drama_prompt.zig").encode(a, bytes.text(), "hi");
    const rows = try a.alloc([11]i32, tokens.len);
    for (rows, tokens) |*row, token| {
        row.* = @splat(0);
        row[0] = @intCast(token);
    }
    var script: Script = .{ .steps = &.{frame}, .expected_rows = rows };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\"}", 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), " 200 ");
    s.join(&script);
    try std.testing.expect(script.rows_equal);
    try std.testing.expectEqual(@as(u32, 0), script.reference_rows[0]);
    try std.testing.expectEqualStrings(&pcm(1), try s.readRest());
}

test "engine voices: code-point prompt, catalog index, listing and an unknown name refused" {
    const names = [_][]const u8{ "juno", "mira" };
    var script: Script = .{ .steps = &.{frame}, .engine_voices = &names };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"h\\u00e9\",\"voice\":\"mira\",\"response_format\":\"pcm\"}", 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), " 200 ");
    s.join(&script);
    try std.testing.expectEqual(@as(u32, 1), script.voice);
    try std.testing.expectEqualSlices(u32, &.{ 'h', 0xe9 }, script.prompt[0..2]);
    var unknown_script: Script = .{ .steps = &.{frame}, .engine_voices = &names };
    var unknown: Served = undefined;
    try unknown.start(&unknown_script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"voice\":\"tone\"}", 2000);
    defer unknown.deinit();
    try expectHeader(try unknown.readHead(), " 400 ");
    unknown.join(&unknown_script);
    try std.testing.expect(std.mem.indexOf(u8, try unknown.readRest(), "unknown voice 'tone'; see /v1/voices") != null);
    var listing_script: Script = .{ .steps = &.{}, .engine_voices = &names, .get_voices = true };
    var listing: Served = undefined;
    try listing.start(&listing_script, "HTTP/1.1", "", 2000);
    defer listing.deinit();
    try expectHeader(try listing.readHead(), " 200 ");
    listing.join(&listing_script);
    try std.testing.expectEqualStrings("{\"voices\": [\"default\", \"juno\", \"mira\"], \"details\": [{\"id\": \"juno\", \"description\": \"\"}, {\"id\": \"mira\", \"description\": \"\"}]}", try listing.readRest());
}

test "engine voices: the catalog outlives the engine's names, as across an idle unload and reload" {
    const juno = try gpa.dupe(u8, "juno");
    const mira = try gpa.dupe(u8, "mira");
    const names = [_][]const u8{ juno, mira };
    var script: Script = .{ .steps = &.{}, .engine_voices = &names };
    var bytes: Bytes = .{};
    const srv = try server.Server.init(gpa, std.testing.io, script.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    for (names) |name| {
        @memset(@constCast(name), 'x'); // the unload frees the engine's catalog
        gpa.free(name);
    }
    try std.testing.expectEqual(@as(usize, 2), srv.voices.len);
    try std.testing.expectEqualStrings("juno", srv.voices[0].name);
    try std.testing.expectEqualStrings("mira", srv.voices[1].name);
}

test "engine voices: Drama's sampling fields are refused, the seed is kept" {
    const names = [_][]const u8{"juno"};
    for ([_][]const u8{ "temperature\":0", "top_p\":0.5", "top_k\":5" }) |field| {
        var script: Script = .{ .steps = &.{frame}, .engine_voices = &names };
        var body: [96]u8 = undefined;
        var s: Served = undefined;
        try s.start(&script, "HTTP/1.1", try std.fmt.bufPrint(&body, "{{\"model\":\"m\",\"input\":\"hi\",\"{s}}}", .{field}), 2000);
        defer s.deinit();
        try expectHeader(try s.readHead(), " 400 ");
        s.join(&script);
        try std.testing.expect(std.mem.indexOf(u8, try s.readRest(), "is not supported by this model") != null);
    }
    var script: Script = .{ .steps = &.{frame}, .engine_voices = &names };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"seed\":7,\"temperature\":null,\"response_format\":\"pcm\"}", 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), " 200 ");
    s.join(&script);
}
