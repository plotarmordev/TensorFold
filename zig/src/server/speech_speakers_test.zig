//! Speaker identities through actual HTTP submissions, long-form history, guidance and redraws.
const std = @import("std");
const server = @import("root.zig");
const json = @import("json.zig");
const prompt = @import("drama_prompt.zig");
const wire = @import("speech_stream_test.zig");
const scripted = @import("speech_long_test.zig");

const suffix = "<|im_end|>\n<|im_start|>assistant\n<|voice|>";
const plain_head = "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n<|im_start|>user\n";
const reference = "[{\"text\":\"  <|speaker:15|> Reference.\",\"vq_codes\":[[7],[1],[2],[3],[4],[5],[6],[7],[8],[9]]}]";

fn request(a: std.mem.Allocator, input: []const u8, stream: bool, referenced: bool) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":{s},\"response_format\":\"pcm\",\"stream\":{}{s}}}", .{
        try json.quote(a, input, .{}), stream, if (referenced) ",\"references\":" ++ reference else "",
    });
}

fn serve(a: std.mem.Allocator, turns: *scripted.Turns, config: server.Config, body: []const u8) ![]const u8 {
    var s: wire.Served = undefined;
    try s.startWith(turns.engine(), config, "HTTP/1.1", body, 2000);
    defer s.deinit();
    _ = try s.readHead();
    _ = try s.readRest();
    s.route.join();
    return a.dupe(u8, s.got.items);
}

fn bytes(a: std.mem.Allocator, text: []const u8) ![]const u32 {
    const out = try a.alloc(u32, text.len);
    for (text, out) |byte, *token| token.* = byte;
    return out;
}

fn userText(a: std.mem.Allocator, tokens: []const u32, text: []const u8) !void {
    const end = try bytes(a, try std.mem.concat(a, u8, &.{ "<|im_start|>user\n", text, suffix }));
    try std.testing.expect(std.mem.endsWith(u32, tokens, end));
}

fn contains(a: std.mem.Allocator, tokens: []const u32, text: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u32, tokens, try bytes(a, text)) != null);
}

test "speakers: every canonical ID reaches the engine without changing dialogue text" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (0..16) |id| {
        const input = try std.fmt.allocPrint(a, "\u{3000}<|speaker:{d}|> First. <|speaker:0|> Reply.", .{id});
        var turns: scripted.Turns = .{ .turns = &.{.{ .frames = 2 }} };
        defer turns.deinit();
        const got = try serve(a, &turns, scripted.config(null), try request(a, input, false, false));
        try std.testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 200 "));
        try std.testing.expectEqual(@as(usize, 1), turns.seen);
        try std.testing.expectEqualSlices(u32, try bytes(a, try std.mem.concat(a, u8, &.{ plain_head, input, suffix })), turns.prompts.items[0]);
    }
}

test "speakers: referenced whole and streamed dialogue keeps reference and guidance identities" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = "<|speaker:15|> First. <|speaker:1|> Reply. <|speaker:15|> Done.";
    var config = scripted.config(null);
    config.cfg = .{ .gamma = 3 };
    for ([_]bool{ false, true }) |stream| {
        var turns: scripted.Turns = .{ .turns = &.{.{ .frames = 2 }} };
        defer turns.deinit();
        const got = try serve(a, &turns, config, try request(a, input, stream, true));
        try std.testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 200 "));
        try std.testing.expectEqual(@as(usize, 1), turns.seen);
        try userText(a, turns.prompts.items[0], input);
        try contains(a, turns.prompts.items[0], "Text: <|speaker:15|>   Reference.\nSpeech: ");
        try std.testing.expectEqualSlices(u32, try bytes(a, plain_head ++ input ++ suffix), turns.unconditioned.items[0]);
        var found: usize = 0;
        for (turns.codes.items[0]) |codes| if (codes[0] != 0) {
            try std.testing.expectEqualSlices(u32, &.{ 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &codes);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        if (stream) try std.testing.expect(std.mem.endsWith(u8, got, "\r\n0\r\n\r\n"));
    }
}

test "speakers: long-form continuations carry the active speaker with and without history or streaming" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = "<|speaker:15|> First sentence. Second sentence. <|speaker:2|> Third sentence. Fourth sentence.";
    const spoken = [_][]const u8{
        "<|speaker:15|> First sentence.", "<|speaker:15|> Second sentence.",
        "<|speaker:2|> Third sentence.",  "<|speaker:2|> Fourth sentence.",
    };
    for ([_]u8{ 0, 1 }) |history| for ([_]bool{ false, true }) |stream| {
        var turns: scripted.Turns = .{ .turns = &.{ .{ .frames = 2 }, .{ .frames = 2 }, .{ .frames = 2 }, .{ .frames = 2 } } };
        defer turns.deinit();
        var config = scripted.config(.{ .chars = 30, .history = history });
        config.cfg = .{ .gamma = 3 };
        const got = try serve(a, &turns, config, try request(a, input, stream, false));
        try std.testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 200 "));
        try std.testing.expectEqual(@as(usize, 4), turns.seen);
        for (spoken, 0..) |text, i| {
            try userText(a, turns.prompts.items[i], text);
            if (history > 0 and i > 0) {
                try contains(a, turns.prompts.items[i], try std.mem.concat(a, u8, &.{ "[Reference 1]\nText: ", spoken[i - 1], "\nSpeech: " }));
                try std.testing.expectEqualSlices(u32, try bytes(a, try std.mem.concat(a, u8, &.{ plain_head, text, suffix })), turns.unconditioned.items[i]);
                var found: usize = 0;
                for (turns.codes.items[i]) |codes| if (codes[0] != 0) {
                    const want: prompt.Codes = @splat(@intCast(i * 100 + found));
                    try std.testing.expectEqualSlices(u32, &want, &codes);
                    found += 1;
                };
                try std.testing.expectEqual(@as(usize, 2), found);
            } else try std.testing.expectEqual(@as(usize, 0), turns.unconditioned.items[i].len);
        }
    };
}

test "speakers: paragraph shifts and plain redraws preserve speaker tags in the next segment's history" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = "<|speaker:15|> First sentence.\u{b6} [pause] Second sentence.\u{b6} Third sentence.";
    for ([_]@import("speech_redraw.zig").Scope{ .cues, .controls }) |scope| {
        var turns: scripted.Turns = .{ .turns = &.{ .{ .frames = 2 }, .{ .frames = 2, .reason = .silence }, .{ .frames = 2 }, .{ .frames = 2 } } };
        defer turns.deinit();
        var config = scripted.config(.{ .chars = 80, .history = 1, .shift_pause_s = 0.5 });
        config.cfg = .{ .gamma = 3 };
        config.silence_redraw = .{ .max = 1, .plain = true, .plain_scope = scope };
        const got = try serve(a, &turns, config, try request(a, input, false, false));
        try std.testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 200 "));
        try std.testing.expectEqual(@as(usize, 4), turns.seen);
        try userText(a, turns.prompts.items[1], "<|speaker:15|> [pause] Second sentence.");
        try userText(a, turns.prompts.items[2], "<|speaker:15|> Second sentence.");
        try userText(a, turns.prompts.items[3], "<|speaker:15|> Third sentence.");
        try contains(a, turns.prompts.items[3], "Text: <|speaker:15|> Second sentence.\nSpeech: ");
        try std.testing.expectEqualSlices(u32, try bytes(a, plain_head ++ "<|speaker:15|> Second sentence." ++ suffix), turns.unconditioned.items[2]);
        try std.testing.expect(std.mem.indexOf(u8, got, "X-Pause-Spans: ") != null);
    }
}

test "speakers: long-form admission counts the carried speaker tag before submitting any segment" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const continuation = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.";
    var turns: scripted.Turns = .{
        .turns = &.{ .{ .frames = 2 }, .{ .frames = 2 } },
        .max_prompt = plain_head.len + "<|speaker:0|> ".len + continuation.len + suffix.len,
    };
    defer turns.deinit();
    const got = try serve(a, &turns, scripted.config(.{ .chars = 40 }), try request(a, "<|speaker:15|> Hi. " ++ continuation, false, false));
    try std.testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 400 "));
    try std.testing.expect(std.mem.indexOf(u8, got, "rows exceeds the maximum allowed length") != null);
    try std.testing.expectEqual(@as(usize, 0), turns.seen);
}

test "speakers: protected hard cuts retain embedded tags and sampling controls through continuation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = "abcdefghijklmnop<|speaker:15|>abcdefghijklmnopqrstuvwxyz";
    const spoken = [_][]const u8{
        "<|speaker:0|> abcdefghijklmnop",
        "<|speaker:15|>abcdef",
        "<|speaker:15|> ghijklmnopqrstuvwxyz",
    };
    for ([_]bool{ false, true }) |stream| {
        var turns: scripted.Turns = .{ .turns = &.{ .{ .frames = 2 }, .{ .frames = 2 }, .{ .frames = 2 } } };
        defer turns.deinit();
        var config = scripted.config(.{ .chars = 20, .history = 1 });
        config.cfg = .{ .gamma = 2 };
        const raw = try request(a, input, stream, false);
        const body = try std.fmt.allocPrint(a, "{s},\"temperature\":0.5,\"top_p\":0.75,\"top_k\":8,\"seed\":9007199254740993}}", .{raw[0 .. raw.len - 1]});
        const got = try serve(a, &turns, config, body);
        try std.testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 200 "));
        try std.testing.expectEqual(spoken.len, turns.seen);
        for (spoken, 0..) |text, i| {
            try userText(a, turns.prompts.items[i], text);
            try std.testing.expectEqualDeep(scripted.Sampling{ .temperature = 0.5, .top_p = 0.75, .top_k = 8 }, turns.sampling.items[i]);
            try std.testing.expectEqual(@as(u64, 9007199254740993), turns.seeds.items[i]);
            if (i > 0) try userText(a, turns.unconditioned.items[i], text);
        }
    }
}

test "speakers: malformed dialogue is refused before submission with or without references" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bad = [_][]const u8{
        "<|speaker:15|> Valid. <|speaker:16|> Bad.",
        "<|speaker:1|> Valid. <|speaker:01|> Bad.",
        "<|speaker:-1|> Bad.",
        "<|speaker:\u{661}|> Bad.",
        "Partial <|speaker:",
    };
    for ([_]bool{ false, true }) |referenced| for (bad) |input| {
        var turns: scripted.Turns = .{ .turns = &.{.{ .frames = 2 }} };
        defer turns.deinit();
        const got = try serve(a, &turns, scripted.config(null), try request(a, input, true, referenced));
        try std.testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 400 "));
        try std.testing.expect(std.mem.indexOf(u8, got, "N in 0..15 and no leading zero") != null);
        try std.testing.expectEqual(@as(usize, 0), turns.seen);
    };
}
