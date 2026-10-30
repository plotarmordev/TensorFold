//! Speech sampling controls through the HTTP route, including long-form retries and carried reference prompts.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const harness = @import("speech_stream_test.zig");
const scripted = @import("speech_long_test.zig");
const drama_prompt = @import("drama_prompt.zig");
const gpa = std.testing.allocator;
const config: server.Config = .{ .served_name = "m", .model_ids = &.{"m"}, .seed_salt = 17 };
const defaults: scripted.Sampling = .{ .temperature = 1, .top_p = 0.9, .top_k = 30 };

fn exchange(turns: *scripted.Turns, cfg: server.Config, body: []const u8) ![]u8 {
    var s: harness.Served = undefined;
    try s.startWith(turns.engine(), cfg, "HTTP/1.1", body, 2000);
    defer s.deinit();
    defer s.route.join();
    _ = try s.readHead();
    _ = try s.readRest();
    return gpa.dupe(u8, s.got.items);
}

fn expectDraw(turns: *const scripted.Turns, index: usize, sampling: scripted.Sampling, seed: u64) !void {
    try std.testing.expectEqualDeep(sampling, turns.sampling.items[index]);
    try std.testing.expectEqual(seed, turns.seeds.items[index]);
}

test "speech sampling: omitted and null controls retain default parameters and PCM" {
    const bodies = [_][]const u8{
        "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\"}",
        "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"temperature\":null,\"top_p\":null,\"top_k\":null,\"seed\":null}",
    };
    var first: ?[]u8 = null;
    defer if (first) |pcm| gpa.free(pcm);
    for (bodies) |body| {
        var turns: scripted.Turns = .{ .turns = &.{.{ .frames = 5 }} };
        defer turns.deinit();
        const reply = try exchange(&turns, config, body);
        defer gpa.free(reply);
        try harness.expectHeader(reply, "HTTP/1.1 200");
        try std.testing.expectEqual(@as(usize, 1), turns.seen);
        try expectDraw(&turns, 0, defaults, api.seedFor(turns.prompts.items[0], config.seed_salt));
        const pcm = reply[(std.mem.indexOf(u8, reply, "\r\n\r\n").? + 4)..];
        if (first) |before| try std.testing.expectEqualSlices(u8, before, pcm) else first = try gpa.dupe(u8, pcm);
    }
}

test "speech sampling: custom parameters and exact integer seeds reach whole and streamed requests" {
    const cases = [_]struct { fields: []const u8, sampling: scripted.Sampling, seed: ?u64 }{
        .{ .fields = "\"temperature\":1.25,\"top_p\":0.8,\"top_k\":64,\"seed\":9007199254740993", .sampling = .{ .temperature = 1.25, .top_p = 0.8, .top_k = 64 }, .seed = 9007199254740993 },
        .{ .fields = "\"temperature\":0,\"top_p\":1,\"top_k\":1,\"seed\":0", .sampling = .{ .temperature = 0, .top_p = 1, .top_k = 1 }, .seed = 0 },
        .{ .fields = "\"seed\":18446744073709551615", .sampling = defaults, .seed = std.math.maxInt(u64) },
        .{ .fields = "\"temperature\":0.5", .sampling = .{ .temperature = 0.5, .top_p = 0.9, .top_k = 30 }, .seed = null },
        .{ .fields = "\"temperature\":1e-400", .sampling = .{ .temperature = 0, .top_p = 0.9, .top_k = 30 }, .seed = null },
        .{ .fields = "\"temperature\":-1e-400", .sampling = .{ .temperature = -0.0, .top_p = 0.9, .top_k = 30 }, .seed = null },
        .{ .fields = "\"temperature\":3.4028234663852886e38,\"top_p\":1e-45", .sampling = .{ .temperature = std.math.floatMax(f32), .top_p = 1e-45, .top_k = 30 }, .seed = null },
    };
    for (cases) |case| for ([_]bool{ false, true }) |stream| {
        const body = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":{s},{s}}}", .{ if (stream) "true" else "false", case.fields });
        defer gpa.free(body);
        var turns: scripted.Turns = .{ .turns = &.{.{ .frames = 5 }} };
        defer turns.deinit();
        const reply = try exchange(&turns, config, body);
        defer gpa.free(reply);
        try harness.expectHeader(reply, "HTTP/1.1 200");
        if (stream) try harness.expectHeader(reply, "Transfer-Encoding: chunked");
        try std.testing.expectEqual(@as(usize, 1), turns.seen);
        try expectDraw(&turns, 0, case.sampling, case.seed orelse api.seedFor(turns.prompts.items[0], config.seed_salt));
    };
}

test "speech sampling: invalid controls fail before any engine submission" {
    const fields = [_][]const u8{
        "\"top_p\":1e-400",               "\"top_p\":-1e-400",
        "\"temperature\":true",           "\"temperature\":\"1\"",
        "\"temperature\":[]",             "\"temperature\":{}",
        "\"temperature\":-1",             "\"temperature\":NaN",
        "\"temperature\":Infinity",       "\"temperature\":1e39",
        "\"temperature\":1e-100",         "\"top_p\":false",
        "\"top_p\":\"0.9\"",              "\"top_p\":0",
        "\"top_p\":-0.1",                 "\"top_p\":1.00000001",
        "\"top_p\":NaN",                  "\"top_p\":1e-100",
        "\"top_p\":Infinity",             "\"top_k\":true",
        "\"top_k\":\"30\"",               "\"top_k\":1.5",
        "\"top_k\":30.0",                 "\"top_k\":0",
        "\"top_k\":-1",                   "\"top_k\":65",
        "\"top_k\":18446744073709551616", "\"seed\":true",
        "\"seed\":\"1\"",                 "\"seed\":1.5",
        "\"seed\":1.0",                   "\"seed\":-1",
        "\"seed\":18446744073709551616",  "\"seed\":NaN",
    };
    for (fields) |field| {
        const body = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"input\":\"hi\",{s}}}", .{field});
        defer gpa.free(body);
        var turns: scripted.Turns = .{ .turns = &.{.{ .frames = 5 }} };
        defer turns.deinit();
        const reply = try exchange(&turns, config, body);
        defer gpa.free(reply);
        try harness.expectHeader(reply, "HTTP/1.1 400");
        try std.testing.expectEqual(@as(usize, 0), turns.seen);
    }
}

test "speech sampling: referenced re-prompts keep overrides and wrap explicit retry seeds" {
    const frames = [_]drama_prompt.Codes{@splat(7)};
    const refs = [_]drama_prompt.Reference{.{ .text = "A small cloud passes.", .frames = &frames }};
    const voices = [_]@import("drama_voice.zig").Voice{.{ .name = "tone", .references = &refs, .source = .@"model-invented" }};
    var cfg = config;
    cfg.voices = &voices;
    cfg.cfg = .{ .gamma = 2 };
    cfg.silence_redraw = .{ .max = 2, .plain = true };
    var turns: scripted.Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence }, .{ .frames = 5, .reason = .silence }, .{ .frames = 5 } } };
    defer turns.deinit();
    const reply = try exchange(&turns, cfg, "{\"model\":\"m\",\"input\":\"hi [sighs] [laughs] there\",\"voice\":\"tone\",\"temperature\":0.5,\"top_p\":0.75,\"top_k\":8,\"seed\":18446744073709551615}");
    defer gpa.free(reply);
    try harness.expectHeader(reply, "HTTP/1.1 200");
    try std.testing.expectEqual(@as(usize, 3), turns.seen);
    for ([_]u64{ std.math.maxInt(u64), 0, 1 }, 0..) |seed, k| {
        try expectDraw(&turns, k, .{ .temperature = 0.5, .top_p = 0.75, .top_k = 8 }, seed);
        var reference_frames: usize = 0;
        for (turns.codes.items[k]) |codes| if (std.meta.eql(frames[0], codes)) {
            reference_frames += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), reference_frames);
        try std.testing.expectEqual(@as(f32, 2), turns.gammas.items[k]);
        try std.testing.expect(turns.unconditioned.items[k].len > 0);
    }
    try std.testing.expect(!std.mem.eql(u32, turns.prompts.items[0], turns.prompts.items[1]));
}

test "speech sampling: long-form segment retries preserve overrides, carried prompts and per-segment seeds" {
    var cfg = config;
    cfg.long_form = .{ .chars = 20 };
    cfg.silence_redraw = .{ .max = 1, .plain = true };
    cfg.cfg = .{ .gamma = 2 };
    for ([_]bool{ false, true }) |explicit| {
        const seed_field = if (explicit) ",\"seed\":18446744073709551615" else "";
        const body = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"input\":\"Sentence 00 is ok. Sentence [sighs] ok. Sentence 02 is ok.\",\"temperature\":0.5,\"top_p\":0.75,\"top_k\":8{s}}}", .{seed_field});
        defer gpa.free(body);
        var turns: scripted.Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 5, .reason = .silence }, .{ .frames = 5, .busy = 1 }, .{ .frames = 5 } } };
        defer turns.deinit();
        const reply = try exchange(&turns, cfg, body);
        defer gpa.free(reply);
        try harness.expectHeader(reply, "HTTP/1.1 200");
        try std.testing.expectEqual(@as(usize, 4), turns.seen);
        for ([_]usize{ 0, 0, 1, 0 }, 0..) |attempt, k| {
            const seed = if (explicit) std.math.maxInt(u64) +% @as(u64, @intCast(attempt)) else api.seedFor(turns.prompts.items[k], config.seed_salt + @as(i64, @intCast(attempt)));
            try expectDraw(&turns, k, .{ .temperature = 0.5, .top_p = 0.75, .top_k = 8 }, seed);
            try std.testing.expectEqual(@as(f32, if (k == 0) 1 else 2), turns.gammas.items[k]);
        }
        try std.testing.expect(!std.mem.eql(u32, turns.prompts.items[1], turns.prompts.items[2]));
        try std.testing.expect(turns.codes.items[3].len > 0);
    }
}

test "speech sampling: a silenced long-form stream keeps controls without redrawing" {
    var cfg = config;
    cfg.long_form = .{ .chars = 20 };
    cfg.silence_redraw = .{ .max = 2 };
    var turns: scripted.Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 5, .reason = .silence } } };
    defer turns.deinit();
    const reply = try exchange(&turns, cfg, "{\"model\":\"m\",\"input\":\"Sentence 00 is ok. Sentence 01 is ok. Sentence 02 is ok.\",\"response_format\":\"pcm\",\"stream\":true,\"temperature\":0.5,\"seed\":2}");
    defer gpa.free(reply);
    try harness.expectHeader(reply, "HTTP/1.1 200");
    try harness.expectHeader(reply, "X-Finish-Reason: silence");
    try std.testing.expectEqual(@as(usize, 2), turns.seen);
    for (0..2) |k| try expectDraw(&turns, k, .{ .temperature = 0.5, .top_p = 0.9, .top_k = 30 }, 2);
}
