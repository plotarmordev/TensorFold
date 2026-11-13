//! Ornith's MTP drafts in the lane core on real weights: TF_QWEN27_DIR the MLX 4-bit target, TF_QWEN27_MTP_DIR the bf16
//! release's folder holding its `mtp.*` shards. Drafted replies are plain decoding's tokens, alone, four at once,
//! beside a stream that does not draft, and request after request on one engine.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_served").lanes_backend;
const gpa = std.testing.allocator;

const prompt = [_]u32{ 760, 1156, 3833, 525, 279, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220, 22, 11, 220, 23, 11, 220, 24, 11, 220, 16, 15, 13, 5005, 1052, 374, 264, 3364, 911 };
const quicksort = [_]u32{ 727, 3841, 6646, 51532, 1590, 198, 262, 4071, 9927, 264, 1103, 440, 3841, 6646, 71483, 198 }; // the bench's `quicksort-raw`
const lengths = [_]u32{ 24, 40, 56, 96 }; // the longest reply ends alone, drafting again

/// Each of `streams`' replies to its end, the streams started together.
fn run(back: *lb.Metal, cfg: *const lanes.Config, io: std.Io, streams: []lanes.Stream) !void {
    var clock = lanes.backend.WallClock{ .io = io };
    var engine = lanes.Engine.init(gpa, cfg, back.backend(), clock.clock());
    defer engine.deinit();
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
}

fn init(out: []lanes.Stream, drafts: bool) !void {
    for (out, lengths[lengths.len - out.len ..]) |*s, n| s.* = try lanes.Stream.init(gpa, .{ .id = "mtp", .prompt = &prompt, .max_new = n, .drafts = drafts });
}

test "Ornith MTP lanes: drafted replies are plain decoding's, alone and four at once, through accepts and rejects" {
    const dir = std.mem.span(std.c.getenv("TF_QWEN27_DIR") orelse return error.SkipZigTest);
    const mtp_dir = std.mem.span(std.c.getenv("TF_QWEN27_MTP_DIR") orelse return error.SkipZigTest);
    const io = std.testing.io;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.model.Model.loadDeferred(gpa, io, dir, 128);
    defer model.deinit();
    try model.loadMtp(io, mtp_dir);
    try model.loadShare(0);
    var runner = try q.decode_round.Runner.initReady(gpa, model, lengths.len, 256, 1);
    defer runner.deinit();
    var drafter = try q.mtp.Drafter.init(&runner);
    defer drafter.deinit();
    const back = try lb.Metal.init(gpa, &runner, null);
    defer back.deinit();
    back.mtp = &drafter;
    try back.measure(io);
    var cfg = try lanes.Config.init(gpa, back.facts(), back.window, back.nodes);
    defer cfg.deinit(gpa);

    var plain: [lengths.len]lanes.Stream = undefined;
    try init(&plain, false);
    defer for (&plain) |*s| s.deinit(gpa);
    for (&plain) |*s| try run(back, &cfg, io, s[0..1]);

    // four at once draft in their shared rounds, each window's kept rows stepping the layer before it drafts; the
    // shortest reply ends while all four are live, so its drafts and accepts are all shared rounds'
    var together: [lengths.len]lanes.Stream = undefined;
    try init(&together, true);
    defer for (&together) |*s| s.deinit(gpa);
    try run(back, &cfg, io, &together);
    for (plain, together) |p, t| try std.testing.expectEqualSlices(u32, p.emitted(), t.emitted());
    for (together) |t| try std.testing.expect(t.drafted > 0 and t.accepted > 0);

    // beside a plain stream (the longest reply, so every round of the others' is shared with its one-row window), the
    // three drafting streams keep drafting and landing drafts
    var mixed: [lengths.len]lanes.Stream = undefined;
    try init(&mixed, true);
    defer for (&mixed) |*s| s.deinit(gpa);
    mixed[3].drafts = false;
    try run(back, &cfg, io, &mixed);
    for (plain, mixed) |p, m| try std.testing.expectEqualSlices(u32, p.emitted(), m.emitted());
    for (mixed[0..3]) |m| try std.testing.expect(m.accepted > 1);
    try std.testing.expect(mixed[3].drafted == 0);

    // alone, in a slot another stream's cache filled: a kept row moves the layer's cache on; a refused draft's row
    // never enters it, so later drafts still land
    var lone: [1]lanes.Stream = undefined;
    try init(&lone, true);
    defer lone[0].deinit(gpa);
    try run(back, &cfg, io, &lone);
    try std.testing.expectEqualSlices(u32, plain[3].emitted(), lone[0].emitted());
    try std.testing.expect(lone[0].accepted > 0 and lone[0].accepted < lone[0].drafted);
    try std.testing.expect(lone[0].accepted * 2 >= lone[0].drafted);

    // request after request on one engine, as a served slot is reused: a quarter of this prompt's drafts are refused,
    // so unless a whole-kept round counts as a landing the refusals alone price the draft out for good
    var clock = lanes.backend.WallClock{ .io = io };
    var served = lanes.Engine.init(gpa, &cfg, back.backend(), clock.clock());
    defer served.deinit();
    var reference = try lanes.Stream.init(gpa, .{ .id = "plain", .prompt = &quicksort, .max_new = 128, .drafts = false });
    defer reference.deinit(gpa);
    try served.addStream(&reference);
    while (served.activeCount() > 0) try served.step();
    for (0..3) |_| {
        var s = try lanes.Stream.init(gpa, .{ .id = "mtp", .prompt = &quicksort, .max_new = 128, .drafts = true });
        defer s.deinit(gpa);
        try served.addStream(&s);
        while (served.activeCount() > 0) try served.step();
        try std.testing.expectEqualSlices(u32, reference.emitted(), s.emitted());
        try std.testing.expect(s.drafted * 10 >= s.rounds * 9);
    }
}
