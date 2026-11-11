//! Ornith's MTP drafts in the lane core on real weights: TF_QWEN27_DIR the MLX 4-bit target, TF_QWEN27_MTP_DIR the bf16
//! release's folder holding its `mtp.*` shards. Drafted replies are plain decoding's tokens, alone and four at once.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_served").lanes_backend;
const gpa = std.testing.allocator;

const prompt = [_]u32{ 760, 1156, 3833, 525, 279, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220, 22, 11, 220, 23, 11, 220, 24, 11, 220, 16, 15, 13, 5005, 1052, 374, 264, 3364, 911 };
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

    // four at once share rounds, each window's kept rows stepping the layer; the longest, left alone, drafts from a
    // whole cache (one with holes drafts none), so its drafts land as a lone stream's do
    var together: [lengths.len]lanes.Stream = undefined;
    try init(&together, true);
    defer for (&together) |*s| s.deinit(gpa);
    try run(back, &cfg, io, &together);
    for (plain, together) |p, t| try std.testing.expectEqualSlices(u32, p.emitted(), t.emitted());
    try std.testing.expect(together[3].drafted > 0 and together[3].accepted * 2 >= together[3].drafted);

    // alone, in a slot another stream's cache filled: a kept row moves the layer's cache on; a refused draft's row
    // never enters it, so later drafts still land
    var lone: [1]lanes.Stream = undefined;
    try init(&lone, true);
    defer lone[0].deinit(gpa);
    try run(back, &cfg, io, &lone);
    try std.testing.expectEqualSlices(u32, plain[3].emitted(), lone[0].emitted());
    try std.testing.expect(lone[0].accepted > 0 and lone[0].accepted < lone[0].drafted);
    try std.testing.expect(lone[0].accepted * 2 >= lone[0].drafted);
}
