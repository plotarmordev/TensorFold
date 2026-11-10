//! Ornith-1.5-122B's MTP layer on real weights: TF_QWEN27_DIR the MLX 4-bit target, TF_QWEN27_MTP_DIR the bf16
//! release's folder holding its `mtp.*` shards. testdata/ornith-mtp.safetensors is tools/zig/ornith_mtp_reference.py's,
//! testdata/ornith-mtp-drafts.safetensors tools/zig/ornith_mtp_drafts.py's.
const std = @import("std");
const platform = @import("sys");
const st = @import("core").safetensors;
const Model = @import("model.zig").Model;
const Runner = @import("decode_round.zig").Runner;
const Round = @import("round_plan.zig").Round;
const Session = @import("session.zig").Session;
const mtp = @import("mtp.zig");
/// Each step's reference draft is its top `candidates` token ids, best first (tools/zig/ornith_mtp_drafts.py).
const candidates = 8;

/// An embedded safetensors file's header and data.
const Fixture = struct {
    header: st.Header,
    data: []const u8,

    fn parse(a: std.mem.Allocator, bytes: []const u8) !Fixture {
        const json_len: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        return .{ .header = try st.parseHeader(a, bytes[8..][0..json_len], bytes.len - 8 - json_len), .data = bytes[8 + json_len ..] };
    }

    /// Tensor `name`, copied out aligned.
    fn tensor(f: Fixture, a: std.mem.Allocator, name: []const u8, comptime T: type) ![]T {
        const e = f.header.get(name) orelse return error.MissingTensor;
        const out = try a.alloc(T, (e.end - e.begin) / @sizeOf(T));
        @memcpy(std.mem.sliceAsBytes(out), f.data[e.begin..e.end]);
        return out;
    }
};

fn f32of(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

test "Ornith MTP layer: bf16 weights load 4-bit and its rows match mlx-lm's" {
    const dir = std.mem.span(platform.c.getenv("TF_QWEN27_DIR") orelse return error.SkipZigTest);
    const mtp_dir = std.mem.span(platform.c.getenv("TF_QWEN27_MTP_DIR") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fixture = try Fixture.parse(arena.allocator(), @embedFile("testdata/ornith-mtp.safetensors"));
    const ids = try fixture.tensor(arena.allocator(), "ids", u32);
    const hidden = try fixture.tensor(arena.allocator(), "hidden", u16);
    const want = try fixture.tensor(arena.allocator(), "out", u16);

    const m = try Model.loadDeferred(a, io, dir, 128);
    defer m.deinit();
    try m.loadMtp(io, mtp_dir);
    try std.testing.expectError(error.MtpLoaded, m.loadMtp(io, mtp_dir));
    try m.loadShare(0);
    try std.testing.expectError(error.MtpAfterShare, m.loadMtp(io, mtp_dir));
    const w = m.weights.mtp orelse return error.MtpNotLoaded;
    const h = m.config.hidden;
    try std.testing.expectEqual(.{ h, 2 * h, @as(u8, 4) }, .{ w.fc.n, w.fc.k, w.fc.bits });
    try std.testing.expect(w.layer.mixer == .attention);
    try std.testing.expectEqual(m.config.experts, w.layer.moe.?.gate.experts);

    var runner = try Runner.init(a, m, 1, 64);
    defer runner.deinit();
    var sc = try mtp.Scratch.init(m.device, m.frame.capacity, h, m.config.vocab);
    defer sc.deinit();
    const rows = ids.len;
    const input = try m.device.buffer(rows * h * 2, @import("metal").ResourceOptions.shared);
    defer input.deinit();
    @memcpy(input.slice(u16, rows * h), hidden);
    var round = try Round.init(a, &.{.{ .slot = 0, .start = 0, .capacity = runner.capacity, .ids = ids }}, @intCast(m.config.conv_kernel), 1, @intCast(m.config.vocab));
    defer round.deinit();
    const out = try mtp.pass(&runner, &round, &sc, .{ .buffer = input });
    const got: [*]const u16 = @ptrCast(@alignCast(out.buffer.contents() + out.offset));
    // bf16 rounding through fc, attention and eight of 256 experts (a near-tied pick can flip), not 4-bit noise: both quantize alike
    for (0..rows) |r| {
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        for (got[r * h ..][0..h], want[r * h ..][0..h]) |g, x| {
            dot += f32of(g) * f32of(x);
            gg += f32of(g) * f32of(g);
            ww += f32of(x) * f32of(x);
        }
        try std.testing.expect(dot / @sqrt(gg * ww) > 0.995);
    }
}

test "Ornith MTP drafts: once a step on the target's rows, its greedy drafts match mlx-lm's on the bench prompts" {
    const dir = std.mem.span(platform.c.getenv("TF_QWEN27_DIR") orelse return error.SkipZigTest);
    const mtp_dir = std.mem.span(platform.c.getenv("TF_QWEN27_MTP_DIR") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fixture = try Fixture.parse(arena.allocator(), @embedFile("testdata/ornith-mtp-drafts.safetensors"));
    const ids = try fixture.tensor(arena.allocator(), "ids", u32);
    const lengths = try fixture.tensor(arena.allocator(), "lengths", u32);
    const want = try fixture.tensor(arena.allocator(), "drafts", u32);
    const logits = try fixture.tensor(arena.allocator(), "logits", f32);
    const steps = want.len / candidates / lengths.len;

    const m = try Model.loadDeferred(a, io, dir, 128);
    defer m.deinit();
    try m.loadMtp(io, mtp_dir);
    try m.loadShare(0);
    var runner = try Runner.init(a, m, 1, 256);
    defer runner.deinit();
    var drafter = try mtp.Drafter.init(&runner);
    defer drafter.deinit();
    const s = Session{ .runner = &runner };
    var same: usize = 0;
    var first: usize = 0;
    for (lengths, 0..) |len, p| {
        const seq = ids[first..][0..len];
        const prompt = len - steps;
        try runner.reset(0);
        // mlx-lm's greedy reply teacher-forced, so both sides draft from the same tokens
        try s.decodeChunk(seq[0..prompt], true);
        for (0..steps) |k| {
            const got = if (k == 0) try drafter.step(0, seq[1 .. prompt + 1]) else blk: {
                try s.step(seq[prompt + k - 1]);
                break :blk try drafter.step(0, seq[prompt + k ..][0..1]);
            };
            const i = (p * steps + k) * candidates;
            // every draft is one of the reference's top picks, near its best: the 4-bit MoE target's rows differ
            // between the two kernels' rounding (mlx-lm's own stepped and whole-pass drafts differ at 131 of 2,048)
            const at = std.mem.indexOfScalar(u32, want[i..][0..candidates], got) orelse return error.DraftNotInReferenceTop;
            try std.testing.expect(logits[i] - logits[i + at] < 6); // 5.4 seen
            same += @intFromBool(at == 0);
        }
        first += len;
    }
    try std.testing.expectError(error.TapsDiffer, drafter.step(0, ids[ids.len - 1 ..])); // no target pass since the last
    try std.testing.expectError(error.RoundNotReady, drafter.step(runner.slots, ids[ids.len - 1 ..]));
    try std.testing.expect(same * 100 >= steps * lengths.len * 93); // 1,912 of 2,048 seen
}
