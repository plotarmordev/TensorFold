//! Ornith-1.5-122B's MTP layer on real weights: TF_QWEN27_DIR the MLX 4-bit target, TF_QWEN27_MTP_DIR the bf16
//! release's folder holding its `mtp.*` shards. testdata/ornith-mtp.safetensors is tools/zig/ornith_mtp_reference.py's.
const std = @import("std");
const platform = @import("sys");
const st = @import("core").safetensors;
const Model = @import("model.zig").Model;
const Runner = @import("decode_round.zig").Runner;
const Round = @import("round_plan.zig").Round;
const mtp = @import("mtp.zig");

const fixture = @embedFile("testdata/ornith-mtp.safetensors");

/// The fixture's tensor `name`, copied out aligned.
fn tensor(a: std.mem.Allocator, header: st.Header, name: []const u8, comptime T: type) ![]T {
    const e = header.get(name) orelse return error.MissingTensor;
    const data = fixture[8 + std.mem.readInt(u64, fixture[0..8], .little) ..][e.begin..e.end];
    const out = try a.alloc(T, data.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), data);
    return out;
}

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
    const json_len: usize = @intCast(std.mem.readInt(u64, fixture[0..8], .little));
    const header = try st.parseHeader(arena.allocator(), fixture[8..][0..json_len], fixture.len - 8 - json_len);
    const ids = try tensor(arena.allocator(), header, "ids", u32);
    const hidden = try tensor(arena.allocator(), header, "hidden", u16);
    const want = try tensor(arena.allocator(), header, "out", u16);

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
    var sc = try mtp.Scratch.init(m.device, m.frame.capacity, h);
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
