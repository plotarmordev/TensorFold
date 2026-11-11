//! drama3-router-compact (GPU): d3_router_compact vs FP64: excluded logits, ties, held counts, poisoned rows.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const experts = 256;
const top = 8;

/// One launch: the held original ids and each row's logits (bf16 bits, every expert).
const Case = struct { held: []const u8, rows: []const [experts]u16 };

fn bf16(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16); // every logit below is exact in bf16
}

/// Every logit `rest`, the listed ids `v`.
fn row(rest: f32, ids: []const u8, v: []const f32) [experts]u16 {
    var r: [experts]u16 = @splat(bf16(rest));
    for (ids, v) |id, x| r[id] = bf16(x);
    return r;
}

fn logit(l: [experts]u16, id: u8) f64 {
    return @as(f32, @bitCast(@as(u32, l[id]) << 16));
}

fn before(l: [experts]u16, a: u8, b: u8) bool {
    return logit(l, a) > logit(l, b) or (logit(l, a) == logit(l, b) and a < b);
}

/// Held logits only: a NaN or +inf among them, or none finite, poisons the row (NaN weights, held slots picked).
fn reference(held: []const u8, logits: [experts]u16, slots: [experts]i32, picks: *[top]i32, weights: *[top]f64) void {
    var order: [experts]u8 = undefined;
    @memcpy(order[0..held.len], held);
    var max: f64 = -std.math.inf(f64);
    var poisoned = false;
    for (held) |id| {
        const v = logit(logits, id);
        poisoned = poisoned or std.math.isNan(v) or v == std.math.inf(f64);
        if (v > max) max = v;
    }
    poisoned = poisoned or max == -std.math.inf(f64);
    if (!poisoned) std.sort.pdq(u8, order[0..held.len], logits, before);
    var sum: f64 = 0;
    for (order[0..top]) |id| sum += @exp(logit(logits, id) - max);
    for (order[0..top], picks, weights) |id, *p, *w| {
        p.* = slots[id];
        w.* = if (poisoned) std.math.nan(f64) else @exp(logit(logits, id) - max) / sum;
    }
}

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.drama3);
    defer mod.unload();
    const f = try mod.function("d3_router_compact");
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const inf = std.math.inf(f32);
    const nan = std.math.nan(f32);
    const eight = [_]u8{ 1, 7, 19, 32, 65, 127, 200, 255 };
    const nine = [_]u8{ 0, 3, 64, 100, 128, 129, 200, 254, 255 };
    var all: [experts]u8 = undefined;
    for (&all, 0..) |*id, i| id.* = @intCast(i);
    const cases = [_]Case{
        .{ .held = &eight, .rows = &.{
            row(100, &eight, &.{ -2, -2, -2, -2, -2, -2, -2, -2 }), // excluded logits highest; held tie in id order
            row(100, &eight, &.{ 1, -1, 3, 0.5, -3, 2, 0, -0.5 }), // graded: weights are the held softmax renormalized
            row(nan, &eight, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }), // a NaN on an excluded id is masked away
            row(0, &eight, &.{ 0, nan, 0, 0, 0, 0, 0, 0 }), // a NaN held logit poisons the row
            row(0, &eight, &.{ 0, 0, 0, inf, 0, 0, 0, 0 }), // +inf held: poisoned
            row(inf, &eight, &.{ -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf }), // no finite held logit: poisoned
        } },
        .{ .held = &nine, .rows = &.{
            row(50, &nine, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }), // nine tie: the highest original id drops
            row(50, &nine, &.{ 1, 1, 2, 1, 1, 1, 1, 1, 1 }),
        } },
        .{ .held = &all, .rows = &.{
            row(0, &.{ 3, 200, 201, 202, 203, 204, 205, 206, 207 }, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }), // nothing masked
        } },
    };
    var n: usize = 0;
    for (cases) |c| {
        var slots: [experts]i32 = @splat(-1);
        for (c.held, 0..) |id, slot| slots[id] = @intCast(slot);
        const t = c.rows.len;
        var logits_d = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(c.rows));
        defer logits_d.free();
        var slots_d = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&slots));
        defer slots_d.free();
        var picks_d = try cuda.DeviceBuffer.alloc(gpu.d, t * top * 4);
        defer picks_d.free();
        var weights_d = try cuda.DeviceBuffer.alloc(gpu.d, t * top * 4);
        defer weights_d.free();
        var a: cuda.Args = .{};
        a.add(logits_d.ptr);
        a.add(@as(i32, experts));
        a.add(@as(i32, 0));
        a.add(picks_d.ptr);
        a.add(weights_d.ptr);
        a.add(slots_d.ptr);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(t) }, .block = .{ .x = experts } }, s, &a);
        try s.synchronize();
        const picks = try gpa.alloc(i32, t * top);
        defer gpa.free(picks);
        const weights = try gpa.alloc(f32, t * top);
        defer gpa.free(weights);
        try picks_d.download(0, std.mem.sliceAsBytes(picks));
        try weights_d.download(0, std.mem.sliceAsBytes(weights));
        for (c.rows, 0..) |logits, r| {
            var want_p: [top]i32 = undefined;
            var want_w: [top]f64 = undefined;
            reference(c.held, logits, slots, &want_p, &want_w);
            for (0..top) |k| {
                const got_p = picks[r * top + k];
                const got_w: f64 = weights[r * top + k];
                const nan_ok = std.math.isNan(want_w[k]) and std.math.isNan(got_w);
                try check.expect(got_p == want_p[k] and (nan_ok or @abs(got_w - want_w[k]) <= 1e-6), "drama3-router-compact: {d} held, row {d} pick {d}: slot {d} weight {e}, FP64 slot {d} weight {e}", .{ c.held.len, r, k, got_p, got_w, want_p[k], want_w[k] });
            }
            n += 1;
        }
    }
    check.pass("drama3-router-compact: {d} rows (8, 9 and 256 held; excluded highs, id ties, poisoned rows) match FP64", .{n});
}
