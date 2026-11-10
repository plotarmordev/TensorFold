//! Portable EXL3/mul1 encoder. The decoder remains the bit-exact reconstruction oracle.
const std = @import("std");
const exl3 = @import("exl3_format.zig");
const Allocator = std.mem.Allocator;

const beam_width = 128;
const Entry = struct { value: f16, state: u32 };
const Node = struct { cost: f64, node: u32, parent: u32 };
const Candidate = struct { cost: f64, node: u32, parent: u32, state: u32 };

pub const Result = struct {
    width: exl3.Width,
    k: usize,
    n: usize,
    trellis: []i16,
    suh: []f16,
    svh: []f16,
    reconstructed: []f64,
    error_exl3: f64,
    error_plain: f64,

    pub fn deinit(self: *Result, gpa: Allocator) void {
        gpa.free(self.trellis);
        gpa.free(self.suh);
        gpa.free(self.svh);
        gpa.free(self.reconstructed);
    }
};

pub fn packTile(w: exl3.Width, states: *const [256]u32, tile: []i16) !void {
    if (tile.len != w.tileWords()) return error.BadTile;
    @memset(tile, 0);
    var ends: [256]u32 = undefined;
    exl3.streamEnds(w, &ends);
    const ring = 128 * @as(usize, w.halves);
    for (states, ends) |state, end| for (0..16) |j| {
        const bit = (end + ring - 16 + j) % ring;
        if ((state >> @intCast(15 - j)) & 1 != 0) {
            const word: *i16 = &tile[2 * (bit / 32) + 1 - (bit % 32) / 16];
            word.* |= @as(i16, @bitCast(@as(u16, 1) << @intCast(15 - bit % 16)));
        }
    };
}

fn fwht(v: []f64, stride: usize) void {
    var d: usize = 1;
    while (d < 128) : (d *= 2) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * d) for (base..base + d) |i| {
            const a = v[i * stride];
            const b = v[(i + d) * stride];
            v[i * stride] = a + b;
            v[(i + d) * stride] = a - b;
        };
    }
}

fn rotateFast(v: []f64, rows: usize, cols: usize, axis: u1) void {
    const norm = 1.0 / @sqrt(128.0);
    if (axis == 0) {
        for (0..rows / 128) |b| for (0..cols) |c| {
            const at = b * 128 * cols + c;
            fwht(v[at..], cols);
            for (0..128) |r| v[at + r * cols] *= norm;
        };
    } else {
        for (0..rows) |r| for (0..cols / 128) |b| {
            const at = r * cols + b * 128;
            fwht(v[at .. at + 128], 1);
            for (0..128) |c| v[at + c] *= norm;
        };
    }
}

fn lessEntry(_: void, a: Entry, b: Entry) bool {
    return a.value < b.value or (a.value == b.value and a.state < b.state);
}

fn nearest(book: []const Entry, x: f64) u32 {
    var lo: usize = 0;
    var hi = book.len;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (book[mid].value <= x) lo = mid else hi = mid;
    }
    if (lo == 0) return book[0].state;
    const a = @abs(@as(f64, book[lo - 1].value) - x);
    const b = @abs(@as(f64, book[lo].value) - x);
    return if (a < b) book[lo - 1].state else book[lo].state;
}

fn plainTile(target: *const [256]f64, book: []const Entry, w: exl3.Width, states: *[256]u32) void {
    const ring = 128 * @as(usize, w.halves);
    var votes: [768]i32 = @splat(0);
    var ends: [256]u32 = undefined;
    exl3.streamEnds(w, &ends);
    for (target, ends, 0..) |x, end, p| {
        const state = nearest(book, x);
        for (0..16) |j| votes[(end + ring - 16 + j) % ring] += if ((state >> @intCast(15 - j)) & 1 != 0) 1 else -1;
        states[p] = 0;
    }
    var tile: [48]i16 = @splat(0);
    for (0..ring) |j| if (votes[j] >= 0) {
        tile[2 * (j / 32) + 1 - (j % 32) / 16] |= @as(i16, @bitCast(@as(u16, 1) << @intCast(15 - j % 16)));
    };
    exl3.tileStates(w, tile[0..w.tileWords()], states);
}

fn lessNode(_: void, a: Node, b: Node) bool {
    return a.cost < b.cost or (a.cost == b.cost and a.node < b.node);
}
fn lessCandidate(_: void, a: Candidate, b: Candidate) bool {
    return a.node < b.node or (a.node == b.node and (a.cost < b.cost or (a.cost == b.cost and a.state < b.state)));
}

fn emitBits(w: exl3.Width, p: usize) u4 {
    return if (w.halves % 2 == 0) @intCast(w.halves / 2) else @intCast(w.halves / 2 + p % 2);
}

fn weightedTile(target: *const [256]f64, importance: *const [256]f64, w: exl3.Width, start: u32, plain: *const [256]u32, states: *[256]u32) void {
    var current: [beam_width]Node = undefined;
    var next: [beam_width]Node = undefined;
    var candidates: [beam_width * 8]Candidate = undefined;
    var history_nodes: [256 * beam_width]u32 = undefined;
    var history_parents: [256 * beam_width]u32 = undefined;
    const first_bits = emitBits(w, 0);
    const first_out = emitBits(w, 1);
    var count: usize = 0;
    for (0..@as(usize, 1) << first_bits) |suffix| {
        const state: u32 = @intCast((@as(u32, start) << first_bits) | suffix);
        const node = state & (@as(u32, 0xffff) >> first_out);
        const d = target[0] - @as(f64, exl3.codebookValue(.mul1, @intCast(state)));
        candidates[count] = .{ .cost = importance[0] * d * d, .node = node, .parent = 0, .state = state };
        count += 1;
    }
    for (0..256) |p| {
        if (p != 0) {
            count = 0;
            const kin = emitBits(w, p);
            const kout = emitBits(w, (p + 1) % 256);
            const out_mask = @as(u32, 0xffff) >> kout;
            for (current[0..beam_width], 0..) |prev, parent| for (0..@as(usize, 1) << kin) |suffix| {
                const state: u32 = @truncate((@as(u32, prev.node) << kin) | suffix);
                const d = target[p] - @as(f64, exl3.codebookValue(.mul1, @intCast(state)));
                candidates[count] = .{ .cost = prev.cost + importance[p] * d * d, .node = state & out_mask, .parent = @intCast(parent), .state = state };
                count += 1;
            };
        }
        std.mem.sort(Candidate, candidates[0..count], {}, lessCandidate);
        var unique: usize = 0;
        for (candidates[0..count]) |candidate| {
            if (unique == 0 or candidates[unique - 1].node != candidate.node) {
                candidates[unique] = candidate;
                unique += 1;
            }
        }
        std.mem.sort(Candidate, candidates[0..unique], {}, struct {
            fn less(_: void, a: Candidate, b: Candidate) bool {
                return a.cost < b.cost or (a.cost == b.cost and a.node < b.node);
            }
        }.less);
        for (0..@min(unique, beam_width)) |j| next[j] = .{ .cost = candidates[j].cost, .node = candidates[j].node, .parent = candidates[j].parent };
        var used = @min(unique, beam_width);
        var found = false;
        for (next[0..used]) |node| found = found or node.node == start;
        if (!found) for (candidates[0..unique]) |candidate| if (candidate.node == start) {
            const at = if (used < beam_width) used else used - 1;
            next[at] = .{ .cost = candidate.cost, .node = candidate.node, .parent = candidate.parent };
            if (used < beam_width) used += 1;
            break;
        };
        while (used < beam_width) : (used += 1) next[used] = .{ .cost = std.math.inf(f64), .node = 0, .parent = 0 };
        std.mem.sort(Node, next[0..beam_width], {}, lessNode);
        for (next, 0..) |node, j| {
            history_nodes[p * beam_width + j] = node.node;
            history_parents[p * beam_width + j] = node.parent;
        }
        current = next;
    }
    var at: usize = 0;
    var found = false;
    for (current, 0..) |node, j| if (node.node == start) {
        at = j;
        found = true;
        break;
    };
    if (!found) {
        states.* = plain.*;
        return;
    }
    var p: usize = 256;
    while (p > 0) {
        p -= 1;
        const node = history_nodes[p * beam_width + at];
        const parent = history_parents[p * beam_width + at];
        const prev = if (p == 0) start else history_nodes[(p - 1) * beam_width + parent];
        states[p] = @truncate((@as(u32, prev) << emitBits(w, p)) | node);
        at = parent;
    }
    var weighted_cost: f64 = 0;
    var plain_cost: f64 = 0;
    for (target, importance, plain, states) |x, weight, p0, p1| {
        const a = x - @as(f64, exl3.codebookValue(.mul1, @intCast(p0)));
        const b = x - @as(f64, exl3.codebookValue(.mul1, @intCast(p1)));
        plain_cost += weight * a * a;
        weighted_cost += weight * b * b;
    }
    if (plain_cost < weighted_cost) states.* = plain.*;
}

/// Exact sum_n (W-Q)^T H (W-Q), with matrices in [K,N] layout.
pub fn hessianError(w: []const f64, q: []const f64, h: []const f64, k: usize, n: usize) f64 {
    var total: f64 = 0;
    var d = std.heap.page_allocator.alloc(f64, k) catch @panic("oom");
    defer std.heap.page_allocator.free(d);
    for (0..n) |col| {
        for (0..k) |i| d[i] = w[i * n + col] - q[i * n + col];
        for (0..k) |i| {
            var v: f64 = 0;
            for (0..k) |j| v += h[i * k + j] * d[j];
            total += d[i] * v;
        }
    }
    return total;
}

fn dequant(gpa: Allocator, width: exl3.Width, k: usize, n: usize, trellis: []const i16, suh: []const f16, svh: []const f16) ![]f64 {
    const out = try gpa.alloc(f64, k * n);
    errdefer gpa.free(out);
    const scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(scratch);
    exl3.dequantize(width, .mul1, trellis, k / 16, n / 16, suh, svh, out, scratch);
    return out;
}

/// Encode one [K,N] matrix with a full [K,K] calibration Hessian. The exact-error fallback is tensor-wide.
pub fn encode(gpa: Allocator, weight: []const f64, hessian: []const f64, k: usize, n: usize, width: exl3.Width) !Result {
    if (k % 128 != 0 or n % 128 != 0 or weight.len != k * n or hessian.len != k * k or width.halves < 4 or width.halves > 6) return error.BadShape;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const rotated = try a.dupe(f64, weight);
    const suh = try gpa.alloc(f16, k);
    errdefer gpa.free(suh);
    for (suh, 0..) |*s, i| {
        var x = @as(u64, i + 1) *% 0x9E3779B97F4A7C15;
        x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
        s.* = if (x & 1 == 0) 1 else -1;
        for (0..n) |j| rotated[i * n + j] /= @as(f64, s.*);
    }
    rotateFast(rotated, k, n, 0);
    const svh = try gpa.alloc(f16, n);
    errdefer gpa.free(svh);
    for (0..n) |j| {
        var sum: f64 = 0;
        for (0..k) |i| sum += rotated[i * n + j] * rotated[i * n + j];
        const scale = @sqrt(sum / @as(f64, @floatFromInt(k))) / 1.24371088;
        svh[j] = @floatCast(if (scale > 1e-12) scale else 1.0);
        for (0..k) |i| rotated[i * n + j] /= @as(f64, svh[j]);
    }
    rotateFast(rotated, k, n, 1);

    const hd = try a.alloc(f64, k);
    for (0..k / 128) |b| for (0..128) |r| {
        const row = b * 128 + r;
        var sum: f64 = 0;
        for (0..128) |i| for (0..128) |j| {
            const u = @as(f64, suh[b * 128 + i]);
            const v = @as(f64, suh[b * 128 + j]);
            sum += u * v * hessian[(b * 128 + i) * k + b * 128 + j] * had(r, i) * had(r, j) / 128.0;
        };
        hd[row] = @max(0, sum);
    };
    const gd = try a.alloc(f64, n);
    for (0..n / 128) |b| {
        var sum: f64 = 0;
        for (svh[b * 128 ..][0..128]) |s| sum += @as(f64, s) * @as(f64, s);
        @memset(gd[b * 128 ..][0..128], sum / 128.0);
    }

    const book = try a.alloc(Entry, 65536);
    for (book, 0..) |*e, i| e.* = .{ .value = exl3.codebookValue(.mul1, @intCast(i)), .state = @intCast(i) };
    std.mem.sort(Entry, book, {}, lessEntry);
    const tiles = (k / 16) * (n / 16);
    const plain_trellis = try a.alloc(i16, tiles * width.tileWords());
    const weighted_trellis = try gpa.alloc(i16, tiles * width.tileWords());
    errdefer gpa.free(weighted_trellis);
    var placement: [256]exl3.Placement = undefined;
    exl3.tilePlacement(&placement);
    for (0..k / 16) |tr| for (0..n / 16) |tc| {
        var target: [256]f64 = undefined;
        var importance: [256]f64 = undefined;
        for (0..256) |p| {
            const at = placement[p];
            const row = tr * 16 + at.row;
            const col = tc * 16 + at.col;
            target[p] = rotated[row * n + col];
            importance[p] = hd[row] * gd[col];
        }
        var plain: [256]u32 = undefined;
        var weighted: [256]u32 = undefined;
        plainTile(&target, book, width, &plain);
        const start = plain[255] & (@as(u32, 0xffff) >> emitBits(width, 0));
        weightedTile(&target, &importance, width, start, &plain, &weighted);
        const tile_at = (tr * (n / 16) + tc) * width.tileWords();
        try packTile(width, &plain, plain_trellis[tile_at..][0..width.tileWords()]);
        try packTile(width, &weighted, weighted_trellis[tile_at..][0..width.tileWords()]);
    };
    const plain_q = try dequant(a, width, k, n, plain_trellis, suh, svh);
    const weighted_q = try dequant(gpa, width, k, n, weighted_trellis, suh, svh);
    const plain_error = hessianError(weight, plain_q, hessian, k, n);
    const weighted_error = hessianError(weight, weighted_q, hessian, k, n);
    if (plain_error < weighted_error) {
        @memcpy(weighted_trellis, plain_trellis);
        @memcpy(weighted_q, plain_q);
        return .{ .width = width, .k = k, .n = n, .trellis = weighted_trellis, .suh = suh, .svh = svh, .reconstructed = weighted_q, .error_exl3 = plain_error, .error_plain = plain_error };
    }
    return .{ .width = width, .k = k, .n = n, .trellis = weighted_trellis, .suh = suh, .svh = svh, .reconstructed = weighted_q, .error_exl3 = weighted_error, .error_plain = plain_error };
}

fn had(r: usize, c: usize) f64 {
    return if (@popCount(r & c) % 2 == 0) 1 else -1;
}

fn expectReconstruction(weight: []const f64, hessian: []const f64, k: usize, n: usize, halves: u4) !void {
    const gpa = std.testing.allocator;
    var result = try encode(gpa, weight, hessian, k, n, .{ .halves = halves });
    defer result.deinit(gpa);
    const actual = try dequant(gpa, result.width, k, n, result.trellis, result.suh, result.svh);
    defer gpa.free(actual);
    try std.testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(result.reconstructed), std.mem.sliceAsBytes(actual)));
    try std.testing.expect(result.error_exl3 <= result.error_plain);
}

test "packTile is tileStates' inverse at every requested width" {
    const gpa = std.testing.allocator;
    const book = try gpa.alloc(Entry, 65536);
    defer gpa.free(book);
    for (book, 0..) |*e, i| e.* = .{ .value = exl3.codebookValue(.mul1, @intCast(i)), .state = @intCast(i) };
    std.mem.sort(Entry, book, {}, lessEntry);
    var target: [256]f64 = undefined;
    for (&target, 0..) |*v, i| v.* = @sin(@as(f64, @floatFromInt(i)) * 0.13);
    for ([_]u4{ 4, 5, 6 }) |halves| {
        const w = exl3.Width{ .halves = halves };
        var states: [256]u32 = undefined;
        plainTile(&target, book, w, &states);
        var tile: [48]i16 = undefined;
        try packTile(w, &states, tile[0..w.tileWords()]);
        var actual: [256]u32 = undefined;
        exl3.tileStates(w, tile[0..w.tileWords()], &actual);
        try std.testing.expectEqualSlices(u32, &states, &actual);
    }
}

test "weighted trellis beats majority rounding" {
    const gpa = std.testing.allocator;
    const book = try gpa.alloc(Entry, 65536);
    defer gpa.free(book);
    for (book, 0..) |*e, i| e.* = .{ .value = exl3.codebookValue(.mul1, @intCast(i)), .state = @intCast(i) };
    std.mem.sort(Entry, book, {}, lessEntry);
    for ([_]u4{ 4, 5, 6 }) |halves| {
        var target: [256]f64 = undefined;
        var importance: [256]f64 = @splat(1);
        for (&target, 0..) |*v, i| v.* = @sin(@as(f64, @floatFromInt(i)) * 0.13);
        var plain: [256]u32 = undefined;
        var weighted: [256]u32 = undefined;
        const w = exl3.Width{ .halves = halves };
        plainTile(&target, book, w, &plain);
        weightedTile(&target, &importance, w, plain[255] & (@as(u32, 0xffff) >> emitBits(w, 0)), &plain, &weighted);
        var bad: usize = 0;
        for (0..256) |p| {
            const b = emitBits(w, p);
            const prev = if (p == 0) weighted[255] else weighted[p - 1];
            const expected: u32 = ((prev << b) | (weighted[p] & ((@as(u32, 1) << b) - 1))) & 0xffff;
            if (expected != weighted[p]) bad += 1;
        }
        try std.testing.expectEqual(@as(usize, 0), bad);
        var pe: f64 = 0;
        var we: f64 = 0;
        for (target, plain, weighted) |x, p, q| {
            pe += (x - @as(f64, exl3.codebookValue(.mul1, @intCast(p)))) * (x - @as(f64, exl3.codebookValue(.mul1, @intCast(p))));
            we += (x - @as(f64, exl3.codebookValue(.mul1, @intCast(q)))) * (x - @as(f64, exl3.codebookValue(.mul1, @intCast(q))));
        }
        try std.testing.expect(we <= pe);
    }
}

test "Drama gate shape is bit exact and Hessian aware" {
    const k = 2048;
    const n = 512;
    const w = try std.testing.allocator.alloc(f64, k * n);
    defer std.testing.allocator.free(w);
    const h = try std.testing.allocator.alloc(f64, k * k);
    defer std.testing.allocator.free(h);
    for (w, 0..) |*v, i| v.* = @sin(@as(f64, @floatFromInt(i)) * 0.013) * 0.1;
    for (h, 0..) |*v, i| v.* = if (i / k == i % k) 1 else 0;
    try expectReconstruction(w, h, k, n, 4);
}

test "Drama down shape half-bit stream is bit exact" {
    const k = 512;
    const n = 2048;
    const w = try std.testing.allocator.alloc(f64, k * n);
    defer std.testing.allocator.free(w);
    const h = try std.testing.allocator.alloc(f64, k * k);
    defer std.testing.allocator.free(h);
    for (w, 0..) |*v, i| v.* = @cos(@as(f64, @floatFromInt(i)) * 0.021) * 0.1;
    for (h, 0..) |*v, i| v.* = if (i / k == i % k) 1 else 0;
    try expectReconstruction(w, h, k, n, 5);
}

test "small three-bit tensor round trip" {
    const k = 128;
    const n = 128;
    var w: [k * n]f64 = undefined;
    var h: [k * k]f64 = @splat(0);
    for (&w, 0..) |*v, i| v.* = @sin(@as(f64, @floatFromInt(i)) * 0.031) * 0.1;
    for (0..k) |i| {
        h[i * k + i] = 2;
        if (i > 0) {
            h[i * k + i - 1] = -0.25;
            h[(i - 1) * k + i] = -0.25;
        }
    }
    try expectReconstruction(&w, &h, k, n, 6);
}
