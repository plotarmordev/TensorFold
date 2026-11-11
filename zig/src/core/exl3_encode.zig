//! Portable EXL3/mul1 encoder. The decoder remains the bit-exact reconstruction oracle.
const std = @import("std");
const exl3 = @import("exl3_format.zig");
const Allocator = std.mem.Allocator;

const beam_width = 128;
const max_workers = 64;
const Entry = struct { value: f16, state: u32 };
const Candidate = struct { cost: f64, node: u32, parent: u32, state: u32 };

pub const Result = struct {
    width: exl3.Width,
    k: usize,
    n: usize,
    trellis: []i16,
    suh: []f16,
    svh: []f16,
    reconstructed: []f64,

    pub fn deinit(self: *Result, gpa: Allocator) void {
        gpa.free(self.trellis);
        gpa.free(self.suh);
        gpa.free(self.svh);
        gpa.free(self.reconstructed);
    }
};

/// The mul1 codebook by state, and sorted by value.
const Book = struct {
    values: [65536]f64,
    sorted: [65536]Entry,

    fn init(book: *Book) void {
        for (&book.values, &book.sorted, 0..) |*v, *e, i| {
            e.* = .{ .value = exl3.codebookValue(.mul1, @intCast(i)), .state = @intCast(i) };
            v.* = e.value;
        }
        std.sort.pdq(Entry, &book.sorted, {}, lessEntry);
    }

    /// Index of the first sorted entry above x.
    fn above(book: *const Book, x: f64) usize {
        return std.sort.partitionPoint(Entry, &book.sorted, x, struct {
            fn below(t: f64, e: Entry) bool {
                return e.value <= t;
            }
        }.below);
    }
};

fn lessEntry(_: void, a: Entry, b: Entry) bool {
    return a.value < b.value or (a.value == b.value and a.state < b.state);
}
fn lessCost(_: void, a: Candidate, b: Candidate) bool {
    return a.cost < b.cost or (a.cost == b.cost and a.state < b.state);
}
fn lessNode(_: void, a: Candidate, b: Candidate) bool {
    return a.node < b.node or (a.node == b.node and lessCost({}, a, b));
}

pub fn packTile(w: exl3.Width, states: *const [256]u32, tile: []i16) void {
    std.debug.assert(tile.len == w.tileWords());
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

/// Stream bits value p shifts into its 16-bit state.
fn emitBits(w: exl3.Width, p: usize) u5 {
    return @intCast(w.halves / 2 + if (w.isHalf()) p % 2 else 0);
}

/// Weighted beam Viterbi over a tile's ring; with `start`, value 0's wrapped window is forced to it, closing the ring.
fn beam(target: *const [256]f64, importance: *const [256]f64, book: *const Book, w: exl3.Width, start: ?u32, states: *[256]u32) void {
    var candidates: [beam_width * 16]Candidate = undefined;
    var history: [256][beam_width]u32 = undefined;
    var parents: [256][beam_width]u32 = undefined;
    var costs: [beam_width]f64 = undefined;
    var ends: [256]u32 = undefined;
    exl3.streamEnds(w, &ends);
    const ring = 128 * @as(u32, w.halves);
    const k0 = emitBits(w, 0);
    const wrapped = 16 - k0;
    var live: usize = 0;
    for (0..256) |p| {
        var count: usize = 0;
        if (p == 0) {
            if (start) |s| {
                for (0..@as(u32, 1) << k0) |suffix| {
                    candidates[count] = .{ .cost = 0, .node = 0, .parent = 0, .state = (s << k0) | @as(u32, @intCast(suffix)) };
                    count += 1;
                }
            } else {
                const lo = @min(book.above(target[0]) -| candidates.len / 2, book.sorted.len - candidates.len);
                for (book.sorted[lo..][0..candidates.len]) |e| {
                    candidates[count] = .{ .cost = 0, .node = 0, .parent = 0, .state = e.state };
                    count += 1;
                }
            }
        } else {
            const kin = emitBits(w, p);
            // the newest stream bits land in the wrapped region at the end of the ring
            const forced: u5 = if (start == null) 0 else @intCast(@min(kin, ends[p] -| (ring - wrapped)));
            const required: u32 = if (forced == 0) 0 else (start.? >> @intCast(ring - ends[p])) & ((@as(u32, 1) << forced) - 1);
            for (history[p - 1][0..live], costs[0..live], 0..) |prev, prev_cost, parent| {
                for (0..@as(u32, 1) << (kin - forced)) |free| {
                    candidates[count] = .{ .cost = prev_cost, .node = 0, .parent = @intCast(parent), .state = ((prev << kin) | (@as(u32, @intCast(free)) << forced) | required) & 0xffff };
                    count += 1;
                }
            }
        }
        // states sharing the bits the next value keeps have identical futures
        const node_mask = @as(u32, 0xffff) >> emitBits(w, (p + 1) % 256);
        for (candidates[0..count]) |*c| {
            const d = target[p] - book.values[c.state];
            c.cost += importance[p] * d * d;
            c.node = c.state & node_mask;
        }
        std.sort.pdq(Candidate, candidates[0..count], {}, lessNode);
        var unique: usize = 0;
        for (candidates[0..count]) |c| if (unique == 0 or candidates[unique - 1].node != c.node) {
            candidates[unique] = c;
            unique += 1;
        };
        std.sort.pdq(Candidate, candidates[0..unique], {}, lessCost);
        live = @min(unique, beam_width);
        for (candidates[0..live], 0..) |c, j| {
            history[p][j] = c.state;
            parents[p][j] = c.parent;
            costs[j] = c.cost;
        }
    }
    var at: u32 = 0;
    var p: usize = 256;
    while (p > 0) {
        p -= 1;
        states[p] = history[p][at];
        at = parents[p][at];
    }
}

/// Tail-biting search: a free pass over the tile turned by half picks the closing state, a second pass closes there.
fn searchTile(target: *const [256]f64, importance: *const [256]f64, book: *const Book, w: exl3.Width, states: *[256]u32) void {
    var turned_target: [256]f64 = undefined;
    var turned_importance: [256]f64 = undefined;
    for (0..256) |p| {
        turned_target[p] = target[(p + 128) % 256];
        turned_importance[p] = importance[(p + 128) % 256];
    }
    beam(&turned_target, &turned_importance, book, w, null, states);
    beam(target, importance, book, w, states[127] & (@as(u32, 0xffff) >> emitBits(w, 0)), states);
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

/// One band of tile rows: the chosen trellis words and their rotated-domain values.
const Job = struct {
    rotated: []const f64,
    hd: []const f64,
    gd: []const f64,
    book: *const Book,
    width: exl3.Width,
    n: usize,
    trellis: []i16,
    q: []f64,

    fn rows(job: *const Job, first: usize, last: usize) void {
        var placement: [256]exl3.Placement = undefined;
        exl3.tilePlacement(&placement);
        const words = job.width.tileWords();
        for (first..last) |tr| for (0..job.n / 16) |tc| {
            var target: [256]f64 = undefined;
            var importance: [256]f64 = undefined;
            for (placement, &target, &importance) |at, *x, *weight| {
                const row = tr * 16 + at.row;
                const col = tc * 16 + at.col;
                x.* = job.rotated[row * job.n + col];
                weight.* = job.hd[row] * job.gd[col];
            }
            var states: [256]u32 = undefined;
            searchTile(&target, &importance, job.book, job.width, &states);
            const tile = (tr * (job.n / 16) + tc) * words;
            packTile(job.width, &states, job.trellis[tile..][0..words]);
            for (placement, states) |at, s| job.q[(tr * 16 + at.row) * job.n + tc * 16 + at.col] = job.book.values[s];
        };
    }
};

/// Encode [K,N] with its full [K,K] Hessian; `reconstructed` comes from the chosen states, not the packed trellis.
pub fn encode(gpa: Allocator, weight: []const f64, hessian: []const f64, k: usize, n: usize, width: exl3.Width) !Result {
    if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0 or weight.len != k * n or hessian.len != k * k or width.halves < 4 or width.halves > 8) return error.BadShape;
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
    for (svh, 0..) |*s, j| {
        var sum: f64 = 0;
        for (0..k) |i| sum += rotated[i * n + j] * rotated[i * n + j];
        // a zero scale keeps a dead output exactly zero; any other leaks codebook noise across its Hadamard block
        if (sum == 0) {
            s.* = 0;
            continue;
        }
        // fp16 must not round a live scale to zero: the targets divide by it
        s.* = @floatCast(@max(@sqrt(sum / @as(f64, @floatFromInt(k))) / 1.24371088, std.math.floatTrueMin(f16)));
        for (0..k) |i| rotated[i * n + j] /= @as(f64, s.*);
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

    const book = try a.create(Book);
    book.init();
    const words = (k / 16) * (n / 16) * width.tileWords();
    const trellis = try gpa.alloc(i16, words);
    errdefer gpa.free(trellis);
    const q = try gpa.alloc(f64, k * n);
    errdefer gpa.free(q);
    const job = Job{ .rotated = rotated, .hd = hd, .gd = gd, .book = book, .width = width, .n = n, .trellis = trellis, .q = q };
    // tile rows are independent: each worker's band matches a serial pass byte for byte
    const tiles_k = k / 16;
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, tiles_k);
    const per = (tiles_k + workers - 1) / workers;
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |t| threads[t] = std.Thread.spawn(.{}, Job.rows, .{ &job, @min(tiles_k, t * per), @min(tiles_k, (t + 1) * per) }) catch null;
    job.rows(0, @min(tiles_k, per));
    for (1..workers) |t| if (threads[t]) |thread| thread.join() else job.rows(@min(tiles_k, t * per), @min(tiles_k, (t + 1) * per));

    exl3.restore(q, k, n, suh, svh);
    return .{ .width = width, .k = k, .n = n, .trellis = trellis, .suh = suh, .svh = svh, .reconstructed = q };
}

fn had(r: usize, c: usize) f64 {
    return if (@popCount(r & c) % 2 == 0) 1 else -1;
}

/// Decode the packed trellis and require the encoder's own reconstruction bit for bit; returns the Hessian error.
fn expectBitExact(weight: []const f64, hessian: []const f64, k: usize, n: usize, halves: u16) !f64 {
    const gpa = std.testing.allocator;
    var result = try encode(gpa, weight, hessian, k, n, .{ .halves = halves });
    defer result.deinit(gpa);
    const decoded = try gpa.alloc(f64, k * n);
    defer gpa.free(decoded);
    const scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(scratch);
    exl3.dequantize(result.width, .mul1, result.trellis, k / 16, n / 16, result.suh, result.svh, decoded, scratch);
    try std.testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(result.reconstructed), std.mem.sliceAsBytes(decoded)));
    return hessianError(weight, result.reconstructed, hessian, k, n);
}

/// Bit exact at 2.0, 2.5, 3.0 and 4.0 bits, with error relative to W^T H W falling under `first` and with every width.
fn expectWidths(weight: []const f64, hessian: []const f64, k: usize, n: usize, first: f64) !void {
    const zero = try std.testing.allocator.alloc(f64, k * n);
    defer std.testing.allocator.free(zero);
    @memset(zero, 0);
    const norm = hessianError(weight, zero, hessian, k, n);
    var last = first;
    for ([_]u16{ 4, 5, 6, 8 }) |halves| {
        const relative = try expectBitExact(weight, hessian, k, n, halves) / norm;
        try std.testing.expect(relative < last);
        last = relative;
    }
}

fn gaussian(seed: u64, out: []f64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (out) |*v| v.* = prng.random().floatNorm(f64);
}

test "the closed ring decodes to its chosen states and stays near the free beam" {
    const book = try std.testing.allocator.create(Book);
    defer std.testing.allocator.destroy(book);
    book.init();
    var target: [256]f64 = undefined;
    const importance: [256]f64 = @splat(1);
    for ([_]u16{ 4, 5, 6, 8 }) |halves| {
        const w = exl3.Width{ .halves = halves };
        var free_error: f64 = 0;
        var closed_error: f64 = 0;
        for (0..16) |seed| {
            gaussian(seed, &target);
            var free: [256]u32 = undefined;
            var closed: [256]u32 = undefined;
            beam(&target, &importance, book, w, null, &free);
            searchTile(&target, &importance, book, w, &closed);
            var tile: [64]i16 = undefined;
            packTile(w, &closed, tile[0..w.tileWords()]);
            var decoded: [256]u32 = undefined;
            exl3.tileStates(w, tile[0..w.tileWords()], &decoded);
            try std.testing.expectEqualSlices(u32, &closed, &decoded);
            for (target, free, closed) |x, f, c| {
                free_error += (x - book.values[f]) * (x - book.values[f]);
                closed_error += (x - book.values[c]) * (x - book.values[c]);
            }
        }
        // the free beam ignores the ring; closing it costs about a tenth more on average
        try std.testing.expect(closed_error <= 1.2 * free_error);
    }
}

test "seeded Gaussian gate [512,2048] with H = I" {
    const k = 2048;
    const n = 512;
    const w = try std.testing.allocator.alloc(f64, k * n);
    defer std.testing.allocator.free(w);
    const h = try std.testing.allocator.alloc(f64, k * k);
    defer std.testing.allocator.free(h);
    gaussian(23, w);
    for (h, 0..) |*v, i| v.* = if (i / k == i % k) 1 else 0;
    try expectWidths(w, h, k, n, 0.15);
}

test "the Hessian-weighted search beats the same search with H = I under an uneven Hessian" {
    const k = 256;
    const n = 256;
    const rank = 4;
    var w: [k * n]f64 = undefined;
    var x: [rank * k]f64 = undefined;
    var h: [k * k]f64 = @splat(0);
    var identity: [k * k]f64 = @splat(0);
    gaussian(26, &w);
    gaussian(27, &x);
    // few calibration rows: the rotated diagonal varies widely within each 128-row block
    for (0..k) |i| for (0..k) |j| for (0..rank) |r| {
        h[i * k + j] += x[r * k + i] * x[r * k + j] / rank;
    };
    for (0..k) |i| identity[i * k + i] = 1;
    for ([_]u16{ 4, 5, 6, 8 }) |halves| {
        const weighted = try expectBitExact(&w, &h, k, n, halves);
        var unweighted = try encode(std.testing.allocator, &w, &identity, k, n, .{ .halves = halves });
        defer unweighted.deinit(std.testing.allocator);
        try std.testing.expect(weighted < 0.9 * hessianError(&w, unweighted.reconstructed, &h, k, n));
    }
}

test "Drama down shape [2048,512] with a correlated Hessian" {
    const k = 512;
    const n = 2048;
    const w = try std.testing.allocator.alloc(f64, k * n);
    defer std.testing.allocator.free(w);
    const h = try std.testing.allocator.alloc(f64, k * k);
    defer std.testing.allocator.free(h);
    gaussian(24, w);
    for (w) |*v| v.* *= 0.02;
    // AR(1) inputs: H[i][j] = rho^|i-j|, positive definite
    for (h, 0..) |*v, i| v.* = std.math.pow(f64, 0.6, @floatFromInt(@max(i / k, i % k) - @min(i / k, i % k)));
    try expectWidths(w, h, k, n, 0.15);
}

test "pruned outputs stay out of the live columns' error" {
    const k = 128;
    const n = 128;
    var w: [k * n]f64 = undefined;
    var h: [k * k]f64 = @splat(0);
    gaussian(25, &w);
    for (&w) |*v| v.* *= 0.1;
    // a fifth of the outputs pruned to zero, as in Drama 3's gate experts
    for (0..k) |i| for (0..n) |j| if (j % 5 == 0) {
        w[i * n + j] = 0;
    };
    for (0..k) |i| h[i * k + i] = 1;
    try expectWidths(&w, &h, k, n, 0.15);
}

test "small tensor with a tridiagonal Hessian" {
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
    try expectWidths(&w, &h, k, n, 1);
}

test "an all-zero matrix decodes to exactly zero at every width" {
    const k = 128;
    const n = 128;
    const w: [k * n]f64 = @splat(0);
    var h: [k * k]f64 = @splat(0);
    for (0..k) |i| h[i * k + i] = 1;
    for ([_]u16{ 4, 5, 6, 8 }) |halves| try std.testing.expectEqual(@as(f64, 0), try expectBitExact(&w, &h, k, n, halves));
}

test "an empty dimension is a bad shape" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.BadShape, encode(gpa, &.{}, &.{}, 0, 128, .{ .halves = 4 }));
    try std.testing.expectError(error.BadShape, encode(gpa, &.{}, &.{}, 128, 0, .{ .halves = 4 }));
}
