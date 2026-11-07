//! Woof's text side over Unicode code points: its character ids, its sentence and clause pieces (textsplit.py) and
//! the stall measure its redraw check uses.
const std = @import("std");

const ascii = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()&/%$€£@#+=*_<>[]";
const ellipsis: u32 = 0x2026;
const em_dash: u32 = 0x2014;
pub const clause_max = 120;
const min_piece = 30;
const conjunctions = [_][]const u8{ "and", "but", "or", "nor", "that", "which", "who", "whom", "whose", "when", "where", "while", "because", "although", "though", "so", "if", "as", "than", "until" };

const alphabet: []const u32 = blk: {
    @setEvalBranchQuota(10000);
    var out: []const u32 = &.{};
    const view = std.unicode.Utf8View.initComptime(ascii);
    var it = view.iterator();
    while (it.nextCodepoint()) |c| out = out ++ [_]u32{c};
    break :blk out ++ [_]u32{ ellipsis, em_dash };
};

/// BOS then one id a character; characters outside the alphabet are 2.
pub fn encode(a: std.mem.Allocator, piece: []const u32) ![]u32 {
    const ids = try a.alloc(u32, piece.len + 1);
    ids[0] = 1;
    for (piece, ids[1..]) |c, *id| id.* = for (alphabet, 0..) |x, i| {
        if (x == c) break @intCast(i + 3);
    } else 2;
    return ids;
}

fn space(c: u32) bool {
    return switch (c) {
        ' ', '\t', '\n', '\r', 0x0b, 0x0c, 0x1c...0x1f, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn word(c: u32) bool {
    return c == '_' or (c < 128 and std.ascii.isAlphanumeric(@intCast(c))) or (c >= 128 and !space(c) and c != ellipsis and c != em_dash);
}

fn strip(s: []const u32) []const u32 {
    var lo: usize = 0;
    var hi = s.len;
    while (lo < hi and space(s[lo])) lo += 1;
    while (hi > lo and space(s[hi - 1])) hi -= 1;
    return s[lo..hi];
}

/// Pieces of `text`, owned by `a`: each sentence, and a sentence over clause_max cut at clauses merged greedily, then
/// once before a conjunction near the middle of a piece still too long.
pub fn split(a: std.mem.Allocator, text: []const u32) ![]const []const u32 {
    var out: std.ArrayList([]const u32) = .empty;
    const t = strip(text);
    var start: usize = 0;
    var i: usize = 0;
    while (i <= t.len) {
        const cut = i == t.len or (i > 0 and space(t[i]) and switch (t[i - 1]) {
            '.', '!', '?', ellipsis => true,
            else => false,
        });
        if (!cut) {
            i += 1;
            continue;
        }
        if (i > start) try sentence(a, &out, t[start..i]);
        while (i < t.len and space(t[i])) i += 1;
        start = i;
        if (i == t.len) break;
    }
    return out.items;
}

fn sentence(a: std.mem.Allocator, out: *std.ArrayList([]const u32), s: []const u32) !void {
    if (s.len <= clause_max) return out.append(a, s);
    var merged: std.ArrayList(u32) = .empty;
    var start: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        const after_dash = c == em_dash or (c == '-' and i > start and s[i - 1] == '-');
        const after_clause = (c == ',' or c == ';' or c == ':') and i + 1 < s.len and space(s[i + 1]);
        if (!after_dash and !after_clause) continue;
        var next = i + 1;
        while (next < s.len and space(s[next])) next += 1;
        try join(a, out, &merged, s[start .. i + 1]);
        start = next;
        i = next - 1;
    }
    if (start < s.len) try join(a, out, &merged, s[start..]);
    if (merged.items.len > 0) try conjunction(a, out, merged.items);
}

/// The greedy clause merge: a clause starts a new piece when the current one cannot take it within clause_max.
fn join(a: std.mem.Allocator, out: *std.ArrayList([]const u32), cur: *std.ArrayList(u32), clause: []const u32) !void {
    if (clause.len == 0) return;
    if (cur.items.len > 0 and cur.items.len + 1 + clause.len > clause_max) {
        try conjunction(a, out, try cur.toOwnedSlice(a));
    } else if (cur.items.len > 0) {
        const last = cur.items[cur.items.len - 1];
        const dash = last == em_dash or (last == '-' and cur.items.len > 1 and cur.items[cur.items.len - 2] == '-');
        if (!dash) try cur.append(a, ' ');
    }
    try cur.appendSlice(a, clause);
}

fn conjunction(a: std.mem.Allocator, out: *std.ArrayList([]const u32), s: []const u32) !void {
    if (s.len <= clause_max) return out.append(a, s);
    var best: ?usize = null;
    for (0..s.len) |c| {
        if (!space(s[c]) or c < min_piece or s.len - c < min_piece or !conjunctionAt(s, c + 1)) continue;
        // |c - len/2|, first wins a tie, as Python's min
        const d = @abs(@as(f64, @floatFromInt(c)) - @as(f64, @floatFromInt(s.len)) / 2);
        if (best == null or d < @abs(@as(f64, @floatFromInt(best.?)) - @as(f64, @floatFromInt(s.len)) / 2)) best = c;
    }
    const c = best orelse return out.append(a, s);
    try conjunction(a, out, strip(s[0..c]));
    try conjunction(a, out, strip(s[c..]));
}

fn conjunctionAt(s: []const u32, at: usize) bool {
    if (at > 0 and word(s[at - 1])) return false;
    for (conjunctions) |w| {
        if (s.len - at < w.len) continue;
        const end = at + w.len;
        for (w, s[at..end]) |x, y| {
            if (x != y) break;
        } else if (end == s.len or !word(s[end])) return true;
    }
    return false;
}

/// `pieces`, each one over `max` characters cut at its last space within `max`, else at `max`: a run the splitter
/// cannot cut still fits one piece's frames.
pub fn fit(a: std.mem.Allocator, pieces: []const []const u32, max: usize) ![]const []const u32 {
    var out: std.ArrayList([]const u32) = .empty;
    for (pieces) |piece| {
        var rest = piece;
        while (rest.len > max) {
            var cut = max;
            while (cut > 0 and !space(rest[cut])) cut -= 1;
            if (cut == 0) cut = max;
            try out.append(a, strip(rest[0..cut]));
            rest = strip(rest[cut..]);
        }
        if (rest.len > 0) try out.append(a, rest);
    }
    return out.items;
}

/// The longest run, in seconds, of 40 ms frames under -50 dBFS between the first and last voiced frame.
pub fn stall(pcm: []const f32, sample_rate: u32) f64 {
    const hop = 960;
    const n = pcm.len / hop;
    if (n < 3) return 0;
    var first: ?usize = null;
    var last: usize = 0;
    var best: usize = 0;
    var run: usize = 0;
    for (0..n) |f| {
        var sum: f64 = 0;
        for (pcm[f * hop ..][0..hop]) |x| sum += @as(f64, x) * x;
        const quiet = 20 * std.math.log10(@sqrt(sum / hop) + 1e-9) < -50.0;
        if (quiet) {
            if (first != null) run += 1;
            continue;
        }
        if (first == null) first = f else best = @max(best, run);
        last = f;
        run = 0;
    }
    if (first == null or first.? == last) return 0;
    return @as(f64, @floatFromInt(best * hop)) / @as(f64, @floatFromInt(sample_rate));
}

fn codepoints(a: std.mem.Allocator, s: []const u8) ![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = (try std.unicode.Utf8View.init(s)).iterator();
    while (it.nextCodepoint()) |c| try out.append(a, c);
    return out.items;
}

fn expectPieces(text: []const u8, want: []const []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try split(a, try codepoints(a, text));
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualSlices(u32, try codepoints(a, w), g);
}

test "split: sentences, clauses merged under 120, then a conjunction near the middle" {
    try expectPieces("  Good morning. The kettle is on!  Is it?", &.{ "Good morning.", "The kettle is on!", "Is it?" });
    try expectPieces("Wait… what? 3.5 stays", &.{ "Wait…", "what?", "3.5 stays" });
    const long = "The committee, which had met every Tuesday since the spring of 2019, finally agreed that the budget would need a second review before anyone could sign it, and that the review should be public.";
    try expectPieces(long, &.{ "The committee, which had met every Tuesday since the spring of 2019,", "finally agreed that the budget would need a second review before anyone could sign it,", "and that the review should be public." });
    const dashes = "She packed the car with every box she owned and every lamp she had—then she drove north for nine hours and never once looked back at the town";
    try expectPieces(dashes, &.{ "She packed the car with every box she owned and every lamp she had—", "then she drove north for nine hours and never once looked back at the town" });
    const as: [50]u8 = @splat('a');
    const bs: [50]u8 = @splat('b');
    try expectPieces(as ++ " the cat sat and the dog ran " ++ bs, &.{ as ++ " the cat sat", "and the dog ran " ++ bs });
}

test "encode: BOS, alphabet ids, unknown 2" {
    const ids = try encode(std.testing.allocator, &.{ ' ', 'a', '—', 0x4e2d });
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 4, 94, 2 }, ids);
}

test "fit: a piece over the limit cut at its last space within it, else at the limit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try fit(a, &.{ try codepoints(a, "ab cd efg"), try codepoints(a, "abcdefg"), try codepoints(a, "abcd") }, 4);
    const want = [_][]const u8{ "ab", "cd", "efg", "abcd", "efg", "abcd" };
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualSlices(u32, try codepoints(a, w), g);
}

test "stall: the longest quiet run between voiced frames" {
    var pcm: [960 * 10]f32 = @splat(0);
    for ([_]usize{ 1, 2, 6, 7 }) |f| @memset(pcm[f * 960 ..][0..960], 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.12), stall(&pcm, 24000), 1e-12);
    try std.testing.expectEqual(@as(f64, 0), stall(pcm[0 .. 960 * 2], 24000));
}
