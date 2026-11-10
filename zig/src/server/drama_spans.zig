//! Indivisible speech text: special tokens, balanced brackets and correctly nested paired tags.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Span = struct { start: usize, end: usize };
const Tag = struct { start: usize, end: usize, name: []const u8, closing: bool };

/// Sorted, disjoint byte ranges; adjacent controls remain separate atoms and all bytes stay in the caller's input.
pub fn collect(a: Allocator, input: []const u8) Allocator.Error![]Span {
    var atoms: std.ArrayList(Span) = .empty;
    defer atoms.deinit(a);
    var brackets: std.ArrayList(usize) = .empty;
    defer brackets.deinit(a);
    var i: usize = 0;
    while (i < input.len) {
        if (specialEnd(input, i)) |end| {
            try atoms.append(a, .{ .start = i, .end = end });
            i = end;
            continue;
        }
        if (input[i] == '[') try brackets.append(a, i) else if (input[i] == ']') {
            if (brackets.pop()) |start| try atoms.append(a, .{ .start = start, .end = i + 1 });
        }
        i += 1;
    }
    merge(&atoms);

    var spans: std.ArrayList(Span) = .empty;
    errdefer spans.deinit(a);
    try spans.appendSlice(a, atoms.items);
    var tags: std.ArrayList(Tag) = .empty;
    defer tags.deinit(a);
    var atom: usize = 0;
    i = 0;
    while (i < input.len) {
        if (atom < atoms.items.len and i == atoms.items[atom].start) {
            i = atoms.items[atom].end;
            atom += 1;
            continue;
        }
        if (tagAt(input, i)) |tag| {
            if (!tag.closing) {
                try tags.append(a, tag);
            } else if (tags.pop()) |open| {
                if (std.mem.eql(u8, open.name, tag.name)) {
                    try spans.append(a, .{ .start = open.start, .end = tag.end });
                } else tags.clearRetainingCapacity();
            }
            i = tag.end;
        } else i += 1;
    }
    merge(&spans);
    return spans.toOwnedSlice(a);
}

fn specialEnd(input: []const u8, start: usize) ?usize {
    if (!std.mem.startsWith(u8, input[start..], "<|")) return null;
    var i = start + 2;
    while (i < input.len and input[i] != '|' and input[i] != '>') : (i += 1) {
        if (input[i] == '<') return null;
    }
    return if (std.mem.startsWith(u8, input[i..], "|>")) i + 2 else null;
}

fn tagAt(input: []const u8, start: usize) ?Tag {
    if (input[start] != '<') return null;
    var i = start + 1;
    const closing = i < input.len and input[i] == '/';
    if (closing) i += 1;
    const first = i;
    if (i == input.len or !std.ascii.isAlphabetic(input[i])) return null;
    i += 1;
    while (i < input.len and (std.ascii.isAlphanumeric(input[i]) or input[i] == '_' or input[i] == '-')) i += 1;
    if (i == input.len or input[i] != '>') return null;
    return .{ .start = start, .end = i + 1, .name = input[first..i], .closing = closing };
}

fn before(_: void, left: Span, right: Span) bool {
    return left.start < right.start or (left.start == right.start and left.end > right.end);
}

fn merge(spans: *std.ArrayList(Span)) void {
    std.mem.sort(Span, spans.items, {}, before);
    var n: usize = 0;
    for (spans.items) |span| {
        if (n > 0 and span.start < spans.items[n - 1].end) {
            spans.items[n - 1].end = @max(spans.items[n - 1].end, span.end);
        } else {
            spans.items[n] = span;
            n += 1;
        }
    }
    spans.shrinkRetainingCapacity(n);
}

fn expectSpans(input: []const u8, want: []const []const u8) !void {
    const spans = try collect(std.testing.allocator, input);
    defer std.testing.allocator.free(spans);
    try std.testing.expectEqual(want.len, spans.len);
    for (spans, want) |span, expected| try std.testing.expectEqualStrings(expected, input[span.start..span.end]);
    for (spans, 0..) |span, i| {
        try std.testing.expect(span.start < span.end and span.end <= input.len);
        if (i > 0) try std.testing.expect(spans[i - 1].end <= span.start);
    }
}

test "special tokens and adjacent or nested bracketed cues are indivisible" {
    try expectSpans("A <|speaker:0|> [soft laugh][pause] B", &.{ "<|speaker:0|>", "[soft laugh]", "[pause]" });
    try expectSpans("[outer [inner] tail] [next]", &.{ "[outer [inner] tail]", "[next]" });
    try expectSpans("[<|speaker:0|> [pause]]", &.{"[<|speaker:0|> [pause]]"});
    try expectSpans("<|a|b|> <|x> <|unterminated", &.{});
}

test "paired tags nest by exact name and ignore tag-shaped text inside bracketed cues" {
    const nested = "<whisper>One. <soft>Two.</soft> <whisper>Three.</whisper></whisper>";
    try expectSpans(nested ++ " <slow>Four.</slow>", &.{ nested, "<slow>Four.</slow>" });
    try expectSpans("<whisper>[literal </other>] Quiet.</whisper>", &.{"<whisper>[literal </other>] Quiet.</whisper>"});
    try expectSpans("<tone-1>\u{4f60}\u{597d}\u{3002}</tone-1>", &.{"<tone-1>\u{4f60}\u{597d}\u{3002}</tone-1>"});
}

test "unclosed and crossed controls do not hide subsequent valid controls" {
    try expectSpans("[unclosed [pause] <soft>Hi.</soft>", &.{ "[pause]", "<soft>Hi.</soft>" });
    try expectSpans("<unclosed><soft>Hi.</soft>", &.{"<soft>Hi.</soft>"});
    try expectSpans("<soft><whisper>Hi.</soft> Bye.</whisper> <slow>End.</slow>", &.{"<slow>End.</slow>"});
    try expectSpans("<soft>Hi.</SOFT> <soft >Hi.</soft> <1>Hi.</1>", &.{});
    try expectSpans("<|unclosed <soft>Hi.</soft>", &.{"<soft>Hi.</soft>"});
    try expectSpans("2 < 3; a > b; [] [unfinished", &.{"[]"});
}
