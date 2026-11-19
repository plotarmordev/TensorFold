//! The prompt cache's contract: keyed by the prompt's exact bytes, states returned verbatim,
//! least recently used evicted inside the budget, nothing kept or lost past it.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const testing = std.testing;

/// ids, tags and text rows for `n` rows, their values the prompt's own.
fn states(a: std.mem.Allocator, n: usize) !struct { []u32, []i32, []u16 } {
    const ids = try a.alloc(u32, n);
    const tags = try a.alloc(i32, n);
    const text = try a.alloc(u16, n * 3);
    for (ids, tags, 0..) |*id, *t, i| {
        id.* = @intCast(1000 + i);
        t.* = 1;
    }
    for (text, 0..) |*v, i| v.* = @intCast(i);
    return .{ ids, tags, text };
}

test "h3 prompt cache: a kept prompt's states come back verbatim, another prompt misses" {
    const a = testing.allocator;
    var c = pc.Cache.init(a, 1 << 20);
    defer c.deinit();
    const s = try states(a, 4);
    defer a.free(s[0]);
    defer a.free(s[1]);
    defer a.free(s[2]);

    try testing.expectEqual(@as(?pc.States, null), c.find("a red fox"));
    c.keep("a red fox", s[0], s[1], s[2]);
    const hit = c.find("a red fox").?;
    try testing.expectEqualSlices(u32, s[0], hit.ids);
    try testing.expectEqualSlices(i32, s[1], hit.tags);
    try testing.expectEqualSlices(u16, s[2], hit.text);
    try testing.expect(c.find("a different prompt") == null);
    try testing.expectEqual(@as(u64, 1), c.hits);
    try testing.expectEqual(@as(u64, 2), c.misses);
}

test "h3 prompt cache: a found entry is newest, the least recently used evicted inside the budget" {
    const a = testing.allocator;
    const s = try states(a, 4);
    defer a.free(s[0]);
    defer a.free(s[1]);
    defer a.free(s[2]);
    // every key eight bytes, so each entry is the same size and two exactly fill the budget
    const entry_bytes = "prompt-1".len + 4 * (4 + 4) + 2 * 12;

    var c = pc.Cache.init(a, 2 * entry_bytes);
    defer c.deinit();
    c.keep("prompt-1", s[0], s[1], s[2]);
    c.keep("prompt-2", s[0], s[1], s[2]);
    try testing.expectEqual(@as(usize, 2 * entry_bytes), c.held);
    // finding "prompt-1" makes it newest, so keeping the third evicts "prompt-2"
    try testing.expect(c.find("prompt-1") != null);
    c.keep("prompt-3", s[0], s[1], s[2]);
    try testing.expect(c.find("prompt-2") == null);
    try testing.expect(c.find("prompt-1") != null);
    try testing.expect(c.find("prompt-3") != null);
    try testing.expectEqual(@as(usize, 2 * entry_bytes), c.held);
    try testing.expectEqual(@as(usize, 2), c.entries.items.len);
}

test "h3 prompt cache: an entry over the budget keeps nothing, a zero budget keeps nothing" {
    const a = testing.allocator;
    const s = try states(a, 4);
    defer a.free(s[0]);
    defer a.free(s[1]);
    defer a.free(s[2]);
    const entry_bytes = "prompt".len + 4 * (4 + 4) + 2 * 12;

    var c = pc.Cache.init(a, entry_bytes - 1);
    defer c.deinit();
    c.keep("prompt", s[0], s[1], s[2]);
    try testing.expectEqual(@as(usize, 0), c.held);
    try testing.expect(c.find("prompt") == null);

    var off = pc.Cache.init(a, 0);
    defer off.deinit();
    off.keep("prompt", s[0], s[1], s[2]);
    try testing.expect(off.find("prompt") == null);
}

test "h3 prompt cache: keeping a prompt already kept changes nothing" {
    const a = testing.allocator;
    const s = try states(a, 4);
    defer a.free(s[0]);
    defer a.free(s[1]);
    defer a.free(s[2]);

    var c = pc.Cache.init(a, 1 << 20);
    defer c.deinit();
    c.keep("prompt", s[0], s[1], s[2]);
    const held = c.held;
    c.keep("prompt", s[0], s[1], s[2]);
    try testing.expectEqual(held, c.held);
    try testing.expectEqual(@as(usize, 1), c.entries.items.len);
    try testing.expectEqual(@as(u64, 0), c.hits);
}

test "h3 prompt cache: H3_PROMPT_CACHE's MiB, default and off" {
    try testing.expectEqual(pc.default_bytes, pc.Cache.budgetOf(null));
    try testing.expectEqual(pc.default_bytes, pc.Cache.budgetOf("not a number"));
    try testing.expectEqual(@as(usize, 0), pc.Cache.budgetOf("0"));
    try testing.expectEqual(@as(usize, 64 << 20), pc.Cache.budgetOf("64"));
}
