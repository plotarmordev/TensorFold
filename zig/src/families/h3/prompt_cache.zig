//! MiniMax-H3-Image's kept prompt states: a prompt's presentation and refined text rows held by the
//! resident prompt server across pictures, least recently used inside a byte budget. The key is the
//! prompt's exact bytes as tokenised; the geometry, marks and tokenizer are the process's own, checked
//! at load, so the bytes alone decide. H3-Image has no negative prompt or CFG, so nothing else varies.
const std = @import("std");

/// Bytes of kept states without H3_PROMPT_CACHE: a few full-length prompts' worth.
pub const default_bytes: usize = 256 << 20;

/// One prompt's states: its rows' ids and tags and the refiner's text rows for them.
const Entry = struct {
    prompt: []u8,
    ids: []u32,
    tags: []i32,
    text: []u16,
    /// What the entry's slices hold, counted against the budget.
    bytes: usize,

    fn free(e: *Entry, a: std.mem.Allocator) void {
        a.free(e.prompt);
        a.free(e.ids);
        a.free(e.tags);
        a.free(e.text);
        a.destroy(e);
    }
};

/// The kept states of a `find` hit: the presentation's ids and tags and the refiner's text rows.
pub const States = struct { ids: []const u32, tags: []const i32, text: []const u16 };

pub const Cache = struct {
    gpa: std.mem.Allocator,
    /// Oldest first; a `find` or `keep` makes its entry the last.
    entries: std.ArrayList(*Entry) = .empty,
    /// `entries`' bytes held against `budget`.
    held: usize = 0,
    budget: usize,
    hits: u64 = 0,
    misses: u64 = 0,

    /// MiB from H3_PROMPT_CACHE, or `default_bytes` unset; 0 keeps nothing, so the cache is off.
    pub fn budgetOf(value: ?[]const u8) usize {
        const v = value orelse return default_bytes;
        const mib = std.fmt.parseInt(usize, v, 10) catch return default_bytes;
        return mib << 20;
    }

    pub fn init(gpa: std.mem.Allocator, budget: usize) Cache {
        return .{ .gpa = gpa, .budget = budget };
    }

    pub fn deinit(c: *Cache) void {
        for (c.entries.items) |e| e.free(c.gpa);
        c.entries.deinit(c.gpa);
    }

    /// The prompt's kept states, its entry now the newest; null on a miss.
    pub fn find(c: *Cache, prompt: []const u8) ?States {
        const i = c.indexOf(prompt) orelse {
            c.misses += 1;
            return null;
        };
        c.hits += 1;
        const e = c.entries.items[i];
        const last = c.entries.items.len - 1;
        if (i < last) {
            std.mem.copyForwards(*Entry, c.entries.items[i..last], c.entries.items[i + 1 .. last + 1]);
            c.entries.items[last] = e;
        }
        return .{ .ids = e.ids, .tags = e.tags, .text = e.text };
    }

    /// The prompt's states kept, its entry newest, least-recently-used evicted until it fits.
    /// An entry over the budget keeps nothing and evicts nothing; neither does an allocation
    /// failure — a picture never fails for the cache.
    pub fn keep(c: *Cache, prompt: []const u8, ids: []const u32, tags: []const i32, text: []const u16) void {
        const bytes = prompt.len + 4 * (ids.len + tags.len) + 2 * text.len;
        if (bytes > c.budget or c.indexOf(prompt) != null) return;
        const e = c.gpa.create(Entry) catch return;
        e.* = .{ .prompt = &.{}, .ids = &.{}, .tags = &.{}, .text = &.{}, .bytes = bytes };
        e.prompt = c.gpa.dupe(u8, prompt) catch return e.free(c.gpa);
        e.ids = c.gpa.dupe(u32, ids) catch return e.free(c.gpa);
        e.tags = c.gpa.dupe(i32, tags) catch return e.free(c.gpa);
        e.text = c.gpa.dupe(u16, text) catch return e.free(c.gpa);
        // the copies are owned: only now may older entries leave for the room they need
        c.held += bytes;
        c.entries.append(c.gpa, e) catch return {
            c.held -= bytes;
            return e.free(c.gpa);
        };
        while (c.held > c.budget) c.dropOldest();
    }

    fn indexOf(c: *const Cache, prompt: []const u8) ?usize {
        for (c.entries.items, 0..) |e, i| if (std.mem.eql(u8, e.prompt, prompt)) return i;
        return null;
    }

    fn dropOldest(c: *Cache) void {
        const e = c.entries.orderedRemove(0);
        c.held -= e.bytes;
        e.free(c.gpa);
    }
};
