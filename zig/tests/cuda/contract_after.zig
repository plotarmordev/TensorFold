//! kernel-contract's words (`only`): the kernels so named, and the suites that run after the gates on a full run.
const std = @import("std");
const g = @import("core").contract.gates;
const check = @import("check.zig");

/// Whether `only` makes a partial run: a plain word names kernels, a `%I/N` takes a shard; `-word`s and `@word`s keep it full.
pub fn filtered(only: []const []const u8) bool {
    for (only) |w| if (!std.mem.startsWith(u8, w, "-") and !std.mem.startsWith(u8, w, "@")) return true;
    return false;
}

/// Whether a plain word (not `-`, `@` or `%`) names the kernels to keep.
fn named(only: []const []const u8) bool {
    for (only) |w| if (mark(w) == 0) return true;
    return false;
}

/// A word's leading `-`, `@` or `%`, else 0.
fn mark(w: []const u8) u8 {
    return if (w.len > 0 and std.mem.indexOfScalar(u8, "-@%", w[0]) != null) w[0] else 0;
}

/// Whether `name` holds no `-word` and, when plain words are given, one of them.
pub fn wanted(name: []const u8, only: []const []const u8) bool {
    var hit_named = !named(only);
    for (only) |w| {
        if (mark(w) == '@' or mark(w) == '%') continue;
        const minus = mark(w) == '-';
        const hit = std.mem.indexOf(u8, name, if (minus) w[1..] else w) != null;
        if (minus and hit) return false;
        hit_named = hit_named or hit;
    }
    return hit_named;
}

/// Keeps the kernels `wanted` keeps, from the first whose name holds an `@word` on (a run resumed there), and of those
/// with `%I/N` every Nth from the Ith (a shard: the N shards cover them once); empty keeps every one.
pub fn keep(kernels: *std.ArrayList(g.Kernel), only: []const []const u8) !void {
    var from: usize = 0;
    var shard: [2]usize = .{ 0, 1 };
    for (only) |w| switch (mark(w)) {
        '@' => from = for (kernels.items, 0..) |k, i| {
            if (std.mem.indexOf(u8, k.spec.name, w[1..]) != null) break i;
        } else kernels.items.len,
        '%' => shard = try parseShard(w[1..]),
        else => {},
    };
    var kept: usize = 0;
    for (kernels.items[from..], 0..) |k, at| if (at % shard[1] == shard[0] and wanted(k.spec.name, only)) {
        kernels.items[kept] = k;
        kept += 1;
    };
    kernels.shrinkRetainingCapacity(kept);
}

fn parseShard(s: []const u8) ![2]usize {
    const slash = std.mem.indexOfScalar(u8, s, '/') orelse return error.BadShard;
    const i = try std.fmt.parseInt(usize, s[0..slash], 10);
    const n = try std.fmt.parseInt(usize, s[slash + 1 ..], 10);
    if (i >= n) return error.BadShard;
    return .{ i, n };
}

/// The served, paged, pending, learning, lane and SwiGLU pair suites of a full run.
pub fn suites(gpu: check.Gpu) !void {
    try @import("contract_qwen_served.zig").run(gpu);
    try @import("contract_qwen_pages.zig").run(gpu);
    try @import("contract_qwen_shared_prefix.zig").run(gpu, &.{});
    try @import("contract_qwen_pending.zig").run(gpu);
    try @import("contract_qwen_learning.zig").numerical(gpu);
    try @import("contract_fn_lanes.zig").run(gpu);
    try @import("contract_prompt_swiglu.zig").run(gpu);
}
