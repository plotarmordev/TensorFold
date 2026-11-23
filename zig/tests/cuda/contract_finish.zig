//! The end of tf-cuda-test kernel-contract: the standalone suites and gate 5, once the registered fixtures are gone.
const std = @import("std");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const live = @import("contract_limits.zig").live;

/// `gates`: how the registered kernels' gates ended. An error there or in a suite skips what follows it, and
/// SUMMARY still prints with what was counted, marked as stopped early. The suites and the offline audit, which
/// are not divided, run on a full run's first shard.
pub fn run(gpu: check.Gpu, only: []const []const u8, shard: g.Shard, t: g.Tally, t0: i96, gates: anyerror!void) !void {
    var t1 = check.now(gpu.io);
    var lim: usize = 0;
    const whole = only.len == 0 and shard.first();
    const ended: anyerror!void = if (gates) |_| suites(gpu, only, shard, whole, &t1, &lim) else |e| e;
    const t2 = check.now(gpu.io);
    var part: [32]u8 = undefined;
    std.debug.print("{s}{s}{s}SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ if (only.len > 0) "(filtered) " else "", if (shard.of > 1) try std.fmt.bufPrint(&part, "(shard {d}/{d}) ", .{ shard.at + 1, shard.of }) else "", if (ended) |_| "" else |_| "(stopped early) ", t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, @divTrunc(t1 - t0, std.time.ns_per_ms), @divTrunc(t2 - t1, std.time.ns_per_ms) });
    try ended;
    try check.expect(t.problems() + lim == 0, "kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}

fn suites(gpu: check.Gpu, only: []const []const u8, shard: g.Shard, whole: bool, t1: *i96, lim: *usize) !void {
    if (whole) try @import("contract_back_suites.zig").run(gpu);
    t1.* = check.now(gpu.io);
    if (whole) try @import("contract_after.zig").suites(gpu);
    lim.* = try live(gpu, only, shard);
}
