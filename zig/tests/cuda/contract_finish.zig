//! The end of tf-cuda-test kernel-contract: the standalone suites and gate 5, once the registered fixtures are gone.
const std = @import("std");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const live = @import("contract_limits.zig").live;

pub fn run(gpu: check.Gpu, only: []const []const u8, t: g.Tally, t0: i96) !void {
    const full = !@import("contract_after.zig").filtered(only);
    if (full) try @import("contract_back_suites.zig").run(gpu);
    const t1 = check.now(gpu.io);
    if (full) try @import("contract_after.zig").suites(gpu);
    const lim = try live(gpu, only);
    const t2 = check.now(gpu.io);
    std.debug.print("{s}SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ if (only.len > 0) "(filtered) " else "", t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, @divTrunc(t1 - t0, std.time.ns_per_ms), @divTrunc(t2 - t1, std.time.ns_per_ms) });
    try check.expect(t.problems() + lim == 0, "kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}

/// One part of a full run's tail on its own (kernel-contract-suites): `back` (the gradient suites), `after` (the
/// served to SwiGLU pair suites) or `limits` (gate 5, every registered launch); a part fails as the full run would.
pub fn suites(gpu: check.Gpu, part: []const u8) !void {
    const t0 = check.now(gpu.io);
    if (std.mem.eql(u8, part, "back")) {
        try @import("contract_back_suites.zig").run(gpu);
    } else if (std.mem.eql(u8, part, "after")) {
        try @import("contract_after.zig").suites(gpu);
    } else if (std.mem.eql(u8, part, "limits")) {
        const lim = try live(gpu, &.{});
        try check.expect(lim == 0, "kernel-contract-suites limits: {d} limit problems", .{lim});
    } else return error.UnknownSuitesPart;
    std.debug.print("SUMMARY kernel-contract-suites {s}: pass in {d} ms\n", .{ part, @divTrunc(check.now(gpu.io) - t0, std.time.ns_per_ms) });
}
