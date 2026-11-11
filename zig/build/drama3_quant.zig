//! tf-drama-quant, Drama 3's CPU checkpoint converter, and its tests: host only on every OS, no model, no GPU.
const std = @import("std");

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    const quant = b.createModule(.{
        .root_source_file = b.path("zig/src/families/drama3/quant.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    });
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = quant }));
    b.step("test-drama3-quant", "tf-drama-quant on a synthetic FP8 checkpoint: NVFP4 bytes, compact maps, provenance").dependOn(&tests.step);
    test_step.dependOn(&tests.step);
    // the provenance file names the commit the converter was built from
    const commit = b.addSystemCommand(&.{ "git", "describe", "--always", "--dirty", "--abbrev=40", "--exclude=*" });
    commit.setCwd(b.path("."));
    commit.has_side_effects = true;
    const tool = b.createModule(.{
        .root_source_file = b.path("zig/src/families/drama3/quant_tool.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "drama3_quant", .module = quant }},
    });
    tool.addAnonymousImport("tool_commit", .{ .root_source_file = commit.captureStdOut(.{}) });
    const exe = b.addExecutable(.{ .name = "tf-drama-quant", .root_module = tool });
    b.installArtifact(exe);
    b.step("drama3-quant", "Build tf-drama-quant, Drama 3's FP8 routed experts to NVFP4 (and --compact-experts) on the CPU").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
