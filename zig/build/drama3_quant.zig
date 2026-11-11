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
    // the provenance file names the commit the converter was built from, when the tree is a git checkout
    b.graph.poisonCache(); // a commit or an edit changes the answer
    const git = b.runFallible(&.{ "git", "-C", b.root.toString(b.allocator) catch @panic("OOM"), "describe", "--always", "--dirty", "--abbrev=40", "--exclude=*" }, .{ .stderr_behavior = .ignore });
    const build_options = b.addOptions();
    build_options.addOption(?[]const u8, "commit", if (git == .success) std.mem.trimEnd(u8, git.success, "\r\n") else null);
    const tool = b.createModule(.{
        .root_source_file = b.path("zig/src/families/drama3/quant_tool.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "drama3_quant", .module = quant }},
    });
    tool.addOptions("build_options", build_options);
    const exe = b.addExecutable(.{ .name = "tf-drama-quant", .root_module = tool });
    b.installArtifact(exe);
    b.step("drama3-quant", "Build tf-drama-quant, Drama 3's FP8 routed experts to NVFP4 (and --compact-experts) on the CPU").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
