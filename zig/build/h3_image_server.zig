//! `zig build tf-h3-image-server` (metal_targets.zig): MiniMax-H3-Image's three tools resident in one process.
const std = @import("std");
const prompt = @import("h3_image_prompt.zig");

pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, h3_kernels: *std.Build.Module, sources: *std.Build.Module, test_step: *std.Build.Step) void {
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = optimize });
    // core's picture files have one owner: the PNG writer, which the prompt path's Qwen copy reads them through
    const writer = b.createModule(.{ .root_source_file = b.path("zig/src/picture_writer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    // one fixture module for the image module's prompt case writer and the tool's edit entries
    const fixture = b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_fixture.zig"), .target = target, .optimize = optimize });
    // one module over the transformer, prompt path and decoder: they share source files
    const image = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "metal", .module = metal },
        .{ .name = "safetensors", .module = st },
        .{ .name = "h3_kernels", .module = h3_kernels },
        .{ .name = "kernel_sources", .module = sources },
        .{ .name = "h3_prompt_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/prompt_kernels.zig") }) },
        .{ .name = "h3_image_decoder_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/image_decoder.zig") }) },
        .{ .name = "qwen_vl_rope", .module = prompt.rope(b, target) },
        .{ .name = "checkpoint_fixture", .module = fixture },
        .{ .name = "picture", .module = writer },
        .{ .name = "picture_writer", .module = writer },
        .{ .name = "gpu_turn", .module = b.createModule(.{ .root_source_file = b.path("zig/src/server/gpu_turn.zig"), .target = target, .optimize = optimize }) },
    } });
    const exe = b.addExecutable(.{ .name = "tf-h3-image-server", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_server.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "metal", .module = metal },
        .{ .name = "h3", .module = image },
        .{ .name = "h3_prompt", .module = image },
        .{ .name = "h3_image_decoder", .module = image },
        .{ .name = "safetensors", .module = st },
        .{ .name = "tokenizer", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }) },
        .{ .name = "checkpoint_fixture", .module = fixture },
        .{ .name = "moss_noise", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/noise.zig"), .target = target, .optimize = optimize }) },
        .{ .name = "picture_writer", .module = writer },
    } }) });
    b.step("tf-h3-image-server", "MiniMax-H3-Image resident: prompt encoder, transformer and decoder loaded once, one picture a stdin line").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const tests = b.addTest(.{ .root_module = exe.root_module });
    const run = b.addRunArtifact(tests);
    b.step("test-h3-image-server", "The resident server's binding reuse, failed-bind unwind and over-long line reader").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
