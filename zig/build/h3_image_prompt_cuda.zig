//! MiniMax-H3-Image's prompt path on CUDA: tf-h3-image-prompt-cuda, its kernel image built by `-Dnvcc`
//! or compile-only without it; host tests cover condition-projection algorithm selection.
const std = @import("std");

/// `zig build tf-h3-image-prompt-cuda`: a prompt to the refiner's text rows on CUDA, and its synthetic
/// selftest against the float64 host reference.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, opts: @import("options.zig").Cuda, test_step: *std.Build.Step) void {
    const image = if (opts.nvcc) |nvcc|
        @import("cuda_fatbin.zig").fatbin(b, nvcc, b.addSystemCommand(&.{ nvcc, "--version" }).captureStdOut(.{}), .{ .name = "h3_image_prompt", .flags = &.{"-O3"}, .min_sm = 80 }, opts.sms)
    else
        b.addWriteFiles().add("h3-image-prompt-compile-only.fatbin", "");
    const cuda = @import("cuda.zig").runtime(b, target, optimize, &.{});
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = optimize });
    const fixture = b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_fixture.zig"), .target = target, .optimize = optimize });
    const rope = @import("h3_image_prompt.zig").rope(b, target);
    const prompt = b.createModule(.{
        .root_source_file = b.path("zig/src/families/h3/prompt_cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "safetensors", .module = st },
            .{ .name = "qwen_vl_rope", .module = rope },
            .{ .name = "checkpoint_fixture", .module = fixture },
        },
    });
    prompt.addAnonymousImport("fatbin_h3_image_prompt", .{ .root_source_file = image });
    const exe = b.addExecutable(.{ .name = "tf-h3-image-prompt-cuda", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/h3_image_prompt_cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "h3_prompt_cuda", .module = prompt },
            .{ .name = "tokenizer", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }) },
            .{ .name = "checkpoint_fixture", .module = fixture },
            .{ .name = "moss_noise", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/noise.zig"), .target = target, .optimize = optimize }) },
        },
    }) });
    b.step("tf-h3-image-prompt-cuda", "MiniMax-H3-Image's prompt to its text rows on CUDA: Qwen3-VL to layer 50, the token refiner").dependOn(&b.addInstallArtifact(exe, .{}).step);
    test_step.dependOn(&exe.step);
    const tests = b.addTest(.{ .root_module = prompt });
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
