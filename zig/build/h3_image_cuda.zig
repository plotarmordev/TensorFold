//! MiniMax-H3-Image's free run on the CUDA blocks: tf-h3-image-cuda, with H3's kernel image built by `-Dnvcc` (on the
//! Spark) or compile-only without it; and the host row-kernel ports' checks.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, opts: @import("options.zig").Cuda, test_step: *std.Build.Step) void {
    const image = if (opts.nvcc) |nvcc| @import("cuda_fatbin.zig").fatbin(b, nvcc, b.addSystemCommand(&.{ nvcc, "--version" }).captureStdOut(.{}), .{ .name = "h3", .flags = &.{"-O3"} }, opts.sms) else b.addWriteFiles().add("h3-compile-only.fatbin", "");
    const cuda = @import("cuda.zig").runtime(b, target, optimize, &.{});
    const st = @import("h3_host.zig").safetensors(b, target);
    const h3 = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "safetensors", .module = st } } });
    h3.addAnonymousImport("fatbin_h3", .{ .root_source_file = image });
    // one module for the host side's shared geometry: the schedule, the picture and the row ports
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/image_host.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "safetensors", .module = st }} });
    const exe = b.addExecutable(.{ .name = "tf-h3-image-cuda", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "h3cuda",      .module = h3 },
        .{ .name = "h3image",     .module = host },
        .{ .name = "safetensors", .module = st },
    } }) });
    b.step("tf-h3-image-cuda", "MiniMax-H3-Image's free run: a prompt case to a latent on the CUDA transformer blocks").dependOn(&b.addInstallArtifact(exe, .{}).step);

    // the host ports' hand-checked values, and the sharded checkpoint reader, no GPU anywhere
    const ports = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/rows_reference.zig"), .target = b.graph.host, .imports = &.{.{ .name = "safetensors", .module = @import("h3_host.zig").safetensors(b, b.graph.host) }} });
    b.step("test-h3-image-cuda-host", "The host ports of the H3 row kernels and the shard reader, no GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ports })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/checkpoint.zig"), .target = b.graph.host, .imports = &.{.{ .name = "safetensors", .module = @import("h3_host.zig").safetensors(b, b.graph.host) } } }) })).step);
}
