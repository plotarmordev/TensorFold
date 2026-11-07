//! Underdog Woof: a speech family over `core` (on macOS, `metal`), its host tests, parity and bench programs.
const std = @import("std");

/// Woof's `core`: npy (its npz checkpoint and voice files) re-exported from the engine module, as drama's shim does.
pub fn core(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, tensorfold: *std.Build.Module) *std.Build.Module {
    const shim = b.addWriteFiles().add("woof_core.zig", "pub const npy = @import(\"tensorfold\").npy;\n");
    return b.createModule(.{ .root_source_file = shim, .target = target, .optimize = optimize, .imports = &.{.{ .name = "tensorfold", .module = tensorfold }} });
}

/// The family module (woof.zig imports `core`'s npy loader; given `metal`, its batched matmuls run on the GPU).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, core_module: *std.Build.Module, metal: ?*std.Build.Module) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "gpu", metal != null);
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/woof/woof.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core_module }, .{ .name = "woof_options", .module = options.createModule() } },
    });
    if (metal) |mtl| m.addImport("metal", mtl);
    return m;
}

/// Woof in a platform's `native_engines` module: the host reads it as `woof`; `core_module` is what the family imports.
pub fn addNative(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, native: *std.Build.Module, core_module: *std.Build.Module, metal: *std.Build.Module) void {
    native.addImport("woof", module(b, target, optimize, core_module, metal));
}

/// The family's tests under `zig build test` and its parity/bench programs in zig-out/bin; `core` is the real root.zig.
pub fn targets(b: *std.Build, test_step: *std.Build.Step, metal: *std.Build.Module) void {
    const host = b.graph.host;
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = host, .optimize = .ReleaseSafe, .link_libc = true });
    const host_core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = host, .optimize = .debug, .link_libc = true });
    host_core.addImport("tokenizer", tokenizer);
    const family = module(b, host, .debug, host_core, metal);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family })).step);
    const family_fast = module(b, host, .ReleaseFast, host_core, metal);
    for ([_]struct { name: []const u8, path: []const u8, about: []const u8 }{
        .{ .name = "tf-woof-parity", .path = "zig/tests/woof_parity.zig", .about = "Woof speech against its numpy runtime's recorded samples" },
        .{ .name = "tf-woof-bench", .path = "zig/tests/woof_bench.zig", .about = "Woof first audio, real-time factor and peak memory over a speech corpus" },
    }) |p| {
        const exe = b.addExecutable(.{ .name = p.name, .root_module = b.createModule(.{
            .root_source_file = b.path(p.path),
            .target = host,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{.{ .name = "woof", .module = family_fast }},
        }) });
        b.step(p.name, p.about).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
}
