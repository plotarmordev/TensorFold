//! Emit only the SASS targets supported by this unit; no eligible targets produces an inert empty image.
const std = @import("std");
const catalog = @import("cuda_kernels.zig");
const Kernel = catalog.Kernel;
const torch_flags = catalog.torch_flags;

/// The `-Dsm` entry and SASS profile rule, kept with the catalog so its host tests cover it.
pub const Target = catalog.Target;
pub const Profile = catalog.Profile;
pub const profile = catalog.profile;
/// On Windows, `nvcc` intermittently exits with code 2 and no diagnostic on a cold toolchain
/// (fresh checkout: first-use scans of its `cicc`/`ptxas`/`cl.exe` subprocesses and `%TEMP%`
/// intermediates abort it; an identical rerun then succeeds). `zig build` cannot retry a `Run`
/// step, so on Windows each compile goes through a generated batch file that retries nvcc.
/// One batch is generated per distinct nvcc path; the `||`-chain re-executes the compile up
/// to four times and yields the last attempt's exit code.
var nvcc_retry_batches: ?std.StringHashMapUnmanaged(std.Build.LazyPath) = null;

fn nvccRetryBatch(b: *std.Build, nvcc: []const u8) std.Build.LazyPath {
    if (nvcc_retry_batches == null) nvcc_retry_batches = .empty;
    const map = &nvcc_retry_batches.?;
    if (map.get(nvcc)) |bat| return bat;
    const bat = b.addWriteFiles().add(b.fmt("nvcc-retry-{d}.bat", .{map.count()}),
        b.fmt("@\"{s}\" %* || @\"{s}\" %* || @\"{s}\" %* || @\"{s}\" %*\r\n", .{ nvcc, nvcc, nvcc, nvcc }));
    map.put(b.allocator, nvcc, bat) catch @panic("OOM");
    return bat;
}

/// An nvcc invocation as a `Run` step: plain on POSIX, but on Windows wrapped in the retry
/// batch (`cmd /c call`, as `hip.zig`'s `hipccRun`), so one cold-toolchain abort cannot fail
/// the build. `%*` forwards the arguments to each retry verbatim, so output and depfile args
/// behave exactly as a direct nvcc call.
pub fn nvccRun(b: *std.Build, nvcc: []const u8, args: []const []const u8) *std.Build.Step.Run {
    const run = if (b.graph.host.result.os.tag == .windows) blk: {
        const r = b.addSystemCommand(&.{ b.graph.environ_map.get("ComSpec") orelse "cmd", "/c", "call" });
        r.addFileArg(nvccRetryBatch(b, nvcc));
        break :blk r;
    } else b.addSystemCommand(&.{nvcc});
    run.addArgs(args);
    return run;
}

/// `nvcc --version` as captured output, for `fatbin`'s version argument, with the Windows retry.
pub fn nvccVersion(b: *std.Build, nvcc: []const u8) std.Build.LazyPath {
    const run = nvccRun(b, nvcc, &.{"--version"});
    run.has_side_effects = true;
    return run.captureStdOut(.{});
}


pub fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    var choices = std.mem.tokenizeScalar(u8, sms, ',');
    var any = false;
    while (choices.next()) |sm| {
        any = any or Target.parse(sm).number >= k.min_sm;
    }
    if (!any) return b.addWriteFiles().add(b.fmt("unsupported-{s}.fatbin", .{k.name}), "");
    const run = nvccRun(b, nvcc, &.{"-fatbin"});
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    // CUDA 13's headers with cl.exe need MSVC's conforming preprocessor and M_SQRT2's math define; under a CJK system
    // code page, C4819 (a header character it cannot represent) buries the real errors in thousands of warnings.
    if (b.graph.host.result.os.tag == .windows) run.addArgs(&.{ "-Xcompiler=/Zc:preprocessor", "-Xcompiler=/wd4819", "-D_USE_MATH_DEFINES" });
    run.addArgs(k.flags);
    // Generated CUDA sources resolve the same maintained headers as checked-in units.
    run.addDirectoryArg2(b.path("zig/kernels/cuda"), .{ .prefix = "-I" });
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| {
        const target = Target.parse(sm);
        const p = profile(k, target) orelse continue;
        // arch-specific features (the block-scaled FP4 mma) from sm_120 on; older targets build the trapping fallback
        const a = if (p.arch_specific and target.suffix.len == 0) "a" else "";
        run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    }
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    if (std.mem.eql(u8, k.name, "qwen4_gdn")) {
        const text = @import("../src/families/flashnext/hybrid_sources.zig").Gdn.cuda(b.allocator, @embedFile("../kernels/cuda/flashnext_gdn.cu")) catch @panic("Flash Next GDN source changed; regenerate Qwen4 specialization");
        const controlled = @import("../src/families/qwen4/gdn_control_source.zig").cuda(b.allocator, text) catch @panic("Qwen4 GDN control boundaries changed");
        run.addFileArg(b.addWriteFiles().add("qwen4_gdn.cu", controlled));
    } else if (std.mem.eql(u8, k.name, "qwen4_ple")) {
        const text = @import("../src/families/flashnext/hybrid_ple_source.zig").cuda(b.allocator) catch @panic("Flash Next PLE source changed; update Qwen4 geometry adapter");
        const controlled = @import("../src/families/qwen4/ple_control_source.zig").cuda(b.allocator, text) catch @panic("Qwen4 PLE controls changed");
        run.addFileArg(b.addWriteFiles().add("qwen4_ple.cu", controlled));
    } else run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}
