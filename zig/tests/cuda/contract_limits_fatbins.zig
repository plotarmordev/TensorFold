//! kernel-contract-limits DIR: the records against nvcc's fatbins by ELF parsing alone; no driver, GPU or contract.
const std = @import("std");
const core = @import("core");
const check = @import("check.zig");

const contract = core.contract;
const limits = contract.limits;
const registry = contract.registry;

/// Says which compiler's records an audit compares against, and why.
pub fn records(what: []const u8, rel: ?limits.Nvcc, set: limits.Nvcc) void {
    if (rel) |v| {
        const own = std.meta.eql(v, set);
        std.debug.print("INFO limits {s}: images built by nvcc {d}.{d}, checked against nvcc {d}.{d}'s records{s}\n", .{ what, v.major, v.minor, set.major, set.minor, if (own) "" else " (the newest set at or below it)" });
    } else std.debug.print("INFO limits {s}: the images name no nvcc release; checked against nvcc {d}.{d}'s records\n", .{ what, set.major, set.minor });
}

/// Only sm_110 cubins need the loader's raised shared-memory limit for these launches.
pub fn auditLaunch(launch: limits.CudaLaunch, sm: u32) limits.CudaLaunch {
    if (sm != 110) return launch;
    const raised = [_][]const u8{
        "qmmf FP8 64-row tile", // zig/src/cuda/qmmf.zig:147
        "qmmf FP8 64-row tile, cluster of 8", // zig/src/cuda/qmmf.zig:147
        "qmmf BF16 32-row tile", // zig/src/cuda/qmmf.zig:147
        "qmmf BF16 32-row tile, cluster of 8", // zig/src/cuda/qmmf.zig:147
        "qmmf shared down 32 stride", // zig/src/cuda/qmmf.zig:189
        "prompt16 FP8", // zig/src/cuda/nvfp4.zig:122
        "qmmf FP8 fp32 out 64-row tile", // zig/src/cuda/qmmf.zig:147
        "qmmf FP8 fp32 out 64-row tile, cluster of 8", // zig/src/cuda/qmmf.zig:147
        "qmmf FP8 fp32 out 64-row tile, fused slices", // zig/src/cuda/qmmf.zig:151
        "qmmf BF16 fp32 out 32-row tile", // zig/src/cuda/qmmf.zig:147
        "qmmf BF16 fp32 out 32-row tile, cluster of 8", // zig/src/cuda/qmmf.zig:147
        "h3 h3_attention", // zig/src/families/h3/cuda.zig:97
    };
    for (raised) |name| if (std.mem.eql(u8, launch.name, name)) {
        var result = launch;
        result.opt_in = true;
        return result;
    };
    return launch;
}

/// kernel-contract-limits DIR: each record against DIR's fatbins; a RECORD line keeps the record for SMs not built.
fn keep(a: std.mem.Allocator, line: *std.ArrayList(u8), r: registry.CudaRecord, compiler: ?limits.Nvcc, i: usize) !void {
    const u = registry.usageFor(r, compiler)[i];
    try line.print(a, ".{{ .regs = {d}, .static = {d}, .bound = {d} }}, ", .{ u.regs, u.static, u.bound });
}

/// The served SM images in DIR's fatbins against the records of `images` (none: every record), controls among them.
pub fn fatbins(a: std.mem.Allocator, io: std.Io, dir: []const u8, images: []const []const u8) !void {
    var n: usize = 0;
    var said: ?limits.Nvcc = null;
    const mode_path = try std.fmt.allocPrint(a, "{s}/norms.fatbin", .{dir});
    defer a.free(mode_path);
    const mode_bytes = try std.Io.Dir.cwd().readFileAlloc(io, mode_path, a, .limited(1 << 28));
    defer a.free(mode_bytes);
    const mode_images = try contract.cubin.images(a, mode_bytes);
    defer contract.cubin.freeImages(a, mode_images);
    var release = false;
    var thor = false;
    for (mode_images) |img| {
        thor = thor or img.sm == 110;
        for (limits.sms) |t| release = release or img.sm == t.sm;
    }
    if (!release and !thor) return error.NoKnownSmInFatbins;
    for (registry.all_cuda ++ registry.cuda_controls) |r| {
        if (images.len > 0) for (images) |name| {
            if (std.mem.eql(u8, r.launch.image, name)) break;
        } else continue;
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        const any_target = for (limits.sms, 0..) |target, index| {
            if (registry.targetDecision(r, target, r.usage[index]) == .applicable) break true;
        } else false;
        const thor_target = thor and r.launch.min_sm <= 110;
        if (!control and !(release and any_target) and !thor_target) {
            std.debug.print("N/A limits {s}: no applicable target\n", .{r.launch.name});
            continue;
        }
        const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ dir, r.launch.image });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
        defer a.free(bytes);
        const imgs = try contract.cubin.images(a, bytes);
        defer contract.cubin.freeImages(a, imgs);
        const compiler: ?limits.Nvcc = if (imgs.len > 0) try contract.cubin.release(a, imgs[0].elf) else null;
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(a);
        if (release) for (limits.sms, 0..) |t, i| {
            const decision = registry.targetDecision(r, t, r.usage[i]);
            if (!control and decision != .applicable) {
                std.debug.print("N/A limits {s} sm_{d}: {t}\n", .{ r.launch.name, t.sm, decision });
                try keep(a, &line, r, compiler, i);
                continue;
            }
            const img = for (imgs) |im| {
                if (im.sm == t.sm) break im;
            } else {
                std.debug.print("FAIL limits {s}: {s} holds no sm_{d} image (build with -Dsm=80,86,89,120,121)\n", .{ r.launch.name, path, t.sm });
                n += 1;
                try keep(a, &line, r, compiler, i);
                continue;
            };
            const ks = try contract.cubin.kernels(a, img.elf, t.sm);
            defer a.free(ks);
            const k = for (ks) |kv| {
                if (std.mem.eql(u8, kv.name, r.launch.symbol)) break kv;
            } else return error.SymbolNotInImage;
            const rel = try contract.cubin.release(a, img.elf);
            const set = registry.setFor(rel);
            if (said == null or !std.meta.eql(said.?, set)) records("fatbins", rel, set);
            said = set;
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(a);
            const measured = registry.targetDecision(r, t, k.usage);
            if (!control and measured != .applicable) {
                std.debug.print("N/A limits {s} sm_{d}: {t}\n", .{ r.launch.name, t.sm, measured });
                try keep(a, &line, r, compiler, i);
                continue;
            }
            const problems = try limits.cuda(r.launch, k.usage, t, &out, a);
            const current = std.meta.eql(k.usage, registry.usageFor(r, rel)[i]);
            if (!control and !current) n += 1;
            if (control and problems == 0) n += 1;
            if (!control) n += problems;
            std.debug.print("{s} limits sm_{d} {s}: {d} registers, {d} static shared bytes, bound {d} (record {s}){s}", .{ if (control) (if (problems == 0) "MISSED" else "RED ") else if (problems == 0) "PASS" else "FAIL", t.sm, r.launch.name, k.usage.regs, k.usage.static, k.usage.bound, if (current) "current" else "stale", if (out.items.len > 0) ": " else "\n" });
            if (out.items.len > 0) std.debug.print("{s}", .{out.items});
            try line.print(a, ".{{ .regs = {d}, .static = {d}, .bound = {d} }}, ", .{ k.usage.regs, k.usage.static, k.usage.bound });
        };
        if (thor_target) {
            const t = limits.target(110).?;
            const img = for (imgs) |im| {
                if (im.sm == 110) break im;
            } else {
                std.debug.print("FAIL limits {s}: {s} holds no sm_110 image\n", .{ r.launch.name, path });
                n += 1;
                continue;
            };
            const ks = try contract.cubin.kernels(a, img.elf, 110);
            defer a.free(ks);
            const k = for (ks) |kv| {
                if (std.mem.eql(u8, kv.name, r.launch.symbol)) break kv;
            } else return error.SymbolNotInImage;
            const decision = registry.targetDecision(r, t, k.usage);
            if (!control and decision != .applicable) {
                std.debug.print("N/A limits {s} sm_110: {t}\n", .{ r.launch.name, decision });
            } else {
                var out: std.ArrayList(u8) = .empty;
                defer out.deinit(a);
                const problems = try limits.cuda(auditLaunch(r.launch, 110), k.usage, t, &out, a);
                if (control and problems == 0) n += 1;
                if (!control) n += problems;
                std.debug.print("{s} limits sm_110 {s}: {d} registers, {d} static shared bytes, bound {d}{s}", .{
                    if (control) (if (problems == 0) "MISSED" else "RED ") else if (problems == 0) "PASS" else "FAIL",
                    r.launch.name,
                    k.usage.regs,
                    k.usage.static,
                    k.usage.bound,
                    if (problems == 0) "\n" else ": ",
                });
                if (problems != 0) std.debug.print("{s}", .{out.items});
            }
        }
        if (release) std.debug.print("RECORD {s}: .usage = .{{ {s}}}\n", .{ r.launch.name, line.items });
    }
    try check.expect(n == 0, "kernel-contract-limits: {d} problems", .{n});
    if (thor) {
        check.pass("kernel-contract-limits: every audited launch fits sm_110 from compiled cubins, every control red", .{});
    } else {
        check.pass("kernel-contract-limits: every audited launch fits sm_80, sm_86, sm_89, sm_120 and sm_121, every control red", .{});
    }
}
