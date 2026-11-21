//! tf-cuda-test kernel-contract: the images the records name, and gate 5 on this GPU.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const options = @import("contract_options");
const check = @import("check.zig");

const contract = core.contract;
const limits = contract.limits;
const registry = contract.registry;
const kk = cuda.kernels;
const records = @import("contract_limits_fatbins.zig").records;
const auditLaunch = @import("contract_limits_fatbins.zig").auditLaunch;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const h3_image = struct {
    pub const bytes align(16) = @embedFile("fatbin_h3").*;
};

/// The named embedded image; only contract_controls may select the planted controls.
const h3_image_decoder = struct {
    pub const bytes align(16) = @embedFile("fatbin_h3_image_decoder").*;
};

const h3_image_prompt = struct {
    pub const bytes align(16) = @embedFile("fatbin_h3_image_prompt").*;
};

const h3_image_modulation = struct {
    pub const bytes align(16) = @embedFile("fatbin_h3_image_modulation").*;
};

pub fn image(name: []const u8) error{UnknownContractImage}![]const u8 {
    if (@import("contract_deepseek_images.zig").image(name)) |bytes| return bytes;
    inline for (.{ "l2pf", "l2pace" }) |n|
        if (std.mem.eql(u8, name, n)) return @field(kk, n);
    if (std.mem.eql(u8, name, "window_rows")) return kk.window_rows;
    if (std.mem.eql(u8, name, "joint_schema")) return kk.joint_schema;
    if (std.mem.eql(u8, name, "fp8_experts")) return kk.fp8_experts;
    if (std.mem.eql(u8, name, "mla")) return kk.mla;
    if (std.mem.eql(u8, name, "h3")) return &h3_image.bytes;
    if (std.mem.eql(u8, name, "h3_image_decoder")) return &h3_image_decoder.bytes;
    if (std.mem.eql(u8, name, "h3_image_prompt")) return &h3_image_prompt.bytes;
    if (std.mem.eql(u8, name, "h3_image_modulation")) return &h3_image_modulation.bytes;
    if (std.mem.eql(u8, name, "flashnext_hcn")) return kk.flashnext_hcn;
    inline for (.{ "qwen4_projection", "qwen4_hc", "qwen4_gdn", "qwen4_ple", "gemma4_ple" }) |n|
        if (std.mem.eql(u8, name, n)) return @field(kk, n);
    inline for (.{ "tree_round", "act_rows", "gdn_prep", "qk_rows", "norms", "moe", "experts", "qwen36_mtp", "nemotron_attention", "nemotron_route", "nemotron_latent", "qmm_group", "qmm_prefill", "lane_gemv", "prompt16", "qmmf", "ck_experts", "nvfp4_experts", "nvfp4_experts_tile", "nvfp4_experts32", "rank_sum", "prefill_attention", "vit", "embed_rows", "route_rows", "flashnext_gdn", "flashnext_batch", "flashnext_hc", "flashnext_attn", "flashnext_mtp", "qwen_vision", "exl3_linear", "exl3_experts", "exl3_candidate", "experts3", "experts3_prefill", "drama3", "drama3_fp8", "drama3_codec", "glm_mm", "glm_dflash", "glm_glue", "glm_ops", "glm_index", "nemotron_mamba", "sample", "qwen35", "qwen35_bf16_rows", "qwen35_keep", "qwen35_kv8", "w4a8_prefill", "qwen35_draft", "ck_lane", "gdn_train", "train_q4", "train_attn", "flashnext_train", "train", "attn_train", "qwen27_train", "tree_attention", "block_attention", "window_attention", "window_pages", "moe_glue", "dense_rows", "qmm", "flashnext_fact", "flashnext_hcf", "flashnext_fexp", "experts_rows", "nemotron_ops", "minimax_m3", "minimax_m3_attn", "experts_bf16", "nemo_audio", "moss_image", "idol", "idol_decoder", "lfm2_moe", "lfm2", "dots_mocr", "muse_glue", "muse_attention", "muse_linear", "muse_experts2", "muse_dflash", "muse_dflash2" }) |n| if (std.mem.eql(u8, name, n)) return @field(kk, n);
    inline for (.{ "inkling2_attn", "inkling2_pattn", "inkling2_ops", "inkling2_rows" }) |n| if (std.mem.eql(u8, name, n)) return @field(kk, n);
    if (std.mem.eql(u8, name, "contract_controls")) return if (options.controls) &controls.bytes else &.{};
    return error.UnknownContractImage;
}

/// Gate 5 on this GPU, each record against the loaded function; `only`: the launches so named, no offline audit.
pub fn live(gpu: check.Gpu, only: []const []const u8) !usize {
    var n: usize = 0;
    if (only.len == 0) n = try registry.checkCuda(gpu.gpa, &registry.all_cuda, false);
    if (only.len == 0) std.debug.print("{s} limits: {d} registered CUDA launches against sm_80/86/89/120/121, {d} problems\n", .{ if (n == 0) "PASS" else "FAIL", registry.all_cuda.len, n });
    const sm: u32 = @intCast(try gpu.ctx.capability());
    const t = limits.target(sm) orelse {
        std.debug.print("N/A  limits live: sm_{d} is not a target the records cover\n", .{sm});
        return n;
    };
    const at: ?usize = for (limits.sms, 0..) |recorded, i| {
        if (recorded.sm == sm) break i;
    } else null;
    const dev = [_]struct { []const u8, cuda.abi.DeviceAttribute, u32 }{
        .{ "most shared bytes a block (opt-in)", .max_shared_memory_per_block_optin, t.smem_optin },
        .{ "threads a block", .max_threads_per_block, t.threads },
    };
    for (dev) |d| {
        const got: u32 = @intCast(try gpu.ctx.attribute(d[1]));
        if (got != d[2]) n += 1;
        std.debug.print("{s} limits live sm_{d} {s}: the driver says {d}, the table {d}\n", .{ if (got == d[2]) "PASS" else "FAIL", sm, d[0], got, d[2] });
    }
    var said: ?limits.Nvcc = null;
    for (registry.all_cuda ++ registry.cuda_controls) |r| {
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        if (only.len > 0) for (only) |w| {
            if (std.mem.indexOf(u8, r.launch.name, w) != null) break;
        } else continue;
        if (!control and t.sm < r.launch.min_sm) continue;
        if (!control and at != null and registry.targetDecision(r, t, r.usage[at.?]) != .applicable) continue;
        const bytes = image(r.launch.image) catch |err| {
            std.debug.print("FAIL kernel-contract: unknown image name: {s}\n", .{r.launch.image});
            return err;
        };
        const rel = if (at != null) try built(gpu.gpa, bytes, sm) else null;
        if (at != null) {
            const set = registry.setFor(rel);
            if (said == null or !std.meta.eql(said.?, set)) records("live", rel, set);
            said = set;
        }
        var mod = try cuda.Module.load(gpu.d, bytes);
        defer mod.unload();
        const sym = try gpu.gpa.dupeSentinel(u8, r.launch.symbol, 0);
        defer gpu.gpa.free(sym);
        const f = try mod.function(sym);
        const regs: u32 = @intCast(try f.attribute(.num_regs));
        const static: u32 = @intCast(try f.attribute(.shared_size_bytes));
        const most: u32 = @intCast(try f.attribute(.max_threads_per_block));
        if (at == null) {
            const usage: limits.Usage = .{ .regs = regs, .static = static, .bound = most };
            if (!control and registry.targetDecision(r, t, usage) != .applicable) continue;
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(gpu.gpa);
            const red = try limits.cuda(auditLaunch(r.launch, sm), usage, t, &out, gpu.gpa);
            if ((control and red == 0) or (!control and red != 0)) n += 1;
            std.debug.print("{s} limits live sm_{d} {s}: {d} registers, {d} static shared bytes, bound {d}{s}", .{
                if (control) (if (red == 0) "MISSED" else "RED ") else if (red == 0) "PASS" else "FAIL",
                sm,
                r.launch.name,
                regs,
                static,
                most,
                if (red == 0) "\n" else ": ",
            });
            if (red != 0) std.debug.print("{s}", .{out.items});
            continue;
        }
        const want = registry.usageFor(r, rel)[at.?];
        const fits = r.launch.threads <= most and static + r.launch.dynamic <= t.smem_optin;
        const current = want.regs == regs and want.static == static;
        if (!control and !(fits and current)) n += 1;
        if (control and fits) n += 1;
        const word = if (control) (if (fits) "MISSED" else "RED ") else if (fits and current) "PASS" else "FAIL";
        std.debug.print("{s} limits live {s}: {d} threads launched, the driver allows {d}; {d} registers, {d} + {d} shared bytes (recorded {d}, {d})\n", .{ word, r.launch.name, r.launch.threads, most, regs, static, r.launch.dynamic, want.regs, want.static });
    }
    return n;
}

/// The nvcc release that built a fatbin's image for `sm`, else the one its images agree on (null: none names one).
fn built(a: std.mem.Allocator, bytes: []const u8, sm: u32) !?limits.Nvcc {
    return contract.cubin.releaseFor(a, bytes, sm);
}
