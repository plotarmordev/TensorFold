//! MiniMax-H3's CUDA kernels: the video transformer's, the image decoder's and the prompt's, on the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const h3_k = @import("contract_h3.zig");
const h3_prep = @import("contract_h3_prep.zig");
const h3_topk_k = @import("contract_h3_topk.zig");
const h3d_k = @import("contract_h3_image_decoder.zig");
const h3p_k = @import("contract_h3_image_prompt.zig");

const g = core.contract.gates;

const images = struct {
    pub const h3 align(16) = @embedFile("fatbin_h3").*;
    pub const decoder align(16) = @embedFile("fatbin_h3_image_decoder").*;
    pub const prompt align(16) = @embedFile("fatbin_h3_image_prompt").*;
};

/// Loads H3's three images and appends every kernel they register.
pub fn append(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
    var h3_mod = try cuda.Module.load(gpu.d, &images.h3);
    defer h3_mod.unload();
    const h3_which = comptime std.enums.values(h3_k.Which);
    var h3_list: [h3_which.len]h3_k.H3 = undefined;
    for (h3_which, &h3_list) |w, *m| {
        const f = try h3_mod.function(h3_k.symbol(w));
        if (w == .attention) try f.allowDynamicShared(h3_prep.attention_shared);
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .f = f, .arena = std.heap.ArenaAllocator.init(a) };
    }
    defer for (&h3_list) |*m| m.deinit();
    for (&h3_list) |*m| try kernels.append(a, m.kernel());
    var h3_topk = h3_topk_k.TopK{ .rig = .{ .gpu = gpu, .s = s }, .f = try h3_mod.function("h3_topk") };
    defer h3_topk.deinit();
    try kernels.append(a, h3_topk.kernel());
    var h3d_mod = try cuda.Module.load(gpu.d, &images.decoder);
    defer h3d_mod.unload();
    const h3d_which = comptime std.enums.values(h3d_k.Which);
    var h3d_list: [h3d_which.len]h3d_k.Dec = undefined;
    for (h3d_which, &h3d_list) |w, *m| {
        const f = try h3d_mod.function(h3d_k.symbol(w));
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .f = f, .arena = std.heap.ArenaAllocator.init(a) };
    }
    defer for (&h3d_list) |*m| m.deinit();
    for (&h3d_list) |*m| try kernels.append(a, m.kernel());
    var h3p_mod = try cuda.Module.load(gpu.d, &images.prompt);
    defer h3p_mod.unload();
    const h3p_which = comptime std.enums.values(h3p_k.Which);
    var h3p_list: [h3p_which.len]h3p_k.Prm = undefined;
    for (h3p_which, &h3p_list) |w, *m| {
        const f = try h3p_mod.function(h3p_k.symbol(w));
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .f = f, .arena = std.heap.ArenaAllocator.init(a) };
    }
    defer for (&h3p_list) |*m| m.deinit();
    for (&h3p_list) |*m| try kernels.append(a, m.kernel());
}
