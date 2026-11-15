//! MiniMax-H3-Image's CUDA launches. Resource usage is measured by kernel-contract-limits on the release images.
const std = @import("std");
const registry = @import("registry.zig");

fn pending(comptime symbol: []const u8, comptime image: []const u8, threads: u32, min_sm: u32) registry.CudaRecord {
    return .{
        .launch = .{ .name = "h3_image " ++ symbol, .symbol = symbol, .image = image, .threads = threads, .min_sm = min_sm },
        .usage = registry.none,
    };
}

pub const records = [_]registry.CudaRecord{
    pending("h3d_post_quant", "h3_image_decoder", 256, 80),
    pending("h3d_embed", "h3_image_decoder", 256, 80),
    pending("h3d_rms_norm", "h3_image_decoder", 256, 80),
    pending("h3d_layer_norm", "h3_image_decoder", 256, 80),
    pending("h3d_finish", "h3_image_decoder", 256, 80),
    pending("h3d_heads", "h3_image_decoder", 32, 80),
    pending("h3d_attention", "h3_image_decoder", 256, 80),
    pending("h3d_swiglu", "h3_image_decoder", 256, 80),
    pending("h3p_norm", "h3_image_prompt", 256, 80),
    pending("h3p_heads", "h3_image_prompt", 256, 80),
    pending("h3p_attention", "h3_image_prompt", 256, 80),
    pending("h3p_glu", "h3_image_prompt", 256, 80),
    pending("h3p_add", "h3_image_prompt", 256, 80),
    pending("h3p_scatter_add", "h3_image_prompt", 256, 80),
    pending("h3p_embed", "h3_image_prompt", 256, 80),
    pending("h3p_splice", "h3_image_prompt", 256, 80),
    pending("h3p_linear", "h3_image_prompt", 256, 80),
    pending("h3p_round", "h3_image_prompt", 256, 80),
    pending("h3m_sinusoid", "h3_image_modulation", 256, 0),
    pending("h3m_embed", "h3_image_modulation", 256, 0),
    pending("h3m_project", "h3_image_modulation", 256, 0),
};

test "MiniMax-H3-Image CUDA: every served symbol has one registered launch" {
    for (records) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(r.usage.len, try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
