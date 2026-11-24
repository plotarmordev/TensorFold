//! Inkling 2 Mini's inkling2_dspark image (the DSpark draft head): every SM's compile records, each pending.
const registry = @import("registry.zig");

/// The head is unmeasured: with no receipt on an SM, its launches refuse there (cuda_pending).
pub const pending = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_fuse", .symbol = "tf_ink2_dsp_fuse", .image = "inkling2_dspark", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_ctx", .symbol = "tf_ink2_dsp_ctx", .image = "inkling2_dspark", .threads = 32, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_init", .symbol = "tf_ink2_dsp_init", .image = "inkling2_dspark", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_add_norm", .symbol = "tf_ink2_dsp_add_norm", .image = "inkling2_dspark", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_qk", .symbol = "tf_ink2_dsp_qk", .image = "inkling2_dspark", .threads = 32, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_attn", .symbol = "tf_ink2_dsp_attn", .image = "inkling2_dspark", .threads = 32, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_swiglu", .symbol = "tf_ink2_dsp_swiglu", .image = "inkling2_dspark", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_kappa", .symbol = "tf_ink2_dsp_kappa", .image = "inkling2_dspark", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_step", .symbol = "tf_ink2_dsp_step", .image = "inkling2_dspark", .threads = 768, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_dspark tf_ink2_dsp_pick", .symbol = "tf_ink2_dsp_pick", .image = "inkling2_dspark", .threads = 1024, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};
