//! Inkling 2's inkling2_loop image: every SM's compile records (nvcc 13.0, 13.2, 13.3), each pending.
const registry = @import("registry.zig");

/// Loop is unmeasured: with no receipt on an SM, its launches refuse there (cuda_pending).
pub const pending = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "inkling2_loop tf_ink2_loop_mix", .symbol = "tf_ink2_loop_mix", .image = "inkling2_loop", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_loop tf_ink2_loop_gate_down", .symbol = "tf_ink2_loop_gate_down", .image = "inkling2_loop", .threads = 256, .dynamic = 18432, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_loop tf_ink2_loop_gate_lin", .symbol = "tf_ink2_loop_gate_lin", .image = "inkling2_loop", .threads = 128, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "inkling2_loop tf_ink2_loop_gate_mix", .symbol = "tf_ink2_loop_gate_mix", .image = "inkling2_loop", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};
