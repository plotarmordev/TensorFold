//! Inkling 2 on CUDA: Small's kernels (attention, glue) and Mini (inkling2_loop) at one rank: config, plan, weights,
//! the loop's gates and messages, the forward over 48 visit slots a stream.

pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");
pub const config = @import("config.zig");
pub const plan = @import("plan.zig");
pub const names = @import("names.zig");
pub const loop = @import("loop.zig");
pub const mini_weights = @import("mini_weights.zig");
pub const mini_model = @import("mini_model.zig");
pub const mini_backend = @import("mini_backend.zig");
pub const mini_snap = @import("mini_snap.zig");
/// The float64 reference the CUDA forward's greedy tokens are checked against.
pub const mini_ref = @import("mini_ref.zig");
/// The float64 reference the CUDA DSpark draft head's logits are checked against.
pub const dspark_ref = @import("dspark_ref.zig");
/// The native CUDA registry's entry for inkling2_loop (native/cuda_families.zig).
pub const native = @import("mini_native.zig");
/// Mini's DSpark draft head (mtp.safetensors) over the target's committed taps.
pub const dspark = @import("dspark.zig");

/// config.json's name in the release.
pub const model_type = "inkling2";

test {
    _ = kernels;
    _ = ops;
    _ = dspark;
    _ = native;
    _ = loop;
    _ = config;
    _ = plan;
    _ = mini_model;
    _ = @import("mini_visit.zig");
    _ = mini_weights;
    _ = mini_backend;
    _ = mini_snap;
    _ = mini_ref;
    _ = dspark_ref;
}
