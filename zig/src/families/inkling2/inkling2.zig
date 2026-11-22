//! Inkling 2 on CUDA: Small's kernels (attention, glue) and Mini (inkling2_loop) at one rank: config, plan, weights,
//! the loop's gates and messages, the forward over 48 visit slots a stream.

pub const image = @import("image.zig");
pub const vision = @import("vision.zig");
pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");
pub const experts = @import("experts.zig");
pub const modelopt = @import("modelopt.zig");
pub const config = @import("config.zig");
pub const plan = @import("plan.zig");
pub const names = @import("names.zig");
pub const loop = @import("loop.zig");
pub const loop_cache = @import("loop_cache.zig");
pub const loop_model = @import("loop_model.zig");
pub const loop_math_cuda = @import("loop_math_cuda.zig");
pub const mini_weights = @import("mini_weights.zig");
pub const mini_model = @import("mini_model.zig");
pub const mini_backend = @import("mini_backend.zig");
pub const mini_snap = @import("mini_snap.zig");
/// The float64 reference the CUDA forward's greedy tokens are checked against.
pub const mini_ref = @import("mini_ref.zig");
pub const mini_native = @import("mini_native.zig");
/// The native CUDA registry's retained Inkling 2 loop entry.
pub const native = @import("cuda_native.zig");

/// config.json's name in the release.
pub const model_type = "inkling2";

test {
    _ = image;
    _ = vision;
    _ = kernels;
    _ = ops;
    _ = experts;
    _ = modelopt;
    _ = native;
    _ = loop;
    _ = loop_cache;
    _ = loop_model;
    _ = loop_math_cuda;
    _ = config;
    _ = plan;
    _ = mini_model;
    _ = @import("mini_visit.zig");
    _ = mini_weights;
    _ = mini_backend;
    _ = mini_snap;
    _ = mini_ref;
    _ = mini_native;
}
