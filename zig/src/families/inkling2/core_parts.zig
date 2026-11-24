//! The core parts the family's host files import by name (safetensors, visit_plan, modelopt), as the CUDA family
//! module reaches them through core; the host-only tests import the core files themselves (build/inkling2_host.zig).
const core = @import("core");

pub const DType = core.safetensors.DType;
pub const Header = core.safetensors.Header;
pub const Entry = core.safetensors.Entry;

pub const Plan = core.visit_plan.Plan;
pub const Loop = core.visit_plan.Loop;
pub const Tap = core.visit_plan.Tap;
pub const straight = core.visit_plan.straight;

pub const Why = core.modelopt.Why;
pub const admitByTensors = core.modelopt.admitByTensors;
