//! Inkling 2 Mini's stored experts (soft-capped NVFP4, MXFP8): nvcc 13.0.88, sm_80/86/89/120/121.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_softcap", .symbol = "tf_nvfp4_softcap", .image = "nvfp4_experts", .threads = 128, .dynamic = 0 }, .usage = .{ .{ .regs = 100, .static = 34816, .bound = 128 }, .{ .regs = 106, .static = 34816, .bound = 128 }, .{ .regs = 134, .static = 34816, .bound = 128 }, .{ .regs = 129, .static = 34816, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_prompt_softcap", .symbol = "tf_nvfp4_prompt_softcap", .image = "nvfp4_experts", .threads = 128, .dynamic = 34816 }, .usage = .{ .{ .regs = 239, .static = 0, .bound = 128 }, .{ .regs = 239, .static = 0, .bound = 128 }, .{ .regs = 239, .static = 0, .bound = 128 }, .{ .regs = 209, .static = 0, .bound = 128 }, .{ .regs = 209, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "fp8_experts tf_mxfp8_softcap", .symbol = "tf_mxfp8_softcap", .image = "fp8_experts", .threads = 128, .dynamic = 0 }, .usage = .{ .{ .regs = 119, .static = 0, .bound = 128 }, .{ .regs = 119, .static = 0, .bound = 128 }, .{ .regs = 119, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "fp8_experts tf_mxfp8_down_f32", .symbol = "tf_mxfp8_down_f32", .image = "fp8_experts", .threads = 128, .dynamic = 0 }, .usage = .{ .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 89, .static = 0, .bound = 128 }, .{ .regs = 89, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "fp8_experts tf_mxfp8_down_bf16", .symbol = "tf_mxfp8_down_bf16", .image = "fp8_experts", .threads = 128, .dynamic = 0 }, .usage = .{ .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 89, .static = 0, .bound = 128 }, .{ .regs = 89, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "fp8_experts tf_mxfp8_prompt_softcap", .symbol = "tf_mxfp8_prompt_softcap", .image = "fp8_experts", .threads = 128, .dynamic = 0 }, .usage = .{ .{ .regs = 146, .static = 33280, .bound = 128 }, .{ .regs = 146, .static = 33280, .bound = 128 }, .{ .regs = 146, .static = 33280, .bound = 128 }, .{ .regs = 160, .static = 33280, .bound = 128 }, .{ .regs = 160, .static = 33280, .bound = 128 } } },
    .{ .launch = .{ .name = "fp8_experts tf_mxfp8_prompt_down", .symbol = "tf_mxfp8_prompt_down", .image = "fp8_experts", .threads = 128, .dynamic = 0 }, .usage = .{ .{ .regs = 140, .static = 33280, .bound = 128 }, .{ .regs = 142, .static = 33280, .bound = 128 }, .{ .regs = 142, .static = 33280, .bound = 128 }, .{ .regs = 154, .static = 33280, .bound = 128 }, .{ .regs = 154, .static = 33280, .bound = 128 } } },
};

/// The 64-row decode entries: unmeasured (no receipt on any SM), so they refuse until one lands (cuda_pending).
pub const pending = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_fact_softcap", .symbol = "tf_nvfp4_fact_softcap", .image = "nvfp4_experts", .threads = 128, .dynamic = 34816 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "fp8_experts tf_mxfp8_staged_down_f32", .symbol = "tf_mxfp8_staged_down_f32", .image = "fp8_experts", .threads = 128, .dynamic = 0 }, .usage = registry.none, .later_pending = true },
};

/// What nvcc 13.2.78 compiles differently.
pub const cuda_132 = [_]registry.Override{
    .{ .name = "nvfp4_experts tf_nvfp4_softcap", .usage = .{ .{ .regs = 100, .static = 34816, .bound = 128 }, .{ .regs = 106, .static = 34816, .bound = 128 }, .{ .regs = 134, .static = 34816, .bound = 128 }, .{ .regs = 128, .static = 34816, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 } } },
    .{ .name = "nvfp4_experts tf_nvfp4_prompt_softcap", .usage = .{ .{ .regs = 239, .static = 0, .bound = 128 }, .{ .regs = 239, .static = 0, .bound = 128 }, .{ .regs = 239, .static = 0, .bound = 128 }, .{ .regs = 208, .static = 0, .bound = 128 }, .{ .regs = 208, .static = 0, .bound = 128 } } },
    .{ .name = "fp8_experts tf_mxfp8_down_f32", .usage = .{ .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 } } },
    .{ .name = "fp8_experts tf_mxfp8_down_bf16", .usage = .{ .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 } } },
};
