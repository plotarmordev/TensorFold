//! CUDA DSpark numerical contracts against independent host arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const dspark = @import("inkling2").dspark;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");

const g = core.contract.gates;

const base = @import("contract_inkling2_dspark_core.zig");
const attention = @import("contract_inkling2_dspark_attention.zig");
const markov = @import("contract_inkling2_dspark_markov.zig");
const AddNorm = base.AddNorm;
const Swiglu = base.Swiglu;
const Pick = base.Pick;
const Fuse = base.Fuse;
const Ctx = base.Ctx;
const Init = base.Init;
const Qk = attention.Qk;
const Attn = attention.Attn;
const Kappa = markov.Kappa;
const Step = markov.Step;
const groups = .{ "norms", "swiglus", "picks", "fuses", "ctxs", "inits", "qks", "attns", "kappas", "steps" };
pub const Set = struct {
    k: dspark.Kernels = undefined,
    loaded: bool = false,
    norms: [2]ref.Gated(AddNorm) = undefined,
    swiglus: [2]ref.Gated(Swiglu) = undefined,
    picks: [2]ref.Checked(Pick) = undefined,
    fuses: [2]ref.Gated(Fuse) = undefined,
    ctxs: [2]ref.Checked(Ctx) = undefined,
    inits: [2]ref.Checked(Init) = undefined,
    qks: [2]ref.Checked(Qk) = undefined,
    attns: [2]ref.Gated(Attn) = undefined,
    kappas: [2]ref.Gated(Kappa) = undefined,
    steps: [2]ref.Checked(Step) = undefined,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try dspark.Kernels.load(gpu.d);
        errdefer set.k.unload();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&set.norms, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .stale = i == 1 } };
        for (&set.swiglus, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .swapped = i == 1 } };
        for (&set.picks, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .short = i == 1 } };
        for (&set.fuses, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .bymax = i == 1 } };
        for (&set.ctxs, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .raw = i == 1 } };
        for (&set.inits, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .sqrt = i == 1 } };
        for (&set.qks, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .twice = i == 1 } };
        for (&set.attns, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .mean = i == 1 } };
        for (&set.kappas, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .bias2 = i == 1 } };
        for (&set.steps, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.k, .flipped = i == 1 } };
        errdefer {
            inline for (groups) |field| for (&@field(set, field)) |*x| x.deinit();
        }
        inline for (groups) |field| for (&@field(set, field)) |*x| try list.append(a, x.kernel());
        set.loaded = true;
    }
    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        inline for (groups) |field| for (&@field(set, field)) |*x| x.deinit();
        set.k.unload();
        set.loaded = false;
    }
};
