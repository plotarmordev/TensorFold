//! Real Metal dispatch for Inkling 2 Mini's M3 NVFP4/MXFP8 routed experts.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
const plan = @import("experts_plan.zig");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Format = plan.Format;
pub const Call = plan.Call;
pub const Operation = enum {
    nvfp4_gu,
    nvfp4_down,
    mxfp8_gu,
    mxfp8_down,
    nvfp4_quantize,
    mxfp8_quantize,
    combine,
    expert_group,
    nvfp4_gu_grouped,
    nvfp4_down_grouped,
    mxfp8_gu_grouped,
    mxfp8_down_grouped,
};

pub const function_names = [_][:0]const u8{
    "tf_ink2_nvfp4_gu",
    "tf_ink2_nvfp4_down",
    "tf_ink2_mxfp8_gu",
    "tf_ink2_mxfp8_down",
    "tf_ink2_nvfp4_quantize",
    "tf_ink2_mxfp8_quantize",
    "tf_ink2_combine",
    "tf_ink2_expert_group",
    "tf_ink2_nvfp4_gu_grouped",
    "tf_ink2_nvfp4_down_grouped",
    "tf_ink2_mxfp8_gu_grouped",
    "tf_ink2_mxfp8_down_grouped",
};

/// The grouping launch, then gate-up and down for each format, on simdgroup matrices for GPUs without tensor units.
const mma_names = [_][:0]const u8{ "tf_ink2_mma_group", "tf_ink2_nvfp4_gu_mma", "tf_ink2_nvfp4_down_mma", "tf_ink2_mxfp8_gu_mma", "tf_ink2_mxfp8_down_mma" };
/// The weight rows a simdgroup-matrix threadgroup takes.
const mma_tile_rows = 64;
/// The route pairs one expert may take: a chunk's rows, each of which names an expert once.
const mma_cap: u32 = 256;
/// A chunk of at least this many rows takes the simdgroup-matrix kernels, where the GPU has no tensor units.
pub const mma_rows: u32 = 16;
pub const max_grouped_rows: u32 = plan.max_grouped_rows;
const units_gu_k: u32 = 2304;
const units_down_k: u32 = 1152;
const units_group_cap: u32 = 1024;
const turned_rows: u32 = 512;
const units_names = [_][:0]const u8{
    "tf_ink2_group_units",
    "tf_ink2_nvfp4_gu_units",
    "tf_ink2_nvfp4_down_units",
    "tf_ink2_mxfp8_gu_units",
    "tf_ink2_mxfp8_down_units",
    "tf_ink2_logits_units",
    "tf_ink2_dense_units",
    "tf_ink2_nvfp4_gu_units_t",
    "tf_ink2_nvfp4_down_units_t",
    "tf_ink2_mxfp8_gu_units_t",
    "tf_ink2_mxfp8_down_units_t",
};

const MmaGroupArgs = extern struct { pairs: u32, experts: u32, cap: u32 };
const MmaArgs = extern struct { k: u32, n: u32, topk: u32, cap: u32 };
pub const Globals = struct { amax: Ref, denominator: Ref };

pub const Pipelines = struct {
    entries: [function_names.len]mtl.Pipeline,
    /// Each expert's count, then its route pairs: launches in one encoder run in order, so they share it.
    groups: ?mtl.Buffer = null,
    mma: ?[mma_names.len]mtl.Pipeline = null,
    units: ?[units_names.len]mtl.Pipeline = null,

    pub fn load(device: mtl.Device) !Pipelines {
        var out = try loadSource(device, src.experts);
        errdefer out.deinit();
        if (device.tensorUnits()) try out.loadUnits(device, src.experts_units) else try out.loadMma(device, src.experts_mma);
        return out;
    }

    /// The simdgroup-matrix kernels from `text`; the contract's controls pass an edited one.
    pub fn loadMma(p: *Pipelines, device: mtl.Device, text: []const u8) !void {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [mma_names.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for (mma_names, &pipes) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        const groups = try device.buffer((plan.expert_count + plan.expert_count * mma_cap) * @sizeOf(u32), mtl.ResourceOptions.shared);
        if (p.mma) |old| for (old) |pipe| pipe.deinit();
        if (p.groups) |old| old.deinit();
        p.mma = pipes;
        p.groups = groups;
    }

    pub fn loadUnits(p: *Pipelines, device: mtl.Device, text: []const u8) !void {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [units_names.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for (units_names, &pipes) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.maxThreads() < (if (made == 1) @as(usize, 32) else 256)) return error.ExpertThreads;
        }
        const groups = try device.buffer((plan.expert_count + plan.expert_count * units_group_cap) * @sizeOf(u32), mtl.ResourceOptions.shared);
        if (p.units) |old| for (old) |pipe| pipe.deinit();
        if (p.groups) |old| old.deinit();
        p.units = pipes;
        p.groups = groups;
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = .{ .entries = undefined };
        var made: usize = 0;
        errdefer for (out.entries[0..made]) |pipe| pipe.deinit();
        for (function_names, 0..) |name, i| {
            out.entries[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            const need: usize = if (i == @backingInt(Operation.expert_group)) 128 else 32;
            if (out.entries[i].maxThreads() < need) return error.ExpertThreads;
        }
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        if (p.units) |pipes| for (pipes) |pipe| pipe.deinit();
        if (p.mma) |pipes| for (pipes) |pipe| pipe.deinit();
        if (p.groups) |buf| buf.deinit();
        for (p.entries) |pipe| pipe.deinit();
    }

    fn get(p: *const Pipelines, op: Operation) mtl.Pipeline {
        return p.entries[@backingInt(op)];
    }
};

fn args(c: Call) plan.MatrixArgs {
    return .{ .k = c.k, .n = c.n, .topk = c.topk, .experts = c.experts };
}

fn groupedArgs(c: Call) plan.GroupedMatrixArgs {
    return .{ .rows = c.rows, .k = c.k, .n = c.n, .topk = c.topk, .experts = c.experts };
}

fn launch(p: *const Pipelines, e: mtl.ComputeEncoder, op: Operation, groups: u32) void {
    e.setPipeline(p.get(op));
    e.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(32, 1, 1));
}

/// Products and combine: groups along outputs (x) by route pairs or tokens (y); 16-byte words need 16-byte offsets.
fn launchGrid(p: *const Pipelines, e: mtl.ComputeEncoder, op: Operation, g: plan.Geometry, words: []const Ref) !void {
    for (words) |r| if (r.off % 16 != 0) return error.ExpertAlignment;
    e.setPipeline(p.get(op));
    e.dispatchGroups(mtl.Size.of(g.blocks, g.groups, 1), mtl.Size.of(g.threads, 1, 1));
}

/// Packed NVFP4 GU: X [rows,k], ids [rows,topk], Y [rows*topk,n].
pub fn nvfp4Gu(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, amax: Ref, denominator: Ref, ids: Ref, out: Ref, c: Call) !void {
    if (c.format != .nvfp4) return error.WrongExpertFormat;
    try plan.check(c);
    for ([_]Ref{ x, w, scales, amax, denominator, ids, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 7);
    try launchGrid(p, e, .nvfp4_gu, plan.geometry(c), &.{ x, w });
}

/// Packed MXFP8 GU with UE8M0 scales per row/block32.
pub fn mxfp8Gu(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, ids: Ref, out: Ref, c: Call) !void {
    if (c.format != .mxfp8) return error.WrongExpertFormat;
    try plan.check(c);
    for ([_]Ref{ x, w, scales, ids, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 5);
    try launchGrid(p, e, .mxfp8_gu, plan.geometry(c), &.{ x, w });
}

pub fn nvfp4Down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, w: Ref, scales: Ref, amax: Ref, denominator: Ref, ids: Ref, out: Ref, c: Call) !void {
    if (c.format != .nvfp4) return error.WrongExpertFormat;
    try plan.check(c);
    for ([_]Ref{ act, w, scales, amax, denominator, ids, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 7);
    try launchGrid(p, e, .nvfp4_down, plan.geometry(c), &.{ act, w });
}

pub fn mxfp8Down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, w: Ref, scales: Ref, ids: Ref, out: Ref, c: Call) !void {
    if (c.format != .mxfp8) return error.WrongExpertFormat;
    try plan.check(c);
    for ([_]Ref{ act, w, scales, ids, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 5);
    try launchGrid(p, e, .mxfp8_down, plan.geometry(c), &.{ act, w });
}

/// Decode one expert matrix to BF16, not an activation quantization operation.
pub fn nvfp4Quantize(p: *const Pipelines, e: mtl.ComputeEncoder, w: Ref, scales: Ref, amax: Ref, denominator: Ref, out: Ref, rows: u32, k: u32) !void {
    if (k == 0 or k % 16 != 0 or rows == 0) return error.BadExpertBlock;
    for ([_]Ref{ w, scales, amax, denominator, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(plan.DecodeArgs{ .rows = rows, .k = k }, 5);
    launch(p, e, .nvfp4_quantize, (rows * k + 31) / 32);
}

pub fn mxfp8Quantize(p: *const Pipelines, e: mtl.ComputeEncoder, w: Ref, scales: Ref, out: Ref, rows: u32, k: u32) !void {
    if (k == 0 or k % 32 != 0 or rows == 0) return error.BadExpertBlock;
    for ([_]Ref{ w, scales, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(plan.DecodeArgs{ .rows = rows, .k = k }, 3);
    launch(p, e, .mxfp8_quantize, (rows * k + 31) / 32);
}

/// Sum per-token BF16 down outputs in ascending expert index and round once to BF16.
pub fn combine(p: *const Pipelines, e: mtl.ComputeEncoder, down: Ref, ids: Ref, weights: Ref, out: Ref, c: Call) !void {
    try plan.check(c);
    // The kernel orders a token's routes in eight registers.
    if (c.topk > 8) return error.BadExpertShape;
    for ([_]Ref{ down, ids, weights, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 4);
    // Combine's rows are tokens, rather than route pairs.
    try launchGrid(p, e, .combine, plan.combineGeometry(c), &.{});
}

pub fn gu(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, nv_globals: ?struct { amax: Ref, denominator: Ref }, ids: Ref, out: Ref, c: Call) !void {
    switch (c.format) {
        .nvfp4 => if (nv_globals) |g| return nvfp4Gu(p, e, x, w, scales, g.amax, g.denominator, ids, out, c) else return error.MissingNvfp4Globals,
        .mxfp8 => return mxfp8Gu(p, e, x, w, scales, ids, out, c),
    }
}

/// Group route pairs in ascending pair order for either M5 tensor units or non-M5 simdgroup matrices.
pub fn group(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, c: Call) !void {
    try plan.check(c);
    const cap = if (p.units != null) units_group_cap else mma_cap;
    if (c.rows > cap or c.experts > plan.expert_count) return error.BadExpertShape;
    const groups = p.groups orelse return error.NoGroupedExperts;
    e.setPipeline(if (p.units) |pipes| pipes[0] else (p.mma orelse return error.NoGroupedExperts)[0]);
    e.setBuffer(ids.buf, ids.off, 0);
    e.setBuffer(groups, 0, 1);
    e.setBuffer(groups, plan.expert_count * @sizeOf(u32), 2);
    e.setValue(MmaGroupArgs{ .pairs = c.rows * c.topk, .experts = c.experts, .cap = cap }, 3);
    if (p.units != null)
        e.dispatchGroups(mtl.Size.of((c.experts + 31) / 32, 1, 1), mtl.Size.of(32, 1, 1))
    else
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(c.experts, 1, 1));
}

/// Gate-up, or down (`down`), after `group`: 64 weight rows by bands of 32 of an expert's pairs a threadgroup.
pub fn mma(p: *const Pipelines, e: mtl.ComputeEncoder, down: bool, x: Ref, w: Ref, scales: Ref, nv_globals: ?Globals, out: Ref, c: Call) !void {
    const pipes = p.mma orelse return error.NoSimdgroupMatrices;
    try plan.check(c);
    const weight_rows = if (down) c.n else 2 * c.n;
    if (weight_rows % mma_tile_rows != 0 or c.k % 64 != 0 or c.experts > plan.expert_count) return error.BadExpertShape;
    const groups = p.groups.?;
    const nv = c.format == .nvfp4;
    e.setPipeline(pipes[1 + (if (nv) @as(usize, 0) else 2) + @intFromBool(down)]);
    var at: usize = 0;
    for ([_]Ref{ x, w, scales }) |b| {
        e.setBuffer(b.buf, b.off, at);
        at += 1;
    }
    if (nv) {
        const g = nv_globals orelse return error.MissingNvfp4Globals;
        for ([_]Ref{ g.amax, g.denominator }) |b| {
            e.setBuffer(b.buf, b.off, at);
            at += 1;
        }
    }
    e.setBuffer(groups, 0, at);
    e.setBuffer(groups, plan.expert_count * @sizeOf(u32), at + 1);
    e.setBuffer(out.buf, out.off, at + 2);
    e.setValue(MmaArgs{ .k = c.k, .n = c.n, .topk = c.topk, .cap = mma_cap }, at + 3);
    e.dispatchGroups(mtl.Size.of(weight_rows / mma_tile_rows, c.experts, 1), mtl.Size.of(256, 1, 1));
}

/// Bucket route pairs by expert ID into <= 8-pair work items on GPU without host readback.
pub fn groupBuckets(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, out: Ref, c: Call) !void {
    try plan.checkGrouped(c);
    e.setBuffer(ids.buf, ids.off, 0);
    e.setBuffer(out.buf, out.off, 1);
    e.setValue(plan.GroupArgs{ .rows = c.rows, .topk = c.topk, .experts = c.experts }, 2);
    e.setPipeline(p.get(.expert_group));
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(128, 1, 1));
}

pub fn nvfp4GuGrouped(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, amax: Ref, denominator: Ref, group_buf: Ref, out: Ref, c: Call) !void {
    if (c.format != .nvfp4) return error.WrongExpertFormat;
    try plan.checkGrouped(c);
    for ([_]Ref{ x, w, scales, amax, denominator, group_buf, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(groupedArgs(c), 7);
    try launchGrid(p, e, .nvfp4_gu_grouped, plan.groupedGeometry(c), &.{});
}

pub fn mxfp8GuGrouped(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, group_buf: Ref, out: Ref, c: Call) !void {
    if (c.format != .mxfp8) return error.WrongExpertFormat;
    try plan.checkGrouped(c);
    for ([_]Ref{ x, w, scales, group_buf, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(groupedArgs(c), 5);
    try launchGrid(p, e, .mxfp8_gu_grouped, plan.groupedGeometry(c), &.{});
}

pub fn nvfp4DownGrouped(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, w: Ref, scales: Ref, amax: Ref, denominator: Ref, group_buf: Ref, out: Ref, c: Call) !void {
    if (c.format != .nvfp4) return error.WrongExpertFormat;
    try plan.checkGrouped(c);
    for ([_]Ref{ act, w, scales, amax, denominator, group_buf, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(groupedArgs(c), 7);
    try launchGrid(p, e, .nvfp4_down_grouped, plan.groupedGeometry(c), &.{});
}

pub fn mxfp8DownGrouped(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, w: Ref, scales: Ref, group_buf: Ref, out: Ref, c: Call) !void {
    if (c.format != .mxfp8) return error.WrongExpertFormat;
    try plan.checkGrouped(c);
    for ([_]Ref{ act, w, scales, group_buf, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(groupedArgs(c), 5);
    try launchGrid(p, e, .mxfp8_down_grouped, plan.groupedGeometry(c), &.{});
}

pub fn guGrouped(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, nv_globals: ?struct { amax: Ref, denominator: Ref }, group_buf: Ref, out: Ref, c: Call) !void {
    switch (c.format) {
        .nvfp4 => if (nv_globals) |g| return nvfp4GuGrouped(p, e, x, w, scales, g.amax, g.denominator, group_buf, out, c) else return error.MissingNvfp4Globals,
        .mxfp8 => return mxfp8GuGrouped(p, e, x, w, scales, group_buf, out, c),
    }
}

const UnitsArgs = extern struct { k: u32, n: u32, topk: u32, cap: u32 };
const DenseUnitsArgs = extern struct { rows: u32, n: u32, k: u32 };
const LogitUnitsArgs = extern struct { rows: u32, vocab: u32 };

/// M5 tensor units cover both decode and wide prompt chunks, with an expert's pairs grouped once.
pub fn units(p: *const Pipelines, e: mtl.ComputeEncoder, down: bool, input: Ref, weight: Ref, scales: Ref, globals: ?Globals, out: Ref, c: Call) !void {
    return launchUnits(p, e, down, c.rows > turned_rows, input, weight, scales, globals, out, c);
}

pub fn unitsTurned(p: *const Pipelines, e: mtl.ComputeEncoder, down: bool, input: Ref, weight: Ref, scales: Ref, globals: ?Globals, out: Ref, c: Call) !void {
    return launchUnits(p, e, down, true, input, weight, scales, globals, out, c);
}

fn launchUnits(p: *const Pipelines, e: mtl.ComputeEncoder, down: bool, turned: bool, input: Ref, weight: Ref, scales: Ref, globals: ?Globals, out: Ref, c: Call) !void {
    try plan.check(c);
    const pipes = p.units orelse return error.NoTensorUnits;
    const n_weight = if (down) c.n else 2 * c.n;
    if (c.k != (if (down) units_down_k else units_gu_k) or n_weight % 32 != 0 or c.rows > units_group_cap) return error.BadExpertShape;
    const groups = p.groups orelse return error.NoTensorUnits;
    const first: usize = if (turned) 7 else 1;
    var slot: usize = 0;
    for ([_]Ref{ input, weight, scales }) |value| {
        e.setBuffer(value.buf, value.off, slot);
        slot += 1;
    }
    if (c.format == .nvfp4) {
        const g = globals orelse return error.MissingNvfp4Globals;
        for ([_]Ref{ g.amax, g.denominator }) |value| {
            e.setBuffer(value.buf, value.off, slot);
            slot += 1;
        }
    }
    e.setBuffer(groups, 0, slot);
    e.setBuffer(groups, plan.expert_count * @sizeOf(u32), slot + 1);
    e.setBuffer(out.buf, out.off, slot + 2);
    e.setValue(UnitsArgs{ .k = c.k, .n = c.n, .topk = c.topk, .cap = units_group_cap }, slot + 3);
    e.setPipeline(pipes[first + (if (c.format == .mxfp8) @as(usize, 2) else 0) + @as(usize, @intFromBool(down))]);
    e.dispatchGroups(mtl.Size.of(if (turned) n_weight / 16 else n_weight / 32, c.experts, 1), mtl.Size.of(256, 1, 1));
}

pub fn denseUnits(p: *const Pipelines, e: mtl.ComputeEncoder, input: Ref, weight: Ref, out: Ref, rows: u32, n: u32, k: u32) !void {
    const pipes = p.units orelse return error.NoTensorUnits;
    if (rows == 0 or n == 0 or (k != 2304 and k != 3072)) return error.BadExpertShape;
    for ([_]Ref{ input, weight, out }, 0..) |value, slot| e.setBuffer(value.buf, value.off, slot);
    e.setValue(DenseUnitsArgs{ .rows = rows, .n = n, .k = k }, 3);
    e.setPipeline(pipes[6]);
    e.dispatchGroups(mtl.Size.of((n + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
}

pub fn logitsUnits(p: *const Pipelines, e: mtl.ComputeEncoder, input: Ref, weight: Ref, out: Ref, rows: u32, vocab: u32) !void {
    const pipes = p.units orelse return error.NoTensorUnits;
    if (rows == 0 or vocab == 0) return error.BadExpertShape;
    for ([_]Ref{ input, weight, out }, 0..) |value, slot| e.setBuffer(value.buf, value.off, slot);
    e.setValue(LogitUnitsArgs{ .rows = rows, .vocab = vocab }, 3);
    e.setPipeline(pipes[5]);
    e.dispatchGroups(mtl.Size.of((vocab + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
}
