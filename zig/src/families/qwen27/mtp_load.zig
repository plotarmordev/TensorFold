//! The MTP layer's bf16 `mtp.*` tensors (the publisher's full-precision release) under the target's 4-bit names.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("core").checkpoint_metal;
const host4 = @import("core").affine4_host;

/// The bf16 tensors' namespace in the checkpoint; the quantized ones take the target's `language_model.`.
pub const source = "bf16.";
const target = "language_model.";

/// Matrices stay in their files until quantized; the norms load whole.
pub fn deferred(_: []const u8, rank: usize) bool {
    return rank == 2;
}

const norms = [_][]const u8{ "mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden", "mtp.norm", "mtp.layers.0.input_layernorm", "mtp.layers.0.post_attention_layernorm", "mtp.layers.0.self_attn.q_norm", "mtp.layers.0.self_attn.k_norm" };
const matrices = [_][]const u8{ "mtp.fc", "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj", "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.mlp.gate", "mtp.layers.0.mlp.shared_expert_gate", "mtp.layers.0.mlp.shared_expert.gate_proj", "mtp.layers.0.mlp.shared_expert.up_proj", "mtp.layers.0.mlp.shared_expert.down_proj" };
const projections = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// Every `mtp.*` matrix as MLX 4-bit g64, the `experts` stacked as `switch_mlp`; the norms' gains as 1 + w.
pub fn stage(a: std.mem.Allocator, device: mtl.Device, ck: *cp.Checkpoint, experts: usize) !void {
    var name: [96]u8 = undefined;
    var to: [96]u8 = undefined;
    for (norms) |n| {
        const t = try ck.get(try std.fmt.bufPrint(&name, source ++ "{s}.weight", .{n}));
        if (t.dtype != .bf16 or t.rank != 1) return error.BadMtpTensor;
        // HF stores Qwen's zero-centred gains; the kernels multiply by the gain, as MLX's converted files store it
        const w: [*]u16 = @ptrCast(@alignCast(t.buffer.contents() + t.offset));
        for (w[0..t.shape[0]]) |*v| v.* = host4.bf16of(host4.f32of(v.*) + 1);
        try ck.set(try std.fmt.bufPrint(&to, target ++ "{s}.weight", .{n}), t);
    }
    for (matrices) |m| {
        const from = try std.fmt.bufPrint(&name, source ++ "{s}.weight", .{m});
        try quantize(a, device, ck, &.{from}, try std.fmt.bufPrint(&to, target ++ "{s}", .{m}));
    }
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const names = try arena.allocator().alloc([]const u8, experts);
    for (projections) |p| {
        for (names, 0..) |*n, e| n.* = try std.fmt.allocPrint(arena.allocator(), source ++ "mtp.layers.0.mlp.experts.{d}.{s}.weight", .{ e, p });
        try quantize(a, device, ck, names, try std.fmt.bufPrint(&to, target ++ "mtp.layers.0.mlp.switch_mlp.{s}", .{p}));
    }
}

/// The bf16 [n, k] matrices `from`, stacked when several, as `module`'s weight, scales and biases in one owned buffer.
fn quantize(a: std.mem.Allocator, device: mtl.Device, ck: *cp.Checkpoint, from: []const []const u8, module: []const u8) !void {
    const first = ck.deferred.get(from[0]) orelse return error.MissingTensor;
    const n = first.shape[0];
    const k = first.shape[1];
    if (k % host4.group != 0) return error.BadMtpTensor;
    const values = from.len * n * k;
    const words = values / 2;
    const groups = values / host4.group * 2;
    const at = [3]usize{ 0, std.mem.alignForward(usize, words, cp.alignment), std.mem.alignForward(usize, words, cp.alignment) + std.mem.alignForward(usize, groups, cp.alignment) };
    const buffer = try device.buffer(at[2] + groups, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    ck.own(buffer, at[2] + groups) catch |err| {
        buffer.deinit();
        return err;
    };
    const base = buffer.contents();
    const src = try a.alloc(u16, n * k);
    defer a.free(src);
    const job = Job{
        .src = src,
        .k = k,
        .words = @as([*]u32, @ptrCast(@alignCast(base)))[0 .. words / 4],
        .scales = @as([*]u16, @ptrCast(@alignCast(base + at[1])))[0 .. groups / 2],
        .biases = @as([*]u16, @ptrCast(@alignCast(base + at[2])))[0 .. groups / 2],
    };
    for (from, 0..) |name, i| {
        const d = ck.deferred.get(name) orelse return error.MissingTensor;
        if (d.dtype != .bf16 or d.rank != 2 or d.shape[0] != n or d.shape[1] != k) return error.BadMtpTensor;
        try ck.read(name, 0, std.mem.sliceAsBytes(src));
        try job.rows(a, i * n, n);
    }
    var name: [128]u8 = undefined;
    const stacked = from.len > 1;
    for ([_][]const u8{ ".weight", ".scales", ".biases" }, at, [_]usize{ k / 8, k / host4.group, k / host4.group }, [_]cp.DType{ .u32, .bf16, .bf16 }) |suffix, offset, cols, dtype| {
        var t = cp.Tensor{ .buffer = buffer, .offset = offset, .bytes = from.len * n * cols * dtype.size(), .dtype = dtype, .rank = if (stacked) 3 else 2 };
        const shape = if (stacked) [3]usize{ from.len, n, cols } else [3]usize{ n, cols, 1 };
        @memcpy(t.shape[0..3], &shape);
        try ck.set(try std.fmt.bufPrint(&name, "{s}{s}", .{ module, suffix }), t);
    }
}

/// One bf16 matrix (`src`) quantized into rows [row, row + n) of a stack's words, scales and biases.
const Job = struct {
    src: []const u16,
    k: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,

    fn rows(j: Job, a: std.mem.Allocator, row: usize, n: usize) !void {
        const threads = @min(std.Thread.getCpuCount() catch 1, 16);
        const handles = try a.alloc(std.Thread, threads);
        defer a.free(handles);
        var failed = std.atomic.Value(bool).init(false);
        {
            const per = (n + threads - 1) / threads;
            var spawned: usize = 0;
            defer for (handles[0..spawned]) |h| h.join();
            while (spawned * per < n) : (spawned += 1)
                handles[spawned] = try std.Thread.spawn(.{}, run, .{ j, row, spawned * per, @min(n, (spawned + 1) * per), &failed });
        }
        if (failed.load(.monotonic)) return error.NonFiniteMtpWeight;
    }

    fn run(j: Job, row: usize, from: usize, to: usize, failed: *std.atomic.Value(bool)) void {
        const g = j.k / host4.group;
        var values: [host4.group]f32 = undefined;
        for (from..to) |r| for (0..g) |i| {
            for (&values, j.src[r * j.k + i * host4.group ..][0..host4.group]) |*v, b| v.* = host4.f32of(b);
            const at = (row + r) * g + i;
            if (host4.quantizeMlx(host4.group, &values, j.words[at * 8 ..][0..8], j.scales[at..][0..1], j.biases[at..][0..1]) != null) failed.store(true, .monotonic);
        };
    }
};
