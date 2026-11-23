//! Mini DSpark's greedy cohort: each launch for `rows` streams against the solo launch it replaces, run a stream at a
//! time on the same inputs. Shape rows are streams; k is the depth row (step, pick) or the first base row (logits,
//! pack).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_dspark_host;
const controls = @import("metal_inkling2_dspark_cohort_controls.zig");

const RANK = host.rank;
const RNN = host.rnn_width;
const H = host.hidden;
const Q = host.query_width;
const KV = host.kv_width;
const V = host.vocab;
const BLOCK = host.block_rows;
const MOST = host.cohort_streams;

const Kind = enum { qkv, step, pick, logits, pack };

fn noSums(_: c.Shape) usize {
    return 1;
}
fn hr(b: rig_.Ref) host.Ref {
    return .{ .buf = b.buf, .off = b.off };
}
fn plus(b: rig_.Ref, bytes: usize) host.Ref {
    return .{ .buf = b.buf, .off = b.off + bytes };
}

/// One cohort kernel's probe; `replaced` runs the solo kernel per stream into twin buffers laid out as the outputs.
pub const Probe = struct {
    kind: Kind,
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    cohort: host.Cohort,
    chain: host.Chain,
    pipes: host.Pipelines,
    control: ?g.Gate,
    shape: c.Shape = undefined,
    ins: [6]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    twin: [3]rig_.Ref = undefined,
    sizes: [3]usize = @splat(0),
    want: std.ArrayList(u8) = .empty,
    active: [host.steps]u32 = @splat(0),
    msg: [160]u8 = undefined,

    fn mutation(kind: Kind) controls.Mutation {
        return switch (kind) {
            .qkv => .qkv_anchor,
            .step => .step_token,
            .pick => .pick_nan_row,
            .logits => .logits_row,
            .pack => .pack_prefix,
        };
    }
    fn label(kind: Kind, control: bool) []const u8 {
        if (control) return mutation(kind).name();
        return switch (kind) {
            inline else => |k| "metal inkling2 dspark cohort_" ++ @tagName(k) ++ " tf_ink2_dspark_cohort_" ++ @tagName(k),
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try controls.apply(a, if (control) mutation(kind) else .none);
        defer a.free(text);
        var cohort = try host.Cohort.load(device, text);
        errdefer cohort.deinit();
        var chain = try host.Chain.load(device, @import("inkling2_metal").inkling2_dspark);
        errdefer chain.deinit();
        return .{ .kind = kind, .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .cohort = cohort, .chain = chain, .pipes = try host.Pipelines.load(device), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Probe) void {
        t.want.deinit(t.a);
        t.rig.deinit();
        t.arena.deinit();
        t.pipes.deinit();
        t.chain.deinit();
        t.cohort.deinit();
    }

    pub fn kernel(t: *Probe) g.Kernel {
        const served: []const c.Shape = switch (t.kind) {
            .qkv => &.{ .{ .rows = 1, .n = Q, .k = 0 }, .{ .rows = 3, .n = Q, .k = 0 }, .{ .rows = MOST, .n = Q, .k = 0 } },
            .step => &.{ .{ .rows = 1, .n = RNN, .k = 0 }, .{ .rows = 5, .n = RNN, .k = 3 }, .{ .rows = MOST, .n = RNN, .k = 6 } },
            .pick => &.{ .{ .rows = 1, .n = V, .k = 0 }, .{ .rows = 3, .n = V, .k = 2 }, .{ .rows = MOST, .n = V, .k = 6 } },
            // the kernel's three widths: up to 4, 8 and 16 streams, each from a later depth's base rows
            .logits => &.{ .{ .rows = 3, .n = V, .k = 2 }, .{ .rows = 7, .n = V, .k = 5 }, .{ .rows = MOST, .n = V, .k = 1 } },
            .pack => &.{ .{ .rows = 1, .n = H, .k = 0 }, .{ .rows = 6, .n = H, .k = 4 }, .{ .rows = MOST, .n = H, .k = 9 } },
        };
        const nan: []const u8 = switch (t.kind) {
            .qkv => "its bytes are tf_ink2_dspark_block_qkv's on each block at its own anchor",
            .step => "its bytes are tf_ink2_dspark_recurrent's on each stream's state, kappa row and held token",
            .pick => "its picks are tf_ink2_dspark_chain_pick's on each stream's row",
            .logits => "its bytes are tf_ink2_dspark_logits' on each stream's base and lambda rows",
            .pack => "its rows are copies of each stream's draft rows, NaNs included",
        };
        return .{ .ctx = t, .vt = &vt, .control = t.control, .spec = .{
            .name = label(t.kind, t.control != null),
            .accum = .none,
            .depth = noSums,
            .out = if (t.kind == .pick) .raw else .bf16,
            .nan = .{ .stated = nan },
            .served = served,
            .replaces = "the solo launch",
        } };
    }
    const vt = g.hooks(Probe);

    fn bf16(t: *Probe, n: usize, rng: std.Random) ![]u16 {
        const bits = try t.arena.allocator().alloc(u16, n);
        for (bits) |*b| b.* = c.toBf16(rng.float(f32) * 0.5 - 0.25);
        return bits;
    }
    fn in(t: *Probe, i: usize, data: anytype) !void {
        t.ins[i] = try t.rig.input(std.mem.sliceAsBytes(data));
    }
    fn outputs(t: *Probe, sizes: []const usize) !void {
        for (sizes, 0..) |bytes, i| {
            t.sizes[i] = bytes;
            t.outs[i] = try t.rig.output(bytes);
            t.twin[i] = try t.rig.scratch(bytes);
        }
    }

    pub fn prepare(t: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n: u32 = switch (t.kind) {
            .qkv => Q,
            .step => RNN,
            .pick, .logits => V,
            .pack => H,
        };
        const depth_ok = switch (t.kind) {
            .step, .pick => shape.k < host.steps,
            .logits => shape.k + shape.rows <= host.steps * MOST,
            .pack => shape.k < host.steps * MOST,
            .qkv => shape.k == 0,
        };
        if (shape.n != n or shape.rows == 0 or shape.rows > MOST or !depth_ok) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        t.shape = shape;
        t.sizes = @splat(0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const s: usize = shape.rows;
        const a = t.arena.allocator();
        switch (t.kind) {
            .qkv => {
                const q_raw = try t.bf16(s * BLOCK * Q, rng);
                if (fill == .nonfinite) q_raw[Q + 5] = c.toBf16(std.math.nan(f32));
                const anchors = try a.alloc(u32, s);
                for (anchors, 0..) |*p, i| p.* = @intCast(1 + 977 * i + rng.uintLessThan(u32, 500));
                try t.in(0, q_raw);
                try t.in(1, try t.bf16(s * BLOCK * KV, rng));
                try t.in(2, try t.bf16(s * BLOCK * KV, rng));
                try t.in(3, try t.bf16(128, rng));
                try t.in(4, try t.bf16(128, rng));
                try t.in(5, anchors);
                try t.outputs(&.{ s * BLOCK * Q * 2, s * BLOCK * KV * 2, s * BLOCK * KV * 2 });
            },
            .step => {
                const state = try t.bf16(s * RANK, rng);
                if (fill == .nonfinite) state[0] = c.toBf16(std.math.nan(f32));
                const kappa = try a.alloc(f32, s * host.steps * RNN);
                for (kappa) |*v| v.* = rng.float(f32) * 0.5 - 0.25;
                // Markov rows of tokens 0 and 1 differ; each stream's held token at the depth differs from stream 0's.
                const held = try a.alloc(u32, s * 8);
                for (held, 0..) |*h, i| h.* = @intFromBool((i / 8) % 2 == 1);
                try t.in(0, state);
                try t.in(1, kappa);
                try t.in(2, try t.bf16(RNN * 5120, rng));
                try t.in(3, try t.bf16(2 * RANK, rng));
                try t.in(4, held);
                try t.outputs(&.{ s * 3 * RANK * 2, s * RANK * 2, s * RANK * 2 });
            },
            .pick => {
                const logits = try a.alloc(u16, s * V);
                // few distinct values, so each row's greatest is tied many times
                for (logits) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 4 - 8);
                // a NaN at id 0 of a later row only: that row picks 0, the others do not
                if (s > 1) logits[V] = c.toBf16(std.math.nan(f32));
                if (fill == .nonfinite) logits[V / 3] = c.toBf16(std.math.inf(f32));
                try t.in(0, logits);
                try t.outputs(&.{s * 8 * @sizeOf(u32)});
            },
            .logits => {
                const base = try t.bf16((shape.k + s) * V, rng);
                if (fill == .nonfinite) base[shape.k * V + 9] = c.toBf16(std.math.nan(f32));
                try t.in(0, base);
                try t.in(1, try t.bf16(V * RANK, rng));
                try t.in(2, try t.bf16(V, rng));
                try t.in(3, try t.bf16(s * RANK, rng));
                try t.outputs(&.{s * V * 2});
            },
            .pack => {
                const x = try t.bf16(s * BLOCK * H, rng);
                if (fill == .nonfinite) x[H + 3] = c.toBf16(std.math.nan(f32));
                // deepest first, one fewer stream at each later depth
                var total: u32 = 0;
                for (&t.active, 0..) |*v, j| {
                    v.* = @intCast(s -| j);
                    total += v.*;
                }
                if (shape.k >= total) return false;
                try t.in(0, x);
                try t.outputs(&.{(total - @as(u32, @intCast(shape.k))) * H * 2});
            },
        }
        return true;
    }

    fn packArgs(t: *const Probe) host.PackArgs {
        var total: u32 = 0;
        for (t.active) |v| total += v;
        return .{ .active = t.active, .first = @intCast(t.shape.k), .count = total - @as(u32, @intCast(t.shape.k)) };
    }

    pub fn run(t: *Probe) !void {
        const s: u32 = @intCast(t.shape.rows);
        const k: u32 = @intCast(t.shape.k);
        const pass = t.rig.begin();
        switch (t.kind) {
            .qkv => host.cohortQkv(&t.cohort, pass.e, hr(t.ins[0]), hr(t.ins[1]), hr(t.ins[2]), hr(t.ins[3]), hr(t.ins[4]), hr(t.outs[0]), hr(t.outs[1]), hr(t.outs[2]), hr(t.ins[5]), s),
            .step => host.cohortStep(&t.cohort, pass.e, hr(t.ins[0]), hr(t.ins[1]), hr(t.ins[2]), hr(t.ins[3]), hr(t.outs[0]), hr(t.outs[1]), hr(t.outs[2]), hr(t.ins[4]), k, s),
            .pick => host.cohortPick(&t.cohort, pass.e, hr(t.ins[0]), hr(t.outs[0]), k, s),
            .logits => host.cohortLogits(&t.cohort, pass.e, hr(t.ins[0]), hr(t.ins[1]), hr(t.ins[2]), hr(t.ins[3]), hr(t.outs[0]), k, s),
            .pack => host.cohortPack(&t.cohort, pass.e, hr(t.ins[0]), hr(t.outs[0]), t.packArgs()),
        }
        try t.rig.finish(pass);
    }

    /// The solo launches, a stream at a time, into the twins; returns their bytes in output order.
    pub fn replaced(t: *Probe) ![]const u8 {
        for (t.sizes, t.twin) |bytes, twin| if (bytes > 0) @memset(t.rig.bytes(twin, bytes), 0xff);
        const k: u32 = @intCast(t.shape.k);
        const pass = t.rig.begin();
        for (0..t.shape.rows) |si| {
            const s: u32 = @intCast(si);
            switch (t.kind) {
                .qkv => {
                    const anchors = std.mem.bytesAsSlice(u32, t.rig.bytes(t.ins[5], t.shape.rows * 4));
                    const rows = s * BLOCK;
                    host.blockQkv(&t.pipes, pass.e, plus(t.ins[0], rows * Q * 2), plus(t.ins[1], rows * KV * 2), plus(t.ins[2], rows * KV * 2), hr(t.ins[3]), hr(t.ins[4]), plus(t.twin[0], rows * Q * 2), plus(t.twin[1], rows * KV * 2), plus(t.twin[2], rows * KV * 2), anchors[s]);
                },
                .step => {
                    const held = std.mem.bytesAsSlice(u32, t.rig.bytes(t.ins[4], t.shape.rows * 8 * 4));
                    host.recurrentStep(&t.pipes, pass.e, plus(t.ins[0], s * RANK * 2), plus(t.ins[1], (s * host.steps + k) * RNN * 4), hr(t.ins[2]), hr(t.ins[3]), plus(t.twin[0], s * 3 * RANK * 2), plus(t.twin[1], s * RANK * 2), plus(t.twin[2], s * RANK * 2), held[s * 8 + k]);
                },
                .pick => host.chainPick(&t.chain, pass.e, plus(t.ins[0], @as(usize, s) * V * 2), plus(t.twin[0], s * 8 * 4), k),
                .logits => host.finalLogits(&t.pipes, pass.e, plus(t.ins[0], (@as(usize, k) + s) * V * 2), hr(t.ins[1]), hr(t.ins[2]), plus(t.ins[3], s * RANK * 2), plus(t.twin[0], @as(usize, s) * V * 2)),
                .pack => {},
            }
        }
        try t.rig.finish(pass);
        t.want.clearRetainingCapacity();
        if (t.kind == .pack) {
            // the solo path's packed rows: depth j's draft row j + 1 of each of its active streams, from row `first` on
            var row: usize = 0;
            for (t.active, 0..) |streams, j| for (0..streams) |si| {
                if (row >= t.shape.k) try t.want.appendSlice(t.a, t.rig.bytes(t.ins[0], (si * BLOCK + j + 2) * H * 2)[(si * BLOCK + j + 1) * H * 2 ..]);
                row += 1;
            };
            return t.want.items;
        }
        for (t.sizes, t.twin) |bytes, twin| if (bytes > 0) try t.want.appendSlice(t.a, t.rig.bytes(twin, bytes));
        return t.want.items;
    }

    pub fn count(t: *Probe) usize {
        var n: usize = 0;
        for (t.sizes) |bytes| n += bytes;
        return n;
    }
    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Probe, _: g.Gate) !?[]const u8 {
        const got = try t.a.dupe(u8, try t.rig.outputBytes());
        defer t.a.free(got);
        const want = try t.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&t.msg, "differs from the solo launches from byte {d}", .{at});
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [2 * std.enums.values(Kind).len]Probe = undefined,
    used: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.used]) |*t| t.deinit();
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |control| for (std.enums.values(Kind)) |kind| {
            if (only != null and std.mem.indexOf(u8, Probe.label(kind, control), only.?) == null) continue;
            s.probes[s.used] = try Probe.init(s.a, s.device, s.queue, kind, control);
            s.used += 1;
            try s.kernels.append(s.a, s.probes[s.used - 1].kernel());
        };
    }
};
