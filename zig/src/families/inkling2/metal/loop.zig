//! Inkling 2 Mini cross-pass messages and message gates: real Metal dispatches.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
pub const plan = @import("inkling2_loop_plan");

pub const hidden: u32 = plan.hidden;
pub const rank: u32 = plan.rank;
pub const message_entries: u32 = plan.message_entries;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const RowsArgs = plan.RowsArgs;

/// Bytes of gate scratch a row: normalized [h | m] in BF16, then zeta's 64 BF16.
const gate_row_bytes: usize = (2 * hidden + rank) * 2;

pub const Pipelines = struct {
    messages: mtl.Pipeline,
    gate: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.loop);
    }
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var messages = try mtl.Pipeline.init(device, lib, plan.Kind.messages.symbol(), false);
        errdefer messages.deinit();
        const gate = try mtl.Pipeline.init(device, lib, plan.Kind.gates.symbol(), false);
        errdefer gate.deinit();
        return .{ .messages = messages, .gate = gate };
    }
    pub fn deinit(p: *Pipelines) void {
        p.gate.deinit();
        p.messages.deinit();
        p.* = undefined;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
}

/// Output layout is [token][12 core layers][q,k,v,mlp][H], all binary32.
pub fn crossPassMessageAttention(p: *const Pipelines, e: mtl.ComputeEncoder, bank0: Ref, bank1: Ref, q_mlp: Ref, kv: Ref, out: Ref, rows: u32) void {
    std.debug.assert(rows > 0 and rows <= 1024);
    e.setPipeline(p.messages);
    bind(e, &.{ bank0, bank1, q_mlp, kv, out });
        e.setValue(GateArgs{ .rows = rows, .stride = hidden, .normed = 0 }, 5);
    e.dispatchGroups(mtl.Size.of(rows * message_entries, 1, 1), mtl.Size.of(256, 1, 1));
}

const GateArgs = extern struct { rows: u32, stride: u32, normed: u32 };

/// One controller tuple; h, m and output are [token][H] binary32 tensors.
pub fn gates(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref, rows: u32) void {
    gatesStrided(p, e, h, m, down, forget, gain, out, rows, hidden);
}

/// As gates, with row r's m at `m_stride` binary32 values from row r - 1's.
pub fn gatesStrided(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref, rows: u32, m_stride: u32) void {
    gatesEach(p, e, &.{.{ .h = h, .m = m, .down = down, .forget = forget, .gain = gain, .out = out }}, null, rows, m_stride);
}

/// A norm fused after the tuple's blend: weight and a BF16 output, the row the blend leaves in FP32.
pub const Norm = struct { weight: Ref, out: Ref };

/// Controller tuples one call gates together: a visit's three attention roles.
pub const max_gates = 3;
pub const Gate = struct { h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref };

/// gatesStrided for each tuple a launch at a time — every launch is the whole gate plus its norm, so the
/// launches need no barrier between them; the tuples' outputs must be distinct.
pub fn gatesEach(p: *const Pipelines, e: mtl.ComputeEncoder, items: []const Gate, norms: ?[]const Norm, rows: u32, m_stride: u32) void {
    std.debug.assert(rows > 0 and rows <= 1024 and items.len > 0 and items.len <= max_gates);
    std.debug.assert(norms == null or norms.?.len == items.len);
    for (items, 0..) |g, i| {
        const normed = norms != null;
        const n: Norm = if (norms) |ns| ns[i] else .{ .weight = g.gain, .out = g.out };
        e.setPipeline(p.gate);
        bind(e, &.{ g.h, g.m, g.down, g.forget, g.gain, g.out, n.weight, n.out });
        e.setValue(GateArgs{ .rows = rows, .stride = m_stride, .normed = @intFromBool(normed) }, 8);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }
}
