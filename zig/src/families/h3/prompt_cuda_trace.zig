//! The prompt encoder's per-operation captures: `encodeObserved` downloads the one device scratch's
//! buffers after each op of layer 0 into a `Capture`, and `check` replays every op's host reference on
//! the captured GPU input that op read — never on a replay's own chained output — comparing each
//! captured buffer against the float64 and float32-sum references as `match` gates whole runs.
//! Diagnostic output only: the selftest's whole gates alone decide failure. Captures exist only under
//! encodeObserved; the production `encode` instantiates encodeImpl without them.
const std = @import("std");
const cuda = @import("cuda");
const rules = @import("picture_prompt.zig");
const ref = @import("prompt_reference.zig");
const bf16_round = @import("bf16_round");
const Runner = @import("prompt_cuda.zig").Runner;

const D = rules.head_dim;

/// The captured buffers, in encodeImpl's op order; `po` and `pd` are the products' pre-rounding fp32
/// output, everything else the bf16 the next op reads.
pub const Op = enum { x0, xn1, q, k, v, qh, kh, att, po, x1, xn2, g, u, act, pd, x2, xd };
pub const ops = std.enums.values(Op);

/// Every op buffer's host copy; `set` marks the ones encodeObserved filled (no vision: `xd` stays unset).
pub const Capture = struct {
    bufs: [ops.len][]align(@alignOf(f32)) u8,
    set: [ops.len]bool,

    /// The buffers sized for a `g`-geometry encode of `rows` rows.
    pub fn init(a: std.mem.Allocator, g: rules.Text, rows: usize) !Capture {
        const T = rows;
        const H = g.hidden;
        var c: Capture = .{ .bufs = undefined, .set = @splat(false) };
        for (ops, 0..) |op, i| {
            const bytes: usize = switch (op) {
                .po, .pd => T * H * 4,
                .q, .qh, .att => T * g.heads * D * 2,
                .k, .v, .kh => T * g.kv_heads * D * 2,
                .g, .u, .act => T * g.ffn * 2,
                else => T * H * 2,
            };
            c.bufs[i] = try a.alignedAlloc(u8, .of(f32), bytes);
        }
        return c;
    }

    /// The buffer after `op`, downloaded once the stream's launches above it have landed.
    pub fn take(c: *Capture, r: *Runner, op: Op, ptr: u64) !void {
        try r.stream.synchronize();
        const buf = c.bufs[@backingInt(op)];
        var src: cuda.DeviceBuffer = .{ .d = &r.driver, .ptr = ptr, .len = buf.len };
        try src.download(0, buf);
        c.set[@backingInt(op)] = true;
    }

    /// A bf16 buffer's elements as fp32 values, freshly allocated.
    fn vals(c: *const Capture, a: std.mem.Allocator, op: Op) ![]f32 {
        const b = std.mem.bytesAsSlice(u16, c.bufs[@backingInt(op)]);
        const out = try a.alloc(f32, b.len);
        for (out, b) |*o, t| o.* = bf16_round.value(t);
        return out;
    }

    /// A raw fp32 product's values as h3p_add reads them: each element rounded to bf16 first.
    fn next(c: *const Capture, a: std.mem.Allocator, op: Op) ![]f32 {
        const sums = std.mem.bytesAsSlice(f32, c.bufs[@backingInt(op)]);
        const out = try a.alloc(f32, sums.len);
        for (out, sums) |*o, v| o.* = @floatCast(bf16_round.bf16(v));
        return out;
    }

    /// A captured buffer's bits for `dist`: bf16 as stored; a raw product rounded to the bf16 the
    /// residual add's first rounding made of it.
    fn bits(c: *const Capture, a: std.mem.Allocator, op: Op) ![]u16 {
        const bytes = c.bufs[@backingInt(op)];
        if (op == .po or op == .pd) {
            const sums = std.mem.bytesAsSlice(f32, bytes);
            const out = try a.alloc(u16, sums.len);
            for (out, sums) |*o, v| o.* = bf16_round.bits(v);
            return out;
        }
        return a.dupe(u16, std.mem.bytesAsSlice(u16, bytes));
    }
};

/// The distance of a captured buffer from a reference's fp32 values, as h3_image_match.match's.
const Dist = struct { cosine: f64, relative: f64, equal: f64 };

fn dist(ours: []const u16, theirs: []const f32) Dist {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var dd: f64 = 0;
    var equal: usize = 0;
    for (ours, theirs) |o, t| {
        const x: f64 = bf16_round.value(o);
        dot += x * t;
        aa += x * x;
        bb += @as(f64, t) * t;
        dd += (x - t) * (x - t);
        equal += @intFromBool(o == bf16_round.bits(t));
    }
    return .{ .cosine = dot / @sqrt(@max(aa * bb, 1e-300)), .relative = @sqrt(dd / @max(bb, 1e-300)), .equal = @as(f64, @floatFromInt(equal)) / @as(f64, @floatFromInt(ours.len)) };
}

/// `vals` (bf16 values held in fp32) as their bf16 bits.
fn bf16s(a: std.mem.Allocator, vals: []const f32) ![]u16 {
    const out = try a.alloc(u16, vals.len);
    for (out, vals) |*o, v| o.* = bf16_round.bits(v);
    return out;
}

/// Each op's reference output at f64 and at f32, on `a`'s lifetime (an arena's in `check`).
const Expected = struct { f64: [ops.len][]f32, f32: [ops.len][]f32 };

/// Every op of layer 0 replayed on the captured buffer that op reads: x0 gathers embed and splices
/// `v`'s rows; products read the captured normed or activation rows; heads read the captured rounded
/// products; attention reads the captured turned heads; each residual adds the captured raw product
/// rounded to bf16, and DeepStack's rows land on the captured x2. Only `xd` is left unfilled without
/// a vision. `l`, `v`, `embed`, `cos`, `sin` are the reference's fp32 weights, vision and tables.
fn expected(
    a: std.mem.Allocator,
    cap: *const Capture,
    g: rules.Text,
    l: ref.Layer,
    v: ?ref.Vision,
    ids: []const u32,
    embed: []const f32,
    cos: []const f32,
    sin: []const f32,
) !Expected {
    const T = ids.len;
    const H = g.hidden;
    const qn = g.heads * D;
    const kvn = g.kv_heads * D;
    var ar_state = std.heap.ArenaAllocator.init(a);
    defer ar_state.deinit();
    const ar = ar_state.allocator();

    var out: Expected = undefined;
    inline for (.{ f64, f32 }) |S| {
        const e = if (S == f64) &out.f64 else &out.f32;
        // embed gather and the vision splice produce x0
        e[@backingInt(Op.x0)] = try a.alloc(f32, T * H);
        for (ids, 0..) |id, r| @memcpy(e[@backingInt(Op.x0)][r * H ..][0..H], embed[id * H ..][0..H]);
        if (v) |vis| for (vis.pads, 0..) |at, i| @memcpy(e[@backingInt(Op.x0)][at * H ..][0..H], vis.rows[i * H ..][0..H]);
        e[@backingInt(Op.xn1)] = try a.alloc(f32, T * H);
        ref.norm(S, try cap.vals(ar, .x0), l.norm1, e[@backingInt(Op.xn1)], H, g.eps, .qwen);
        const xn = try cap.vals(ar, .xn1);
        e[@backingInt(Op.q)] = try a.alloc(f32, T * qn);
        try ref.gemm(S, xn, l.q, null, e[@backingInt(Op.q)], T, qn, H);
        e[@backingInt(Op.k)] = try a.alloc(f32, T * kvn);
        try ref.gemm(S, xn, l.k, null, e[@backingInt(Op.k)], T, kvn, H);
        e[@backingInt(Op.v)] = try a.alloc(f32, T * kvn);
        try ref.gemm(S, xn, l.v, null, e[@backingInt(Op.v)], T, kvn, H);
        // heads on the GPU's own rounded q/k: in place on a copy, the uploaded rotary tables
        e[@backingInt(Op.qh)] = try cap.vals(a, .q);
        ref.heads(S, e[@backingInt(Op.qh)], g.heads, qn, D, l.q_norm, .{ cos, sin }, T, g.eps, .qwen);
        e[@backingInt(Op.kh)] = try cap.vals(a, .k);
        ref.heads(S, e[@backingInt(Op.kh)], g.kv_heads, kvn, D, l.k_norm, .{ cos, sin }, T, g.eps, .qwen);
        e[@backingInt(Op.att)] = try a.alloc(f32, T * qn);
        try ref.attention(S, ar, try cap.vals(ar, .qh), .{ .count = g.heads, .row = qn, .head = D }, try cap.vals(ar, .kh), try cap.vals(ar, .v), .{ .count = g.kv_heads, .row = kvn, .head = D }, e[@backingInt(Op.att)], T, true);
        e[@backingInt(Op.po)] = try a.alloc(f32, T * H);
        try ref.gemm(S, try cap.vals(ar, .att), l.o, null, e[@backingInt(Op.po)], T, H, qn);
        // the residual reads x as captured and the product's own bf16 rounding, as h3p_add does
        e[@backingInt(Op.x1)] = try cap.vals(a, .x0);
        ref.add(e[@backingInt(Op.x1)], try cap.next(ar, .po));
        e[@backingInt(Op.xn2)] = try a.alloc(f32, T * H);
        ref.norm(S, try cap.vals(ar, .x1), l.norm2, e[@backingInt(Op.xn2)], H, g.eps, .qwen);
        const xn2 = try cap.vals(ar, .xn2);
        e[@backingInt(Op.g)] = try a.alloc(f32, T * g.ffn);
        try ref.gemm(S, xn2, l.gate, null, e[@backingInt(Op.g)], T, g.ffn, H);
        e[@backingInt(Op.u)] = try a.alloc(f32, T * g.ffn);
        try ref.gemm(S, xn2, l.up, null, e[@backingInt(Op.u)], T, g.ffn, H);
        e[@backingInt(Op.act)] = try a.alloc(f32, T * g.ffn);
        ref.glu(S, try cap.vals(ar, .g), try cap.vals(ar, .u), e[@backingInt(Op.act)], T, g.ffn, g.ffn, g.ffn, .qwen);
        e[@backingInt(Op.pd)] = try a.alloc(f32, T * H);
        try ref.gemm(S, try cap.vals(ar, .act), l.down, null, e[@backingInt(Op.pd)], T, H, g.ffn);
        e[@backingInt(Op.x2)] = try cap.vals(a, .x1);
        ref.add(e[@backingInt(Op.x2)], try cap.next(ar, .pd));
        if (v) |vis| {
            e[@backingInt(Op.xd)] = try cap.vals(a, .x2);
            for (vis.pads, 0..) |at, i| ref.add(e[@backingInt(Op.xd)][at * H ..][0..H], vis.deepstack[0][i * H ..][0..H]);
        }
    }
    return out;
}

fn say(label: []const u8, ours: Dist, noise: Dist) void {
    std.debug.print("  diagnostic {s}: cosine {d:.7}, relative error {e:.2}, {d:.4} bit-equal; the same-input float32 replay's: {e:.2}, {d:.4}\n", .{ label, ours.cosine, ours.relative, ours.equal, noise.relative, noise.equal });
}

/// One diagnostic line a captured op: its distance from the float64 replay beside the float32 replay's,
/// so a divergence names the op that owns it. Prints only; the selftest's whole gates decide failure.
pub fn check(
    a: std.mem.Allocator,
    cap: *const Capture,
    g: rules.Text,
    l: ref.Layer,
    v: ?ref.Vision,
    ids: []const u32,
    embed: []const f32,
    cos: []const f32,
    sin: []const f32,
) !void {
    var ar_state = std.heap.ArenaAllocator.init(a);
    defer ar_state.deinit();
    const ar = ar_state.allocator();
    const e = try expected(ar, cap, g, l, v, ids, embed, cos, sin);
    const labels = std.enums.EnumArray(Op, []const u8).init(.{
        .x0 = "embed gather and vision splice",
        .xn1 = "input norm",
        .q = "q product and round",
        .k = "k product and round",
        .v = "v product and round",
        .qh = "q heads: norm and rotary",
        .kh = "k heads: norm and rotary",
        .att = "attention",
        .po = "o product rounded to BF16",
        .x1 = "attention residual add",
        .xn2 = "FFN norm",
        .g = "gate product and round",
        .u = "up product and round",
        .act = "SwiGLU",
        .pd = "down product rounded to BF16",
        .x2 = "FFN residual add",
        .xd = "DeepStack scatter add",
    });
    for (ops) |op| {
        if (!cap.set[@backingInt(op)]) continue;
        const ours = dist(try cap.bits(ar, op), e.f64[@backingInt(op)]);
        const noise = dist(try bf16s(ar, e.f32[@backingInt(op)]), e.f64[@backingInt(op)]);
        say(labels.get(op), ours, noise);
    }
}
