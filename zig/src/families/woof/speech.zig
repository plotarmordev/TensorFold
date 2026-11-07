//! Woof's speech model: 8-layer 448-wide causal transformer over voice prefix, text and BOS, one latent frame a step.
const std = @import("std");
const ops = @import("ops.zig");
const Loader = @import("weights.zig").Loader;
const Randn = @import("rng.zig").Randn;
const Linear = ops.Linear;

pub const layers = 8;
pub const d = 448;
const heads = 7;
const dh = d / heads;
const ff = 1216;
const hw = 320;
const depth = 4;
pub const latent = 32;
pub const voice_rows = 16;
pub const max_len = 1100;
const vocab = 95;
const eps = std.math.floatEps(f32);

const Layer = struct { qkv: Linear, o: Linear, up: Linear, down: Linear };

pub const Speech = struct {
    layer: [layers]Layer,
    ce: Linear,
    ada: Linear,
    blocks: [depth][2]Linear,
    lat_in: Linear,
    h_inp: Linear,
    h_out: Linear,
    text_emb: []const f32,
    bos: []const f32,
    cos: []f32,
    sin: []f32,
    /// Per layer, K then V, (max_len, d) each.
    kv: []f32,
    x: []f32,
    xn: []f32,
    big: []f32,
    mid: []f32,
    scores: []f32,

    pub fn load(l: *Loader) !Speech {
        var s: Speech = undefined;
        for (&s.layer, 0..) |*y, i| y.* = .{
            .qkv = try l.linear(3 * d, d, "speech.layers.{d}.0", .{i}),
            .o = try l.linear(d, d, "speech.layers.{d}.1", .{i}),
            .up = try l.linear(2 * ff, d, "speech.layers.{d}.2", .{i}),
            .down = try l.linear(d, ff, "speech.layers.{d}.3", .{i}),
        };
        s.ce = try l.linear(hw + 1, d, "speech.ce", .{});
        s.ada = try l.linear(3 * hw * depth + 2 * hw, hw, "speech.ada", .{});
        for (&s.blocks, 0..) |*b, j| b.* = .{ try l.linear(2 * hw, hw, "speech.hblocks.{d}.0", .{j}), try l.linear(hw, 2 * hw, "speech.hblocks.{d}.1", .{j}) };
        s.lat_in = try l.linear(d, latent, "speech.lat_in", .{});
        s.h_inp = try l.linear(hw, latent, "speech.h_inp", .{});
        s.h_out = try l.linear(latent, hw, "speech.h_out", .{});
        s.text_emb = try l.floats(vocab * d, "speech.text_emb", .{});
        s.bos = try l.floats(d, "speech.bos", .{});
        const a = l.a;
        s.cos = try a.alloc(f32, max_len * dh);
        s.sin = try a.alloc(f32, max_len * dh);
        for (0..max_len) |p| for (0..dh / 2) |j| {
            const inv: f32 = 1 / std.math.pow(f32, 10000, @as(f32, @floatFromInt(2 * j)) / dh);
            const ang = @as(f32, @floatFromInt(p)) * inv;
            s.cos[p * dh + j] = @cos(ang);
            s.cos[p * dh + j + dh / 2] = @cos(ang);
            s.sin[p * dh + j] = @sin(ang);
            s.sin[p * dh + j + dh / 2] = @sin(ang);
        };
        s.kv = try a.alloc(f32, layers * 2 * max_len * d);
        s.x = try a.alloc(f32, max_len * d);
        s.xn = try a.alloc(f32, max_len * d);
        s.big = try a.alloc(f32, max_len * 2 * ff);
        s.mid = try a.alloc(f32, max_len * ff);
        s.scores = try a.alloc(f32, max_len);
        return s;
    }

    fn norm(s: *Speech, n: usize) void {
        for (0..n) |r| {
            const x = s.x[r * d ..][0..d];
            const inv = ops.rmsInv(x, eps);
            for (s.xn[r * d ..][0..d], x) |*y, v| y.* = v * inv;
        }
    }

    /// Layer i over rows s.x[0..n] at positions p0.., their K and V cached.
    fn block(s: *Speech, i: usize, n: usize, p0: usize) void {
        const y = s.layer[i];
        const k_cache = s.kv[(2 * i) * max_len * d ..][0 .. max_len * d];
        const v_cache = s.kv[(2 * i + 1) * max_len * d ..][0 .. max_len * d];
        s.norm(n);
        const qkv = s.big;
        ops.rows(y.qkv, s.xn, d, n, qkv, 3 * d, null);
        for (0..n) |r| {
            const p = p0 + r;
            const row = qkv[r * 3 * d ..];
            for (0..2 * heads) |h| ops.rope(row[h * dh ..][0..dh], s.cos[p * dh ..][0..dh], s.sin[p * dh ..][0..dh]);
            @memcpy(k_cache[p * d ..][0..d], row[d..][0..d]);
            @memcpy(v_cache[p * d ..][0..d], row[2 * d ..][0..d]);
        }
        const att = s.xn;
        const scale: f32 = 1 / @sqrt(@as(f32, dh));
        for (0..n) |r| {
            const p = p0 + r;
            for (0..heads) |h| {
                const q = qkv[r * 3 * d + h * dh ..][0..dh];
                const sc = s.scores[0 .. p + 1];
                for (sc, 0..) |*v, t| v.* = ops.dot(k_cache[t * d + h * dh ..][0..dh], q) * scale;
                ops.softmax(sc);
                const out = att[r * d + h * dh ..][0..dh];
                @memset(out, 0);
                for (sc, 0..) |w, t| for (out, v_cache[t * d + h * dh ..][0..dh]) |*o, v| {
                    o.* += w * v;
                };
            }
        }
        ops.rows(y.o, att, d, n, s.x, d, s.x);
        s.norm(n);
        ops.rows(y.up, s.xn, d, n, s.big, 2 * ff, null);
        for (0..n) |r| {
            const gu = s.big[r * 2 * ff ..];
            for (s.mid[r * ff ..][0..ff], gu[0..ff], gu[ff..][0..ff]) |*m, g, u| m.* = ops.silu(g) * u;
        }
        ops.rows(y.down, s.mid, ff, n, s.x, d, s.x);
    }

    /// The flow condition of row x and its end logit.
    fn cond(s: *Speech, x: []const f32, c: *[hw]f32) f32 {
        var xn: [d]f32 = undefined;
        const inv = ops.rmsInv(x, eps);
        for (&xn, x) |*y, v| y.* = v * inv;
        var out: [hw + 1]f32 = undefined;
        ops.vec(s.ce, &xn, &out);
        for (c, out[0..hw]) |*y, v| y.* = ops.silu(v);
        return out[hw];
    }

    /// One flow step: z = noise + head(cond, noise).
    fn head(s: *Speech, c: *const [hw]f32, noise: *const [latent]f32, z: *[latent]f32) void {
        var a: [3 * hw * depth + 2 * hw]f32 = undefined;
        ops.vec(s.ada, c, &a);
        var h: [hw]f32 = undefined;
        ops.vec(s.h_inp, noise, &h);
        var y: [hw]f32 = undefined;
        var u: [2 * hw]f32 = undefined;
        var o: [hw]f32 = undefined;
        for (s.blocks, 0..) |b, j| {
            const m = a[3 * hw * j ..];
            modulate(&y, &h, m);
            ops.vec(b[0], &y, &u);
            for (&u) |*v| v.* = ops.silu(v.*);
            ops.vec(b[1], &u, &o);
            for (&h, m[2 * hw ..][0..hw], o) |*x, g, v| x.* += g * v;
        }
        modulate(&y, &h, a[3 * hw * depth ..]);
        ops.vec(s.h_out, &y, z);
        for (z, noise) |*v, e| v.* += e;
    }

    fn modulate(y: *[hw]f32, h: *const [hw]f32, m: []const f32) void {
        y.* = h.*;
        ops.layerNorm(y, 1e-5);
        for (y, m[0..hw], m[hw..][0..hw]) |*v, shift, scale| v.* = shift + v.* * scale;
    }

    /// Latent frames for one piece (ids from text.encode) into out; returns the count.
    pub fn generate(s: *Speech, voice: []const f32, ids: []const u32, rng: *Randn, temp: f32, eos: f64, out: [][latent]f32) usize {
        const p0 = voice_rows + ids.len + 1;
        std.debug.assert(p0 < max_len);
        @memcpy(s.x[0 .. voice_rows * d], voice);
        for (ids, 0..) |id, r| @memcpy(s.x[(voice_rows + r) * d ..][0..d], s.text_emb[id * d ..][0..d]);
        @memcpy(s.x[(p0 - 1) * d ..][0..d], s.bos);
        for (0..layers) |i| s.block(i, p0, 0);
        @memmove(s.x[0..d], s.x[(p0 - 1) * d ..][0..d]);
        var c: [hw]f32 = undefined;
        var e = s.cond(s.x[0..d], &c);
        const eos_logit = @log(eos / (1 - eos));
        const frames = @min(out.len, max_len - p0);
        var noise: [latent]f32 = undefined;
        for (0..frames) |f| {
            if (f > 3 and e > eos_logit) return f;
            rng.fill(&noise);
            for (&noise) |*v| v.* *= temp;
            s.head(&c, &noise, &out[f]);
            ops.vec(s.lat_in, &out[f], s.x[0..d]);
            for (0..layers) |i| s.block(i, 1, p0 + f);
            e = s.cond(s.x[0..d], &c);
        }
        return frames;
    }
};
