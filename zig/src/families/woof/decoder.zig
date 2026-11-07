//! Woof's audio decoder (adapted from Kyutai's Mimi): a 2-layer windowed transformer over the latents, then a causal
//! SEANet that upsamples 25 Hz to 24 kHz, run 8 latent frames at a time with each convolution's state carried over.
const std = @import("std");
const ops = @import("ops.zig");
const Loader = @import("weights.zig").Loader;
const Linear = ops.Linear;

const latent = 32;
const dim = 512;
const heads = 8;
const hd = 64;
const window = 250;
const chunk = 8;
const eps = 1e-5;

/// A pack source over a (cout, cin, k) conv weight. Conv: output o, input t * cin + i (its window row t).
/// ConvT: output t * cout + o (tap t), input i.
const Taps = struct {
    raw: []const f32,
    cout: usize,
    cin: usize,
    k: usize,
    transposed: bool,

    pub fn get(s: Taps, out: usize, in: usize) f32 {
        const o, const i, const t = if (s.transposed) .{ out % s.cout, in, out / s.cout } else .{ out, in % s.cin, in / s.cin };
        return s.raw[(o * s.cin + i) * s.k + t];
    }
};

const Conv = struct {
    w: Linear,
    cin: usize,
    state: []f32,

    fn load(l: *Loader, cout: usize, cin: usize, k: usize, comptime name: []const u8, args: anytype) !Conv {
        const raw = try l.grouped(&.{ cout, cin, k }, 0, name ++ ".w", args);
        defer l.tmp.free(raw);
        const taps: Taps = .{ .raw = raw, .cout = cout, .cin = cin, .k = k, .transposed = false };
        return .{ .w = try .pack(l.a, cout, k * cin, taps, try l.floats(cout, name ++ ".b", args)), .cin = cin, .state = try l.a.alloc(f32, (k - 1) * cin) };
    }

    /// out = conv(x) (+ res); x has n rows; win holds the state rows then x.
    fn run(c: *Conv, x: []const f32, n: usize, win: []f32, out: []f32, res: ?[]const f32) void {
        const s = c.state.len;
        @memcpy(win[0..s], c.state);
        @memcpy(win[s..][0 .. n * c.cin], x[0 .. n * c.cin]);
        @memcpy(c.state, win[n * c.cin ..][0..s]);
        ops.rows(c.w, win, c.cin, n, out, c.w.out, res);
    }
};

const ConvT = struct {
    /// Taps (k, cout) of each output, k = 2 * stride.
    w: Linear,
    b: []const f32,
    stride: usize,
    cout: usize,
    carry: []f32,

    fn load(l: *Loader, cin: usize, cout: usize, stride: usize, comptime name: []const u8, args: anytype) !ConvT {
        const k = 2 * stride;
        // grouped along cout, so laid out (cout, cin, k)
        const raw = try l.grouped(&.{ cin, cout, k }, 1, name ++ ".w", args);
        defer l.tmp.free(raw);
        const taps: Taps = .{ .raw = raw, .cout = cout, .cin = cin, .k = k, .transposed = true };
        return .{ .w = try .pack(l.a, k * cout, cin, taps, null), .b = try l.floats(cout, name ++ ".b", args), .stride = stride, .cout = cout, .carry = try l.a.alloc(f32, stride * cout) };
    }

    /// x (n, cin) -> out (n * stride, cout), using z (n, 2 * stride * cout).
    fn run(c: *ConvT, x: []const f32, n: usize, z: []f32, out: []f32) void {
        const half = c.stride * c.cout;
        ops.rows(c.w, x, c.w.in, n, z, 2 * half, null);
        for (0..n) |r| {
            const o = out[r * half ..][0..half];
            const prev = if (r == 0) c.carry else z[(r - 1) * 2 * half + half ..][0..half];
            for (o, z[r * 2 * half ..][0..half], prev, 0..) |*y, v, p, j| y.* = v + p + c.b[j % c.cout];
        }
        @memcpy(c.carry, z[(n - 1) * 2 * half + half ..][0..half]);
    }
};

const Res = struct { a: Conv, b: Conv };
const Stage = struct { up: ConvT, res: Res };

const TfLayer = struct { ln1: [2][]const f32, ln2: [2][]const f32, qkv: Linear, o: Linear, fc1: Linear, fc2: Linear };

pub const Decoder = struct {
    inp: Linear,
    inv_freq: []const f32,
    tf: [2]TfLayer,
    first: Conv,
    stages: [4]Stage,
    last: Conv,
    bufs: [3][]f32,

    pub fn load(l: *Loader) !Decoder {
        var dec: Decoder = undefined;
        const inp_w = try l.scratch(dim * latent, "decoder.inp.0", .{});
        defer l.tmp.free(inp_w);
        dec.inp = try .pack(l.a, dim, latent, ops.Strided{ .m = inp_w, .so = latent, .si = 1 }, try l.floats(dim, "decoder.inp.1", .{}));
        dec.inv_freq = try l.floats(hd / 2, "decoder.inv_freq", .{});
        for (&dec.tf, 0..) |*t, i| {
            t.* = .{
                .ln1 = .{ try l.floats(dim, "decoder.tf.{d}.ln1.0", .{i}), try l.floats(dim, "decoder.tf.{d}.ln1.1", .{i}) },
                .ln2 = .{ try l.floats(dim, "decoder.tf.{d}.ln2.0", .{i}), try l.floats(dim, "decoder.tf.{d}.ln2.1", .{i}) },
                .qkv = try l.groupedLinear(3 * dim, dim, "decoder.tf.{d}.qkv", .{i}),
                .o = try l.groupedLinear(dim, dim, "decoder.tf.{d}.o", .{i}),
                .fc1 = try l.groupedLinear(4 * dim, dim, "decoder.tf.{d}.fc1", .{i}),
                .fc2 = try l.groupedLinear(dim, 4 * dim, "decoder.tf.{d}.fc2", .{i}),
            };
        }
        dec.first = try Conv.load(l, 768, dim, 7, "decoder.seanet.0.1", .{});
        const chans = [_]usize{ 768, 384, 192, 96, 48 };
        const strides = [_]usize{ 8, 6, 5, 4 };
        var largest: usize = chunk * dim * 7;
        var rate: usize = chunk;
        for (&dec.stages, 0..) |*s, i| {
            const at = 2 + 3 * i;
            s.up = try ConvT.load(l, chans[i], chans[i + 1], strides[i], "decoder.seanet.{d}.1", .{at});
            s.res = .{
                .a = try Conv.load(l, chans[i + 1] / 2, chans[i + 1], 3, "decoder.seanet.{d}.1.0", .{at + 1}),
                .b = try Conv.load(l, chans[i + 1], chans[i + 1] / 2, 1, "decoder.seanet.{d}.1.1", .{at + 1}),
            };
            rate *= strides[i];
            largest = @max(largest, 2 * rate * chans[i + 1], rate * chans[i] / strides[i] + 2 * chans[i + 1]);
        }
        dec.last = try Conv.load(l, 1, 48, 3, "decoder.seanet.14.1", .{});
        for (&dec.bufs) |*b| b.* = try l.a.alloc(f32, largest);
        return dec;
    }

    /// Mono audio for `lat` (T latent frames, already de-normalized) into pcm (T * 960 samples).
    pub fn run(dec: *Decoder, a: std.mem.Allocator, lat: []const [latent]f32, pcm: []f32) !void {
        const t = lat.len;
        const h = try a.alloc(f32, t * dim);
        defer a.free(h);
        ops.rows(dec.inp, std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(lat)), latent, t, h, dim, null);
        try dec.transformer(a, h, t);
        for ([_]*Conv{ &dec.first, &dec.last }) |c| @memset(c.state, 0);
        for (&dec.stages) |*s| {
            @memset(s.up.carry, 0);
            @memset(s.res.a.state, 0);
            @memset(s.res.b.state, 0);
        }
        var out: usize = 0;
        var at: usize = 0;
        while (at < t) : (at += chunk) {
            const n: usize = @min(chunk, t - at);
            const y = dec.seanet(h[at * dim ..][0 .. n * dim], n);
            @memcpy(pcm[out..][0..y.len], y);
            out += y.len;
        }
    }

    /// One chunk: b[0] holds each stage's output, b[1] the window scratch, b[2] the residual branch's copy.
    fn seanet(dec: *Decoder, x: []const f32, rows: usize) []const f32 {
        const b = dec.bufs;
        var n = rows;
        dec.first.run(x, n, b[1], b[0], null);
        elu(b[0][0 .. n * 768]);
        for (&dec.stages) |*s| {
            s.up.run(b[0], n, b[1], b[2]);
            n *= s.up.stride;
            const c = s.up.cout;
            @memcpy(b[0][0 .. n * c], b[2][0 .. n * c]);
            elu(b[2][0 .. n * c]);
            s.res.a.run(b[2], n, b[1], b[2], null);
            elu(b[2][0 .. n * c / 2]);
            s.res.b.run(b[2], n, b[1], b[0], b[0]);
            elu(b[0][0 .. n * c]);
        }
        dec.last.run(b[0], n, b[1], b[2], null);
        return b[2][0..n];
    }

    fn transformer(dec: *Decoder, a: std.mem.Allocator, h: []f32, t: usize) !void {
        const y = try a.alloc(f32, t * dim);
        defer a.free(y);
        const big = try a.alloc(f32, t * 4 * dim);
        defer a.free(big);
        const att = try a.alloc(f32, t * dim);
        defer a.free(att);
        var cos: [hd]f32 = undefined;
        var sin: [hd]f32 = undefined;
        var scores: [window]f32 = undefined;
        const scale: f32 = 1 / @sqrt(@as(f32, hd));
        for (dec.tf) |l| {
            norm(h, y, t, l.ln1);
            ops.rows(l.qkv, y, dim, t, big, 3 * dim, null);
            for (0..t) |p| {
                for (0..hd / 2) |j| {
                    const ang = @as(f32, @floatFromInt(p)) * dec.inv_freq[j];
                    cos[j] = @cos(ang);
                    cos[j + hd / 2] = cos[j];
                    sin[j] = @sin(ang);
                    sin[j + hd / 2] = sin[j];
                }
                for (0..2 * heads) |k| ops.rope(big[p * 3 * dim + k * hd ..][0..hd], &cos, &sin);
            }
            for (0..t) |p| for (0..heads) |k| {
                const lo = (p + 1) -| window;
                const sc = scores[0 .. p + 1 - lo];
                const q = big[p * 3 * dim + k * hd ..][0..hd];
                for (sc, lo..) |*v, j| v.* = ops.dot(q, big[j * 3 * dim + dim + k * hd ..][0..hd]) * scale;
                ops.softmax(sc);
                const o = att[p * dim + k * hd ..][0..hd];
                @memset(o, 0);
                for (sc, lo..) |w, j| for (o, big[j * 3 * dim + 2 * dim + k * hd ..][0..hd]) |*z, v| {
                    z.* += w * v;
                };
            };
            ops.rows(l.o, att, dim, t, h, dim, h);
            norm(h, y, t, l.ln2);
            ops.rows(l.fc1, y, dim, t, big, 4 * dim, null);
            for (big[0 .. t * 4 * dim]) |*v| v.* = gelu(v.*);
            ops.rows(l.fc2, big, 4 * dim, t, h, dim, h);
        }
    }
};

fn norm(h: []const f32, y: []f32, t: usize, wb: [2][]const f32) void {
    for (0..t) |r| {
        const o = y[r * dim ..][0..dim];
        @memcpy(o, h[r * dim ..][0..dim]);
        ops.layerNorm(o, eps);
        for (o, wb[0], wb[1]) |*v, w, b| v.* = v.* * w + b;
    }
}

fn elu(x: []f32) void {
    for (x) |*v| if (v.* < 0) {
        v.* = std.math.expm1(v.*);
    };
}

/// GELU with Abramowitz-Stegun 7.1.26 erf (|error| <= 1.5e-7), as Woof's numpy runtime computes it.
fn gelu(x: f32) f32 {
    return 0.5 * x * (1 + erf(x / std.math.sqrt2));
}

fn erf(x: f32) f32 {
    const a = @abs(x);
    const t = 1 / (1 + 0.3275911 * a);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * @exp(-a * a);
    return std.math.copysign(y, x);
}
