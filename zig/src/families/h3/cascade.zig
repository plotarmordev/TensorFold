//! MiniMax-H3-Image's resolution cascade: early steps on a half-side canvas, then a lift to the full one.
const std = @import("std");
const geometry = @import("geometry.zig");

/// A latent's grid in cells. Latents are float32, channel-major [channels, rows, columns].
pub const Grid = struct { rows: usize, columns: usize };

/// The early steps' grid; both keep even sides (two cells a transformer row), so canvas sides are multiples of 64.
pub fn half(full: Grid) !Grid {
    if (full.rows == 0 or full.columns == 0 or full.rows % 4 != 0 or full.columns % 4 != 0) return error.H3BadCascadeGrid;
    return .{ .rows = full.rows / 2, .columns = full.columns / 2 };
}

fn planes(len: usize, g: Grid) !usize {
    const plane = try geometry.product(&.{ g.rows, g.columns });
    if (len == 0 or len % plane != 0) return error.H3BadLatentLength;
    return len / plane;
}

/// The sum of the 2 x 2 full cells under low cell (r, q) of plane c, over 2: unit variance again for unit noise.
fn pooled(full_noise: []const f32, full: Grid, c: usize, r: usize, q: usize) f32 {
    const at = c * full.rows * full.columns + 2 * r * full.columns + 2 * q;
    const below = at + full.columns;
    return 0.5 * (full_noise[at] + full_noise[at + 1] + full_noise[below] + full_noise[below + 1]);
}

/// The half grid's starting noise, from the seed's full-canvas draw, so both canvases share one seed's picture.
pub fn pool(full_noise: []const f32, full: Grid, out: []f32) !void {
    const low = try half(full);
    const n = try planes(full_noise.len, full);
    if (out.len != n * low.rows * low.columns) return error.H3BadLatentLength;
    for (0..n) |c| for (0..low.rows) |r| for (0..low.columns) |q| {
        out[(c * low.rows + r) * low.columns + q] = pooled(full_noise, full, c, r, q);
    };
}

/// The low cells and weights of a 2x bilinear upsample at full index k: half-cell centres, edges clamped.
fn taps(k: usize, len: usize) [2]usize {
    const near = k / 2;
    return .{ near, if (k % 2 == 0) near -| 1 else @min(near + 1, len - 1) };
}
const weights = [2]f32{ 0.75, 0.25 };

/// Half step `x` at sigma `from`, velocity `v` (noise to data), lifted to sigma `to` with the seed's re-pooled noise.
pub fn lift(x: []const f32, v: []const f32, from: f32, to: f32, full_noise: []const f32, full: Grid, noalias out: []f32) !void {
    const low = try half(full);
    const n = try planes(full_noise.len, full);
    const low_len = n * low.rows * low.columns;
    if (x.len != low_len or v.len != low_len or out.len != full_noise.len) return error.H3BadLatentLength;
    if (!(from > 0 and from <= 1 and to >= 0 and to < from)) return error.H3BadCascadeSigma;
    for (0..n) |c| for (0..full.rows) |r| for (0..full.columns) |q| {
        const base = c * low.rows * low.columns;
        var clean: f32 = 0;
        for (taps(r, low.rows), weights) |lr, wr| for (taps(q, low.columns), weights) |lq, wq| {
            const i = base + lr * low.columns + lq;
            clean += wr * wq * (x[i] + from * v[i]);
        };
        const i = base + (r / 2) * low.columns + q / 2;
        const shift = 0.5 * (x[i] - (1 - from) * v[i] - pooled(full_noise, full, c, r / 2, q / 2));
        const o = (c * full.rows + r) * full.columns + q;
        out[o] = (1 - to) * clean + to * (full_noise[o] + shift);
    };
}

test {
    _ = @import("cascade_test.zig");
}
