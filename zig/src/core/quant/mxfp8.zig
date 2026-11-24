//! MXFP8 weights (e4m3 codes, one UE8M0 power of two a 32 inputs) encoded on the host from wider values.
const std = @import("std");
const fp8 = @import("fp8.zig");
const e4m3FromF32 = @import("../exlr8_codec.zig").e4m3FromF32;

/// Inputs one scale byte covers.
pub const group = 32;

/// Each group's least power of two putting its largest magnitude at or under e4m3's 448 (no code saturates), then
/// every input / 2^e to the nearest e4m3. `row` [k] into codes [k] and scale bytes [k / 32] (2^e as e + 127).
pub fn encodeRow(row: []const f32, codes: []u8, scales: []u8) void {
    std.debug.assert(row.len % group == 0 and codes.len == row.len and scales.len == row.len / group);
    for (scales, 0..) |*s, g| {
        const in = row[g * group ..][0..group];
        var amax: f32 = 0;
        for (in) |x| amax = @max(amax, @abs(x));
        var e: i32 = if (amax == 0) 0 else @intFromFloat(@ceil(std.math.log2(amax / 448.0)));
        if (amax > 0 and std.math.ldexp(@as(f32, 448.0), e) < amax) e += 1; // log2's last-bit error on an exact power
        e = std.math.clamp(e, -127, 127);
        s.* = @intCast(e + 127);
        for (in, codes[g * group ..][0..group]) |x, *c| c.* = e4m3FromF32(std.math.ldexp(x, -e));
    }
}

test "every group takes the least scale that keeps its codes finite, and each code is its input's nearest e4m3" {
    var prng = std.Random.DefaultPrng.init(252);
    const r = prng.random();
    var row: [4 * group]f32 = undefined;
    for (&row, 0..) |*x, i| x.* = (r.float(f32) - 0.5) * std.math.ldexp(@as(f32, 1), @as(i32, @intCast(i / group)) * 7 - 12);
    @memset(row[3 * group ..], 0);
    row[2 * group] = 448.0 * 0x1p2; // a group whose amax is exactly 448 times a power of two
    var codes: [row.len]u8 = undefined;
    var scales: [row.len / group]u8 = undefined;
    encodeRow(&row, &codes, &scales);
    try std.testing.expectEqual(@as(u8, 2 + 127), scales[2]);
    try std.testing.expectEqual(@as(u8, 127), scales[3]);
    for (0..3) |g| {
        var amax: f32 = 0;
        for (row[g * group ..][0..group]) |x| amax = @max(amax, @abs(x));
        const e: i32 = @as(i32, scales[g]) - 127;
        try std.testing.expect(amax <= std.math.ldexp(@as(f32, 448), e) and amax > std.math.ldexp(@as(f32, 448), e - 1));
    }
    for (row, codes, 0..) |x, c, i| {
        try std.testing.expect(c & 0x7f != 0x7f);
        const s = std.math.ldexp(@as(f64, 1), @as(i32, scales[i / group]) - 127);
        const v = fp8.e4m3(c) * s;
        try std.testing.expectEqual(v, fp8.toE4m3(x / s) * s);
    }
}
