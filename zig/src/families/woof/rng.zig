//! torch.manual_seed(seed); torch.randn(n) on the CPU, as Woof's runtime draws its flow noise: mt19937 24-bit
//! uniforms, then Box-Muller over each 16: out[j] = r cos, out[j + 8] = r sin, with u1 = 1 - u[j], u2 = u[j + 8].
const std = @import("std");

pub const Randn = struct {
    mt: [624]u32,
    i: usize,

    /// mt19937 init_genrand; torch keeps the low 32 bits of the seed.
    pub fn init(seed: u64) Randn {
        var r: Randn = .{ .mt = undefined, .i = 624 };
        r.mt[0] = @truncate(seed);
        for (1..624) |k| r.mt[k] = 1812433253 *% (r.mt[k - 1] ^ (r.mt[k - 1] >> 30)) +% @as(u32, @intCast(k));
        return r;
    }

    fn next(r: *Randn) u32 {
        if (r.i == 624) {
            for (0..624) |k| {
                const y = (r.mt[k] & 0x80000000) | (r.mt[(k + 1) % 624] & 0x7fffffff);
                r.mt[k] = r.mt[(k + 397) % 624] ^ (y >> 1) ^ (if (y & 1 != 0) @as(u32, 0x9908b0df) else 0);
            }
            r.i = 0;
        }
        var y = r.mt[r.i];
        r.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        return y ^ (y >> 18);
    }

    /// Fills `out` (a multiple of 16) with standard normals.
    pub fn fill(r: *Randn, out: []f32) void {
        std.debug.assert(out.len % 16 == 0);
        var block: usize = 0;
        while (block < out.len) : (block += 16) {
            var u: [16]f64 = undefined;
            for (&u) |*x| x.* = @as(f64, @floatFromInt(r.next() & 0xffffff)) * 0x1p-24;
            for (0..8) |j| {
                const radius = @sqrt(-2.0 * @log(1.0 - u[j]));
                const theta = 2.0 * std.math.pi * u[j + 8];
                out[block + j] = @floatCast(radius * @cos(theta));
                out[block + j + 8] = @floatCast(radius * @sin(theta));
            }
        }
    }
};

test "randn: torch.manual_seed(7); torch.randn(32), twice" {
    var r = Randn.init(7);
    var x: [32]f32 = undefined;
    r.fill(&x);
    const first = [_]f32{ -0.82013446, 0.395631, 0.8989082, -1.388404 };
    for (first, x[0..4]) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    r.fill(&x);
    try std.testing.expectApproxEqAbs(@as(f32, -0.355266), x[30], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7541926), x[31], 1e-6);
}
