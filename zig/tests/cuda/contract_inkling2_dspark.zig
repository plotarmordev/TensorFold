//! CUDA DSpark's device kernels against independent inputs, with poisoned outputs and planted controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const full_ = @import("contract_inkling2_dspark_full_set.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const dspark = ink.dspark;
const hidden = dspark.hidden;

const tap_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 4 * hidden, .k = hidden },
    .{ .rows = 5, .n = 4 * hidden, .k = hidden },
    .{ .rows = 16, .n = 4 * hidden, .k = hidden },
    .{ .rows = 128, .n = 4 * hidden, .k = hidden },
    .{ .rows = 2048, .n = 4 * hidden, .k = hidden },
};

/// A wrong tap selector must fail the served-shape gate; unwritten visits stay poisoned.
const Tap = struct {
    rig: rig_.Rig,
    k: *const dspark.Kernels,
    wrong: bool = false,
    rows: usize = 0,
    src: []u16 = &.{},
    expected: []u8 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    fn kernel(t: *Tap) g.Kernel {
        return .{ .spec = .{
            .name = if (t.wrong) "control: CUDA DSpark tap copies a different visit" else "CUDA DSpark tf_ink2_dsp_tap",
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .served = &tap_shapes,
            .tiles = .{ .{ .size = 1 }, .{}, .{} },
            .nan = .{ .stated = "each input BF16 bit pattern is copied unchanged to its selected visit; all others stay poisoned" },
        }, .ctx = t, .vt = &vt, .control = if (t.wrong) .shapes else null };
    }

    const vt = g.hooks(Tap);

    fn depth(_: c.Shape) usize {
        return 0;
    }

    fn free(t: *Tap) void {
        const a = t.rig.gpu.gpa;
        a.free(t.src);
        a.free(t.expected);
        t.src = &.{};
        t.expected = &.{};
    }

    fn deinit(t: *Tap) void {
        t.free();
        t.rig.deinit();
    }

    pub fn prepare(t: *Tap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.n != 4 * hidden or shape.k != hidden) return false;
        t.rig.reset();
        t.free();
        const a = t.rig.gpu.gpa;
        t.rows = shape.rows;
        t.src = try a.alloc(u16, t.rows * hidden);
        t.expected = try a.alloc(u8, t.rows * 4 * hidden * 2);
        @memset(t.expected, 0xff);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.src, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress) @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) else rng.float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            t.src[0] = 0x7fc1;
            t.src[1] = 0x7f80;
            t.src[2] = 0xff80;
        }
        for (0..t.rows) |row| {
            const src = std.mem.sliceAsBytes(t.src[row * hidden ..][0..hidden]);
            @memcpy(t.expected[(row * 4 + 1) * hidden * 2 ..][0 .. hidden * 2], src);
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.src));
        t.out = try t.rig.output(t.expected.len);
        return true;
    }

    pub fn run(t: *Tap) !void {
        try t.k.launchTap(t.rig.s, t.in, t.out, t.rows, if (t.wrong) 2 else 1);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Tap) usize {
        return t.expected.len;
    }

    pub fn reference(t: *Tap, want: []f64, mag: []f64) !void {
        for (want, mag, t.expected) |*w, *m, v| {
            w.* = @floatFromInt(v);
            m.* = 0;
        }
    }

    pub fn check(t: *Tap, _: g.Gate) !?[]const u8 {
        return if (std.mem.eql(u8, t.expected, try t.rig.outputBytes())) null else "the visit tap differs from the source's BF16 bytes or an unselected visit changed";
    }
};

pub const Set = struct {
    taps: [2]Tap = undefined,
    full: full_.Set = .{},
    loaded: bool = false,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        try set.full.init(gpu, s, a, list);
        errdefer set.full.deinit();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        var made: usize = 0;
        errdefer for (set.taps[0..made]) |*tap| tap.deinit();
        for (&set.taps, 0..) |*tap, i| {
            tap.* = .{ .rig = rig, .k = &set.full.k, .wrong = i == 1 };
            made += 1;
            try list.append(a, tap.kernel());
        }
        set.loaded = true;
    }

    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        for (&set.taps) |*tap| tap.deinit();
        set.full.deinit();
        set.loaded = false;
    }
};
