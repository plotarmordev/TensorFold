//! Drama 3's RMS norms (d3_rmsnorm, d3_add_rmsnorm, d3_gated_norm) and decode GEMV d3_gemv under the contract on CUDA.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const Dense = contract.dense.Dense;

pub const Which = enum { rms, add_rms, gated };

/// ceil(k / 256) squares a thread, then a warp's tree and the block's tree of 8 warps.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

/// A lane's 4 squares of its 128-wide head, then the warp's tree.
fn headDepth(_: c.Shape) usize {
    return 9;
}

const why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

const hidden = [_]c.Shape{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 168, .n = 2048, .k = 2048 } };

/// The GDN projection's row: q, k and v (8192), the gate z (4096), b and a (32 each).
const ps = 12352;

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .rms => .{ .name = "cuda drama3 d3_rmsnorm (bf16 rows, plain gamma)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &hidden, .sums = .{ .bf16 = 0, .why = why } },
        // d3_rmsnorm's ops and the gamma's 1 + w
        .add_rms => .{ .name = "cuda drama3 d3_add_rmsnorm (bf16 residual, zero-centred gamma)", .accum = .f32, .depth = depth, .ops = 9, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &hidden, .sums = .{ .bf16 = 0, .why = why } },
        // a frame's 32 value heads of 128; d3_rmsnorm's ops and silu's exp, add, divide and product
        .gated => .{ .name = "cuda drama3 d3_gated_norm (128-wide heads, silu gate)", .accum = .f32, .depth = headDepth, .ops = 13, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 4096, .k = 128 }, .{ .rows = 168, .n = 4096, .k = 128 } }, .sums = .{ .bf16 = 0, .why = why } },
    };
}

/// One of the norms on prepared rows; d3_add_rmsnorm's residual is uploaded again before every launch.
pub const Norm = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    z: u64 = 0,
    res: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module, which: Which) !Norm {
        const symbol: [:0]const u8 = switch (which) {
            .rms => "d3_rmsnorm",
            .add_rms => "d3_add_rmsnorm",
            .gated => "d3_gated_norm",
        };
        return .{ .rig = .{ .gpu = gpu, .s = s }, .which = which, .f = try mod.function(symbol) };
    }

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Norm);

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const gated = self.which == .gated;
        const fits = if (gated) shape.n == 4096 and shape.k == 128 else shape.n == shape.k;
        if (!fits) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        // d3_gated_norm's rows are its heads
        const op: rows.Op = switch (self.which) {
            .rms => .rms,
            .add_rms => .add_rms,
            .gated => .gated,
        };
        const r = try rows.Rows.init(a, op, if (gated) .{ .rows = shape.rows * 32, .n = 128, .k = 128 } else shape);
        self.r = r;
        r.fill(fill, seed);
        switch (self.which) {
            .rms => self.x = try self.rig.input(std.mem.sliceAsBytes(r.x)),
            .add_rms => self.z = try self.rig.input(std.mem.sliceAsBytes(r.z)),
            .gated => {
                const o = try a.alloc(f32, r.x.len);
                defer a.free(o);
                for (o, r.x) |*v, b| v.* = c.fromBf16(b);
                self.x = try self.rig.input(std.mem.sliceAsBytes(o));
                const proj = try a.alloc(u16, shape.rows * ps);
                defer a.free(proj);
                @memset(proj, 0);
                for (0..shape.rows) |t| @memcpy(proj[t * ps + 8192 ..][0..4096], r.z[t * 4096 ..][0..4096]);
                self.z = try self.rig.input(std.mem.sliceAsBytes(proj));
            },
        }
        self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        if (self.which == .add_rms) self.res = try self.rig.output(r.x.len * 2);
        self.out = try self.rig.output(r.x.len * c.size(spec(self.which).out));
        return true;
    }

    pub fn run(self: *Norm) !void {
        const r = self.r.?;
        const k: c_int = @intCast(r.shape.k);
        var a: cuda.Args = .{};
        var grid: cuda.Dim3 = .{ .x = @intCast(r.shape.rows), .y = 1, .z = 1 };
        switch (self.which) {
            .rms => {
                a.add(self.x);
                a.add(k);
                a.add(self.w);
                a.add(self.out);
                a.add(k);
                a.add(k);
                a.add(r.eps);
                a.add(@as(c_int, 0));
            },
            .add_rms => {
                const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.res, .len = r.x.len * 2 };
                try b.upload(0, std.mem.sliceAsBytes(r.x));
                a.add(self.res);
                a.add(self.z);
                a.add(k);
                a.add(self.w);
                a.add(self.out);
                a.add(k);
                a.add(r.eps);
            },
            .gated => {
                a.add(self.x);
                a.add(self.z);
                a.add(@as(c_int, ps));
                a.add(self.w);
                a.add(self.out);
                a.add(r.eps);
                grid = .{ .x = @intCast(r.shape.rows / 32), .y = 4, .z = 1 };
            },
        }
        try cuda.launch.launch(self.f, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Norm) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};

/// A lane's 8-deep dot of each 8-wide chunk, its ceil(k / 256) chunk adds, then the warp's tree.
fn gemvDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 13;
}

/// x and W rows are read 8 bf16 at a time; a block covers 8 outputs.
pub fn gemvSpec() c.Spec {
    return .{ .name = "cuda drama3 d3_gemv (bf16 decode rows)", .accum = .f32, .depth = gemvDepth, .out = .bf16, .tiles = .{ .{}, .{ .size = 8 }, .{ .size = 256, .step = 8, .min = 8 } }, .served = &.{ .{ .rows = 1, .n = 12352, .k = 2048 }, .{ .rows = 1, .n = 2048, .k = 4096 }, .{ .rows = 1, .n = 2048, .k = 512 }, .{ .rows = 1, .n = 2048, .k = 6144 } } };
}

/// x [rows, k] bf16 against W [n, k] bf16, a launch a row as the engine's decode calls it.
pub const Gemv = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module) !Gemv {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .f = try mod.function("d3_gemv") };
    }

    pub fn deinit(self: *Gemv) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Gemv) g.Kernel {
        return .{ .spec = gemvSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Gemv);

    pub fn prepare(self: *Gemv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 8 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x77);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(d.outputs() * 2);
        return true;
    }

    pub fn run(self: *Gemv) !void {
        const d = self.d.?;
        const k = d.shape.k;
        const n = d.shape.n;
        for (0..d.shape.rows) |row| {
            var a: cuda.Args = .{};
            a.add(self.w);
            a.add(self.x + row * k * 2);
            a.add(@as(c_int, @intCast(k)));
            a.add(@as(c_int, @intCast(n)));
            a.add(self.out + row * n * 2);
            a.add(@as(u64, 0));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((n + 7) / 8), .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = @intCast(k * 2) }, self.rig.s, &a);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Gemv) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Gemv, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Gemv, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};
