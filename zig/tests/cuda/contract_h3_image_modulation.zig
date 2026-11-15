//! MiniMax-H3-Image's modulation CUDA kernels (h3_image_modulation.cu) under the contract,
//! each against its float64 formula in contract_h3_image_ref.zig.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_h3_image_ref.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

pub const Which = enum { sinusoid, embed, project };

/// embed's binary64 dot over k: declared a f32-orderable bound (the tightest the gate models); its real error is ~exact.
fn dotK(s: c.Shape) usize {
    return s.k;
}
/// project's lane partials: a serial chain over every 32nd term, then the 32 lanes' serial chain.
fn laneChain(s: c.Shape) usize {
    return (s.k + 31) / 32 + 32;
}
fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .sinusoid => .{ .name = "cuda h3 image modulation h3m_sinusoid", .accum = .none, .depth = none, .ops = 4, .out = .f32, .served = &.{ .{ .rows = 8, .n = 256, .k = 1 }, .{ .rows = 77, .n = 512, .k = 1 } } },
        .embed => .{ .name = "cuda h3 image modulation h3m_embed", .accum = .f32, .depth = dotK, .ops = 4, .out = .f32, .served = &.{ .{ .rows = 8, .n = 64, .k = 256 }, .{ .rows = 8, .n = 128, .k = 1024 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "the kernel sums in binary64: a bf16 or f16 sum's error is a few f32 output ulps" } },
        .project => .{ .name = "cuda h3 image modulation h3m_project", .accum = .f32, .depth = laneChain, .ops = 2, .out = .bf16, .store = .f32, .served = &.{ .{ .rows = 8, .n = 64, .k = 256 }, .{ .rows = 8, .n = 128, .k = 1024 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "the bf16-in-f32 output hides a bf16 sum's error" } },
    };
}

pub fn symbol(w: Which) [:0]const u8 {
    return switch (w) {
        .sinusoid => "h3m_sinusoid",
        .embed => "h3m_embed",
        .project => "h3m_project",
    };
}

pub const Mod = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    shape: c.Shape = undefined,
    arena: std.heap.ArenaAllocator,
    in: ref.In = .{},
    bufs: [6]u64 = undefined,
    nbufs: usize = 0,
    outs: usize = 0,

    pub fn deinit(self: *Mod) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Mod) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Mod);

    fn values(self: *Mod, n: usize, lo: f64, hi: f64, fill: g.Fill, rng: std.Random, half: bool, plant: bool) ![]f64 {
        const v = try self.arena.allocator().alloc(f64, n);
        for (v, 0..) |*x, i| {
            const r = lo + (hi - lo) * rng.float(f64);
            x.* = switch (fill) {
                .stress => @floatFromInt(@as(i64, @intFromFloat(@round(r * 8)))),
                else => r,
            };
            if (i % 61 == 7 and fill != .stress) x.* *= 6;
            x.* = if (half) ref.bf(x.*) else @as(f32, @floatCast(x.*));
        }
        if (fill == .nonfinite and plant) {
            const at = [_]usize{ n / 3, n / 2, n - 1 };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |at_, x| v[at_] = x;
        }
        return v;
    }

    fn upload(self: *Mod, v: []const f64, half: bool) !u64 {
        return self.bytes(try self.encode(v, half));
    }

    fn encode(self: *Mod, v: []const f64, half: bool) ![]const u8 {
        const al = self.arena.allocator();
        if (half) {
            const h = try al.alloc(u16, v.len);
            for (h, v) |*o, x| o.* = c.toBf16(@floatCast(x));
            return std.mem.sliceAsBytes(h);
        }
        const f = try al.alloc(f32, v.len);
        for (f, v) |*o, x| o.* = @floatCast(x);
        return std.mem.sliceAsBytes(f);
    }

    fn bytes(self: *Mod, b: []const u8) !u64 {
        self.bufs[self.nbufs] = try self.rig.input(b);
        self.nbufs += 1;
        return self.bufs[self.nbufs - 1];
    }

    fn output(self: *Mod, len: usize) !u64 {
        self.bufs[self.nbufs] = try self.rig.output(len);
        self.nbufs += 1;
        return self.bufs[self.nbufs - 1];
    }

    pub fn prepare(self: *Mod, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.nbufs = 0;
        self.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (self.which) {
            .sinusoid => {
                // s.rows times through a table of s.n channels (even).
                if (s.n % 2 != 0) return false;
                const times = try self.values(s.rows, 0, 4, fill, rng, false, true);
                self.in = .{ .a = times, .rows = s.rows, .n = s.n };
                _ = try self.upload(times, false);
                _ = try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
            },
            .embed, .project => {
                // x [rows, k] through w [n, k] + b [n]: bf16 weights for the served variants.
                if (s.k > 8192) return false;
                const x = try self.values(s.rows * s.k, -1, 1, fill, rng, false, true);
                const w = try self.values(s.n * s.k, -1, 1, .random, rng, true, false);
                const b = try self.values(s.n, -0.5, 0.5, .random, rng, true, false);
                self.in = .{ .a = x, .b = w, .d = b, .rows = s.rows, .n = s.n, .k = s.k };
                _ = try self.upload(x, false);
                _ = try self.upload(w, true);
                _ = try self.upload(b, true);
                _ = try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
            },
        }
        return true;
    }

    pub fn run(self: *Mod) !void {
        const s = self.shape;
        const b = self.bufs;
        var a: cuda.Args = .{};
        const ci = struct {
            fn of(n: usize) c_int {
                return @intCast(n);
            }
        }.of;
        switch (self.which) {
            .sinusoid => {
                for (b[0..2]) |p| a.add(p);
                for ([_]usize{ s.n, s.rows }) |v| a.add(ci(v));
                try self.go(.{ .x = @intCast((s.rows * s.n + 255) / 256) }, &a);
            },
            .embed => {
                // x f32; w, b read by h3m_read with wide flags 0 (bf16) here.
                for (b[0..4]) |p| a.add(p);
                for ([_]usize{ s.rows, s.n, s.k, 0, 0 }) |v| a.add(ci(v));
                try self.go(.{ .x = @intCast((s.rows * s.n + 255) / 256) }, &a);
            },
            .project => {
                for (b[0..4]) |p| a.add(p);
                for ([_]usize{ s.rows, s.n, s.k, 0, 0 }) |v| a.add(ci(v));
                try self.go(.{ .x = @intCast((s.n + 7) / 8), .y = @intCast(s.rows) }, &a);
            },
        }
        try self.rig.s.synchronize();
    }

    fn go(self: *Mod, grid: cuda.Dim3, args: *cuda.Args) !void {
        try cuda.launch.launch(self.f, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, args);
    }

    pub fn count(self: *Mod) usize {
        return self.outs;
    }

    fn compute(self: *Mod, s_: ref.Sum, want: []f64, mag: []f64) !void {
        switch (self.which) {
            .sinusoid => ref.sinusoid(self.in, want, mag),
            .embed => ref.membed(self.in, s_, want, mag),
            .project => ref.mproject(self.in, s_, want, mag),
        }
    }

    pub fn reference(self: *Mod, want: []f64, mag: []f64) !void {
        return self.compute(.{}, want, mag);
    }

    pub fn simulate(self: *Mod, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        return self.compute(.{ .acc = acc, .order = order }, out, mag);
    }
};
