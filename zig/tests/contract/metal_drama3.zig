//! Drama 3's RMS norms on Metal (d3_rmsnorm, d3_add_rmsnorm, d3_gated_norm) under the contract, and gate 5's sources.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// A drama3 source with d3_common.h inlined, by the file its gate 5 label names ("drama3 <file> <function>").
pub fn source(a: std.mem.Allocator, label: []const u8) ![]u8 {
    const d = sources.drama3;
    const files = .{ .{ "dense", d.dense }, .{ "gdn", d.gdn }, .{ "attn", d.attn }, .{ "moe", d.moe }, .{ "sample", d.sample }, .{ "codec", d.codec } };
    var it = std.mem.tokenizeScalar(u8, label, ' ');
    _ = it.next();
    const file = it.next() orelse return error.UnknownLaunch;
    inline for (files) |f| if (std.mem.eql(u8, file, f[0])) return std.mem.replaceOwned(u8, a, f[1], "#include \"d3_common.h\"", d.common);
    return error.UnknownLaunch;
}

pub const Which = enum { rms, add_rms, gated };

/// ceil(k / 256) squares a thread, a simdgroup's tree, then the 8 simdgroups' partials in order.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 13;
}

/// A lane's 4 squares of its 128-wide head, then the simdgroup's tree.
fn headDepth(_: c.Shape) usize {
    return 9;
}

const why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

const hidden = [_]c.Shape{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 168, .n = 2048, .k = 2048 } };

/// The GDN projection's row: q, k and v (8192), the gate z (4096), b and a (32 each).
const ps = 12352;

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .rms => .{ .name = "metal drama3 d3_rmsnorm (bf16 rows, plain gamma)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &hidden, .sums = .{ .bf16 = 0, .why = why } },
        // d3_rmsnorm's ops and the gamma's 1 + w
        .add_rms => .{ .name = "metal drama3 d3_add_rmsnorm (bf16 residual, zero-centred gamma)", .accum = .f32, .depth = depth, .ops = 9, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &hidden, .sums = .{ .bf16 = 0, .why = why } },
        // a frame's 32 value heads of 128; d3_rmsnorm's ops and silu's exp, add, divide and product
        .gated => .{ .name = "metal drama3 d3_gated_norm (128-wide heads, silu gate)", .accum = .f32, .depth = headDepth, .ops = 13, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 4096, .k = 128 }, .{ .rows = 168, .n = 4096, .k = 128 } }, .sums = .{ .bf16 = 0, .why = why } },
    };
}

const NormArgs = extern struct { xs: u32, ys: u32, D: u32, zc: u32, eps: f32 };
const AddNormArgs = extern struct { as: u32, D: u32, eps: f32 };
const GatedNormArgs = extern struct { ps: u32, eps: f32 };

/// One of the norms on prepared rows; d3_add_rmsnorm's residual is copied in again before every launch.
pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    pipe: mtl.Pipeline,
    r: ?rows.Rows = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    z: rig_.Ref = undefined,
    res: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which) !Norm {
        const label: []const u8 = switch (which) {
            .rms => "drama3 dense d3_rmsnorm",
            .add_rms => "drama3 dense d3_add_rmsnorm",
            .gated => "drama3 gdn d3_gated_norm",
        };
        const src = try source(a, label);
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .pipe = try rig_.pipeline(device, src, label[std.mem.lastIndexOfScalar(u8, label, ' ').? + 1 ..]) };
    }

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
        self.pipe.deinit();
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
        // d3_gated_norm's rows are its heads
        const op: rows.Op = switch (self.which) {
            .rms => .rms,
            .add_rms => .add_rms,
            .gated => .gated,
        };
        const r = try rows.Rows.init(self.a, op, if (gated) .{ .rows = shape.rows * 32, .n = 128, .k = 128 } else shape);
        self.r = r;
        r.fill(fill, seed);
        switch (self.which) {
            .rms => self.x = try self.rig.input(std.mem.sliceAsBytes(r.x)),
            .add_rms => self.z = try self.rig.input(std.mem.sliceAsBytes(r.z)),
            .gated => {
                const o = try self.a.alloc(f32, r.x.len);
                defer self.a.free(o);
                for (o, r.x) |*v, b| v.* = c.fromBf16(b);
                self.x = try self.rig.input(std.mem.sliceAsBytes(o));
                const proj = try self.a.alloc(u16, shape.rows * ps);
                defer self.a.free(proj);
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
        const k: u32 = @intCast(r.shape.k);
        if (self.which == .add_rms) @memcpy(self.rig.bytes(self.res, r.x.len * 2), std.mem.sliceAsBytes(r.x));
        const p = self.rig.begin();
        const e = p.e;
        e.setPipeline(self.pipe);
        switch (self.which) {
            .rms => {
                e.setBuffer(self.x.buf, self.x.off, 0);
                e.setBuffer(self.w.buf, self.w.off, 1);
                e.setBuffer(self.out.buf, self.out.off, 2);
                e.setValue(NormArgs{ .xs = k, .ys = k, .D = k, .zc = 0, .eps = r.eps }, 3);
                e.dispatchGroups(mtl.Size.of(r.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
            },
            .add_rms => {
                e.setBuffer(self.res.buf, self.res.off, 0);
                e.setBuffer(self.z.buf, self.z.off, 1);
                e.setBuffer(self.w.buf, self.w.off, 2);
                e.setBuffer(self.out.buf, self.out.off, 3);
                e.setValue(AddNormArgs{ .as = k, .D = k, .eps = r.eps }, 4);
                e.dispatchGroups(mtl.Size.of(r.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
            },
            .gated => {
                e.setBuffer(self.x.buf, self.x.off, 0);
                e.setBuffer(self.z.buf, self.z.off, 1);
                e.setBuffer(self.w.buf, self.w.off, 2);
                e.setBuffer(self.out.buf, self.out.off, 3);
                e.setValue(GatedNormArgs{ .ps = ps, .eps = r.eps }, 4);
                e.dispatchGroups(mtl.Size.of(r.shape.rows / 32, 4, 1), mtl.Size.of(256, 1, 1));
            },
        }
        try self.rig.finish(p);
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
