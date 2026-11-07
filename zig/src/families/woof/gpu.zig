//! Woof's batched matmuls on the Mac's GPU: memory from `allocator()` is Metal shared storage, so rows run in place.
const std = @import("std");
const mtl = @import("metal");
const Linear = @import("ops.zig").Linear;

/// Built with the `metal` module (the Metal engines); elsewhere the Metal import is never analyzed.
pub const enabled = @import("woof_options").gpu;
pub const Handle = if (enabled) *Gpu else void;
/// Rows from which one GPU round trip beats the CPU pool.
pub const min_rows = 32;

/// `h`'s shared memory with the GPU, else `gpa`.
pub fn heap(gpa: std.mem.Allocator, h: Handle) std.mem.Allocator {
    return if (enabled) h.allocator() else gpa;
}

const Args = extern struct { out: u32, in: u32, n: u32, x_stride: u32, y_stride: u32, has_b: u32, has_res: u32 };
const Held = struct { buf: mtl.Buffer, base: usize, len: usize };

pub const Gpu = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    pipe: mtl.Pipeline,
    held: std.ArrayList(Held) = .empty,

    pub fn init(gpa: std.mem.Allocator) !*Gpu {
        const device = try mtl.Device.init();
        const lib = try mtl.Library.fromSource(device, @embedFile("rows.metal"), mtl.CompileOptions.mlx());
        defer lib.deinit();
        const g = try gpa.create(Gpu);
        g.* = .{ .gpa = gpa, .device = device, .queue = try device.queue(), .pipe = try mtl.Pipeline.init(device, lib, "woof_rows", false) };
        return g;
    }

    /// After every block from `allocator()` is freed.
    pub fn deinit(g: *Gpu) void {
        std.debug.assert(g.held.items.len == 0);
        g.held.deinit(g.gpa);
        mtl.objc.release(g.pipe.id);
        g.queue.deinit();
        g.device.deinit();
        g.gpa.destroy(g);
    }

    /// Each block its own shared buffer; for arenas, whose few large nodes it backs.
    pub fn allocator(g: *Gpu) std.mem.Allocator {
        return .{ .ptr = g, .vtable = &.{ .alloc = alloc, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap, .free = free } };
    }

    fn alloc(ctx: *anyopaque, len: usize, _: std.mem.Alignment, _: usize) ?[*]u8 {
        const g: *Gpu = @ptrCast(@alignCast(ctx));
        g.held.ensureUnusedCapacity(g.gpa, 1) catch return null;
        const buf = g.device.buffer(len, mtl.ResourceOptions.shared) catch return null;
        const p = buf.contents();
        g.held.appendAssumeCapacity(.{ .buf = buf, .base = @intFromPtr(p), .len = len });
        return p;
    }

    fn free(ctx: *anyopaque, memory: []u8, _: std.mem.Alignment, _: usize) void {
        const g: *Gpu = @ptrCast(@alignCast(ctx));
        for (g.held.items, 0..) |h, i| if (h.base == @intFromPtr(memory.ptr)) {
            h.buf.deinit();
            _ = g.held.swapRemove(i);
            return;
        };
        unreachable;
    }

    /// The buffer holding `s` and its byte offset there.
    fn find(g: *const Gpu, s: []const f32) ?struct { mtl.Buffer, usize } {
        const at = @intFromPtr(s.ptr);
        for (g.held.items) |h| if (at >= h.base and at + s.len * 4 <= h.base + h.len) return .{ h.buf, at - h.base };
        return null;
    }

    /// ops.rows on the GPU; false (nothing done) when an operand is outside `allocator()`'s memory.
    pub fn rows(g: *const Gpu, l: Linear, x: []const f32, x_stride: usize, n: usize, y: []f32, y_stride: usize, res: ?[]const f32) bool {
        const w = g.find(l.w) orelse return false;
        const xb = g.find(x[0 .. (n - 1) * x_stride + l.in]) orelse return false;
        const yb = g.find(y[0 .. (n - 1) * y_stride + l.out]) orelse return false;
        const bb = if (l.b) |b| g.find(b) orelse return false else w;
        const rb = if (res) |r| g.find(r[0 .. (n - 1) * y_stride + l.out]) orelse return false else w;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(g.pipe);
        for ([_]struct { mtl.Buffer, usize }{ w, bb, xb, rb, yb }, 0..) |s, i| e.setBuffer(s[0], s[1], i);
        e.setValue(Args{ .out = @intCast(l.out), .in = @intCast(l.in), .n = @intCast(n), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .has_b = @intFromBool(l.b != null), .has_res = @intFromBool(res != null) }, 5);
        const panels = std.math.divCeil(usize, l.out, 16) catch unreachable;
        e.dispatchGroups(mtl.Size.of(panels, std.math.divCeil(usize, n, 8) catch unreachable, 1), mtl.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| std.debug.panic("woof rows on the GPU: {s}", .{why});
        return true;
    }
};

test "rows: the GPU matches the CPU pool's sums over strides, bias, residual and a tail block; refuses foreign memory" {
    const ops = @import("ops.zig");
    const Pool = @import("pool.zig").Pool;
    const g = try Gpu.init(std.testing.allocator);
    defer g.deinit();
    var arena = std.heap.ArenaAllocator.init(g.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const out = 37;
    const in = 300;
    const n = 45;
    const w = try a.alloc(f32, out * in);
    for (w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)));
    const l = try Linear.pack(a, out, in, ops.Strided{ .m = w, .so = in, .si = 1 }, w[0..out]);
    const x = try a.alloc(f32, n * (in + 3));
    for (x, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.3);
    const want = try a.alloc(f32, n * 40);
    const got = try a.alloc(f32, n * 40);
    for (want, got, 0..) |*u, *v, i| {
        u.* = @floatFromInt(i % 5);
        v.* = u.*;
    }
    var one: Pool = undefined;
    try one.init(std.testing.allocator, std.testing.io, 1);
    defer one.deinit(std.testing.allocator);
    ops.rows(&one, l, x, in + 3, n, want, 40, want);
    try std.testing.expect(g.rows(l, x, in + 3, n, got, 40, got));
    for (want, got) |u, v| try std.testing.expectApproxEqAbs(u, v, 1e-3);
    var outside: [n * 40]f32 = undefined;
    try std.testing.expect(!g.rows(l, x, in + 3, n, &outside, 40, null));
}
