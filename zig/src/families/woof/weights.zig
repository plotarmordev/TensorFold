//! Checkpoint tensors as fp32, dequantized at load: speech linears int8 rows, per-row scale; decoder fp16 or int8.
const std = @import("std");
const npy = @import("core").npy;
const Npz = @import("npz.zig").Npz;
const ops = @import("ops.zig");

pub const Loader = struct {
    z: Npz,
    /// Holds what the model keeps.
    a: std.mem.Allocator,
    /// Holds dequantized tensors until they are packed.
    tmp: std.mem.Allocator,
    name: [128]u8 = undefined,

    fn array(l: *Loader, comptime fmt: []const u8, args: anytype) !npy.Array {
        return l.z.array(try std.fmt.bufPrint(&l.name, fmt, args));
    }

    /// Any float tensor of `count` values as fp32.
    pub fn floats(l: *Loader, count: usize, comptime fmt: []const u8, args: anytype) ![]f32 {
        return toF32(l.a, try l.array(fmt, args), count);
    }

    /// As floats, but in l.tmp; the caller frees it.
    pub fn scratch(l: *Loader, count: usize, comptime fmt: []const u8, args: anytype) ![]f32 {
        return toF32(l.tmp, try l.array(fmt, args), count);
    }

    /// A speech linear `name`: int8 q (out, in) with scale s (out,), or fp32 w; optional bias b.
    pub fn linear(l: *Loader, out: usize, in: usize, comptime name: []const u8, args: anytype) !ops.Linear {
        const b = try l.bias(out, name, args);
        if (l.array(name ++ ".q", args)) |q| {
            const s = try toF32(l.tmp, try l.array(name ++ ".s", args), out);
            defer l.tmp.free(s);
            return ops.Linear.pack(l.a, out, in, Scaled{ .q = try int8(q, out * in), .s = s, .in = in }, b);
        } else |e| if (e != error.MissingTensor) return e;
        const w = try l.scratch(out * in, name ++ ".w", args);
        defer l.tmp.free(w);
        return ops.Linear.pack(l.a, out, in, ops.Strided{ .m = w, .so = in, .si = 1 }, b);
    }

    fn bias(l: *Loader, out: usize, comptime name: []const u8, args: anytype) !?[]f32 {
        return l.floats(out, name ++ ".b", args) catch |e| if (e == error.MissingTensor) null else e;
    }

    /// A decoder tensor of `shape` stored int8 in groups of 32 along `axis`, as fp32 in l.tmp; the caller frees it.
    pub fn grouped(l: *Loader, shape: []const usize, axis: usize, comptime name: []const u8, args: anytype) ![]f32 {
        var n: usize = 1;
        for (shape) |d| n *= d;
        const rows = shape[axis];
        const cols = n / rows;
        const q = try int8(try l.array(name ++ ".q8", args), n);
        const s_arr = try l.array(name ++ ".s", args);
        const g = 32;
        if (s_arr.rank != 2 or s_arr.shape[0] != rows or s_arr.shape[1] != std.math.divCeil(usize, cols, g) catch unreachable) return error.BadShape;
        const s = try toF32(l.tmp, s_arr, rows * s_arr.shape[1]);
        defer l.tmp.free(s);
        const out = try l.tmp.alloc(f32, n);
        for (0..rows) |r| for (0..cols) |c| {
            out[r * cols + c] = @as(f32, @floatFromInt(q[r * cols + c])) * s[r * s_arr.shape[1] + c / g];
        };
        return out;
    }

    /// A decoder linear (out, in), stored int8 grouped along its outputs.
    pub fn groupedLinear(l: *Loader, out: usize, in: usize, comptime name: []const u8, args: anytype) !ops.Linear {
        const w = try l.grouped(&.{ out, in }, 0, name, args);
        defer l.tmp.free(w);
        return ops.Linear.pack(l.a, out, in, ops.Strided{ .m = w, .so = in, .si = 1 }, null);
    }
};

/// A pack source: int8 rows times a per-row scale.
const Scaled = struct {
    q: []const i8,
    s: []const f32,
    in: usize,

    pub fn get(w: Scaled, o: usize, i: usize) f32 {
        return @as(f32, @floatFromInt(w.q[o * w.in + i])) * w.s[o];
    }
};

fn int8(a: npy.Array, count: usize) ![]const i8 {
    if (!std.mem.eql(u8, a.descr, "|i1") or a.count() != count) return error.BadShape;
    return @ptrCast(a.data);
}

fn toF32(alloc: std.mem.Allocator, a: npy.Array, count: usize) ![]f32 {
    if (a.count() != count) return error.BadShape;
    const out = try alloc.alloc(f32, count);
    errdefer alloc.free(out);
    if (std.mem.eql(u8, a.descr, "<f4")) {
        for (out, 0..) |*x, i| x.* = @bitCast(std.mem.readInt(u32, a.data[i * 4 ..][0..4], .little));
    } else if (std.mem.eql(u8, a.descr, "<f2")) {
        for (out, 0..) |*x, i| x.* = @as(f16, @bitCast(std.mem.readInt(u16, a.data[i * 2 ..][0..2], .little)));
    } else return error.BadDtype;
    return out;
}
