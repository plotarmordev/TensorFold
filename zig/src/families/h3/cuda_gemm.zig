//! The H3 CUDA family's products on the runtime's cuBLASLt `Linear`: bf16 in and out, fp32 sums, one algorithm a shape.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;

pub const workspace_bytes = 32 << 20;

pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: lt_.Library,
    workspace: cuda.DeviceBuffer,
    plans: std.AutoHashMapUnmanaged([3]u64, lt_.Linear) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try lt_.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        g.plans.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    /// y (rows, n) = x (rows, k) . w (n, k)^T in bf16 with fp32 sums, on `stream`.
    pub fn run(g: *Gemms, stream: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        const p = try g.plan(rows, n, k);
        try p.run(x, w, y, g.workspace.ptr, workspace_bytes, stream.handle);
    }

    /// The shape's plan: the algorithm cuBLASLt's heuristic picks at the rows this picture runs, not a
    /// canonical batch. The plans asked at 16,384 rows, so a 4,120-row picture ran an algorithm chosen for
    /// four times the work and reached ~35 TFLOPS of the GB10's BF16 rate.
    fn plan(g: *Gemms, rows: usize, n: usize, k: usize) !*const lt_.Linear {
        const got = try g.plans.getOrPut(g.gpa, .{ rows, n, k });
        if (got.found_existing) return got.value_ptr;
        errdefer g.plans.removeByPtr(got.key_ptr);
        var l = try lt_.Linear.init(&g.lt, rows, n, k, .bf16, workspace_bytes);
        errdefer l.deinit();
        got.value_ptr.* = l;
        return got.value_ptr;
    }
};
