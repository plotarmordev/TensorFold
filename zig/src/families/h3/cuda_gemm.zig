//! The H3 CUDA family's products on the runtime's cuBLASLt `Linear`: bf16 in and out, fp32 sums, one algorithm a shape.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;
const report = @import("report.zig");

pub const workspace_bytes = 32 << 20;
/// The rows each (n, k)'s algorithm is chosen at: smaller batches retain that algorithm.
const canonical_rows = 16384;

pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: lt_.Library,
    workspace: cuda.DeviceBuffer,
    algos: std.AutoHashMapUnmanaged([2]u64, lt_.Algo) = .empty,
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
        g.algos.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    /// y (rows, n) = x (rows, k) . w (n, k)^T in bf16 with fp32 sums, on `stream`.
    pub fn run(g: *Gemms, stream: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        const p = try g.plan(rows, n, k);
        try p.run(x, w, y, g.workspace.ptr, workspace_bytes, stream.handle);
    }

    /// `plan`, its algorithm then the fastest of the heuristic's candidates on these operands (once a shape).
    /// Host-side, before a step is captured: it waits for `stream`, and y gets one of the products.
    pub fn tuned(g: *Gemms, stream: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        if (g.plans.contains(.{ rows, n, k })) return;
        const p = try g.plan(rows, n, k);
        try fastest(&g.lt, p, stream, .{ rows, n, k }, "bf16", .{ x, w, y, g.workspace.ptr, workspace_bytes, stream.handle });
    }

    /// The shape's plan, running the algorithm cuBLASLt's heuristic picks for (canonical_rows, n, k) when it
    /// accepts this shape, else the one it picks for the shape itself.
    pub fn plan(g: *Gemms, rows: usize, n: usize, k: usize) !*lt_.Linear {
        const got = try g.plans.getOrPut(g.gpa, .{ rows, n, k });
        if (got.found_existing) return got.value_ptr;
        errdefer g.plans.removeByPtr(got.key_ptr);
        const chosen = try g.algoFor(n, k);
        var l = try lt_.Linear.init(&g.lt, rows, n, k, .bf16, workspace_bytes);
        errdefer l.deinit();
        var checked: lt_.Heuristic = undefined;
        const s = g.lt.api.cublasLtMatmulAlgoCheck(l.handle, l.desc, l.a, l.b, l.c, l.c, &chosen, &checked);
        if (s == 0 and checked.state == 0 and checked.workspace_size <= workspace_bytes) {
            l.algo = chosen;
            l.workspace_need = checked.workspace_size;
        }
        got.value_ptr.* = l;
        return got.value_ptr;
    }

    fn algoFor(g: *Gemms, n: usize, k: usize) !lt_.Algo {
        if (g.algos.get(.{ n, k })) |a| return a;
        var l = try lt_.Linear.init(&g.lt, canonical_rows, n, k, .bf16, workspace_bytes);
        defer l.deinit();
        try g.algos.put(g.gpa, .{ n, k }, l.algo);
        return l.algo;
    }
};

/// Runs `p`'s own algorithm (a `Linear` or an int8 `Gemm`, main's pick) and each of the heuristic's candidates for it
/// on `args`, its run's or runInt's operands, and keeps the fastest. Logs the two in TOPS.
pub fn fastest(lt: *const lt_.Library, p: anytype, stream: cuda.Stream, shape: [3]usize, kind: []const u8, args: anytype) !void {
    const reps = 5;
    var buf: [lt_.max_candidates]lt_.Heuristic = undefined;
    const ranked = try lt_.candidates(lt, p.handle, p.desc, p.a, p.b, p.c, workspace_bytes, &buf);
    var start = try cuda.Event.init(stream.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(stream.d, true);
    defer end.deinit();
    const own: lt_.Heuristic = .{ .algo = p.algo, .workspace_size = p.workspace_need, .state = 0, .waves_count = 0, .reserved = @splat(0) };
    var best = own;
    var best_ms = std.math.inf(f32);
    var own_ms: f32 = 0;
    for (0..ranked.len + 1) |i| {
        const c = if (i == 0) own else ranked[i - 1];
        p.algo = c.algo;
        p.workspace_need = c.workspace_size;
        try launch(p, args); // warm-up
        try start.record(stream);
        for (0..reps) |_| try launch(p, args);
        try end.record(stream);
        try end.synchronize();
        const ms = try cuda.Event.elapsedMs(start, end) / reps;
        if (i == 0) own_ms = ms;
        if (ms < best_ms) {
            best_ms = ms;
            best = c;
        }
    }
    p.algo = best.algo;
    p.workspace_need = best.workspace_size;
    const tera = 2e-9 * @as(f64, @floatFromInt(shape[0] * shape[1] * shape[2]));
    report.say("h3 {s} product ({d}, {d}, {d}): heuristic #1 {d:.1} TOPS, fastest of {d} candidates {d:.1} TOPS", .{ kind, shape[0], shape[1], shape[2], tera / own_ms, ranked.len, tera / best_ms });
}

fn launch(p: anytype, args: anytype) !void {
    try @call(.auto, if (@TypeOf(p.*) == lt_.Gemm) lt_.Gemm.runInt else lt_.Linear.run, .{p} ++ args);
}
