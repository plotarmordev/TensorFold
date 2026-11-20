//! The H3 CUDA family's products on the runtime's cuBLASLt `Linear`: bf16 in and out, fp32 sums, one
//! algorithm a (n, k) shape kept in the model's picks file (`cuda_algo_cache.zig`) so a cold start
//! skips the heuristic's search.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;
const algo_cache = @import("cuda_algo_cache.zig");

pub const workspace_bytes = 32 << 20;
/// The rows each (n, k)'s algorithm is chosen at: smaller batches retain that algorithm.
const canonical_rows = 16384;

pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: lt_.Library,
    workspace: cuda.DeviceBuffer,
    /// This device's stored picks, or null when the cache is off or has no home; `Int8` shares it.
    picks: ?*algo_cache.Picks = null,
    algos: std.AutoHashMapUnmanaged([2]u64, lt_.Algo) = .empty,
    plans: std.AutoHashMapUnmanaged([3]u64, lt_.Linear) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try lt_.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    /// The device identity's picks, read after `ctx` exists; kept until `deinit` saves the table.
    pub fn loadPicks(g: *Gemms, d: *const cuda.Driver, ctx: *const cuda.Context) !void {
        g.picks = try algo_cache.Picks.load(g.gpa, d, ctx, &g.lt);
    }

    pub fn deinit(g: *Gemms) void {
        if (g.picks) |p| {
            p.save(g.gpa);
            p.deinit(g.gpa);
        }
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

    /// The shape's plan, running the algorithm cuBLASLt picked for (canonical_rows, n, k) — the
    /// picks file's own when it matches, proven on this shape by `initChecked`. A record the device
    /// refuses (a bit flip behind a valid header, a workspace need that grew) is dropped, the search
    /// reruns and its fresh pick replaces it — a corrupt file never repeats.
    pub fn plan(g: *Gemms, rows: usize, n: usize, k: usize) !*const lt_.Linear {
        const got = try g.plans.getOrPut(g.gpa, .{ rows, n, k });
        if (got.found_existing) return got.value_ptr;
        errdefer g.plans.removeByPtr(got.key_ptr);
        const chosen = try g.algoFor(n, k);
        got.value_ptr.* = lt_.Linear.initChecked(&g.lt, rows, n, k, .bf16, workspace_bytes, chosen) catch |e| blk: {
            if (e != error.NoAlgorithm) return e;
            g.stale(.{ canonical_rows, n, k });
            const fresh = try g.searched(n, k);
            break :blk try lt_.Linear.initChecked(&g.lt, rows, n, k, .bf16, workspace_bytes, fresh);
        };
        return got.value_ptr;
    }

    /// The canonical shape's stored pick dropped after the device refused it; the (n, k) in `algos`
    /// cleared the same way, so the next caller searches too.
    fn stale(g: *Gemms, shape: [3]u64) void {
        if (g.picks) |p| p.remove(.bf16, shape);
        _ = g.algos.remove(.{ shape[1], shape[2] });
    }

    /// A fresh heuristic pick for (n, k), remembered in both maps; used on a miss and on a refusal.
    fn searched(g: *Gemms, n: usize, k: usize) !lt_.Algo {
        var l = try lt_.Linear.init(&g.lt, canonical_rows, n, k, .bf16, workspace_bytes);
        defer l.deinit();
        try g.algos.put(g.gpa, .{ n, k }, l.algo);
        if (g.picks) |p| try p.put(g.gpa, .bf16, .{ canonical_rows, n, k }, l.algo);
        return l.algo;
    }

    /// (n, k)'s canonical algorithm: the picks file's, this process's, else the heuristic's, which
    /// is then remembered both ways.
    fn algoFor(g: *Gemms, n: usize, k: usize) !lt_.Algo {
        if (g.algos.get(.{ n, k })) |a| return a;
        const shape = [3]u64{ canonical_rows, n, k };
        if (g.picks) |p| if (p.get(.bf16, shape)) |a| {
            try g.algos.put(g.gpa, .{ n, k }, a);
            return a;
        };
        return g.searched(n, k);
    }
};
