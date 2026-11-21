//! Device regression of the picks file's refusal path: bytes a valid header can carry but the device
//! will not run are dropped and the search's fresh pick replaces them, through `cuda_gemm`'s and
//! `cuda_int8`'s own `initChecked` fallbacks — the same code a cold start runs on a corrupt record.
const std = @import("std");
const h3 = @import("cuda.zig");
const algo_cache = @import("cuda_algo_cache.zig");
const lt_ = @import("cuda").cublaslt;

/// An algorithm id far outside the library's: `cublasLtMatmulAlgoCheck` refuses it on any shape.
const refused: lt_.Algo = .{ .data = .{ 0xffff_ffff_ffff_ffff, 0, 0, 0, 0, 0, 0, 0 } };

pub fn run(a: std.mem.Allocator) !void {
    const model = try h3.Model.setup(a);
    // this test's own table, not the start's file: the records the model read stay as they were
    if (model.gemms.picks) |loaded| loaded.deinit(model.gpa);
    var picks: algo_cache.Picks = .{ .key = algo_cache.Picks.identity(&model.driver, &model.ctx, &model.gemms.lt) };
    defer picks.entries.deinit(a);
    model.gemms.picks = &picks;
    defer {
        model.gemms.picks = null; // this table is not the model's to save and destroy
        model.deinit();
    }

    // bf16: the stored pick is proved by initChecked, refused, dropped and re-searched by plan()
    const bf16_shape = [3]u64{ 16384, 64, 64 };
    try picks.put(a, .bf16, bf16_shape, refused);
    _ = try model.gemms.plan(4, 64, 64);
    try replaced(&picks, .bf16, bf16_shape);
    _ = try model.gemms.plan(8, 64, 64); // the fresh pick serves other row counts as-is

    // int8: the same refusal through warm()'s plans, the real cold-start path
    model.inner = 1024;
    model.mlp = 2048;
    model.hidden = 4096;
    const int8 = try model.gpa.create(h3.Int8);
    int8.* = .{ .blocks = &.{}, .handle = null, .scratch = .{ .d = &model.driver, .ptr = 0, .len = 0 } };
    defer int8.deinit(model); // its plans, the scratch warm reserved, its handle
    try model.gemms.lt.check(model.gemms.lt.api.cublasLtCreate(&int8.handle), "cublasLtCreate");
    const int8_shape = [3]u64{ 8, 3 * model.inner, model.hidden };
    try picks.put(a, .int8, int8_shape, refused);
    try int8.warm(model, 8);
    try replaced(&picks, .int8, int8_shape);

    std.debug.print("H3 CUDA picks: a stored algorithm the device refuses is re-searched and replaced, bf16 and int8\n", .{});
}

/// The shape's one record is a fresh pick, not the refused bytes.
fn replaced(picks: *algo_cache.Picks, kind: algo_cache.Kind, shape: [3]u64) !void {
    var n: usize = 0;
    var fresh: ?lt_.Algo = null;
    for (picks.entries.items) |e| if (e.kind == kind and std.mem.eql(u64, &e.shape, &shape)) {
        n += 1;
        fresh = e.algo;
    };
    if (n != 1 or fresh == null) return error.H3PickNotReplaced;
    if (std.meta.eql(fresh.?, refused)) return error.H3RefusedPickKept;
}
