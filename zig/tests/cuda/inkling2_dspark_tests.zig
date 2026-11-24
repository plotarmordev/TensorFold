//! Inkling 2 Mini's DSpark draft head on the GPU against its float64 host reference on fixed hidden rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const check = @import("check.zig");

const ds = ink.dspark;
const hidden = ds.hidden;

/// Context rows: past the local layers' 1024-row window, so their ring wraps and drops keys the global layer keeps.
const context_rows = 1100;
const start = 37;
const pending: u32 = 1000;
/// The draft logits' RMS error relative to the reference's RMS: bf16 weights and activations sit well under it.
const tolerance = 0.02;
/// Any one logit's error (cap 30), so a wrong column is not diluted by the vocabulary; it is also the pick margin: the
/// GPU's pick must be within this of the reference's best logit.
const column_tolerance = 0.25;

/// tf-cuda-test inkling2-dspark DIR: seven greedy drafts after a seeded context, every step's logits and pick against the
/// float64 reference teacher-forced on the device's picks; INKLING2_REF_PLANT=1 plants a context-blind reference
/// that must go red.
pub fn run(gpu: check.Gpu, model: []const u8) !void {
    const gpa = gpu.gpa;
    var tw = try ink.mini_weights.load(gpa, gpu.io, gpu.d, model);
    defer tw.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, @intCast(try gpu.ctx.capability() / 10));
    defer lane.unload();
    var w = try ds.load(gpa, gpu.io, gpu.d, model);
    defer w.deinit(gpa);
    var k = try ds.Kernels.load(gpu.d);
    defer k.unload();
    var head = try ds.Head.init(gpa, gpu.d, &w, &k, .{ .w = &tw, .lane = &lane });
    defer head.deinit(gpa);
    var c = try ds.Context.init(gpu.d, 2048);
    defer c.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();

    const taps = try gpa.alloc(u16, context_rows * 4 * hidden);
    defer gpa.free(taps);
    var rng = std.Random.DefaultPrng.init(236);
    for (taps) |*t| t.* = @truncate(@as(u32, @bitCast(rng.random().floatNorm(f32))) >> 16);
    var tap_buf = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(taps));
    defer tap_buf.free();
    var picks = cuda.DeviceBuffer{ .d = gpu.d, .ptr = head.picks, .len = (1 + ds.depth) * 4, .borrowed = true };
    try picks.upload(0, std.mem.asBytes(&pending));

    const t0 = check.now(gpu.io);
    try head.absorb(s, &c, tap_buf.ptr, start, context_rows);
    try head.draft(s, &c);
    try s.synchronize();
    const t1 = check.now(gpu.io);
    const vocab: usize = tw.config.text.unpadded_vocab;
    const got = try gpa.alloc(f32, ds.depth * vocab);
    defer gpa.free(got);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = head.logits, .len = got.len * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(got));
    var ids: [1 + ds.depth]u32 = undefined;
    try picks.download(0, std.mem.asBytes(&ids));

    var ck = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    try ck.add(model, "mtp.safetensors");
    const plant = if (std.c.getenv("INKLING2_REF_PLANT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    const r: ink.dspark_ref.Ref = .{ .gpa = gpa, .ck = &ck, .eps = tw.config.text.eps, .vocab = vocab, .cap = tw.config.text.logit_cap, .plant = plant };
    const want = try gpa.alloc(f64, got.len);
    defer gpa.free(want);
    try r.draft(taps, start, pending, ids[1..], want);

    var worst: f64 = 0;
    var worst_column: f64 = 0;
    var bad_picks: usize = 0;
    for (0..ds.depth) |step| {
        const ref_logits = want[step * vocab ..][0..vocab];
        var err: f64 = 0;
        var norm: f64 = 0;
        var column: f64 = 0;
        var best: usize = 0;
        for (got[step * vocab ..][0..vocab], ref_logits, 0..) |g, x, v| {
            err += (g - x) * (g - x);
            norm += x * x;
            column = @max(column, @abs(g - x));
            if (x > ref_logits[best]) best = v;
        }
        const rel = @sqrt(err / norm);
        const margin = ref_logits[best] - ref_logits[ids[step + 1]];
        worst = @max(worst, rel);
        worst_column = @max(worst_column, column);
        if (margin > column_tolerance) bad_picks += 1;
        std.debug.print("INFO inkling2 dspark step {d}: GPU pick {d}, reference pick {d} ({d:.4} below), relative logit error {d:.5}, worst logit error {d:.4}\n", .{ step, ids[step + 1], best, margin, rel, column });
    }
    const ok = worst <= tolerance and worst_column <= column_tolerance and bad_picks == 0;
    std.debug.print("RESULT inkling2 dspark vs float64 reference{s}: worst relative logit error {d:.5} (tolerance {d}), worst logit error {d:.4} (tolerance {d}), {d} of {d} picks past the margin; GPU absorb + draft {d:.1} ms\n", .{ if (plant) " (PLANTED: no context)" else "", worst, tolerance, worst_column, column_tolerance, bad_picks, ds.depth, @as(f64, @floatFromInt(t1 - t0)) / 1e6 });
    if (plant) return check.expect(!ok, "inkling2 dspark planted control: a context-blind reference still agrees", .{});
    try check.expect(ok, "inkling2 dspark logits == reference: relative {d:.5}, worst logit {d:.4}, {d} picks past the margin", .{ worst, worst_column, bad_picks });
}
