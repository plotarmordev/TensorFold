//! Inkling 2 Mini on the released checkpoint at one rank: greedy tokens, a prompt cut in chunks against one chunk
//! (resumed == fresh), and four streams decoding together against each alone (concurrent == solo).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const check = @import("check.zig");
const Gpu = check.Gpu;

const mm = ink.mini_model;

pub const Rig = struct {
    gpu: Gpu,
    w: ink.mini_weights.Weights,
    k: mm.Kernels,
    s: cuda.Stream,
    m: mm.Model,

    pub fn init(r: *Rig, gpu: Gpu, model: []const u8, context: usize, slots: usize) !void {
        r.gpu = gpu;
        r.w = try ink.mini_weights.load(gpu.gpa, gpu.io, gpu.d, model);
        errdefer r.w.deinit();
        ink.mini_weights.roundCaps(&r.w, roundCaps());
        r.k = try mm.Kernels.load(gpu.ctx);
        errdefer r.k.deinit();
        r.s = try cuda.Stream.init(gpu.d, false);
        errdefer r.s.deinit();
        r.m = try mm.Model.init(gpu.gpa, gpu.d, r.s, &r.w, &r.k, context, slots);
    }

    pub fn deinit(r: *Rig) void {
        r.m.deinit();
        r.s.deinit();
        r.k.deinit();
        r.w.deinit();
    }

    fn vocab(r: *const Rig) usize {
        return r.w.config.text.unpadded_vocab;
    }

    /// The host copy of the logits of `heads` rows the last forward kept.
    fn logits(r: *Rig, heads: usize) ![]f32 {
        try r.s.synchronize();
        const out = try r.gpu.gpa.alloc(f32, heads * r.vocab());
        try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = r.m.s.logits, .len = out.len * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(out));
        return out;
    }

    /// A prompt into `slot` in chunks of `chunk` rows; the last row's logits.
    pub fn prompt(r: *Rig, slot: usize, tokens: []const u32, p0: usize, chunk: usize) ![]f32 {
        var at: usize = 0;
        while (at < tokens.len) {
            const n = @min(chunk, tokens.len - at);
            const last = at + n == tokens.len;
            try r.m.forward(&r.k, r.s, &.{.{ .slot = slot, .p0 = p0 + at, .tokens = tokens[at..][0..n] }}, true, if (last) n - 1 else 0, @intFromBool(last));
            at += n;
        }
        return r.logits(1);
    }

    /// One decode row a stream, the streams in one forward: each row's logits.
    pub fn step(r: *Rig, chains: []const mm.Chain) ![]f32 {
        try r.m.forward(&r.k, r.s, chains, false, 0, chains.len);
        return r.logits(chains.len);
    }
};

/// INKLING2_ROUND_CAPS=1: the experts' gate and up rounded to bf16 before the soft caps (engine and reference).
pub fn roundCaps() bool {
    const v = std.c.getenv("INKLING2_ROUND_CAPS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

pub fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return @intCast(best);
}

fn ms(a: i96, b: i96) f64 {
    return @as(f64, @floatFromInt(b - a)) / 1e6;
}

/// Greedy tokens of `slot` from its prompt's logits, one decode row at a time; `rows` (when given) keeps each
/// token's logits, `first` included.
fn decode(r: *Rig, slot: usize, first: []const f32, p0: usize, out: []u32, rows: ?[]f32) !void {
    const v = r.vocab();
    if (rows) |k| @memcpy(k[0..v], first);
    out[0] = argmax(first);
    for (1..out.len) |i| {
        const row = try r.step(&.{.{ .slot = slot, .p0 = p0 + i - 1, .tokens = out[i - 1 .. i] }});
        defer r.gpu.gpa.free(row);
        if (rows) |k| @memcpy(k[i * v ..][0..v], row);
        out[i] = argmax(row);
    }
}

/// KL(reference || engine) in nats of one position's soft-capped logits.
fn kl(ref: []const f64, got: []const f32) f64 {
    var rmax: f64 = -std.math.inf(f64);
    var gmax: f64 = -std.math.inf(f64);
    for (ref, got) |a, b| {
        rmax = @max(rmax, a);
        gmax = @max(gmax, b);
    }
    var rs: f64 = 0;
    var gs: f64 = 0;
    for (ref, got) |a, b| {
        rs += @exp(a - rmax);
        gs += @exp(@as(f64, b) - gmax);
    }
    const rl = rmax + @log(rs);
    const gl = gmax + @log(gs);
    var sum: f64 = 0;
    for (ref, got) |a, b| sum += @exp(a - rl) * ((a - rl) - (@as(f64, b) - gl));
    return sum;
}

/// tf-cuda-test inkling2-mini DIR N TEXT: greedy, resumed == fresh, concurrent == solo, decode speed at 1 and 4.
pub fn run(gpu: Gpu, model: []const u8, count_text: []const u8, text: []const u8) !void {
    const gpa = gpu.gpa;
    const count = try std.fmt.parseInt(usize, count_text, 10);
    var tok = try core.tokenizer.loadTokenizer(gpu.io, gpa, model);
    defer tok.deinit();
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    var rig: Rig = undefined;
    const t_load = check.now(gpu.io);
    try rig.init(gpu, model, 4096, 4);
    defer rig.deinit();
    std.debug.print("INFO inkling2 mini: {d} prompt tokens, {d:.2} GB of weights on the GPU, loaded in {d:.1} s\n", .{ ids.len, @as(f64, @floatFromInt(rig.w.bytes)) / 1e9, ms(t_load, check.now(gpu.io)) / 1e3 });
    const n0 = ids.len;
    const t0 = check.now(gpu.io);
    const first = try rig.prompt(0, ids, 0, mm.prompt_rows);
    defer gpa.free(first);
    const t1 = check.now(gpu.io);
    const plain = try gpa.alloc(u32, count);
    defer gpa.free(plain);
    try decode(&rig, 0, first, n0, plain, null);
    const t2 = check.now(gpu.io);
    const said = try tok.decode(gpa, plain, false);
    defer gpa.free(said);
    std.debug.print("RESULT inkling2 mini greedy: {d} tokens, prompt {d:.1} ms, decode {d:.2} ms a token ({d:.1} tok/s): \"{s}\"\n", .{ count, ms(t0, t1), ms(t1, t2) / @as(f64, @floatFromInt(count - 1)), @as(f64, @floatFromInt(count - 1)) * 1e3 / ms(t1, t2), said });
    std.debug.print("RESULT inkling2 mini ids:", .{});
    for (plain) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
    // resumed == fresh: the prompt in chunks (a prompt-cache hit resumes at a chunk edge), then the same greedy tokens
    var bad: usize = 0;
    for ([_]usize{ 1, 7, 64 }) |chunk| {
        if (chunk >= n0) continue;
        const again = try rig.prompt(1, ids, 0, chunk);
        defer gpa.free(again);
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(again), std.mem.sliceAsBytes(first));
        const resumed = try gpa.alloc(u32, @min(count, 32));
        defer gpa.free(resumed);
        try decode(&rig, 1, again, n0, resumed, null);
        const tokens_same = std.mem.eql(u32, resumed, plain[0..resumed.len]);
        if (!same or !tokens_same) bad += 1;
        std.debug.print("{s} inkling2 mini prompt in chunks of {d}: last-row logits {s}, {d} greedy tokens {s}\n", .{ if (same and tokens_same) "PASS" else "FAIL", chunk, if (same) "bit-equal" else "differ", resumed.len, if (tokens_same) "equal" else "differ" });
    }
    try check.expect(bad == 0, "inkling2 mini resumed == fresh: {d} chunkings differ", .{bad});
    try graphs(&rig, ids, first, plain);
    try snapshots(&rig, ids, first);
    try concurrent(&rig, ids, plain);
}

/// A prompt-cache state saved mid-prompt in slot 0, restored into slot 1, the rest prefilled there: the last row's
/// logits bit-equal to the fresh pass. Planted: a state of another prompt restored instead must differ.
fn snapshots(rig: *Rig, ids: []const u32, first: []const f32) !void {
    const gpa = rig.gpu.gpa;
    const snap = ink.mini_snap;
    const mid: u32 = @intCast(ids.len / 2);
    gpa.free(try rig.prompt(0, ids[0..mid], 0, mm.prompt_rows));
    const k = try snap.save(gpa, &rig.m, rig.s, 0, mid);
    defer snap.drop(gpa, rig.s, k);
    // the planted control: slot 0 refilled with another prefix (the ids reversed) and its state restored
    const other = try gpa.alloc(u32, mid);
    defer gpa.free(other);
    for (other, 0..) |*o, i| o.* = ids[mid - 1 - i];
    gpa.free(try rig.prompt(0, other, 0, mm.prompt_rows));
    const wrong = try snap.save(gpa, &rig.m, rig.s, 0, mid);
    defer snap.drop(gpa, rig.s, wrong);
    var same: [2]bool = undefined;
    for ([_]*const snap.Saved{ wrong, k }, 0..) |state, i| {
        try snap.restore(&rig.m, rig.s, 1, state);
        const got = try rig.prompt(1, ids[mid..], mid, mm.prompt_rows);
        defer gpa.free(got);
        same[i] = std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(first));
    }
    try check.expect(!same[0], "inkling2 mini planted control: another prompt's state restored gives the fresh logits", .{});
    try check.expect(same[1], "inkling2 mini snapshot: restored at {d} != fresh", .{mid});
    check.pass("inkling2 mini prompt-cache state at {d} of {d} ({d} bytes): restored == fresh bit-equal; planted (another prompt's state) red", .{ mid, ids.len, snap.bytes(&rig.m, mid) });
}

/// The decode graphs against direct launches: the first step's logits bit-equal, then 32 greedy tokens equal.
fn graphs(rig: *Rig, ids: []const u32, first: []const f32, plain: []const u32) !void {
    const gpa = rig.gpu.gpa;
    var rows: [2][]f32 = undefined;
    for ([_]bool{ true, false }, 0..) |on, i| {
        rig.m.graphed = on;
        gpa.free(try rig.prompt(2 + i, ids, 0, mm.prompt_rows));
        rows[i] = try rig.step(&.{.{ .slot = 2 + i, .p0 = ids.len, .tokens = plain[0..1] }});
    }
    defer for (rows) |r| gpa.free(r);
    const same = std.mem.eql(u8, std.mem.sliceAsBytes(rows[0]), std.mem.sliceAsBytes(rows[1]));
    const direct = try gpa.alloc(u32, @min(plain.len, 32));
    defer gpa.free(direct);
    gpa.free(try rig.prompt(3, ids, 0, mm.prompt_rows)); // afresh: the step above moved slot 3's conv states
    try decode(rig, 3, first, ids.len, direct);
    rig.m.graphed = true;
    const tokens_same = std.mem.eql(u32, direct, plain[0..direct.len]);
    try check.expect(same and tokens_same, "inkling2 mini graphs == direct: logits {s}, tokens {s}", .{ if (same) "equal" else "differ", if (tokens_same) "equal" else "differ" });
    check.pass("inkling2 mini decode graphs == direct launches: first step's logits bit-equal, {d} greedy tokens equal", .{direct.len});
}

/// Four streams at once (prompts of different lengths) against each alone: equal tokens; the round's decode speed.
fn concurrent(rig: *Rig, ids: []const u32, plain: []const u32) !void {
    const gpa = rig.gpu.gpa;
    const steps = @min(plain.len, 64);
    var solo: [4][]u32 = undefined;
    var lens: [4]usize = undefined;
    var firsts: [4][]f32 = undefined;
    for (0..4) |i| {
        lens[i] = @max(1, ids.len - i * ids.len / 8); // prefixes of the prompt: different lengths and contexts
        firsts[i] = try rig.prompt(i, ids[0..lens[i]], 0, mm.prompt_rows);
        solo[i] = try gpa.alloc(u32, steps);
        try decode(rig, i, firsts[i], lens[i], solo[i]);
    }
    defer for (0..4) |i| {
        gpa.free(solo[i]);
        gpa.free(firsts[i]);
    };
    var together: [4][]u32 = undefined;
    for (0..4) |i| {
        gpa.free(try rig.prompt(i, ids[0..lens[i]], 0, mm.prompt_rows));
        together[i] = try gpa.alloc(u32, steps);
        together[i][0] = argmax(firsts[i]);
    }
    defer for (together) |t| gpa.free(t);
    const t0 = check.now(rig.gpu.io);
    for (1..steps) |j| {
        var chains: [4]mm.Chain = undefined;
        for (0..4) |i| chains[i] = .{ .slot = i, .p0 = lens[i] + j - 1, .tokens = together[i][j - 1 .. j] };
        const rows = try rig.step(&chains);
        defer gpa.free(rows);
        for (0..4) |i| together[i][j] = argmax(rows[i * rig.vocab() ..][0..rig.vocab()]);
    }
    const t1 = check.now(rig.gpu.io);
    var bad: usize = 0;
    for (0..4) |i| {
        if (!std.mem.eql(u32, together[i], solo[i])) bad += 1;
    }
    try check.expect(bad == 0, "inkling2 mini concurrent == solo: {d} of 4 streams' tokens differ", .{bad});
    const per = ms(t0, t1) / @as(f64, @floatFromInt(steps - 1));
    check.pass("inkling2 mini concurrent == solo: 4 streams x {d} greedy tokens equal; c=4 round {d:.2} ms ({d:.1} tok/s total)", .{ steps, per, 4e3 / per });
}

/// tf-cuda-test inkling2-mini-ref DIR N TEXT: N greedy tokens on the GPU, then the float64 reference teacher-forced
/// over the prompt and them: each generated token against the reference's pick (top-1 agreement), every difference
/// with its margin, and the mean KL of the N positions' distributions. A bf16 trunk must agree everywhere; a
/// quantised one (TF_INKLING2_TRUNK) passes at mean KL <= `trunk_kl` and at most one pick differing.
pub fn reference(gpu: Gpu, model: []const u8, count_text: []const u8, text: []const u8) !void {
    const gpa = gpu.gpa;
    const count = try std.fmt.parseInt(usize, count_text, 10);
    var tok = try core.tokenizer.loadTokenizer(gpu.io, gpa, model);
    defer tok.deinit();
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    var rig: Rig = undefined;
    try rig.init(gpu, model, 4096, 1);
    defer rig.deinit();
    const first = try rig.prompt(0, ids, 0, mm.prompt_rows);
    defer gpa.free(first);
    const all = try gpa.alloc(u32, ids.len + count);
    defer gpa.free(all);
    @memcpy(all[0..ids.len], ids);
    const v = rig.vocab();
    const rows = try gpa.alloc(f32, count * v);
    defer gpa.free(rows);
    try decode(&rig, 0, first, ids.len, all[ids.len..], rows);
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    // INKLING2_REF_PLANT=1: the reference drops the cross-pass messages; the check must go red
    const plant = if (std.c.getenv("INKLING2_REF_PLANT")) |p| std.mem.eql(u8, std.mem.span(p), "1") else false;
    const full = try gpa.alloc(f64, count * v);
    defer gpa.free(full);
    var r: ink.mini_ref.Ref = .{ .gpa = gpa, .c = rig.w.config, .p = rig.w.plan, .ck = &ck, .plant = plant, .round_caps = roundCaps(), .full = .{ .from = ids.len - 1, .rows = full } };
    const picks = try gpa.alloc(ink.mini_ref.Pick, all.len);
    defer gpa.free(picks);
    const t0 = check.now(gpu.io);
    try r.run(all[0 .. all.len - 1], picks[0 .. all.len - 1]);
    std.debug.print("INFO inkling2 mini reference: {d} positions in {d:.1} s\n", .{ all.len - 1, ms(t0, check.now(gpu.io)) / 1e3 });
    var differ: usize = 0;
    var smallest: f64 = std.math.inf(f64);
    var kl_sum: f64 = 0;
    for (0..count) |j| kl_sum += kl(full[j * v ..][0..v], rows[j * v ..][0..v]);
    const mean_kl = kl_sum / @as(f64, @floatFromInt(count));
    for (ids.len - 1..all.len - 1) |i| {
        const want = picks[i];
        smallest = @min(smallest, want.margin);
        if (want.id != all[i + 1]) {
            differ += 1;
            std.debug.print("DIFF inkling2 mini position {d}: GPU {d}, reference {d} (reference margin {d:.4})\n", .{ i + 1, all[i + 1], want.id, want.margin });
        }
    }
    const said = try tok.decode(gpa, all[ids.len..], false);
    defer gpa.free(said);
    const trunk = @tagName(rig.w.trunk);
    std.debug.print("RESULT inkling2 mini {s} trunk vs float64 reference{s}: top-1 {d} of {d}, mean KL {e:.3} nats, smallest reference margin {d:.4}: \"{s}\"\n", .{ trunk, if (plant) " (PLANTED: no messages)" else "", count - differ, count, mean_kl, smallest, said });
    if (plant) return check.expect(differ > 0, "inkling2 mini planted control: the reference without messages still agrees", .{});
    if (rig.w.trunk == .bf16) return check.expect(differ == 0, "inkling2 mini greedy == reference: {d} of {d} tokens differ", .{ differ, count });
    try check.expect(differ <= 1 and mean_kl <= trunk_kl, "inkling2 mini {s} trunk gate: top-1 {d} of {d}, mean KL {e:.3} (at most 1 differing, KL <= {d})", .{ trunk, count - differ, count, mean_kl, trunk_kl });
}

/// A quantised trunk's mean-KL limit against the float64 reference, in nats.
const trunk_kl = 0.01;
