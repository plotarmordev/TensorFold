//! MiniMax-H3-Image's picture rules (picture.zig) against hand-worked examples, without a Metal device.
const std = @import("std");
const p = @import("picture.zig");
const g = @import("geometry.zig");
const testing = std.testing;

fn arena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(testing.allocator);
}

fn expectCanvas(w: usize, h: usize, area: u64, cw: usize, ch: usize) !void {
    const c = try p.canvas(w, h, area);
    try testing.expectEqual(p.Size{ .width = cw, .height = ch }, c);
}

test "picture canvas: small sizes grow to the area at their ratio, others snap to 32" {
    const area = p.default_min_area;
    try expectCanvas(1024, 1024, area, 2048, 2048);
    try expectCanvas(1280, 720, area, 2720, 1536);
    try expectCanvas(720, 1280, area, 1536, 2720);
    try expectCanvas(1024, 768, area, 2368, 1760);
    try expectCanvas(1536, 1024, area, 2496, 1664);
    try expectCanvas(2100, 900, area, 3136, 1344);
    try expectCanvas(4736, 3552, area, 4736, 3552);
    try expectCanvas(4096, 4096, area, 4096, 4096);
    // 3000 / 32 = 93.75: snapped up; at or above the area nothing grows
    try expectCanvas(4000, 3000, area, 4000, 3008);
    // the smallest area never grows a picture, only snaps it
    try expectCanvas(1024, 1024, p.min_area_floor, 1024, 1024);
    try expectCanvas(1000, 1000, p.min_area_floor, 992, 992);
    try expectCanvas(1, 1, p.min_area_floor, 32, 32);
    try testing.expectError(error.H3BadMinArea, p.canvas(1024, 1024, p.min_area_floor - 1));
    try testing.expectError(error.H3BadMinArea, p.canvas(1024, 1024, p.min_area_ceiling + 1));
    try testing.expectError(error.H3BadPictureSize, p.canvas(0, 1024, area));
}

test "picture canvas: sides round half to even, as Python's round" {
    try testing.expectEqual(@as(f64, 2), p.roundEven(2.5));
    try testing.expectEqual(@as(f64, 4), p.roundEven(3.5));
    try testing.expectEqual(@as(f64, 3), p.roundEven(2.5000001));
    try testing.expectEqual(@as(f64, 2), p.roundEven(2.4999999));
    // 1040 / 32 = 32.5 and 1072 / 32 = 33.5: to 32 and 34 rows of 32 pixels
    try expectCanvas(1040, 1072, p.min_area_floor, 1024, 1088);
}

test "picture draft schedule: mmh3's renoised grid, drafts then the renoise or the next sigma" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    const base = try p.pdd(x, "{\"num_inference_steps\": 6, \"sigmas\": [1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232, 0.0]}");
    // two drafts, no renoise: the draft sigmas, then the rest of the grid from sigma 3
    const two = try p.draftSchedule(x, base, 2, null);
    try testing.expectEqualSlices(f32, base.sigmas, two.sigmas);
    try testing.expectEqual(@as(usize, 6), two.steps());
    // three drafts renoised to 0.5: [1, .9675, .8875] at half size, [.5, .232] and 0 at full size
    const ren = try p.draftSchedule(x, base, 3, 0.5);
    try testing.expectEqualSlices(f32, &.{ 1.0, 0.9675, 0.8875, 0.5, 0.232, 0.0 }, ren.sigmas);
    try testing.expectEqual(@as(usize, 5), ren.steps());
    for (ren.timesteps, ren.sigmas[0..5]) |t, sg| try testing.expectEqual(@as(f32, 1), t + sg);
    for (ren.deltas, 0..) |d, k| try testing.expectEqual(ren.sigmas[k] - ren.sigmas[k + 1], d);
    // every step drafted is allowed only with a renoise below the last draft sigma (.232):
    // the six drafts, then [.12, 0] at full size
    const all = try p.draftSchedule(x, base, 6, 0.12);
    const all_want = [_]f32{ 1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232, 0.12, 0.0 };
    try testing.expectEqual(all_want.len, all.sigmas.len);
    for (all_want, all.sigmas) |want, got| try testing.expectApproxEqAbs(want, got, 1e-6);
    try testing.expectError(error.H3BadRenoiseSigma, p.draftSchedule(x, base, 6, 0.5));
    try testing.expectError(error.H3DraftLeavesNoStep, p.draftSchedule(x, base, 6, null));
    try testing.expectError(error.H3BadDraftSteps, p.draftSchedule(x, base, 0, null));
    try testing.expectError(error.H3BadDraftSteps, p.draftSchedule(x, base, 7, null));
    try testing.expectError(error.H3BadRenoiseSigma, p.draftSchedule(x, base, 3, 0.9)); // above the last draft sigma
    try testing.expectError(error.H3BadRenoiseSigma, p.draftSchedule(x, base, 3, 0));
    try testing.expectError(error.H3BadRenoiseSigma, p.draftSchedule(x, base, 3, -0.5));
}

test "picture head span: on-grid steps keep their own head, off-grid blends the spanned intervals" {
    const grid = [_]f32{ 1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232, 0.0 };
    // the six-step run: every step's span is its own head at weight 1
    for (0..6) |s| {
        const sp = try p.headSpan(&grid, &grid, s);
        try testing.expectEqual(s, sp.first);
        try testing.expectEqual(s, sp.last);
        try testing.expectEqual(@as(f32, 1), sp.weights[0]);
    }
    // the renoised run [1, .9675, .8875, .5, .232, 0]: the .5 step spans [.5328, .232] → head 4 alone,
    // and the .232 → 0 step keeps head 5
    const sched = [_]f32{ 1.0, 0.9675, 0.8875, 0.5, 0.232, 0.0 };
    const mid = try p.headSpan(&grid, &sched, 3);
    try testing.expectEqual(@as(usize, 4), mid.first);
    try testing.expectEqual(@as(usize, 4), mid.last);
    try testing.expectEqual(@as(f32, 1), mid.weights[0]);
    const last = try p.headSpan(&grid, &sched, 4);
    try testing.expectEqual(@as(usize, 5), last.first);
    try testing.expectEqual(@as(usize, 5), last.last);
    // a step from .9 to 0 starts at the head nearest .9 on the grid (index 2, at .8875) and covers
    // heads 2..5, weighted by each interval's length
    const wide = [_]f32{ 0.9, 0.0 };
    const span = try p.headSpan(&grid, &wide, 0);
    try testing.expectEqual(@as(usize, 2), span.first);
    try testing.expectEqual(@as(usize, 5), span.last);
    var total: f64 = 0;
    for (2..6) |i| total += @as(f64, grid[i]) - @as(f64, grid[i + 1]);
    for (2..6, 0..) |i, w| try testing.expectApproxEqAbs(@as(f32, @floatCast((@as(f64, grid[i]) - @as(f64, grid[i + 1])) / total)), span.weights[w], 1e-6);
}

test "picture enlarge draft: mmh3's bilinear taps on the channel-major latent" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    // one channel of a 4x4-cell draft (2x2 patch rows): channel 0's plane is values 0..15 row-major,
    // each patch row holding that channel's 2x2 cells; the other 23 channels stay zero
    const g_d = p.Grid{ .rows = 4, .columns = 4 };
    const g_full = p.Grid{ .rows = 8, .columns = 8 };
    var rows: [4 * p.row_width]f32 = @splat(0);
    rows[0] = 0;
    rows[1] = 1;
    rows[2] = 4;
    rows[3] = 5; // patch (0,0): cells (0,0),(0,1),(1,0),(1,1)
    rows[p.row_width + 0] = 2;
    rows[p.row_width + 1] = 3;
    rows[p.row_width + 2] = 6;
    rows[p.row_width + 3] = 7; // patch (0,1): cells (0,2),(0,3),(1,2),(1,3)
    rows[2 * p.row_width + 0] = 8;
    rows[2 * p.row_width + 1] = 9;
    rows[2 * p.row_width + 2] = 12;
    rows[2 * p.row_width + 3] = 13; // patch (1,0): cells (2,0),(2,1),(3,0),(3,1)
    rows[3 * p.row_width + 0] = 10;
    rows[3 * p.row_width + 1] = 11;
    rows[3 * p.row_width + 2] = 14;
    rows[3 * p.row_width + 3] = 15; // patch (1,1): cells (2,2),(2,3),(3,2),(3,3)
    const big = try p.enlargeDraft(x, &rows, g_d, g_full);
    try testing.expectEqual(@as(usize, 16 * p.row_width), big.len);
    // mmh3's taps put source position (i + 0.5) * 4/8 - 0.5 = i/2 - 0.25 under output cell i:
    // output cell (0,0) clamps to source cell (0,0)
    try testing.expectApproxEqAbs(@as(f32, 0), big[0], 1e-6);
    // output cell (1,1) → patch (0,0) slot 3: taps rows 0,1 and columns 0,1 at .25 each:
    // upper 0*.75+1*.25 = .25, lower 4*.75+5*.25 = 4.25 → .25*.75 + 4.25*.25 = 1.25
    try testing.expectApproxEqAbs(@as(f32, 1.25), big[3], 1e-6);
    // output cell (3,3) → patch (1,1) slot 3: taps rows 1,2 and columns 1,2 at .25:
    // upper 5*.75+6*.25 = 5.25, lower 9*.75+10*.25 = 9.25 → 5.25*.75 + 9.25*.25 = 6.25
    try testing.expectApproxEqAbs(@as(f32, 6.25), big[5 * p.row_width + 3], 1e-6);
    // the far corner (7,7) → patch (3,3) slot 3: clamps to source cell (3,3) = 15
    try testing.expectApproxEqAbs(@as(f32, 15), big[15 * p.row_width + 3], 1e-6);
    try testing.expectError(error.H3BadPictureGrid, p.enlargeDraft(x, &rows, g_d, p.Grid{ .rows = 12, .columns = 8 }));
}

const pdd_json =
    \\{"num_inference_steps": 4, "sigmas": [1.0, 0.9675236940383911, 0.3, 0.125, 0.0]}
;

test "picture pdd: sigmas through float32 timesteps, t + sigma exactly 1, float32 deltas" {
    var a = arena();
    defer a.deinit();
    const s = try p.pdd(a.allocator(), pdd_json);
    try testing.expectEqual(@as(usize, 4), s.steps());
    try testing.expectEqual(@as(usize, 5), s.sigmas.len);
    try testing.expectEqual(@as(f32, 1), s.sigmas[0]);
    try testing.expectEqual(@as(f32, 0), s.timesteps[0]);
    try testing.expectEqual(@as(f32, 0), s.sigmas[4]);
    for (s.timesteps, s.sigmas[0..4]) |t, sg| try testing.expectEqual(@as(f32, 1), t + sg);
    // sigma 0.3: u = f32(0.7), and 1 - u is exact
    try testing.expectEqual(@as(f32, 1.0) - @as(f32, @floatCast(@as(f64, 0.7))), s.sigmas[2]);
    // sigma near 1: u is small, so 1 - u rounds and t is 1 - sigma', not u
    const u: f32 = @floatCast(1.0 - 0.9675236940383911);
    try testing.expectEqual(@as(f32, 1.0) - u, s.sigmas[1]);
    try testing.expectEqual(@as(f32, 1.0) - s.sigmas[1], s.timesteps[1]);
    for (s.deltas, 0..) |d, k| try testing.expectEqual(s.sigmas[k] - s.sigmas[k + 1], d);
}

test "picture pdd: a config other than steps + 1 sigmas falling strictly from 1 to 0 is refused" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    try testing.expectError(error.H3PddSigmaCount, p.pdd(x, "{\"num_inference_steps\": 3, \"sigmas\": [1.0, 0.5, 0.0]}"));
    try testing.expectError(error.H3PddEnds, p.pdd(x, "{\"num_inference_steps\": 2, \"sigmas\": [0.9, 0.5, 0.0]}"));
    try testing.expectError(error.H3PddEnds, p.pdd(x, "{\"num_inference_steps\": 2, \"sigmas\": [1.0, 0.5, 0.1]}"));
    try testing.expectError(error.H3PddNotFalling, p.pdd(x, "{\"num_inference_steps\": 3, \"sigmas\": [1.0, 0.5, 0.5, 0.0]}"));
    try testing.expectError(error.H3PddNotFalling, p.pdd(x, "{\"num_inference_steps\": 3, \"sigmas\": [1.0, 0.4, 0.6, 0.0]}"));
    try testing.expectError(error.H3BadPdd, p.pdd(x, "{\"num_inference_steps\": 0, \"sigmas\": [1.0]}"));
    try testing.expectError(error.H3BadPdd, p.pdd(x, "{\"num_inference_steps\": 2.0, \"sigmas\": [1.0, 0.5, 0.0]}"));
    try testing.expectError(error.H3BadPdd, p.pdd(x, "{\"sigmas\": [1.0, 0.5, 0.0]}"));
    try testing.expectError(error.H3BadPdd, p.pdd(x, "[1.0, 0.0]"));
    try testing.expectError(error.H3BadPdd, p.pdd(x, "{\"num_inference_steps\": 1, \"sigmas\": [1.0, \"0\"]}"));
}

test "picture euler: the plain form, x + delta * v, not the clip tool's ratio form" {
    var x = [_]f32{ 1, -2, 0.5 };
    p.euler(&x, &.{ 2, 4, -1 }, 0.25);
    try testing.expectEqualSlices(f32, &.{ 1.5, -1, 0.25 }, &x);
}

test "picture positions: text on t, each block centred on 16 by its own area" {
    var a = arena();
    defer a.deinit();
    // three text rows, a 1024-pixel square reference (64 x 64 cells), a 2048-pixel square target (128 x 128)
    const refs = [_]p.Grid{.{ .rows = 64, .columns = 64 }};
    const l = p.Layout{ .text = 3, .references = &refs, .target = .{ .rows = 128, .columns = 128 } };
    try testing.expectEqual(@as(usize, 3 + 32 * 32 + 64 * 64), try l.rows());
    const pos = try p.positions(a.allocator(), l);
    for (0..3) |i| try testing.expectEqual([3]f32{ @floatFromInt(i), 0, 0 }, pos[i]);
    // the reference: t = text + 0, steps of 1 from 0
    try testing.expectEqual([3]f32{ 3, 0, 0 }, pos[3]);
    try testing.expectEqual([3]f32{ 3, 0, 1 }, pos[4]);
    try testing.expectEqual([3]f32{ 3, 31, 31 }, pos[3 + 32 * 32 - 1]);
    // the target: t = text + one reference, steps of 0.5 from 0, inside [0, 32)
    const t0 = 3 + 32 * 32;
    try testing.expectEqual([3]f32{ 4, 0, 0 }, pos[t0]);
    try testing.expectEqual([3]f32{ 4, 0, 0.5 }, pos[t0 + 1]);
    try testing.expectEqual([3]f32{ 4, 0.5, 0 }, pos[t0 + 64]);
    try testing.expectEqual([3]f32{ 4, 31.5, 31.5 }, pos[pos.len - 1]);
}

test "picture positions: a wide block's long axis passes [0, 32), its short one stays inside" {
    var a = arena();
    defer a.deinit();
    // 4736 x 3552: 296 x 222 cells, 148 x 111 rows
    const l = p.Layout{ .text = 0, .references = &.{}, .target = .{ .rows = 222, .columns = 296 } };
    const pos = try p.positions(a.allocator(), l);
    try testing.expectEqual(@as(usize, 111 * 148), pos.len);
    const area = @sqrt(@as(f64, 222.0 * 296.0));
    const h0: f32 = @floatCast((1.0 - 222.0 / area) / 2.0 * 32.0);
    const w0: f32 = @floatCast((1.0 - 296.0 / area) / 2.0 * 32.0);
    try testing.expectEqual([3]f32{ 0, h0, w0 }, pos[0]);
    try testing.expect(h0 > 2.14 and h0 < 2.15);
    try testing.expect(w0 < -2.47 and w0 > -2.48);
    const last = pos[pos.len - 1];
    try testing.expect(last[1] < 32 - h0 and last[1] > 29);
    try testing.expect(last[2] > 32 and last[2] < 32 - w0);
}

test "picture layout: odd or empty grids are refused" {
    try testing.expectError(error.H3BadPictureGrid, p.gridRows(.{ .rows = 63, .columns = 64 }));
    try testing.expectError(error.H3BadPictureGrid, p.gridRows(.{ .rows = 64, .columns = 0 }));
    try testing.expectEqual(p.Grid{ .rows = 222, .columns = 296 }, try p.gridOf(.{ .width = 4736, .height = 3552 }));
    try testing.expectError(error.H3BadPictureSize, p.gridOf(.{ .width = 4736, .height = 3550 }));
}

test "picture rotary: 48 angles a row from rope.inv_freq, each laid twice over the 96 rotated dims" {
    var a = arena();
    defer a.deinit();
    var freq: [16]f32 = undefined;
    for (&freq, 0..) |*f, j| f.* = 1.0 / @as(f32, @floatFromInt(j + 1));
    const pos = [_][3]f32{ .{ 7, 0, 0 }, .{ 9, 2.5, -1.25 } };
    const r = try p.rotary(a.allocator(), &pos, &freq);
    try testing.expectEqual(@as(usize, 2 * 96), r.cos.len);
    for (0..2) |i| for (0..3) |ax| for (freq, 0..) |f, j| {
        const at = i * 96 + ax * 16 + j;
        const angle = pos[i][ax] * f;
        try testing.expectEqual(@cos(angle), r.cos[at]);
        try testing.expectEqual(@sin(angle), r.sin[at]);
        try testing.expectEqual(r.cos[at], r.cos[at + 48]);
        try testing.expectEqual(r.sin[at], r.sin[at + 48]);
    };
    // a text row turns only on t
    for (16..48) |c| {
        try testing.expectEqual(@as(f32, 1), r.cos[c]);
        try testing.expectEqual(@as(f32, 0), r.sin[c]);
    }
    try testing.expectError(error.H3BadRotary, p.rotary(a.allocator(), &pos, freq[0..15]));
}

test "picture rows: channel-major 2 x 2 patches in row-major order, and back" {
    var a = arena();
    defer a.deinit();
    const grid = p.Grid{ .rows = 4, .columns = 6 };
    const n = 24 * 4 * 6;
    const latent = try a.allocator().alloc(f32, n);
    for (latent, 0..) |*v, i| v.* = @floatFromInt(i);
    const rows = try a.allocator().alloc(f32, n);
    try p.patchify(latent, grid, rows);
    // row 4 is patch (1, 1): cells (2..3, 2..3); its channel 5's (dy 1, dx 0) is at 5 * 4 + 2
    const plane = 4 * 6;
    try testing.expectEqual(latent[5 * plane + 3 * 6 + 2], rows[4 * 96 + 5 * 4 + 2]);
    try testing.expectEqual(latent[0], rows[0]);
    try testing.expectEqual(latent[1], rows[1]);
    try testing.expectEqual(latent[6], rows[2]);
    try testing.expectEqual(latent[plane], rows[4]);
    const back = try a.allocator().alloc(f32, n);
    try p.unpatchify(rows, grid, back);
    try testing.expectEqualSlices(f32, latent, back);
    try testing.expectError(error.H3BadLatentLength, p.patchify(latent[1..], grid, rows[1..]));
}

test "picture tiles: prefix then target in tiles of up to 64, admitted by the case reader's counts" {
    var a = arena();
    defer a.deinit();
    const t = try p.tiles(a.allocator(), 65, 130);
    try testing.expectEqual(@as(usize, 2), t.prefix);
    try testing.expectEqual(@as(usize, 3), t.target);
    try testing.expectEqualSlices(i32, &.{ 64, 1, 64, 64, 2 }, t.sizes);
    try testing.expectEqual(@as(i32, 64), t.slot[64]);
    try testing.expectEqual(@as(i32, 128), t.slot[65]);
    try testing.expectEqual(@as(i32, 4 * 64 + 1), t.slot[194]);
    // 60 text rows and 5 reference rows: the counts a picture's case carries
    const r = try g.countsFor(&.{ 2, 3, 3, 60, 0, 130, 5 }, false);
    try testing.expectEqual(@as(usize, 195), r.rows);
    try testing.expectError(error.H3EmptyMedia, p.tiles(a.allocator(), 0, 1));
}

test "picture model folder: the image pipeline and partition, its heads file, an audio-free transformer" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    const index =
        \\{"_class_name": "MiniMaxH3ImagePipeline", "_minimax_h3": {"partition": "image", "pdd_fused_heads": "pdd_fused_heads.safetensors", "tasks": ["t2i", "edit"]}}
    ;
    try testing.expectEqualStrings("pdd_fused_heads.safetensors", try p.checkIndex(x, index));
    try testing.expectError(error.H3NotPictureModel, p.checkIndex(x, "{\"_class_name\": \"MiniMaxH3Pipeline\", \"_minimax_h3\": {\"partition\": \"image\", \"pdd_fused_heads\": \"h.safetensors\"}}"));
    try testing.expectError(error.H3NotPictureModel, p.checkIndex(x, "{\"_class_name\": \"MiniMaxH3ImagePipeline\", \"_minimax_h3\": {\"partition\": \"video\", \"pdd_fused_heads\": \"h.safetensors\"}}"));
    try testing.expectError(error.H3NotPictureModel, p.checkIndex(x, "{\"_class_name\": \"MiniMaxH3ImagePipeline\", \"_minimax_h3\": {\"partition\": \"image\", \"pdd_fused_heads\": \"../h.safetensors\"}}"));
    try testing.expectError(error.H3NotPictureModel, p.checkIndex(x, "{\"_class_name\": \"MiniMaxH3ImagePipeline\", \"_minimax_h3\": {\"partition\": \"image\"}}"));
    const config =
        \\{"latents_dim": 24, "patch_size": [1, 2, 2], "rope_inv_freq_len": 16, "attention_head_dim": 128, "audio_enabled": false}
    ;
    try p.checkTransformer(x, config);
    try testing.expectError(error.H3PictureHasAudio, p.checkTransformer(x, "{\"latents_dim\": 24, \"patch_size\": [1, 2, 2], \"rope_inv_freq_len\": 16, \"attention_head_dim\": 128, \"audio_enabled\": true}"));
    try testing.expectError(error.H3UnsupportedPictureModel, p.checkTransformer(x, "{\"latents_dim\": 24, \"patch_size\": [1, 1, 2], \"rope_inv_freq_len\": 16, \"attention_head_dim\": 128, \"audio_enabled\": false}"));
    try testing.expectError(error.H3UnsupportedPictureModel, p.checkTransformer(x, "{\"latents_dim\": 32, \"patch_size\": [1, 2, 2], \"rope_inv_freq_len\": 16, \"attention_head_dim\": 128, \"audio_enabled\": false}"));
}

test "picture shard count from the transformer's index" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    try testing.expectEqual(@as(usize, 12), try p.shardCount(x, "{\"weight_map\": {\"a\": \"model-00001-of-00012.safetensors\", \"b\": \"model-00012-of-00012.safetensors\"}}"));
    try testing.expectError(error.H3BadShardIndex, p.shardCount(x, "{\"weight_map\": {\"a\": \"model-00001-of-00012.safetensors\", \"b\": \"model-00001-of-00011.safetensors\"}}"));
    try testing.expectError(error.H3BadShardIndex, p.shardCount(x, "{\"weight_map\": {\"a\": \"diffusion-00001-of-00002.safetensors\"}}"));
    try testing.expectError(error.H3BadShardIndex, p.shardCount(x, "{\"weight_map\": {}}"));
}

test "picture distance and gates: cosine, relative error, worst row; bf16 ULPs across zero" {
    const d = try p.distance(&.{ 1, 0, 0, 2 }, &.{ 1, 0, 0, 2 }, 2);
    try testing.expectEqual(@as(f64, 1), d.cosine);
    try testing.expectEqual(@as(f64, 0), d.relative);
    const e = try p.distance(&.{ 1, 0, 0, 3 }, &.{ 1, 0, 0, 2 }, 2);
    try testing.expectEqual(@as(f64, 0.5), e.worst_row);
    try testing.expect(!p.stepPasses(e));
    try testing.expect(p.stepPasses(d));
    try testing.expectError(error.H3BadLatentLength, p.distance(&.{ 1, 2, 3 }, &.{ 1, 2, 3 }, 2));
    try testing.expectEqual(@as(u32, 0), p.ulps(0x3f80, 0x3f80));
    try testing.expectEqual(@as(u32, 1), p.ulps(0x3f80, 0x3f81));
    try testing.expectEqual(@as(u32, 2), p.ulps(0x0001, 0x8001));
    try testing.expectEqual(@as(u16, 0x3f80), p.bf16(1.0));
}

test "picture start rows: one draw of the whole channel-major latent, seeded, then patchified" {
    const fake = struct {
        var seen: u64 = 0;
        var len: usize = 0;
        fn draw(seed: u64, out: []f32) void {
            seen = seed;
            len = out.len;
            for (out, 0..) |*v, i| v.* = @floatFromInt(i);
        }
    };
    const grid = p.Grid{ .rows = 4, .columns = 6 };
    const rows = try p.startRows(testing.allocator, grid, 42, &fake.draw);
    defer testing.allocator.free(rows);
    try testing.expectEqual(@as(u64, 42), fake.seen);
    try testing.expectEqual(@as(usize, 24 * 4 * 6), fake.len);
    // row 4 is patch (1, 1); its channel 5's (dy 1, dx 0) is draw index 5 * 24 + 3 * 6 + 2
    try testing.expectEqual(@as(f32, 5 * 24 + 3 * 6 + 2), rows[4 * 96 + 5 * 4 + 2]);
    try testing.expectError(error.H3BadPictureGrid, p.startRows(testing.allocator, .{ .rows = 3, .columns = 6 }, 0, &fake.draw));
}
