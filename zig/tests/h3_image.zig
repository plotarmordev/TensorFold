//! MiniMax-H3-Image's transformer on Metal from a picture case: six steps teacher-forced and gated, then free.
const std = @import("std");
const mtl = @import("metal");
const st = @import("core").safetensors;
const h3 = @import("h3");
const fixture = @import("checkpoint_fixture");

const picture = h3.picture;

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;
const usage =
    \\usage: tf-h3-image MODEL_DIR picture.safetensors OUT_PREFIX [once | free] [--draft-steps N] [--renoise-sigma X]
    \\  MODEL_DIR    the MiniMax-H3-Image folder: model_index.json, transformer/, pdd_config.json and its heads
    \\  picture      the case: canvas, text (the token refiner's rows), noise, and optionally text_tags,
    \\               references and anchors, the reference's step_in, velocity and latent, and its positions,
    \\               timesteps, deltas and rotary_cos / rotary_sin for the exact checks
    \\  once         the first step alone, teacher-forced; free: the free run alone, a picture's generation
    \\  --draft-steps N runs the first N of the checkpoint's steps on the half-size canvas, as mmh3's
    \\               --draft-steps does (the free run only): the clean draft is enlarged bilinearly, noised to
    \\               the next step's sigma (or to --renoise-sigma X) and redrawn at full size for the rest.
    \\               It draws a different picture, not a faster copy of the six-step one.
    \\Exit 0: every gate passed (or a free run had none); 1: a gate failed; 2: no reference to gate (inconclusive).
    \\Between transformer blocks it waits while the speech server speaks. Exclude timing_valid=false steps
    \\from benchmarks, even when their wait is 0: speech can start and finish between polls.
    \\
;

fn read(init: std.process.Init, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(64 << 20)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

fn write(a: std.mem.Allocator, prefix: []const u8, suffix: []const u8, values: []const f32) !void {
    const path = try std.fmt.allocPrintSentinel(a, "{s}{s}", .{ prefix, suffix }, 0);
    const file = fopen(path.ptr, "wb") orelse return error.CannotWrite;
    defer _ = fclose(file);
    if (fwrite(values.ptr, 4, values.len, file) != values.len) return error.CannotWrite;
}

/// A tensor of the checkpoint's in its own bytes, checked for its type and size.
fn raw(ck: *const h3.Checkpoint, name: []const u8, dtype: st.DType, count: usize) ![]const u8 {
    for (ck.shards) |*shard| if (shard.names.get(name)) |e| {
        if (e.dtype != dtype or e.end - e.begin != count * dtype.size()) {
            std.debug.print("{s}: unexpected type or size\n", .{name});
            return error.BadTensor;
        }
        return shard.map.bytes[shard.data + e.begin ..][0 .. e.end - e.begin];
    };
    std.debug.print("no tensor {s}\n", .{name});
    return error.MissingTensor;
}

/// The case's optional tensor `name` of `dtype` and `count` values, else null.
fn optional(case: *const h3.Tensors, comptime T: type, name: []const u8, dtype: st.DType, count: usize) !?[]const T {
    if (case.names.get(name) == null) return null;
    return case.values(T, try case.at(name, dtype, count), count);
}

fn bytes(comptime T: type, values: []const T) []const u8 {
    return std.mem.sliceAsBytes(values);
}

/// A safetensors image with aligned tensors: data from byte 16 on, four-byte tensors first, then two-byte ones.
fn image(a: std.mem.Allocator, entries: []fixture.Entry) ![]u8 {
    std.mem.sort(fixture.Entry, entries, {}, struct {
        fn wide(_: void, x: fixture.Entry, y: fixture.Entry) bool {
            return !std.mem.eql(u8, x.dtype, "BF16") and std.mem.eql(u8, y.dtype, "BF16");
        }
    }.wide);
    const plain = try fixture.image(a, entries);
    const header: usize = @intCast(std.mem.readInt(u64, plain[0..8], .little));
    const pad = (16 - (8 + header) % 16) % 16;
    const out = try a.alloc(u8, plain.len + pad);
    std.mem.writeInt(u64, out[0..8], header + pad, .little);
    @memcpy(out[8..][0..header], plain[8..][0..header]);
    @memset(out[8 + header ..][0..pad], ' ');
    @memcpy(out[8 + header + pad ..], plain[8 + header ..]);
    return out;
}

const Gate = struct { failed: bool = false, gated: bool = false };

/// The model folder: the transformer's weights read in, the distilled schedule and its per-step heads.
pub const Folder = struct {
    dir: []const u8,
    ck: h3.Checkpoint,
    heads: h3.Tensors,
    pdd: picture.Pdd,
    hidden: usize,
    layers: usize,
    /// Every model bound from the folder gives way to live speech between blocks.
    turn: h3.gpu_turn.Turn,

    pub fn open(init: std.process.Init, a: std.mem.Allocator, dir: []const u8) !Folder {
        // the folder: the image pipeline, an audio-free transformer, the distilled schedule and its heads
        const heads_name = try picture.checkIndex(a, try read(init, a, dir, "model_index.json"));
        const transformer_dir = try std.fs.path.join(a, &.{ dir, "transformer" });
        try picture.checkTransformer(a, try read(init, a, transformer_dir, "config.json"));
        const shards = try picture.shardCount(a, try read(init, a, transformer_dir, "model.safetensors.index.json"));
        const pdd = try picture.pdd(a, try read(init, a, std.fs.path.dirname(try std.fs.path.join(a, &.{ dir, heads_name })) orelse dir, "pdd_config.json"));
        const ck = try h3.Model.open(init.gpa, init.io, transformer_dir, shards);
        const heads = try h3.Tensors.open(init.gpa, try mtl.Device.init(), try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, heads_name }, 0));
        const hidden = (ck.entry("final_layer.norm.weight") orelse return error.MissingTensor).shape[0];
        var name: [32]u8 = undefined;
        var layers: usize = 0;
        while (ck.entry(try std.fmt.bufPrint(&name, "blocks.{d}.norm1.weight", .{layers})) != null) layers += 1;
        return .{ .dir = dir, .ck = ck, .heads = heads, .pdd = pdd, .hidden = hidden, .layers = layers, .turn = .{ .io = init.io } };
    }

    /// A model for `case`'s rows over the folder's weights, its AdaLN tables for `sched` (the distilled
    /// schedule by default; the draft run's when --draft-steps renoises a step).
    pub fn bind(f: *const Folder, gpa: std.mem.Allocator, a: std.mem.Allocator, case: *const Case, run: ?picture.Pdd) !*h3.Model {
        const sched = run orelse f.pdd;
        const model = try h3.Model.bind(gpa, f.ck, case.file, .{ .give_way = f.turn });
        errdefer model.release();
        if (!model.base or model.step_heads != sched.steps() or model.audio != 0) return error.H3NotPictureModel;
        model.dense = true;
        const schedule = h3.schedule.Schedule{ .sigmas = sched.sigmas, .timesteps = sched.timesteps };
        _ = try h3.modulation.reschedule(model, a, schedule, schedule, false);
        return model;
    }
};

/// A picture case laid out for the transformer: its rows' positions and rotary table, and the case file.
pub const Case = struct {
    canvas: picture.Size,
    grid: picture.Grid,
    layout: picture.Layout,
    rows: usize,
    pos: [][3]f32,
    cos: []f32,
    sin: []f32,
    noise: []const f32,
    file: *h3.Tensors,

    /// A step's output head as Case.write writes it: video_out.weight [steps,width,hidden] and .bias.
    pub const StepHeads = struct { w: []const f32, b: []const f32 };

    /// The pieces a draft run's case overrides: the run's schedule (sigmas/deltas/video_step and the
    /// step count adaln, times and video_out are sized by), the target's grid, the rows' starting noise
    /// and the per-step output heads (off-grid steps' heads blended by picture.headSpan). Null fields
    /// write the picture's own: the distilled schedule, the canvas's grid, its noise and the heads file.
    pub const Plan = struct {
        sched: ?picture.Pdd = null,
        grid: ?picture.Grid = null,
        noise: ?[]const f32 = null,
        video_out: ?StepHeads = null,
    };

    /// `pic`'s rows packed as the transformer reads them, written to OUT_PREFIX.h3case.safetensors and opened.
    /// `plan` overrides the schedule, target grid, noise and step heads for a draft run's cases.
    pub fn write(init: std.process.Init, a: std.mem.Allocator, f: *const Folder, pic: *const h3.Tensors, prefix: []const u8, plan: ?Plan) !Case {
        // the picture's rows: the canvas's target grid, the text, each reference's grid
        const width = picture.row_width;
        const hidden = f.hidden;
        const sched = if (plan) |p| p.sched orelse f.pdd else f.pdd;
        const steps = sched.steps();
        const canvas_v = pic.values(i32, try pic.at("canvas", .i32, 2), 2);
        const canvas = picture.Size{ .width = @intCast(@max(canvas_v[0], 0)), .height = @intCast(@max(canvas_v[1], 0)) };
        const target_grid = if (plan) |p| p.grid orelse try picture.gridOf(canvas) else try picture.gridOf(canvas);
        const target = try picture.gridRows(target_grid);
        const text_entry = pic.names.get("text") orelse return error.MissingTensor;
        if (text_entry.rank != 2 or text_entry.shape[1] != hidden) return error.BadTensor;
        const text = text_entry.shape[0];
        var refs: []picture.Grid = &.{};
        if (pic.names.get("references")) |e| {
            if (e.rank != 2 or e.shape[1] != 2) return error.BadTensor;
            const v = pic.values(i32, try pic.at("references", .i32, 2 * e.shape[0]), 2 * e.shape[0]);
            refs = try a.alloc(picture.Grid, e.shape[0]);
            for (refs, 0..) |*g, k| g.* = .{ .rows = @intCast(@max(v[2 * k], 0)), .columns = @intCast(@max(v[2 * k + 1], 0)) };
        }
        const layout = picture.Layout{ .text = text, .references = refs, .target = target_grid };
        const held = try layout.referenceRows();
        const rows = try layout.rows();
        const tags = try optional(pic, i32, "text_tags", .i32, text);
        if (tags) |t| for (t) |tag| if (tag != 0 and tag != 1) return error.BadTensor;

        // positions and the rotary table from the checkpoint's frequencies, the tile map of a dense forward
        const pos = try picture.positions(a, layout);
        const rot = try picture.rotary(a, pos, try f.ck.floats(a, "rope.inv_freq"));
        const map = try picture.tiles(a, text + held, target);

        // the transformer's case: these rows, the float projections, the step heads, and placeholder tables
        const noise = if (plan) |p| p.noise orelse pic.values(f32, try pic.at("noise", .f32, target * width), target * width) else pic.values(f32, try pic.at("noise", .f32, target * width), target * width);
        if (noise.len != target * width) return error.BadTensor;
        // a plan's case is a draft run's: the teacher-forcing reference belongs to the picture's own steps
        const reference: ?[]const f32 = if (plan == null) try optional(pic, f32, "velocity", .f32, steps * target * width) else null;
        const anchors: ?[]const f32 = if (held > 0) pic.values(f32, try pic.at("anchors", .f32, held * width), held * width) else null;
        const geometry = [_]i32{ @intCast(map.prefix), @intCast(map.target), @intCast(map.target), @intCast(text), 0, @intCast(target), @intCast(held) };
        const adaln = try a.alloc(i32, steps * rows);
        for (0..steps) |k| for (0..rows) |r| {
            adaln[k * rows + r] = if (r < text) (if (tags) |t| t[r] else 1) else 0;
        };
        const times = try a.alloc(i32, steps * rows);
        @memset(times, 0);
        const video_step = try a.alloc(f32, 2 * steps);
        for (0..steps) |k| video_step[2 * k ..][0..2].* = .{ sched.sigmas[k], sched.deltas[k] };
        const zero_velocity = try a.alloc(f32, target * width);
        @memset(zero_velocity, 0);
        const tables = try a.alloc(u16, 6 * 3 * hidden);
        @memset(tables, 0);
        const final = try a.alloc(u16, 2 * hidden);
        @memset(final, 0);
        const text_bytes = pic.map.bytes[pic.data + text_entry.begin ..][0 .. text * hidden * 2];
        const heads = &f.heads;
        const out_w = if (plan) |p| (if (p.video_out) |v| v.w else heads.values(f32, try heads.at("video_out.weight", .f32, steps * width * hidden), steps * width * hidden)) else heads.values(f32, try heads.at("video_out.weight", .f32, steps * width * hidden), steps * width * hidden);
        const out_b = if (plan) |p| (if (p.video_out) |v| v.b else heads.values(f32, try heads.at("video_out.bias", .f32, steps * width), steps * width)) else heads.values(f32, try heads.at("video_out.bias", .f32, steps * width), steps * width);
        if (out_w.len != steps * width * hidden or out_b.len != steps * width) return error.BadTensor;
        var entries: std.ArrayList(fixture.Entry) = .empty;
        const shape = struct {
            fn of(al: std.mem.Allocator, dims: anytype) ![]const usize {
                return al.dupe(usize, &dims);
            }
        };
        try entries.appendSlice(a, &.{
            .{ .name = "geometry", .dtype = "I32", .shape = try shape.of(a, [_]usize{7}), .bytes = bytes(i32, &geometry) },
            .{ .name = "text", .dtype = "BF16", .shape = try shape.of(a, [_]usize{ text, hidden }), .bytes = text_bytes },
            .{ .name = "final", .dtype = "BF16", .shape = try shape.of(a, [_]usize{ 1, 2 * hidden }), .bytes = bytes(u16, final) },
            .{ .name = "final_norm.weight", .dtype = "BF16", .shape = try shape.of(a, [_]usize{hidden}), .bytes = try raw(&f.ck, "final_layer.norm.weight", .bf16, hidden) },
            .{ .name = "video_in.weight", .dtype = "F32", .shape = try shape.of(a, [_]usize{ hidden, width }), .bytes = try raw(&f.ck, "video_patch_proj.weight", .f32, hidden * width) },
            .{ .name = "video_in.bias", .dtype = "F32", .shape = try shape.of(a, [_]usize{hidden}), .bytes = try raw(&f.ck, "video_patch_proj.bias", .f32, hidden) },
            .{ .name = "video_out.weight", .dtype = "F32", .shape = try shape.of(a, [_]usize{ steps, width, hidden }), .bytes = bytes(f32, out_w) },
            .{ .name = "video_out.bias", .dtype = "F32", .shape = try shape.of(a, [_]usize{ steps, width }), .bytes = bytes(f32, out_b) },
            .{ .name = "video", .dtype = "F32", .shape = try shape.of(a, [_]usize{ target, width }), .bytes = bytes(f32, noise) },
            .{ .name = "video_velocity", .dtype = "F32", .shape = try shape.of(a, [_]usize{ target, width }), .bytes = bytes(f32, if (reference) |v| v[0 .. target * width] else zero_velocity) },
            .{ .name = "video_step", .dtype = "F32", .shape = try shape.of(a, [_]usize{ steps, 2 }), .bytes = bytes(f32, video_step) },
            .{ .name = "cos", .dtype = "F32", .shape = try shape.of(a, [_]usize{ rows, picture.rotated }), .bytes = bytes(f32, rot.cos) },
            .{ .name = "sin", .dtype = "F32", .shape = try shape.of(a, [_]usize{ rows, picture.rotated }), .bytes = bytes(f32, rot.sin) },
            .{ .name = "adaln", .dtype = "I32", .shape = try shape.of(a, [_]usize{ steps, rows }), .bytes = bytes(i32, adaln) },
            .{ .name = "times", .dtype = "I32", .shape = try shape.of(a, [_]usize{ steps, rows }), .bytes = bytes(i32, times) },
            .{ .name = "tile_slot", .dtype = "I32", .shape = try shape.of(a, [_]usize{rows}), .bytes = bytes(i32, map.slot) },
            .{ .name = "tile_sizes", .dtype = "I32", .shape = try shape.of(a, [_]usize{map.sizes.len}), .bytes = bytes(i32, map.sizes) },
        });
        if (anchors) |v| try entries.append(a, .{ .name = "condition", .dtype = "F32", .shape = try shape.of(a, [_]usize{ held, width }), .bytes = bytes(f32, v) });
        for (0..f.layers) |i| try entries.append(a, .{ .name = try std.fmt.allocPrint(a, "tables.{d}", .{i}), .dtype = "BF16", .shape = try shape.of(a, [_]usize{ 6, 3, hidden }), .bytes = bytes(u16, tables) });
        const case_path = try std.fmt.allocPrintSentinel(a, "{s}.h3case.safetensors", .{prefix}, 0);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = case_path, .data = try image(a, entries.items) });
        const file = try a.create(h3.Tensors);
        file.* = try h3.Tensors.open(init.gpa, try mtl.Device.init(), case_path);
        return .{ .canvas = canvas, .grid = target_grid, .layout = layout, .rows = rows, .pos = pos, .cos = rot.cos, .sin = rot.sin, .noise = noise, .file = file };
    }
};

/// Steps `from` to `to` of `sched` on `model`: each step's velocity, then the Euler update. The last
/// step's velocity is left in `last` when set (a draft's lift input: x stays at its sigma, no update).
fn runSteps(model: *h3.Model, a: std.mem.Allocator, sched: picture.Pdd, from: usize, to: usize, x: []f32, last: ?[]f32) !void {
    const velocity = try a.alloc(f32, x.len);
    defer a.free(velocity);
    for (from..to) |k| {
        const took = try model.forward(x, &.{}, k, velocity, &.{});
        if (last != null and k == to - 1) {
            @memcpy(last.?, velocity);
        } else picture.euler(x, velocity, sched.deltas[k]);
        std.debug.print("step {d}/{d}: {d:.2} s wall, {d:.2} s GPU, {d:.2} s waited for speech, timing_valid={}\n", .{ k + 1, sched.steps(), took.wall, took.gpu, took.paused, took.timing_valid });
    }
}

/// The run's rows and latent: {prefix}.rows.f32 and {prefix}.latent.f32 written, both returned.
pub const LatentOut = struct { rows: []f32, latent: []f32 };

fn finish(a: std.mem.Allocator, prefix: []const u8, grid: picture.Grid, x: []f32) !LatentOut {
    try write(a, prefix, ".rows.f32", x);
    const out = try a.alloc(f32, x.len);
    try picture.unpatchify(x, grid, out);
    try write(a, prefix, ".latent.f32", out);
    std.debug.print("wrote {s}.latent.f32: [{d}, {d}, {d}] normalised float32\n", .{ prefix, picture.channels, grid.rows, grid.columns });
    return .{ .rows = x, .latent = out };
}

/// The six steps from the case's noise: OUT_PREFIX.rows.f32 and .latent.f32 written, both returned.
pub fn free(a: std.mem.Allocator, model: *h3.Model, f: *const Folder, case: *const Case, prefix: []const u8) !LatentOut {
    const x = try a.dupe(f32, case.noise);
    try runSteps(model, a, f.pdd, 0, f.pdd.steps(), x, null);
    return finish(a, prefix, case.grid, x);
}

/// mmh3's --draft-steps on Metal: `drafts` steps on the half-size draft case (the last one's velocity
/// kept, its video lifted), then the rest of `sched`'s steps at full size on the lifted draft.
/// `noise_latent` is the seed's full-size noise unpatchified; `cascade.lift` reuses its detail.
pub fn drafted(
    a: std.mem.Allocator,
    model: *h3.Model,
    draft_model: *h3.Model,
    case: *const Case,
    draft_case: *const Case,
    sched: picture.Pdd,
    drafts: usize,
    noise_latent: []const f32,
    prefix: []const u8,
) !LatentOut {
    const steps = sched.steps();
    const width = picture.row_width;
    const draft_grid = draft_case.grid;
    const draft_target = try picture.gridRows(draft_grid);

    // the draft steps on the half-size model: the last one's video stays at sigma[drafts-1] and its
    // velocity goes to cascade.lift for the clean prediction
    const draft = try a.dupe(f32, draft_case.noise);
    const draft_vel = try a.alloc(f32, draft.len);
    try runSteps(draft_model, a, sched, 0, drafts, draft, draft_vel);
    std.debug.print("{d}x{d} draft done, lifting\n", .{ draft_case.canvas.width / 2, draft_case.canvas.height / 2 });

    // cascade.lift predicts the clean latent x + sigma*v at half size, upsamples it bilinearly and
    // renoises it to sigma[drafts] at full size with the seed's own detail
    const full_grid = h3.cascade.Grid{ .rows = case.grid.rows, .columns = case.grid.columns };
    const x_latent = try a.alloc(f32, draft_target * width);
    const v_latent = try a.alloc(f32, draft_target * width);
    try picture.unpatchify(draft, draft_grid, x_latent);
    try picture.unpatchify(draft_vel, draft_grid, v_latent);
    const lifted = try a.alloc(f32, noise_latent.len);
    try h3.cascade.lift(x_latent, v_latent, sched.sigmas[drafts - 1], sched.sigmas[drafts], noise_latent, full_grid, lifted);
    const x = try a.alloc(f32, noise_latent.len);
    try picture.patchify(lifted, case.grid, x);
    try runSteps(model, a, sched, drafts, steps, x, null);
    return finish(a, prefix, case.grid, x);
}

/// The schedule and case plans for a --draft-steps run: mmh3's draft schedule, the half-size grid's
/// pooled noise and each step's output head — an off-grid step blends the distilled heads its sigma
/// span covers (picture.headSpan). `heads` is null when every step sits on the distilled grid.
pub const DraftPlan = struct {
    sched: picture.Pdd,
    draft_grid: picture.Grid,
    draft_noise: []f32,
    noise_latent: []f32,
    heads: ?Case.StepHeads,
};

pub fn draftPlan(a: std.mem.Allocator, f: *const Folder, case: *const Case, drafts: usize, renoise: ?f32) !DraftPlan {
    const width = picture.row_width;
    const sched = try picture.draftSchedule(a, f.pdd, drafts, renoise);
    const full_grid = h3.cascade.Grid{ .rows = case.grid.rows, .columns = case.grid.columns };
    const half = try h3.cascade.half(full_grid);
    const draft_grid = picture.Grid{ .rows = half.rows, .columns = half.columns };
    const draft_target = try picture.gridRows(draft_grid);

    // cascade.pool draws the draft's half-grid noise from the seed's full draw, so both canvases
    // share one seed's picture
    const noise_latent = try a.alloc(f32, picture.channels * case.grid.rows * case.grid.columns);
    try picture.unpatchify(case.noise, case.grid, noise_latent);
    const draft_latent = try a.alloc(f32, picture.channels * draft_grid.rows * draft_grid.columns);
    try h3.cascade.pool(noise_latent, full_grid, draft_latent);
    const draft_noise = try a.alloc(f32, draft_target * width);
    try picture.patchify(draft_latent, draft_grid, draft_noise);

    // each step's output head: the distilled head alone on the grid, the blend of the heads its
    // sigma span covers off it (mmh3's step_heads, as ComfyUI takes them). The override is written
    // whenever a step blends or a renoise changed the run's step count — the heads file holds
    // pdd's count, and the case's video_out needs one row a step.
    const steps = sched.steps();
    var blended = steps != f.pdd.steps();
    const spans = try a.alloc(picture.HeadSpan, steps);
    for (spans, 0..) |*sp, s| {
        sp.* = try picture.headSpan(f.pdd.sigmas, sched.sigmas, s);
        blended = blended or sp.first != sp.last;
    }
    var heads: ?Case.StepHeads = null;
    if (blended) {
        const hidden = f.hidden;
        const hw = f.heads.values(f32, try f.heads.at("video_out.weight", .f32, f.pdd.steps() * width * hidden), f.pdd.steps() * width * hidden);
        const hb = f.heads.values(f32, try f.heads.at("video_out.bias", .f32, f.pdd.steps() * width), f.pdd.steps() * width);
        const w = try a.alloc(f32, steps * width * hidden);
        const b = try a.alloc(f32, steps * width);
        for (spans, 0..) |sp, s| {
            @memset(w[s * width * hidden ..][0 .. width * hidden], 0);
            @memset(b[s * width ..][0..width], 0);
            for (sp.first..sp.last + 1, 0..) |head, j| {
                const weight = sp.weights[j];
                for (w[s * width * hidden ..][0 .. width * hidden], hw[head * width * hidden ..][0 .. width * hidden]) |*o, v| o.* += weight * v;
                for (b[s * width ..][0..width], hb[head * width ..][0..width]) |*o, v| o.* += weight * v;
            }
        }
        heads = .{ .w = w, .b = b };
    }
    return .{ .sched = sched, .draft_grid = draft_grid, .draft_noise = draft_noise, .noise_latent = noise_latent, .heads = heads };
}

pub fn main(init: std.process.Init) !void {
    const all = try init.minimal.args.toSlice(init.arena.allocator());
    const a = init.arena.allocator();
    const gpa = init.gpa;
    var draft_steps: usize = 0;
    var renoise: ?f32 = null;
    var positional: std.ArrayList([:0]const u8) = .empty;
    var arg_i: usize = 1;
    while (arg_i < all.len) : (arg_i += 1) {
        const arg = all[arg_i];
        if (std.mem.eql(u8, arg, "--draft-steps")) {
            arg_i += 1;
            if (arg_i >= all.len) return error.BadArguments;
            draft_steps = std.fmt.parseInt(usize, all[arg_i], 10) catch return error.BadArguments;
        } else if (std.mem.eql(u8, arg, "--renoise-sigma")) {
            arg_i += 1;
            if (arg_i >= all.len) return error.BadArguments;
            renoise = std.fmt.parseFloat(f32, all[arg_i]) catch return error.BadArguments;
        } else try positional.append(a, arg);
    }
    const args = positional.items;
    if (args.len != 3 and args.len != 4) {
        std.debug.print("{s}", .{usage});
        return error.BadArguments;
    }
    const mode: enum { all, once, free } = if (args.len == 3) .all else if (std.mem.eql(u8, args[3], "once")) .once else if (std.mem.eql(u8, args[3], "free")) .free else {
        std.debug.print("{s}", .{usage});
        return error.BadArguments;
    };
    // drafts redraw a picture: there is no reference to teacher-force against
    if (draft_steps > 0 and mode == .once) return error.BadArguments;
    if (draft_steps == 0 and renoise != null) return error.BadArguments; // --renoise-sigma noises a draft

    const loading = mtl.clock.seconds();
    var folder = try Folder.open(init, a, args[0]);
    defer folder.heads.deinit();
    const pdd = folder.pdd;
    const steps = pdd.steps();
    var pic = try h3.Tensors.open(gpa, try mtl.Device.init(), args[1]);
    defer pic.deinit();
    const case = try Case.write(init, a, &folder, &pic, args[2], null);
    defer case.file.deinit();
    const rows = case.rows;
    const pos = case.pos;
    var gate = Gate{};
    exact: {
        const theirs = (try optional(&pic, f32, "positions", .f32, rows * 3)) orelse break :exact;
        var same: usize = 0;
        for (pos, 0..) |q, i| for (q, 0..) |v, ax| {
            same += @intFromBool(@as(u32, @bitCast(v)) == @as(u32, @bitCast(theirs[3 * i + ax])));
        };
        std.debug.print("positions against the reference's: {d} of {d} bit-identical\n", .{ same, rows * 3 });
        gate.failed = gate.failed or same != rows * 3;
    }
    exact: {
        const c = (try optional(&pic, u16, "rotary_cos", .bf16, rows * picture.angles)) orelse break :exact;
        const s = (try optional(&pic, u16, "rotary_sin", .bf16, rows * picture.angles)) orelse break :exact;
        var same: usize = 0;
        var worst: u32 = 0;
        for (0..rows) |i| for (0..picture.angles) |j| {
            const ours = [2]u16{ picture.bf16(case.cos[i * picture.rotated + j]), picture.bf16(case.sin[i * picture.rotated + j]) };
            const want = [2]u16{ c[i * picture.angles + j], s[i * picture.angles + j] };
            for (ours, want) |x, y| {
                same += @intFromBool(x == y);
                worst = @max(worst, picture.ulps(x, y));
            }
        };
        std.debug.print("rotary table against the reference's: {d} of {d} bf16 values identical, at most {d} ULP apart\n", .{ same, 2 * rows * picture.angles, worst });
        gate.failed = gate.failed or worst > picture.gates.rotary_ulps;
    }
    inline for (.{ .{ "timesteps", pdd.timesteps }, .{ "deltas", pdd.deltas } }) |kv| exact: {
        const theirs = (try optional(&pic, f32, kv[0], .f32, steps)) orelse break :exact;
        const same = std.mem.eql(u32, @as([]const u32, @ptrCast(kv[1])), @as([]const u32, @ptrCast(theirs)));
        std.debug.print("{s} against the reference's: {s}\n", .{ kv[0], if (same) "bit-identical" else "DIFFERENT" });
        gate.failed = gate.failed or !same;
    }

    const target = try picture.gridRows(case.grid);
    const width = picture.row_width;
    const reference = try optional(&pic, f32, "velocity", .f32, steps * target * width);
    const step_in = try optional(&pic, f32, "step_in", .f32, steps * target * width);
    const latent = try optional(&pic, f32, "latent", .f32, target * width);
    const model = try folder.bind(gpa, a, &case, null);
    const layout = case.layout;
    std.debug.print("{s}: {d}x{d} canvas, {d} rows (text {d}, references {d}, target {d}), {d} tiles, {d} blocks, {d} steps, {d} timesteps, loaded in {d:.1} s\n", .{ model.device.name(), case.canvas.width, case.canvas.height, rows, layout.text, try layout.referenceRows(), target, model.tiles, model.blocks.len, steps, model.lines / h3.schedule.modalities, mtl.clock.seconds() - loading });

    const velocity = try a.alloc(f32, target * width);
    const x = try a.alloc(f32, target * width);
    // teacher-forced: every step on the reference's own input, its target velocity gated
    if (mode != .free and draft_steps == 0) teacher: {
        const ins = step_in orelse {
            std.debug.print("no step_in in the case: the teacher-forced gate is inconclusive\n", .{});
            break :teacher;
        };
        const want = reference orelse break :teacher;
        for (0..if (mode == .once) 1 else steps) |k| {
            @memcpy(x, ins[k * target * width ..][0 .. target * width]);
            const took = try model.forward(x, &.{}, k, velocity, &.{});
            const d = try picture.distance(velocity, want[k * target * width ..][0 .. target * width], width);
            const ok = picture.stepPasses(d);
            gate.gated = true;
            gate.failed = gate.failed or !ok;
            std.debug.print("step {d}/{d} teacher-forced: cosine {d:.6}, relative error {d:.5}, worst row {d:.5} ({s}), {d:.2} s wall, {d:.2} s GPU, {d:.2} s waited for speech, timing_valid={}\n", .{ k + 1, steps, d.cosine, d.relative, d.worst_row, if (ok) "pass" else "FAIL", took.wall, took.gpu, took.paused, took.timing_valid });
            if (k == 0) try write(a, args[2], ".first-velocity.f32", velocity);
        }
    }
    // free-running: our own rows from the noise through every step — the first `draft_steps` of them
    // on the half-size canvas with --draft-steps, mmh3's fewer-steps path
    if (mode != .once) {
        const out = if (draft_steps == 0)
            try free(a, model, &folder, &case, args[2])
        else draft: {
            if (try case.layout.referenceRows() != 0) return error.H3DraftWithReferences;
            const plan = try draftPlan(a, &folder, &case, draft_steps, renoise);
            const draft_prefix = try std.fmt.allocPrint(a, "{s}.draft", .{args[2]});
            const draft_case = try Case.write(init, a, &folder, &pic, draft_prefix, .{ .sched = plan.sched, .grid = plan.draft_grid, .noise = plan.draft_noise, .video_out = plan.heads });
            defer draft_case.file.deinit();
            const draft_model = try folder.bind(gpa, a, &draft_case, plan.sched);
            defer draft_model.release();
            // the full-size steps of the draft schedule need their own binding when a renoise moved
            // a step off the distilled grid (other AdaLN lines and blended heads); the plain draft
            // keeps pdd's, so the normal binding's tables and heads serve it as it stands
            var full = model;
            if (plan.heads != null) {
                const run_case = try Case.write(init, a, &folder, &pic, try std.fmt.allocPrint(a, "{s}.renoise", .{args[2]}), .{ .sched = plan.sched, .video_out = plan.heads });
                defer run_case.file.deinit();
                full = try folder.bind(gpa, a, &run_case, plan.sched);
            }
            defer if (full != model) full.release();
            break :draft try drafted(a, full, draft_model, &case, &draft_case, plan.sched, draft_steps, plan.noise_latent, args[2]);
        };
        // a draft draws a different picture on purpose: nothing to gate its latent against
        if (latent != null and draft_steps == 0) {
            const want = latent.?;
            const d = try picture.distance(out.rows, want, width);
            const ok = d.cosine >= picture.gates.latent_cosine;
            gate.gated = true;
            gate.failed = gate.failed or !ok;
            std.debug.print("final latent against the reference's: cosine {d:.6}, relative error {d:.5}, worst row {d:.5} ({s})\n", .{ d.cosine, d.relative, d.worst_row, if (ok) "pass" else "FAIL" });
        }
    }
    if (gate.failed) std.process.exit(1);
    if (!gate.gated and mode != .free and draft_steps == 0) std.process.exit(2);
}
