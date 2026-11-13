//! MiniMax-H3-Image's transformer on CUDA from a picture case: the free run of the distilled schedule,
//! the blocks on the GPU in the checkpoint's bf16, everything around them on the host at the Metal
//! kernels' rounding: rows in, the AdaLN tables, the final norm and each step's head, then the Euler step.
const std = @import("std");
const h3 = @import("h3cuda");
const h3image = @import("h3image");
const picture = h3image.picture;
const schedule = h3image.schedule;
const rows = h3image.rows;

const usage =
    \\usage: tf-h3-image-cuda MODEL_DIR picture.safetensors OUT_PREFIX
    \\  MODEL_DIR    the MiniMax-H3-Image folder: model_index.json, transformer/, pdd_config.json and its heads
    \\  picture      the prompt stage's case: canvas, text (the token refiner's rows), noise from the seed
    \\Writes OUT_PREFIX.rows.f32 (the target's rows) and OUT_PREFIX.latent.f32, as tf-h3-image's free run does.
    \\
;

fn read(init: std.process.Init, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(64 << 20)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

/// The bytes of a tensor as values; safetensors data need not be aligned.
fn values(comptime T: type, bytes: []const u8) []align(1) const T {
    return std.mem.bytesAsSlice(T, bytes);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("{s}", .{usage});
        return error.BadArguments;
    }
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const started = h3.seconds();

    // the folder: the image pipeline, an audio-free transformer, the distilled schedule and its heads
    const heads_name = try picture.checkIndex(a, try read(init, a, args[1], "model_index.json"));
    const transformer_dir = try std.fs.path.join(a, &.{ args[1], "transformer" });
    try picture.checkTransformer(a, try read(init, a, transformer_dir, "config.json"));
    const shard_count = try picture.shardCount(a, try read(init, a, transformer_dir, "model.safetensors.index.json"));
    const heads_dir = std.fs.path.dirname(try std.fs.path.join(a, &.{ args[1], heads_name })) orelse args[1];
    const pdd = try picture.pdd(a, try read(init, a, heads_dir, "pdd_config.json"));
    const steps = pdd.steps();

    var ck = try h3.checkpoint.Shards.open(gpa, transformer_dir, shard_count);
    defer ck.deinit(gpa);
    const model = try h3.Model.loadSharded(gpa, &ck);
    defer model.deinit();
    if (model.gated) return error.H3NotPictureModel; // a picture's blocks are dense: every tile sees every tile
    const hidden = model.hidden;

    // the case's rows: the canvas's target grid, the text, each reference's grid
    var case = try h3.checkpoint.Checkpoint.open(args[2]);
    defer case.deinit();
    const canvas_v = values(i32, try case.at("canvas", .i32, 2));
    const canvas = picture.Size{ .width = @intCast(@max(canvas_v[0], 0)), .height = @intCast(@max(canvas_v[1], 0)) };
    const target_grid = try picture.gridOf(canvas);
    const target = try picture.gridRows(target_grid);
    const text_entry = case.names.get("text") orelse return error.MissingTensor;
    if (text_entry.rank != 2 or text_entry.shape[1] != hidden) return error.BadTensor;
    const text = text_entry.shape[0];
    const text_rows = values(u16, try case.at("text", .bf16, text * hidden));
    var refs: []picture.Grid = &.{};
    if (case.names.get("references")) |e| {
        if (e.rank != 2 or e.shape[1] != 2) return error.BadTensor;
        const v = values(i32, try case.at("references", .i32, 2 * e.shape[0]));
        refs = try a.alloc(picture.Grid, e.shape[0]);
        for (refs, 0..) |*g, k| g.* = .{ .rows = @intCast(@max(v[2 * k], 0)), .columns = @intCast(@max(v[2 * k + 1], 0)) };
    }
    const layout = picture.Layout{ .text = text, .references = refs, .target = target_grid };
    const held = try layout.referenceRows();
    const count = try layout.rows();
    const anchors: ?[]align(1) const f32 = if (held > 0) values(f32, try case.at("anchors", .f32, held * picture.row_width)) else null;
    var tags: ?[]align(1) const i32 = null;
    if (case.names.get("text_tags") != null) {
        tags = values(i32, try case.at("text_tags", .i32, text));
        for (tags.?) |tag| if (tag != 0 and tag != 1) return error.BadTensor;
    }
    const noise = values(f32, try case.at("noise", .f32, target * picture.row_width));

    // positions and the rotary table from the checkpoint's frequencies, the tile map of a dense forward
    const pos = try picture.positions(a, layout);
    const rot = try picture.rotary(a, pos, try ck.floats(a, "rope.inv_freq"));
    const map = try picture.tiles(a, text + held, target);
    const tiles = map.prefix + map.target;
    // the blocks' rotary tables are 48 wide: the first half of each of picture.rotary's 96-wide rows
    const cos = try a.alloc(f32, count * picture.angles);
    const sin = try a.alloc(f32, count * picture.angles);
    for (0..count) |r| {
        @memcpy(cos[r * picture.angles ..][0..picture.angles], rot.cos[r * picture.rotated ..][0..picture.angles]);
        @memcpy(sin[r * picture.angles ..][0..picture.angles], rot.sin[r * picture.rotated ..][0..picture.angles]);
    }

    // the AdaLN tables: the embedder on the host, each block's projection at the Metal kernel's rounding,
    // laid out (line, part, hidden) in float32 as the CUDA blocks read them, line = timestep x 3 + modality
    const plan = try schedule.plan(a, .{ .text = text, .condition = held, .audio = 0, .video = target }, pdd.timesteps, pdd.timesteps);
    const times = plan.table.len;
    const lines = times * schedule.modalities;
    const embed = try rows.embed(a, plan.table, try embeddingDim(&ck), try ck.floats(a, "time_embedder.proj_in.weight"), try ck.floats(a, "time_embedder.proj_in.bias"), try ck.floats(a, "time_embedder.proj_out.weight"), try ck.floats(a, "time_embedder.proj_out.bias"));
    const k = embed.len / times;
    const block_table = lines * 6 * hidden;
    const tables = try a.alloc(f32, model.blocks.len * block_table);
    var name: [128]u8 = undefined;
    for (0..model.blocks.len) |i| {
        try rows.modulation(f32, embed, try ck.floats(a, try std.fmt.bufPrint(&name, "blocks.{d}.adaln_proj.linear.weight", .{i})), try ck.floats(a, try std.fmt.bufPrint(&name, "blocks.{d}.adaln_proj.linear.bias", .{i})), tables[i * block_table ..], times, k, schedule.modalities, 6, hidden);
    }
    const final = try a.alloc(u16, times * 2 * hidden);
    try rows.modulation(u16, embed, try ck.floats(a, "final_layer.adaln_proj.linear.weight"), try ck.floats(a, "final_layer.adaln_proj.linear.bias"), final, times, k, 1, 1, 2 * hidden);
    const final_norm = values(u16, try ck.at("final_layer.norm.weight", .bf16, hidden));

    // the step heads: one a step (a picture's distilled heads) or one shared, and the rows' projections
    const width = picture.row_width;
    const out_entry = ck.entry("video_out.weight") orelse return error.MissingTensor;
    const head_count: usize = if (out_entry.rank == 3) out_entry.shape[0] else 1;
    if (head_count != 1 and head_count != steps) return error.BadTensor;
    const w_out = try ck.floats(a, "video_out.weight");
    const b_out = try ck.floats(a, "video_out.bias");
    if (w_out.len != head_count * width * hidden or b_out.len != head_count * width) return error.BadTensor;
    const w_in = try ck.floats(a, "video_patch_proj.weight");
    const b_in = try ck.floats(a, "video_patch_proj.bias");

    // each row's line in the blocks' tables and in the final layer's, for every step of the schedule
    const adaln = try a.alloc(i32, steps * count);
    const line_of = try a.alloc(i32, steps * count);
    for (0..steps) |s| for (0..count) |r| {
        const time = plan.index[s * count + r];
        const tag: i32 = if (r < text) (if (tags) |t| t[r] else 1) else 0;
        adaln[s * count + r] = time * schedule.modalities + tag;
        line_of[s * count + r] = time;
    };

    // the packed stream, the video's own rows, and the final norm's output
    const x = try a.alloc(u16, count * hidden);
    const video = try a.alloc(f32, noise.len);
    @memcpy(video, noise);
    const velocity = try a.alloc(f32, target * width);
    const normed = try a.alloc(u16, count * hidden);
    var in: h3.Model.Inputs = .{
        .x = @intFromPtr(x.ptr),
        .tables = @intFromPtr(tables.ptr),
        .line = @intFromPtr(adaln.ptr),
        .cos = @intFromPtr(cos.ptr),
        .sin = @intFromPtr(sin.ptr),
        .slot = @intFromPtr(map.slot.ptr),
        .sizes = @intFromPtr(map.sizes.ptr),
        .rows = @intCast(count),
        .lines = @intCast(lines),
        .rot = picture.angles,
        .tiles = @intCast(tiles),
        .prefix_tiles = @intCast(map.prefix),
        .keep = 0,
    };
    std.debug.print("{d}x{d} canvas, {d} rows (text {d}, references {d}, target {d}), {d} tiles, {d} blocks, {d} steps, loaded in {d:.1} s\n", .{ canvas.width, canvas.height, count, text, held, target, tiles, model.blocks.len, steps, h3.seconds() - started });

    const transformer = h3.seconds();
    for (0..steps) |s| {
        const step = h3.seconds();
        // the stream: the text's rows, each reference's, then the video's through the float projection
        @memcpy(x[0 .. text * hidden], text_rows);
        if (anchors) |v| rows.rowsIn(v, w_in, b_in, x, text, held, width, hidden);
        rows.rowsIn(video, w_in, b_in, x, text + held, target, width, hidden);
        in.line = @intFromPtr(adaln[s * count ..].ptr);
        _ = try h3.host.forwardHost(model, &in);
        // the final norm over every row, then this step's head over the target's rows
        rows.finalNorm(x, final_norm, final, line_of[s * count ..], normed, hidden, h3.eps);
        const at = try h3image.geometry.headOffsets(head_count, s, steps, width, hidden);
        rows.rowsOut(normed, w_out[at[0] / 4 ..], b_out[at[1] / 4 ..], velocity, text + held, target, hidden, width);
        picture.euler(video, velocity, pdd.deltas[s]);
        std.debug.print("step {d}/{d}: {d:.2} s\n", .{ s + 1, steps, h3.seconds() - step });
    }
    std.debug.print("transformer: {d:.2} s\n", .{h3.seconds() - transformer});

    const rows_path = try std.fmt.allocPrintSentinel(a, "{s}.rows.f32", .{args[3]}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = rows_path, .data = std.mem.sliceAsBytes(video) });
    const latent = try a.alloc(f32, video.len);
    try picture.unpatchify(video, target_grid, latent);
    const latent_path = try std.fmt.allocPrintSentinel(a, "{s}.latent.f32", .{args[3]}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = latent_path, .data = std.mem.sliceAsBytes(latent) });
    std.debug.print("wrote {s}.latent.f32: [{d}, {d}, {d}] normalised float32\n", .{ args[3], picture.channels, target_grid.rows, target_grid.columns });
}

/// The embedder's width, checked against its four tensors before any allocation.
fn embeddingDim(ck: *const h3.checkpoint.Shards) !usize {
    const in_w = ck.entry("time_embedder.proj_in.weight") orelse return error.MissingTensor;
    const in_b = ck.entry("time_embedder.proj_in.bias") orelse return error.MissingTensor;
    const out_w = ck.entry("time_embedder.proj_out.weight") orelse return error.MissingTensor;
    const out_b = ck.entry("time_embedder.proj_out.bias") orelse return error.MissingTensor;
    return rows.shape.embedding(in_w, in_b, out_w, out_b, 1);
}
