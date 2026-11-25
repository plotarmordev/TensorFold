//! MiniMax-H3-Image's transformer on CUDA from a picture case: the free run of the distilled schedule,
//! the blocks, timestep embedder and AdaLN tables on CUDA; row projections and Euler updates on the host.
const std = @import("std");
const h3 = @import("h3cuda");
const h3image = h3.image_host;
const picture = h3image.picture;
const schedule = h3image.schedule;
const rows = h3image.rows;

const usage =
    \\usage: tf-h3-image-cuda MODEL_DIR picture.safetensors OUT_PREFIX
    \\       tf-h3-image-cuda MODEL_DIR --serve
    \\       tf-h3-image-cuda --self-test
    \\  MODEL_DIR    the MiniMax-H3-Image folder: model_index.json, transformer/, pdd_config.json and its heads
    \\  picture      the prompt stage's case: canvas, text (the token refiner's rows), noise from the seed
    \\Writes OUT_PREFIX.rows.f32 (the target's rows) and OUT_PREFIX.latent.f32, as tf-h3-image's free run does.
    \\--serve loads the model once, then renders one `picture.safetensors OUT_PREFIX` line of stdin at a time,
    \\answering each on stdout with `done OUT_PREFIX` or `failed OUT_PREFIX`.
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

/// The folder's transformer, schedule and step heads, loaded once for every picture.
const Folder = struct {
    ck: h3.checkpoint.Shards,
    model: *h3.Model,
    heads: h3.checkpoint.Checkpoint,
    pdd: picture.Pdd,
    modulation: ?h3.modulation.Tables = null,
    times: []f32 = &.{},

    fn load(init: std.process.Init, a: std.mem.Allocator, dir: []const u8) !Folder {
        const heads_name = try picture.checkIndex(a, try read(init, a, dir, "model_index.json"));
        const transformer_dir = try std.fs.path.join(a, &.{ dir, "transformer" });
        try picture.checkTransformer(a, try read(init, a, transformer_dir, "config.json"));
        const shard_count = try picture.shardCount(a, try read(init, a, transformer_dir, "model.safetensors.index.json"));
        const heads_dir = std.fs.path.dirname(try std.fs.path.join(a, &.{ dir, heads_name })) orelse dir;
        const pdd = try picture.pdd(a, try read(init, a, heads_dir, "pdd_config.json"));
        var ck = try h3.checkpoint.Shards.open(init.gpa, transformer_dir, shard_count);
        errdefer ck.deinit(init.gpa);
        const model = try h3.Model.loadSharded(init.gpa, &ck);
        errdefer model.deinit();
        if (model.gated) return error.H3NotPictureModel; // a picture's blocks are dense: every tile sees every tile
        ck.evict(); // the prompt and decoder stages share this memory; each render evicts what it read back
        const heads = try h3.checkpoint.Checkpoint.open(try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, heads_name }, 0));
        return .{ .ck = ck, .model = model, .heads = heads, .pdd = pdd };
    }

    fn deinit(f: *Folder, gpa: std.mem.Allocator) void {
        if (f.modulation) |*m| m.deinit();
        gpa.free(f.times);
        f.heads.deinit();
        f.model.deinit();
        f.ck.deinit(gpa);
    }

    /// The AdaLN tables for `times`, kept across --serve pictures: they depend on the timesteps alone, and preparing
    /// them reads ~26 GB of projection weights. A case with other timesteps (a reference's keyframe row) replaces them.
    fn tables(f: *Folder, gpa: std.mem.Allocator, times: []const f32) !*h3.modulation.Tables {
        if (f.modulation != null and std.mem.eql(f32, f.times, times)) return &f.modulation.?;
        if (f.modulation) |*m| m.deinit();
        f.modulation = null;
        gpa.free(f.times);
        f.times = &.{};
        f.modulation = try h3.modulation.prepare(gpa, &f.model.driver, f.model.stream, &f.ck, times, f.model.blocks.len, f.model.hidden);
        f.times = try gpa.dupe(f32, times);
        return &f.modulation.?;
    }
};

pub fn main(init: std.process.Init) !void {
    const all = try init.minimal.args.toSlice(init.arena.allocator());
    if (all.len == 2 and std.mem.eql(u8, all[1], "--self-test")) return h3.modulation_test.run(init.gpa);
    const args = all;
    const serving = args.len == 3 and std.mem.eql(u8, args[2], "--serve");
    if (args.len != 4 and !serving) {
        std.debug.print("{s}", .{usage});
        return error.BadArguments;
    }
    const started = h3.seconds();
    var folder = try Folder.load(init, init.arena.allocator(), args[1]);
    defer folder.deinit(init.gpa);
    std.debug.print("model loaded in {d:.1} s\n", .{h3.seconds() - started});
    if (!serving) return render(init, init.arena.allocator(), &folder, args[2], args[3]);
    var in_buf: [4096]u8 = undefined;
    var in = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    while (try in.interface.takeDelimiter('\n')) |line| {
        var it = std.mem.tokenizeAny(u8, line, " \t\r");
        const case_path = it.next() orelse continue;
        const prefix = it.next() orelse case_path;
        var request = std.heap.ArenaAllocator.init(init.gpa);
        defer request.deinit();
        const verdict = if (render(init, request.allocator(), &folder, case_path, prefix)) "done" else |err| blk: {
            std.debug.print("{s} failed: {s}\n", .{ prefix, @errorName(err) });
            break :blk "failed";
        };
        try std.Io.File.stdout().writeStreamingAll(init.io, try std.fmt.allocPrint(request.allocator(), "{s} {s}\n", .{ verdict, prefix }));
    }
}

/// One picture case through the distilled schedule on the loaded folder: its rows and latent next to OUT_PREFIX.
fn render(init: std.process.Init, a: std.mem.Allocator, folder: *Folder, case_path: []const u8, prefix: []const u8) !void {
    const gpa = init.gpa;
    const started = h3.seconds();
    const model = folder.model;
    const ck = &folder.ck;
    const pdd = folder.pdd;
    const steps = pdd.steps();
    const hidden = model.hidden;

    // the case's rows: the canvas's target grid, the text, each reference's grid
    var case = try h3.checkpoint.Checkpoint.open(try a.dupeSentinel(u8, case_path, 0));
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

    // AdaLN tables stay resident across every denoising step; only the final host norm's table returns.
    const plan = try schedule.plan(a, .{ .text = text, .condition = held, .audio = 0, .video = target }, pdd.timesteps, pdd.timesteps);
    const times = plan.table.len;
    const lines = times * schedule.modalities;
    const modulation = try folder.tables(gpa, plan.table);
    const final = modulation.final;
    const final_norm = values(u16, try ck.at("final_layer.norm.weight", .bf16, hidden));
    ck.evict(); // the weights this case read (~26 GB when it prepared the tables): a cached copy would crowd the next prompt

    // the step heads: the picture's distilled heads, one a step, in model_index.json's heads file
    const width = picture.row_width;
    const heads = &folder.heads;
    const w_out = values(f32, try heads.at("video_out.weight", .f32, steps * width * hidden));
    const b_out = values(f32, try heads.at("video_out.bias", .f32, steps * width));
    const w_in = values(f32, try ck.at("video_patch_proj.weight", .f32, hidden * width));
    const b_in = values(f32, try ck.at("video_patch_proj.bias", .f32, hidden));

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
        .tables = modulation.blocks.ptr,
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
    std.debug.print("{d}x{d} canvas, {d} rows (text {d}, references {d}, target {d}), {d} tiles, {d} blocks, {d} steps, case ready in {d:.1} s\n", .{ canvas.width, canvas.height, count, text, held, target, tiles, model.blocks.len, steps, h3.seconds() - started });

    const transformer = h3.seconds();
    for (0..steps) |s| {
        const step = h3.seconds();
        // the stream: the text's rows, each reference's, then the video's through the float projection
        @memcpy(x[0 .. text * hidden], text_rows);
        if (anchors) |v| rows.rowsIn(v, w_in, b_in, x, text, held, width, hidden);
        rows.rowsIn(video, w_in, b_in, x, text + held, target, width, hidden);
        in.line = @intFromPtr(adaln[s * count ..][0..count].ptr);
        const checked = try model.plan(&in);
        try model.drain();
        // stage() treats every address as host memory; stage the small inputs without the resident table.
        var upload = checked;
        upload.input[1] = 0;
        var device = try h3.host.stage(model, &in, upload);
        device.tables = modulation.blocks.ptr;
        _ = try model.forward(&device);
        try h3.host.readback(model, &in, checked);
        // the final norm over every row, then this step's head over the target's rows
        rows.finalNorm(x, final_norm, final, line_of[s * count ..][0..count], normed, hidden, h3.eps);
        rows.rowsOut(normed, w_out[s * width * hidden ..], b_out[s * width ..], velocity, text + held, target, hidden, width);
        picture.euler(video, velocity, pdd.deltas[s]);
        std.debug.print("step {d}/{d}: {d:.2} s\n", .{ s + 1, steps, h3.seconds() - step });
    }
    std.debug.print("transformer: {d:.2} s\n", .{h3.seconds() - transformer});

    const rows_path = try std.fmt.allocPrintSentinel(a, "{s}.rows.f32", .{prefix}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = rows_path, .data = std.mem.sliceAsBytes(video) });
    const latent = try a.alloc(f32, video.len);
    try picture.unpatchify(video, target_grid, latent);
    const latent_path = try std.fmt.allocPrintSentinel(a, "{s}.latent.f32", .{prefix}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = latent_path, .data = std.mem.sliceAsBytes(latent) });
    std.debug.print("wrote {s}.latent.f32: [{d}, {d}, {d}] normalised float32\n", .{ prefix, picture.channels, target_grid.rows, target_grid.columns });
}
