//! MiniMax-H3-Image's transformer on CUDA from a picture case: the free run of the distilled schedule,
//! the blocks, timestep embedder, AdaLN tables, row projections and Euler updates on CUDA.
const std = @import("std");
const h3 = @import("h3cuda");
const h3image = h3.image_host;
const picture = h3image.picture;
const schedule = h3image.schedule;

const usage =
    \\usage: tf-h3-image-cuda MODEL_DIR picture.safetensors OUT_PREFIX [--bf16]
    \\       tf-h3-image-cuda MODEL_DIR --serve [--bf16]
    \\  MODEL_DIR    the MiniMax-H3-Image folder: model_index.json, transformer/, pdd_config.json and its heads
    \\  picture      the prompt stage's case: canvas, text (the token refiner's rows), noise from the seed
    \\Writes OUT_PREFIX.rows.f32 (the target's rows) and OUT_PREFIX.latent.f32, as tf-h3-image's free run does.
    \\--serve loads the model once, then renders one `picture.safetensors OUT_PREFIX` line of stdin at a time,
    \\answering each on stdout with `done OUT_PREFIX` or `failed OUT_PREFIX`.
    \\--bf16 runs every projection in bf16; by default qkvc and fc1 run as mmh3 does: rotated int8 weights and
    \\rows, exact int32 sums, and the picture passes episode 1's gates.
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

    fn load(init: std.process.Init, a: std.mem.Allocator, dir: []const u8, int8: bool) !Folder {
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
        if (int8) try h3.Int8.quantize(model);
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
    if (all.len == 2 and std.mem.eql(u8, all[1], "--self-test")) {
        try h3.modulation_test.run(init.gpa);
        return h3.rows_test.run(init.gpa);
    }
    const bf16 = all.len > 2 and std.mem.eql(u8, all[all.len - 1], "--bf16");
    const args = all[0 .. all.len - @intFromBool(bf16)];
    const serving = args.len == 3 and std.mem.eql(u8, args[2], "--serve");
    if (args.len != 4 and !serving) {
        std.debug.print("{s}", .{usage});
        return error.BadArguments;
    }
    const started = h3.seconds();
    var folder = try Folder.load(init, init.arena.allocator(), args[1], !bf16);
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

    // the packed stream on the device across every step; the host keeps only the latent's rows
    const x = try a.alloc(u16, count * hidden);
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
    var device_rows = try h3.Rows.init(model, .{
        .text = std.mem.sliceAsBytes(text_rows),
        .anchors = if (anchors) |v| std.mem.sliceAsBytes(v) else &.{},
        .noise = std.mem.sliceAsBytes(noise),
        .w_in = std.mem.sliceAsBytes(w_in),
        .b_in = std.mem.sliceAsBytes(b_in),
        .w_out = std.mem.sliceAsBytes(w_out),
        .b_out = std.mem.sliceAsBytes(b_out),
        .final_norm = std.mem.sliceAsBytes(final_norm),
        .final = std.mem.sliceAsBytes(final),
        .line_of = std.mem.sliceAsBytes(line_of),
        .text_rows = text,
        .held = held,
        .target = target,
        .width = width,
    });
    defer device_rows.deinit();
    std.debug.print("{d}x{d} canvas, {d} rows (text {d}, references {d}, target {d}), {d} tiles, {d} blocks, {d} steps, case ready in {d:.1} s\n", .{ canvas.width, canvas.height, count, text, held, target, tiles, model.blocks.len, steps, h3.seconds() - started });

    // the steps run on their own non-blocking stream so each captures as one CUDA graph; --serve's
    // next case shares this model, so its stream comes back when render returns
    var work = try @import("cuda").Stream.init(&model.driver, true);
    const kept = model.stream;
    defer {
        model.stream = kept;
        work.deinit();
    }
    model.stream = work;

    // every step's AdaLN lines staged once; each step's graph points `line` at its own slice
    var adaln_dev = try @import("cuda").DeviceBuffer.fromHost(&model.driver, std.mem.sliceAsBytes(adaln));
    defer adaln_dev.free();

    // stage() treats every address as host memory: the resident table stays out, x is room for the stream,
    // and `line` carries this step's count only (the validation reads one step's lines)
    const checked = try model.plan(&in);
    try model.drain();
    var upload = checked;
    upload.input[1] = 0;
    var device = try h3.host.stage(model, &in, upload);
    device.tables = modulation.blocks.ptr;
    try h3.geometry.prepare(model, &device, checked);
    try model.warm(count);
    try model.drain(); // the staging above is pageable host copies: the first launch must wait on them
    const transformer = h3.seconds();

    // H3_PROFILE keeps the eager path: its per-stage waits can't be captured
    const profiling = if (init.environ_map.get("H3_PROFILE")) |v| v.len > 0 else false;
    model.profile = profiling;
    model.spent = @splat(0);
    const graphs = !profiling;
    var execs = try a.alloc(?@import("cuda").graph.Exec, steps);
    defer {
        for (execs) |*e| if (e.*) |*ex| ex.deinit();
        a.free(execs);
    }
    for (execs) |*e| e.* = null;
    for (0..steps) |s| {
        const mark = h3.seconds();
        device.line = adaln_dev.ptr + s * count * 4;
        if (graphs) {
            if (execs[s] == null) execs[s] = stepGraph(model, &device, checked, &device_rows, s, pdd.deltas[s]) catch |err| blk: {
                std.debug.print("step {d} runs without its graph: {s}\n", .{ s + 1, @errorName(err) });
                break :blk null;
            };
            if (execs[s]) |e| {
                try e.launchOn(work);
            } else try stepBody(model, &device, checked, &device_rows, s, pdd.deltas[s]);
        } else try stepBody(model, &device, checked, &device_rows, s, pdd.deltas[s]);
        try model.stream.synchronize();
        std.debug.print("step {d}/{d}: {d:.2} s\n", .{ s + 1, steps, h3.seconds() - mark });
    }
    const video = try a.alloc(f32, noise.len);
    try device_rows.latent(video);
    const spent = h3.seconds() - transformer;
    std.debug.print("transformer: {d:.2} s\n", .{spent});
    if (profiling) {
        var blocks: f64 = 0;
        for (h3.stages, model.spent) |name, s| {
            std.debug.print("  {s}: {d:.2} s\n", .{ name, s });
            blocks += s;
        }
        std.debug.print("  outside the blocks: {d:.2} s\n", .{spent - blocks});
    }

    const rows_path = try std.fmt.allocPrintSentinel(a, "{s}.rows.f32", .{prefix}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = rows_path, .data = std.mem.sliceAsBytes(video) });
    const latent = try a.alloc(f32, video.len);
    try picture.unpatchify(video, target_grid, latent);
    const latent_path = try std.fmt.allocPrintSentinel(a, "{s}.latent.f32", .{prefix}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = latent_path, .data = std.mem.sliceAsBytes(latent) });
    std.debug.print("wrote {s}.latent.f32: [{d}, {d}, {d}] normalised float32\n", .{ prefix, picture.channels, target_grid.rows, target_grid.columns });
}

/// The whole step captured once: rows in, the blocks' launches, the step's head and the Euler update.
/// `device.line` already points at step `s`'s AdaLN lines; the graph replays those pointers.
fn stepGraph(model: *h3.Model, device: *const h3.Model.Inputs, checked: h3.Model.Plan, rows: *const h3.Rows, s: usize, delta: f32) !@import("cuda").graph.Exec {
    const graph_api = @import("cuda").graph;
    try graph_api.beginCapture(model.stream, .thread_local);
    stepBody(model, device, checked, rows, s, delta) catch |err| {
        if (graph_api.endCapture(model.stream)) |g| {
            var dropped = g;
            dropped.deinit();
        } else |_| {}
        return err;
    };
    var graph = try graph_api.endCapture(model.stream);
    defer graph.deinit();
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(model.stream);
    return exec;
}

fn stepBody(model: *h3.Model, device: *const h3.Model.Inputs, checked: h3.Model.Plan, rows: *const h3.Rows, s: usize, delta: f32) !void {
    try rows.in(device.x);
    try model.step(device, checked);
    try rows.out(device.x, s, delta);
}
