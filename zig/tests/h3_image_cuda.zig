//! MiniMax-H3-Image's transformer on CUDA from a picture case: the free run of the distilled schedule,
//! the blocks, timestep embedder, AdaLN tables, row projections and Euler updates on CUDA.
const std = @import("std");
const h3 = @import("h3cuda");
const h3image = h3.image_host;
const picture = h3image.picture;
const schedule = h3image.schedule;
const cascade = h3image.cascade;

const usage =
    \\usage: tf-h3-image-cuda MODEL_DIR picture.safetensors OUT_PREFIX [--bf16|--fp4] [--load-threads N] [--draft-steps N] [--renoise-sigma X] [--routed-steps N]
    \\       tf-h3-image-cuda MODEL_DIR --serve [--bf16|--fp4] [--load-threads N] [--draft-steps N] [--renoise-sigma X] [--routed-steps N]
    \\  MODEL_DIR    the MiniMax-H3-Image folder: model_index.json, transformer/, pdd_config.json and its heads
    \\  picture      the prompt stage's case: canvas, text (the token refiner's rows), noise from the seed
    \\Writes OUT_PREFIX.rows.f32 (the target's rows) and OUT_PREFIX.latent.f32, as tf-h3-image's free run does.
    \\--serve loads the model once, then renders one `picture.safetensors OUT_PREFIX` line of stdin at a time,
    \\answering each on stdout with `done OUT_PREFIX` or `failed OUT_PREFIX`.
    \\--bf16 runs every projection in bf16; by default qkvc and fc1 run as mmh3 does: rotated int8 weights and
    \\rows, exact int32 sums, and the picture passes episode 1's gates. --fp4 instead quantizes every projection
    \\to block-scaled NVFP4 (e2m1, one e4m3 scale every 16 values) on GB10's FP4 tensor cores: faster products,
    \\a different picture — its numbers and gates are the recipe's, and it never runs unless asked.
    \\--draft-steps N runs the first N of the checkpoint's steps on the half-size canvas, as mmh3's
    \\--draft-steps does: the clean draft is enlarged bilinearly, noised to the next step's sigma (or
    \\to --renoise-sigma X) and redrawn at full size for the rest. It draws a different picture, not a
    \\faster copy of the six-step one.
    \\--load-threads N reads the transformer's weights with N reader threads straight into the pinned
    \\stages that alternate with the GPU upload, as mmh3's eight-thread loader does; 0 (default) reads
    \\the shards in place.
    \\--routed-steps N routes the last N steps' attention as FastH3 does: each picture tile attends the prompt's
    \\tiles and the fifth of the picture tiles its pooled query scores best (h3_topk), every other tile dropped.
    \\It draws a different picture; 0 (default) keeps every step dense.
;

/// The transformer's linear layers: the int8 default, all-bf16, or the opt-in block-scaled NVFP4 products.
const Linear = enum { int8, bf16, fp4 };

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
    /// The last steps of a picture whose attention is routed (`--routed-steps`); 0 keeps every step dense.
    routed: usize = 0,

    fn load(init: std.process.Init, a: std.mem.Allocator, dir: []const u8, linear: Linear, threads: usize) !Folder {
        const heads_name = try picture.checkIndex(a, try read(init, a, dir, "model_index.json"));
        const transformer_dir = try std.fs.path.join(a, &.{ dir, "transformer" });
        try picture.checkTransformer(a, try read(init, a, transformer_dir, "config.json"));
        const shard_count = try picture.shardCount(a, try read(init, a, transformer_dir, "model.safetensors.index.json"));
        const heads_dir = std.fs.path.dirname(try std.fs.path.join(a, &.{ dir, heads_name })) orelse dir;
        const pdd = try picture.pdd(a, try read(init, a, heads_dir, "pdd_config.json"));
        var ck = try h3.checkpoint.Shards.open(init.gpa, transformer_dir, shard_count);
        errdefer ck.deinit(init.gpa);
        const model = try h3.Model.loadSharded(init.gpa, &ck, threads);
        errdefer model.deinit();
        if (model.gated) return error.H3NotPictureModel; // a picture's blocks are dense: every tile sees every tile
        if (linear == .int8) try h3.Int8.quantize(model);
        if (linear == .fp4) try h3.Fp4.quantize(model);
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
        try h3.fused_test.run(init.gpa);
        return h3.rows_test.run(init.gpa);
    }
    var draft_steps: usize = 0;
    var routed_steps: usize = 0;
    var load_threads: usize = 0;
    var renoise: ?f32 = null;
    var linear: Linear = .int8;
    var args: std.ArrayList([]const u8) = .empty;
    var i: usize = 1;
    while (i < all.len) : (i += 1) {
        const arg = all[i];
        if (std.mem.eql(u8, arg, "--bf16")) {
            if (linear != .int8) return error.BadArguments;
            linear = .bf16;
        } else if (std.mem.eql(u8, arg, "--fp4")) {
            if (linear != .int8) return error.BadArguments;
            linear = .fp4;
        } else if (std.mem.eql(u8, arg, "--load-threads")) {
            i += 1;
            if (i >= all.len) return error.BadArguments;
            load_threads = std.fmt.parseInt(usize, all[i], 10) catch return error.BadArguments;
        } else if (std.mem.eql(u8, arg, "--draft-steps")) {
            i += 1;
            if (i >= all.len) return error.BadArguments;
            draft_steps = std.fmt.parseInt(usize, all[i], 10) catch return error.BadArguments;
        } else if (std.mem.eql(u8, arg, "--renoise-sigma")) {
            i += 1;
            if (i >= all.len) return error.BadArguments;
            renoise = std.fmt.parseFloat(f32, all[i]) catch return error.BadArguments;
        } else if (std.mem.eql(u8, arg, "--routed-steps")) {
            i += 1;
            if (i >= all.len) return error.BadArguments;
            routed_steps = std.fmt.parseInt(usize, all[i], 10) catch return error.BadArguments;
        } else try args.append(init.arena.allocator(), arg);
    }
    const serving = args.items.len == 2 and std.mem.eql(u8, args.items[1], "--serve");
    if (args.items.len != 3 and !serving) {
        std.debug.print("{s}", .{usage});
        return error.BadArguments;
    }
    if (draft_steps == 0 and renoise != null) return error.BadArguments; // --renoise-sigma noises a draft
    const started = h3.seconds();
    var folder = try Folder.load(init, init.arena.allocator(), args.items[0], linear, load_threads);
    defer folder.deinit(init.gpa);
    folder.routed = routed_steps;
    std.debug.print("model loaded in {d:.1} s\n", .{h3.seconds() - started});
    if (!serving) return render(init, init.arena.allocator(), &folder, args.items[1], args.items[2], draft_steps, renoise);
    var in_buf: [4096]u8 = undefined;
    var in = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    while (try in.interface.takeDelimiter('\n')) |line| {
        var it = std.mem.tokenizeAny(u8, line, " \t\r");
        const case_path = it.next() orelse continue;
        const prefix = it.next() orelse case_path;
        var request = std.heap.ArenaAllocator.init(init.gpa);
        defer request.deinit();
        const verdict = if (render(init, request.allocator(), &folder, case_path, prefix, draft_steps, renoise)) "done" else |err| blk: {
            std.debug.print("{s} failed: {s}\n", .{ prefix, @errorName(err) });
            break :blk "failed";
        };
        try std.Io.File.stdout().writeStreamingAll(init.io, try std.fmt.allocPrint(request.allocator(), "{s} {s}\n", .{ verdict, prefix }));
    }
}

/// The tensors of a picture case every phase shares: its canvas's full-size target grid, the text
/// rows, the references' grids and anchors, and the seed's noise for the full target.
const Case = struct {
    canvas: picture.Size,
    target_grid: picture.Grid,
    text: usize,
    text_rows: []align(1) const u16,
    refs: []picture.Grid,
    held: usize,
    anchors: ?[]align(1) const f32,
    tags: ?[]align(1) const i32,
    noise: []align(1) const f32,
    target: usize,

    fn load(a: std.mem.Allocator, case_path: []const u8, hidden: usize) !struct { case: Case, file: h3.checkpoint.Checkpoint } {
        var file = try h3.checkpoint.Checkpoint.open(try a.dupeSentinel(u8, case_path, 0));
        errdefer file.deinit();
        const canvas_v = values(i32, try file.at("canvas", .i32, 2));
        const canvas = picture.Size{ .width = @intCast(@max(canvas_v[0], 0)), .height = @intCast(@max(canvas_v[1], 0)) };
        const target_grid = try picture.gridOf(canvas);
        const target = try picture.gridRows(target_grid);
        const text_entry = file.names.get("text") orelse return error.MissingTensor;
        if (text_entry.rank != 2 or text_entry.shape[1] != hidden) return error.BadTensor;
        const text = text_entry.shape[0];
        const text_rows = values(u16, try file.at("text", .bf16, text * hidden));
        var refs: []picture.Grid = &.{};
        if (file.names.get("references")) |e| {
            if (e.rank != 2 or e.shape[1] != 2) return error.BadTensor;
            const v = values(i32, try file.at("references", .i32, 2 * e.shape[0]));
            refs = try a.alloc(picture.Grid, e.shape[0]);
            for (refs, 0..) |*g, k| g.* = .{ .rows = @intCast(@max(v[2 * k], 0)), .columns = @intCast(@max(v[2 * k + 1], 0)) };
        }
        const held = try (picture.Layout{ .text = text, .references = refs, .target = target_grid }).referenceRows();
        const anchors: ?[]align(1) const f32 = if (held > 0) values(f32, try file.at("anchors", .f32, held * picture.row_width)) else null;
        var tags: ?[]align(1) const i32 = null;
        if (file.names.get("text_tags") != null) {
            tags = values(i32, try file.at("text_tags", .i32, text));
            for (tags.?) |tag| if (tag != 0 and tag != 1) return error.BadTensor;
        }
        const noise = values(f32, try file.at("noise", .f32, target * picture.row_width));
        return .{ .case = .{ .canvas = canvas, .target_grid = target_grid, .text = text, .text_rows = text_rows, .refs = refs, .held = held, .anchors = anchors, .tags = tags, .noise = noise, .target = target }, .file = file };
    }
};

/// Steps `first .. first + run` of `sched` over one stream on the device: the grid's rows in through
/// the blocks, each step's head and its Euler update, each step captured as one CUDA graph. The
/// video rows start from `video` (target, row_width) f32: the case's pooled noise for a fresh picture,
/// the lifted draft for the full-size steps after drafts. `lift_last` makes the last step compute only
/// its velocity, downloaded into `velocity_out` for cascade.lift. Returns the stream's video rows.
fn streamSteps(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    folder: *Folder,
    case: Case,
    grid: picture.Grid,
    video: []align(1) const f32,
    sched: picture.Pdd,
    spans: []const picture.HeadSpan,
    first: usize,
    run: usize,
    lift_last: bool,
    velocity_out: ?[]f32,
    graphs: bool,
) ![]f32 {
    const model = folder.model;
    const ck = &folder.ck;
    const hidden = model.hidden;
    const steps = sched.steps();
    const target = try picture.gridRows(grid);
    const layout = picture.Layout{ .text = case.text, .references = case.refs, .target = grid };
    const count = try layout.rows();
    const pos = try picture.positions(a, layout);
    const rot = try picture.rotary(a, pos, try ck.floats(a, "rope.inv_freq"));
    const map = try picture.tiles(a, case.text + case.held, target);
    const tiles = map.prefix + map.target;
    // the blocks' rotary tables are 48 wide: the first half of each of picture.rotary's 96-wide rows
    const cos = try a.alloc(f32, count * picture.angles);
    const sin = try a.alloc(f32, count * picture.angles);
    for (0..count) |r| {
        @memcpy(cos[r * picture.angles ..][0..picture.angles], rot.cos[r * picture.rotated ..][0..picture.angles]);
        @memcpy(sin[r * picture.angles ..][0..picture.angles], rot.sin[r * picture.rotated ..][0..picture.angles]);
    }

    // AdaLN tables stay resident across every denoising step and both draft phases; only the final
    // host norm's table returns. The plan's table depends on the timesteps alone, so a phase that
    // adds no timestep reuses it.
    const plan = try schedule.plan(a, .{ .text = case.text, .condition = case.held, .audio = 0, .video = target }, sched.timesteps, sched.timesteps);
    const times = plan.table.len;
    const lines = times * schedule.modalities;
    const modulation = try folder.tables(gpa, plan.table);
    const final = modulation.final;
    const final_norm = values(u16, try ck.at("final_layer.norm.weight", .bf16, hidden));
    ck.evict(); // the weights this case read (~26 GB when it prepared the tables): a cached copy would crowd the next prompt

    // the step heads: the picture's distilled heads, one a step, in model_index.json's heads file
    const width = picture.row_width;
    const heads = &folder.heads;
    const head_count: usize = @intCast((heads.names.get("video_out.weight") orelse return error.MissingTensor).shape[0]);
    const w_out = values(f32, try heads.at("video_out.weight", .f32, head_count * width * hidden));
    const b_out = values(f32, try heads.at("video_out.bias", .f32, head_count * width));
    const w_in = values(f32, try ck.at("video_patch_proj.weight", .f32, hidden * width));
    const b_in = values(f32, try ck.at("video_patch_proj.bias", .f32, hidden));

    // each row's line in the blocks' tables and in the final layer's, for every step of the schedule
    const adaln = try a.alloc(i32, steps * count);
    const line_of = try a.alloc(i32, steps * count);
    for (0..steps) |s| for (0..count) |r| {
        const time = plan.index[s * count + r];
        const tag: i32 = if (r < case.text) (if (case.tags) |t| t[r] else 1) else 0;
        adaln[s * count + r] = time * schedule.modalities + tag;
        line_of[s * count + r] = time;
    };

    // the packed stream on the device across this phase's steps; the host keeps only the video rows
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
        .text = std.mem.sliceAsBytes(case.text_rows),
        .anchors = if (case.anchors) |v| std.mem.sliceAsBytes(v) else &.{},
        .noise = std.mem.sliceAsBytes(video),
        .w_in = std.mem.sliceAsBytes(w_in),
        .b_in = std.mem.sliceAsBytes(b_in),
        .w_out = std.mem.sliceAsBytes(w_out),
        .b_out = std.mem.sliceAsBytes(b_out),
        .final_norm = std.mem.sliceAsBytes(final_norm),
        .final = std.mem.sliceAsBytes(final),
        .line_of = std.mem.sliceAsBytes(line_of),
        .text_rows = case.text,
        .held = case.held,
        .target = target,
        .width = width,
    });
    defer device_rows.deinit();

    // every step's AdaLN lines staged once; each step's graph points `line` at its own slice
    var adaln_dev = try @import("cuda").DeviceBuffer.fromHost(&model.driver, std.mem.sliceAsBytes(adaln));
    defer adaln_dev.free();

    // stage() treats every address as host memory: the resident table stays out, x is room for the
    // stream, and `line` carries this step's count only (the validation reads one step's lines)
    const checked = try model.plan(&in);
    // the routed steps' plan: a picture tile keeps the prompt's tiles and a fifth of the picture's, as FastH3 keeps;
    // the geometry is sized for it, and the dense steps read none of its routing buffers
    var routed_in = in;
    routed_in.keep = @intCast((map.target + 4) / 5);
    const routed = try model.plan(&routed_in);
    const routed_from = steps - folder.routed;
    const geometry = if (first + run > routed_from) routed else checked;
    try model.drain();
    var upload = checked;
    upload.input[1] = 0;
    var device = try h3.host.stage(model, &in, upload);
    device.tables = modulation.blocks.ptr;
    try h3.geometry.prepare(model, &device, geometry);
    try model.warm(count);
    try model.drain(); // the staging above is pageable host copies: the first launch must wait on them

    for (first..first + run) |s| {
        const mark = h3.seconds();
        device.line = adaln_dev.ptr + s * count * 4;
        const last = lift_last and s == first + run - 1;
        const step_plan = if (s >= routed_from) routed else checked;
        // the exec outlives its in-flight launch: graph execs are destroyed only after the stream syncs
        var launched: ?@import("cuda").graph.Exec = null;
        defer if (launched) |*e| e.deinit();
        if (graphs) {
            const exec = stepGraph(model, &device, step_plan, &device_rows, s, sched.deltas[s], spans[s], last) catch |err| blk: {
                std.debug.print("step {d} runs without its graph: {s}\n", .{ s + 1, @errorName(err) });
                break :blk null;
            };
            if (exec) |e| {
                try e.launchOn(model.stream);
                launched = e;
            } else try stepBody(model, &device, step_plan, &device_rows, s, sched.deltas[s], spans[s], last);
        } else try stepBody(model, &device, step_plan, &device_rows, s, sched.deltas[s], spans[s], last);
        try model.stream.synchronize();
        std.debug.print("step {d}/{d}{s}{s}: {d:.2} s\n", .{ s + 1, steps, if (last) " (draft's clean)" else "", if (s >= routed_from) " (routed)" else "", h3.seconds() - mark });
    }
    const rows = try a.alloc(f32, video.len);
    try device_rows.latent(rows);
    if (velocity_out) |v| try device_rows.lastVelocity(v);
    return rows;
}

/// One picture case through the distilled schedule on the loaded folder: its rows and latent next to OUT_PREFIX.
fn render(init: std.process.Init, a: std.mem.Allocator, folder: *Folder, case_path: []const u8, prefix: []const u8, draft_steps: usize, renoise: ?f32) !void {
    const started = h3.seconds();
    const model = folder.model;
    const pdd = folder.pdd;
    const hidden = model.hidden;

    const loaded = try Case.load(a, case_path, hidden);
    var case_file = loaded.file;
    defer case_file.deinit();
    const case = loaded.case;
    const width = picture.row_width;
    std.debug.print("{d}x{d} canvas, {d} steps, case ready in {d:.1} s\n", .{ case.canvas.width, case.canvas.height, pdd.steps(), h3.seconds() - started });

    // the run's schedule: the checkpoint's six distilled sigmas, or mmh3's draft schedule — the first
    // `draft_steps` steps on the half-size canvas, then the rest at full size on the enlarged draft,
    // noised to `renoise` (or to the next sigma without one)
    const sched = if (draft_steps == 0) pdd else try picture.draftSchedule(a, pdd, draft_steps, renoise);
    const steps = sched.steps();
    if (folder.routed > steps) return error.BadArguments;
    const spans = try a.alloc(picture.HeadSpan, steps);
    for (spans, 0..) |*sp, s| sp.* = try picture.headSpan(pdd.sigmas, sched.sigmas, s);
    // cascade.half enforces the half-canvas grid: both sides multiples of 64 pixels (mmh3's rule)
    const full_grid = cascade.Grid{ .rows = case.target_grid.rows, .columns = case.target_grid.columns };
    const draft_grid = if (draft_steps > 0) try cascade.half(full_grid) else undefined;
    if (draft_steps > 0 and case.held > 0) return error.H3DraftWithReferences;

    // the steps run on their own non-blocking stream so each captures as one CUDA graph; --serve's
    // next case shares this model, so its stream comes back when render returns
    var work = try @import("cuda").Stream.init(&model.driver, true);
    const kept = model.stream;
    defer {
        model.stream = kept;
        work.deinit();
    }
    model.stream = work;

    // H3_PROFILE keeps the eager path: its per-stage waits can't be captured
    const profiling = if (init.environ_map.get("H3_PROFILE")) |v| v.len > 0 else false;
    model.profile = profiling;
    model.spent = @splat(0);
    const graphs = !profiling;
    const transformer = h3.seconds();

    var video: []f32 = undefined;
    if (draft_steps == 0) {
        video = try streamSteps(a, init.gpa, folder, case, case.target_grid, case.noise, sched, spans, 0, steps, false, null, graphs);
    } else {
        // cascade.pool draws the draft's half-grid noise from the seed's full draw, so both canvases
        // share one seed's picture; case.noise's mmap'd rows copy to an aligned buffer, then
        // unpatchify to the [channels, rows, columns] latent pool and lift work on
        const noise_aligned = try a.alloc(f32, case.noise.len);
        @memcpy(std.mem.sliceAsBytes(noise_aligned), std.mem.sliceAsBytes(case.noise));
        const noise_latent = try a.alloc(f32, picture.channels * full_grid.rows * full_grid.columns);
        try picture.unpatchify(noise_aligned, case.target_grid, noise_latent);
        const draft_target = try picture.gridRows(.{ .rows = draft_grid.rows, .columns = draft_grid.columns });
        const draft_latent = try a.alloc(f32, picture.channels * draft_grid.rows * draft_grid.columns);
        try cascade.pool(noise_latent, full_grid, draft_latent);
        const draft_noise = try a.alloc(f32, draft_target * width);
        try picture.patchify(draft_latent, .{ .rows = draft_grid.rows, .columns = draft_grid.columns }, draft_noise);

        // the draft steps leave their video at sigma[draft_steps-1] and their last velocity out for
        // cascade.lift to predict the clean latent and renoise it to sigma[draft_steps] at full size
        const draft_vel = try a.alloc(f32, draft_target * width);
        const draft = try streamSteps(a, init.gpa, folder, case, .{ .rows = draft_grid.rows, .columns = draft_grid.columns }, draft_noise, sched, spans, 0, draft_steps, true, draft_vel, graphs);
        std.debug.print("{d}x{d} draft done, lifting\n", .{ case.canvas.width / 2, case.canvas.height / 2 });
        const x_latent = try a.alloc(f32, draft_latent.len);
        const v_latent = try a.alloc(f32, draft_latent.len);
        try picture.unpatchify(draft, .{ .rows = draft_grid.rows, .columns = draft_grid.columns }, x_latent);
        try picture.unpatchify(draft_vel, .{ .rows = draft_grid.rows, .columns = draft_grid.columns }, v_latent);
        const lifted = try a.alloc(f32, noise_latent.len);
        try cascade.lift(x_latent, v_latent, sched.sigmas[draft_steps - 1], sched.sigmas[draft_steps], noise_latent, full_grid, lifted);
        video = try a.alloc(f32, case.target * width);
        try picture.patchify(lifted, case.target_grid, video);
        video = try streamSteps(a, init.gpa, folder, case, case.target_grid, video, sched, spans, draft_steps, steps - draft_steps, false, null, graphs);
    }
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
    try picture.unpatchify(video, case.target_grid, latent);
    const latent_path = try std.fmt.allocPrintSentinel(a, "{s}.latent.f32", .{prefix}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = latent_path, .data = std.mem.sliceAsBytes(latent) });
    std.debug.print("wrote {s}.latent.f32: [{d}, {d}, {d}] normalised float32\n", .{ prefix, picture.channels, case.target_grid.rows, case.target_grid.columns });
}

/// The whole step captured once: rows in, the blocks' launches, the step's head and the Euler update.
/// `device.line` already points at step `s`'s AdaLN lines; the graph replays those pointers.
/// `last_velocity` is the draft phase's last step: the head's velocity alone, no Euler advance, so the
/// host can lift it with cascade.lift.
fn stepGraph(model: *h3.Model, device: *const h3.Model.Inputs, checked: h3.Model.Plan, rows: *const h3.Rows, s: usize, delta: f32, span: picture.HeadSpan, last_velocity: bool) !@import("cuda").graph.Exec {
    const graph_api = @import("cuda").graph;
    try graph_api.beginCapture(model.stream, .thread_local);
    stepBody(model, device, checked, rows, s, delta, span, last_velocity) catch |err| {
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

fn stepBody(model: *h3.Model, device: *const h3.Model.Inputs, checked: h3.Model.Plan, rows: *const h3.Rows, s: usize, delta: f32, span: picture.HeadSpan, last_velocity: bool) !void {
    try rows.in(device.x);
    try model.step(device, checked);
    if (last_velocity) {
        // the draft's last step computes the velocity cascade.lift turns into the clean latent:
        // the video stays at this step's sigma and the host lifts it, so there is no Euler update
        try rows.velocityOf(device.x, s, span.first, span.last, &span.weights);
    } else try rows.out(device.x, s, delta, span.first, span.last, &span.weights);
}
