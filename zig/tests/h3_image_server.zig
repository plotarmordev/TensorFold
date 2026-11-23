//! tf-h3-image-server: MiniMax-H3-Image's prompt encoder, transformer and decoder loaded once, then one picture per
//! request line on stdin, each step as the one-shot tools run it.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3");
const prompt_tool = @import("h3_image_prompt.zig");
const image_tool = @import("h3_image.zig");
const decoder_tool = @import("h3_image_decoder.zig");
const keepalive = @import("keepalive");

const keep_warm_ns: i64 = std.math.maxInt(i64);

const usage =
    \\usage: tf-h3-image-server TEXT_ENCODER_DIR MODEL_DIR [profile] [--draft-steps N] [--renoise-sigma X] [--far-pool] [--routed-steps N]
    \\  then one request a line on stdin: WIDTHxHEIGHT SEED OUT_PREFIX prompt.txt
    \\  each writes OUT_PREFIX.safetensors, .h3case.safetensors, .rows.f32, .latent.f32 and .png as
    \\  tf-h3-image-prompt, tf-h3-image (free) and tf-h3-image-decoder do, and prints a timing line;
    \\  a request with the last one's canvas, rows and tags reuses its transformer binding;
    \\  a prompt's states are kept for its repeats (H3_PROMPT_CACHE MiB, 256)
    \\  profile: each request's six steps by kernel, each launch timed by the GPU's timestamp counters (slower; diagnostic)
    \\  --draft-steps N runs every request's first N steps on the half-size canvas, as mmh3's does:
    \\  a different picture, faster (the free run only; no reference edits)
    \\  --far-pool pools distant keys and values 2 x 2 at 4096 x 4096 only (tf-h3-image's --far-pool); on the
    \\  steps it pools it wins over --routed-steps
    \\  --routed-steps N routes the last N steps' attention, each picture tile to its best fifth of picture
    \\  tiles (FastH3's top-k): a different picture, measured in the Metal recipe
    \\
;

/// The transformer bound to one picture layout, its host copies in its own arena.
const Binding = struct {
    arena: *std.heap.ArenaAllocator,
    model: *h3.Model,
    canvas: h3.picture.Size,
    rows: usize,
    references: usize,
    tags: []const i32,
    weights: ?mtl.ResidencySet,

    fn init(gpa: std.mem.Allocator, folder: *const image_tool.Folder, case: *const image_tool.Case, tags: []const i32, weights: ?mtl.ResidencySet, sched: ?h3.picture.Pdd) !Binding {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        errdefer gpa.destroy(arena);
        arena.* = .init(gpa);
        errdefer arena.deinit();
        const model = try folder.bind(arena.allocator(), arena.allocator(), case, sched);
        errdefer model.release();
        const own_tags = try arena.allocator().dupe(i32, tags);
        // the binding's int8 weights join the weights' set while it lives: a step's encode drops from 0.19 to 0.04 s
        if (weights) |set| {
            model.queue.addResidencySet(set);
            set.add(model.wq);
            set.add(model.ws);
            set.commit();
        }
        return .{ .arena = arena, .model = model, .canvas = case.canvas, .rows = case.rows, .references = case.layout.references.len, .tags = own_tags, .weights = weights };
    }

    // Text-to-image only: serve creates no reference rows; canvas, rows and tags determine every row and rotary position.
    fn fits(b: *const Binding, case: *const image_tool.Case, tags: []const i32) bool {
        return std.meta.eql(b.canvas, case.canvas) and b.rows == case.rows and b.references == case.layout.references.len and std.mem.eql(i32, b.tags, tags);
    }

    fn release(b: *Binding) void {
        if (b.weights) |set| {
            set.remove(b.model.wq);
            set.remove(b.model.ws);
            set.commit();
        }
        b.model.release();
        const gpa = b.arena.child_allocator;
        b.arena.deinit();
        gpa.destroy(b.arena);
    }
};

/// `binding` for `case`: its text rows replaced when the layout fits, else released and bound anew; true when reused.
fn rebind(gpa: std.mem.Allocator, folder: *const image_tool.Folder, binding: *?Binding, case: *const image_tool.Case, tags: []const i32, weights: ?mtl.ResidencySet, sched: ?h3.picture.Pdd) !bool {
    if (binding.*) |*b| if (b.fits(case, tags)) {
        try b.model.retext(case.file);
        return true;
    };
    if (binding.*) |*b| b.release();
    binding.* = null;
    binding.* = try Binding.init(gpa, folder, case, tags, weights, sched);
    return false;
}

/// Every weight file in one residency set. macOS unwires GPU memory idle for a second or two, so without it each
/// stage's first commit of a request re-wires its weights: 0.3-0.7 s of step 1, up to 1.8 s of the prompt. A refusal
/// past `limit` stops startup rather than serve unwarmed.
fn holdWeights(init: std.process.Init, buffers: []const mtl.Buffer, device: mtl.Device, queue: mtl.Queue, limit: ?u64) !mtl.resident.Resident {
    const held = try mtl.resident.Resident.holdUnder(init.gpa, init.io, device, queue, buffers, "h3 image weights", limit);
    return held orelse error.WeightsNotResident;
}

/// The idle ticker over a held set; a failure stops startup, since unwarmed stages re-wire every request.
fn warmStart(init: std.process.Init, gpa: std.mem.Allocator, resident: *mtl.resident.Resident, target: *mtl.keepalive.Target) !*keepalive.Keepalive {
    target.* = resident.target();
    return keepalive.Keepalive.start(gpa, init.io, .{ .ctx = target, .tick = mtl.keepalive.Target.tick }, keep_warm_ns);
}

pub fn main(init: std.process.Init) !u8 {
    const all = try init.minimal.args.toSlice(init.arena.allocator());
    const a0 = init.arena.allocator();
    var profile = false;
    var draft_steps: usize = 0;
    var routed_steps: usize = 0;
    var renoise: ?f32 = null;
    var far_pool = false;
    var positional: std.ArrayList([]const u8) = .empty;
    var i: usize = 1;
    while (i < all.len) : (i += 1) {
        const arg = all[i];
        if (std.mem.eql(u8, arg, "profile")) {
            profile = true;
        } else if (std.mem.eql(u8, arg, "--draft-steps")) {
            i += 1;
            if (i >= all.len) return 2;
            draft_steps = std.fmt.parseInt(usize, all[i], 10) catch return 2;
        } else if (std.mem.eql(u8, arg, "--renoise-sigma")) {
            i += 1;
            if (i >= all.len) return 2;
            renoise = std.fmt.parseFloat(f32, all[i]) catch return 2;
        } else if (std.mem.eql(u8, arg, "--far-pool")) {
            far_pool = true;
        } else if (std.mem.eql(u8, arg, "--routed-steps")) {
            i += 1;
            if (i >= all.len) return 2;
            routed_steps = std.fmt.parseInt(usize, all[i], 10) catch return 2;
        } else try positional.append(a0, arg);
    }
    const args = positional.items;
    if (args.len != 2 or (draft_steps == 0 and renoise != null)) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const loading = mtl.clock.seconds();
    var encoder = try prompt_tool.Encoder.load(init, a, args[0], args[1]);
    defer encoder.deinit(init.gpa);
    var folder = try image_tool.Folder.open(init, a, args[1]);
    folder.far_pool = far_pool;
    defer folder.heads.deinit();
    folder.routed_steps = routed_steps;
    const vae = try decoder_tool.Vae.load(init, a, args[1]);
    defer vae.d.deinit();
    std.debug.print("loaded the prompt encoder, transformer weights and decoder in {d:.1} s; ready\n", .{mtl.clock.seconds() - loading});
    var buffers: std.ArrayList(mtl.Buffer) = .empty;
    for ([_][]const h3.Tensors{ encoder.text_w.shards.files, encoder.front_w.shards.files, folder.ck.shards, (&folder.heads)[0..1] }) |mapped|
        for (mapped) |f| try buffers.append(a, f.buffer);
    try buffers.append(a, vae.d.weights);
    var resident = try holdWeights(init, buffers.items, encoder.device, encoder.runner.queue, null);
    defer resident.deinit();
    vae.d.queue.addResidencySet(resident.held.set);
    std.debug.print("{d:.1} GiB of weights held resident\n", .{mtl.ResidencySet.gibs(resident.held.bytes)});
    var warm: mtl.keepalive.Target = undefined;
    const ticker = try warmStart(init, init.gpa, &resident, &warm);
    defer ticker.stop();

    var binding: ?Binding = null;
    defer if (binding) |*b| b.release();
    var draft_binding: ?Binding = null;
    defer if (draft_binding) |*b| b.release();
    var in_buf: [4096]u8 = undefined;
    var in = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    while (true) {
        const line = (nextLine(&in.interface) catch |err| switch (err) {
            error.StreamTooLong => {
                std.debug.print("request failed: StreamTooLong\n", .{});
                continue;
            },
            else => |e| return e,
        }) orelse break;
        const fields = std.mem.trim(u8, line, " \t\r");
        if (fields.len == 0) continue;
        var request = std.heap.ArenaAllocator.init(init.gpa);
        defer request.deinit();
        ticker.begin();
        defer ticker.end();
        serve(init, request.allocator(), &encoder, &folder, &vae, &binding, &draft_binding, resident.held.set, fields, profile, draft_steps, renoise) catch |err| {
            std.debug.print("request failed: {s}\n", .{@errorName(err)});
        };
    }
    return 0;
}

/// The next request line; an over-long one is drained to its end and reported as `error.StreamTooLong`.
fn nextLine(in: *std.Io.Reader) !?[]const u8 {
    return in.takeDelimiter('\n') catch |err| switch (err) {
        error.StreamTooLong => {
            while (true) {
                _ = in.takeDelimiter('\n') catch |e| switch (e) {
                    error.StreamTooLong => {
                        in.toss(in.buffered().len);
                        continue;
                    },
                    else => |other| return other,
                };
                break;
            }
            return error.StreamTooLong;
        },
        else => |e| return e,
    };
}

/// One request line: its prompt's case, the six steps on the bound (or a new) transformer, its PNG, a timing line.
fn serve(init: std.process.Init, a: std.mem.Allocator, encoder: *prompt_tool.Encoder, folder: *const image_tool.Folder, vae: *const decoder_tool.Vae, binding: *?Binding, draft_binding: *?Binding, weights: mtl.ResidencySet, line: []const u8, profile: bool, draft_steps: usize, renoise: ?f32) !void {
    var it = std.mem.tokenizeAny(u8, line, " \t");
    const size = try prompt_tool.size(it.next() orelse return error.BadRequest);
    const seed = try std.fmt.parseInt(u64, it.next() orelse return error.BadRequest, 10);
    const prefix = it.next() orelse return error.BadRequest;
    const prompt_path = it.next() orelse return error.BadRequest;
    if (it.next() != null) return error.BadRequest;
    var prompt: []const u8 = try std.Io.Dir.cwd().readFileAlloc(init.io, prompt_path, a, .limited(16 << 20));
    if (std.mem.endsWith(u8, prompt, "\n")) prompt = prompt[0 .. prompt.len - 1];
    if (std.mem.endsWith(u8, prompt, "\r")) prompt = prompt[0 .. prompt.len - 1];

    const started = mtl.clock.seconds();
    const request = prompt_tool.Request{ .size = size, .seed = seed, .min_area = h3.picture.default_min_area };
    const pic_path = try std.fmt.allocPrintSentinel(a, "{s}.safetensors", .{prefix}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = pic_path, .data = try encoder.case(init, a, prompt, request, .{}) });
    const encoded = mtl.clock.seconds();

    var pic = try h3.Tensors.open(init.gpa, try mtl.Device.init(), pic_path);
    defer pic.deinit();
    // mmh3's --draft-steps: draftPlan needs the case's grid and seed noise, so the normal case is
    // written first; a renoise that moved a step off the distilled grid rewrites it with the draft
    // schedule and blended heads
    var case = try image_tool.Case.write(init, a, folder, &pic, prefix, null);
    defer case.file.deinit();
    const plan: ?image_tool.DraftPlan = if (draft_steps == 0) null else plan: {
        if (case.layout.references.len != 0) return error.H3DraftWithReferences;
        break :plan try image_tool.draftPlan(a, folder, &case, draft_steps, renoise);
    };
    if (plan) |p| if (p.heads != null) {
        const next = try image_tool.Case.write(init, a, folder, &pic, prefix, .{ .sched = p.sched, .video_out = p.heads });
        case.file.deinit();
        case = next;
    };
    const tags = pic.values(i32, try pic.at("text_tags", .i32, case.layout.text), case.layout.text);
    const cased = mtl.clock.seconds();
    const reused = try rebind(init.gpa, folder, binding, &case, tags, weights, if (plan) |p| p.sched else null);
    const bound = mtl.clock.seconds();

    const model = binding.*.?.model;
    if (profile) {
        model.profile = true;
        model.spent = @splat(0);
    }
    defer {
        if (profile) model.profile = false;
    }
    const steps = if (plan) |p| p.sched.steps() else folder.pdd.steps();
    const out = if (plan) |p| out: {
        // the half-size draft's own case and binding: its steps index the same schedule positions
        const draft_case = try image_tool.Case.write(init, a, folder, &pic, try std.fmt.allocPrint(a, "{s}.draft", .{prefix}), .{ .sched = p.sched, .grid = p.draft_grid, .noise = p.draft_noise, .video_out = p.heads });
        defer draft_case.file.deinit();
        _ = try rebind(init.gpa, folder, draft_binding, &draft_case, tags, weights, p.sched);
        break :out try image_tool.drafted(a, model, draft_binding.*.?.model, &case, &draft_case, p.sched, draft_steps, p.noise_latent, prefix);
    } else try image_tool.free(a, model, folder, &case, prefix);
    if (profile) {
        std.debug.print("{s}: steps by kernel, each launch timed by the GPU's timestamp counters:\n", .{prefix});
        for (h3.stages, model.spent) |stage, seconds| std.debug.print("  {s}: {d:.3} s\n", .{ stage, seconds });
    }
    const denoised = mtl.clock.seconds();
    try vae.png(init, a, out.latent, case.grid.rows, case.grid.columns, .{ .width = size.width, .height = size.height }, try std.fmt.allocPrint(a, "{s}.png", .{prefix}));
    const done = mtl.clock.seconds();
    std.debug.print("{s}: prompt {d:.2} s, case {d:.2} s, transformer {s} {d:.2} s, {d} steps {d:.2} s, decoder {d:.2} s, total {d:.2} s\n", .{ prefix, encoded - started, cased - encoded, if (reused) "reused" else "bound", bound - cased, steps, denoised - bound, done - denoised, done - started });
}

const testing = std.testing;
const fixture = @import("checkpoint_fixture");
const files = @import("h3_image_files.zig");

/// A tiny synthetic MiniMax-H3-Image transformer: one block, hidden 256, inner 256, mlp 256, six steps.
const synthetic = struct {
    const hidden = 256;
    const inner = 256;
    const mlp = 256;
    const width = h3.picture.row_width;
    const steps = 6;
    const lines = 3;
    const times = 1;
    const rot = h3.picture.rotated;
    const adaln_n = 3 * 6 * hidden;
    const final_n = 2 * hidden;

    fn fill(a: std.mem.Allocator, comptime T: type, n: usize, seed: u32) ![]const u8 {
        const out = try a.alloc(T, n);
        var x: u32 = seed;
        for (out) |*v| {
            x = x *% 1664525 +% 1013904223;
            const f = @as(f32, @floatFromInt((x >> 8) & 0xffff)) / 65536.0 - 0.5;
            v.* = switch (T) {
                f32 => f * 0.1,
                u16 => @truncate(@as(u32, @bitCast(f * 0.1)) >> 16),
                i32 => @intCast((x >> 16) & 1),
                else => unreachable,
            };
        }
        return std.mem.sliceAsBytes(out);
    }

    fn add(a: std.mem.Allocator, entries: *std.ArrayList(fixture.Entry), name: []const u8, dtype: []const u8, shape: []const usize, seed: u32) !void {
        var n: usize = 1;
        for (shape) |d| n *= d;
        const bytes = switch (dtype[0]) {
            'B' => try fill(a, u16, n, seed),
            'F' => try fill(a, f32, n, seed),
            else => try fill(a, i32, n, seed),
        };
        try entries.append(a, .{ .name = name, .dtype = dtype, .shape = shape, .bytes = bytes });
    }
};

/// The folder `Folder.open` reads: the image pipeline's index, the distilled schedule, one shard and its heads.
fn syntheticFolder(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    try dir.writeFile(io, .{ .sub_path = "model_index.json", .data = "{\"_class_name\": \"MiniMaxH3ImagePipeline\", \"_minimax_h3\": {\"partition\": \"image\", \"pdd_fused_heads\": \"pdd_fused_heads.safetensors\", \"tasks\": [\"t2i\"]}}" });
    try dir.writeFile(io, .{ .sub_path = "pdd_config.json", .data = "{\"num_inference_steps\": 6, \"sigmas\": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]}" });
    var transformer = try dir.createDirPathOpen(io, "transformer", .{});
    defer transformer.close(io);
    try transformer.writeFile(io, .{ .sub_path = "config.json", .data = "{\"audio_enabled\": false, \"latents_dim\": 24, \"rope_inv_freq_len\": 16, \"attention_head_dim\": 128, \"patch_size\": [1, 2, 2]}" });
    try transformer.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\": {\"blocks.0.attn.qkv_proj.weight\": \"model-00001-of-00001.safetensors\"}}" });
    const s = synthetic;
    var entries: std.ArrayList(fixture.Entry) = .empty;
    try s.add(a, &entries, "blocks.0.attn.qkv_proj.weight", "BF16", &.{ 3 * s.inner, s.hidden }, 1);
    try s.add(a, &entries, "blocks.0.attn.out_proj.weight", "BF16", &.{ s.hidden, s.inner }, 2);
    try s.add(a, &entries, "blocks.0.mlp.fc1.weight", "BF16", &.{ 2 * s.mlp, s.hidden }, 3);
    try s.add(a, &entries, "blocks.0.mlp.fc2.weight", "BF16", &.{ s.hidden, s.mlp }, 4);
    try s.add(a, &entries, "blocks.0.norm1.weight", "BF16", &.{s.hidden}, 5);
    try s.add(a, &entries, "blocks.0.norm2.weight", "BF16", &.{s.hidden}, 6);
    try s.add(a, &entries, "blocks.0.attn.q_norm.weight", "BF16", &.{128}, 7);
    try s.add(a, &entries, "blocks.0.attn.k_norm.weight", "BF16", &.{128}, 8);
    try s.add(a, &entries, "blocks.0.adaln_proj.linear.weight", "BF16", &.{ s.adaln_n, s.hidden }, 9);
    try s.add(a, &entries, "blocks.0.adaln_proj.linear.bias", "BF16", &.{s.adaln_n}, 10);
    try s.add(a, &entries, "final_layer.norm.weight", "BF16", &.{s.hidden}, 11);
    try s.add(a, &entries, "final_layer.adaln_proj.linear.weight", "BF16", &.{ s.final_n, s.hidden }, 12);
    try s.add(a, &entries, "final_layer.adaln_proj.linear.bias", "BF16", &.{s.final_n}, 13);
    try s.add(a, &entries, "time_embedder.proj_in.weight", "BF16", &.{ s.hidden, s.hidden }, 14);
    try s.add(a, &entries, "time_embedder.proj_in.bias", "BF16", &.{s.hidden}, 15);
    try s.add(a, &entries, "time_embedder.proj_out.weight", "BF16", &.{ s.hidden, s.hidden }, 16);
    try s.add(a, &entries, "time_embedder.proj_out.bias", "BF16", &.{s.hidden}, 17);
    try s.add(a, &entries, "rope.inv_freq", "F32", &.{16}, 18);
    try s.add(a, &entries, "video_patch_proj.weight", "F32", &.{ s.hidden, s.width }, 19);
    try s.add(a, &entries, "video_patch_proj.bias", "F32", &.{s.hidden}, 20);
    try fixture.write(a, io, transformer, "model-00001-of-00001.safetensors", entries.items);
    var heads: std.ArrayList(fixture.Entry) = .empty;
    try s.add(a, &heads, "video_out.weight", "F32", &.{ s.steps, s.width, s.hidden }, 21);
    try s.add(a, &heads, "video_out.bias", "F32", &.{ s.steps, s.width }, 22);
    try fixture.write(a, io, dir, "pdd_fused_heads.safetensors", heads.items);
}

/// A prompt tool's picture case: the canvas, the prompt's rows, their tags and the seed's noise.
fn pictureCase(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8, canvas: [2]i32, text: usize, seed: u32) !void {
    const s = synthetic;
    var entries: std.ArrayList(fixture.Entry) = .empty;
    const canvas_values = try a.alloc(i32, 2);
    canvas_values[0] = canvas[0];
    canvas_values[1] = canvas[1];
    try entries.append(a, .{ .name = "canvas", .dtype = "I32", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(canvas_values) });
    try s.add(a, &entries, "text", "BF16", &.{ text, s.hidden }, seed);
    const tags = try a.alloc(i32, text);
    @memset(tags, 1);
    try entries.append(a, .{ .name = "text_tags", .dtype = "I32", .shape = &.{text}, .bytes = std.mem.sliceAsBytes(tags) });
    const grid = try h3.picture.gridOf(.{ .width = @intCast(canvas[0]), .height = @intCast(canvas[1]) });
    try s.add(a, &entries, "noise", "F32", &.{try h3.picture.gridRows(grid) * s.width}, seed + 100);
    try dir.writeFile(io, .{ .sub_path = name, .data = try files.image(a, entries.items) });
}

var idle_ticks = std.atomic.Value(u32).init(0);

/// The keepalive's tick, counted so the idle gate sees it ran.
fn countedTick(ctx: *anyopaque) void {
    _ = idle_ticks.fetchAdd(1, .monotonic);
    mtl.keepalive.Target.tick(ctx);
}

const Spec = struct { name: []const u8, canvas: [2]i32, text: usize, seed: u32 };
const Built = struct { case: image_tool.Case, tags: []const i32 };

fn buildCase(init: std.process.Init, a: std.mem.Allocator, dir: std.Io.Dir, dir_path: []const u8, folder: *const image_tool.Folder, device: mtl.Device, spec: Spec) !Built {
    const pic_name = try std.fmt.allocPrint(a, "{s}.pic.safetensors", .{spec.name});
    try pictureCase(a, init.io, dir, pic_name, spec.canvas, spec.text, spec.seed);
    const pic_path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, pic_name }, 0);
    var pic = try h3.Tensors.open(testing.allocator, device, pic_path);
    defer pic.deinit();
    const prefix = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir_path, spec.name });
    var case = try image_tool.Case.write(init, a, folder, &pic, prefix, null);
    // the noise is a view into `pic`, which closes on return
    case.noise = try a.dupe(f32, case.noise);
    const tags = try a.dupe(i32, pic.values(i32, try pic.at("text_tags", .i32, case.layout.text), case.layout.text));
    return .{ .case = case, .tags = tags };
}

fn freshLatent(a: std.mem.Allocator, folder: *const image_tool.Folder, case: *const image_tool.Case, tags: []const i32, prefix: []const u8) ![]f32 {
    var b = try Binding.init(testing.allocator, folder, case, tags, null, null);
    defer b.release();
    return (try image_tool.free(a, b.model, folder, case, prefix)).latent;
}

fn residentLatent(a: std.mem.Allocator, folder: *const image_tool.Folder, binding: *?Binding, case: *const image_tool.Case, tags: []const i32, weights: mtl.ResidencySet, prefix: []const u8) ![]f32 {
    _ = try rebind(testing.allocator, folder, binding, case, tags, weights, null);
    return (try image_tool.free(a, binding.*.?.model, folder, case, prefix)).latent;
}

fn expectSameLatent(want: []const f32, got: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    try testing.expect(std.mem.eql(u32, @as([]const u32, @ptrCast(want)), @as([]const u32, @ptrCast(got))));
}

/// The folder and the four cases the resident tests share; the caller frees the checkpoint's shards.
fn residentFixture(init: std.process.Init, a: std.mem.Allocator, dir: std.Io.Dir, dir_path: []const u8, device: mtl.Device) !struct { folder: image_tool.Folder, cases: [4]Built } {
    try syntheticFolder(a, init.io, dir);
    var folder = try image_tool.Folder.open(init, a, dir_path);
    errdefer {
        folder.heads.deinit();
        for (folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(folder.ck.shards);
    }
    const specs = [_]Spec{
        .{ .name = "a", .canvas = .{ 64, 64 }, .text = 2, .seed = 42 },
        .{ .name = "b", .canvas = .{ 64, 64 }, .text = 2, .seed = 43 },
        .{ .name = "c", .canvas = .{ 64, 64 }, .text = 3, .seed = 44 },
        .{ .name = "d", .canvas = .{ 96, 96 }, .text = 2, .seed = 45 },
    };
    var cases: [4]Built = undefined;
    for (&cases, specs) |*c, spec| c.* = try buildCase(init, a, dir, dir_path, &folder, device, spec);
    return .{ .folder = folder, .cases = cases };
}

/// The process init a tool function reads: no arguments, the test's arena and allocator.
fn testInit(arena: *std.heap.ArenaAllocator, environ_map: *std.process.Environ.Map) std.process.Init {
    return .{
        .minimal = .{ .environ = .empty, .args = .{ .vector = &.{} } },
        .arena = arena,
        .gpa = testing.allocator,
        .io = std.testing.io,
        .environ_map = environ_map,
        .preopens = .empty,
    };
}

test "resident binding, its weights held resident, matches fresh bindings across prompts, rows and canvas" {
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = testInit(&arena, &environ_map);
    var f = try residentFixture(init, a, tmp.dir, dir_path, device);
    defer {
        for (&f.cases) |*c| c.case.file.deinit();
        f.folder.heads.deinit();
        for (f.folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(f.folder.ck.shards);
    }
    var buffers: std.ArrayList(mtl.Buffer) = .empty;
    for (f.folder.ck.shards) |shard| try buffers.append(a, shard.buffer);
    var held = (try mtl.resident.Resident.hold(testing.allocator, init.io, device, try device.queue(), buffers.items, "h3 image test")) orelse return error.SkipZigTest;
    defer held.deinit();
    const weights = held.held.set;
    var warm = held.target();
    idle_ticks.store(0, .release);
    var ticker = try keepalive.Keepalive.startEvery(testing.allocator, std.testing.io, .{ .ctx = &warm, .tick = countedTick }, std.time.ns_per_s, 10 * std.time.ns_per_ms);
    defer ticker.stop();
    var binding: ?Binding = null;
    defer if (binding) |*b| b.release();

    const fresh_a = try freshLatent(a, &f.folder, &f.cases[0].case, f.cases[0].tags, try std.fmt.allocPrint(a, "{s}/fresh-a", .{dir_path}));
    const resident_a = try residentLatent(a, &f.folder, &binding, &f.cases[0].case, f.cases[0].tags, weights, try std.fmt.allocPrint(a, "{s}/resident-a", .{dir_path}));
    try expectSameLatent(fresh_a, resident_a);

    // a different prompt of the same length reuses the binding and replaces its text rows
    try testing.expect(binding.?.fits(&f.cases[1].case, f.cases[1].tags));
    const fresh_b = try freshLatent(a, &f.folder, &f.cases[1].case, f.cases[1].tags, try std.fmt.allocPrint(a, "{s}/fresh-b", .{dir_path}));
    const resident_b = try residentLatent(a, &f.folder, &binding, &f.cases[1].case, f.cases[1].tags, weights, try std.fmt.allocPrint(a, "{s}/resident-b", .{dir_path}));
    try expectSameLatent(fresh_b, resident_b);

    // a changed token count and a changed canvas each release the binding and bind anew
    try testing.expect(!binding.?.fits(&f.cases[2].case, f.cases[2].tags));
    const fresh_c = try freshLatent(a, &f.folder, &f.cases[2].case, f.cases[2].tags, try std.fmt.allocPrint(a, "{s}/fresh-c", .{dir_path}));
    const resident_c = try residentLatent(a, &f.folder, &binding, &f.cases[2].case, f.cases[2].tags, weights, try std.fmt.allocPrint(a, "{s}/resident-c", .{dir_path}));
    try expectSameLatent(fresh_c, resident_c);

    try testing.expect(!binding.?.fits(&f.cases[3].case, f.cases[3].tags));
    const fresh_d = try freshLatent(a, &f.folder, &f.cases[3].case, f.cases[3].tags, try std.fmt.allocPrint(a, "{s}/fresh-d", .{dir_path}));
    const resident_d = try residentLatent(a, &f.folder, &binding, &f.cases[3].case, f.cases[3].tags, weights, try std.fmt.allocPrint(a, "{s}/resident-d", .{dir_path}));
    try expectSameLatent(fresh_d, resident_d);

    // back to the first request: a fresh binding again, the same latent
    const resident_a2 = try residentLatent(a, &f.folder, &binding, &f.cases[0].case, f.cases[0].tags, weights, try std.fmt.allocPrint(a, "{s}/resident-a2", .{dir_path}));
    try expectSameLatent(fresh_a, resident_a2);

    // macOS unwires idle GPU memory after 1-2 s: the ticker keeps the binding's weights held across that
    std.Io.sleep(init.io, .fromMilliseconds(1500), .awake) catch {};
    try testing.expect(idle_ticks.load(.acquire) > 0);
    const resident_idle = try residentLatent(a, &f.folder, &binding, &f.cases[0].case, f.cases[0].tags, weights, try std.fmt.allocPrint(a, "{s}/resident-idle", .{dir_path}));
    try expectSameLatent(fresh_a, resident_idle);
}

test "the steps' Euler on the GPU leaves main's host-Euler latent bit for bit" {
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = testInit(&arena, &environ_map);
    var f = try residentFixture(init, a, tmp.dir, dir_path, device);
    defer {
        for (&f.cases) |*c| c.case.file.deinit();
        f.folder.heads.deinit();
        for (f.folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(f.folder.ck.shards);
    }
    const case = &f.cases[0].case;
    var b = try Binding.init(testing.allocator, &f.folder, case, f.cases[0].tags, null, null);
    defer b.release();
    const resident = (try image_tool.free(a, b.model, &f.folder, case, try std.fmt.allocPrint(a, "{s}/resident", .{dir_path}))).rows;

    // main's loop on the same binding: each step's velocity down, picture.euler on the host, the latent up again
    const x = try a.dupe(f32, case.noise);
    const velocity = try a.alloc(f32, x.len);
    for (0..f.folder.pdd.steps()) |k| {
        _ = try b.model.forward(x, &.{}, k, null, velocity, &.{});
        h3.picture.euler(x, velocity, f.folder.pdd.deltas[k]);
    }
    try expectSameLatent(x, resident);
}

test "routed steps that keep every picture tile leave the dense latent bit for bit" {
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = testInit(&arena, &environ_map);
    var f = try residentFixture(init, a, tmp.dir, dir_path, device);
    defer {
        for (&f.cases) |*c| c.case.file.deinit();
        f.folder.heads.deinit();
        for (f.folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(f.folder.ck.shards);
    }
    const case = &f.cases[0].case;
    var dense = try Binding.init(testing.allocator, &f.folder, case, f.cases[0].tags, null, null);
    defer dense.release();
    const want = (try image_tool.free(a, dense.model, &f.folder, case, try std.fmt.allocPrint(a, "{s}/dense", .{dir_path}))).rows;

    // every step routed: the prefix tile attends densely, each picture tile to the prefix and its chosen tiles
    f.folder.routed_steps = f.folder.pdd.steps();
    var routed = try Binding.init(testing.allocator, &f.folder, case, f.cases[0].tags, null, null);
    defer routed.release();
    try testing.expectEqual(routed.model.tiles - routed.model.prefix_tiles, routed.model.keep);
    const got = (try image_tool.free(a, routed.model, &f.folder, case, try std.fmt.allocPrint(a, "{s}/routed", .{dir_path}))).rows;
    try expectSameLatent(got, want);
}

test "profile times each picture kernel on the GPU's counters and leaves the latent bit for bit" {
    const device = try mtl.Device.init();
    if (!device.tensorUnits() or mtl.counters.query(device).mode() == .unsupported) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = testInit(&arena, &environ_map);
    var f = try residentFixture(init, a, tmp.dir, dir_path, device);
    defer {
        for (&f.cases) |*c| c.case.file.deinit();
        f.folder.heads.deinit();
        for (f.folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(f.folder.ck.shards);
    }
    const case = &f.cases[0].case;
    var b = try Binding.init(testing.allocator, &f.folder, case, f.cases[0].tags, null, null);
    defer b.release();
    const plain = (try image_tool.free(a, b.model, &f.folder, case, try std.fmt.allocPrint(a, "{s}/plain", .{dir_path}))).rows;
    b.model.profile = true;
    const profiled = (try image_tool.free(a, b.model, &f.folder, case, try std.fmt.allocPrint(a, "{s}/profiled", .{dir_path}))).rows;
    try expectSameLatent(plain, profiled);
    // a picture's blocks are dense: no routing and no pooled branch
    for (h3.stages, b.model.spent) |stage, seconds| {
        if (std.mem.eql(u8, stage, "routing") or std.mem.eql(u8, stage, "gate mix")) continue;
        errdefer std.debug.print("stage {s}: {d} s\n", .{ stage, seconds });
        try testing.expect(seconds > 0 and seconds < 1);
    }
    // a profiled step's GPU time spans every kernel's counter-timed encoder (within the clocks' conversion)
    b.model.spent = @splat(0);
    const took = try b.model.forward(case.noise, &.{}, 0, null, &.{}, &.{});
    var kernels: f64 = 0;
    for (b.model.spent) |seconds| kernels += seconds;
    errdefer std.debug.print("step GPU {d} s, kernels {d} s\n", .{ took.gpu, kernels });
    try testing.expect(took.gpu > 0.9 * kernels);
}

test "a failed bind releases every Metal object it made" {
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = testInit(&arena, &environ_map);
    var f = try residentFixture(init, a, tmp.dir, dir_path, device);
    defer {
        for (&f.cases) |*c| c.case.file.deinit();
        f.folder.heads.deinit();
        for (f.folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(f.folder.ck.shards);
    }
    const baseline = device.allocated();
    var counting = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = std.math.maxInt(usize) });
    const model = try h3.Model.bind(counting.allocator(), f.folder.ck, f.cases[0].case.file, .{});
    // Metal rounds buffers up and may keep freed pages a while: a release need only return seven eighths of a bind
    const ceiling = baseline + (device.allocated() - baseline) / 8;
    model.release();
    try testing.expectEqual(counting.allocated_bytes, counting.freed_bytes);
    try testing.expect(device.allocated() <= ceiling);

    // the last allocation is the block table: every buffer and pipeline exists when it fails
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = counting.alloc_index - 1 });
    try testing.expectError(error.OutOfMemory, h3.Model.bind(failing.allocator(), f.folder.ck, f.cases[0].case.file, .{}));
    try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    try testing.expect(device.allocated() <= ceiling);

    var binding_alloc = std.testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    var bound = try Binding.init(binding_alloc.allocator(), &f.folder, &f.cases[0].case, &.{}, null, null);
    const tag_count = bound.arena.queryCapacity() / @sizeOf(i32) + 1;
    bound.release();
    try testing.expect(device.allocated() <= ceiling);
    const tags = try a.alloc(i32, tag_count);
    @memset(tags, 1);

    // Force the tag copy to grow the arena, then fail that allocation after the model has bound successfully.
    var tag_alloc = std.testing.FailingAllocator.init(testing.allocator, .{
        .fail_index = binding_alloc.alloc_index,
        .resize_fail_index = 0,
    });
    try testing.expectError(error.OutOfMemory, Binding.init(tag_alloc.allocator(), &f.folder, &f.cases[0].case, tags, null, null));
    try testing.expect(tag_alloc.has_induced_failure);
    try testing.expectEqual(tag_alloc.allocated_bytes, tag_alloc.freed_bytes);
    try testing.expect(device.allocated() <= ceiling);
}

/// An io whose `nth` exclusive tryLock (a stage's speech probe) first takes speech for `ms` on the real io: speech
/// arrives mid-stage, after the passes probed before it were submitted. One at a time; `join` after the stage.
const Arrival = struct {
    var base: std.Io = undefined;
    var vtable: std.Io.VTable = undefined;
    var lock: []const u8 = undefined;
    var nth: usize = 0;
    var hold_ms: i64 = 0;
    var speech: ?std.Thread = null;

    fn at(io: std.Io, path: []const u8, probe: usize, ms: i64) h3.gpu_turn.Turn {
        base = io;
        vtable = io.vtable.*;
        vtable.fileTryLock = tryLock;
        lock = path;
        nth = probe;
        hold_ms = ms;
        speech = null;
        return .{ .io = .{ .userdata = io.userdata, .vtable = &vtable }, .at = path };
    }

    fn tryLock(userdata: ?*anyopaque, file: std.Io.File, kind: std.Io.File.Lock) std.Io.File.LockError!bool {
        if (kind == .exclusive and nth > 0) {
            nth -= 1;
            if (nth == 0) speech = speakFor(base, lock, hold_ms) catch null;
        }
        return base.vtable.fileTryLock(userdata, file, kind);
    }

    fn join() !void {
        (speech orelse return error.NoSpeechArrived).join();
    }
};

/// Speech holding `lock` for `ms` from now, released from its own thread.
fn speakFor(io: std.Io, lock: []const u8, ms: i64) !std.Thread {
    const held = h3.gpu_turn.speak(io, lock) orelse return error.NoSpeechHold;
    return std.Thread.spawn(.{}, struct {
        fn release(i: std.Io, file: std.Io.File, wait: i64) void {
            i.sleep(.fromMilliseconds(wait), .awake) catch {};
            h3.gpu_turn.done(i, file);
        }
    }.release, .{ io, held, ms });
}

test "speech arriving during the prompt encoder, denoising or the decoder holds its next pass, output unchanged" {
    const io = testing.io;
    const device = try mtl.Device.init();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const lock = try std.fs.path.join(a, &.{ dir_path, "speech.lock" });
    const hold_ms = 500;
    const least = 0.25; // seconds the next pass must wait: the hold less any delay before its probe, with room to spare

    // the prompt encoder and the token refiner, on the prompt selftest's tiny checkpoint
    var prng = std.Random.DefaultPrng.init(0x5133);
    const tiny = try prompt_tool.Tiny.write(io, a, prng.random(), dir_path);
    const g = prompt_tool.tiny_text;
    var runner = try h3.Runner.init(device);
    defer runner.deinit();
    var text_w = try h3.weights.loadText(testing.allocator, io, device, tiny.text_dir, g);
    defer text_w.deinit(testing.allocator);
    var front_w = try h3.weights.loadRefiner(testing.allocator, io, device, tiny.front_dir, prompt_tool.tiny_front);
    defer front_w.deinit(testing.allocator);
    const T = 70;
    const ids = try a.alloc(u32, T);
    for (ids, 0..) |*id, i| id.* = @intCast(i % g.vocab);
    const pos = try a.alloc([3]i32, T);
    for (pos, 0..) |*p, i| p.* = @splat(@intCast(i));
    const rows = try a.alloc(u16, T * g.hidden);
    const text = try a.alloc(u16, T * prompt_tool.tiny_front.hidden);
    try runner.encode(&text_w, ids, pos, null, rows);
    try runner.refine(&front_w, rows, text);
    const held_rows = try a.alloc(u16, rows.len);
    const held_text = try a.alloc(u16, text.len);
    // the second probe: speech arrives once the first of the two layers (or of the refiner's three passes) is submitted
    runner.give_way = Arrival.at(io, lock, 2, hold_ms);
    try runner.encode(&text_w, ids, pos, null, held_rows);
    try Arrival.join();
    try testing.expect(runner.paused >= least);
    runner.give_way = Arrival.at(io, lock, 2, hold_ms);
    try runner.refine(&front_w, held_rows, held_text);
    try Arrival.join();
    try testing.expect(runner.paused >= least);
    try testing.expectEqualSlices(u16, rows, held_rows);
    try testing.expectEqualSlices(u16, text, held_text);

    // the decoder: 2 x 2 tiles of a 20 x 28 cell latent, one pass each
    const d = try h3.Decoder.init(testing.allocator, device, 2, 100);
    defer d.deinit();
    try d.synthetic(0x5eed);
    const config: h3.vae.Config = .{ .mean = @splat(0), .std = @splat(1), .tile = 256, .overlap = 64, .rope_theta = 100 };
    const latent = try a.alloc(f32, try h3.host.latentLength(20, 28));
    for (latent, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) / 8 - 1;
    const pixels = try a.alloc(f32, 3 * 20 * 28 * h3.vae.cell_pixels * h3.vae.cell_pixels);
    _ = try d.decode(a, &config, latent, 20, 28, 2, 1, pixels, null);
    const held_pixels = try a.alloc(f32, pixels.len);
    d.give_way = Arrival.at(io, lock, 2, hold_ms); // after the first of four tiles
    const took = try d.decode(a, &config, latent, 20, 28, 2, 1, held_pixels, null);
    try Arrival.join();
    try testing.expect(took.paused >= least);
    try testing.expectEqualSlices(f32, pixels, held_pixels);

    // denoising: one step of the resident transformer
    if (!device.tensorUnits()) return;
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = std.process.Init{
        .minimal = .{ .environ = .empty, .args = .{ .vector = &.{} } },
        .arena = &arena,
        .gpa = testing.allocator,
        .io = io,
        .environ_map = &environ_map,
        .preopens = .empty,
    };
    var f = try residentFixture(init, a, tmp.dir, dir_path, device);
    defer {
        for (&f.cases) |*c| c.case.file.deinit();
        f.folder.heads.deinit();
        for (f.folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(f.folder.ck.shards);
    }
    f.folder.turn = .{ .io = io, .at = lock };
    const case = &f.cases[0].case;
    var b = try Binding.init(testing.allocator, &f.folder, case, f.cases[0].tags, null, null);
    defer b.release();
    const velocity = try a.alloc(f32, case.noise.len);
    _ = try b.model.forward(case.noise, &.{}, 0, null, velocity, &.{});
    const held_velocity = try a.alloc(f32, velocity.len);
    // the third probe, after `mark` and the forward's first yield: speech arrives once the first block is submitted
    b.model.give_way = Arrival.at(io, lock, 3, hold_ms);
    const step = try b.model.forward(case.noise, &.{}, 0, null, held_velocity, &.{});
    try Arrival.join();
    try testing.expect(step.paused >= least and !step.timing_valid);
    try testing.expectEqualSlices(f32, velocity, held_velocity);
}

test "an over-long request line is drained and the next one read" {
    var input: [5031]u8 = undefined;
    @memset(input[0..5000], 'x');
    input[5000] = '\n';
    @memcpy(input[5001..], "2048x2048 42 out/a prompt.txt\n");
    var fixed = std.Io.Reader.fixed(&input);
    var buf: [4096]u8 = undefined;
    var limited = fixed.limited(.unlimited, &buf);
    try testing.expectError(error.StreamTooLong, nextLine(&limited.interface));
    try testing.expectEqualStrings("2048x2048 42 out/a prompt.txt", (try nextLine(&limited.interface)).?);
    try testing.expectEqual(@as(?[]const u8, null), try nextLine(&limited.interface));
}

test "startup stops when the weights cannot be held resident" {
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const buffer = try device.buffer(64, mtl.ResourceOptions.shared);
    defer buffer.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    // a limit no weight set passes: the server refuses to start rather than serve unwarmed
    try testing.expectError(error.WeightsNotResident, holdWeights(testInit(&arena, &environ_map), &.{buffer}, device, try device.queue(), 0));
}

test "startup stops when the idle ticker cannot start" {
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const buffer = try device.buffer(64, mtl.ResourceOptions.shared);
    defer buffer.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = testInit(&arena, &environ_map);
    var held = (try mtl.resident.Resident.hold(testing.allocator, init.io, device, try device.queue(), &.{buffer}, "h3 image test")) orelse return error.SkipZigTest;
    defer held.deinit();
    var target: mtl.keepalive.Target = undefined;
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    try testing.expectError(error.OutOfMemory, warmStart(init, failing.allocator(), &held, &target));
}

// A --draft-steps request on the synthetic model: the draft plan's pooled noise and the full run's
// determinism; bad draft counts and a bad half-grid refuse.
test "draft plan pools the seed's noise and the draft run is deterministic" {
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = testInit(&arena, &environ_map);
    var f = try residentFixture(init, a, tmp.dir, dir_path, device);
    defer {
        for (&f.cases) |*c| c.case.file.deinit();
        f.folder.heads.deinit();
        for (f.folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(f.folder.ck.shards);
    }
    const case = &f.cases[0].case; // 64x64 canvas: a 4x4 latent grid, whose half is 2x2

    // the draft schedule of a plain 3-step draft is pdd's prefix (steps 0,1,2 then the lift from
    // sigma[2] to sigma[3]); the half-grid noise is cascade.pool of the unpatchified seed noise
    const plan = try image_tool.draftPlan(a, &f.folder, case, 3, null);
    try testing.expectEqual(f.folder.pdd.steps(), plan.sched.steps());
    try testing.expect(plan.heads == null); // no renoise: distilled heads as they stand
    const full_grid = h3.cascade.Grid{ .rows = case.grid.rows, .columns = case.grid.columns };
    const seed_latent = try a.alloc(f32, plan.noise_latent.len);
    try h3.picture.unpatchify(case.noise, case.grid, seed_latent);
    const want_pool = try a.alloc(f32, h3.picture.channels * plan.draft_grid.rows * plan.draft_grid.columns);
    try h3.cascade.pool(seed_latent, full_grid, want_pool);
    const got_latent = try a.alloc(f32, want_pool.len);
    try h3.picture.unpatchify(plan.draft_noise, plan.draft_grid, got_latent);
    try testing.expectEqualSlices(f32, want_pool, got_latent);

    // the draft case and both bindings, then two draft runs: bit-identical latents
    const pic_name = try std.fmt.allocPrintSentinel(a, "{s}/a.pic.safetensors", .{dir_path}, 0);
    var pic = try h3.Tensors.open(testing.allocator, device, pic_name);
    defer pic.deinit();
    const draft_prefix = try std.fmt.allocPrint(a, "{s}/draft", .{dir_path});
    const draft_case = try image_tool.Case.write(init, a, &f.folder, &pic, draft_prefix, .{ .sched = plan.sched, .grid = plan.draft_grid, .noise = plan.draft_noise, .video_out = plan.heads });
    defer draft_case.file.deinit();
    var b = try Binding.init(testing.allocator, &f.folder, case, f.cases[0].tags, null, null);
    defer b.release();
    var db = try Binding.init(testing.allocator, &f.folder, &draft_case, f.cases[0].tags, null, plan.sched);
    defer db.release();
    const first = try image_tool.drafted(a, b.model, db.model, case, &draft_case, plan.sched, 3, plan.noise_latent, try std.fmt.allocPrint(a, "{s}/one", .{dir_path}));
    const second = try image_tool.drafted(a, b.model, db.model, case, &draft_case, plan.sched, 3, plan.noise_latent, try std.fmt.allocPrint(a, "{s}/two", .{dir_path}));
    try expectSameLatent(first.latent, second.latent);

    // a renoise at sigma .5 runs the 3 drafts to .6, lifts to .5, then resumes the grid under it:
    // seven steps, each video_out row the headSpan blend of the distilled heads it covers
    const renoised = try image_tool.draftPlan(a, &f.folder, case, 3, 0.5);
    try testing.expectEqual(@as(usize, 7), renoised.sched.steps());
    const blended = renoised.heads orelse return error.TestUnexpectedResult;
    const sw = synthetic.width;
    const hidden = f.folder.hidden;
    const hw = f.folder.heads.values(f32, try f.folder.heads.at("video_out.weight", .f32, f.folder.pdd.steps() * sw * hidden), f.folder.pdd.steps() * sw * hidden);
    const hb = f.folder.heads.values(f32, try f.folder.heads.at("video_out.bias", .f32, f.folder.pdd.steps() * sw), f.folder.pdd.steps() * sw);
    var saw_off_grid = false;
    for (0..renoised.sched.steps()) |s| {
        const span = try h3.picture.headSpan(f.folder.pdd.sigmas, renoised.sched.sigmas, s);
        saw_off_grid = saw_off_grid or span.first != s;
        const want_w = try a.alloc(f32, sw * hidden);
        @memset(want_w, 0);
        const want_b = try a.alloc(f32, sw);
        @memset(want_b, 0);
        for (span.first..span.last + 1, 0..) |head, j| {
            for (want_w, hw[head * sw * hidden ..][0 .. sw * hidden]) |*o, v| o.* += span.weights[j] * v;
            for (want_b, hb[head * sw ..][0..sw]) |*o, v| o.* += span.weights[j] * v;
        }
        try testing.expectEqualSlices(f32, want_w, blended.w[s * sw * hidden ..][0 .. sw * hidden]);
        try testing.expectEqualSlices(f32, want_b, blended.b[s * sw ..][0..sw]);
    }
    try testing.expect(saw_off_grid); // the .5 step's head is a mid-grid blend

    // the run case with the blended heads and its rescheduled binding, then the renoised draft run
    const run_prefix = try std.fmt.allocPrint(a, "{s}/run", .{dir_path});
    var run_case = try image_tool.Case.write(init, a, &f.folder, &pic, run_prefix, .{ .sched = renoised.sched, .video_out = renoised.heads });
    defer run_case.file.deinit();
    const renoised_draft_prefix = try std.fmt.allocPrint(a, "{s}/draft-renoised", .{dir_path});
    var renoised_draft_case = try image_tool.Case.write(init, a, &f.folder, &pic, renoised_draft_prefix, .{ .sched = renoised.sched, .grid = renoised.draft_grid, .noise = renoised.draft_noise, .video_out = renoised.heads });
    defer renoised_draft_case.file.deinit();
    var rb = try Binding.init(testing.allocator, &f.folder, &run_case, f.cases[0].tags, null, renoised.sched);
    defer rb.release();
    var rdb = try Binding.init(testing.allocator, &f.folder, &renoised_draft_case, f.cases[0].tags, null, renoised.sched);
    defer rdb.release();
    const done_a = try image_tool.drafted(a, rb.model, rdb.model, &run_case, &renoised_draft_case, renoised.sched, 3, renoised.noise_latent, try std.fmt.allocPrint(a, "{s}/done-a", .{dir_path}));
    const done_b = try image_tool.drafted(a, rb.model, rdb.model, &run_case, &renoised_draft_case, renoised.sched, 3, renoised.noise_latent, try std.fmt.allocPrint(a, "{s}/done-b", .{dir_path}));
    try expectSameLatent(done_a.latent, done_b.latent);

    // serve's rewrite is write-then-deinit: a failed rewrite leaves the first case alone, deinited once
    var keep = try image_tool.Case.write(init, a, &f.folder, &pic, try std.fmt.allocPrint(a, "{s}/keep", .{dir_path}), null);
    defer keep.file.deinit();
    var oom_init = init;
    var oom_gpa = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    oom_init.gpa = oom_gpa.allocator();
    try testing.expectError(error.OutOfMemory, image_tool.Case.write(oom_init, a, &f.folder, &pic, run_prefix, .{ .sched = renoised.sched, .video_out = renoised.heads }));
    try testing.expect(oom_gpa.has_induced_failure);

    // refusals: no draft steps, more than pdd has, all of them without a renoise, a renoise that
    // doesn't sit under sigma[drafts-1]
    try testing.expectError(error.H3BadDraftSteps, image_tool.draftPlan(a, &f.folder, case, 0, null));
    try testing.expectError(error.H3BadDraftSteps, image_tool.draftPlan(a, &f.folder, case, 7, null));
    try testing.expectError(error.H3DraftLeavesNoStep, image_tool.draftPlan(a, &f.folder, case, 6, null));
    try testing.expectError(error.H3BadRenoiseSigma, image_tool.draftPlan(a, &f.folder, case, 3, 0.9));

    // a canvas whose latent grid doesn't halve cleanly refuses: 96x96 is a 6x6 latent grid
    try testing.expectError(error.H3BadCascadeGrid, image_tool.draftPlan(a, &f.folder, &f.cases[3].case, 3, null));
}
