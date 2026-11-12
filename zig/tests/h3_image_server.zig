//! tf-h3-image-server: MiniMax-H3-Image's prompt encoder, transformer and decoder loaded once, then one picture per
//! request line on stdin, each step as the one-shot tools run it.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3");
const prompt_tool = @import("h3_image_prompt.zig");
const image_tool = @import("h3_image.zig");
const decoder_tool = @import("h3_image_decoder.zig");

const usage =
    \\usage: tf-h3-image-server TEXT_ENCODER_DIR MODEL_DIR
    \\  then one request a line on stdin: WIDTHxHEIGHT SEED OUT_PREFIX prompt.txt
    \\  each writes OUT_PREFIX.safetensors, .h3case.safetensors, .rows.f32, .latent.f32 and .png as
    \\  tf-h3-image-prompt, tf-h3-image (free) and tf-h3-image-decoder do, and prints a timing line;
    \\  a request with the last one's canvas, rows and tags reuses its transformer binding
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

    fn init(gpa: std.mem.Allocator, folder: *const image_tool.Folder, case: *const image_tool.Case, tags: []const i32) !Binding {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        errdefer gpa.destroy(arena);
        arena.* = .init(gpa);
        errdefer arena.deinit();
        const model = try folder.bind(arena.allocator(), arena.allocator(), case);
        return .{ .arena = arena, .model = model, .canvas = case.canvas, .rows = case.rows, .references = case.layout.references.len, .tags = try arena.allocator().dupe(i32, tags) };
    }

    // Text-to-image only: serve creates no reference rows; canvas, rows and tags determine every row and rotary position.
    fn fits(b: *const Binding, case: *const image_tool.Case, tags: []const i32) bool {
        return std.meta.eql(b.canvas, case.canvas) and b.rows == case.rows and b.references == case.layout.references.len and std.mem.eql(i32, b.tags, tags);
    }

    fn release(b: *Binding) void {
        b.model.release();
        const gpa = b.arena.child_allocator;
        b.arena.deinit();
        gpa.destroy(b.arena);
    }
};

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const loading = mtl.clock.seconds();
    var encoder = try prompt_tool.Encoder.load(init, a, args[1], args[2]);
    defer encoder.deinit(init.gpa);
    var folder = try image_tool.Folder.open(init, a, args[2]);
    defer folder.heads.deinit();
    const vae = try decoder_tool.Vae.load(init, a, args[2]);
    defer vae.d.deinit();
    std.debug.print("loaded the prompt encoder, transformer weights and decoder in {d:.1} s; ready\n", .{mtl.clock.seconds() - loading});

    var binding: ?Binding = null;
    defer if (binding) |*b| b.release();
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
        serve(init, request.allocator(), &encoder, &folder, &vae, &binding, fields) catch |err| {
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
fn serve(init: std.process.Init, a: std.mem.Allocator, encoder: *prompt_tool.Encoder, folder: *const image_tool.Folder, vae: *const decoder_tool.Vae, binding: *?Binding, line: []const u8) !void {
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
    const case = try image_tool.Case.write(init, a, folder, &pic, prefix);
    defer case.file.deinit();
    const tags = pic.values(i32, try pic.at("text_tags", .i32, case.layout.text), case.layout.text);
    const cased = mtl.clock.seconds();
    const reused = if (binding.*) |*b| b.fits(&case, tags) else false;
    if (reused) {
        try binding.*.?.model.retext(case.file);
    } else {
        if (binding.*) |*b| b.release();
        binding.* = null;
        binding.* = try Binding.init(init.gpa, folder, &case, tags);
    }
    const bound = mtl.clock.seconds();

    const out = try image_tool.free(a, binding.*.?.model, folder, &case, prefix);
    const denoised = mtl.clock.seconds();
    try vae.png(init, a, out.latent, case.grid.rows, case.grid.columns, .{ .width = size.width, .height = size.height }, try std.fmt.allocPrint(a, "{s}.png", .{prefix}));
    const done = mtl.clock.seconds();
    std.debug.print("{s}: prompt {d:.2} s, case {d:.2} s, transformer {s} {d:.2} s, 6 steps {d:.2} s, decoder {d:.2} s, total {d:.2} s\n", .{ prefix, encoded - started, cased - encoded, if (reused) "reused" else "bound", bound - cased, denoised - bound, done - denoised, done - started });
}

const testing = std.testing;
const fixture = @import("checkpoint_fixture");

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
    try s.add(a, &entries, "noise", "F32", &.{ try h3.picture.gridRows(grid) * s.width }, seed + 100);
    try fixture.write(a, io, dir, name, entries.items);
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
    const case = try image_tool.Case.write(init, a, folder, &pic, prefix);
    const tags = try a.dupe(i32, pic.values(i32, try pic.at("text_tags", .i32, case.layout.text), case.layout.text));
    return .{ .case = case, .tags = tags };
}

fn freshLatent(a: std.mem.Allocator, folder: *const image_tool.Folder, case: *const image_tool.Case, tags: []const i32, prefix: []const u8) ![]f32 {
    var b = try Binding.init(testing.allocator, folder, case, tags);
    defer b.release();
    return (try image_tool.free(a, b.model, folder, case, prefix)).latent;
}

fn residentLatent(a: std.mem.Allocator, folder: *const image_tool.Folder, binding: *?Binding, case: *const image_tool.Case, tags: []const i32, prefix: []const u8) ![]f32 {
    const reused = if (binding.*) |*b| b.fits(case, tags) else false;
    if (reused) {
        try binding.*.?.model.retext(case.file);
    } else {
        if (binding.*) |*b| b.release();
        binding.* = null;
        binding.* = try Binding.init(testing.allocator, folder, case, tags);
    }
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

test "resident binding matches fresh bindings across prompts, rows and canvas" {
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
    const init = std.process.Init{
        .minimal = .{ .environ = .empty, .args = .{ .vector = &.{} } },
        .arena = &arena,
        .gpa = testing.allocator,
        .io = std.testing.io,
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
    var binding: ?Binding = null;
    defer if (binding) |*b| b.release();

    const fresh_a = try freshLatent(a, &f.folder, &f.cases[0].case, f.cases[0].tags, try std.fmt.allocPrint(a, "{s}/fresh-a", .{dir_path}));
    const resident_a = try residentLatent(a, &f.folder, &binding, &f.cases[0].case, f.cases[0].tags, try std.fmt.allocPrint(a, "{s}/resident-a", .{dir_path}));
    try expectSameLatent(fresh_a, resident_a);

    // a different prompt of the same length reuses the binding and replaces its text rows
    try testing.expect(binding.?.fits(&f.cases[1].case, f.cases[1].tags));
    const fresh_b = try freshLatent(a, &f.folder, &f.cases[1].case, f.cases[1].tags, try std.fmt.allocPrint(a, "{s}/fresh-b", .{dir_path}));
    const resident_b = try residentLatent(a, &f.folder, &binding, &f.cases[1].case, f.cases[1].tags, try std.fmt.allocPrint(a, "{s}/resident-b", .{dir_path}));
    try expectSameLatent(fresh_b, resident_b);

    // a changed token count and a changed canvas each release the binding and bind anew
    try testing.expect(!binding.?.fits(&f.cases[2].case, f.cases[2].tags));
    const fresh_c = try freshLatent(a, &f.folder, &f.cases[2].case, f.cases[2].tags, try std.fmt.allocPrint(a, "{s}/fresh-c", .{dir_path}));
    const resident_c = try residentLatent(a, &f.folder, &binding, &f.cases[2].case, f.cases[2].tags, try std.fmt.allocPrint(a, "{s}/resident-c", .{dir_path}));
    try expectSameLatent(fresh_c, resident_c);

    try testing.expect(!binding.?.fits(&f.cases[3].case, f.cases[3].tags));
    const fresh_d = try freshLatent(a, &f.folder, &f.cases[3].case, f.cases[3].tags, try std.fmt.allocPrint(a, "{s}/fresh-d", .{dir_path}));
    const resident_d = try residentLatent(a, &f.folder, &binding, &f.cases[3].case, f.cases[3].tags, try std.fmt.allocPrint(a, "{s}/resident-d", .{dir_path}));
    try expectSameLatent(fresh_d, resident_d);

    // back to the first request: a fresh binding again, the same latent
    const resident_a2 = try residentLatent(a, &f.folder, &binding, &f.cases[0].case, f.cases[0].tags, try std.fmt.allocPrint(a, "{s}/resident-a2", .{dir_path}));
    try expectSameLatent(fresh_a, resident_a2);
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
    const init = std.process.Init{
        .minimal = .{ .environ = .empty, .args = .{ .vector = &.{} } },
        .arena = &arena,
        .gpa = testing.allocator,
        .io = std.testing.io,
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
    const baseline = device.allocated();
    var counting = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = std.math.maxInt(usize) });
    const model = try h3.Model.bind(counting.allocator(), f.folder.ck, f.cases[0].case.file, .{});
    model.release();
    try testing.expectEqual(counting.allocated_bytes, counting.freed_bytes);
    try testing.expectEqual(baseline, device.allocated());

    // the last allocation is the block table: every buffer and pipeline exists when it fails
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = counting.alloc_index - 1 });
    try testing.expectError(error.OutOfMemory, h3.Model.bind(failing.allocator(), f.folder.ck, f.cases[0].case.file, .{}));
    try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    try testing.expectEqual(baseline, device.allocated());
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
