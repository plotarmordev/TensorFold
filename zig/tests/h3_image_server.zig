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
    \\  a request with the last one's canvas, text length and tags reuses its transformer binding
    \\
;

/// The transformer bound to one picture layout, its host copies in its own arena.
const Binding = struct {
    arena: *std.heap.ArenaAllocator,
    model: *h3.Model,
    canvas: h3.picture.Size,
    tags: []const i32,

    fn init(gpa: std.mem.Allocator, folder: *const image_tool.Folder, case: *const image_tool.Case, tags: []const i32) !Binding {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        errdefer gpa.destroy(arena);
        arena.* = .init(gpa);
        errdefer arena.deinit();
        const model = try folder.bind(arena.allocator(), arena.allocator(), case);
        return .{ .arena = arena, .model = model, .canvas = case.canvas, .tags = try arena.allocator().dupe(i32, tags) };
    }

    fn fits(b: *const Binding, case: *const image_tool.Case, tags: []const i32) bool {
        return std.meta.eql(b.canvas, case.canvas) and std.mem.eql(i32, b.tags, tags);
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
    while (in.interface.takeDelimiter('\n') catch |err| return err) |line| {
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
