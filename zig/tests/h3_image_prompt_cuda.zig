//! tf-h3-image-prompt-cuda: MiniMax-H3-Image's prompt stage on CUDA, the same case
//! tf-h3-image-prompt(-host) writes — Qwen3-VL's text to layer 50 and the two-block refiner on the
//! GPU, or its synthetic selftest against the float64 host reference.
const std = @import("std");
const hp = @import("h3_prompt_cuda");
const tokenizer = @import("tokenizer");
const fixture = @import("checkpoint_fixture");
const bf16_round = @import("bf16_round");
const noise = @import("moss_noise");

const rules = hp.rules;
const ref = hp.reference;
const stage = hp.case;
const picture = hp.picture;
const ck = hp.checkpoint;
const files = @import("h3_image_files.zig");
const match = @import("h3_image_match.zig");

const usage =
    \\usage: tf-h3-image-prompt-cuda TEXT_ENCODER_DIR MODEL_DIR prompt.txt out.safetensors [WIDTHxHEIGHT [SEED [MIN_AREA]]]
    \\       tf-h3-image-prompt-cuda TEXT_ENCODER_DIR MODEL_DIR --serve
    \\       tf-h3-image-prompt-cuda selftest [WORK_DIR]
    \\  TEXT_ENCODER_DIR  MiniMax-H3's text_encoder/ (Qwen3-VL): config.json, its index and shards
    \\  MODEL_DIR         the MiniMax-H3-Image folder: model_index.json, tokenizer/, transformer/
    \\  prompt.txt        the prompt, verbatim but for one trailing line ending
    \\  out               text BF16 [rows, 5376] and text_tags I32 [rows], as tf-h3-image's picture case reads them,
    \\                    and ids I32 [rows]
    \\  WIDTHxHEIGHT      a text-to-image case: the canvas (grown to MIN_AREA, default 2048 x 2048, at the size's
    \\                    ratio, sides snapped to 32), its starting rows from SEED (default 0) as torch's CPU randn
    \\                    draws them, and the requested size
    \\  --serve           the weights loaded once, then one `prompt.txt out.safetensors WxH [SEED [MIN_AREA]]` line
    \\                    of stdin per picture, each answered `done OUT` or `failed OUT` on stdout
    \\  selftest          the GPU kernels on a tiny synthetic checkpoint against the float64 host reference;
    \\                    WORK_DIR (default under TMPDIR) holds a new folder with the checkpoint while it runs
    \\Exit 0: done or every gate passed; 1: a gate failed.
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(a);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "selftest") and args.len <= 3) {
        const tmp = init.environ_map.get("TMPDIR") orelse "/tmp";
        const dir = if (args.len == 3) args[2] else try std.fs.path.join(a, &.{ tmp, "tf-h3-image-prompt-cuda-selftest" });
        return selftest(init, a, gpa, dir);
    }
    const serving = args.len == 4 and std.mem.eql(u8, args[3], "--serve");
    if ((args.len < 5 or args.len > 8) and !serving) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var loaded = try Loaded.open(init.io, a, gpa, args[1], args[2]);
    defer loaded.deinit(gpa);
    if (serving) return serve(init, gpa, &loaded);
    try pictureRun(init.io, a, &loaded, args[3], args[4], args[5..]);
    return 0;
}

/// What every picture shares: the checked geometry, the tokenizer, the runner and the weights
/// resident on the device, the shard mappings kept only for their names — their pages evicted once
/// the resident copy landed, so a served process pins no host copy of the weights.
const Loaded = struct {
    enc: rules.Encoder,
    front: rules.Refiner,
    tok: tokenizer.Tokenizer,
    shards: Shards,
    front_shards: Shards,
    runner: *hp.Runner,
    res: hp.Resident,

    fn open(io: std.Io, a: std.mem.Allocator, gpa: std.mem.Allocator, encoder_dir: []const u8, model_dir: []const u8) !Loaded {
        // the folder and its tokenizer, checked as the host prompt path checks them
        const enc = try rules.checkEncoder(a, try read(io, a, &.{ encoder_dir, "config.json" }));
        _ = try picture.checkIndex(a, try read(io, a, &.{ model_dir, "model_index.json" }));
        const transformer = try std.fs.path.join(a, &.{ model_dir, "transformer" });
        const front = try rules.checkRefiner(a, try read(io, a, &.{ transformer, "config.json" }));
        var tok = try tokenizer.loadTokenizer(io, a, try std.fs.path.join(a, &.{ model_dir, "tokenizer" }));
        errdefer tok.deinit();

        const runner = try hp.Runner.init(gpa);
        errdefer runner.deinit();

        var shards = try Shards.open(io, a, gpa, encoder_dir);
        errdefer shards.deinit(gpa);
        const prefix = try rules.textPrefix(shards.where.keys());
        const embed = try shards.tensorBytes(try std.fmt.allocPrint(a, "{s}embed_tokens.weight", .{prefix}), .{ enc.text.vocab, enc.text.hidden });
        var front_shards = try Shards.open(io, a, gpa, transformer);
        errdefer front_shards.deinit(gpa);
        const front_w = try frontTensors(&front_shards, front);

        // the whole stage's weights into one device arena; the shards' pages then leave the cache
        const started = seconds();
        var layer_ctx: LayerCtx = .{ .shards = &shards, .g = enc.text, .prefix = prefix };
        var block_ctx: BlockCtx = .{ .shards = &front_shards, .g = front };
        const res = try hp.loadResident(runner, a, enc.text, front, embed, layerOf, &layer_ctx, blockOf, &block_ctx, front_w.cond_w, front_w.cond_b, front_w.final_norm);
        std.debug.print("{d:.2} GiB of prompt weights resident in {d:.2} s\n", .{ @as(f64, @floatFromInt(res.arena.len)) / (1 << 30), seconds() - started });
        for (shards.folder.files.values()) |*file| file.evict();
        for (front_shards.folder.files.values()) |*file| file.evict();
        return .{ .enc = enc, .front = front, .tok = tok, .shards = shards, .front_shards = front_shards, .runner = runner, .res = res };
    }

    fn deinit(l: *Loaded, gpa: std.mem.Allocator) void {
        l.res.deinit();
        l.runner.deinit();
        l.front_shards.deinit(gpa);
        l.shards.deinit(gpa);
        l.tok.deinit();
    }
};

/// One line of --serve's stdin per picture: `prompt.txt out.safetensors WxH [SEED [MIN_AREA]]`,
/// answered `done OUT` or `failed OUT` on stdout.
fn serve(init: std.process.Init, gpa: std.mem.Allocator, loaded: *Loaded) !u8 {
    var in_buf: [4096]u8 = undefined;
    var in = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    while (try in.interface.takeDelimiter('\n')) |line| {
        var request = std.heap.ArenaAllocator.init(gpa);
        defer request.deinit();
        var it = std.mem.tokenizeAny(u8, line, " \t\r");
        const prompt_path = it.next() orelse continue;
        const out_path = it.next() orelse prompt_path;
        var extra: [3][]const u8 = undefined;
        var rest: usize = 0;
        while (it.next()) |arg| {
            if (rest == extra.len) {
                rest += 1;
                break;
            }
            extra[rest] = arg;
            rest += 1;
        }
        const verdict = blk: {
            if (rest > extra.len) break :blk "failed";
            pictureRun(init.io, request.allocator(), loaded, prompt_path, out_path, extra[0..rest]) catch |err| {
                std.debug.print("{s} failed: {s}\n", .{ out_path, @errorName(err) });
                break :blk "failed";
            };
            break :blk "done";
        };
        try std.Io.File.stdout().writeStreamingAll(init.io, try std.fmt.allocPrint(request.allocator(), "{s} {s}\n", .{ verdict, out_path }));
    }
    return 0;
}

/// One picture: its prompt presented and run through the resident weights to `out`'s case file.
/// `rest` is the optional WIDTHxHEIGHT, SEED and MIN_AREA. `prompt: {d:.2} s` closes the log; the
/// bench parser reads it.
fn pictureRun(io: std.Io, a: std.mem.Allocator, loaded: *Loaded, prompt_path: []const u8, out_path: []const u8, rest: []const []const u8) !void {
    if (rest.len > 3) return error.BadArguments;
    const request: ?stage.Request = if (rest.len >= 1) .{
        .size = try stage.size(rest[0]),
        .seed = if (rest.len >= 2) try std.fmt.parseInt(u64, rest[1], 10) else 0,
        .min_area = if (rest.len == 3) try std.fmt.parseInt(u64, rest[2], 10) else picture.default_min_area,
    } else null;
    const started = seconds();

    var prompt: []const u8 = try std.Io.Dir.cwd().readFileAlloc(io, prompt_path, a, .limited(16 << 20));
    if (std.mem.endsWith(u8, prompt, "\n")) prompt = prompt[0 .. prompt.len - 1];
    if (std.mem.endsWith(u8, prompt, "\r")) prompt = prompt[0 .. prompt.len - 1];

    const p = try rules.present(a, &loaded.tok, loaded.enc.marks, &.{}, prompt);
    const pos = try rules.positions(a, p, loaded.enc.marks, &.{});
    const T = p.ids.len;
    const runner = loaded.runner;

    const enc_rows = try a.alloc(u16, T * loaded.enc.text.hidden);
    var gpu_started = seconds();
    try runner.encode(a, loaded.enc.text, &loaded.res, p.ids, pos, null, enc_rows);
    std.debug.print("{d} tokens; text encoder, {d} layers: {d:.2} s wall, {d:.2} s GPU\n", .{ T, loaded.enc.text.layers, seconds() - gpu_started, runner.gpu });

    const text = try a.alloc(u16, T * loaded.front.hidden);
    gpu_started = seconds();
    try runner.refine(loaded.front, &loaded.res, enc_rows, text);
    std.debug.print("condition_proj and the token refiner: {d:.2} s wall, {d:.2} s GPU\n", .{ seconds() - gpu_started, runner.gpu });

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = try stage.write(a, loaded.front.hidden, p, text, request, &noise.randn, &.{}) });
    std.debug.print("wrote {s}: text [{d}, {d}] bf16, text_tags and ids\n", .{ out_path, T, loaded.front.hidden });
    std.debug.print("prompt: {d:.2} s\n", .{seconds() - started});
}

fn read(io: std.Io, a: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    const path = try std.fs.path.join(a, parts);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

const seconds = hp.seconds;

/// A checkpoint folder on the host, as tf-h3-image-prompt-host's: the index's name-to-shard map, each
/// shard mapped once on first use; a layer's tensors are read, uploaded and freed before the next's.
const Shards = struct {
    gpa: std.mem.Allocator,
    /// Tensor name to shard file, both in the caller's arena.
    where: std.StringArrayHashMapUnmanaged([]const u8) = .empty,
    folder: ck.Folder,

    fn open(io: std.Io, a: std.mem.Allocator, gpa: std.mem.Allocator, dir: []const u8) !Shards {
        const text = try read(io, a, &.{ dir, "model.safetensors.index.json" });
        const doc = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.H3BadShardIndex;
        if (doc != .object) return error.H3BadShardIndex;
        const map = doc.object.get("weight_map") orelse return error.H3BadShardIndex;
        if (map != .object or map.object.count() == 0) return error.H3BadShardIndex;
        var s: Shards = .{ .gpa = gpa, .folder = .{ .dir = try a.dupe(u8, dir) } };
        var it = map.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string or std.mem.indexOf(u8, e.value_ptr.string, "/") != null) return error.H3BadShardIndex;
            try s.where.put(a, e.key_ptr.*, e.value_ptr.string);
        }
        return s;
    }

    fn deinit(s: *Shards, gpa: std.mem.Allocator) void {
        s.folder.deinit(gpa);
    }

    /// The shard holding `name`.
    fn shard(s: *Shards, name: []const u8) !*ck.Checkpoint {
        const file = s.where.get(name) orelse {
            std.debug.print("no tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        return s.folder.shard(s.gpa, file);
    }

    /// The bf16 tensor `name` at `shape` as stored bytes (columns 0: a vector of `rows` values).
    fn tensorBytes(s: *Shards, name: []const u8, shape: rules.Shape) !hp.Tensor {
        const e = try (try s.shard(name)).bf16(name);
        const rows_: usize = if (shape[1] == 0) 1 else shape[0];
        const cols: usize = if (shape[1] == 0) shape[0] else shape[1];
        if (e.rows != rows_ or e.cols != cols) {
            std.debug.print("{s} is [{d}, {d}]; the prompt path reads [{d}, {d}]\n", .{ name, e.rows, e.cols, rows_, cols });
            return error.H3UnexpectedTensor;
        }
        return e.bytes;
    }
};

/// `encode`'s layer callback: layer `li`'s 11 bf16 tensors from the mapped shards.
const LayerCtx = struct { shards: *Shards, g: rules.Text, prefix: []const u8 };

fn layerOf(ctx: *anyopaque, li: usize) anyerror!hp.LayerSet {
    const c: *LayerCtx = @ptrCast(@alignCast(ctx));
    var set: hp.LayerSet = undefined;
    var name: [176]u8 = undefined;
    inline for (comptime std.meta.fieldNames(hp.LayerSet), rules.layerTensors(c.g)) |field, spec| {
        std.debug.assert(std.mem.eql(u8, field, spec[0]));
        @field(set, field) = try c.shards.tensorBytes(try std.fmt.bufPrint(&name, "{s}layers.{d}.{s}", .{ c.prefix, li, spec[1] }), spec[2]);
    }
    return set;
}

/// `refine`'s block callback: block `bi`'s 8 bf16 tensors from the transformer's shards.
const BlockCtx = struct { shards: *Shards, g: rules.Refiner };

fn blockOf(ctx: *anyopaque, bi: usize) anyerror!hp.BlockSet {
    const c: *BlockCtx = @ptrCast(@alignCast(ctx));
    var set: hp.BlockSet = undefined;
    var name: [96]u8 = undefined;
    inline for (comptime std.meta.fieldNames(hp.BlockSet), rules.blockTensors(c.g)) |field, spec| {
        std.debug.assert(std.mem.eql(u8, field, spec[0]));
        @field(set, field) = try c.shards.tensorBytes(try std.fmt.bufPrint(&name, "token_refiner.blocks.{d}.{s}", .{ bi, spec[1] }), spec[2]);
    }
    return set;
}

/// condition_proj's weight and bias and the final norm, checked at the refiner's shapes.
fn frontTensors(s: *Shards, g: rules.Refiner) !struct { cond_w: hp.Tensor, cond_b: hp.Tensor, final_norm: hp.Tensor } {
    var w: struct { cond_w: hp.Tensor, cond_b: hp.Tensor, final_norm: hp.Tensor } = undefined;
    inline for (.{ "cond_w", "cond_b", "final_norm" }, rules.frontTensors(g)) |field, spec| {
        std.debug.assert(std.mem.eql(u8, field, spec[0]));
        @field(w, field) = try s.tensorBytes(spec[1], spec[2]);
    }
    return .{ .cond_w = w.cond_w, .cond_b = w.cond_b, .final_norm = w.final_norm };
}

/// The selftest's geometry, as tf-h3-image-prompt's: two layers of a 256-wide text model
/// (4 query and 2 KV heads), a 384-wide refiner; 70 rows, past one 64-row tile.
const tiny_text: rules.Text = .{ .hidden = 256, .heads = 4, .kv_heads = 2, .ffn = 512, .vocab = 96, .layers = 2, .eps = 1e-6, .theta = 1e4, .sections = .{ 24, 20, 20 } };
const tiny_front: rules.Refiner = .{ .text = 256, .hidden = 384, .heads = 2, .mlp = 512, .blocks = 2, .eps = 1e-5, .qk_eps = 1e-5, .final_eps = 1e-5 };
const tiny_rows = 70;
/// The tiny checkpoint's stored bf16 bytes for `name`, missing ones refused rather than unwrapped.
fn bitsOf(tiny: *const Tiny, name: []const u8) !hp.Tensor {
    return tiny.bits.get(name) orelse return error.MissingTensor;
}

/// A synthetic checkpoint folder: bf16 tensors drawn uniformly, one shard and its index.
const Synth = struct {
    a: std.mem.Allocator,
    rand: std.Random,
    entries: std.ArrayList(fixture.Entry) = .empty,

    /// A tensor `name` of `shape` drawn in center +- spread, kept in the shard; its values as fp32.
    fn tensor(s: *Synth, name: []const u8, shape: rules.Shape, center: f32, spread: f32) ![]const f32 {
        const n = shape[0] * @max(shape[1], 1);
        const vals = try s.a.alloc(f32, n);
        const bits = try s.a.alloc(u16, n);
        for (vals, bits) |*v, *b| {
            b.* = bf16_round.bits(center + spread * (2 * s.rand.float(f32) - 1));
            v.* = bf16_round.value(b.*);
        }
        const dims: []const usize = if (shape[1] == 0) try s.a.dupe(usize, shape[0..1]) else try s.a.dupe(usize, &shape);
        try s.entries.append(s.a, .{ .name = try s.a.dupe(u8, name), .dtype = "BF16", .shape = dims, .bytes = std.mem.sliceAsBytes(bits) });
        return vals;
    }

    /// A weight by its field: norms near 1, projections of unit gain, biases small.
    fn weight(s: *Synth, name: []const u8, field: []const u8, shape: rules.Shape) ![]const f32 {
        if (std.mem.indexOf(u8, field, "norm") != null) return s.tensor(name, shape, 1, 0.2);
        if (shape[1] == 0) return s.tensor(name, shape, 0, 0.1);
        return s.tensor(name, shape, 0, @sqrt(3.0 / @as(f32, @floatFromInt(shape[1]))));
    }

    fn write(s: *Synth, io: std.Io, dir: []const u8) !void {
        try files.checkpoint(io, s.a, dir, s.entries.items);
    }
};

/// The tiny text model and refiner written under `dir` (text_encoder/, transformer/), their host
/// copies for the float64 reference and their stored bf16 bytes for the GPU path.
const Tiny = struct {
    text_dir: []const u8,
    front_dir: []const u8,
    host: ref.Text,
    front: ref.Refiner,
    /// The stored bf16 tensors, keyed as the checkpoint names them.
    bits: std.StringArrayHashMapUnmanaged([]const u8),

    fn write(io: std.Io, a: std.mem.Allocator, rand: std.Random, dir: []const u8) !Tiny {
        const g = tiny_text;
        const f = tiny_front;
        var tiny: Tiny = .{ .text_dir = try std.fs.path.join(a, &.{ dir, "text_encoder" }), .front_dir = try std.fs.path.join(a, &.{ dir, "transformer" }), .host = undefined, .front = undefined, .bits = .empty };
        // the text model: three layers written, two run, beside a vision tensor the loader must leave alone
        var text = Synth{ .a = a, .rand = rand };
        const prefix = "model.language_model.";
        const embed = try text.tensor(prefix ++ "embed_tokens.weight", .{ g.vocab, g.hidden }, 0, 1);
        try tiny.bits.put(a, prefix ++ "embed_tokens.weight", text.entries.items[text.entries.items.len - 1].bytes);
        _ = try text.tensor("model.visual.blocks.0.norm1.weight", .{ 128, 0 }, 1, 0.2);
        const layers = try a.alloc(ref.Layer, g.layers + 1);
        for (layers, 0..) |*l, i| inline for (comptime std.meta.fieldNames(ref.Layer), rules.layerTensors(g)) |field, s| {
            const name = try std.fmt.allocPrint(a, "{s}layers.{d}.{s}", .{ prefix, i, s[1] });
            @field(l, field) = try text.weight(name, field, s[2]);
            try tiny.bits.put(a, name, text.entries.items[text.entries.items.len - 1].bytes);
        };
        var front = Synth{ .a = a, .rand = rand };
        inline for (.{ "cond_w", "cond_b", "final_norm" }, rules.frontTensors(f)) |field, s| {
            @field(tiny.front, field) = try front.weight(s[1], field, s[2]);
            try tiny.bits.put(a, try a.dupe(u8, s[1]), front.entries.items[front.entries.items.len - 1].bytes);
        }
        const blocks = try a.alloc(ref.Block, f.blocks);
        for (blocks, 0..) |*b, i| inline for (comptime std.meta.fieldNames(ref.Block), rules.blockTensors(f)) |field, s| {
            const name = try std.fmt.allocPrint(a, "token_refiner.blocks.{d}.{s}", .{ i, s[1] });
            @field(b, field) = try front.weight(name, field, s[2]);
            try tiny.bits.put(a, name, front.entries.items[front.entries.items.len - 1].bytes);
        };
        tiny.front.blocks = blocks;
        tiny.host = .{ .embed = embed, .layers = layers };
        try text.write(io, tiny.text_dir);
        try front.write(io, tiny.front_dir);
        return tiny;
    }
};

/// A whole run's rows against float64, gated by a float32-sum host run's distance from it, as
/// tf-h3-image-prompt's `whole` gates: 1 - cosine and relative error at most twice the host run's,
/// cosine at least 0.999.
fn whole(a: std.mem.Allocator, label: []const u8, ours: []const u16, sum32: []const f32, exact: []const f32, width: usize) !bool {
    const host_noise = match.match(try match.bf16s(a, sum32), exact, width);
    const m = match.match(ours, exact, width);
    const pass = match.within(m, host_noise, 2, 0.999);
    match.report(label, m, pass);
    std.debug.print("  the float32-sum host run: cosine {d:.7}, relative error {e:.2}, {d:.4} bit-equal\n", .{ host_noise.cosine, host_noise.relative, host_noise.equal });
    return pass;
}

fn selftest(init: std.process.Init, a: std.mem.Allocator, gpa: std.mem.Allocator, dir: []const u8) !u8 {
    const io = init.io;
    var prng = std.Random.DefaultPrng.init(0x5133);
    const rand = prng.random();
    const g = tiny_text;
    const f = tiny_front;
    const scratch = try files.Scratch.init(io, a, dir);
    defer scratch.deinit(io);
    const tiny = try Tiny.write(io, a, rand, scratch.path);

    const runner = try hp.Runner.init(gpa);
    defer runner.deinit();
    std.debug.print("synthetic text model {d} wide ({d} layers run of {d}), refiner {d} wide, {d} rows\n", .{ g.hidden, g.layers, g.layers + 1, f.hidden, tiny_rows });
    var failed = false;

    const T = tiny_rows;
    const ids = try a.alloc(u32, T);
    for (ids) |*id| id.* = rand.uintLessThan(u32, @intCast(g.vocab));
    const pos = try a.alloc([3]i32, T);
    for (pos, 0..) |*p, i| p.* = @splat(@intCast(i));

    // the resident weights, loaded once from the tiny checkpoint's stored bf16 bytes
    const prefix = "model.language_model.";
    var bits_layer: Ctx = .{ .tiny = &tiny, .a = a, .g = g };
    var bits_front: Ctx = .{ .tiny = &tiny, .a = a, .g = g };
    var res = try hp.loadResident(runner, a, g, f, try bitsOf(&tiny, try std.fmt.allocPrint(a, "{s}embed_tokens.weight", .{prefix})), bitsLayer, &bits_layer, bitsBlock, &bits_front, try bitsOf(&tiny, "condition_proj.weight"), try bitsOf(&tiny, "condition_proj.bias"), try bitsOf(&tiny, "token_refiner.final_norm.weight"));
    defer res.deinit();
    const enc = try a.alloc(u16, T * g.hidden);
    const started = seconds();
    try runner.encode(a, g, &res, ids, pos, null, enc);
    const enc_s = seconds() - started;
    const out = try a.alloc(u16, T * f.hidden);
    try runner.refine(f, &res, enc, out);

    // the float64 oracle and the float32-sum host run over the same weights
    const cs = try a.alloc(f32, T * 64);
    const sn = try a.alloc(f32, T * 64);
    rules.rotary(cs, sn, pos, g.theta, g.sections);
    const host = tiny.host;
    const enc_ref = try a.alloc(f32, T * g.hidden);
    try ref.encode(f64, a, g, host, ids, cs, sn, null, enc_ref);
    const enc_32 = try a.alloc(f32, T * g.hidden);
    try ref.encode(f32, a, g, host, ids, cs, sn, null, enc_32);
    failed = !try whole(a, "text encoder rows, a text prompt", enc, enc_32, enc_ref, g.hidden) or failed;
    const enc_f = try a.alloc(f32, enc.len);
    for (enc_f, enc) |*o, v| o.* = bf16_round.value(v);
    const out_ref = try a.alloc(f32, T * f.hidden);
    try ref.refine(f64, a, f, tiny.front, enc_f, out_ref);
    const out_32 = try a.alloc(f32, T * f.hidden);
    try ref.refine(f32, a, f, tiny.front, enc_f, out_32);
    failed = !try whole(a, "condition_proj, refiner and final norm on the GPU's encoder rows", out, out_32, out_ref, f.hidden) or failed;

    // a second run, byte for byte the same
    const again = try a.alloc(u16, enc.len);
    try runner.encode(a, g, &res, ids, pos, null, again);
    const out_again = try a.alloc(u16, out.len);
    try runner.refine(f, &res, again, out_again);
    const same = std.mem.eql(u16, enc, again) and std.mem.eql(u16, out, out_again);
    std.debug.print("encoded in {d:.3} s; again, byte for byte the same: {}\n", .{ enc_s, same });
    failed = failed or !same;

    // an edit's rows: a 6-pad vision block spliced at the marks, DeepStack's blocks scattered after
    // layers 0 to 2 — the full run, then each part alone so a failure names splice, scatter or layer growth
    failed = !try withPicture(runner, &res, a, rand, &tiny) or failed;
    std.debug.print("{s}\n", .{if (failed) "FAIL" else "PASS every gate"});
    return @intFromBool(failed);
}

/// Preserve the failing full vision case, then separate no-op writes, layer growth, splice and DeepStack.
fn withPicture(runner: *hp.Runner, res: *const hp.Resident, a: std.mem.Allocator, rand: std.Random, tiny: *const Tiny) !bool {
    const g = tiny_text;
    const H = g.hidden;
    const n = 6;
    var blocks: [4][]f32 = undefined;
    const bits = try a.alloc(u16, 4 * n * H);
    for (&blocks, 0..) |*v, j| {
        v.* = try a.alloc(f32, n * H);
        for (v.*, bits[j * n * H ..][0 .. n * H]) |*x, *b| {
            b.* = bf16_round.bits(2 * rand.float(f32) - 1);
            x.* = bf16_round.value(b.*);
        }
    }
    // the presentation the pads land in: 62 text rows, the vision block's start mark, 6 pads, its end
    const T = 62 + 1 + n + 1;
    const ids = try a.alloc(u32, T);
    for (ids, 0..) |*id, i| id.* = if (i < 62) @intCast(i % 80) else if (i < 62 + n) 91 else 92;
    const pads = try a.alloc(u32, n);
    for (pads, 0..) |*p, i| p.* = @intCast(63 + i);
    const pos = try a.alloc([3]i32, T);
    for (pos, 0..) |*p, i| p.* = @splat(@intCast(i));
    const cs = try a.alloc(f32, T * 64);
    const sn = try a.alloc(f32, T * 64);
    rules.rotary(cs, sn, pos, g.theta, g.sections);
    const embed = try bitsOf(tiny, "model.language_model.embed_tokens.weight");

    // Preserve embedding rows for no-op splices; zero DeepStack blocks separate the two vision operations.
    const zeros = try a.alloc(u16, n * H);
    @memset(zeros, 0);
    const zeros_f = try a.alloc(f32, n * H);
    @memset(zeros_f, 0);
    const keep = try a.alloc(u16, n * H);
    const keep_f = try a.alloc(f32, n * H);
    for (0..n) |i| {
        const row = ids[pads[i]];
        @memcpy(std.mem.sliceAsBytes(keep[i * H ..][0..H]), embed[row * H * 2 ..][0 .. H * 2]);
        @memcpy(keep_f[i * H ..][0..H], tiny.host.embed[row * H ..][0..H]);
    }
    const vision: hp.Vision = .{ .pads = pads, .rows = bits[0 .. n * H], .deepstack = .{ bits[n * H ..][0 .. n * H], bits[2 * n * H ..][0 .. n * H], bits[3 * n * H ..][0 .. n * H] } };
    const oracle: ref.Vision = .{ .pads = pads, .rows = blocks[0], .deepstack = blocks[1..4].* };

    // Equal inputs must retain the same arithmetic with the vision copy/add path enabled.
    const plain = try a.alloc(u16, T * H);
    try runner.encode(a, g, res, ids, pos, null, plain);
    const noop = try a.alloc(u16, T * H);
    const noop_vision: hp.Vision = .{ .pads = pads, .rows = keep, .deepstack = .{ zeros, zeros, zeros } };
    try runner.encode(a, g, res, ids, pos, noop_vision, noop);
    const noop_same = std.mem.eql(u16, plain, noop);
    std.debug.print("a no-op vision (embedding rows spliced back, DeepStack zero) byte for byte the text-only run: {}\n", .{noop_same});

    // one layer alone: the probes below separate its growth from splice and DeepStack
    var first_layer = g;
    first_layer.layers = 1;

    const Probe = struct { label: []const u8, geometry: rules.Text, gpu: hp.Vision, host: ref.Vision };
    const probes = [_]Probe{
        .{ .label = "text encoder rows, a reference's vision and DeepStack rows at equal-axis positions", .geometry = g, .gpu = vision, .host = oracle },
        .{ .label = "text encoder rows after layer 0 only, vision and DeepStack", .geometry = first_layer, .gpu = vision, .host = oracle },
        .{
            .label = "text encoder rows, vision spliced and DeepStack zero",
            .geometry = g,
            .gpu = .{ .pads = pads, .rows = vision.rows, .deepstack = .{ zeros, zeros, zeros } },
            .host = .{ .pads = pads, .rows = oracle.rows, .deepstack = .{ zeros_f, zeros_f, zeros_f } },
        },
        .{
            .label = "text encoder rows, DeepStack on embedding-row pads",
            .geometry = g,
            .gpu = .{ .pads = pads, .rows = keep, .deepstack = vision.deepstack },
            .host = .{ .pads = pads, .rows = keep_f, .deepstack = oracle.deepstack },
        },
    };
    const enc = try a.alloc(u16, T * H);
    const exact = try a.alloc(f32, T * H);
    const sum32 = try a.alloc(f32, T * H);
    var ok = noop_same;
    for (probes) |probe| {
        try runner.encode(a, probe.geometry, res, ids, pos, probe.gpu, enc);
        try ref.encode(f64, a, probe.geometry, tiny.host, ids, cs, sn, probe.host, exact);
        try ref.encode(f32, a, probe.geometry, tiny.host, ids, cs, sn, probe.host, sum32);
        ok = try whole(a, probe.label, enc, sum32, exact, H) and ok;
    }
    return ok;
}

/// The selftest's weight callbacks' context: the tiny checkpoint's stored bf16 bytes by name.
const Ctx = struct { tiny: *const Tiny, a: std.mem.Allocator, g: rules.Text };

fn bitsLayer(ctx: *anyopaque, li: usize) anyerror!hp.LayerSet {
    const c: *Ctx = @ptrCast(@alignCast(ctx));
    var set: hp.LayerSet = undefined;
    inline for (comptime std.meta.fieldNames(hp.LayerSet), rules.layerTensors(c.g)) |field, spec| {
        const name = try std.fmt.allocPrint(c.a, "model.language_model.layers.{d}.{s}", .{ li, spec[1] });
        @field(set, field) = try bitsOf(c.tiny, name);
    }
    return set;
}

fn bitsBlock(ctx: *anyopaque, bi: usize) anyerror!hp.BlockSet {
    const c: *Ctx = @ptrCast(@alignCast(ctx));
    var set: hp.BlockSet = undefined;
    inline for (comptime std.meta.fieldNames(hp.BlockSet), rules.blockTensors(tiny_front)) |field, spec| {
        const name = try std.fmt.allocPrint(c.a, "token_refiner.blocks.{d}.{s}", .{ bi, spec[1] });
        @field(set, field) = try bitsOf(c.tiny, name);
    }
    return set;
}

test "the CUDA selftest exposes its checkpoint embedding to the encoder" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var prng = std.Random.DefaultPrng.init(0x5133);
    const rand = prng.random();
    const scratch = try files.Scratch.init(io, a, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer scratch.deinit(io);
    const tiny = try Tiny.write(io, a, rand, scratch.path);
    const g = tiny_text;

    // the written text shard: the embedding the GPU path feeds, read back and checked byte for byte
    var shards = try Shards.open(io, a, a, tiny.text_dir);
    defer shards.deinit(a);
    const embed = try shards.tensorBytes("model.language_model.embed_tokens.weight", .{ g.vocab, g.hidden });
    try std.testing.expectEqual(2 * g.vocab * g.hidden, embed.len);
    try std.testing.expectEqualSlices(u8, embed, try bitsOf(&tiny, "model.language_model.embed_tokens.weight"));
    try std.testing.expectError(error.MissingTensor, bitsOf(&tiny, "model.language_model.embed_tokens.weight.missing"));
}
