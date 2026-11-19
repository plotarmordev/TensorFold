//! tf-h3-image-decoder-cuda: MiniMax-H3-Image's image decoder on CUDA, a latent to a PNG, or its self-test.
const std = @import("std");
const dec = @import("h3_image_decoder_cuda");
const hostdec = dec.host_decoder;
const selftest = dec.selftest;

const vae = dec.vae;

const usage =
    \\usage: tf-h3-image-decoder-cuda MODEL_DIR latent.f32 ROWS COLUMNS out.png [WIDTHxHEIGHT] [--exact | --tc]
    \\       tf-h3-image-decoder-cuda MODEL_DIR rows.f32 ROWS COLUMNS out.png [WIDTHxHEIGHT] [--exact | --tc]
    \\       tf-h3-image-decoder-cuda reference MODEL_DIR latent.f32 ROWS COLUMNS gpu.png host.png [WIDTHxHEIGHT] [--exact | --tc]
    \\       tf-h3-image-decoder-cuda MODEL_DIR --serve [--exact | --tc]
    \\       tf-h3-image-decoder-cuda selftest [--exact | --tc]
    \\  --tc          (default) products and attention on tensor cores (cuBLASLt, fp32 sums); selftest's gate
    \\  --exact       attention and products in ordered fp32; selftest keeps its 90% block-identity bar
    \\  MODEL_DIR     the MiniMax-H3-Image folder: vae/config.json and vae/diffusion_pytorch_model.safetensors
    \\  latent.f32    the normalised latent [24, ROWS, COLUMNS] in float32, as tf-h3-image writes it
    \\  rows.f32      the transformer's target rows, streamed: the decoder reads them progressively, so a
    \\                fifo the transformer still fills overlaps the last step's tail and the first tiles
    \\  WIDTHxHEIGHT  the requested size: the 16 COLUMNS x 16 ROWS canvas is resized to it (Pillow's LANCZOS)
    \\reference decodes the same latent on the GPU and on the host's reference decoder, writes both PNGs and compares
    \\the pictures against the selftest's full-chain relative bound; it exits 1 past it.
    \\--serve loads the decoder once, then decodes one `latent.f32|rows.f32 ROWS COLUMNS out.png [WIDTHxHEIGHT]` line
    \\of stdin per picture, answering each `done OUT` or `failed OUT` on stdout.
    \\selftest runs the kernels on synthetic weights against a float64 host reference; it exits 1 past a bound.
    \\
;

/// Tiles a pass decodes at most.
const batch = 16;

/// The image decoder loaded once: its config and weights, its kernels and cuBLASLt plans.
pub const Vae = struct {
    config: vae.Config,
    d: *dec.Decoder,

    pub fn load(init: std.process.Init, a: std.mem.Allocator, model_dir: []const u8, tc: bool) !Vae {
        const vae_dir = try std.fs.path.join(a, &.{ model_dir, "vae" });
        const config = try vae.config(a, try hostdec.read(init.io, a, try std.fs.path.join(a, &.{ vae_dir, "config.json" }), 1 << 20));
        const loading = dec.seconds();
        const d = try dec.Decoder.init(init.gpa, vae.layers, config.rope_theta);
        errdefer d.deinit();
        d.fast = tc;
        try d.load(init.io, a, try std.fmt.allocPrint(a, "{s}/diffusion_pytorch_model.safetensors", .{vae_dir}));
        // H3_PROFILE: wait after every stage and print where a picture's decoder seconds went
        d.profile = if (init.environ_map.get("H3_PROFILE")) |v| v.len > 0 else false;
        std.debug.print("decoder weights in bf16 in {d:.1} s\n", .{dec.seconds() - loading});
        return .{ .config = config, .d = d };
    }

    /// The normalised `latent` [24, rows, cols] decoded and written to `path` as a PNG of `want` (default the canvas).
    pub fn png(v: *const Vae, init: std.process.Init, a: std.mem.Allocator, latent: []f32, rows: usize, cols: usize, want: ?dec.host.Size, path: []const u8) !void {
        if (latent.len != try dec.host.latentLength(rows, cols)) return error.H3BadLatentLength;
        const tiles = try dec.host.tiles(a, &v.config, rows, cols);
        const width = cols * vae.cell_pixels;
        const height = rows * vae.cell_pixels;
        const size = want orelse dec.host.Size{ .width = width, .height = height };
        try vae.denormalize(&v.config, latent);
        const started = dec.seconds();
        const decoded = try a.alloc(f32, 3 * width * height);
        const took = try v.d.decode(a, &v.config, latent, rows, cols, vae.layers, batch, decoded);
        try hostdec.write(init.io, a, decoded, height, width, size.width, size.height, path);
        std.debug.print("{d}x{d} canvas: decoded {d} tiles in {d:.2} s ({d:.2} s on the GPU), {d:.2} s in all; wrote {s} at {d}x{d}\n", .{ width, height, tiles.total(), took.wall, took.gpu, dec.seconds() - started, path, size.width, size.height });
    }
};

/// The transformer's rows stream: `path` is read band by band as bytes arrive (a fifo overlaps the
/// render's last step), each band unpatchified to latent rows, denormalised and uploaded, and every
/// tile pass launched as soon as the rows its tiles read have landed.
fn pngRows(v: *const Vae, init: std.process.Init, a: std.mem.Allocator, path: []const u8, rows: usize, cols: usize, want: ?dec.host.Size, out_path: []const u8) !void {
    if (cols % 2 != 0 or rows % 2 != 0) return error.H3BadPictureGrid;
    const grid: dec.picture.Grid = .{ .rows = rows, .columns = cols };
    const width = cols * vae.cell_pixels;
    const height = rows * vae.cell_pixels;
    const size = want orelse dec.host.Size{ .width = width, .height = height };
    const t = try dec.host.tiles(a, &v.config, rows, cols);
    const d = v.d;
    const per = @min(batch, @min(dec.most_batch, t.total()));
    if (vae.layers > d.plan.layers) return error.H3BadTiles;
    d.spent = @splat(0);
    try d.prepareEmpty(rows, cols, t.h, t.w, per);

    var file = try std.Io.Dir.cwd().openFile(init.io, path, .{});
    defer file.close(init.io);
    const band_rows = 16; // latent cell rows a read delivers: 8 patch rows of the video
    const banded = try a.alloc(f32, dec.host.channels * band_rows * cols);
    const video = try a.alloc(f32, band_rows / 2 * (cols / 2) * dec.picture.row_width);
    var assembly = try vae.Assembler.init(a, t.rows, t.cols, 3, try a.alloc(f32, 3 * width * height));
    const decoded = assembly.out;
    const tile = try a.alloc(f32, 3 * t.h * t.w * vae.cell_pixels * vae.cell_pixels);
    const origins = try a.alloc([2]u32, per);
    var took: dec.Timing = .{};
    var k: usize = 0; // next tile the gate is holding
    var row: usize = 0; // latent cell rows fed so far
    const started = dec.seconds();
    while (k < t.total()) {
        // pass k needs every origin's rows; feed bands until the batch's deepest tile is covered
        const n = @min(per, t.total() - k);
        var need: usize = 0;
        for (origins[0..n], 0..) |*o, i| {
            const at = t.at(k + i);
            o.* = .{ @intCast(at.row), @intCast(at.col) };
            need = @max(need, at.row + t.h);
        }
        while (row < need) {
            const m = @min(band_rows, rows - row);
            const cells = m * cols;
            const band_video = m / 2 * (cols / 2) * dec.picture.row_width;
            try readFull(init.io, file, std.mem.sliceAsBytes(video[0..band_video]));
            try dec.picture.unpatchifyRows(video[0..band_video], grid, banded[0 .. dec.host.channels * cells]);
            try vae.denormalize(&v.config, banded[0 .. dec.host.channels * cells]);
            try d.feed(banded[0 .. dec.host.channels * cells], row, m);
            row += m;
        }
        const one = try d.pass(origins[0..n], .{ .to = vae.layers });
        took.wall += one.wall;
        took.gpu += one.gpu;
        const projected_rows = try d.projected();
        for (0..n) |i| {
            try dec.host.unpatchify(projected_rows[i * d.tokens * dec.host.patch ..][0 .. t.h * t.w * dec.host.patch], t.h, t.w, tile);
            const at = t.at(k + i);
            try assembly.place(at.r, at.c, tile);
        }
        k += n;
    }
    try d.queue.synchronize();
    try hostdec.write(init.io, a, decoded, height, width, size.width, size.height, out_path);
    std.debug.print("{d}x{d} canvas: decoded {d} tiles in {d:.2} s ({d:.2} s on the GPU), {d:.2} s in all; wrote {s} at {d}x{d}\n", .{ width, height, t.total(), took.wall, took.gpu, dec.seconds() - started, out_path, size.width, size.height });
}

/// `file`'s next `bytes.len` bytes, retrying short reads until filled or the stream ends.
fn readFull(io: std.Io, file: std.Io.File, bytes: []u8) !void {
    var got: usize = 0;
    while (got < bytes.len) {
        const n = file.readStreaming(io, &.{bytes[got..]}) catch |err| switch (err) {
            error.EndOfStream => return error.H3BadLatentLength,
            else => return err,
        };
        if (n == 0) return error.H3BadLatentLength;
        got += n;
    }
}

/// The latent file's [24, rows, cols] float32 values.
fn latentOf(init: std.process.Init, a: std.mem.Allocator, path: []const u8, rows: usize, cols: usize) ![]f32 {
    const n = try dec.host.latentLength(rows, cols);
    const bytes = try hostdec.read(init.io, a, path, n * 4 + 1);
    if (bytes.len != n * 4) {
        std.debug.print("{s} holds {d} bytes; a [24, {d}, {d}] float32 latent is {d}\n", .{ path, bytes.len, rows, cols, n * 4 });
        return error.H3BadLatentLength;
    }
    const latent = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(latent), bytes);
    return latent;
}

/// `latent` decoded on the GPU and on the host reference decoder, both written as PNGs and the pictures compared.
fn reference(init: std.process.Init, a: std.mem.Allocator, model_dir: []const u8, latent: []f32, rows: usize, cols: usize, gpu_path: []const u8, host_path: []const u8, want: ?dec.host.Size, fast: bool) !bool {
    const v = try Vae.load(init, a, model_dir, fast);
    defer v.d.deinit();
    const width = cols * vae.cell_pixels;
    const height = rows * vae.cell_pixels;
    const size = want orelse dec.host.Size{ .width = width, .height = height };
    try vae.denormalize(&v.config, latent);
    const ours = try a.alloc(f32, 3 * width * height);
    const started = dec.seconds();
    const took = try v.d.decode(a, &v.config, latent, rows, cols, vae.layers, batch, ours);
    try hostdec.write(init.io, a, ours, height, width, size.width, size.height, gpu_path);
    std.debug.print("{d}x{d} canvas: GPU decoder {d:.2} s in all ({d:.2} s on the GPU); wrote {s} at {d}x{d}\n", .{ width, height, dec.seconds() - started, took.gpu, gpu_path, size.width, size.height });
    var h = try hostdec.Decoder.init(init.gpa, vae.layers);
    defer h.deinit();
    const loading = dec.seconds();
    try h.load(init.io, a, model_dir);
    const theirs = try a.alloc(f32, ours.len);
    const host_started = dec.seconds();
    try h.decode(a, &v.config, latent, rows, cols, vae.layers, theirs);
    try hostdec.write(init.io, a, theirs, height, width, size.width, size.height, host_path);
    std.debug.print("reference decoder: weights in {d:.1} s, {d:.2} s in all; wrote {s} at {d}x{d}\n", .{ host_started - loading, dec.seconds() - host_started, host_path, size.width, size.height });
    return compare(a, ours, theirs, height, width);
}

/// The two decoded pictures [3, height, width]: the bytes' share identical and within 1, the largest difference,
/// PSNR, and the relative error against the selftest's full-chain bound.
fn compare(a: std.mem.Allocator, ours: []const f32, theirs: []const f32, height: usize, width: usize) !bool {
    var dd: f64 = 0;
    var norm: f64 = 0;
    for (ours, theirs) |x, y| {
        dd += (@as(f64, x) - y) * (@as(f64, x) - y);
        norm += @as(f64, y) * y;
    }
    const relative = @sqrt(dd / @max(norm, 1e-300));
    const ours_rgb = try a.alloc(u8, ours.len);
    const theirs_rgb = try a.alloc(u8, theirs.len);
    try vae.pixels(ours, height, width, ours_rgb);
    try vae.pixels(theirs, height, width, theirs_rgb);
    var same: usize = 0;
    var within: usize = 0;
    var most: u8 = 0;
    var squares: f64 = 0;
    for (ours_rgb, theirs_rgb) |x, y| {
        const d = if (x > y) x - y else y - x;
        same += @intFromBool(d == 0);
        within += @intFromBool(d <= 1);
        most = @max(most, d);
        squares += @as(f64, @floatFromInt(d)) * @as(f64, @floatFromInt(d));
    }
    const count: f64 = @floatFromInt(ours_rgb.len);
    const psnr = if (squares == 0) std.math.inf(f64) else 10 * @log10(255 * 255 / (squares / count));
    const ok = relative <= selftest.bounds.deep_relative;
    std.debug.print("picture: {d:.3}% bytes identical, {d:.3}% within 1, at most {d} apart, PSNR {d:.2} dB; relative error {e:.2} (bound {e:.2}) ({s})\n", .{ 100 * @as(f64, @floatFromInt(same)) / count, 100 * @as(f64, @floatFromInt(within)) / count, most, psnr, relative, selftest.bounds.deep_relative, if (ok) "pass" else "FAIL" });
    return ok;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const all = try init.minimal.args.toSlice(a);
    const kept = try a.alloc([]const u8, all.len);
    var mode: ?bool = null;
    var n_kept: usize = 0;
    for (all) |arg| {
        if (std.mem.eql(u8, arg, "--tc")) {
            mode = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--exact")) {
            mode = false;
            continue;
        }
        kept[n_kept] = arg;
        n_kept += 1;
    }
    const fast = mode orelse true;
    const args = kept[0..n_kept];
    if (args.len == 2 and (std.mem.eql(u8, args[1], "selftest") or std.mem.eql(u8, args[1], "--self-test"))) {
        if (!try selftest.run(init.gpa, a, fast)) std.process.exit(1);
        return;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "reference")) {
        if (args.len != 8 and args.len != 9) {
            std.debug.print("{s}", .{usage});
            std.process.exit(2);
        }
        const rows = std.fmt.parseInt(usize, args[4], 10) catch return error.H3BadPictureGrid;
        const cols = std.fmt.parseInt(usize, args[5], 10) catch return error.H3BadPictureGrid;
        const latent = try latentOf(init, a, args[3], rows, cols);
        const ok = try reference(init, a, args[2], latent, rows, cols, args[6], args[7], if (args.len == 9) try dec.host.size(args[8]) else null, fast);
        if (!ok) std.process.exit(1);
        return;
    }
    if (args.len != 6 and args.len != 7 and !(args.len == 3 and std.mem.eql(u8, args[2], "--serve"))) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    if (args.len == 3) {
        var v = try Vae.load(init, a, args[1], fast);
        defer v.d.deinit();
        return serve(init, &v);
    }
    const rows = std.fmt.parseInt(usize, args[3], 10) catch return error.H3BadPictureGrid;
    const cols = std.fmt.parseInt(usize, args[4], 10) catch return error.H3BadPictureGrid;
    const v = try Vae.load(init, a, args[1], fast);
    defer v.d.deinit();
    const want: ?dec.host.Size = if (args.len == 7) try dec.host.size(args[6]) else null;
    if (std.mem.endsWith(u8, args[2], ".rows.f32")) {
        try pngRows(&v, init, a, args[2], rows, cols, want, args[5]);
        return;
    }
    const latent = try latentOf(init, a, args[2], rows, cols);
    try v.png(init, a, latent, rows, cols, want, args[5]);
}

/// `--serve`'s stdin: one `latent.f32 ROWS COLUMNS out.png [WIDTHxHEIGHT]` line per picture, each answered
/// `done OUT` or `failed OUT` on stdout, the decoder's weights loaded once and kept.
fn serve(init: std.process.Init, v: *Vae) !void {
    var in_buf: [4096]u8 = undefined;
    var in = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    while (try in.interface.takeDelimiter('\n')) |line| {
        var request = std.heap.ArenaAllocator.init(init.gpa);
        defer request.deinit();
        const a = request.allocator();
        var fields: [5][]const u8 = undefined;
        var count: usize = 0;
        var it = std.mem.tokenizeAny(u8, line, " \t\r");
        while (it.next()) |field| {
            if (count == fields.len) {
                count += 1;
                break;
            }
            fields[count] = field;
            count += 1;
        }
        const verdict = blk: {
            if (count < 4 or count > fields.len) break :blk "failed";
            const rows = std.fmt.parseInt(usize, fields[1], 10) catch break :blk "failed";
            const cols = std.fmt.parseInt(usize, fields[2], 10) catch break :blk "failed";
            const want: ?dec.host.Size = if (count == 5) dec.host.size(fields[4]) catch break :blk "failed" else null;
            if (std.mem.endsWith(u8, fields[0], ".rows.f32")) {
                pngRows(v, init, a, fields[0], rows, cols, want, fields[3]) catch |err| {
                    std.debug.print("{s} failed: {s}\n", .{ fields[3], @errorName(err) });
                    break :blk "failed";
                };
            } else {
                const latent = latentOf(init, a, fields[0], rows, cols) catch break :blk "failed";
                v.png(init, a, latent, rows, cols, want, fields[3]) catch |err| {
                    std.debug.print("{s} failed: {s}\n", .{ fields[3], @errorName(err) });
                    break :blk "failed";
                };
            }
            break :blk "done";
        };
        try std.Io.File.stdout().writeStreamingAll(init.io, try std.fmt.allocPrint(a, "{s} {s}\n", .{ verdict, if (count >= 4) fields[3] else "-" }));
    }
}
