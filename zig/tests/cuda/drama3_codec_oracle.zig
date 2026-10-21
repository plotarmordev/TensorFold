//! The frozen codec gate's native side on Drama 3's fp32 codec: the code prefixes of 1, 2, 7, 128 and 1024 frames,
//! each decoded whole from a reset, written as native-{frames}.f32 (little-endian) for comparison with the reference.
//! Input is exactly 1024 frame-major rows of ten little-endian u32 codes. Each code is first clamped to its codebook's
//! last entry, as the reference's decode clamps it; the codec itself reads 0xffffffff as -1 and takes entry 0.
//! The caller owns the GPU lock, admission, memory cap and lifetime.
const std = @import("std");
const d3 = @import("drama3");

const max_frames = 1024;
const spf = d3.codec.samples_per_frame;

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 4) {
        std.debug.print("usage: drama3-codec-oracle CODEC_PTH CODES_U32LE OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], arena, .limited(max_frames * 10 * 4 + 1));
    if (bytes.len != max_frames * 10 * 4) return error.Expected1024Frames;
    // the codec's frame rows: int32 [n, 11], codes in columns 1..10
    const frames = try arena.alloc([11]i32, max_frames);
    for (frames, 0..) |*f, t| {
        f[0] = 0;
        for (f[1..], 0..) |*code, j| {
            const v = std.mem.readInt(u32, bytes[(t * 10 + j) * 4 ..][0..4], .little);
            code.* = @intCast(@min(v, @as(u32, if (j == 0) 4095 else 1023)));
        }
    }
    const pcm = try arena.alloc(f32, max_frames * spf);
    const c = try d3.tts.CodecOnly.open(init.gpa, io, args[1]);
    defer c.close();
    std.debug.print("weight_bytes={d}\n", .{c.weightBytes()});
    const fb = try c.buffer(@sizeOf([11]i32) * max_frames);
    try c.upload(fb, std.mem.sliceAsBytes(frames));
    const ab = try c.buffer(pcm.len * @sizeOf(f32));
    for ([_]usize{ 1, 2, 7, 128, 1024 }) |n| {
        try c.reset();
        try c.sync();
        const t0 = d3.engine.nowNs(io);
        try c.decode(fb, n, ab);
        const out = pcm[0 .. n * spf];
        try c.download(ab, std.mem.sliceAsBytes(out));
        const elapsed = d3.engine.nowNs(io) - t0;
        for (out) |x| if (!std.math.isFinite(x)) return error.NonFiniteAudio;
        const path = try std.fmt.allocPrint(arena, "{s}/native-{d}.f32", .{ args[3], n });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(out) });
        std.debug.print("frames={d} samples={d} elapsed_s={d:.6}\n", .{ n, out.len, @as(f64, @floatFromInt(elapsed)) / 1e9 });
    }
    return 0;
}
