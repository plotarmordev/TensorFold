//! Woof over a speech bench corpus (JSON lines with "id" and "text"): one take per item at SEED in VOICE on THREADS, timings, OUT/<id>.wav.
const std = @import("std");
const woof = @import("woof");

const Clock = std.Io.Clock;

const First = struct {
    io: std.Io,
    start: Clock.Timestamp,
    seconds: ?f64 = null,

    fn piece(ctx: *anyopaque, _: usize) bool {
        const f: *First = @ptrCast(@alignCast(ctx));
        if (f.seconds == null) f.seconds = seconds(f.start.untilNow(f.io));
        return true;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = init.minimal.args.iterate();
    _ = args.next();
    const dir = args.next() orelse return usage();
    const corpus = args.next() orelse return usage();
    const out_path = args.next() orelse return usage();
    const seed = try std.fmt.parseInt(u64, args.next() orelse "0", 10);
    const voice_name = args.next() orelse "juno";
    const threads = try std.fmt.parseInt(usize, args.next() orelse "0", 10);
    const load_start = Clock.Timestamp.now(io, .awake);
    const m = try woof.Model.load(gpa, io, dir, null, threads);
    defer m.deinit();
    const voice: u32 = for (m.voice_names, 0..) |v, i| {
        if (std.mem.eql(u8, v, voice_name)) break @intCast(i);
    } else return usage();
    const load_s = seconds(load_start.untilNow(io));
    const lines = try std.Io.Dir.cwd().readFileAlloc(io, corpus, gpa, .limited(1 << 20));
    defer gpa.free(lines);
    try std.Io.Dir.cwd().createDirPath(io, out_path);
    var out = try std.Io.Dir.cwd().openDir(io, out_path, .{});
    defer out.close(io);
    const pcm = try gpa.alloc(f32, woof.max_frames * woof.frame_length);
    defer gpa.free(pcm);
    var total_s: f64 = 0;
    var audio_s: f64 = 0;
    var firsts: std.ArrayList(f64) = .empty;
    defer firsts.deinit(gpa);
    const cpu_start = cpuSeconds();
    var it = std.mem.tokenizeScalar(u8, lines, '\n');
    while (it.next()) |line| {
        const item = try std.json.parseFromSlice(struct { id: []const u8, text: []const u8 }, gpa, line, .{ .ignore_unknown_fields = true });
        defer item.deinit();
        const cps = try codepoints(gpa, item.value.text);
        defer gpa.free(cps);
        var first: First = .{ .io = io, .start = .now(io, .awake) };
        const n = (try m.speak(cps, voice, seed, pcm, .{ .ctx = &first, .piece = First.piece })).samples;
        const took = seconds(first.start.untilNow(io));
        const audio = @as(f64, @floatFromInt(n)) / woof.sample_rate;
        total_s += took;
        audio_s += audio;
        try writeWav(gpa, io, out, item.value.id, pcm[0..n]);
        std.debug.print("{s}: first audio {d:.3} s, total {d:.3} s, audio {d:.2} s\n", .{ item.value.id, first.seconds.?, took, audio });
        try firsts.append(gpa, first.seconds.?);
    }
    const busy = (cpuSeconds() - cpu_start) / total_s;
    var usage_: std.c.rusage = undefined;
    _ = std.c.getrusage(std.c.rusage.SELF, &usage_);
    // ru_maxrss is bytes on macOS, KiB on Linux
    const rss_mib = @as(f64, @floatFromInt(usage_.maxrss)) / if (@import("builtin").os.tag == .macos) 1 << 20 else 1 << 10;
    std.mem.sort(f64, firsts.items, {}, std.sort.asc(f64));
    std.debug.print("load {d:.3} s, first audio p50 {d:.3} s p90 {d:.3} s, real-time factor {d:.4}, peak RSS {d:.1} MiB, threads {d}, cores busy {d:.2}\n", .{
        load_s, quantile(firsts.items, 0.5), quantile(firsts.items, 0.9), total_s / audio_s, rss_mib, m.pool.size(), busy,
    });
}

/// Linear between order statistics, as Python's statistics.quantiles(method="inclusive").
fn quantile(sorted: []const f64, q: f64) f64 {
    const at = q * @as(f64, @floatFromInt(sorted.len - 1));
    const lo: usize = @intFromFloat(@floor(at));
    const hi = @min(lo + 1, sorted.len - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - @floor(at));
}

fn cpuSeconds() f64 {
    var u: std.c.rusage = undefined;
    _ = std.c.getrusage(std.c.rusage.SELF, &u);
    const us = (u.utime.sec + u.stime.sec) * 1_000_000 + u.utime.usec + u.stime.usec;
    return @as(f64, @floatFromInt(us)) / 1e6;
}

fn seconds(d: Clock.Duration) f64 {
    return @as(f64, @floatFromInt(d.raw.nanoseconds)) / 1e9;
}

fn writeWav(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, id: []const u8, pcm: []const f32) !void {
    const name = try std.fmt.allocPrint(gpa, "{s}.wav", .{id});
    defer gpa.free(name);
    const f = try dir.createFile(io, name, .{});
    defer f.close(io);
    var buf: [1 << 16]u8 = undefined;
    var w = f.writer(io, &buf);
    const data: u32 = @intCast(pcm.len * 2);
    try w.interface.writeAll("RIFF");
    try w.interface.writeInt(u32, 36 + data, .little);
    try w.interface.writeAll("WAVEfmt ");
    for ([_]u32{ 16, 1 | (1 << 16), woof.sample_rate, woof.sample_rate * 2, 2 | (16 << 16) }) |v| try w.interface.writeInt(u32, v, .little);
    try w.interface.writeAll("data");
    try w.interface.writeInt(u32, data, .little);
    // as Woof's write_wav: clip, scale by 32767, round (ties away from zero; numpy's go to even, off by one at most)
    for (pcm) |x| try w.interface.writeInt(i16, @intFromFloat(@round(std.math.clamp(x, -1, 1) * 32767)), .little);
    try w.interface.flush();
}

fn codepoints(gpa: std.mem.Allocator, s: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = (try std.unicode.Utf8View.init(s)).iterator();
    while (it.nextCodepoint()) |c| try out.append(gpa, c);
    return out.toOwnedSlice(gpa);
}

fn usage() error{Usage} {
    std.debug.print("usage: tf-woof-bench WOOF_DIR CORPUS OUT [SEED] [VOICE] [THREADS (0: performance cores)]\n", .{});
    return error.Usage;
}
