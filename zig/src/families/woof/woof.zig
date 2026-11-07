//! Underdog Woof (underdog.ai, Underdog Non-Commercial License 1.0): English text to 24 kHz speech on the CPU from
//! its int8 checkpoint, with its voice prefixes, sentence pieces and redraw of a take that stalls or runs long.
const std = @import("std");
const npy = @import("core").npy;
const Npz = @import("npz.zig").Npz;
const Loader = @import("weights.zig").Loader;
const Randn = @import("rng.zig").Randn;
const text = @import("text.zig");
const speech = @import("speech.zig");
const Speech = speech.Speech;
const Decoder = @import("decoder.zig").Decoder;

const weights_file = "woof_int8.npz";
pub const sample_rate: u32 = 24000;
pub const frame_length: u32 = 960;
/// Frames one request may write: about eight minutes.
pub const max_frames: u32 = 12000;
/// Frames one piece may take (the checkpoint's max_frames); fewer when its text fills the speech model's positions.
const piece_frames = 750;
/// Characters one piece may hold: its prefix (voice, BOS id, characters, speech BOS) must leave one position to speak.
const piece_chars = speech.max_len - speech.voice_rows - 3;
const builtin_voices = [_][]const u8{ "juno", "mira" };

/// The runtime's defaults: flow temperature, end probability, and the redraw check (seconds a character, longest
/// mid-piece silence, extra takes).
const temperature: f32 = 0.5;
const end_probability = 0.9;
const max_seconds_per_char = 0.095;
const stall_seconds = 1.0;
const redraws = 3;

pub const Progress = struct { ctx: *anyopaque, piece: *const fn (ctx: *anyopaque, samples: usize) bool };
pub const Reason = enum { stop, length, cancelled };
pub const Spoken = struct { samples: usize, reason: Reason };

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    gpa: std.mem.Allocator,
    speech: Speech,
    decoder: Decoder,
    mean: []const f32,
    std: []const f32,
    voice_names: []const []const u8,
    voices: []const []const f32,
    latents: [][speech.latent]f32,
    takes: [2][]f32,

    /// `dir`/woof_int8.npz, plus each `voices_dir`/NAME.npy (float32 (16, 448)) as voice NAME after the built-ins.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, voices_dir: ?[]const u8) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .arena = .init(gpa), .gpa = gpa, .speech = undefined, .decoder = undefined, .mean = &.{}, .std = &.{}, .voice_names = &.{}, .voices = &.{}, .latents = &.{}, .takes = undefined };
        errdefer m.arena.deinit();
        const a = m.arena.allocator();
        const path = try std.fs.path.join(gpa, &.{ dir, weights_file });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(256 << 20));
        defer gpa.free(bytes);
        var l: Loader = .{ .z = try Npz.open(bytes), .a = a, .tmp = gpa };
        m.speech = try Speech.load(&l);
        m.decoder = try Decoder.load(&l);
        m.mean = try l.floats(speech.latent, "mean", .{});
        m.std = try l.floats(speech.latent, "std", .{});
        var names: std.ArrayList([]const u8) = .empty;
        var voices: std.ArrayList([]const f32) = .empty;
        for (builtin_voices) |name| {
            try names.append(a, name);
            try voices.append(a, try l.floats(speech.voice_rows * speech.d, "voices.{s}", .{name}));
        }
        if (voices_dir) |vd| try catalog(a, io, vd, &names, &voices);
        m.voice_names = names.items;
        m.voices = voices.items;
        m.latents = try a.alloc([speech.latent]f32, piece_frames);
        for (&m.takes) |*t| t.* = try a.alloc(f32, piece_frames * frame_length);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.arena.deinit();
        m.gpa.destroy(m);
    }

    /// Speaks `input` (code points) in `voice` into pcm, a piece at a time; the seed fixes every draw.
    pub fn speak(m: *Model, input: []const u32, voice: u32, seed: u64, pcm: []f32, progress: Progress) !Spoken {
        var scratch = std.heap.ArenaAllocator.init(m.gpa);
        defer scratch.deinit();
        const a = scratch.allocator();
        const pieces = try text.split(a, input);
        var rng = Randn.init(seed);
        var at: usize = 0;
        for (if (pieces.len == 0) &[_][]const u32{input} else pieces) |piece| {
            if (piece.len > piece_chars) return error.PieceTooLong;
            const take = try m.bestTake(a, try text.encode(a, piece), piece.len, voice, &rng);
            if (take.len > pcm.len - at) return .{ .samples = at, .reason = .length };
            @memcpy(pcm[at..][0..take.len], take);
            at += take.len;
            if (!progress.piece(progress.ctx, at)) return .{ .samples = at, .reason = .cancelled };
        }
        return .{ .samples = at, .reason = .stop };
    }

    /// The first take that neither stalls nor runs long, else the best of 1 + redraws by (bad, stall, pace).
    fn bestTake(m: *Model, a: std.mem.Allocator, ids: []const u32, chars: usize, voice: u32, rng: *Randn) ![]const f32 {
        var best: ?struct { bad: u2, run: f64, pace: f64, slot: usize, len: usize } = null;
        var slot: usize = 0;
        for (0..1 + redraws) |_| {
            const n = m.speech.generate(m.voices[voice], ids, rng, temperature, end_probability, m.latents);
            for (m.latents[0..n]) |*f| for (f, m.std, m.mean) |*z, s, u| {
                z.* = z.* * s + u;
            };
            const wav = m.takes[slot][0 .. n * frame_length];
            try m.decoder.run(a, m.latents[0..n], wav);
            const run = text.stall(wav, sample_rate);
            const pace = @as(f64, @floatFromInt(wav.len)) / sample_rate / @as(f64, @floatFromInt(@max(1, chars)));
            const bad = @as(u2, @intFromBool(run >= stall_seconds)) + @intFromBool(pace > max_seconds_per_char or n >= piece_frames);
            if (best == null or less(.{ bad, run, pace }, .{ best.?.bad, best.?.run, best.?.pace })) {
                best = .{ .bad = bad, .run = run, .pace = pace, .slot = slot, .len = wav.len };
                slot = 1 - slot;
            }
            if (bad == 0) break;
        }
        return m.takes[best.?.slot][0..best.?.len];
    }
};

fn less(x: struct { u2, f64, f64 }, y: struct { u2, f64, f64 }) bool {
    if (x[0] != y[0]) return x[0] < y[0];
    if (x[1] != y[1]) return x[1] < y[1];
    return x[2] < y[2];
}

fn catalog(a: std.mem.Allocator, io: std.Io, path: []const u8, names: *std.ArrayList([]const u8), voices: *std.ArrayList([]const f32)) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var found: std.ArrayList([]const u8) = .empty;
    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".npy")) continue;
        try found.append(a, try a.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, found.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (found.items) |file| {
        const name = file[0 .. file.len - 4];
        for (names.items) |n| if (std.mem.eql(u8, n, name)) return error.DuplicateVoice;
        const arr = try npy.parse(try dir.readFileAlloc(io, file, a, .limited(1 << 20)));
        if (!std.mem.eql(u8, arr.descr, "<f4") or arr.rank != 2 or arr.shape[0] != speech.voice_rows or arr.shape[1] != speech.d) return error.BadVoice;
        const v = try a.alloc(f32, speech.voice_rows * speech.d);
        for (v, 0..) |*x, i| x.* = @bitCast(std.mem.readInt(u32, arr.data[i * 4 ..][0..4], .little));
        try names.append(a, name);
        try voices.append(a, v);
    }
}

test {
    _ = @import("npz.zig");
    _ = @import("rng.zig");
    _ = @import("text.zig");
    _ = @import("ops.zig");
}
