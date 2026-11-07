//! Woof against its numpy runtime: speaks each line of TEXTS (seed SEED) and compares the samples with
//! REF/<n>.f32 from tools/zig/woof_reference.py. The first second must agree within 2e-3 and the length within 4 frames.
//! Past that the speech loop is chaotic: fp32 summation order alone (numpy against itself with its weights moved by
//! 1e-7) can change a later frame by 1 and a piece's length by a frame, so no fixed seed gives equal samples there.
//! Usage: tf-woof-parity WOOF_DIR TEXTS REF [SEED]
const std = @import("std");
const woof = @import("woof");

const head_samples = woof.sample_rate;
const max_abs = 2e-3;
const max_frames_apart = 4;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = init.minimal.args.iterate();
    _ = args.next();
    const dir = args.next() orelse return usage();
    const texts_path = args.next() orelse return usage();
    const ref_dir = args.next() orelse return usage();
    const seed = try std.fmt.parseInt(u64, args.next() orelse "0", 10);
    const m = try woof.Model.load(gpa, io, dir, null);
    defer m.deinit();
    const texts = try std.Io.Dir.cwd().readFileAlloc(io, texts_path, gpa, .limited(1 << 20));
    defer gpa.free(texts);
    const pcm = try gpa.alloc(f32, woof.max_frames * woof.frame_length);
    defer gpa.free(pcm);
    var failed = false;
    var lines = std.mem.splitScalar(u8, std.mem.trimEnd(u8, texts, "\n"), '\n');
    var n: usize = 0;
    while (lines.next()) |line| : (n += 1) {
        const cps = try codepoints(gpa, line);
        defer gpa.free(cps);
        const got = (try m.speak(cps, 0, seed, pcm, .{ .ctx = undefined, .piece = keepGoing })).samples;
        var name: [32]u8 = undefined;
        const ref_path = try std.fs.path.join(gpa, &.{ ref_dir, try std.fmt.bufPrint(&name, "{d}.f32", .{n}) });
        defer gpa.free(ref_path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, gpa, .limited(64 << 20));
        defer gpa.free(bytes);
        const want = std.mem.bytesAsSlice(f32, bytes);
        var worst: f32 = 0;
        const head = @min(got, want.len, head_samples);
        for (pcm[0..head], want[0..head]) |a, b| worst = @max(worst, @abs(a - b));
        const apart = @max(got, want.len) - @min(got, want.len);
        const ok = worst <= max_abs and apart <= max_frames_apart * woof.frame_length;
        failed = failed or !ok;
        std.debug.print("{s} {d}: {d} samples (numpy {d}), first second max |diff| {e:.2}\n", .{ if (ok) "ok" else "FAIL", n, got, want.len, worst });
    }
    if (failed) return error.ParityFailed;
}

fn keepGoing(_: *anyopaque, _: usize) bool {
    return true;
}

fn codepoints(gpa: std.mem.Allocator, s: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = (try std.unicode.Utf8View.init(s)).iterator();
    while (it.nextCodepoint()) |c| try out.append(gpa, c);
    return out.toOwnedSlice(gpa);
}

fn usage() error{Usage} {
    std.debug.print("usage: tf-woof-parity WOOF_DIR TEXTS REF [SEED]\n", .{});
    return error.Usage;
}
