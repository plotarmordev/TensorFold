//! tf-drama-quant: Drama 3's FP8 routed experts to NVFP4, or kept, optionally compacted to a held set, on the CPU.
const std = @import("std");
const builtin = @import("builtin");
const quant = @import("drama3_quant");

const usage = "usage: tf-drama-quant SRC_CHECKPOINT DST_DIR [--experts nvfp4|keep] [--compact-experts MAP]\n";

pub fn main(init: std.process.Init) !u8 {
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var paths: [2][]const u8 = undefined;
    var given: usize = 0;
    var options: quant.Options = .{
        .commit = std.mem.trimEnd(u8, @embedFile("tool_commit"), "\n"),
        .threads = std.Thread.getCpuCount() catch 1,
    };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--experts")) {
            i += 1;
            if (i == args.len) return fail();
            options.nvfp4 = if (std.mem.eql(u8, args[i], "nvfp4")) true else if (std.mem.eql(u8, args[i], "keep")) false else return fail();
        } else if (std.mem.eql(u8, a, "--compact-experts")) {
            i += 1;
            if (i == args.len) return fail();
            options.compact_experts = args[i];
        } else if (given < paths.len and !std.mem.startsWith(u8, a, "--")) {
            paths[given] = a;
            given += 1;
        } else return fail();
    }
    if (given != paths.len) return fail();
    const s = try quant.convert(init.gpa, init.io, paths[0], paths[1], options);
    std.debug.print("tf-drama-quant: {d} shards, {d} expert projections NVFP4, {d} files copied, {d} tensor bytes, {d} threads\n", .{ s.shards, s.converted, s.copied, s.bytes, options.threads });
    return 0;
}

fn fail() u8 {
    std.debug.print(usage, .{});
    return 2;
}
