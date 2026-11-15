//! tf-cuda-limits DIR [IMAGE...]: kernel-contract-limits without tf-cuda-test's contracts, so it builds and runs
//! wherever nvcc does, with no GPU; IMAGE names the records' fatbins to audit (none: every record).
const std = @import("std");
const fatbins = @import("contract_limits_fatbins.zig").fatbins;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-cuda-limits FATBIN_DIR [IMAGE...]\n", .{});
        return 2;
    }
    const images = try init.arena.allocator().alloc([]const u8, args.len - 2);
    for (images, args[2..]) |*image, arg| image.* = arg;
    fatbins(init.gpa, init.io, args[1], images) catch return 1;
    return 0;
}
