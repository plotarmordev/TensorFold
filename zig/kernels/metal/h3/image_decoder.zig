//! MiniMax-H3-Image's image decoder kernels, compiled from source at load on any Apple GPU.
const std = @import("std");

pub const source = @embedFile("image_decoder.metal");

/// `text` (the source, or an edit of it) as the GPU compiles it: nax.h inlined over its include, and its
/// simdgroup-matrix fragments on a GPU without tensor units.
pub fn assemble(a: std.mem.Allocator, tensor_units: bool, text: []const u8) ![]u8 {
    const body = try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", @embedFile("nax.h"));
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ if (tensor_units) "" else "#define TF_SIMD_FRAGS 1\n", body });
}
