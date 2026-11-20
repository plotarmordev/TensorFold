//! MiniMax-H3-Image's image decoder kernels, compiled from source at load: tensor-unit products on the M5, main's
//! simdgroup-matrix products and separate SwiGLU on every other Apple GPU.
const std = @import("std");

pub const source = @embedFile("image_decoder.metal");

/// `text` (the source, or an edit of it) as the GPU compiles it: with tensor units, H3D_TENSOR set and nax.h inlined
/// over its include; without, as written (main's decoder).
pub fn assemble(a: std.mem.Allocator, tensor_units: bool, text: []const u8) ![]u8 {
    if (!tensor_units) return a.dupe(u8, text);
    const body = try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", @embedFile("nax.h"));
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ "#define H3D_TENSOR 1\n", body });
}
