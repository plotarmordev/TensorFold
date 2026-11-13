//! Writing a decoded picture from a tool beside the engine: MOSS-Image's PNG writer and core's resampler, one module.
//! It owns core's picture files, so a tool that also reads pictures (the Qwen copy's `Rgb` and `patches`) takes them here.
pub const png = @import("families/moss_image/png.zig");
pub const resample = @import("core/picture/resample.zig");
pub const patches = @import("core/picture/patches.zig");
pub const Rgb = @import("core/picture/picture.zig").Rgb;
