//! MiniMax-H3-Image's transformer's host side around the CUDA blocks: one module so the picture
//! geometry, the schedule and the row-kernel ports share their `geometry.zig` types with the tool.
pub const picture = @import("picture.zig");
pub const schedule = @import("schedule.zig");
pub const geometry = @import("geometry.zig");
pub const rows = @import("rows_reference.zig");
