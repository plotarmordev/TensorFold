//! Mini DSpark's greedy cohort launches (zig/tests/contract/metal_inkling2_dspark_cohort.zig): up to 16 streams each.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 dspark cohort_qkv tf_ink2_dspark_cohort_qkv", .threads = 128, .tg_memory = 16, .reserved = 128 },
    .{ .name = "metal inkling2 dspark cohort_step tf_ink2_dspark_cohort_step", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark cohort_pick tf_ink2_dspark_cohort_pick", .threads = 1024, .tg_memory = 256, .reserved = 1024 },
    .{ .name = "metal inkling2 dspark cohort_logits tf_ink2_dspark_cohort_logits", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark cohort_pack tf_ink2_dspark_cohort_pack", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

test "the cohort's launches reserve their threads" {
    for (records) |record| {
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 dspark cohort_"));
    }
}
