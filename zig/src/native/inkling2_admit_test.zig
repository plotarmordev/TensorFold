//! Inkling 2 Mini's demand on the host: the 48 visits' state and what four RAM classes leave beside the weights.
const std = @import("std");
const st = @import("../families/inkling2/metal/state.zig");
const dspark = @import("../families/inkling2/metal/dspark.zig");
const admit = @import("../families/inkling2/metal/admit.zig");
const native = @import("inkling2.zig");

const GiB: u64 = 1 << 30;
/// The release's weights file and the two scratch sets: what a Mini serve holds whatever the stream count.
const weights_bytes: u64 = 19_526_019_168;

fn fixed() u64 {
    return weights_bytes + st.scratchBytes(st.batch_rows) + st.scratchBytes(st.prompt_rows);
}

/// A Mac's room for state with no INKLING2_LOAD_LIMIT_GB: 75% of RAM, less the weights, scratch and the 2 GiB margin.
fn roomOf(ram_gib: u64) u64 {
    return native.cacheBudget(ram_gib * GiB, ram_gib * GiB / 4 * 3, null, fixed()) catch unreachable;
}

test "Mini's state: a slot per visit, 40 sliding rings that stay put and 8 global caches that grow with the window" {
    const short = admit.demandAt(fixed(), 8192, false);
    const long = admit.demandAt(fixed(), 131072, false);
    try std.testing.expectEqual(st.cacheBytes(8192), short.stream);
    // a global visit holds one key and one value row a position; a sliding visit's ring and the rest do not grow
    try std.testing.expectEqual(@as(u64, 8) * (131072 - 8192) * 2 * st.kv_width * 2, long.stream - short.stream);
    try std.testing.expectEqual(@as(u64, 0), long.shared);
    try std.testing.expectEqual(short.fixed + short.stream * 3, short.of(3));
}

test "DSpark's lane and score buffers join the demand at the window" {
    const plain = admit.demandAt(fixed(), 8192, false);
    const drafted = admit.demandAt(fixed(), 8192, true);
    try std.testing.expectEqual(dspark.Drafter.streamBytes(8192), drafted.stream - plain.stream);
    try std.testing.expectEqual(dspark.Drafter.scoreBytes(8192) - dspark.Drafter.scoreBytes(0), drafted.shared);
    try std.testing.expectEqual(drafted.fixed + drafted.shared + 2 * drafted.stream, drafted.of(2));
}

test "larger RAM classes leave increasing room for Mini streams after fixed prompt scratch" {
    try std.testing.expectEqual(@as(u64, 0), roomOf(16));
    try std.testing.expectEqual(@as(u64, 0), roomOf(24));
    try std.testing.expect(roomOf(36) > 4 * GiB);
    try std.testing.expect(roomOf(48) > 12 * GiB);
    try std.testing.expect(roomOf(36) < roomOf(48));
    try std.testing.expect(roomOf(36) >= admit.demandAt(fixed(), 8192, false).stream);
    try std.testing.expect(roomOf(48) >= 2 * admit.demandAt(fixed(), 131072, false).stream);
}

test "a drafted serve's demand counts the lane of every stream and the scores once" {
    const d = admit.demandAt(fixed(), 8192, true);
    const n: u64 = 4;
    try std.testing.expectEqual(fixed() + d.shared + n * (st.cacheBytes(8192) + dspark.Drafter.streamBytes(8192)), d.of(n));
}
