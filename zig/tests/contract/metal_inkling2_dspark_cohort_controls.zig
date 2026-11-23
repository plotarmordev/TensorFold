//! Count-checked mutations of Mini DSpark's five cohort kernels: each breaks one stream's or one row's indexing.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    qkv_anchor,
    step_token,
    pick_nan_row,
    logits_row,
    pack_prefix,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .qkv_anchor => "control: metal inkling2 dspark cohort_qkv tf_ink2_dspark_cohort_qkv first anchor",
            .step_token => "control: metal inkling2 dspark cohort_step tf_ink2_dspark_cohort_step first stream's token",
            .pick_nan_row => "control: metal inkling2 dspark cohort_pick tf_ink2_dspark_cohort_pick first row's NaN",
            .logits_row => "control: metal inkling2 dspark cohort_logits tf_ink2_dspark_cohort_logits depth row",
            .pack_prefix => "control: metal inkling2 dspark cohort_pack tf_ink2_dspark_cohort_pack prefix",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .qkv_anchor => .{ .from = "anchors[row / 8] + row % 8, work, d.x);", .to = "anchors[0] + row % 8, work, d.x);" },
            .step_token => .{ .from = "lambda + s * DSP_RANK, held[s * 8 + a.row], i);", .to = "lambda + s * DSP_RANK, held[a.row], i);" },
            .pick_nan_row => .{ .from = "if (t == 0) held[s * 8 + a.row + 1] = isnan(float(row[0])) ? 0u : at;", .to = "if (t == 0) held[s * 8 + a.row + 1] = isnan(float(logits[0])) ? 0u : at;" },
            .logits_row => .{ .from = "const device bfloat* rows = base + size_t(a.row) * 200058u;", .to = "const device bfloat* rows = base;" },
            .pack_prefix => .{ .from = "for (uint j = 0; j < tg.y; ++j) row += a.active[j];", .to = "for (uint j = 0; j < tg.y; ++j) row += a.active[0];" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.inkling2_dspark);
    const change = m.edit();
    if (std.mem.count(u8, src.inkling2_dspark, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.inkling2_dspark, change.from, change.to);
}

test "each cohort control edits exactly its planted line" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 dspark cohort_"));
    }
    inline for (.{ "qkv", "step", "pick", "logits", "pack" }) |kernel|
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.inkling2_dspark, "tf_ink2_dspark_cohort_" ++ kernel ++ "("));
}
