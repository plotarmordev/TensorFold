//! An uncompressed .npz (numpy.savez): a stored ZIP of .npy members, read from memory.
const std = @import("std");
const npy = @import("core").npy;

pub const Npz = struct {
    bytes: []const u8,
    directory: []const u8,
    count: u16,

    pub fn open(bytes: []const u8) !Npz {
        if (bytes.len < 22) return error.NotZip;
        const end = std.mem.lastIndexOf(u8, bytes, "PK\x05\x06") orelse return error.NotZip;
        if (bytes.len - end < 22) return error.NotZip;
        const count = std.mem.readInt(u16, bytes[end + 10 ..][0..2], .little);
        const size = std.mem.readInt(u32, bytes[end + 12 ..][0..4], .little);
        const at = std.mem.readInt(u32, bytes[end + 16 ..][0..4], .little);
        if (@as(u64, at) + size > end) return error.BadDirectory;
        return .{ .bytes = bytes, .directory = bytes[at..][0..size], .count = count };
    }

    /// The member `name` + ".npy"; numpy writes zip64 placeholders locally, so sizes come from the directory.
    pub fn array(z: Npz, name: []const u8) !npy.Array {
        var at: usize = 0;
        for (0..z.count) |_| {
            if (z.directory.len - at < 46 or !std.mem.eql(u8, z.directory[at..][0..4], "PK\x01\x02")) return error.BadDirectory;
            const e = z.directory[at..];
            const method = std.mem.readInt(u16, e[10..12], .little);
            const size = std.mem.readInt(u32, e[24..28], .little);
            const name_len = std.mem.readInt(u16, e[28..30], .little);
            const extra_len = std.mem.readInt(u16, e[30..32], .little);
            const comment_len = std.mem.readInt(u16, e[32..34], .little);
            const local = std.mem.readInt(u32, e[42..46], .little);
            if (e.len - 46 < name_len) return error.BadDirectory;
            const member = e[46..][0..name_len];
            at += 46 + @as(usize, name_len) + extra_len + comment_len;
            if (!(member.len == name.len + 4 and std.mem.startsWith(u8, member, name) and std.mem.endsWith(u8, member, ".npy"))) continue;
            if (method != 0) return error.CompressedMember;
            if (z.bytes.len < @as(u64, local) + 30) return error.BadMember;
            const h = z.bytes[local..];
            const data = 30 + @as(usize, std.mem.readInt(u16, h[26..28], .little)) + std.mem.readInt(u16, h[28..30], .little);
            if (h.len < @as(u64, data) + size) return error.BadMember;
            return npy.parse(h[data..][0..size]);
        }
        return error.MissingTensor;
    }
};

test "npz: stored members by name, missing and compressed ones refused" {
    const a = std.testing.allocator;
    const dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }";
    const pad: [0x76 - dict.len - 1]u8 = @splat(' ');
    const payload = "\x93NUMPY\x01\x00\x76\x00" ++ dict ++ pad ++ "\n" ++ "\x00\x00\x80\x3f";
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const name = "x.npy";
    var local: [30]u8 = @splat(0);
    @memcpy(local[0..4], "PK\x03\x04");
    std.mem.writeInt(u16, local[26..28], name.len, .little);
    try out.appendSlice(a, &local);
    try out.appendSlice(a, name);
    try out.appendSlice(a, payload);
    const dir_at = out.items.len;
    var central: [46]u8 = @splat(0);
    @memcpy(central[0..4], "PK\x01\x02");
    std.mem.writeInt(u32, central[24..28], payload.len, .little);
    std.mem.writeInt(u16, central[28..30], name.len, .little);
    try out.appendSlice(a, &central);
    try out.appendSlice(a, name);
    var end: [22]u8 = @splat(0);
    @memcpy(end[0..4], "PK\x05\x06");
    std.mem.writeInt(u16, end[10..12], 1, .little);
    std.mem.writeInt(u32, end[12..16], @intCast(out.items.len - dir_at), .little);
    std.mem.writeInt(u32, end[16..20], @intCast(dir_at), .little);
    try out.appendSlice(a, &end);

    const z = try Npz.open(out.items);
    const x = try z.array("x");
    try std.testing.expectEqualStrings("<f4", x.descr);
    try std.testing.expectEqual(@as(f32, 1), std.mem.bytesToValue(f32, x.data[0..4]));
    try std.testing.expectError(error.MissingTensor, z.array("y"));
    out.items[dir_at + 10] = 8;
    try std.testing.expectError(error.CompressedMember, (try Npz.open(out.items)).array("x"));
}
