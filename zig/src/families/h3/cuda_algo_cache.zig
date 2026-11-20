//! The model's cuBLASLt picks kept on disk: the bf16 Linears' (n, k) algorithms and the INT8 products'
//! (rows, n, k) ones, under the GPU's name and driver and cuBLASLt's versions. A start whose file
//! matches skips every heuristic search; a key mismatch, a missing or a torn file is a miss — the
//! search runs as before and `save` rewrites the table.
//!
//! `TF_H3_ALGO_CACHE` steers the file: a path of its own (the bench's A/B), or `off` to skip it;
//! otherwise it lives at ~/.cache/tensorfold/h3-cublaslt-algos.bin.
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;

const magic = "TFH3LT1\n";
const key_bytes = 256;
const head_bytes = magic.len + key_bytes;
const max_path = 1024;

/// Which product a record is for: the algo bytes cuBLASLt gives are opaque, so a kind separates a
/// bf16 Linear's pick from an int8 Gemm's at the same dimensions.
pub const Kind = enum(u64) { bf16 = 0, int8 = 1 };

/// One stored pick (96 bytes): the shape it was chosen for and the 64-byte Algo the heuristic gave.
pub const Entry = extern struct {
    kind: Kind,
    shape: [3]u64,
    algo: lt_.Algo,
};

/// The loaded file: this device's identity line and the records it carried. A lookup that misses
/// searches and `put`s; `save` publishes the whole table again under the current key.
pub const Picks = struct {
    key: [key_bytes]u8,
    entries: std.ArrayList(Entry) = .empty,
    dir: [max_path]u8 = undefined,
    dir_len: usize = 0,
    file: [max_path]u8 = undefined,
    file_len: usize = 0,
    off: bool = false,

    /// The file's key: GPU name, driver and cuBLASLt versions — a pick under any other is useless.
    pub fn identity(d: *const cuda.Driver, ctx: *const cuda.Context, lt: *const lt_.Library) [key_bytes]u8 {
        var line: [key_bytes]u8 = @splat(0);
        var name_buf: [192]u8 = undefined;
        const name = ctx.name(&name_buf) catch "gpu?";
        const v = d.version() catch 0;
        const lt_v = lt.api.cublasLtGetVersion();
        var w = std.Io.Writer.fixed(&line);
        w.print("{s}\ndriver {d}\ncublaslt {d}\n", .{ name, v, lt_v }) catch unreachable;
        return line;
    }

    /// This device's stored picks, or an empty table. Anything unreadable, foreign or absent is a
    /// miss, never a failure: the searches then run and the table is rewritten.
    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, lt: *const lt_.Library) !*Picks {
        const p = try gpa.create(Picks);
        errdefer gpa.destroy(p);
        p.* = .{ .key = identity(d, ctx, lt) };
        const spec = env("TF_H3_ALGO_CACHE");
        if (spec) |s| if (std.mem.eql(u8, s, "off")) {
            p.off = true;
            return p;
        };
        var file: []const u8 = undefined;
        if (spec) |s| {
            file = s;
        } else {
            const home = env("HOME") orelse {
                p.off = true;
                return p;
            };
            const dir = std.fmt.bufPrint(&p.dir, "{s}/.cache/tensorfold", .{home}) catch {
                p.off = true;
                return p;
            };
            p.dir_len = dir.len;
            file = std.fmt.bufPrint(&p.file, "{s}/h3-cublaslt-algos.bin", .{dir}) catch {
                p.off = true;
                return p;
            };
            p.file_len = file.len;
        }
        if (spec != null) {
            if (file.len > p.file.len) {
                p.off = true;
                return p;
            }
            @memcpy(p.file[0..file.len], file);
            p.file_len = file.len;
            // a named file's own directory, for the .part sibling
            if (std.fs.path.dirname(file)) |dir| {
                if (dir.len > p.dir.len) {
                    p.off = true;
                    return p;
                }
                @memcpy(p.dir[0..dir.len], dir);
                p.dir_len = dir.len;
            }
        }
        std.debug.print("h3 cublaslt picks: file {s}{s}\n", .{ p.file[0..p.file_len], if (p.off) " (off)" else "" });
        p.readFile(gpa);
        return p;
    }

    pub fn deinit(p: *Picks, gpa: std.mem.Allocator) void {
        p.entries.deinit(gpa);
        gpa.destroy(p);
    }

    /// The stored algorithm for (`kind`, `shape`), or null.
    pub fn get(p: *const Picks, kind: Kind, shape: [3]u64) ?lt_.Algo {
        for (p.entries.items) |e| {
            if (e.kind == kind and std.mem.eql(u64, &e.shape, &shape)) return e.algo;
        }
        return null;
    }


    /// Remembers a search's pick; the next `save` publishes it.
    pub fn put(p: *Picks, gpa: std.mem.Allocator, kind: Kind, shape: [3]u64, algo: lt_.Algo) !void {
        try p.entries.append(gpa, .{ .kind = kind, .shape = shape, .algo = algo });
    }

    /// The file's bytes as written: magic, this device's key, the records.
    pub fn bytes(p: *const Picks, gpa: std.mem.Allocator) ![]u8 {
        var out = try gpa.alloc(u8, head_bytes + p.entries.items.len * @sizeOf(Entry));
        @memcpy(out[0..magic.len], magic);
        @memcpy(out[magic.len..head_bytes], &p.key);
        @memcpy(out[head_bytes..], std.mem.sliceAsBytes(p.entries.items));
        return out;
    }

    /// Writes the table whole through a sibling renamed into place: a reader sees the old file or
    /// the new, never a torn one. A foreign-keyed file is rewritten under this device's key, the
    /// cache's whole point; a failed write only costs the next start its searches again.
    pub fn save(p: *Picks, gpa: std.mem.Allocator) void {
        if (p.off or p.file_len == 0) {
            std.debug.print("h3 cublaslt picks: not saved (off={} file_len={d})\n", .{ p.off, p.file_len });
            return;
        }
        const data = p.bytes(gpa) catch return;
        defer gpa.free(data);
        saveAt(p.dir[0..p.dir_len], p.file[0..p.file_len], data);
        std.debug.print("h3 cublaslt picks: {d} records to {s}\n", .{ p.entries.items.len, p.file[0..p.file_len] });
    }

    /// Everything but the identity check and the device, for tests: `data` at `file` becomes the
    /// table only when its key is `key` — anything else is left for the search to redo.
    pub fn readBytes(p: *Picks, gpa: std.mem.Allocator, data: []const u8) void {
        if (data.len < head_bytes or !std.mem.eql(u8, data[0..magic.len], magic)) return;
        if (!std.mem.eql(u8, data[magic.len..head_bytes], &p.key)) return;
        const body = data[head_bytes..];
        if (body.len % @sizeOf(Entry) != 0) return;
        for (0..body.len / @sizeOf(Entry)) |i| {
            var e: Entry = undefined;
            @memcpy(std.mem.asBytes(&e), body[i * @sizeOf(Entry) ..][0..@sizeOf(Entry)]);
            p.entries.append(gpa, e) catch return;
        }
    }

    fn readFile(p: *Picks, gpa: std.mem.Allocator) void {
        var zfile: [max_path]u8 = undefined;
        const f = p.file[0..p.file_len];
        @memcpy(zfile[0..f.len], f);
        zfile[f.len] = 0;
        const fd = platform.c.open(zfile[0..f.len :0], .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
        if (fd < 0) return;
        defer _ = platform.c.close(fd);
        var data = gpa.alloc(u8, 4 << 20) catch return;
        defer gpa.free(data);
        var got: usize = 0;
        while (got < data.len) {
            const n = platform.c.read(fd, data.ptr + got, data.len - got);
            if (n <= 0) break;
            got += @intCast(n);
        }
        p.readBytes(gpa, data[0..got]);
    }
};

/// `data` at `file`, written to a sibling and renamed over it; `dir`'s parents are made first.
/// POSIX-only like the rows store's counters: a file this small is one write and one rename.
pub fn saveAt(dir: []const u8, file: []const u8, data: []const u8) void {
    var zdir: [max_path]u8 = undefined;
    if (dir.len > zdir.len) return;
    @memcpy(zdir[0..dir.len], dir);
    zdir[dir.len] = 0;
    makeParents(zdir[0..dir.len :0]);
    var zfile: [max_path]u8 = undefined;
    var ztmp: [max_path + 8]u8 = undefined;
    if (file.len > zfile.len) return;
    @memcpy(zfile[0..file.len], file);
    zfile[file.len] = 0;
    const tmp = std.fmt.bufPrintSentinel(&ztmp, "{s}.part", .{file}, 0) catch return;
    const fd = platform.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) {
        std.debug.print("h3 cublaslt picks: open {s}.part failed, errno {t}\n", .{ file, platform.c.errno(fd) });
        return;
    }
    defer _ = platform.c.close(fd);
    var done: usize = 0;
    while (done < data.len) {
        const n = platform.c.write(fd, data.ptr + done, data.len - done);
        if (n <= 0) {
            _ = platform.c.unlink(tmp);
            return;
        }
        done += @intCast(n);
    }
    _ = platform.c.rename(tmp, zfile[0..file.len :0]);
}

/// `dir`'s parents made 0700 the way rows_store's are; the last component is the file's own name.
fn makeParents(path: [:0]const u8) void {
    var buf: [max_path]u8 = undefined;
    if (path.len > buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    for (buf[1..path.len], 1..) |ch, i| if (ch == '/') {
        buf[i] = 0;
        _ = platform.c.mkdir(@ptrCast(&buf), 0o700);
        buf[i] = '/';
    };
}

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (platform.c.getenv(name)) |v| std.mem.span(v) else null;
}

comptime {
    std.debug.assert(@sizeOf(Entry) == 96);
}

test "h3 cublaslt picks: a stored pick answers its shape, kinds separate, and a written file reads back" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var key: [key_bytes]u8 = @splat(0);
    @memcpy(key[0..10], "a gpu\nd 1\n");
    var p: Picks = .{ .key = key };
    defer p.entries.deinit(a);
    const algo: lt_.Algo = .{ .data = .{ 1, 2, 3, 4, 5, 6, 7, 8 } };
    try p.put(a, .bf16, .{ 16384, 4096, 4096 }, algo);
    try p.put(a, .int8, .{ 16408, 12288, 4096 }, .{ .data = .{ 8, 7, 6, 5, 4, 3, 2, 1 } });
    try std.testing.expectEqual(algo, p.get(.bf16, .{ 16384, 4096, 4096 }).?);
    try std.testing.expect(p.get(.int8, .{ 16384, 4096, 4096 }) == null); // same dims, another kind
    // the bytes a save writes come back as the same table under the same key
    const data = try p.bytes(a);
    defer a.free(data);
    var q: Picks = .{ .key = key };
    defer q.entries.deinit(a);
    q.readBytes(a, data);
    try std.testing.expectEqual(@as(usize, 2), q.entries.items.len);
    try std.testing.expectEqual(algo, q.get(.bf16, .{ 16384, 4096, 4096 }).?);
    // a foreign key, a truncated file and a missing file each leave an empty table
    var other: Picks = .{ .key = @splat(7) };
    defer other.entries.deinit(a);
    other.readBytes(a, data);
    try std.testing.expectEqual(@as(usize, 0), other.entries.items.len);
    var r: Picks = .{ .key = key };
    defer r.entries.deinit(a);
    for ([_]usize{ 0, magic.len + 3, data.len - 1 }) |len| {
        r.readBytes(a, data[0..len]);
        try std.testing.expectEqual(@as(usize, 0), r.entries.items.len);
    }
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const file = try std.fs.path.join(a, &.{ dir, "picks.bin" });
    defer a.free(file);
    r.file_len = file.len;
    @memcpy(r.file[0..file.len], file);
    r.readFile(a);
    try std.testing.expectEqual(@as(usize, 0), r.entries.items.len); // nothing there yet
    saveAt(dir, file, data);
    r.readFile(a);
    try std.testing.expectEqual(@as(usize, 2), r.entries.items.len);
}
