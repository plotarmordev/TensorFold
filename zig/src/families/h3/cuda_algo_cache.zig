//! The model's cuBLASLt picks kept on disk: the bf16 Linears' (n, k) algorithms and the INT8 products'
//! (rows, n, k) ones, under the GPU's name and driver and cuBLASLt's versions. A start whose file
//! matches skips every heuristic search; a key mismatch, a missing, torn or corrupt file is a miss —
//! the search runs as before and `save` rewrites the table.
//!
//! `TF_H3_ALGO_CACHE` steers the file: a path of its own (the bench's A/B), or `off` to skip it;
//! otherwise it lives at ~/.cache/tensorfold/h3-cublaslt-algos.bin.
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;

const magic = "TFH3LT2\n";
const key_bytes = 256;
/// Magic, key and the record count: a file cut anywhere, even between records, fails the count.
const head_bytes = magic.len + key_bytes + @sizeOf(u64);
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
        const placed = if (spec) |s|
            p.namedPath(s)
        else if (env("HOME")) |home|
            p.defaultPath(home)
        else
            false;
        if (!placed) {
            p.off = true;
            return p;
        }
        std.debug.print("h3 cublaslt picks: file {s}\n", .{p.file[0..p.file_len]});
        p.readFile(gpa);
        return p;
    }

    /// The `~/.cache/tensorfold` default into `p`'s fixed buffers, or false when it does not fit
    /// with a byte to spare: `readFile`'s NUL would land one past a file path exactly `max_path`.
    fn defaultPath(p: *Picks, home: []const u8) bool {
        const dir = std.fmt.bufPrint(&p.dir, "{s}/.cache/tensorfold", .{home}) catch return false;
        p.dir_len = dir.len;
        const file = std.fmt.bufPrint(&p.file, "{s}/h3-cublaslt-algos.bin", .{dir}) catch return false;
        if (file.len >= max_path) return false;
        p.file_len = file.len;
        return true;
    }

    /// A `TF_H3_ALGO_CACHE` path into `p`'s fixed buffers, with its own directory for the staging
    /// sibling, under the same rule.
    fn namedPath(p: *Picks, file: []const u8) bool {
        if (file.len >= max_path) return false;
        @memcpy(p.file[0..file.len], file);
        p.file_len = file.len;
        if (std.fs.path.dirname(file)) |dir| {
            if (dir.len >= max_path) return false;
            @memcpy(p.dir[0..dir.len], dir);
            p.dir_len = dir.len;
        }
        return true;
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

    /// Remembers a pick, replacing the shape's old one: a search that reran after a record failed
    /// `AlgoCheck` keeps no stale bytes behind.
    pub fn put(p: *Picks, gpa: std.mem.Allocator, kind: Kind, shape: [3]u64, algo: lt_.Algo) !void {
        for (p.entries.items) |*e| if (e.kind == kind and std.mem.eql(u64, &e.shape, &shape)) {
            e.algo = algo;
            return;
        };
        try p.entries.append(gpa, .{ .kind = kind, .shape = shape, .algo = algo });
    }

    /// Drops a record the device refused (`initChecked` NoAlgorithm), so `save` writes a clean table.
    pub fn remove(p: *Picks, kind: Kind, shape: [3]u64) void {
        for (p.entries.items, 0..) |e, i| if (e.kind == kind and std.mem.eql(u64, &e.shape, &shape)) {
            _ = p.entries.swapRemove(i);
            return;
        };
    }

    /// The file's bytes as written: magic, this device's key, the record count, the records.
    pub fn bytes(p: *const Picks, gpa: std.mem.Allocator) ![]u8 {
        var out = try gpa.alloc(u8, head_bytes + p.entries.items.len * @sizeOf(Entry));
        @memcpy(out[0..magic.len], magic);
        @memcpy(out[magic.len..][0..key_bytes], &p.key);
        std.mem.writeInt(u64, out[magic.len + key_bytes ..][0..8], p.entries.items.len, .little);
        @memcpy(out[head_bytes..], std.mem.sliceAsBytes(p.entries.items));
        return out;
    }

    /// Writes the table whole through a sibling renamed into place: a reader sees the old file or
    /// the new, never a torn one. A foreign-keyed file is rewritten under this device's key, the
    /// cache's whole point; a failed write only costs the next start its searches again.
    pub fn save(p: *Picks, gpa: std.mem.Allocator) void {
        if (p.off or p.file_len == 0) return;
        const data = p.bytes(gpa) catch return;
        defer gpa.free(data);
        if (saveAt(gpa, p.dir[0..p.dir_len], p.file[0..p.file_len], data)) {
            std.debug.print("h3 cublaslt picks: {d} records to {s}\n", .{ p.entries.items.len, p.file[0..p.file_len] });
        }
    }

    /// Everything but the identity check and the device, for tests: `data` becomes the table only
    /// when its key is `key`, its count matches its length and every record's kind is known —
    /// anything else is left for the search to redo.
    pub fn readBytes(p: *Picks, gpa: std.mem.Allocator, data: []const u8) void {
        if (data.len < head_bytes or !std.mem.eql(u8, data[0..magic.len], magic)) return;
        if (!std.mem.eql(u8, data[magic.len..][0..key_bytes], &p.key)) return;
        const count = std.mem.readInt(u64, data[magic.len + key_bytes ..][0..8], .little);
        const body = data[head_bytes..];
        if (body.len % @sizeOf(Entry) != 0 or body.len / @sizeOf(Entry) != count) return;
        for (0..count) |i| {
            const tag = std.mem.readInt(u64, body[i * @sizeOf(Entry) ..][0..8], .little);
            if (std.enums.fromInt(Kind, tag) == null) return;
        }
        p.entries.ensureTotalCapacity(gpa, count) catch return;
        for (0..count) |i| {
            var e: Entry = undefined;
            @memcpy(std.mem.asBytes(&e), body[i * @sizeOf(Entry) ..][0..@sizeOf(Entry)]);
            p.entries.appendAssumeCapacity(e);
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

/// `data` at `file`, staged in a unique, exclusively created no-follow sibling and renamed over it
/// (a planted link is never written through); `dir`'s parents are made first, none for a bare
/// filename, which lands in the working directory.
pub fn saveAt(gpa: std.mem.Allocator, dir: []const u8, file: []const u8, data: []const u8) bool {
    var zdir: [max_path]u8 = undefined;
    if (dir.len >= zdir.len or file.len >= max_path) return false;
    @memcpy(zdir[0..dir.len], dir);
    zdir[dir.len] = 0;
    if (dir.len > 0) makeParents(zdir[0..dir.len :0]);
    platform.atomic_file.replacePrivate(gpa, file, &.{data}) catch {
        std.debug.print("h3 cublaslt picks: write {s} failed, {s}\n", .{ file, platform.atomic_file.causeName() });
        return false;
    };
    return true;
}

/// `dir` made 0700 the way rows_store's parents are — but here the last component is a directory.
fn makeParents(path: [:0]const u8) void {
    var buf: [max_path]u8 = undefined;
    if (path.len >= buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    for (buf[1..path.len], 1..) |ch, i| if (ch == '/') {
        buf[i] = 0;
        _ = platform.c.mkdir(@ptrCast(&buf), 0o700);
        buf[i] = '/';
    };
    _ = platform.c.mkdir(@ptrCast(&buf), 0o700);
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
    // a foreign key, a truncated file (mid-record or between records) and a missing file each
    // leave an empty table
    var other: Picks = .{ .key = @splat(7) };
    defer other.entries.deinit(a);
    other.readBytes(a, data);
    try std.testing.expectEqual(@as(usize, 0), other.entries.items.len);
    var r: Picks = .{ .key = key };
    defer r.entries.deinit(a);
    for ([_]usize{ 0, magic.len + 3, data.len - 1, data.len - @sizeOf(Entry) }) |len| {
        r.readBytes(a, data[0..len]);
        try std.testing.expectEqual(@as(usize, 0), r.entries.items.len);
    }
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const nested = try std.fs.path.join(a, &.{ dir, "tensorfold" }); // absent: saveAt makes it, the NOENT bug
    defer a.free(nested);
    const file = try std.fs.path.join(a, &.{ nested, "picks.bin" });
    defer a.free(file);
    r.file_len = file.len;
    @memcpy(r.file[0..file.len], file);
    r.readFile(a);
    try std.testing.expectEqual(@as(usize, 0), r.entries.items.len); // nothing there yet
    try std.testing.expect(saveAt(a, nested, file, data));
    r.readFile(a);
    try std.testing.expectEqual(@as(usize, 2), r.entries.items.len);
    // a record the device would refuse is dropped and the search's fresh pick replaces it
    p.remove(.bf16, .{ 16384, 4096, 4096 });
    try std.testing.expect(p.get(.bf16, .{ 16384, 4096, 4096 }) == null);
    try p.put(a, .bf16, .{ 16384, 4096, 4096 }, .{ .data = .{ 9, 9, 9, 9, 9, 9, 9, 9 } });
    try std.testing.expectEqual(@as(usize, 2), p.entries.items.len); // put replaced, not appended
    // a path at the sentinel buffers' limit is refused, never written past
    const long_dir = try a.alloc(u8, max_path);
    defer a.free(long_dir);
    @memset(long_dir, 'x');
    const long_file = try std.fs.path.join(a, &.{ long_dir, "picks.bin" });
    defer a.free(long_file);
    try std.testing.expect(!saveAt(a, long_dir, long_file, data));
}

test "h3 cublaslt picks: a link planted at the old .part sibling is never written through" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "victim", .data = "untouched" });
    try tmp.dir.symLink(io, "victim", "picks.bin.part", .{});
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const file = try std.fs.path.join(a, &.{ dir, "picks.bin" });
    defer a.free(file);
    var p: Picks = .{ .key = @splat(3) };
    defer p.entries.deinit(a);
    try p.put(a, .bf16, .{ 64, 64, 64 }, .{ .data = @splat(6) });
    const data = try p.bytes(a);
    defer a.free(data);
    try std.testing.expect(saveAt(a, dir, file, data));
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("untouched", try tmp.dir.readFile(io, "victim", &buf));
}

test "h3 cublaslt picks: a default path that fills the file buffer exactly is off, never read past" {
    const a = std.testing.allocator;
    var p: Picks = .{ .key = @splat(0) };
    defer p.entries.deinit(a);
    // a 984-byte HOME fills `file` with exactly `max_path` bytes (dir 1002 + name 22): readFile's
    // NUL would write one past; 983 leaves it the last byte, 1006 overflows the dir first
    var exact: [984]u8 = @splat('x');
    exact[0] = '/';
    try std.testing.expect(!p.defaultPath(&exact));
    var full_dir: [1006]u8 = @splat('x');
    full_dir[0] = '/';
    try std.testing.expect(!p.defaultPath(&full_dir));
    var fits: [983]u8 = @splat('x');
    fits[0] = '/';
    try std.testing.expect(p.defaultPath(&fits));
    try std.testing.expectEqual(@as(usize, max_path - 1), p.file_len);
    try std.testing.expect(std.mem.endsWith(u8, p.file[0..p.file_len], "/h3-cublaslt-algos.bin"));
    try std.testing.expect(std.mem.endsWith(u8, p.dir[0..p.dir_len], "/.cache/tensorfold"));
}

test "h3 cublaslt picks: a record with an unknown kind tag makes the whole file a miss" {
    const a = std.testing.allocator;
    var p: Picks = .{ .key = @splat(1) };
    defer p.entries.deinit(a);
    try p.put(a, .bf16, .{ 64, 64, 64 }, .{ .data = @splat(3) });
    try p.put(a, .int8, .{ 8, 64, 64 }, .{ .data = @splat(4) });
    const data = try p.bytes(a);
    defer a.free(data);
    std.mem.writeInt(u64, data[head_bytes + @sizeOf(Entry) ..][0..8], 7, .little);
    var q: Picks = .{ .key = @splat(1) };
    defer q.entries.deinit(a);
    q.readBytes(a, data);
    try std.testing.expectEqual(@as(usize, 0), q.entries.items.len);
}

test "h3 cublaslt picks: a bare TF_H3_ALGO_CACHE filename saves in the working directory" {
    const a = std.testing.allocator;
    const name = "h3-picks-bare-test.bin";
    var p: Picks = .{ .key = @splat(2) };
    defer p.entries.deinit(a);
    try std.testing.expect(p.namedPath(name));
    try std.testing.expectEqual(@as(usize, 0), p.dir_len);
    try p.put(a, .bf16, .{ 64, 64, 64 }, .{ .data = @splat(5) });
    p.save(a);
    defer _ = platform.c.unlink(name);
    var q: Picks = .{ .key = @splat(2), .file = p.file, .file_len = p.file_len };
    defer q.entries.deinit(a);
    q.readFile(a);
    try std.testing.expectEqual(@as(usize, 1), q.entries.items.len);
}
