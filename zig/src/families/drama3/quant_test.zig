//! tf-drama-quant on a synthetic FP8 checkpoint: Drama 3's 40 layers, 9 experts each at small shapes, in two shards.
const std = @import("std");
const core = @import("core");
const quant = @import("quant.zig");
const format = @import("metal/quant_format.zig");

const st = core.safetensors;
const shard_edit = core.shard_edit;
const nvfp4 = core.quant.nvfp4;
const fp8 = core.quant.fp8;
const Allocator = std.mem.Allocator;
const testing = std.testing;
const io = testing.io;

const layers = 40;
const experts = 9;
const hidden = 48; // three NVFP4 groups
const inner = 32;
const shards = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
const commit = "0123456789abcdef0123456789abcdef01234567";

fn fill(rnd: std.Random, arena: Allocator, n: usize, fp8_codes: bool) ![]u8 {
    const b = try arena.alloc(u8, n);
    for (b) |*x| x.* = if (fp8_codes) rnd.uintLessThan(u8, 0x7F) | (rnd.int(u8) & 0x80) else rnd.int(u8);
    return b;
}

fn scaleInv(rnd: std.Random, arena: Allocator, n: usize, poison: bool) ![]u8 {
    const v = try arena.alloc(f32, n);
    for (v) |*x| x.* = if (poison) std.math.nan(f32) else @floatCast(1e-4 + rnd.float(f64) * 1e-2);
    return std.mem.sliceAsBytes(v);
}

/// How `Fixture.init` breaks layer 30's last expert: a NaN scale_inv, or no rows or columns.
const Fault = enum { none, nan_scale, no_rows, no_cols };

const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    root: []const u8,

    /// SRC: layers 0-19 in the first shard and 20-39 in the second, each a BF16 norm then its FP8 experts (expert 0's
    /// gate with an input_scale); an index, config.json, a nested tokenizer file, two safetensors files the index does
    /// not name and SRC's own provenance.
    /// `fault` breaks layer 30's last expert in the second shard.
    fn init(fault: Fault) !Fixture {
        var f: Fixture = .{ .arena_state = .init(testing.allocator), .tmp = testing.tmpDir(.{}), .root = undefined };
        errdefer f.deinit();
        const arena = f.arena_state.allocator();
        f.root = try f.tmp.dir.realPathFileAlloc(io, ".", arena);
        try f.tmp.dir.createDirPath(io, "src/tokenizer");
        var prng = std.Random.DefaultPrng.init(7);
        const rnd = prng.random();
        var tensors: [2]std.ArrayList(shard_edit.New) = .{ .empty, .empty };
        for (0..layers) |l| {
            const s = &tensors[l / 20];
            const pre = try std.fmt.allocPrint(arena, "model.language_model.layers.{d}.", .{l});
            try s.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}input_layernorm.weight", .{pre}), .dtype = .bf16, .shape = &.{hidden}, .bytes = try fill(rnd, arena, 2 * hidden, false) });
            for (0..experts) |e| for (quant.projections) |p| {
                const broken = l == 30 and e == experts - 1;
                const n: usize = if (broken and fault == .no_rows) 0 else if (p[0] == 'd') hidden else inner;
                const k: usize = if (broken and fault == .no_cols) 0 else if (p[0] == 'd') inner else hidden;
                const name = try std.fmt.allocPrint(arena, "{s}mlp.experts.{d}.{s}.weight", .{ pre, e, p });
                try s.append(arena, .{ .name = name, .dtype = .f8_e4m3, .shape = try arena.dupe(usize, &.{ n, k }), .bytes = try fill(rnd, arena, n * k, true) });
                const blocks = @intFromBool(n > 0) * @intFromBool(k > 0);
                try s.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}_scale_inv", .{name}), .dtype = .f32, .shape = try arena.dupe(usize, &.{ @intFromBool(n > 0), @intFromBool(k > 0) }), .bytes = try scaleInv(rnd, arena, blocks, broken and fault == .nan_scale) });
                if (e == 0 and p[0] == 'g') try s.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}mlp.experts.0.gate_proj.input_scale", .{pre}), .dtype = .f32, .shape = &.{}, .bytes = try scaleInv(rnd, arena, 1, false) });
            };
        }
        try tensors[1].append(arena, .{ .name = "lm_head.weight", .dtype = .bf16, .shape = &.{ 4, hidden }, .bytes = try fill(rnd, arena, 8 * hidden, false) });
        var index: std.ArrayList(u8) = .empty;
        try index.appendSlice(arena, "{\"metadata\": {\"total_size\": 1}, \"weight_map\": {");
        for (tensors, shards, 0..) |t, file, i| {
            try shard_edit.create(arena, try f.path(try std.fs.path.join(arena, &.{ "src", file })), &.{.{ "format", "pt" }}, t.items);
            for (t.items, 0..) |x, j| try index.print(arena, "{s}\"{s}\": \"{s}\"", .{ if (i + j == 0) "" else ", ", x.name, file });
        }
        try index.appendSlice(arena, "}}\n");
        try f.tmp.dir.writeFile(io, .{ .sub_path = "src/" ++ quant.index_name, .data = index.items });
        try f.tmp.dir.writeFile(io, .{ .sub_path = "src/config.json", .data = "{\"model_type\": \"fish_omni\"}\n" });
        try f.tmp.dir.writeFile(io, .{ .sub_path = "src/tokenizer/tokenizer.json", .data = "{\"tokens\": []}\n" });
        try f.tmp.dir.writeFile(io, .{ .sub_path = "src/codec.safetensors", .data = "a codec, not a shard\n" });
        try f.tmp.dir.writeFile(io, .{ .sub_path = "src/tokenizer/vocoder.safetensors", .data = "nor this\n" });
        try f.tmp.dir.writeFile(io, .{ .sub_path = "src/" ++ quant.provenance_name, .data = "{\"producer\": \"modelopt\"}\n" });
        return f;
    }

    fn deinit(f: *Fixture) void {
        f.tmp.cleanup();
        f.arena_state.deinit();
    }

    fn path(f: *Fixture, sub: []const u8) ![]const u8 {
        return std.fs.path.join(f.arena_state.allocator(), &.{ f.root, sub });
    }

    fn read(f: *Fixture, sub: []const u8) ![]u8 {
        return f.tmp.dir.readFileAlloc(io, sub, f.arena_state.allocator(), .limited(1 << 24));
    }

    fn json(f: *Fixture, sub: []const u8) !std.json.Value {
        return std.json.parseFromSliceLeaky(std.json.Value, f.arena_state.allocator(), try f.read(sub), .{});
    }

    fn open(f: *Fixture, dir: []const u8, shard: usize) !st.File {
        return st.File.open(testing.allocator, io, try f.path(try std.fs.path.join(f.arena_state.allocator(), &.{ dir, shards[shard] })));
    }

    fn convert(f: *Fixture, dst: []const u8, o: quant.Options) !quant.Summary {
        return quant.convert(testing.allocator, io, try f.path("src"), try f.path(dst), o);
    }

    fn exists(f: *Fixture, sub: []const u8) bool {
        f.tmp.dir.access(io, sub, .{}) catch return false;
        return true;
    }

    /// A refused run: neither DST nor its staging directory is left behind.
    fn refused(f: *Fixture, dst: []const u8, want: anyerror, o: quant.Options) !void {
        try testing.expectError(want, f.convert(dst, o));
        try testing.expect(!f.exists(dst));
        try testing.expect(!f.exists(try std.fmt.allocPrint(f.arena_state.allocator(), "{s}.partial", .{dst})));
    }

    /// A --compact-experts file of `text`, by path.
    fn map(f: *Fixture, name: []const u8, text: []const u8) ![]const u8 {
        try f.tmp.dir.writeFile(io, .{ .sub_path = name, .data = text });
        return f.path(name);
    }
};

/// A shard's tensor names in data order.
fn ordered(arena: Allocator, file: *const st.File) ![]const []const u8 {
    const names = try arena.dupe([]const u8, file.names.keys());
    std.mem.sort([]const u8, names, file, struct {
        fn lt(f: *const st.File, a: []const u8, b: []const u8) bool {
            return f.names.get(a).?.begin < f.names.get(b).?.begin;
        }
    }.lt);
    return names;
}

fn isExpert(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".mlp.experts.") != null;
}

/// `name`'s NVFP4 triple in `b` against the encoder run on its FP8 source in `a`, and the loader's shape check.
fn expectEncoded(arena: Allocator, a: *const st.File, b: *const st.File, name: []const u8) !void {
    const w = a.get(name).?;
    const n, const k = .{ w.shape[0], w.shape[1] };
    const x = try arena.alloc(f64, n * k);
    try fp8.reference(.{ .weight = w, .scale_inv = a.get(try std.fmt.allocPrint(arena, "{s}_scale_inv", .{name})).? }, x);
    const codes = try arena.alloc(u8, n * k / 2);
    const scales = try arena.alloc(u8, n * k / nvfp4.group);
    const sum = nvfp4.encode(x, n, k, codes, scales);
    const got = b.get(name).?;
    const s = b.get(try std.fmt.allocPrint(arena, "{s}_scale", .{name})).?;
    const g = b.get(try std.fmt.allocPrint(arena, "{s}_scale_2", .{name})).?;
    try format.nvfp4(got, s, g, n, k);
    try testing.expectEqualSlices(u8, codes, got.bytes);
    try testing.expectEqualSlices(u8, scales, s.bytes);
    try testing.expectEqualSlices(u8, std.mem.asBytes(&sum.global), g.bytes);
}

/// DST's shards hold SRC's tensors in SRC's order: other tensors with their bytes, each held expert weight as its
/// NVFP4 triple (or, kept, with its scale_inv and input_scale as they were), and no tensor of an unheld expert.
fn expectShards(f: *Fixture, dst: []const u8, held: ?*const format.Held, nvfp4_experts: bool) !void {
    const arena = f.arena_state.allocator();
    for (0..shards.len) |i| {
        var a = try f.open("src", i);
        defer a.close(io);
        var b = try f.open(dst, i);
        defer b.close(io);
        var want: std.ArrayList([]const u8) = .empty;
        for (try ordered(arena, &a)) |name| {
            if (!isExpert(name)) {
                const x, const y = .{ a.get(name).?, b.get(name).? };
                try testing.expect(y.is(x.dtype, x.shape[0..x.rank]));
                try testing.expectEqualSlices(u8, x.bytes, y.bytes);
                try want.append(arena, name);
                continue;
            }
            var it = std.mem.splitScalar(u8, name["model.language_model.layers.".len..], '.');
            const l = try std.fmt.parseInt(u8, it.next().?, 10);
            _ = it.next();
            _ = it.next();
            const e = try std.fmt.parseInt(u8, it.next().?, 10);
            if (held) |h| if (std.mem.indexOfScalar(u8, h.ids[l][0..h.counts[l]], e) == null) continue;
            if (!nvfp4_experts) {
                try testing.expectEqualSlices(u8, a.get(name).?.bytes, b.get(name).?.bytes);
                try want.append(arena, name);
            } else if (std.mem.endsWith(u8, name, ".weight")) {
                try expectEncoded(arena, &a, &b, name);
                try want.appendSlice(arena, &.{ name, try std.fmt.allocPrint(arena, "{s}_scale", .{name}), try std.fmt.allocPrint(arena, "{s}_scale_2", .{name}) });
            }
        }
        const got = try ordered(arena, &b);
        try testing.expectEqual(want.items.len, got.len);
        for (want.items, got) |x, y| try testing.expectEqualStrings(x, y);
    }
}

/// DST's index maps exactly its shards' tensors.
fn expectIndex(f: *Fixture, dst: []const u8) !std.json.Value {
    const arena = f.arena_state.allocator();
    const index = try f.json(try std.fs.path.join(arena, &.{ dst, quant.index_name }));
    const map = index.object.get("weight_map").?.object;
    var count: usize = 0;
    for (0..shards.len) |i| {
        var b = try f.open(dst, i);
        defer b.close(io);
        for (b.names.keys()) |name| try testing.expectEqualStrings(shards[i], map.get(name).?.string);
        count += b.names.count();
    }
    try testing.expectEqual(count, map.count());
    return index;
}

test "an FP8 checkpoint converts: NVFP4 experts, every other tensor and file unchanged, provenance, SRC untouched" {
    var f = try Fixture.init(.none);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const before = try f.read("src/" ++ shards[1]);
    const s = try f.convert("dst", .{ .commit = commit, .threads = 3 });
    try testing.expectEqual(2, s.shards);
    try testing.expectEqual(layers * experts * 3, s.converted);
    try testing.expectEqual(4, s.copied);
    try testing.expect(!f.exists("dst.partial"));
    try expectShards(&f, "dst", null, true);
    const index = try expectIndex(&f, "dst");
    try testing.expect(index.object.get("metadata").?.object.get(format.experts_key) == null);
    for ([_][]const u8{ "config.json", "tokenizer/tokenizer.json", "codec.safetensors", "tokenizer/vocoder.safetensors" }) |name|
        try testing.expectEqualStrings(try f.read(try std.fs.path.join(arena, &.{ "src", name })), try f.read(try std.fs.path.join(arena, &.{ "dst", name })));
    try testing.expectEqualSlices(u8, before, try f.read("src/" ++ shards[1]));

    const p = try f.json("dst/" ++ quant.provenance_name);
    try testing.expectEqualStrings(commit, p.object.get("tool_commit").?.string);
    try testing.expectEqualStrings(try f.path("src"), p.object.get("source").?.string);
    try testing.expectEqualStrings("nvfp4", p.object.get("experts").?.string);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(try f.read("src/config.json"), &digest, .{});
    try testing.expectEqualStrings(&std.fmt.bytesToHex(digest, .lower), p.object.get("config_sha256").?.string);
    try testing.expectEqualStrings(try f.read("src/" ++ quant.provenance_name), p.object.get("source_provenance").?.object.get("text").?.string);
    try testing.expect(p.object.get("compact_experts").? == .null);
    const tensors = p.object.get("tensors").?.array.items;
    try testing.expectEqual(layers * experts * 3, tensors.len);
    try testing.expect(tensors[0].object.get("rel").?.float < 0.2);

    // the same run, one thread: the same bytes
    _ = try f.convert("one", .{ .commit = commit });
    for (shards) |shard| try testing.expectEqualSlices(u8, try f.read(try std.fs.path.join(arena, &.{ "dst", shard })), try f.read(try std.fs.path.join(arena, &.{ "one", shard })));
    // DST exists: refused before staging
    try testing.expectError(error.DestinationExists, f.convert("dst", .{}));
    try testing.expect(!f.exists("dst.partial"));
    try testing.expectError(error.DestinationInSource, f.convert("src/dst", .{}));
}

/// Layer l holds every expert but l % 9, as the loader's map reads.
fn dropOne(arena: Allocator) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    try text.append(arena, '{');
    for (0..layers) |l| {
        try text.print(arena, "{s}\"{d}\": [", .{ if (l == 0) "" else ", ", l });
        var first = true;
        for (0..experts) |e| if (e != l % experts) {
            try text.print(arena, "{s}{d}", .{ if (first) "" else ", ", e });
            first = false;
        };
        try text.append(arena, ']');
    }
    try text.append(arena, '}');
    return text.items;
}

test "--compact-experts writes the held experts alone and a map the loader reads back" {
    var f = try Fixture.init(.none);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const text = try dropOne(arena);
    const path = try f.map("held.json", text);
    const held = try format.parseHeld(try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}));
    for ([_]bool{ true, false }) |encode| {
        const dst = if (encode) "nvfp4" else "keep";
        const s = try f.convert(dst, .{ .commit = commit, .compact_experts = path, .nvfp4 = encode });
        try testing.expectEqual(@as(usize, if (encode) layers * (experts - 1) * 3 else 0), s.converted);
        try expectShards(&f, dst, &held, encode);
        const index = try expectIndex(&f, dst);
        const got = try format.heldMetadata(index.object.get("metadata"));
        try testing.expectEqualDeep(held, got);
        var names: std.StringHashMapUnmanaged(usize) = .empty;
        for (index.object.get("weight_map").?.object.keys()) |name| try names.put(arena, name, 0);
        try format.checkNames(&got, names.keyIterator());
        const p = try f.json(try std.fs.path.join(arena, &.{ dst, quant.provenance_name }));
        const c = p.object.get("compact_experts").?.object;
        try testing.expectEqual(layers * (experts - 1), c.get("held").?.integer);
        try testing.expectEqualStrings("held.json", c.get("file").?.string);
    }
    // a compact checkpoint is not compacted again
    try testing.expectError(error.AlreadyCompact, quant.convert(testing.allocator, io, try f.path("keep"), try f.path("again"), .{ .compact_experts = path }));
    try testing.expect(!f.exists("again") and !f.exists("again.partial"));
}

test "--compact-experts refuses a malformed map and leaves nothing behind" {
    var f = try Fixture.init(.none);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const good = try dropOne(arena);
    const cut = std.mem.lastIndexOf(u8, good, ", \"39\"").?;
    const cases = [_]struct { []const u8, anyerror }{
        .{ "not json", error.BadExpertMap },
        .{ "[]", error.BadExpertMap },
        .{ try std.fmt.allocPrint(arena, "{s}}}", .{good[0..cut]}), error.BadExpertMap }, // layer 39 missing
        .{ try std.mem.replaceOwned(u8, arena, good, "\"5\": [0, 1, 2, 3, 4, 6, 7, 8]", "\"5\": [0, 1, 2, 3, 4, 7, 6, 8]"), error.BadExpertMap }, // not ascending
        .{ try std.mem.replaceOwned(u8, arena, good, "\"5\": [0, 1, 2, 3, 4, 6, 7, 8]", "\"5\": [0, 1, 2, 3, 4, 4, 7, 8]"), error.BadExpertMap }, // twice
        .{ try std.mem.replaceOwned(u8, arena, good, "\"5\": [0, 1, 2, 3, 4, 6, 7, 8]", "\"5\": [0, 1, 2, 3, 4, 6, 7]"), error.BadExpertMap }, // under top_k
        .{ try std.mem.replaceOwned(u8, arena, good, "\"5\": [0, 1, 2, 3, 4, 6, 7, 8]", "\"5\": [0, 1, 2, 3, 4, 6, 7, 256]"), error.BadExpertMap }, // past 255
        .{ try std.mem.replaceOwned(u8, arena, good, "\"5\": [0, 1, 2, 3, 4, 6, 7, 8]", "\"5\": [0, 1, 2, 3, 4, 6, 7, 9]"), error.CompactExpertNotInSource },
    };
    for (cases, 0..) |c, i| try f.refused("dst", c[1], .{ .compact_experts = try f.map(try std.fmt.allocPrint(arena, "bad{d}.json", .{i}), c[0]) });
}

test "a run that fails part way leaves no DST behind, an empty expert as a bad scale" {
    for ([_]struct { Fault, anyerror }{ .{ .nan_scale, error.BadSourceScale }, .{ .no_rows, error.ExpectedFp8Expert }, .{ .no_cols, error.ExpectedFp8Expert } }) |c| {
        var f = try Fixture.init(c[0]);
        defer f.deinit();
        try f.refused("dst", c[1], .{ .threads = 2 });
    }
}

test "SRC's index and shards must agree, and its metadata comes through" {
    var f = try Fixture.init(.none);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const index = try f.read("src/" ++ quant.index_name);
    const scale = "\"model.language_model.layers.3.mlp.experts.2.up_proj.weight_scale_inv\": \"" ++ shards[0] ++ "\", ";
    const edits = [_]struct { []const u8, []const u8, anyerror }{
        .{ scale, "", error.UnindexedTensor }, // a shard tensor the index leaves out
        .{ "\"weight_map\": {", "\"weight_map\": {\"model.language_model.layers.0.extra.weight\": \"" ++ shards[0] ++ "\", ", error.IndexedTensorMissing }, // an indexed tensor no shard holds
        .{ "\"weight_map\": {", "\"weight_map\": {\"model.language_model.layers.0.mlp.experts.18446744073709551615.up_proj.weight\": \"" ++ shards[0] ++ "\", ", error.UnexpectedExpert }, // past Drama 3's 256 experts
    };
    for (edits) |e| {
        try testing.expect(std.mem.indexOf(u8, index, e[0]) != null);
        try f.tmp.dir.writeFile(io, .{ .sub_path = "src/" ++ quant.index_name, .data = try std.mem.replaceOwned(u8, arena, index, e[0], e[1]) });
        try f.refused("dst", e[2], .{});
    }
    try f.tmp.dir.writeFile(io, .{ .sub_path = "src/" ++ quant.index_name, .data = try std.mem.replaceOwned(u8, arena, index, "\"total_size\": 1", "\"total_size\": 1, \"drama_head_rows\": [0]") });
    _ = try f.convert("dst", .{});
    const meta = (try f.json("dst/" ++ quant.index_name)).object.get("metadata").?;
    try testing.expectError(error.UnsupportedCompactHead, format.heldMetadata(meta)); // as the loader refuses SRC
}
