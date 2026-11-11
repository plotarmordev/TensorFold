//! tf-drama-quant: a new Drama 3 checkpoint, its FP8 routed experts NVFP4 (or kept) and optionally compacted to a held set.
const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");
const format = @import("metal/quant_format.zig");

const st = core.safetensors;
const shard_edit = core.shard_edit;
const nvfp4 = core.quant.nvfp4;
const fp8 = core.quant.fp8;
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const index_name = "model.safetensors.index.json";
pub const provenance_name = "quant-provenance.json";
pub const projections = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

pub const Options = struct {
    /// Encode the routed experts to NVFP4; false copies every expert tensor byte for byte.
    nvfp4: bool = true,
    /// A held-expert map (`metal/quant_format.zig` parseHeld): only its experts are written, under their own names.
    compact_experts: ?[]const u8 = null,
    threads: usize = 1,
    commit: []const u8 = "unknown",
};

pub const Summary = struct { shards: usize, converted: usize, copied: usize, bytes: u64 };

/// A routed expert projection's tensor: `model.language_model.layers.L.mlp.experts.E.PROJ.FIELD`.
const Name = struct { layer: usize, expert: usize, field: []const u8 };

fn parseName(name: []const u8) ?Name {
    const prefix = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    var it = std.mem.splitScalar(u8, name[prefix.len..], '.');
    const layer = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "mlp") or !std.mem.eql(u8, it.next() orelse return null, "experts")) return null;
    const expert = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    const proj = it.next() orelse return null;
    for (projections) |p| if (std.mem.eql(u8, proj, p)) return .{ .layer = layer, .expert = expert, .field = it.rest() };
    return null;
}

/// A converted projection's tensor: its weight, its consumed scale_inv, or an activation input_scale NVFP4 drops.
const Role = enum { weight, scale_inv, dropped };

fn role(field: []const u8) !Role {
    if (std.mem.eql(u8, field, "weight")) return .weight;
    if (std.mem.eql(u8, field, "weight_scale_inv")) return .scale_inv;
    if (std.mem.eql(u8, field, "input_scale")) return .dropped;
    if (std.mem.eql(u8, field, "weight_scale") or std.mem.eql(u8, field, "weight_scale_2")) return error.ExpectedFp8Expert; // NVFP4 already
    return error.UnknownExpertCompanion;
}

const Failure = error{ NonFiniteWeight, BadSourceScale, BadGlobal };

/// One projection to encode and the slices its NVFP4 codes and scales land in.
const Job = struct {
    name: []const u8,
    weight: st.Tensor,
    scale_inv: st.Tensor,
    codes: []u8 = &.{},
    scales: []u8 = &.{},
    summary: nvfp4.Summary = .{},
    failure: ?Failure = null,

    fn size(j: Job) usize {
        return j.weight.shape[0] * j.weight.shape[1];
    }

    /// The FP8 source in f64 as fp8.reference decodes it, then NVFP4.
    fn encode(j: *Job, buf: []f64) Failure!void {
        for (0..j.scale_inv.numel()) |i| {
            const s: f32 = @bitCast(std.mem.readInt(u32, j.scale_inv.bytes[i * 4 ..][0..4], .little));
            if (!std.math.isFinite(s) or s <= 0) return error.BadSourceScale;
        }
        const w = buf[0..j.size()];
        fp8.reference(.{ .weight = j.weight, .scale_inv = j.scale_inv }, w) catch unreachable; // shapes checked on admission
        for (w) |x| if (!std.math.isFinite(x)) return error.NonFiniteWeight;
        j.summary = nvfp4.encode(w, j.weight.shape[0], j.weight.shape[1], j.codes, j.scales);
        const g = j.summary.global;
        if (!(std.math.isFinite(g) and g > 0)) return error.BadGlobal; // amax / 2688 left f32
    }
};

/// Encodes `jobs` on `threads` threads; each job's bytes depend on its inputs alone, so any count writes the same.
fn encodeAll(gpa: Allocator, jobs: []Job, threads: usize) !void {
    var most: usize = 0;
    for (jobs) |j| most = @max(most, j.size());
    const t = @max(1, @min(threads, jobs.len));
    const bufs = try gpa.alloc(f64, t * most);
    defer gpa.free(bufs);
    const Work = struct {
        jobs: []Job,
        next: std.atomic.Value(usize) = .init(0),

        fn run(work: *@This(), buf: []f64) void {
            while (true) {
                const i = work.next.fetchAdd(1, .monotonic);
                if (i >= work.jobs.len) return;
                work.jobs[i].encode(buf) catch |e| {
                    work.jobs[i].failure = e;
                };
            }
        }
    };
    var work: Work = .{ .jobs = jobs };
    const pool = try gpa.alloc(std.Thread, t - 1);
    defer gpa.free(pool);
    {
        var spawned: usize = 0;
        defer for (pool[0..spawned]) |th| th.join();
        for (pool, 1..) |*th, i| {
            th.* = try std.Thread.spawn(.{}, Work.run, .{ &work, bufs[i * most ..][0..most] });
            spawned += 1;
        }
        work.run(bufs[0..most]);
    }
    for (jobs) |j| if (j.failure) |e| return e; // the first in tensor order, whichever thread met it
}

/// A source tensor copied as it is (its index in the shard's header), or an encode job's NVFP4 triple.
const Item = union(enum) { keep: usize, job: usize };

/// SRC shard `src` rewritten at `out` in its tensor order: SRC's own bytes, or an NVFP4 triple in an expert weight's place.
/// Each of its tensors must be listed in SRC's index under `shard`; `listed` counts them.
fn convertShard(gpa: Allocator, arena: Allocator, io: Io, src: []const u8, out: []const u8, source: *const std.json.ObjectMap, held: ?*const format.Held, o: Options, records: *std.ArrayList(Provenance.Tensor), weight_map: *std.json.ObjectMap, shard: []const u8, listed: *usize) !u64 {
    var file = try st.File.open(gpa, io, src);
    defer file.close(io);
    listed.* += file.names.count();
    var scratch_state = std.heap.ArenaAllocator.init(gpa);
    defer scratch_state.deinit();
    const scratch = scratch_state.allocator();
    const raw = file.map.memory;
    const header = try std.json.parseFromSliceLeaky(std.json.Value, scratch, raw[8..file.data], .{});
    var meta: std.ArrayList([2][]const u8) = .empty;
    if (header.object.get("__metadata__")) |m| {
        if (m != .object) return error.BadSafetensors;
        var it = m.object.iterator();
        while (it.next()) |kv| try meta.append(scratch, .{ kv.key_ptr.*, if (kv.value_ptr.* == .string) kv.value_ptr.string else return error.BadSafetensors });
    }
    const keys = file.names.keys();
    const entries = file.names.values();
    const order = try scratch.alloc(usize, entries.len);
    for (order, 0..) |*x, i| x.* = i;
    std.mem.sort(usize, order, @as([]const st.Entry, entries), struct {
        fn lt(e: []const st.Entry, a: usize, b: usize) bool {
            return e[a].begin < e[b].begin;
        }
    }.lt);
    var items: std.ArrayList(Item) = .empty;
    var jobs: std.ArrayList(Job) = .empty;
    var encoded: usize = 0;
    for (order) |i| {
        const name = keys[i];
        const indexed = source.get(name) orelse return error.UnindexedTensor;
        if (!std.mem.eql(u8, indexed.string, shard)) return error.UnindexedTensor;
        const x = parseName(name) orelse {
            try items.append(scratch, .{ .keep = i });
            continue;
        };
        if (held) |h| if (!holds(h, x.layer, x.expert)) continue; // every tensor of a dropped expert
        if (!o.nvfp4) {
            try items.append(scratch, .{ .keep = i });
            continue;
        }
        switch (try role(x.field)) {
            .weight => {},
            .scale_inv => {
                if (!file.names.contains(name[0 .. name.len - "_scale_inv".len])) return error.SplitExpert;
                continue;
            },
            .dropped => continue,
        }
        const weight = file.get(name).?;
        const scale_inv = file.get(try std.fmt.allocPrint(scratch, "{s}_scale_inv", .{name})) orelse return error.SplitExpert;
        if (weight.dtype != .f8_e4m3 or weight.rank != 2 or weight.shape[0] == 0 or weight.shape[1] == 0 or weight.shape[1] % nvfp4.group != 0) return error.ExpectedFp8Expert;
        const n, const k = .{ weight.shape[0], weight.shape[1] };
        if (!scale_inv.is(.f32, &.{ (n + fp8.block - 1) / fp8.block, (k + fp8.block - 1) / fp8.block })) return error.ExpectedFp8Expert;
        try items.append(scratch, .{ .job = jobs.items.len });
        try jobs.append(scratch, .{ .name = try arena.dupe(u8, name), .weight = weight, .scale_inv = scale_inv });
        encoded += n * k / 2 + n * k / nvfp4.group;
    }
    const buffer = try gpa.alloc(u8, encoded);
    defer gpa.free(buffer);
    var used: usize = 0;
    for (jobs.items) |*j| {
        const nk = j.size();
        j.codes = buffer[used..][0 .. nk / 2];
        j.scales = buffer[used + nk / 2 ..][0 .. nk / nvfp4.group];
        used += nk / 2 + nk / nvfp4.group;
    }
    try encodeAll(gpa, jobs.items, o.threads);
    var tensors: std.ArrayList(shard_edit.New) = .empty;
    for (items.items) |item| switch (item) {
        .keep => |i| {
            const e = &entries[i];
            try tensors.append(scratch, .{ .name = keys[i], .dtype = e.dtype, .shape = e.shape[0..e.rank], .bytes = raw[file.data + e.begin .. file.data + e.end] });
        },
        .job => |ji| {
            const j = &jobs.items[ji];
            const n, const k = .{ j.weight.shape[0], j.weight.shape[1] };
            try tensors.appendSlice(scratch, &.{
                .{ .name = j.name, .dtype = .u8, .shape = try scratch.dupe(usize, &.{ n, k / 2 }), .bytes = j.codes },
                .{ .name = try std.fmt.allocPrint(scratch, "{s}_scale", .{j.name}), .dtype = .f8_e4m3, .shape = try scratch.dupe(usize, &.{ n, k / nvfp4.group }), .bytes = j.scales },
                .{ .name = try std.fmt.allocPrint(scratch, "{s}_scale_2", .{j.name}), .dtype = .f32, .shape = &.{}, .bytes = std.mem.asBytes(&j.summary.global) },
            });
        },
    };
    try shard_edit.create(gpa, out, meta.items, tensors.items);
    var bytes: u64 = 0;
    for (tensors.items) |t| {
        try weight_map.put(arena, try arena.dupe(u8, t.name), .{ .string = shard });
        bytes += t.bytes.len;
    }
    for (jobs.items) |j| {
        const rel = j.summary.rel();
        try records.append(arena, .{ .name = j.name, .amax = j.summary.amax, .global = j.summary.global, .rel = if (rel) |v| .{ .float = v } else .{ .string = "undefined" } });
    }
    return bytes;
}

/// SRC rewritten at DST (new, outside SRC): experts NVFP4 or kept, compacted to a held set or not, all else as it was.
pub fn convert(gpa: Allocator, io: Io, src: []const u8, dst: []const u8, o: Options) !Summary {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cwd = Io.Dir.cwd();
    const from = try cwd.realPathFileAlloc(io, src, arena);
    const parent = try cwd.realPathFileAlloc(io, std.fs.path.dirname(dst) orelse ".", arena);
    const to = try std.fs.path.join(arena, &.{ parent, std.fs.path.basename(dst) });
    if (std.mem.startsWith(u8, to, from) and (to.len == from.len or to[from.len] == '/')) return error.DestinationInSource;
    if (cwd.access(io, to, .{ .follow_symlinks = false })) return error.DestinationExists else |e| if (e != error.FileNotFound) return e; // the rename below guards a race

    // the index and the map are checked before DST is staged; shapes and scales fail later, under the errdefer
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, try cwd.readFileAlloc(io, try join(arena, from, index_name), arena, .limited(st.index.max_bytes)), .{});
    const map = try st.index.plainWeightMap(index);
    const source_meta = index.object.get("metadata") orelse std.json.Value{ .object = .empty };
    if (source_meta != .object) return error.BadIndex;
    var shards: std.StringArrayHashMapUnmanaged(void) = .empty;
    var experts: [cfg.layers]usize = @splat(0);
    for (map.keys(), map.values()) |name, shard| {
        try shards.put(arena, shard.string, {});
        const x = parseName(name) orelse continue;
        if (x.layer >= experts.len or x.expert >= cfg.experts) return error.UnexpectedExpert;
        experts[x.layer] = @max(experts[x.layer], x.expert + 1);
        if (o.nvfp4) _ = try role(x.field);
    }
    const held: ?format.Held = if (o.compact_experts) |path| try heldExperts(arena, io, path, map, experts) else null;
    if (held != null and source_meta.object.contains(format.experts_key)) return error.AlreadyCompact;
    const shard_names = try arena.dupe([]const u8, shards.keys());
    std.mem.sort([]const u8, shard_names, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    const others = try ancillary(gpa, arena, io, from, &shards);
    var record: Provenance = .{
        .source = src,
        .tool_commit = o.commit,
        .config_sha256 = try fileSha256(arena, io, try join(arena, from, "config.json")),
        .experts = if (o.nvfp4) "nvfp4" else "keep",
        .compact_experts = if (o.compact_experts) |path| .{
            .file = std.fs.path.basename(path),
            .sha256 = try fileSha256(arena, io, path),
            .held = held.?.total(),
        } else null,
        .source_provenance = try sourceProvenance(arena, io, from),
        .tensors = &.{},
    };

    // written into DST.partial, which becomes DST only once complete: a failed run leaves no DST behind
    const partial = try std.fmt.allocPrint(arena, "{s}.partial", .{to});
    cwd.createDir(io, partial, .default_dir) catch |e| return if (e == error.PathAlreadyExists) error.PartialExists else e;
    errdefer cwd.deleteTree(io, partial) catch {};
    var out_dir = try cwd.openDir(io, partial, .{});
    defer out_dir.close(io);
    var weight_map: std.json.ObjectMap = .empty;
    var tensors: std.ArrayList(Provenance.Tensor) = .empty;
    var bytes: u64 = 0;
    var listed: usize = 0;
    for (shard_names) |shard|
        bytes += try convertShard(gpa, arena, io, try join(arena, from, shard), try join(arena, partial, shard), &map, if (held) |*h| h else null, o, &tensors, &weight_map, shard, &listed);
    if (listed != map.count()) return error.IndexedTensorMissing; // every shard tensor is indexed there, so the rest are absent
    var meta = try source_meta.object.clone(arena); // SRC's own keys hold on
    try meta.put(arena, "total_size", .{ .integer = @intCast(bytes) });
    if (held) |*h| try meta.put(arena, format.experts_key, try heldJson(arena, h));
    var root: std.json.ObjectMap = .empty;
    try root.put(arena, "metadata", .{ .object = meta });
    try root.put(arena, "weight_map", .{ .object = weight_map });
    try emit(arena, io, out_dir, index_name, std.json.Value{ .object = root });
    var src_dir = try cwd.openDir(io, from, .{});
    defer src_dir.close(io);
    for (others.dirs) |d| try out_dir.createDir(io, d, .default_dir);
    for (others.files) |f| try src_dir.copyFile(f, out_dir, f, io, .{ .replace = false });
    record.tensors = tensors.items;
    try emit(arena, io, out_dir, provenance_name, record);
    cwd.renamePreserve(partial, cwd, to, io) catch |e| return if (e == error.PathAlreadyExists) error.DestinationExists else e;
    return .{ .shards = shard_names.len, .converted = tensors.items.len, .copied = others.files.len, .bytes = bytes };
}

/// quant-provenance.json: what this run did, and SRC's own record verbatim.
pub const Provenance = struct {
    schema_version: u32 = 1,
    producer: []const u8 = "tf-drama-quant",
    source: []const u8,
    tool_commit: []const u8,
    config_sha256: []const u8,
    /// nvfp4: every routed expert DST holds was encoded here, a lossy repack of SRC's FP8; keep: copied byte for byte.
    experts: []const u8,
    /// null without --compact-experts; DST's index maps the ids each layer holds (`drama_experts`).
    compact_experts: ?struct { file: []const u8, sha256: []const u8, held: usize },
    /// SRC's quant-provenance.json; null when SRC has none.
    source_provenance: ?Record,
    rel: []const u8 = "||W - Q|| / ||W|| against SRC's FP8 decoded in f64; \"undefined\" when only its denominator is 0",
    tensors: []Tensor,

    pub const Record = struct { sha256: []const u8, text: []const u8 };
    pub const Tensor = struct { name: []const u8, amax: f64, global: f32, rel: std.json.Value };
};

fn holds(h: *const format.Held, layer: usize, expert: usize) bool {
    return std.mem.indexOfScalar(u8, h.ids[layer][0..h.counts[layer]], std.math.cast(u8, expert) orelse return false) != null;
}

/// --compact-experts' map as the loader reads it, refused unless SRC's index lists each held projection's weight and scale_inv.
fn heldExperts(arena: Allocator, io: Io, path: []const u8, map: std.json.ObjectMap, experts: [cfg.layers]usize) !format.Held {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 20));
    const value = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch return error.BadExpertMap;
    const h = try format.parseHeld(value);
    for (0..cfg.layers) |l| for (h.ids[l][0..h.counts[l]]) |e| {
        if (e >= experts[l]) return error.CompactExpertNotInSource;
        for (projections) |p| for ([_][]const u8{ "", "_scale_inv" }) |suffix| if (!map.contains(try std.fmt.allocPrint(arena, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.weight{s}", .{ l, e, p, suffix })))
            return error.CompactExpertNotInSource;
    };
    return h;
}

fn heldJson(arena: Allocator, h: *const format.Held) !std.json.Value {
    var layers: std.json.ObjectMap = .empty;
    for (0..cfg.layers) |l| {
        var ids: std.json.Array = .init(arena);
        for (h.ids[l][0..h.counts[l]]) |e| try ids.append(.{ .integer = e });
        try layers.put(arena, try std.fmt.allocPrint(arena, "{d}", .{l}), .{ .array = ids });
    }
    return .{ .object = layers };
}

/// SRC's directories and files other than its index's shards by relative path, parents first; a symlink is refused, as it
/// could reach outside SRC.
const Ancillary = struct { dirs: []const []const u8, files: []const []const u8 };

fn ancillary(gpa: Allocator, arena: Allocator, io: Io, src: []const u8, shards: *const std.StringArrayHashMapUnmanaged(void)) !Ancillary {
    var dir = try Io.Dir.cwd().openDir(io, src, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(gpa);
    defer walker.deinit();
    var dirs: std.ArrayList([]const u8) = .empty;
    var files: std.ArrayList([]const u8) = .empty;
    while (try walker.next(io)) |e| switch (e.kind) {
        .directory => try dirs.append(arena, try arena.dupe(u8, e.path)),
        .file => {
            if (shards.contains(e.path)) continue;
            if (std.mem.eql(u8, e.path, index_name) or std.mem.eql(u8, e.path, provenance_name)) continue; // DST writes its own
            try files.append(arena, try arena.dupe(u8, e.path));
        },
        .sym_link => return error.SymlinkInSource,
        else => return error.UnsupportedSourceEntry,
    };
    return .{ .dirs = dirs.items, .files = files.items };
}

fn sourceProvenance(arena: Allocator, io: Io, from: []const u8) !?Provenance.Record {
    const text = Io.Dir.cwd().readFileAlloc(io, try join(arena, from, provenance_name), arena, .limited(1 << 28)) catch |e|
        return if (e == error.FileNotFound) null else e;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    return .{ .sha256 = try arena.dupe(u8, &std.fmt.bytesToHex(digest, .lower)), .text = text };
}

/// A new JSON file in `dir`, never over an existing one.
fn emit(arena: Allocator, io: Io, dir: Io.Dir, name: []const u8, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(arena, value, .{ .whitespace = .indent_2 });
    try dir.writeFile(io, .{ .sub_path = name, .data = try std.mem.concat(arena, u8, &.{ text, "\n" }), .flags = .{ .exclusive = true } });
}

fn fileSha256(arena: Allocator, io: Io, path: []const u8) ![]const u8 {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buffer: [1 << 16]u8 = undefined;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var offset: u64 = 0;
    while (true) {
        const got = try file.readPositionalAll(io, &buffer, offset);
        hash.update(buffer[0..got]);
        offset += got;
        if (got < buffer.len) break;
    }
    return arena.dupe(u8, &std.fmt.bytesToHex(hash.finalResult(), .lower));
}

fn join(arena: Allocator, dir: []const u8, name: []const u8) ![]u8 {
    return std.fs.path.join(arena, &.{ dir, name });
}

test {
    _ = @import("quant_test.zig");
}
