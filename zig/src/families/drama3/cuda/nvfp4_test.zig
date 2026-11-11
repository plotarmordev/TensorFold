//! The CUDA loader's compact NVFP4 experts on the host: a held map's experts staged in slot order as stored, decoded
//! from the packed blocks the kernels read and checked against the FP64 scalar reference. No CUDA context.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("../config.zig");
const wts = @import("weights.zig");
const quant = @import("../metal/quant_format.zig");
const ref = @import("../metal/quant_reference.zig");
const Shards = @import("../shards.zig").Shards;

const testing = std.testing;
const io = testing.io;
const E = cuda.experts.Nvfp4Experts;
const held_ids = [_]u8{ 3, 9, 250 };
const projections = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
const shape = [3][2]usize{ .{ cfg.expert_width, cfg.hidden }, .{ cfg.expert_width, cfg.hidden }, .{ cfg.hidden, cfg.expert_width } };

/// Layer 0's held experts in NVFP4 as tf-drama-quant writes them; layer 1's expert 0 gate in FP8.
fn writeCheckpoint(arena: std.mem.Allocator, dir: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(27);
    const rnd = prng.random();
    var tensors: std.ArrayList(core.shard_edit.New) = .empty;
    for (held_ids) |e| for (projections, shape) |p, s| {
        const name = try std.fmt.allocPrint(arena, "model.language_model.layers.0.mlp.experts.{d}.{s}.weight", .{ e, p });
        const codes = try arena.alloc(u8, s[0] * s[1] / 2);
        rnd.bytes(codes);
        const scales = try arena.alloc(u8, s[0] * s[1] / 16);
        for (scales) |*b| b.* = 0x30 + rnd.uintLessThan(u8, 0x30);
        const global = try arena.dupe(u8, std.mem.asBytes(&(0.5 + rnd.float(f32))));
        try tensors.append(arena, .{ .name = name, .dtype = .u8, .shape = try arena.dupe(usize, &.{ s[0], s[1] / 2 }), .bytes = codes });
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}_scale", .{name}), .dtype = .f8_e4m3, .shape = try arena.dupe(usize, &.{ s[0], s[1] / 16 }), .bytes = scales });
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}_scale_2", .{name}), .dtype = .f32, .shape = &.{}, .bytes = global });
    };
    const fp8 = try arena.alloc(u8, cfg.expert_width * cfg.hidden);
    rnd.bytes(fp8);
    try tensors.append(arena, .{ .name = "model.language_model.layers.1.mlp.experts.0.gate_proj.weight", .dtype = .f8_e4m3, .shape = &.{ cfg.expert_width, cfg.hidden }, .bytes = fp8 });
    try core.shard_edit.create(arena, try std.fs.path.join(arena, &.{ dir, "model.safetensors" }), &.{.{ "format", "pt" }}, tensors.items);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(arena, "{\"metadata\": {}, \"weight_map\": {");
    for (tensors.items, 0..) |t, i| try index.print(arena, "{s}\"{s}\": \"model.safetensors\"", .{ if (i == 0) "" else ", ", t.name });
    try index.appendSlice(arena, "}}");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" }), .data = index.items });
}

/// Code and e4m3 scale of (row, input) in matrix `part` of `parts` packed blocks, as experts.zig documents the layout.
fn unpacked(blocks: []const u32, parts: usize, part: usize, k: usize, row: usize, input: usize) struct { u8, u8 } {
    const rr = row % E.cols;
    const ii = input % 32;
    const h = ii / 16;
    const block = blocks[(((row / E.cols) * (k / 32) + input / 32) * parts + part) * E.words ..][0..E.words];
    const word = block[((rr % 8) * 4 + (ii % 16) / 4) * 4 + rr / 8];
    const scale = std.mem.sliceAsBytes(block[128..])[((((rr % 8) / 2) * 2 + h) * 4 + rr / 8) * 2 + rr % 2];
    return .{ @intCast((word >> E.slot(h, ii % 4)) & 0xF), scale };
}

test "compact NVFP4 experts stage as stored in held-slot order and decode to the FP64 reference" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", arena);
    try writeCheckpoint(arena, dir);
    var sh = try Shards.open(testing.allocator, io, dir);
    defer sh.close(io);

    var held: quant.Held = .{};
    held.counts[0] = held_ids.len;
    @memcpy(held.ids[0][0..held_ids.len], &held_ids);
    try testing.expect(try wts.nvfp4Layer(&sh, 0, &held));
    var stage = try wts.stageNvfp4(testing.allocator, &sh, 0, held.ids[0][0..held.counts[0]], 2);
    defer stage.deinit(testing.allocator);

    var prng = std.Random.DefaultPrng.init(5);
    var x: [cfg.hidden]u16 = undefined;
    for (&x) |*v| v.* = 0x3c00 + prng.random().uintLessThan(u16, 0x400) + (@as(u16, prng.random().int(u1)) << 15);
    const slots = held.slots(0);
    const per_up = stage.up.len / held_ids.len;
    const per_down = stage.down.len / held_ids.len;
    // every original id the router can pick reaches the slot holding its own weights; the rest are masked
    for (slots, 0..) |slot, id| {
        if (std.mem.indexOfScalar(u8, &held_ids, @intCast(id)) == null) {
            try testing.expectEqual(@as(i32, -1), slot);
            continue;
        }
        const s: usize = @intCast(slot);
        for (projections, shape, 0..) |p, dims, j| {
            var nb: [3][128]u8 = undefined;
            const base = "model.language_model.layers.0.mlp.experts.{d}.{s}.weight";
            const w = try sh.get(try std.fmt.bufPrint(&nb[0], base, .{ id, p }));
            const sc = try sh.get(try std.fmt.bufPrint(&nb[1], base ++ "_scale", .{ id, p }));
            const g = try sh.get(try std.fmt.bufPrint(&nb[2], base ++ "_scale_2", .{ id, p }));
            const global: f32 = @bitCast(std.mem.readInt(u32, g.bytes[0..4], .little));
            const up = j < 2;
            const blocks = if (up) stage.up[s * per_up ..][0..per_up] else stage.down[s * per_down ..][0..per_down];
            const staged_global = if (up) stage.up_scale[2 * s + j] else stage.down_scale[s];
            try testing.expectEqual(global, staged_global);
            const n, const k = dims;
            for (0..n) |row| {
                var sum: f64 = 0;
                for (0..k) |i| {
                    const code, const scale = unpacked(blocks, if (up) 2 else 1, if (up) j else 0, k, row, i);
                    sum += ref.fp4(code) * ref.fp8(scale) * ref.fromBf16(x[i]);
                }
                try testing.expectEqual(ref.projection(w.bytes, sc.bytes, global, row, k, x[0..k]), sum * staged_global);
            }
        }
    }
}

test "a compact map over FP8 experts is refused; a full FP8 layer keeps the FP8 path" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", arena);
    try writeCheckpoint(arena, dir);
    var sh = try Shards.open(testing.allocator, io, dir);
    defer sh.close(io);

    var held: quant.Held = .{};
    try testing.expect(!try wts.nvfp4Layer(&sh, 1, &held));
    held.counts[1] = cfg.top_k;
    try testing.expectError(error.CompactFp8Experts, wts.nvfp4Layer(&sh, 1, &held));
}
