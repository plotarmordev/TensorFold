//! Greedy CUDA DSpark integration: target equality after accepted and rejected windows, including chunked prompts.
const std = @import("std");
const lanes = @import("lanes");
const ink = @import("inkling2");
const check = @import("check.zig");

const Result = struct {
    ids: []u32,
    proposed: u64,
    accepted: u64,

    fn deinit(r: Result, gpa: std.mem.Allocator) void {
        gpa.free(r.ids);
    }
};

fn idsFrom(gpu: check.Gpu, path: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpu.gpa, bytes, .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases") orelse return error.DSparkTestFixture;
    for (cases.array.items) |case| {
        const text = case.object.get("text") orelse continue;
        if (!std.mem.startsWith(u8, text.string, "def f(")) continue;
        const saved = case.object.get("ids") orelse return error.DSparkTestFixture;
        const ids = try gpu.gpa.alloc(u32, saved.array.items.len);
        for (saved.array.items, ids) |number, *id| id.* = @intCast(number.integer);
        return ids;
    }
    return error.DSparkTestFixture;
}

fn generate(gpu: check.Gpu, backend: lanes.backend.Backend, facts: lanes.Model, prompt: []const u32, drafts: bool, count: u32) !Result {
    var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "inkling-dspark", .prompt = prompt, .max_new = count, .drafts = drafts });
    defer stream.deinit(gpu.gpa);
    var config = try lanes.Config.init(gpu.gpa, facts, ink.native.prompt_rows, ink.native.window_rows);
    defer config.deinit(gpu.gpa);
    config.timed = false;
    config.lone_copies = false;
    var clock: lanes.fake.FixedClock = .{};
    var runner = lanes.Engine.init(gpu.gpa, &config, backend, clock.clock());
    defer runner.deinit();
    try runner.addStream(&stream);
    while (!stream.finished) try runner.step();
    if (stream.reason != .length) return error.DSparkServingDidNotFinish;
    return .{ .ids = try gpu.gpa.dupe(u32, stream.emitted()), .proposed = stream.drafted, .accepted = stream.accepted };
}

pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8) !void {
    const short = try idsFrom(gpu, ids_path);
    defer gpu.gpa.free(short);
    if (short.len < 16 or short.len > 512) return error.DSparkTestPromptLength;
    const long = try gpu.gpa.alloc(u32, 2053);
    defer gpu.gpa.free(long);
    for (long, 0..) |*id, i| id.* = short[i % short.len];
    var loaded = try ink.native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = 4096, .drafts = true });
    defer loaded.deinit(loaded.ctx);
    const owned = try ink.native.openDrafter(gpu.gpa, gpu.io, loaded.ctx, model, 4);
    defer owned.close(owned.ctx);
    var drafted = try lanes.drafted.Drafted.init(gpu.gpa, loaded.backend, owned.drafter);
    defer drafted.deinit();
    try owned.reserve(owned.ctx, 1);
    for ([2][]const u32{ short, long }, 0..) |prompt, case| {
        const count: u32 = if (case == 0) 96 else 24;
        const plain = try generate(gpu, loaded.backend, loaded.facts, prompt, false, count);
        defer plain.deinit(gpu.gpa);
        const proposed = try generate(gpu, drafted.backend(), drafted.facts(loaded.facts), prompt, true, count);
        defer proposed.deinit(gpu.gpa);
        if (!std.mem.eql(u32, plain.ids, proposed.ids)) return error.DSparkChangedGreedyTokens;
        if (proposed.proposed == 0 or proposed.accepted == 0) return error.DSparkNotExercised;
        check.pass("Mini CUDA DSpark greedy target equality: case {d}, {d} proposed, {d} accepted", .{ case, proposed.proposed, proposed.accepted });
    }
}
