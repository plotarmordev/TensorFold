//! Real Nemotron facts and common lane rounds with deterministic head confidences, without device work.
const std = @import("std");
const nemo = @import("nemotron");
const lanes = @import("lanes");
const a = std.testing.allocator;
const be = lanes.backend;
const Fake = lanes.fake.Fake;
const prompt = [_]u32{ 1, 3, 7, 11, 19, 23, 29, 31 };

test "mtp policy: actual family facts carry positive caps and confidence into the common rounds" {
    const e = try a.create(nemo.Engine);
    defer a.destroy(e);
    e.* = undefined;
    e.c.latent = 512;
    e.c.bolt = false;
    var head: nemo.Head = undefined;
    var b: nemo.Lanes = undefined;
    b.e = e;
    b.head = &head;
    b.copy_drafts = true;
    b.cost_count = 0;
    b.mtp_ms = 0;
    b.mtp_drafts = 2;
    b.mtp_confidence = 0.5;
    const facts = b.facts();
    try std.testing.expectEqual(@as(u32, 2), facts.drafts);
    try std.testing.expectEqual(@as(?f64, 0.5), facts.draft_confidence);
    try std.testing.expect(b.backend().vtable.probabilities != null);
    try std.testing.expect(facts.copy_drafts and facts.hidden_rows);
}

const Record = struct {
    var most: usize = 0;
    fn probabilities(_: *anyopaque, _: *lanes.Stream, out: []f64) !bool {
        var chance: f64 = 1;
        for (out) |*v| {
            chance *= 0.8;
            v.* = chance;
        }
        return true;
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const target: *Fake = @ptrCast(@alignCast(ptr));
        for (windows) |w| if (std.mem.eql(u8, w.stream.id, "subject")) {
            most = @max(most, w.held + w.tokens.len);
        };
        try target.backend().vtable.verify(ptr, windows, out);
    }
};
const Options = struct { neighbor: bool = false, resumed: bool = false, handoff: bool = false, sampled: bool = false, drafted: bool = true, floor: ?f64 = 0.5, cap: u32 = 8, width: u32 = 16 };
fn run(o: Options) ![]u32 {
    Record.most = 0;
    const costs = [_]lanes.config.Cost{ .{ .width = 1, .ms = 1 }, .{ .width = 16, .ms = 1 } };
    var cfg = try lanes.Config.init(a, .{ .exact_width = o.width, .copy_drafts = false, .mtp = o.drafted, .speculate = true, .speculate_early = false, .drafts = o.cap, .draft_probabilities = true, .draft_confidence = o.floor, .window_costs = &costs, .hidden_rows = true, .batch_rows = o.width, .max_streams = 4, .draft_streams = true }, o.width, o.width - 1);
    defer cfg.deinit(a);
    var target: Fake = .{ .gpa = a };
    defer target.deinit();
    const original = target.backend();
    var vt = original.vtable.*;
    vt.probabilities = Record.probabilities;
    vt.verify = Record.verify;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, .{ .ptr = original.ptr, .vtable = &vt }, clock.clock());
    defer e.deinit();
    var kept: std.ArrayList(u32) = .empty;
    defer kept.deinit(a);
    try kept.appendSlice(a, prompt[0..4]);
    var subject = try lanes.Stream.init(a, .{ .id = "subject", .prompt = &prompt, .max_new = 80, .eos = &.{}, .drafts = o.drafted, .sampling = if (o.sampled) .{ .seed = 87, .temperature = 0.7 } else null, .reuse = if (o.resumed) .{ .saved = &kept, .at = 4 } else .{} });
    defer subject.deinit(a);
    var neighbor = try lanes.Stream.init(a, .{ .id = "neighbor", .prompt = &.{ 4, 6, 8, 12, 14 }, .max_new = 80, .eos = &.{}, .drafts = o.drafted });
    defer neighbor.deinit(a);
    try e.addStream(&subject);
    if (o.neighbor) try e.addStream(&neighbor);
    var steps: usize = 0;
    while (e.activeCount() > 0) : (steps += 1) {
        if (steps > 200) return error.DidNotFinish;
        try e.step();
        if (o.handoff and steps == 1 and !subject.finished) {
            try std.testing.expect(e.detach(&subject));
            try e.adopt(&subject);
        }
    }
    if (o.drafted) {
        try std.testing.expect(Record.most > 0);
        const ceiling: usize = if (o.floor == 0.5) @min(o.cap, 3) else if (o.floor == 1) 1 else o.cap;
        try std.testing.expect(Record.most <= ceiling);
    }
    return a.dupe(u32, subject.emitted());
}
test "mtp policy: shared confidence controls keep equality across neighbors resume and handoff" {
    for ([_]bool{ false, true }) |sampled| {
        const plain = try run(.{ .drafted = false, .sampled = sampled });
        defer a.free(plain);
        for ([_]bool{ false, true }) |neighbor| for ([_]bool{ false, true }) |resumed| for ([_]bool{ false, true }) |handoff| {
            const got = try run(.{ .neighbor = neighbor, .resumed = resumed, .handoff = handoff, .sampled = sampled });
            defer a.free(got);
            try std.testing.expectEqualSlices(u32, plain, got);
        };
    }
}
test "mtp policy: confidence endpoints and a positive cap are enforced by actual verification" {
    for ([_]?f64{ 0, 0.5, 1 }) |floor| for ([_]u32{ 1, 2, 8 }) |cap| {
        const got = try run(.{ .floor = floor, .cap = cap, .neighbor = true });
        defer a.free(got);
        try std.testing.expectEqual(@as(usize, 80), got.len);
    };
}

test "mtp policy: production probability read binds each stream before a completed copy" {
    const Seq = struct { p: [4]f32 };
    const B = struct {
        head: ?u8 = 1,
        mtp_confidence: ?f64 = 0.5,
        lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, struct { seq: *Seq }) = .empty,
        bound: ?*Seq = null,
        copies: usize = 0,
    };
    const Ops = struct {
        pub fn bind(b: *B, seq: *Seq) void {
            b.bound = seq;
        }
        pub fn download(b: *B, raw: []f32) !void {
            b.copies += 1;
            @memcpy(raw, b.bound.?.p[0..raw.len]);
        }
    };
    var b: B = .{};
    defer b.lanes.deinit(a);
    var s1 = try lanes.Stream.init(a, .{ .id = "one", .prompt = &prompt, .max_new = 8 });
    defer s1.deinit(a);
    var s2 = try lanes.Stream.init(a, .{ .id = "two", .prompt = &prompt, .max_new = 8 });
    defer s2.deinit(a);
    var one: Seq = .{ .p = @splat(0.5) };
    var two: Seq = .{ .p = @splat(0.75) };
    b.bound = &two;
    try b.lanes.put(a, &s1, .{ .seq = &one });
    try b.lanes.put(a, &s2, .{ .seq = &two });
    var out: [4]f64 = undefined;
    try std.testing.expect(try nemo.mtp_policy.read(&b, &s1, &out, Ops));
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.25, 0.125, 0.0625 }, &out);
    try std.testing.expect(try nemo.mtp_policy.read(&b, &s2, &out, Ops));
    try std.testing.expectEqualSlices(f64, &.{ 0.75, 0.5625, 0.421875, 0.31640625 }, &out);
    try std.testing.expect(try nemo.mtp_policy.read(&b, &s1, &out, Ops));
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.25, 0.125, 0.0625 }, &out);
    try std.testing.expectEqual(@as(usize, 3), b.copies);
    b.mtp_confidence = null; // no floor: the chances shared rounds price, at the lone rule's calibration
    try std.testing.expect(try nemo.mtp_policy.read(&b, &s2, &out, Ops));
    for ([_]f64{ 0.5625, 0.31640625, 0.177978515625, 0.1001129150390625 }, out) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-15);
    b.head = null;
    try std.testing.expect(!try nemo.mtp_policy.read(&b, &s1, &out, Ops));
    try std.testing.expectEqual(@as(usize, 4), b.copies);
}

/// The head in Super's shared-round runs: sure of every draft, unsure every third read per stream, or declining.
const Chances = enum { sure, varying, declined };
const Super = struct { drafted: bool = true, shared: bool = true, neighbor: bool = true, sampled: bool = false, resumed: bool = false, handoff: bool = false, chances: Chances = .sure };
/// A run's reads of the head's chances (`held`: the drafts it asks about) and verified windows, in order.
const Event = struct { subject: bool, read: bool, held: usize = 0, high: bool = false, drafts: usize = 0 };
const Trace = struct {
    var events: [1024]Event = undefined;
    var n: usize = 0;
    var reads: [2]usize = .{ 0, 0 };

    fn add(e: Event) void {
        events[n] = e;
        n += 1;
    }
    fn probabilities(_: *anyopaque, s: *lanes.Stream, out: []f64) !bool {
        const subject = std.mem.eql(u8, s.id, "subject");
        const k = reads[@intFromBool(subject)];
        reads[@intFromBool(subject)] += 1;
        // varying: the subject is unsure every third read, the neighbor on another phase
        const high = chances != .varying or (k + @intFromBool(!subject)) % 3 != 2;
        add(.{ .subject = subject, .read = true, .held = out.len, .high = high });
        var chance: f64 = 1;
        for (out, 0..) |*v, j| {
            chance *= if (high) 0.99 else if (j == 0) 0.5 else 0.05;
            v.* = chance;
        }
        return chances != .declined;
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        for (windows) |w| add(.{ .subject = std.mem.eql(u8, w.stream.id, "subject"), .read = false, .drafts = w.held + w.tokens.len });
        const target: *Fake = @ptrCast(@alignCast(ptr));
        try target.backend().vtable.verify(ptr, windows, out);
    }
    var chances: Chances = .sure;
};
/// Both replies of a run in rounds the actual Super facts set up (no floor), and its trace in `Trace`.
fn superRun(o: Super) ![2][]u32 {
    Trace.n = 0;
    Trace.reads = .{ 0, 0 };
    Trace.chances = o.chances;
    const e = try a.create(nemo.Engine);
    defer a.destroy(e);
    e.* = undefined;
    e.c.latent = 512;
    e.c.bolt = false;
    var head: nemo.Head = undefined;
    var b: nemo.Lanes = undefined;
    b.e = e;
    b.head = if (o.drafted) &head else null;
    b.copy_drafts = false;
    b.mtp_drafts = null;
    b.mtp_confidence = null;
    b.mtp_ms = 3.6;
    b.cost_count = 32;
    for (b.costs[0..32], 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 46.0 + 4.0 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(a, b.facts(), 32, 31);
    defer cfg.deinit(a);
    cfg.shared_chances = cfg.shared_chances and o.shared;
    var target: Fake = .{ .gpa = a };
    defer target.deinit();
    const original = target.backend();
    var vt = original.vtable.*;
    vt.probabilities = Trace.probabilities;
    vt.verify = Trace.verify;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, .{ .ptr = original.ptr, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var kept: std.ArrayList(u32) = .empty;
    defer kept.deinit(a);
    try kept.appendSlice(a, prompt[0..4]);
    const sampling: ?lanes.Sampling = if (o.sampled) .{ .seed = 87, .temperature = 0.7 } else null;
    var subject = try lanes.Stream.init(a, .{ .id = "subject", .prompt = &prompt, .max_new = 80, .eos = &.{}, .drafts = o.drafted, .sampling = sampling, .reuse = if (o.resumed) .{ .saved = &kept, .at = 4 } else .{} });
    defer subject.deinit(a);
    var neighbor = try lanes.Stream.init(a, .{ .id = "neighbor", .prompt = &.{ 4, 6, 8, 12, 14 }, .max_new = 80, .eos = &.{}, .drafts = o.drafted, .sampling = sampling });
    defer neighbor.deinit(a);
    try core.addStream(&subject);
    if (o.neighbor) try core.addStream(&neighbor);
    var steps: usize = 0;
    while (core.activeCount() > 0) : (steps += 1) {
        if (steps > 200) return error.DidNotFinish;
        try core.step();
        if (o.handoff and steps == 1 and !subject.finished) {
            try std.testing.expect(core.detach(&subject));
            try core.adopt(&subject);
        }
    }
    return .{ try a.dupe(u32, subject.emitted()), try a.dupe(u32, neighbor.emitted()) };
}
fn freeBoth(r: [2][]u32) void {
    for (r) |x| a.free(x);
}
/// The widest window a stream verified in a run.
fn widest(subject: bool) usize {
    var most: usize = 0;
    for (Trace.events[0..Trace.n]) |x| if (!x.read and x.subject == subject) {
        most = @max(most, x.drafts);
    };
    return most;
}
test "mtp policy: Super's shared rounds draft as deep as the head's chances pay for, its replies unchanged" {
    const plain = try superRun(.{ .drafted = false });
    defer freeBoth(plain);
    const got = try superRun(.{});
    defer freeBoth(got);
    for (plain, got) |p, g| try std.testing.expectEqualSlices(u32, p, g);
    try std.testing.expect(widest(true) >= 12); // the priors alone price three or four drafts a stream
}
test "mtp policy: each stream's next shared depth follows its own latest chances, never an earlier read" {
    const plain = try superRun(.{ .drafted = false });
    defer freeBoth(plain);
    const got = try superRun(.{ .chances = .varying });
    defer freeBoth(got);
    for (plain, got) |p, g| try std.testing.expectEqualSlices(u32, p, g);
    var deep = [2]usize{ 0, 0 };
    var unsure = [2]usize{ 0, 0 };
    var last: [2]?Event = .{ null, null };
    for (Trace.events[0..Trace.n], 0..) |x, i| {
        const s = @intFromBool(x.subject);
        if (!x.read) continue;
        // the next round's request: one draft after an unsure chain, deep again after a sure one
        if (last[s]) |before| {
            if (!before.high) try std.testing.expect(x.held <= 1);
            if (before.high) deep[s] = @max(deep[s], x.held);
        }
        // this round's trim: an unsure chain verifies its first draft alone
        if (!x.high) for (Trace.events[i + 1 .. Trace.n]) |y| if (!y.read and y.subject == x.subject) {
            try std.testing.expect(y.drafts <= 1);
            unsure[s] += 1;
            break;
        };
        last[s] = x;
    }
    for (deep, unsure) |d, u| try std.testing.expect(d >= 8 and u > 0);
}
test "mtp policy: Super's shared chances keep sampled, resumed and handed-over replies equal to undrafted" {
    for ([_]bool{ false, true }) |sampled| for ([_]bool{ false, true }) |resumed| for ([_]bool{ false, true }) |handoff| {
        const plain = try superRun(.{ .drafted = false, .sampled = sampled, .resumed = resumed, .handoff = handoff });
        defer freeBoth(plain);
        const got = try superRun(.{ .sampled = sampled, .resumed = resumed, .handoff = handoff, .chances = .varying });
        defer freeBoth(got);
        try std.testing.expect(Trace.reads[0] > 0 and Trace.reads[1] > 0); // the shared no-floor path ran
        for (plain, got) |p, g| try std.testing.expectEqualSlices(u32, p, g);
    };
}
test "mtp policy: shared chances off, declined or alone leave the prior depth rule as it was" {
    const off = try superRun(.{ .shared = false });
    defer freeBoth(off);
    try std.testing.expectEqual(@as(usize, 0), Trace.reads[0] + Trace.reads[1]);
    var before: [1024]Event = undefined;
    const count = Trace.n;
    @memcpy(before[0..count], Trace.events[0..count]);
    const declined = try superRun(.{ .chances = .declined });
    defer freeBoth(declined);
    try std.testing.expect(Trace.reads[0] > 0);
    var windows: usize = 0;
    for (Trace.events[0..Trace.n]) |x| if (!x.read) {
        while (before[windows].read) windows += 1;
        try std.testing.expectEqual(before[windows].subject, x.subject);
        try std.testing.expectEqual(before[windows].drafts, x.drafts);
        windows += 1;
    };
    try std.testing.expectEqual(count, windows);
    for (off, declined) |p, g| try std.testing.expectEqualSlices(u32, p, g);
    const alone = try superRun(.{ .neighbor = false });
    defer freeBoth(alone);
    try std.testing.expectEqual(@as(usize, 0), Trace.reads[0] + Trace.reads[1]); // a lone stream keeps its own rule
}

test "mtp policy: explicit cap leaves copy capacity independent and invalid confidence is refused" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .drafts = 2, .mtp = true, .speculate = true, .draft_confidence = 0.5 }, 16, 15);
    defer cfg.deinit(a);
    try std.testing.expectEqual(@as(u32, 2), cfg.most_drafts);
    try std.testing.expectEqual(@as(u32, 15), cfg.max_copy);
    for ([_]f64{ -1, 2, std.math.nan(f64) }) |floor| try std.testing.expectError(error.InvalidDraftConfidence, lanes.Config.init(a, .{ .exact_width = 16, .draft_confidence = floor }, 16, 15));
    const empty: lanes.table.Table = .{};
    const counts = try lanes.allocate.allocate(a, &.{1}, &.{&.{ 0, 0, 0 }}, &empty, 0, 4);
    defer a.free(counts);
    try std.testing.expectEqualSlices(u32, &.{3}, counts);
}

test "mtp policy: no-head models retain the landed shared copy lane" {
    const e = try a.create(nemo.Engine);
    defer a.destroy(e);
    e.* = undefined;
    e.c.latent = 0;
    e.c.bolt = true;
    var b: nemo.Lanes = undefined;
    b.e = e;
    b.head = null;
    b.copy_drafts = true;
    b.cost_count = 0;
    b.mtp_ms = 0;
    b.mtp_drafts = null;
    b.mtp_confidence = null;
    const facts = b.facts();
    try std.testing.expectEqual(@as(u32, 16), facts.exact_width);
    try std.testing.expect(facts.hidden_rows and facts.max_streams > 1 and facts.copy_drafts);
    try std.testing.expect(!facts.mtp);
}

test "mtp policy: only Super widens and Bolt keeps its qualified copy window" {
    var cfg: nemo.Config = undefined;
    cfg.latent = 512;
    cfg.bolt = false;
    try std.testing.expectEqual(@as(usize, 32), nemo.state.windowRows(cfg));
    cfg.latent = 0;
    try std.testing.expectEqual(@as(usize, 16), nemo.state.windowRows(cfg));
    cfg.latent = 512;
    cfg.bolt = true;
    try std.testing.expectEqual(@as(usize, 16), nemo.state.windowRows(cfg));
}

test "mtp policy: the inherited policy works through 31 drafts and a resumed shared 32 row round" {
    const plain = try run(.{ .drafted = false });
    defer a.free(plain);
    for ([_]bool{ false, true }) |neighbor| for ([_]bool{ false, true }) |resumed| for ([_]bool{ false, true }) |handoff| {
        const got = try run(.{ .width = 32, .cap = 31, .floor = 0, .neighbor = neighbor, .resumed = resumed, .handoff = handoff });
        defer a.free(got);
        try std.testing.expectEqualSlices(u32, plain, got);
    };
}

test "mtp policy: shared latent windows avoid per-layer side events while lone graphs retain overlap" {
    var f: nemo.Forward = undefined;
    f.c.latent = 512;
    f.side = .{ .s = undefined, .fork = undefined, .join = undefined };
    f.tp = null;
    f.route = .{};
    var buffers: nemo.state.Buffers = undefined;
    const Seg = @typeInfo(@TypeOf(f.segs)).pointer.child;
    var parts: [4]Seg = undefined;
    for (&parts, 0..) |*p, i| p.* = .{ .b = &buffers, .row0 = i * 8, .rows = 8, .sampled = false, .slot = @intCast(i), .untaught = i == 2 };
    const shared = f.routedWindow(&parts);
    try std.testing.expect(shared.side == null);
    try std.testing.expect(shared.untaught);
    try std.testing.expectEqual(@as(usize, 4), shared.segs.len);
    try std.testing.expectEqual(@as(u32, 4), shared.route.count);
    try std.testing.expect(f.routedWindow(parts[0..1]).side != null);
    f.tp = @ptrFromInt(4096);
    try std.testing.expect(f.routedWindow(parts[0..1]).side == null);
    f.tp = null;
    f.c.latent = 0;
    try std.testing.expect(f.routedWindow(&parts).side != null);
}

test "mtp policy: rank-local cached plans survive sibling follow after a resumed sequence is bound" {
    var c: nemo.Config = undefined;
    c.latent = 1024;
    c.bolt = false;
    c.shares = 2;
    var owner: nemo.state.Buffers = undefined;
    owner.shared = 0x100000;
    owner.sids = 0x200000;
    owner.plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0x300000, .hist = 0x400000 };
    owner.table = 0x500000;
    owner.state_bytes = @splat(64);
    inline for (nemo.state.seq_fields, 0..) |name, i| @field(owner, name) = 0x600000 + i * 4096;
    for ([_]usize{ 1, 2 }) |world| {
        c.world = world;
        var sibling = owner;
        sibling.plan.rank = 0x700000;
        sibling.plan.hist = 0x800000;
        for ([_]usize{ 1, 16, 32 }) |rows| {
            const before = sibling.latentPlan(c, rows);
            sibling.follow(&owner);
            const after = sibling.latentPlan(c, rows);
            try std.testing.expectEqual(before, after);
            try std.testing.expectEqual(owner.sids, sibling.sids);
            try std.testing.expectEqual(owner.shared, sibling.shared);
            try std.testing.expectEqual(owner.table, sibling.table);
            try std.testing.expectEqual(@as(u64, 0x700000), after.rank);
            try std.testing.expect(after.items >= after.members + rows * c.ownShares() * 4);
            const items = nemo.kernels.maxItems(rows * c.ownShares(), c.ownShares(), 16);
            try std.testing.expect(after.counts >= after.items + 3 * items * 4);
        }
    }
}
