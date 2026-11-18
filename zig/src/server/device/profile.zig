//! ~/.tensorfold/device.json: the last scan, versioned, read back until the hardware or this binary changes.
const std = @import("std");
const sys = @import("sys");
const machine = @import("machine.zig");
const support = @import("support.zig");
const Allocator = std.mem.Allocator;

pub const schema: u32 = 1;

/// quick: the OS's own listing, a card's compute capability from its name; full: the GPU's driver was asked too.
pub const Scan = enum { quick, full };

pub const Doc = struct {
    schema: u32 = schema,
    scanned_at: []const u8,
    binary_version: []const u8,
    /// machine.fingerprint of the quick listing, which both scans make.
    fingerprint: []const u8,
    scan: Scan,
    machine: machine.Machine,
    /// What the binary that scanned derived, kept as a record; readers derive their own.
    support: support.Support,
};

/// ~/.tensorfold/device.json (HOME, or USERPROFILE on Windows); null when neither is set.
pub fn path(a: Allocator, env: ?*const std.process.Environ.Map) !?[]const u8 {
    const e = env orelse return null;
    const home = e.get("HOME") orelse e.get("USERPROFILE") orelse return null;
    if (home.len == 0) return null;
    return try std.fs.path.join(a, &.{ home, ".tensorfold", "device.json" });
}

/// The profile at `file`; null when it is missing, unreadable, malformed or of another schema.
pub fn load(a: Allocator, io: std.Io, file: []const u8) ?Doc {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(1 << 20)) catch return null;
    return parse(a, bytes);
}

/// The profile read through JSON values: a typed parse links compiler-rt's quad floats, and with them its memcpy over libSystem's.
pub fn parse(a: Allocator, bytes: []const u8) ?Doc {
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{ .allocate = .alloc_always }) catch return null;
    if (root != .object) return null;
    const doc = readDoc(a, root.object) catch return null;
    return if (doc.schema == schema) doc else null;
}

fn readDoc(a: Allocator, o: std.json.ObjectMap) !Doc {
    return .{
        .schema = try int(u32, o, "schema", schema),
        .scanned_at = try text(o, "scanned_at", null),
        .binary_version = try text(o, "binary_version", null),
        .fingerprint = try text(o, "fingerprint", null),
        .scan = try enumeration(Scan, o, "scan", null),
        .machine = try readMachine(a, try object(o, "machine")),
        .support = try readSupport(a, try object(o, "support")),
    };
}

fn readMachine(a: Allocator, o: std.json.ObjectMap) !machine.Machine {
    return .{
        .os = try enumeration(machine.Os, o, "os", null),
        .os_version = try text(o, "os_version", ""),
        .arch = try text(o, "arch", null),
        .cpu = try text(o, "cpu", ""),
        .cores = try int(u32, o, "cores", 0),
        .performance_cores = try int(u32, o, "performance_cores", 0),
        .efficiency_cores = try int(u32, o, "efficiency_cores", 0),
        .ram = try int(u64, o, "ram", 0),
        .gpu_wired_limit = try int(u64, o, "gpu_wired_limit", 0),
        .wsl = try flag(o, "wsl", false),
        .gpus = try gpus(a, o.get("gpus")),
    };
}

fn gpus(a: Allocator, v: ?std.json.Value) ![]const machine.Gpu {
    const list = v orelse return &.{};
    if (list != .array) return error.BadProfile;
    const out = try a.alloc(machine.Gpu, list.array.items.len);
    for (out, list.array.items) |*g, item| {
        if (item != .object) return error.BadProfile;
        const o = item.object;
        g.* = .{
            .vendor = try enumeration(machine.Vendor, o, "vendor", null),
            .name = try text(o, "name", null),
            .memory = try int(u64, o, "memory", 0),
            .unified = try flag(o, "unified", false),
            .sm = try int(u32, o, "sm", 0),
            .gfx = try text(o, "gfx", ""),
            .pci = try text(o, "pci", ""),
            .source = try enumeration(machine.Source, o, "source", .os),
        };
    }
    return out;
}

fn readSupport(a: Allocator, o: std.json.ObjectMap) !support.Support {
    return .{
        .backend = try optionalText(o, "backend"),
        .gpu = try optionalInt(usize, o, "gpu"),
        .reason = try text(o, "reason", ""),
        .certain = try flag(o, "certain", true),
        .formats = try strings(a, o.get("formats")),
        .unavailable = try unavailable(a, o.get("unavailable")),
        .notes = try strings(a, o.get("notes")),
        .budget = try int(u64, o, "budget", 0),
        .budget_rule = try text(o, "budget_rule", ""),
    };
}

fn unavailable(a: Allocator, v: ?std.json.Value) ![]const support.Unavailable {
    const list = v orelse return &.{};
    if (list != .array) return error.BadProfile;
    const out = try a.alloc(support.Unavailable, list.array.items.len);
    for (out, list.array.items) |*u, item| {
        if (item != .object) return error.BadProfile;
        u.* = .{ .family = try text(item.object, "family", null), .reason = try text(item.object, "reason", null) };
    }
    return out;
}

fn strings(a: Allocator, v: ?std.json.Value) ![]const []const u8 {
    const list = v orelse return &.{};
    if (list != .array) return error.BadProfile;
    const out = try a.alloc([]const u8, list.array.items.len);
    for (out, list.array.items) |*s, item| {
        if (item != .string) return error.BadProfile;
        s.* = item.string;
    }
    return out;
}

fn object(o: std.json.ObjectMap, key: []const u8) !std.json.ObjectMap {
    const v = o.get(key) orelse return error.BadProfile;
    if (v != .object) return error.BadProfile;
    return v.object;
}

fn int(comptime T: type, o: std.json.ObjectMap, key: []const u8, default: ?T) !T {
    const v = o.get(key) orelse return default orelse error.BadProfile;
    if (v == .integer) return std.math.cast(T, v.integer) orelse error.BadProfile;
    if (v == .number_string) return std.fmt.parseInt(T, v.number_string, 10) catch error.BadProfile;
    return error.BadProfile;
}

fn optionalInt(comptime T: type, o: std.json.ObjectMap, key: []const u8) !?T {
    const v = o.get(key) orelse return null;
    if (v == .null) return null;
    return try int(T, o, key, null);
}

fn text(o: std.json.ObjectMap, key: []const u8, default: ?[]const u8) ![]const u8 {
    const v = o.get(key) orelse return default orelse error.BadProfile;
    if (v != .string) return error.BadProfile;
    return v.string;
}

fn optionalText(o: std.json.ObjectMap, key: []const u8) !?[]const u8 {
    const v = o.get(key) orelse return null;
    if (v == .null) return null;
    if (v != .string) return error.BadProfile;
    return v.string;
}

fn flag(o: std.json.ObjectMap, key: []const u8, default: ?bool) !bool {
    const v = o.get(key) orelse return default orelse error.BadProfile;
    if (v != .bool) return error.BadProfile;
    return v.bool;
}

fn enumeration(comptime E: type, o: std.json.ObjectMap, key: []const u8, default: ?E) !E {
    const v = o.get(key) orelse return default orelse error.BadProfile;
    if (v != .string) return error.BadProfile;
    inline for (@typeInfo(E).@"enum".field_names, @typeInfo(E).@"enum".field_values) |name, value| {
        if (std.mem.eql(u8, v.string, name)) return @as(E, @enumFromInt(value));
    }
    return error.BadProfile;
}

pub fn write(w: *std.Io.Writer, doc: Doc) !void {
    try std.json.Stringify.value(doc, .{ .whitespace = .indent_2 }, w);
    try w.writeByte('\n');
}

/// Writes `doc` to `file` through a unique temporary file renamed into place, its folder made first.
pub fn save(a: Allocator, io: std.Io, file: []const u8, doc: Doc) !void {
    if (std.fs.path.dirname(file)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
    var out: std.Io.Writer.Allocating = .init(a);
    try write(&out.writer, doc);
    try sys.atomic_file.replace(a, file, &.{out.written()});
}

/// Why a scan is due, or null when the saved profile still describes this machine and binary.
pub const Due = enum { missing, asked, binary, hardware, quick };

/// `want_full`: the caller can ask the driver (`tensorfold device`) and a quick profile is not enough for it.
pub fn due(old: ?Doc, fingerprint: []const u8, version: []const u8, asked: bool, want_full: bool) ?Due {
    const d = old orelse return .missing;
    if (asked) return .asked;
    if (!std.mem.eql(u8, d.binary_version, version)) return .binary;
    if (!std.mem.eql(u8, d.fingerprint, fingerprint)) return .hardware;
    if (want_full and d.scan == .quick) return .quick;
    return null;
}

/// UTC time as ISO 8601 to the second ("2026-11-13T01:22:01Z").
pub fn stamp(buf: *[20]u8, seconds: i64) []const u8 {
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(0, seconds)) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        yd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(),
    }) catch unreachable;
}
