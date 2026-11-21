//! A typed std.json.Value reader that avoids std.json's int path: its quad-float fallback pulls compiler-rt's memcpy into the native server.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// `bytes` parsed as T, unknown fields ignored, the Value tree and every dupe inside `parsed.arena` (deinit as usual).
pub fn parse(comptime T: type, a: Allocator, bytes: []const u8) !std.json.Parsed(T) {
    var parsed = std.json.Parsed(T){
        .arena = try a.create(std.heap.ArenaAllocator),
        .value = undefined,
    };
    errdefer a.destroy(parsed.arena);
    parsed.arena.* = .init(a);
    errdefer parsed.arena.deinit();
    const json = try std.json.parseFromSliceLeaky(std.json.Value, parsed.arena.allocator(), bytes, .{ .parse_numbers = false });
    parsed.value = try value(T, parsed.arena.allocator(), json);
    return parsed;
}

/// The fields of T filled from `source`, ints read from their decimal text without floats;
/// `source` must come from a `.parse_numbers = false` parse (numbers as `.number_string`).
pub fn value(comptime T: type, a: Allocator, source: std.json.Value) std.json.ParseFromValueError!T {
    switch (@typeInfo(T)) {
        .int => switch (source) {
            .number_string, .string => |raw| return decimalInteger(T, raw),
            else => return error.UnexpectedToken,
        },
        .float => switch (source) {
            .number_string, .string => |raw| return std.fmt.parseFloat(T, raw),
            else => return error.UnexpectedToken,
        },
        .optional => |info| {
            if (source == .null) return null;
            return try value(info.child, a, source);
        },
        .@"struct" => |info| {
            if (source != .object) return error.UnexpectedToken;
            if (@hasField(T, "map")) {
                const V = @typeInfo(@FieldType(@FieldType(T, "map").GetOrPutResult, "value_ptr")).pointer.child;
                var result: T = .{};
                var it = source.object.iterator();
                while (it.next()) |kv| try result.map.put(a, kv.key_ptr.*, try value(V, a, kv.value_ptr.*));
                return result;
            }
            var result: T = undefined;
            inline for (info.field_names, info.field_types, info.field_attrs) |name, Field, attrs| {
                @field(result, name) = if (source.object.get(name)) |v|
                    try value(Field, a, v)
                else
                    attrs.defaultValue(Field) orelse return error.MissingField;
            }
            return result;
        },
        .array => |info| {
            if (source != .array or source.array.items.len != info.len) return error.LengthMismatch;
            var result: T = undefined;
            for (&result, source.array.items) |*out, item| out.* = try value(info.child, a, item);
            return result;
        },
        .pointer => |info| {
            if (info.size != .slice) @compileError("JSON documents only contain slice pointers");
            if (info.child == u8 and source == .string) return a.dupe(u8, source.string);
            if (source != .array) return error.UnexpectedToken;
            const values = try a.alloc(info.child, source.array.items.len);
            for (values, source.array.items) |*out, item| out.* = try value(info.child, a, item);
            return values;
        },
        .bool, .@"enum" => return std.json.innerParseFromValue(T, a, source, .{}),
        else => @compileError("unsupported JSON reader type " ++ @typeName(T)),
    }
}

fn decimalInteger(comptime T: type, raw: []const u8) std.json.ParseFromValueError!T {
    if (std.mem.indexOfAny(u8, raw, ".eE") == null) return std.fmt.parseInt(T, raw, 10);
    const negative = raw[0] == '-';
    const unsigned = if (negative or raw[0] == '+') raw[1..] else raw;
    const split = std.mem.indexOfAny(u8, unsigned, "eE") orelse unsigned.len;
    const mantissa = unsigned[0..split];
    const exponent = if (split == unsigned.len) 0 else try std.fmt.parseInt(i64, unsigned[split + 1 ..], 10);
    const point = std.mem.indexOfScalar(u8, mantissa, '.') orelse mantissa.len;
    const decimal_places: i64 = @intCast(mantissa.len -| (point + 1));
    const significant = std.mem.trimEnd(u8, mantissa, "0");
    const zeros: i64 = @intCast(mantissa.len - significant.len);
    var seen_digit = zeros != 0;
    var n: T = 0;
    for (significant, 0..) |ch, index| {
        if (index == point) continue;
        if (ch < '0' or ch > '9') return error.InvalidNumber;
        seen_digit = true;
        n = try std.math.add(T, try std.math.mul(T, n, 10), ch - '0');
    }
    if (!seen_digit) return error.InvalidNumber;
    if (n == 0) return 0;
    if (negative) return error.Overflow;
    const places = try std.math.add(i64, exponent, zeros - decimal_places);
    if (places < 0) return error.InvalidNumber;
    const power = std.math.cast(T, places) orelse return error.Overflow;
    const scale = std.math.powi(T, 10, power) catch return error.Overflow;
    return std.math.mul(T, n, scale);
}
