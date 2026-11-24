//! A speech server's voices: Drama 3's catalog store, loaded at startup and listed on `/v1/voices`.
const std = @import("std");
const server = @import("server.zig");

/// Fills `srv.voices` from the engine's own catalog or `--voices` and selects the default voice; voice flags without a
/// speech model are refused, and so are inline and managed voices on an engine-owned catalog.
pub fn start(srv: *server.Server, gpa: std.mem.Allocator, io: std.Io, config: server.Config) !void {
    const voice = config.voice;
    srv.voices = .{ .gpa = gpa, .io = io, .dir = voice.dir };
    if (srv.info.speech == null and (voice.dir != null or voice.inline_voices or voice.manage)) return error.VoicesRequireSpeech;
    const engine_voices = if (srv.info.speech) |sp| sp.voices else &.{};
    if (engine_voices.len > 0) {
        // the engine's own catalog answers by index: inline objects and admin edits the store never reaches it
        if (voice.inline_voices or voice.manage) return error.EngineOwnedVoices;
        // the server's own copy: an idle unload frees the engine's names while /v1/voices still lists them
        const voices = try srv.arena.allocator().alloc(@import("drama_voice.zig").Voice, engine_voices.len);
        for (engine_voices, voices) |name, *v| v.* = .{ .name = try srv.arena.allocator().dupe(u8, name), .references = &.{} };
        srv.engine_voices = voices;
        try srv.voices.swap(voices);
    } else if (voice.dir) |dir| try srv.voices.load(dir);
    _ = try srv.voices.select(srv.arena.allocator(), voice.default, "default");
}

/// The `/v1/voices` reply.
pub fn listing(srv: *server.Server, a: std.mem.Allocator) !@import("json").Value {
    return srv.voices.listing(a);
}
