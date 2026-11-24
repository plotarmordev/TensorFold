//! Inkling 2 Mini's slot snapshots: every visit's valid K/V rows and all convolution states, saved and restored.
const cuda = @import("cuda");
const Model = @import("loop_engine.zig").Model;

pub const Saved = struct {
    at: u32,
    visits: u32,
    passes: u32,
    hidden: u32,
    buf: cuda.DeviceBuffer,
};

/// Compact bytes for all valid K/V rows and every convolution state of all 48 visits.
pub fn snapshotBytes(m: *const Model, at: usize) u64 {
    var bytes = m.layout.bytes - m.layout.at(0).conv_offset;
    for (0..m.w.p.n_visits) |v| {
        const layer = m.w.p.at(m.w.p.visits[v].layer);
        const global = layer.kind == .global;
        const count = if (global) at else @min(at, m.w.c.text.window - 1);
        const row = @as(u64, layer.heads.kv) * layer.heads.dim * 2;
        bytes += 2 * count * row;
    }
    return bytes;
}

fn copyState(m: *const Model, s: cuda.Stream, saved: *const Saved, out: bool) !void {
    const at: usize = saved.at;
    var off: usize = 0;
    for (0..m.w.p.n_visits) |v| {
        const visit = m.w.p.visits[v];
        const layer = m.w.p.at(visit.layer);
        const geometry = m.layerGeometry(visit.layer);
        const rb = @as(usize, layer.heads.dim) * 2; // the caches are head-major: [kv head][ring][dim]
        const global = layer.kind == .global;
        const count = if (global) at else @min(at, m.w.c.text.window - 1);
        const first = at - count;
        const cache = m.cacheOf(visit.cache_slot);
        for ([2]u64{ cache.k, cache.v }) |k_or_v| for (0..layer.heads.kv) |head| {
            const base = k_or_v + head * geometry.ring * rb;
            var pos = first;
            while (pos < at) {
                const row = pos % geometry.ring;
                const n = @min(at - pos, geometry.ring - row);
                const dev = base + row * rb;
                const src = if (out) dev else saved.buf.ptr + off;
                const dst = if (out) saved.buf.ptr + off else dev;
                try m.copy(s, dst, src, n * rb);
                off += n * rb;
                pos += n;
            }
        };
    }
    const conv = m.slotBase() + m.layout.at(0).conv_offset;
    const bytes = m.layout.bytes - m.layout.at(0).conv_offset;
    try m.copy(s, if (out) saved.buf.ptr + off else conv, if (out) conv else saved.buf.ptr + off, bytes);
}

pub fn save(m: *const Model, s: cuda.Stream, at: u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const gpa = m.gpa;
    const saved = try gpa.create(Saved);
    errdefer gpa.destroy(saved);
    const lp = m.w.c.text.loop.?;
    saved.* = .{ .at = at, .visits = m.w.p.n_visits, .passes = lp.passes, .hidden = m.w.c.text.hidden, .buf = try cuda.DeviceBuffer.alloc(m.d, @intCast(@max(snapshotBytes(m, at), 256))) };
    errdefer saved.buf.free();
    try copyState(m, s, saved, true);
    return saved;
}

pub fn restore(m: *const Model, s: cuda.Stream, saved: *const Saved) !void {
    const lp = m.w.c.text.loop.?;
    if (saved.visits != m.w.p.n_visits or saved.passes != lp.passes or saved.hidden != m.w.c.text.hidden)
        return error.StateOfAnotherModel;
    if (saved.at == 0 or saved.at > m.context or saved.buf.len < snapshotBytes(m, saved.at)) return error.BadLoopSnapshot;
    try copyState(m, s, saved, false);
}

pub fn drop(m: *const Model, s: cuda.Stream, saved: *Saved) void {
    s.synchronize() catch {};
    saved.buf.free();
    m.gpa.destroy(saved);
}
