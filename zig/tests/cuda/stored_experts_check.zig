//! The stored experts' checkpoint checks: one-hot rows read each decoded weight back, random rows the outputs, 64-row items the 16-row items' bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const cs = @import("contract_stored_experts.zig");
const st = @import("stored_experts_tests.zig");

const Gpu = check.Gpu;
const grouped = cuda.grouped;
const ex_ = cuda.experts;
const mx = cuda.experts_mx;
const c = core.contract.spec;

const chunk = 16; // experts a pass of the one-hot check

/// Every weight of both matrices of all 128 experts through the fp32 down kernel: x one-hot, so out = w (x factor).
pub fn decoded(gpu: Gpu, s: st.Stored, layer: usize) !void {
    const a = gpu.gpa;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k4 = try ex_.Nvfp4Experts.load(gpu.d, sms);
    defer k4.unload();
    var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
    defer k8.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    for ([_]st.Matrix{ s.w13, s.w2 }, [_][]const u8{ "w13", "w2" }) |m, name| {
        const n = m.n;
        const k = m.k;
        const words = (if (s.nv) ex_.Nvfp4Experts.bytes(1, n, k, 1) else mx.Mxfp8Experts.bytes(1, n, k, 1)) / 4;
        const packed_ = try a.alloc(u32, st.experts * words);
        defer a.free(packed_);
        for (0..st.experts) |e| {
            const o = packed_[e * words ..][0..words];
            if (s.nv) ex_.Nvfp4Experts.packOne(o, m.expertCodes(e), m.expertScales(e), n, k) else mx.Mxfp8Experts.packOne(o, m.expertCodes(e), m.expertScales(e), n, k);
        }
        var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_));
        defer w.free();
        var fs = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&m.factor));
        defer fs.free();
        // x: the k x k identity; pair i * chunk + j reads row i (slots = chunk) and expert base + j
        const eye = try a.alloc(u16, k * k);
        defer a.free(eye);
        @memset(eye, 0);
        for (0..k) |i| eye[i * k + i] = 0x3F80;
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(eye));
        defer x.free();
        const pairs = k * chunk;
        var pb = try st.PlanBufs.init(gpu, pairs, 16);
        defer pb.deinit();
        const ids = try a.alloc(i32, pairs);
        defer a.free(ids);
        var dids = try cuda.DeviceBuffer.alloc(gpu.d, pairs * 4);
        defer dids.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, pairs * n * 4);
        defer out.free();
        const got = try a.alloc(f32, pairs * n);
        defer a.free(got);
        var differ: usize = 0;
        var zeros: usize = 0;
        var base: usize = 0;
        while (base < st.experts) : (base += chunk) {
            for (ids, 0..) |*v, p| v.* = @intCast(base + p % chunk);
            try dids.upload(0, std.mem.sliceAsBytes(ids));
            try router.route(stream, dids.ptr, pairs, st.experts, 16, pb.plan);
            try out.fill8(0xff, stream.handle);
            const in: ex_.Rows = .{ .x = x.ptr, .stride = k, .slots = chunk };
            const nb = n / 32;
            if (s.nv) {
                // the existing fp32 down (decode form) read with token rows: Nvfp4Experts.run passes slots only to up
                var args: cuda.Args = .{};
                args.add(in.x);
                for ([_]usize{ k, chunk }) |v| args.add(@as(i32, @intCast(v)));
                args.add(w.ptr);
                args.add(fs.ptr);
                for ([_]usize{ k / 32, nb }) |v| args.add(@as(i32, @intCast(v)));
                for ([_]u64{ pb.plan.items, pb.plan.counts, pb.plan.members, out.ptr }) |v| args.add(v);
                args.add(@as(i32, @intCast(n)));
                args.add(@as(f32, 0));
                args.add(@as(i32, -1));
                const grid = @min((pb.items * nb + k4.warps - 1) / k4.warps, k4.resident[1]);
                try cuda.launch.launch(k4.fns[1], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = @intCast(k4.warps * 32) } }, stream, &args);
            } else {
                var args = mx.argsOf(in, true, w.ptr, null, k / 32, nb, pb.plan, out.ptr, n, .{ 0, 0 }, -1);
                const grid = @min((pb.items * nb + 3) / 4, k8.resident[1]);
                try cuda.launch.launch(k8.fns[1], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 128 } }, stream, &args);
            }
            try stream.synchronize();
            try out.download(0, std.mem.sliceAsBytes(got));
            for (0..k) |i| for (0..chunk) |j| for (0..n) |row| {
                const want = m.f32At(s.nv, base + j, row, i);
                const v = got[(i * chunk + j) * n + row];
                if (v != want) differ += 1 else if (@as(u32, @bitCast(v)) != @as(u32, @bitCast(want))) zeros += 1;
            };
        }
        try check.expect(differ == 0, "layer {d} {s}: {d} of {d} decoded weights differ from the host decode", .{ layer, name, differ, st.experts * n * k });
        check.pass("layer {d} {s} ({s}): all {d} weights of 128 experts [{d}, {d}] equal the host decode ({d} zeros differ in sign only)", .{ layer, name, if (s.nv) "NVFP4" else "MXFP8", st.experts * n * k, n, k, zeros });
    }
}

/// Gate-up then down on 4 tokens (decode form) and 64 (prompt form), each output within the contract's fp32 bound.
pub fn outputs(gpu: Gpu, s: st.Stored, layer: usize) !void {
    const a = gpu.gpa;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k4 = try ex_.Nvfp4Experts.load(gpu.d, sms);
    defer k4.unload();
    var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
    defer k8.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var bufs: [4]cuda.DeviceBuffer = undefined;
    const l = try s.upload(gpu, &bufs);
    defer for (&bufs) |*b| b.free();
    const gx: ex_.Grouped = .{ .nvfp4 = &k4, .mxfp8 = &k8 };
    for ([_]usize{ 4, 64 }) |tokens| {
        const prompt = tokens > 4;
        const pairs = tokens * st.top;
        const tile: usize = if (prompt) 64 else 16;
        var prng = std.Random.DefaultPrng.init(0x5eed + tokens + layer);
        const xs = try a.alloc(u16, tokens * st.hidden);
        defer a.free(xs);
        for (xs) |*v| v.* = c.toBf16((prng.random().float(f32) * 2 - 1) * 2);
        const ids = try st.picks(a, tokens, 0xc3 + tokens);
        defer a.free(ids);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs));
        defer dx.free();
        var dids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids));
        defer dids.free();
        var pb = try st.PlanBufs.init(gpu, pairs, tile);
        defer pb.deinit();
        try router.route(stream, dids.ptr, pairs, st.experts, tile, pb.plan);
        var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * st.width * 2);
        defer act.free();
        const yw: usize = if (prompt) 2 else 4;
        var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * st.hidden * yw);
        defer y.free();
        try act.fill8(0xff, stream.handle);
        try y.fill8(0xff, stream.handle);
        const up_in: ex_.Rows = .{ .x = dx.ptr, .stride = st.hidden, .slots = st.top };
        const down_in: ex_.Rows = .{ .x = act.ptr, .stride = st.width };
        if (prompt) {
            try gx.prompt(stream, .up, up_in, l, pb.plan, act.ptr, pb.items, pairs, tile, -1);
            try gx.prompt(stream, .down_bf16, down_in, l, pb.plan, y.ptr, pb.items, pairs, tile, -1);
        } else {
            try gx.decode(stream, .up, up_in, l, pb.plan, act.ptr, pb.items, pairs, tile, -1);
            try gx.decode(stream, .down_f32, down_in, l, pb.plan, y.ptr, pb.items, pairs, tile, -1);
        }
        try stream.synchronize();
        const ga = try a.alloc(u16, pairs * st.width);
        defer a.free(ga);
        try act.download(0, std.mem.sliceAsBytes(ga));
        const gy = try a.alloc(u8, pairs * st.hidden * yw);
        defer a.free(gy);
        try y.download(0, gy);
        const fmt = if (s.nv) cs.Format.nvfp4 else cs.Format.mxfp8;
        const up_spec = cs.spec(.{ .name = "up", .format = fmt, .kind = .up, .prompt = prompt });
        const down_spec = cs.spec(.{ .name = "down", .format = fmt, .kind = if (prompt) .down_bf16 else .down_f32, .prompt = prompt });
        var worst: [2]f64 = .{ 0, 0 };
        var bad: [2]usize = .{ 0, 0 };
        for (0..pairs) |p| {
            const e: usize = @intCast(ids[p]);
            const xr = xs[(p / st.top) * st.hidden ..][0..st.hidden];
            for (0..st.width) |j| {
                var sum: [2]f64 = .{ 0, 0 };
                var abs: [2]f64 = .{ 0, 0 };
                for (0..2) |m| for (0..st.hidden) |i| {
                    const t = c.fromBf16(xr[i]) * s.w13.f64At(s.nv, e, 2 * j + m, i);
                    sum[m] += t;
                    abs[m] += @abs(t);
                };
                const r = cs.softcap(sum[0], sum[1]);
                const mag = @abs(r[1]) * abs[0] + @abs(r[2]) * abs[1];
                const err = @abs(@as(f64, c.fromBf16(ga[p * st.width + j])) - r[0]);
                const b = c.bound(up_spec, .{ .rows = tokens, .n = st.width, .k = st.hidden }, r[0], mag);
                worst[0] = @max(worst[0], err / b);
                if (!(err <= b)) bad[0] += 1;
            }
            const ar = ga[p * st.width ..][0..st.width];
            for (0..st.hidden) |row| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..st.width) |i| {
                    const t = c.fromBf16(ar[i]) * s.w2.f64At(s.nv, e, row, i);
                    sum += t;
                    abs += @abs(t);
                }
                const o = p * st.hidden + row;
                const v: f64 = if (prompt) c.fromBf16(std.mem.bytesToValue(u16, gy[o * 2 ..][0..2])) else std.mem.bytesToValue(f32, gy[o * 4 ..][0..4]);
                const b = c.bound(down_spec, .{ .rows = tokens, .n = st.hidden, .k = st.width }, sum, abs);
                worst[1] = @max(worst[1], @abs(v - sum) / b);
                if (!(@abs(v - sum) <= b)) bad[1] += 1;
            }
        }
        try check.expect(bad[0] == 0 and bad[1] == 0, "layer {d} {d} tokens: {d} gate-up and {d} down outputs outside the fp32 bound", .{ layer, tokens, bad[0], bad[1] });
        check.pass("layer {d} {s} {d} tokens ({s}): {d} gate-up and {d} down outputs within the fp32 bound; worst err/bound {d:.3} and {d:.3}", .{ layer, if (s.nv) "NVFP4" else "MXFP8", tokens, if (prompt) "prompt form" else "decode form", pairs * st.width, pairs * st.hidden, worst[0], worst[1] });
    }
}

/// Elements of a 64-row plan's output against a 16-row plan's: how many differ, how many of each still hold the 0xff fill, the first that differs.
const Tally = struct { differ: usize = 0, poison: [2]usize = .{ 0, 0 }, first: ?usize = null };

fn tally(new: []const u8, old: []const u8, wide: usize) Tally {
    var t: Tally = .{};
    for (0..old.len / wide) |i| {
        const p = new[i * wide ..][0..wide];
        const q = old[i * wide ..][0..wide];
        if (!std.mem.eql(u8, p, q)) {
            if (t.first == null) t.first = i;
            t.differ += 1;
        }
        t.poison[0] += @as(usize, @intFromBool(std.mem.allEqual(u8, p, 0xff)));
        t.poison[1] += @as(usize, @intFromBool(std.mem.allEqual(u8, q, 0xff)));
    }
    return t;
}

/// Element `i` of a `wide`-byte output, little endian, for a failure's line.
fn word(bytes: []const u8, i: usize, wide: usize) u32 {
    const at = i * wide;
    return if (wide == 2) std.mem.readInt(u16, bytes[at..][0..2], .little) else std.mem.readInt(u32, bytes[at..][0..4], .little);
}

/// 64 tokens sharing one top-8 pick on plans of 16 rows (the released decode form) and of 64 (Mini's n > 16 decode): gate-up BF16 and
/// down FP32 byte for byte, the 16-row plan's outputs read back before the 64-row plan runs; the items each plan holds on the device
/// beside the host's launch bound (maxItems over every expert, what the kernels are launched with).
pub fn rows64(gpu: Gpu, s: st.Stored, layer: usize) !void {
    const a = gpu.gpa;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k4 = try ex_.Nvfp4Experts.load(gpu.d, sms);
    defer k4.unload();
    var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
    defer k8.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var bufs: [4]cuda.DeviceBuffer = undefined;
    const l = try s.upload(gpu, &bufs);
    defer for (&bufs) |*b| b.free();
    const gx: ex_.Grouped = .{ .nvfp4 = &k4, .mxfp8 = &k8 };
    const tokens = 64;
    const pairs = tokens * st.top;
    var prng = std.Random.DefaultPrng.init(0x7ed0 + layer);
    const xs = try a.alloc(u16, tokens * st.hidden);
    defer a.free(xs);
    for (xs) |*v| v.* = c.toBf16((prng.random().float(f32) * 2 - 1) * 2);
    const one = try st.picks(a, 1, 0x64);
    defer a.free(one);
    const ids = try a.alloc(i32, pairs);
    defer a.free(ids);
    var chosen: [st.experts]bool = @splat(false);
    for (one) |e| chosen[@intCast(e)] = true;
    for (0..tokens) |t| @memcpy(ids[t * st.top ..][0..st.top], one);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs));
    defer dx.free();
    var dids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids));
    defer dids.free();
    const act_len = pairs * st.width * 2;
    const y_len = pairs * st.hidden * 4;
    const act16 = try a.alloc(u8, act_len);
    defer a.free(act16);
    const y16 = try a.alloc(u8, y_len);
    defer a.free(y16);
    const act64 = try a.alloc(u8, act_len);
    defer a.free(act64);
    const y64 = try a.alloc(u8, y_len);
    defer a.free(y64);
    var bound: [2]usize = undefined; // the host's launch bound: maxItems over all 128 experts
    var held: [2]usize = undefined; // counts[0]: the items the plan holds on the device
    for ([_]usize{ 16, 64 }, 0..) |tile, ti| {
        const per: usize = tokens / tile; // items a picked expert's rows fill
        var pb = try st.PlanBufs.init(gpu, pairs, tile);
        defer pb.deinit();
        bound[ti] = pb.items;
        try router.route(stream, dids.ptr, pairs, st.experts, tile, pb.plan);
        try stream.synchronize();
        var counts: [2]i32 = undefined; // items the plan holds, experts with rows
        try cuda.memory.download(gpu.d, pb.plan.counts, std.mem.sliceAsBytes(&counts));
        held[ti] = @intCast(@max(@as(i32, 0), @min(counts[0], @as(i32, @intCast(bound[ti])))));
        const raw = try a.alloc(i32, 3 * held[ti]);
        defer a.free(raw);
        try cuda.memory.download(gpu.d, pb.plan.items, std.mem.sliceAsBytes(raw));
        var items_of: [st.experts]usize = @splat(0);
        var rows_of: [st.experts]usize = @splat(0);
        var prefix: [st.experts]usize = undefined;
        var seen: usize = 0;
        for (chosen, 0..) |on, e| {
            prefix[e] = seen;
            if (on) seen += tokens;
        }
        var bad: usize = 0; // items outside their expert's rows, in plan order
        for (0..held[ti]) |i| {
            const e = raw[3 * i];
            if (e < 0 or e >= st.experts or raw[3 * i + 1] < 0 or raw[3 * i + 2] < 0) {
                bad += 1;
                continue;
            }
            const eu: usize = @intCast(e);
            const first: usize = @intCast(raw[3 * i + 1]);
            const rows: usize = @intCast(raw[3 * i + 2]);
            if (!chosen[eu] or rows != tile or first != prefix[eu] + rows_of[eu]) bad += 1;
            items_of[eu] += 1;
            rows_of[eu] += rows;
        }
        var off: usize = 0; // experts without their rows in whole `tile`-row items
        for (chosen, 0..) |on, e| {
            if (rows_of[e] != (if (on) tokens else 0) or items_of[e] != (if (on) per else 0)) off += 1;
        }
        try check.expect(bound[ti] == grouped.maxItems(pairs, st.experts, tile) and held[ti] == st.top * per and counts[1] == st.top and bad == 0 and off == 0, "layer {d}, tile {d}: launch bound {d}, {d} items held over {d} experts, {d} misplaced, {d} experts not covered", .{ layer, tile, bound[ti], held[ti], counts[1], bad, off });
        var act = try cuda.DeviceBuffer.alloc(gpu.d, act_len);
        defer act.free();
        var y = try cuda.DeviceBuffer.alloc(gpu.d, y_len);
        defer y.free();
        try act.fill8(0xff, stream.handle);
        try y.fill8(0xff, stream.handle);
        const up_in: ex_.Rows = .{ .x = dx.ptr, .stride = st.hidden, .slots = st.top };
        const down_in: ex_.Rows = .{ .x = act.ptr, .stride = st.width };
        try gx.decode(stream, .up, up_in, l, pb.plan, act.ptr, bound[ti], pairs, tile, -1);
        try gx.decode(stream, .down_f32, down_in, l, pb.plan, y.ptr, bound[ti], pairs, tile, -1);
        try stream.synchronize();
        try act.download(0, if (ti == 0) act16 else act64);
        try y.download(0, if (ti == 0) y16 else y64);
    }
    for ([_][2][]u8{ .{ act64, act16 }, .{ y64, y16 } }, [2]usize{ 2, 4 }, [2]usize{ st.width, st.hidden }, [_][]const u8{ "gate-up BF16", "down FP32" }) |o, w, n, name| {
        const t = tally(o[0], o[1], w);
        const first = t.first orelse 0;
        try check.expect(t.differ == 0 and t.poison[0] == 0 and t.poison[1] == 0, "layer {d} {s}: {d} of {d} outputs of the 64-row items differ from the 16-row items, {d} and {d} still hold the 0xff fill; first at pair {d} (expert {d}), column {d}: {x} for {x}", .{ layer, name, t.differ, o[1].len / w, t.poison[0], t.poison[1], first / n, ids[first / n], first % n, word(o[0], first, w), word(o[1], first, w) });
    }
    check.pass("layer {d} {s}: 64 tokens sharing one top-{d} pick ({d} pairs over {d} experts): launch bound {d} -> {d} (maxItems over {d} experts), device items {d} -> {d}, {d} -> {d} an expert at {d} rows; {d} gate-up BF16 and {d} down FP32 outputs of the 64-row items equal the 16-row items byte for byte", .{ layer, if (s.nv) "NVFP4" else "MXFP8", st.top, pairs, st.top, bound[0], bound[1], st.experts, held[0], held[1], tokens / 16, tokens / 64, tokens, pairs * st.width, pairs * st.hidden });
}

/// MXFP8's E8M0 edges through tf_mxfp8_down_f32: scale byte 0 is 2^-127 (not 0, not 2^-126).
pub fn e8m0Edges(gpu: Gpu) !void {
    const a = gpu.gpa;
    const n = 32;
    const k = 32;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
    defer k8.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const edges = [_]u8{ 0, 1, 2, 126, 127, 128, 200, 240 }; // 448 x 2^113 still fits fp32
    const values = [_]u8{ 0x38, 0x7E, 0x30, 0xB8 }; // 1, 448, 0.5, -1: each times 2^-127 stays a bf16 (>= 2^-133)
    var codes: [n * k]u8 = undefined;
    var scales: [n]u8 = undefined;
    for (0..n) |r| {
        scales[r] = edges[r % edges.len];
        for (0..k) |i| codes[r * k + i] = values[(r / edges.len + i) % values.len];
    }
    var blocks: [mx.Mxfp8Experts.words]u32 = undefined;
    mx.Mxfp8Experts.packOne(&blocks, &codes, &scales, n, k);
    var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&blocks));
    defer w.free();
    var eye: [k * k]u16 = @splat(0);
    for (0..k) |i| eye[i * k + i] = 0x3F80;
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&eye));
    defer x.free();
    var ids: [k]i32 = @splat(0);
    var dids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&ids));
    defer dids.free();
    var pb = try st.PlanBufs.init(gpu, k, 16);
    defer pb.deinit();
    try router.route(stream, dids.ptr, k, st.experts, 16, pb.plan);
    var out = try cuda.DeviceBuffer.alloc(gpu.d, k * n * 4);
    defer out.free();
    try out.fill8(0xff, stream.handle);
    const l: ex_.Layer = .{ .format = .mxfp8, .up = 0, .down = w.ptr, .width = k, .dims = n, .experts = st.experts, .caps = .{ 4, 25 } };
    const gx: ex_.Grouped = .{ .mxfp8 = &k8 };
    try gx.decode(stream, .down_f32, .{ .x = x.ptr, .stride = k }, l, pb.plan, out.ptr, pb.items, k, 16, -1);
    try stream.synchronize();
    const got = try a.alloc(f32, k * n);
    defer a.free(got);
    try out.download(0, std.mem.sliceAsBytes(got));
    for (0..k) |i| for (0..n) |r| {
        const want: f32 = @floatCast(core.expert_decode.mxfp8(&codes, &scales, k, r, i));
        try check.expect(got[i * n + r] == want, "E8M0 byte {d}, code {x}: {e}, host {e}", .{ scales[r], codes[r * k + i], got[i * n + r], want });
    };
    try check.expect(core.expert_decode.e8m0(0) == 0x1p-127, "host E8M0 byte 0", .{});
    check.pass("MXFP8 E8M0 bytes {any} decode exactly on the GPU (byte 0 = 2^-127)", .{edges});
}
