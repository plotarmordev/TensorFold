//! Device regression of the rows outside the blocks (h3_rows_in, h3_rows_out, h3_euler through `Rows`) against
//! rows_reference's ordered host sums over two steps: text, a reference and a ragged hidden width.
const std = @import("std");
const h3 = @import("cuda.zig");
const rows = @import("rows_reference.zig");
const picture = @import("picture.zig");
const bf = @import("bf16_round");

pub fn run(a: std.mem.Allocator) !void {
    const hidden = 300; // not a multiple of the 32 lanes or 256 threads
    const width = 96;
    const text = 2;
    const held = 3;
    const target = 5;
    const count = text + held + target;
    const steps = 2;
    const model = try h3.Model.setup(a);
    defer model.deinit();
    model.hidden = hidden;

    const values = try a.alloc(f32, held * width + target * width + 2 * hidden * width + steps * width * hidden + steps * width);
    defer a.free(values);
    for (values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 7919 % 61)) - 30)) / 64;
    var at: usize = 0;
    const take = struct {
        fn f(v: []f32, from: *usize, n: usize) []f32 {
            defer from.* += n;
            return v[from.*..][0..n];
        }
    }.f;
    const anchors = take(values, &at, held * width);
    const noise = take(values, &at, target * width);
    const w_in = take(values, &at, hidden * width);
    const b_in = take(values, &at, hidden);
    const w_out = take(values, &at, steps * width * hidden);
    const b_out = take(values, &at, steps * width);
    var text_rows: [text * hidden]u16 = undefined;
    var final_norm: [hidden]u16 = undefined;
    var final: [2 * 2 * hidden]u16 = undefined;
    for (&text_rows, 0..) |*v, i| v.* = bf.bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8);
    for (&final_norm, 0..) |*v, i| v.* = bf.bits(1 + @as(f32, @floatFromInt(i % 5)) / 16);
    for (&final, 0..) |*v, i| v.* = bf.bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 16);
    var line_of: [steps * count]i32 = undefined;
    for (&line_of, 0..) |*v, i| v.* = @intCast(i % 2);
    const deltas = [steps]f32{ -0.375, -0.2109375 };

    var device = try h3.Rows.init(model, .{
        .text = std.mem.sliceAsBytes(&text_rows),
        .anchors = std.mem.sliceAsBytes(anchors),
        .noise = std.mem.sliceAsBytes(noise),
        .w_in = std.mem.sliceAsBytes(w_in),
        .b_in = std.mem.sliceAsBytes(b_in),
        .w_out = std.mem.sliceAsBytes(w_out),
        .b_out = std.mem.sliceAsBytes(b_out),
        .final_norm = std.mem.sliceAsBytes(&final_norm),
        .final = std.mem.sliceAsBytes(&final),
        .line_of = std.mem.sliceAsBytes(&line_of),
        .text_rows = text,
        .held = held,
        .target = target,
        .width = width,
    });
    defer device.deinit();
    var x = try @import("cuda").DeviceBuffer.alloc(&model.driver, count * hidden * 2);
    defer x.free();

    // the host path the image tool ran before: rows in, the blocks (here none), the final norm, the head, Euler
    var video: [target * width]f32 = undefined;
    @memcpy(&video, noise);
    var stream: [count * hidden]u16 = undefined;
    var normed: [count * hidden]u16 = undefined;
    var velocity: [target * width]f32 = undefined;
    var got: [count * hidden]u16 = undefined;
    for (0..steps) |s| {
        @memcpy(stream[0 .. text * hidden], &text_rows);
        rows.rowsIn(anchors, w_in, b_in, &stream, text, held, width, hidden);
        rows.rowsIn(&video, w_in, b_in, &stream, text + held, target, width, hidden);
        try device.in(x.ptr);
        try model.stream.synchronize();
        try x.download(0, std.mem.sliceAsBytes(&got));
        if (!std.mem.eql(u16, &got, &stream)) return error.H3RowsInMismatch;
        const line = line_of[s * count ..][0..count];
        rows.finalNorm(&stream, &final_norm, &final, line, &normed, hidden, h3.eps);
        rows.rowsOut(&normed, w_out[s * width * hidden ..][0 .. width * hidden], b_out[s * width ..][0..width], &velocity, text + held, target, hidden, width);
        picture.euler(&video, &velocity, deltas[s]);
        try device.out(x.ptr, s, deltas[s]);
    }
    var latent: [target * width]f32 = undefined;
    try device.latent(&latent);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(&latent), std.mem.sliceAsBytes(&video))) return error.H3RowsOutMismatch;
    std.debug.print("H3 CUDA rows: rows in, final norm, head and Euler over two steps match the ordered host sums\n", .{});
}
