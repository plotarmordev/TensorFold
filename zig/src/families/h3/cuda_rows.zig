//! A picture's rows outside the blocks on the device: the stream's rows in, the final norm, the step's head and the
//! Euler update, each in rows_reference's sum order, so the latent is the host path's bit for bit. The stream stays
//! on the device across steps; only the latent's rows come back.
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const Model = h3.Model;

pub const Rows = struct {
    model: *Model,
    /// The text's bf16 stream rows, copied in ahead of every step.
    text: cuda.DeviceBuffer,
    /// The references' f32 rows and the video's, the step's velocity.
    anchors: cuda.DeviceBuffer,
    video: cuda.DeviceBuffer,
    velocity: cuda.DeviceBuffer,
    /// video_patch_proj's weight and bias; the steps' heads (steps, width, hidden) and biases; the final norm's
    /// weight, the final table and each step's row lines in it.
    w_in: cuda.DeviceBuffer,
    b_in: cuda.DeviceBuffer,
    w_out: cuda.DeviceBuffer,
    b_out: cuda.DeviceBuffer,
    final_norm: cuda.DeviceBuffer,
    final: cuda.DeviceBuffer,
    line_of: cuda.DeviceBuffer,
    text_rows: usize,
    held: usize,
    target: usize,
    rows: usize,
    width: usize,

    pub const Host = struct {
        text: []const u8,
        anchors: []const u8,
        noise: []const u8,
        w_in: []const u8,
        b_in: []const u8,
        w_out: []const u8,
        b_out: []const u8,
        final_norm: []const u8,
        final: []const u8,
        line_of: []const u8,
        text_rows: usize,
        held: usize,
        target: usize,
        width: usize,
    };

    pub fn init(model: *Model, h: Host) !Rows {
        if (h.width > 128) return error.BadGeometry; // h3_rows_in's shared row
        var self: Rows = undefined;
        self.model = model;
        const fields = .{ "text", "anchors", "video", "w_in", "b_in", "w_out", "b_out", "final_norm", "final", "line_of" };
        const bytes = [_][]const u8{ h.text, h.anchors, h.noise, h.w_in, h.b_in, h.w_out, h.b_out, h.final_norm, h.final, h.line_of };
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| if (i < made) @field(self, name).free();
        inline for (fields, 0..) |name, i| {
            @field(self, name) = try cuda.DeviceBuffer.fromHost(&model.driver, bytes[i]);
            made += 1;
        }
        self.velocity = try cuda.DeviceBuffer.alloc(&model.driver, h.noise.len);
        self.text_rows = h.text_rows;
        self.held = h.held;
        self.target = h.target;
        self.rows = h.text_rows + h.held + h.target;
        self.width = h.width;
        return self;
    }

    pub fn deinit(self: *Rows) void {
        inline for (.{ "text", "anchors", "video", "velocity", "w_in", "b_in", "w_out", "b_out", "final_norm", "final", "line_of" }) |name| @field(self, name).free();
    }

    /// The stream x (rows, hidden) for a step: the text's rows, then the references' and the video's projected.
    pub fn in(self: *const Rows, x: u64) !void {
        const m = self.model;
        try m.driver.check(m.driver.api.cuMemcpyDtoDAsync_v2(x, self.text.ptr, self.text.len, m.stream.handle), "cuMemcpyDtoDAsync");
        if (self.held > 0) try self.project(self.anchors.ptr, x, self.held, self.text_rows);
        try self.project(self.video.ptr, x, self.target, self.text_rows + self.held);
    }

    fn project(self: *const Rows, from: u64, x: u64, count: usize, first: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ from, self.w_in.ptr, self.b_in.ptr, x }) |p| args.add(p);
        for ([_]usize{ self.width, self.model.hidden, first }) |n| args.add(@as(c_int, @intCast(n)));
        try self.model.launch("h3_rows_in", .{ .x = @intCast(count) }, 256, 0, &args);
    }

    /// Step `s` after the blocks: the target's rows of x through the final norm and the step's head.
    /// An on-grid step reads head `s` (`first` = `last` = `s`, weight 1); an off-grid one blends the
    /// spanned distilled heads as mmh3's `step_heads` does (`picture.headSpan`): the run chains one
    /// launch a spanned head, `weights[0]` applied bare and each later one accumulate-and-scaled.
    pub fn velocityOf(self: *const Rows, x: u64, s: usize, first: usize, last: usize, weights: []const f32) !void {
        const m = self.model;
        const hidden = m.hidden;
        const head = self.width * hidden * 4;
        const line = self.line_of.ptr + s * self.rows * 4;
        // h3_rows_out applies scale only against an accumulator, so the first head's share
        // weights[0] needs the velocity zeroed first: every launch then runs ACC + scale · head.
        try m.driver.check(m.driver.api.cuMemsetD8Async(self.velocity.ptr, 0, self.velocity.len, m.stream.handle), "cuMemsetD8");
        for (first..last + 1, 0..) |k, w| {
            var args: cuda.Args = .{};
            for ([_]u64{ x, self.final_norm.ptr, self.final.ptr, line, self.w_out.ptr + k * head, self.b_out.ptr + k * self.width * 4, self.velocity.ptr, self.velocity.ptr }) |p| args.add(p);
            for ([_]usize{ hidden, self.width, self.text_rows + self.held }) |n| args.add(@as(c_int, @intCast(n)));
            args.add(h3.eps);
            args.add(weights[w]);
            try m.launch("h3_rows_out", .{ .x = @intCast(self.target) }, 256, @intCast(hidden * 4), &args);
        }
    }

    /// `out` for a step whose head span was resolved on the host: `first` head through `last` at
    /// `weights`, the velocity written, then the Euler update of `delta`.
    pub fn out(self: *const Rows, x: u64, s: usize, delta: f32, first: usize, last: usize, weights: []const f32) !void {
        try self.velocityOf(x, s, first, last, weights);
        var args: cuda.Args = .{};
        args.add(self.video.ptr);
        args.add(self.velocity.ptr);
        args.add(delta);
        args.add(@as(c_int, @intCast(self.target * self.width)));
        try self.model.launch("h3_euler", .{ .x = 512 }, 256, 0, &args);
    }


    /// video − sigma · velocity in place on the device: the draft's predicted clean latent at the end
    /// of the draft steps (`x − sigma · v` is `h3_euler`'s `video + delta · velocity` at delta −sigma),
    /// kept resident so the host enlargement downloads it once.
    pub fn clean(self: *const Rows, sigma: f32) !void {
        var args: cuda.Args = .{};
        args.add(self.video.ptr);
        args.add(self.velocity.ptr);
        args.add(-sigma);
        args.add(@as(c_int, @intCast(self.target * self.width)));
        try self.model.launch("h3_euler", .{ .x = 512 }, 256, 0, &args);
    }

    /// The video's rows after the last step, (target, width) f32.
    pub fn latent(self: *const Rows, into: []f32) !void {
        try self.model.stream.synchronize();
        try self.video.download(0, @import("std").mem.sliceAsBytes(into));
    }
};
