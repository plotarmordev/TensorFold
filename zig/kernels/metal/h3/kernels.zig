//! MiniMax H3's Metal kernels: three sources joined and compiled at load, after the prefill kernels' gemm_nax.
pub const source = @embedFile("h3_rows.metal") ++ @embedFile("h3_routing.metal") ++ @embedFile("h3_attention.metal");
/// H3's own products, after `source`: they need gemm_nax's fragments ahead of them.
pub const gemm_source = @embedFile("h3_gemm.metal");
/// MLX's bf16 GEMM, x w^T with N a multiple of 128 and M a multiple of 64 (whole) or not (ragged).
pub const gemm_whole = "custom_kernel_tf_gemm_nax_bf16_n_t_t_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";
pub const gemm_ragged = "custom_kernel_tf_gemm_nax_bf16_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";
/// H3's products at 64 x 128 a threadgroup and 32 x 64 a warp.
pub const gemm_32x64 = "h3_gemm_32x64";
pub const gemm_32x64_ragged = "h3_gemm_32x64_ragged";
/// The same products on int8 operands quantized by h3_quantize_rows, the weights packed (`packedAt`).
pub const gemm_32x64_i8 = "h3_gemm_32x64_i8";

/// h3_gemm.metal's h3_packed: weight (r, c) of an N x K int8 matrix in the order h3_gemm_32x64_i8 reads it.
pub fn packedAt(r: usize, c: usize, k: usize) usize {
    const x = c & 15;
    const y = r & 15;
    const lane = (x & 8) | (x & 4) >> 2 | (y & 3) << 1 | (y & 4) << 2;
    return (r >> 5) * 32 * k + (c >> 4) * 512 + lane * 16 + (r & 16) / 2 + (y & 8) / 2 + (x & 3);
}

pub const names = [_][:0]const u8{ "h3_rows_in", "h3_norm", "h3_norm_quant", "h3_gate_add", "h3_gate_mix", "h3_swiglu", "h3_head_norm", "h3_euler", "h3_heads", "h3_pool", "h3_tile_scores", "h3_topk", "h3_coarse", "h3_attention", "h3_far_pool", "h3_modulation", gemm_32x64, gemm_32x64_ragged, gemm_32x64_i8, "h3_quantize_rows" };
