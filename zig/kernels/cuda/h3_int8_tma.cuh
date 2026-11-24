// MiniMax-H3-Image's int8 products on TMA and a persistent grid, ported from mmh3's int8_gemm.cu
// (mmh3-org/mmh3 9c0652f, MIT): TMA copies 128-byte K slices of both int8 operands into a two-stage
// ring under a 128B swizzle, one CTA per SM walks the output tiles in bands, eight warps multiply
// with mma.sync.m16n8k32.s8 into int32 accumulators, and the epilogue applies the row and column
// scales in registers before the single bf16 rounding — no int32 round trip through h3_dequantize.
// With SWIGLU the epilogue writes silu(gate) * up, [m, n / 2], for weights whose rows interleave a
// feature's gate and up rows pairwise (h3_swiglu_rows); h3_swiglu_int8's rounding: the scaled sums
// round to bf16 first, silu's value rounds to bf16, then the product rounds to bf16.
//
// Dropped from the port: the cp.async fallback (the picture tools run only where TMA exists), the
// low-rank adapter, the bias and the column-maxima tracking — no H3 picture projection uses them.
// The TMA machinery traps under a build for a pre-Hopper SM, which the host never launches.
#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace h3_int8 {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define H3_TMA_TRAP 1
#endif

constexpr int tma_block_k = 128;
constexpr int tma_stages = 2;
constexpr int tma_warps = 8;
constexpr int tma_group_m = 8; // tile rows a band walks before moving right (mmh3's default)

__device__ __forceinline__ uint32_t h3_shared_address(const void *pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void h3_barrier_init(uint32_t barrier, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier), "r"(count));
}

// Makes the initialized barriers visible to the copy engine.
__device__ __forceinline__ void h3_barrier_init_fence() {
#ifdef H3_TMA_TRAP
    __trap();
#else
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void h3_barrier_expect_bytes(uint32_t barrier, uint32_t bytes) {
#ifdef H3_TMA_TRAP
    __trap();
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier), "r"(bytes) : "memory");
#endif
}

__device__ __forceinline__ bool h3_barrier_try_wait(uint32_t barrier, uint32_t parity) {
#ifdef H3_TMA_TRAP
    __trap();
    return true;
#else
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred arrived;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 arrived, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, arrived;\n"
                 "}\n"
                 : "=r"(done)
                 : "r"(barrier), "r"(parity)
                 : "memory");
    return done != 0;
#endif
}

__device__ __forceinline__ void h3_barrier_wait(uint32_t barrier, uint32_t parity) {
    while (!h3_barrier_try_wait(barrier, parity)) {
    }
}

// Orders this CTA's reads of a stage before the copies that refill it.
__device__ __forceinline__ void h3_async_proxy_fence() {
#ifdef H3_TMA_TRAP
    __trap();
#else
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

// The destination is this CTA's shared memory, not a cluster's: on sm_120 a copy into shared::cluster
// is a system call whose first launch takes 1.5 GiB of the device.
__device__ __forceinline__ void h3_copy_tile(uint32_t destination, const CUtensorMap *map, uint32_t barrier, int column, int row) {
#ifdef H3_TMA_TRAP
    __trap();
#else
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(destination),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(column), "r"(row), "r"(barrier)
                 : "memory");
#endif
}

__device__ __forceinline__ void h3_load_matrix_x4(uint32_t (&registers)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3])
                 : "r"(address));
}

__device__ __forceinline__ void h3_mma_s8(int32_t (&accumulator)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, "
                 "%7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0]), "+r"(accumulator[1]), "+r"(accumulator[2]), "+r"(accumulator[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The TMA 128B swizzle stores 16-byte chunk c of a 128-byte row r at chunk c ^ (r % 8), so every
// ldmatrix phase touches eight distinct bank groups.
__device__ __forceinline__ uint32_t h3_swizzled_offset(int row, int chunk) {
    return row * 128 + (((chunk & 7) ^ (row & 7)) << 4);
}

template <int BLOCK_M, int BLOCK_N, int WARPS_M> struct GemmConfig {
    static constexpr int block_m = BLOCK_M;
    static constexpr int block_n = BLOCK_N;
    static constexpr int warps_n = tma_warps / WARPS_M;
    static constexpr int warp_m = BLOCK_M / WARPS_M;
    static constexpr int warp_n = BLOCK_N / warps_n;
    static constexpr int m_tiles = warp_m / 16;
    static constexpr int n_tiles = warp_n / 8;
    static constexpr int a_bytes = BLOCK_M * tma_block_k;
    static constexpr int stage_bytes = (BLOCK_M + BLOCK_N) * tma_block_k;
    // The stages, alignment slack for the swizzle, and a barrier and release count per stage.
    static constexpr int shared_bytes = tma_stages * stage_bytes + 1024 + tma_stages * 16;
    static_assert(BLOCK_M <= 256 && BLOCK_N <= 256, "one TMA box per operand and stage");
    static_assert(warp_m % 16 == 0 && warp_n % 32 == 0, "warp tiles hold whole groups of four MMA tiles");
};

struct TileGrid {
    int tiles_m;
    int tiles_n;
    int tiles;
    int k_blocks;

    // Tiles go down tma_group_m rows of tiles before moving right, so the CTAs in flight share rows
    // of activations and columns of weights in L2.
    __device__ void coordinates(int tile, int &tile_m, int &tile_n) const {
        const int tiles_per_group = tma_group_m * tiles_n;
        const int first_tile_m = tile / tiles_per_group * tma_group_m;
        const int group_height = min(tiles_m - first_tile_m, tma_group_m);
        tile_m = first_tile_m + tile % tiles_per_group % group_height;
        tile_n = tile % tiles_per_group / group_height;
    }
};

// Fills a stage with this CTA's `sequence`-th K block over all of its tiles.
template <typename Config>
__device__ void fill_stage(const TileGrid &grid, int sequence, uint32_t stage, uint32_t barrier, const CUtensorMap *map_a, const CUtensorMap *map_b) {
    const int tile = blockIdx.x + sequence / grid.k_blocks * gridDim.x;
    if (tile >= grid.tiles) {
        return;
    }
    const int block = sequence % grid.k_blocks;
    int tile_m, tile_n;
    grid.coordinates(tile, tile_m, tile_n);
    h3_barrier_expect_bytes(barrier, Config::stage_bytes);
    h3_copy_tile(stage, map_a, barrier, block * tma_block_k, tile_m * Config::block_m);
    h3_copy_tile(stage + Config::a_bytes, map_b, barrier, block * tma_block_k, tile_n * Config::block_n);
}

// Multiplies one stage into the warp's int32 accumulators: m16n8k32.s8 MMAs, 32 bytes of K a step.
template <typename Config>
__device__ __forceinline__ void multiply_stage(int32_t (&accumulators)[Config::m_tiles][Config::n_tiles][4], uint32_t stage_a, int warp_row, int warp_column, int lane) {
    const uint32_t stage_b = stage_a + Config::a_bytes;
    const int matrix = lane / 8;
    const int matrix_row = lane % 8;
    #pragma unroll
    for (int k_step = 0; k_step < tma_block_k / 32; k_step++) {
        uint32_t a_fragments[Config::m_tiles][4];
        uint32_t b_fragments[Config::n_tiles][2];
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            const int row = warp_row + m_tile * 16 + (matrix % 2) * 8 + matrix_row;
            h3_load_matrix_x4(a_fragments[m_tile], stage_a + h3_swizzled_offset(row, k_step * 2 + matrix / 2));
        }
        #pragma unroll
        for (int n_pair = 0; n_pair < Config::n_tiles / 2; n_pair++) {
            // Within each group of 32 weight rows, MMA column 2t + e of n-tile j reads row
            // 8t + 2((j + t) % 4) + e. Lane t of every quad then holds eight adjacent output
            // columns for a 16-byte store, and the eight rows of each ldmatrix phase still differ
            // modulo 8.
            const int n_tile = n_pair * 2 + matrix / 2;
            const int row_quad_lane = matrix_row / 2;
            const int row = warp_column + n_tile / 4 * 32 + row_quad_lane * 8 + ((n_tile + row_quad_lane) & 3) * 2 + matrix_row % 2;
            uint32_t registers[4];
            h3_load_matrix_x4(registers, stage_b + h3_swizzled_offset(row, k_step * 2 + matrix % 2));
            b_fragments[n_pair * 2][0] = registers[0];
            b_fragments[n_pair * 2][1] = registers[1];
            b_fragments[n_pair * 2 + 1][0] = registers[2];
            b_fragments[n_pair * 2 + 1][1] = registers[3];
        }
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            #pragma unroll
            for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                h3_mma_s8(accumulators[m_tile][n_tile], a_fragments[m_tile], b_fragments[n_tile][0], b_fragments[n_tile][1]);
            }
        }
    }
}

// Ring position of a warp, shared by every tile's K blocks.
struct Ring {
    int stage = 0;
    uint32_t phase = 0;
    int sequence = 0;

    __device__ void advance() {
        sequence++;
        if (++stage == tma_stages) {
            stage = 0;
            phase ^= 1;
        }
    }
};

__device__ __forceinline__ uint32_t h3_pack_bf16(float low, float high) {
    __nv_bfloat162 value = __floats2bfloat162_rn(low, high);
    return *reinterpret_cast<uint32_t *>(&value);
}

// silu(gate) * up of one feature whose packed columns hold (gate, up), rounded as h3_swiglu_int8
// rounds them: silu's value rounded to bf16 before the product (the caller's pack rounds again).
__device__ __forceinline__ float h3_swiglu_value(uint32_t word) {
    const float2 gv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&word));
    const float silu = __bfloat162float(__float2bfloat16(gv.x / (1.0f + expf(-gv.x))));
    return silu * gv.y;
}

// GROUPED's tile: after each K block g the warp's int32 sums (exact: 127² · 128 < 2²⁴, so their
// float is too) fold into fp32 at xs[g * m + row] · ws[g * n + column] and restart; the fp32 sums
// land at output[row, column]. `row0` is the lane's first row, `column0` its first column.
template <typename Config, typename Consume>
__device__ __forceinline__ void grouped_tile(Consume &consume, int32_t (&accumulators)[Config::m_tiles][Config::n_tiles][4], int k_blocks,
                                             const float *__restrict__ activation_scales, const float *__restrict__ weight_scales,
                                             float *__restrict__ output, int m, int n, int row0, int column0) {
    float sums[Config::m_tiles][Config::n_tiles][4] = {};
    for (int g = 0; g < k_blocks; g++) {
        consume(accumulators);
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const int row = row0 + m_tile * 16 + half * 8;
                const float row_scale = row < m ? activation_scales[static_cast<size_t>(g) * m + row] : 0.0f;
                #pragma unroll
                for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                    const int column = column0 + n_tile / 4 * 32 + ((n_tile % 4 + threadIdx.x % 4) & 3) * 2;
                    const float2 column_scale = *reinterpret_cast<const float2 *>(weight_scales + static_cast<size_t>(g) * n + column);
                    int32_t *products = accumulators[m_tile][n_tile] + half * 2;
                    float *sum = sums[m_tile][n_tile] + half * 2;
                    sum[0] += static_cast<float>(products[0]) * row_scale * column_scale.x;
                    sum[1] += static_cast<float>(products[1]) * row_scale * column_scale.y;
                    products[0] = 0;
                    products[1] = 0;
                }
            }
        }
    }
    #pragma unroll
    for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int row = row0 + m_tile * 16 + half * 8;
            if (row >= m) {
                continue;
            }
            #pragma unroll
            for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                const int column = column0 + n_tile / 4 * 32 + ((n_tile % 4 + threadIdx.x % 4) & 3) * 2;
                const float *sum = sums[m_tile][n_tile] + half * 2;
                *reinterpret_cast<float2 *>(output + static_cast<size_t>(row) * n + column) = make_float2(sum[0], sum[1]);
            }
        }
    }
}

// output[m, n] = bf16(acc[m, n] * xs[m] * ws[n]) where acc holds the exact int32 sums, or
// silu(gate) * up at [m, n / 2] when SWIGLU — the same products h3_dequantize and h3_swiglu_int8
// produce from the int32 sums, without the sums' round trip through memory. One CTA per SM walks
// the tiles; the warp that releases a ring stage last issues the TMA copies that refill it, so no
// producer warp costs every warp its registers. The tensor maps bind by reference so their
// addresses stay in parameter space — a by-value copy would land in local space, which
// cp.async.bulk.tensor refuses. N is a multiple of 8 (a lane's eight columns are one store); the
// last column tile may be partial: TMA zero-fills its weight rows past n and its lanes past n skip
// the store.
// GROUPED takes a scale per 128 of K on both operands instead (group-major: xs[g * m + row],
// ws[g * n + column]): each K block's exact int32 sums fold into fp32 at its two scales, and the
// fp32 rows are written unrounded. Its epilogue has no edge tile: N a multiple of BLOCK_N.
template <typename Config, bool SWIGLU, bool GROUPED = false, typename Out = std::conditional_t<GROUPED, float, __nv_bfloat16>>
__device__ __forceinline__ void
    gemm_kernel(const CUtensorMap &map_a, const CUtensorMap &map_b,
                const float *__restrict__ activation_scales, const float *__restrict__ weight_scales,
                Out *__restrict__ output, int m, int n, int k) {
    extern __shared__ __align__(1024) uint8_t shared_memory[];
    const uint32_t shared_base = (h3_shared_address(shared_memory) + 1023) & ~1023u;
    const uint32_t barriers = shared_base + tma_stages * Config::stage_bytes;
    int *release_counts = reinterpret_cast<int *>(shared_memory + (barriers - h3_shared_address(shared_memory)) + tma_stages * 8);

    TileGrid grid;
    grid.tiles_m = (m + Config::block_m - 1) / Config::block_m;
    grid.tiles_n = (n + Config::block_n - 1) / Config::block_n;
    grid.tiles = grid.tiles_m * grid.tiles_n;
    grid.k_blocks = k / tma_block_k;

    if (threadIdx.x == 0) {
        for (int stage = 0; stage < tma_stages; stage++) {
            h3_barrier_init(barriers + stage * 8, 1);
            release_counts[stage] = 0;
        }
        h3_barrier_init_fence();
        for (int stage = 0; stage < tma_stages; stage++) {
            fill_stage<Config>(grid, stage, shared_base + stage * Config::stage_bytes, barriers + stage * 8, &map_a, &map_b);
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warp_row = warp / Config::warps_n * Config::warp_m;
    const int warp_column = warp % Config::warps_n * Config::warp_n;
    // Position of the lane in the MMA accumulator layout.
    const int group_id = lane / 4;
    const int quad_lane = lane % 4;

    // Waits for the ring's current stage, multiplies it and releases it; the warp that releases it
    // last refills it.
    Ring ring;
    auto consume = [&](int32_t (&accumulators)[Config::m_tiles][Config::n_tiles][4]) {
        const uint32_t stage_a = shared_base + ring.stage * Config::stage_bytes;
        const uint32_t barrier = barriers + ring.stage * 8;
        h3_barrier_wait(barrier, ring.phase);
        multiply_stage<Config>(accumulators, stage_a, warp_row, warp_column, lane);
        __syncwarp();
        if (lane == 0) {
            __threadfence_block();
            if (atomicAdd(release_counts + ring.stage, 1) == tma_warps - 1) {
                release_counts[ring.stage] = 0;
                h3_async_proxy_fence();
                fill_stage<Config>(grid, ring.sequence + tma_stages, stage_a, barrier, &map_a, &map_b);
            }
        }
        ring.advance();
    };

    for (int tile = blockIdx.x; tile < grid.tiles; tile += gridDim.x) {
        int tile_m, tile_n;
        grid.coordinates(tile, tile_m, tile_n);
        int32_t accumulators[Config::m_tiles][Config::n_tiles][4];
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            #pragma unroll
            for (int n_tile = 0; n_tile < Config::n_tiles; n_tile++) {
                #pragma unroll
                for (int index = 0; index < 4; index++) {
                    accumulators[m_tile][n_tile][index] = 0;
                }
            }
        }
        if constexpr (GROUPED) {
            grouped_tile<Config>(consume, accumulators, grid.k_blocks, activation_scales, weight_scales, output, m, n,
                                 tile_m * Config::block_m + warp_row + group_id,
                                 tile_n * Config::block_n + warp_column + quad_lane * 8);
            continue; // what follows is the per-row epilogue
        }
        for (int k_block = 0; k_block < grid.k_blocks; k_block++) {
            consume(accumulators);
        }

        // Scale the int32 products. Slot j of each quad of n-tiles holds output columns 2((j + t) %
        // 4) and the one after among the lane's eight.
        float values[Config::m_tiles][Config::n_tiles][4];
        float row_scales[Config::m_tiles][2];
        #pragma unroll
        for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const int row = tile_m * Config::block_m + warp_row + m_tile * 16 + half * 8 + group_id;
                row_scales[m_tile][half] = row < m ? activation_scales[row] : 0.0f;
            }
        }
        #pragma unroll
        for (int quad = 0; quad < Config::n_tiles / 4; quad++) {
            // An edge tile's lanes past n read the last scales (their sums are TMA's zero fill), never stored.
            const int column = min(tile_n * Config::block_n + warp_column + quad * 32 + quad_lane * 8, n - 8);
            #pragma unroll
            for (int slot = 0; slot < 4; slot++) {
                const float2 column_scale = *reinterpret_cast<const float2 *>(weight_scales + column + ((slot + quad_lane) & 3) * 2);
                #pragma unroll
                for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        const int32_t *products = accumulators[m_tile][quad * 4 + slot] + half * 2;
                        float *scaled = values[m_tile][quad * 4 + slot] + half * 2;
                        scaled[0] = static_cast<float>(products[0]) * row_scales[m_tile][half] * column_scale.x;
                        scaled[1] = static_cast<float>(products[1]) * row_scales[m_tile][half] * column_scale.y;
                    }
                }
            }
        }

        #pragma unroll
        for (int quad = 0; quad < Config::n_tiles / 4; quad++) {
            const int column = tile_n * Config::block_n + warp_column + quad * 32 + quad_lane * 8;
            #pragma unroll
            for (int m_tile = 0; m_tile < Config::m_tiles; m_tile++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    uint32_t words[4];
                    #pragma unroll
                    for (int slot = 0; slot < 4; slot++) {
                        const float *scaled = values[m_tile][quad * 4 + slot] + half * 2;
                        words[slot] = h3_pack_bf16(scaled[0], scaled[1]);
                    }
                    // Rotate the slots right by t to put the columns in order.
                    if (quad_lane & 1) {
                        const uint32_t last = words[3];
                        words[3] = words[2];
                        words[2] = words[1];
                        words[1] = words[0];
                        words[0] = last;
                    }
                    if (quad_lane & 2) {
                        uint32_t swapped = words[0];
                        words[0] = words[2];
                        words[2] = swapped;
                        swapped = words[1];
                        words[1] = words[3];
                        words[3] = swapped;
                    }
                    const int row = tile_m * Config::block_m + warp_row + m_tile * 16 + half * 8 + group_id;
                    if (row < m && column < n) {
                        if constexpr (SWIGLU) {
                            // The lane's eight columns hold the (gate, up) pairs of features
                            // column / 2 to column / 2 + 3 (the weights' rows interleave pairwise).
                            *reinterpret_cast<uint2 *>(output + static_cast<size_t>(row) * (n / 2) + column / 2) =
                                make_uint2(h3_pack_bf16(h3_swiglu_value(words[0]), h3_swiglu_value(words[1])),
                                           h3_pack_bf16(h3_swiglu_value(words[2]), h3_swiglu_value(words[3])));
                        } else {
                            *reinterpret_cast<uint4 *>(output + static_cast<size_t>(row) * n + column) =
                                make_uint4(words[0], words[1], words[2], words[3]);
                        }
                    }
                }
            }
        }
    }
}

using TallTile = GemmConfig<256, 128, 4>;
// The prompt encoder's few rows: 64 × 64 tiles, so even a 40-row prompt spreads N over every SM.
using PromptTile = GemmConfig<64, 64, 4>;

} // namespace h3_int8
