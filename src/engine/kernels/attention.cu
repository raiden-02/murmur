#include "engine/kernels/attention.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/block_reduce.cuh"
#include "engine/kernels/launch.cuh"
#include "engine/kernels/tensor_core.cuh"

#include <cmath>
#include <cstddef>
#include <cuda_fp16.h>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

// Shared memory available for the query and the scores: the default
// per-block limit (no opt-in needed) minus BlockReduce's per-warp scratch.
constexpr std::size_t kDynamicSharedBytes = 48 * 1024 - kWarpSize * sizeof(float);

// One block per (head, query row). The block holds the query and one score
// per visible key in shared memory and runs the three steps of attention:
//   1. scores:  s_j = (q . k_j) * scale        for every visible key j
//   2. softmax: w_j = e^(s_j - max) / sum       (subtracting the max avoids overflow)
//   3. mix:     out = sum_j w_j * v_j
// Keys and values are found through the row's sequence's page table. Step 1
// gives each warp different keys; its 32 lanes split the dot product and
// combine it with WarpSum. Step 3 gives each thread one output dimension, so
// neighbouring threads read neighbouring floats of v_j (coalesced).
template <KvFormat F>
__global__ void CausalAttentionKernel(const float* q, KvLayer cache, PagedKv kv, QueryRows rows, float* out,
                                      int num_heads, int num_kv_heads, int head_dim, float scale) {
    extern __shared__ float shared[];
    float* query = shared;                // head_dim floats
    float* weights = shared + head_dim;   // one per visible key

    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int position = rows.positions[row];   // keys 0..position are visible (causal)
    const int* pages = kv.page_tables + static_cast<std::size_t>(rows.sequences[row]) * kv.page_table_stride;
    const int kv_head = head / (num_heads / num_kv_heads);
    const std::size_t q_width = static_cast<std::size_t>(num_heads) * head_dim;
    const int kv_width = num_kv_heads * head_dim;
    const int kv_offset = kv_head * head_dim;

    const float* q_row = q + row * q_width + static_cast<std::size_t>(head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        query[d] = q_row[d];
    }
    __syncthreads();

    // 1. Scores.
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    for (int j = warp; j <= position; j += kWarpsPerBlock) {
        const std::size_t key_row = PagedRow(pages, kv.page_shift, j);
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += kWarpSize) {
            dot += query[d] * LoadKv<F>(cache.keys, cache.key_scales, key_row, kv_width, head_dim, kv_offset + d);
        }
        dot = WarpSum(dot);
        if (lane == 0) {
            weights[j] = dot * scale;
        }
    }
    __syncthreads();

    // 2. Softmax.
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j <= position; j += blockDim.x) {
        local_max = fmaxf(local_max, weights[j]);
    }
    const float max_score = BlockMax(local_max);
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j <= position; j += blockDim.x) {
        weights[j] = expf(weights[j] - max_score);
        local_sum += weights[j];
    }
    const float inverse_sum = 1.0f / BlockSum(local_sum);
    for (int j = threadIdx.x; j <= position; j += blockDim.x) {
        weights[j] *= inverse_sum;
    }
    __syncthreads();

    // 3. Weighted mix of the values.
    float* out_row = out + row * q_width + static_cast<std::size_t>(head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j <= position; ++j) {
            acc += weights[j] * LoadKv<F>(cache.values, cache.value_scales, PagedRow(pages, kv.page_shift, j),
                                          kv_width, head_dim, kv_offset + d);
        }
        out_row[d] = acc;
    }
}


constexpr int kFlashRows = 64;   // query rows per block, 16 per warp
constexpr int kFlashKeys = 64;   // keys per tile
constexpr int kFlashPad = 8;     // FP16 values after each shared row: ldmatrix's 8 rows hit distinct banks

// 8 floats as 8 FP16 values, stored with one 16-byte write.
__device__ __forceinline__ void StoreHalf8(__half* to, const float (&f)[8]) {
    uint4 packed;
    packed.x = Bits(__floats2half2_rn(f[0], f[1]));
    packed.y = Bits(__floats2half2_rn(f[2], f[3]));
    packed.z = Bits(__floats2half2_rn(f[4], f[5]));
    packed.w = Bits(__floats2half2_rn(f[6], f[7]));
    *reinterpret_cast<uint4*>(to) = packed;
}

// Causal attention for prompt rows on tensor cores, the FlashAttention-2
// way: block (head, tile) takes 64 query rows of one head, each warp 16 of
// them, and walks the keys in tiles of 64. Each key/value tile is loaded
// once into shared memory (as FP16, whatever the cache format) for all 64
// rows, then per warp:
//   1. S = Q . K^T on tensor cores (Q in FP16 fragments, kept in registers);
//   2. online softmax per row, in registers: masked scores are -inf, the
//      running max and sum are updated, and what was summed so far shrinks
//      by e^(old max - new max);
//   3. O += P . V on tensor cores: the probabilities P go straight from the
//      S accumulators into A fragments (same layout), V is read transposed.
// Products are summed in FP32. Only the end divides by the row's sum.
//
// Rows of one tile can belong to two prompts (a chunk boundary): the block
// walks each run of rows of one sequence in turn, over that sequence's keys
// up to its run's last position, and the other rows are masked out.
template <KvFormat F, int HeadDim>
__global__ void __launch_bounds__(kThreadsPerBlock)
FlashAttentionKernel(const float* q, KvLayer cache, PagedKv kv, QueryRows rows, int row_count, float* out,
                     int num_heads, int num_kv_heads, float scale_log2) {
    constexpr int kPitch = HeadDim + kFlashPad;   // FP16 values per shared row
    constexpr int kSteps = HeadDim / 16;          // k-steps of Q . K^T
    constexpr int kDimTiles = HeadDim / 8;        // n-tiles of O
    constexpr int kKeyTiles = kFlashKeys / 8;     // n-tiles of S
    __shared__ __align__(16) __half q_or_v[kFlashRows * kPitch];   // the queries, then each value tile
    __shared__ __align__(16) __half keys[kFlashKeys * kPitch];
    __shared__ int row_position[kFlashRows];
    __shared__ int row_sequence[kFlashRows];

    const int head = blockIdx.x;
    const int first_row = blockIdx.y * kFlashRows;
    const int tile_rows = min(kFlashRows, row_count - first_row);
    const int kv_width = num_kv_heads * HeadDim;
    const int kv_offset = head / (num_heads / num_kv_heads) * HeadDim;
    const std::size_t q_width = static_cast<std::size_t>(num_heads) * HeadDim;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    for (int r = threadIdx.x; r < kFlashRows; r += blockDim.x) {
        row_position[r] = r < tile_rows ? rows.positions[first_row + r] : -1;
        row_sequence[r] = r < tile_rows ? rows.sequences[first_row + r] : -1;
    }
    for (int e = threadIdx.x * 8; e < kFlashRows * HeadDim; e += blockDim.x * 8) {
        const int r = e / HeadDim;
        const int d = e % HeadDim;
        float f[8] = {};
        if (r < tile_rows) {
            const float4* from = reinterpret_cast<const float4*>(q + (first_row + r) * q_width + head * HeadDim + d);
            const float4 a = from[0];
            const float4 b = from[1];
            f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
        }
        StoreHalf8(&q_or_v[r * kPitch + d], f);
    }
    __syncthreads();
    unsigned qa[kSteps][4];   // this warp's 16 query rows as A fragments
#pragma unroll
    for (int k = 0; k < kSteps; ++k) {
        LoadMatrixX4(qa[k], &q_or_v[(warp * 16 + lane % 16) * kPitch + k * 16 + lane / 16 * 8]);
    }

    // This thread's two rows of the warp's 16 (fragment rows lane / 4 and
    // lane / 4 + 8) and their running softmax state.
    const int row_lo = warp * 16 + lane / 4;
    const int row_hi = row_lo + 8;
    const int position_lo = row_position[row_lo];
    const int position_hi = row_position[row_hi];
    const int sequence_lo = row_sequence[row_lo];
    const int sequence_hi = row_sequence[row_hi];
    float o[kDimTiles][4] = {};
    float max_lo = -INFINITY, max_hi = -INFINITY, sum_lo = 0.0f, sum_hi = 0.0f;
    __syncthreads();   // q_or_v now holds value tiles

    for (int start = 0; start < tile_rows;) {
        const int sequence = row_sequence[start];
        int end = start;
        int last_key = 0;   // the run's last position: its rows see keys 0 .. last_key
        while (end < tile_rows && row_sequence[end] == sequence) {
            last_key = max(last_key, row_position[end]);
            ++end;
        }
        const int* pages = kv.page_tables + static_cast<std::size_t>(sequence) * kv.page_table_stride;
        for (int key0 = 0; key0 <= last_key; key0 += kFlashKeys) {
            for (int e = threadIdx.x * 8; e < kFlashKeys * HeadDim; e += blockDim.x * 8) {
                const int j = e / HeadDim;
                const int d = e % HeadDim;
                const std::size_t row = PagedRow(pages, kv.page_shift, min(key0 + j, last_key));
                float f[8];
                LoadKv8<F>(cache.keys, cache.key_scales, row, kv_width, HeadDim, kv_offset + d, f);
                StoreHalf8(&keys[j * kPitch + d], f);
                LoadKv8<F>(cache.values, cache.value_scales, row, kv_width, HeadDim, kv_offset + d, f);
                StoreHalf8(&q_or_v[j * kPitch + d], f);
            }
            __syncthreads();

            // 1. S = Q . K^T for the warp's 16 rows and the tile's 64 keys.
            float s[kKeyTiles][4] = {};
#pragma unroll
            for (int k = 0; k < kSteps; ++k) {
#pragma unroll
                for (int n = 0; n < kKeyTiles; n += 2) {
                    unsigned b[4];
                    LoadMatrixX4(b, &keys[(n * 8 + lane % 8 + lane / 16 * 8) * kPitch + k * 16 + lane / 8 % 2 * 8]);
                    Mma(s[n], qa[k], b[0], b[1]);
                    Mma(s[n + 1], qa[k], b[2], b[3]);
                }
            }

            // 2. Online softmax (log2 domain: exp2 is one instruction).
            float tile_lo = -INFINITY, tile_hi = -INFINITY;
#pragma unroll
            for (int n = 0; n < kKeyTiles; ++n) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int key = key0 + n * 8 + lane % 4 * 2 + c;
                    s[n][c] = sequence_lo == sequence && key <= position_lo ? s[n][c] * scale_log2 : -INFINITY;
                    s[n][c + 2] = sequence_hi == sequence && key <= position_hi ? s[n][c + 2] * scale_log2 : -INFINITY;
                    tile_lo = fmaxf(tile_lo, s[n][c]);
                    tile_hi = fmaxf(tile_hi, s[n][c + 2]);
                }
            }
            tile_lo = fmaxf(tile_lo, __shfl_xor_sync(kFullWarpMask, tile_lo, 1));
            tile_lo = fmaxf(tile_lo, __shfl_xor_sync(kFullWarpMask, tile_lo, 2));
            tile_hi = fmaxf(tile_hi, __shfl_xor_sync(kFullWarpMask, tile_hi, 1));
            tile_hi = fmaxf(tile_hi, __shfl_xor_sync(kFullWarpMask, tile_hi, 2));
            const float new_lo = fmaxf(max_lo, tile_lo);
            const float new_hi = fmaxf(max_hi, tile_hi);
            // A row that has seen no key yet keeps -inf: subtract 0, not -inf.
            const float base_lo = new_lo == -INFINITY ? 0.0f : new_lo;
            const float base_hi = new_hi == -INFINITY ? 0.0f : new_hi;
            const float shrink_lo = exp2f(max_lo - base_lo);
            const float shrink_hi = exp2f(max_hi - base_hi);
            float tile_sum_lo = 0.0f, tile_sum_hi = 0.0f;
#pragma unroll
            for (int n = 0; n < kKeyTiles; ++n) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    s[n][c] = exp2f(s[n][c] - base_lo);
                    s[n][c + 2] = exp2f(s[n][c + 2] - base_hi);
                    tile_sum_lo += s[n][c];
                    tile_sum_hi += s[n][c + 2];
                }
            }
            tile_sum_lo += __shfl_xor_sync(kFullWarpMask, tile_sum_lo, 1);
            tile_sum_lo += __shfl_xor_sync(kFullWarpMask, tile_sum_lo, 2);
            tile_sum_hi += __shfl_xor_sync(kFullWarpMask, tile_sum_hi, 1);
            tile_sum_hi += __shfl_xor_sync(kFullWarpMask, tile_sum_hi, 2);
            sum_lo = sum_lo * shrink_lo + tile_sum_lo;
            sum_hi = sum_hi * shrink_hi + tile_sum_hi;
            max_lo = new_lo;
            max_hi = new_hi;
#pragma unroll
            for (int n = 0; n < kDimTiles; ++n) {
                o[n][0] *= shrink_lo;
                o[n][1] *= shrink_lo;
                o[n][2] *= shrink_hi;
                o[n][3] *= shrink_hi;
            }

            // 3. O += P . V, 16 keys at a time.
#pragma unroll
            for (int k = 0; k < kFlashKeys / 16; ++k) {
                const unsigned p[4] = {Bits(__floats2half2_rn(s[2 * k][0], s[2 * k][1])),
                                       Bits(__floats2half2_rn(s[2 * k][2], s[2 * k][3])),
                                       Bits(__floats2half2_rn(s[2 * k + 1][0], s[2 * k + 1][1])),
                                       Bits(__floats2half2_rn(s[2 * k + 1][2], s[2 * k + 1][3]))};
#pragma unroll
                for (int n = 0; n < kDimTiles; n += 2) {
                    unsigned b[4];
                    LoadMatrixX4Transposed(
                        b, &q_or_v[(k * 16 + lane % 8 + lane / 8 % 2 * 8) * kPitch + n * 8 + lane / 16 * 8]);
                    Mma(o[n], p, b[0], b[1]);
                    Mma(o[n + 1], p, b[2], b[3]);
                }
            }
            __syncthreads();   // the next tile overwrites keys and values
        }
        start = end;
    }

    const float inverse_lo = sum_lo > 0.0f ? 1.0f / sum_lo : 0.0f;
    const float inverse_hi = sum_hi > 0.0f ? 1.0f / sum_hi : 0.0f;
#pragma unroll
    for (int n = 0; n < kDimTiles; ++n) {
        const int d = head * HeadDim + n * 8 + lane % 4 * 2;
        if (row_lo < tile_rows) {
            *reinterpret_cast<float2*>(out + (first_row + row_lo) * q_width + d) =
                make_float2(o[n][0] * inverse_lo, o[n][1] * inverse_lo);
        }
        if (row_hi < tile_rows) {
            *reinterpret_cast<float2*>(out + (first_row + row_hi) * q_width + d) =
                make_float2(o[n][2] * inverse_hi, o[n][3] * inverse_hi);
        }
    }
}

template <int HeadDim>
void LaunchFlash(const float* q, const KvLayer& cache, const PagedKv& kv, const QueryRows& rows, int row_count,
                 float* out, const AttentionShape& shape, float scale_log2, cudaStream_t stream) {
    const dim3 grid(shape.num_heads, (row_count + kFlashRows - 1) / kFlashRows);
    const auto launch = [&](auto kernel) {
        kernel<<<grid, kThreadsPerBlock, 0, stream>>>(q, cache, kv, rows, row_count, out, shape.num_heads,
                                                      shape.num_kv_heads, scale_log2);
    };
    switch (cache.format) {
    case KvFormat::kFp32:
        launch(FlashAttentionKernel<KvFormat::kFp32, HeadDim>);
        break;
    case KvFormat::kBf16:
        launch(FlashAttentionKernel<KvFormat::kBf16, HeadDim>);
        break;
    case KvFormat::kFp8:
        launch(FlashAttentionKernel<KvFormat::kFp8, HeadDim>);
        break;
    }
}

}  // namespace

// The longest context (cached + new tokens) one launch supports: the scores
// for every visible key must fit in shared memory next to the query.
int MaxAttentionContext(int head_dim) {
    return static_cast<int>(kDynamicSharedBytes / sizeof(float)) - head_dim;
}

// Causal scaled dot-product attention with grouped-query heads, for query
// rows that may come from different sequences (chunks of several prompts).
//
//   q:      row_count x (num_heads * head_dim), already rotated by RoPE
//   cache:  every visible key and value already written (including the
//           rows' own), in any KvFormat, found through each row's page table
//   rows:   per query row, its position and its sequence's page-table row
//   max_context: at least the largest position + 1 among the rows
//   out:    row_count x (num_heads * head_dim)
//
// Query head h reads key/value head h / (num_heads / num_kv_heads).
void CausalAttention(const float* q, const KvLayer& cache, const PagedKv& kv, const QueryRows& rows, int row_count,
                     int max_context, float* out, const AttentionShape& shape, cudaStream_t stream) {
    RequireArgument(row_count >= 0 && max_context > 0 && shape.num_heads > 0 && shape.num_kv_heads > 0 &&
                        shape.num_heads % shape.num_kv_heads == 0 && shape.head_dim > 0,
                    "CausalAttention: bad shape");
    RequireArgument(max_context <= MaxAttentionContext(shape.head_dim), "CausalAttention: context too long");
    if (row_count == 0) {
        return;
    }
    const dim3 grid(shape.num_heads, row_count);
    const std::size_t shared_bytes = (static_cast<std::size_t>(shape.head_dim) + max_context) * sizeof(float);
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    const auto launch = [&](auto kernel) {
        kernel<<<grid, kThreadsPerBlock, shared_bytes, stream>>>(q, cache, kv, rows, out, shape.num_heads,
                                                                 shape.num_kv_heads, shape.head_dim, scale);
    };
    switch (cache.format) {
    case KvFormat::kFp32:
        launch(CausalAttentionKernel<KvFormat::kFp32>);
        break;
    case KvFormat::kBf16:
        launch(CausalAttentionKernel<KvFormat::kBf16>);
        break;
    case KvFormat::kFp8:
        launch(CausalAttentionKernel<KvFormat::kFp8>);
        break;
    }
    CheckKernelLaunch();
}


// CausalAttention on tensor cores, for prompt chunks: the same inputs and
// result, with queries, keys and values rounded to FP16 (products summed in
// FP32) and no limit on the context. Needs head_dim 64 or 128.
void FlashAttention(const float* q, const KvLayer& cache, const PagedKv& kv, const QueryRows& rows, int row_count,
                    float* out, const AttentionShape& shape, cudaStream_t stream) {
    RequireArgument(row_count >= 0 && shape.num_heads > 0 && shape.num_kv_heads > 0 &&
                        shape.num_heads % shape.num_kv_heads == 0 && (shape.head_dim == 64 || shape.head_dim == 128),
                    "FlashAttention: bad shape");
    if (row_count == 0) {
        return;
    }
    const float scale_log2 = 1.4426950408889634f / std::sqrt(static_cast<float>(shape.head_dim));   // log2(e) / sqrt(d)
    if (shape.head_dim == 128) {
        LaunchFlash<128>(q, cache, kv, rows, row_count, out, shape, scale_log2, stream);
    } else {
        LaunchFlash<64>(q, cache, kv, rows, row_count, out, shape, scale_log2, stream);
    }
    CheckKernelLaunch();
}

}  // namespace murmur
