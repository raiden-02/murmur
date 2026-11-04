#include "engine/kernels/decode_attention.cuh"

#include "common/cuda_check.hpp"
#include "common/device_info.hpp"
#include "engine/kernels/block_reduce.cuh"
#include "engine/kernels/launch.cuh"

#include <cfloat>
#include <cmath>
#include <cstddef>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kMaxSplits = 128;        // most slices one sequence's keys are cut into
static_assert(kMaxSplits == kThreadsPerBlock, "the combine kernel gives each split one thread");
constexpr int kTile = 64;              // keys per slice: scored, softmaxed and summed together
constexpr int kMaxGroup = 8;           // query heads per key/value head
constexpr int kBlocksPerSm = 8;       // blocks per SM that keep the GPU busy
constexpr int kVector = 8;             // values one thread loads at once

// Slices of a `context`-key sequence. Fixed-size slices make the result
// depend on the context alone, never on the batch: a sequence decoded alone
// and the same sequence checked in a batch (speculative decoding) add the
// same numbers in the same order, bit for bit.
__host__ __device__ constexpr int SplitsFor(int context) {
    return (context + kTile - 1) / kTile;
}

// Blocks per sequence and KV head: enough to fill the GPU, and never more
// than the longest context has slices. Each block takes slices y, y +
// gridDim.y, ...: how many blocks share the work never changes a slice's
// result, so the batch size can choose it freely.
int BlocksPerSequence(int batch, int num_kv_heads, int max_context) {
    const int pairs = batch * num_kv_heads;
    const int target = kBlocksPerSm * CurrentDeviceSmCount();
    const int want = (target + pairs - 1) / pairs;
    return want < 1 ? 1 : (want > SplitsFor(max_context) ? SplitsFor(max_context) : want);
}

// Attention for one new token per sequence over a slice of its cache
// ("split-K", the flash-decoding idea): block (kv_head, split, sequence)
// handles keys [split * kTile, ...) of that sequence for the Group query
// heads that share kv_head, so each cached key and value is read once, not
// once per query head. It leaves an unnormalized partial result per query
// head (the slice's max score m, its sum of e^(s - m), and sum of
// e^(s - m) * v) for CombineSplitsKernel to merge.
//
// Decode attention reads every cached key and value once and does little
// math on them, so it is bound by memory, and what matters is keeping many
// loads in flight:
//   - every load brings 8 values (16 bytes of BF16);
//   - scores: kLanesPerKey lanes share a key row, each holding its 8
//     dimensions of the Group queries in registers; the passes over the
//     slice are unrolled and their loads don't wait for each other (rows
//     past the slice load a valid row and are ignored);
//   - softmax: a warp per head, two keys per lane;
//   - values: thread (dims, key lane) sums 8 dimensions over every
//     kKeyLanes-th key; at the end the key lanes are summed by shuffles
//     within each warp, then across the warps.
//
// Context lengths are read from GPU memory and the grid covers the longest
// context, so the launch shape never changes between decode steps (a
// requirement for CUDA Graphs); slices past the end of a sequence's context
// return immediately.
template <KvFormat F, int HeadDim, int Group>
__global__ void __launch_bounds__(kThreadsPerBlock)
SplitAttentionKernel(const float* q, KvLayer cache, PagedKv kv, const int* context_lengths, int num_heads,
                     int num_kv_heads, float scale, float* partial_outputs, float* partial_maxes,
                     float* partial_sums) {
    constexpr int kWarps = kThreadsPerBlock / kWarpSize;
    constexpr int kLanesPerKey = HeadDim / kVector;               // lanes sharing one key row for scores
    constexpr int kKeysPerPass = kThreadsPerBlock / kLanesPerKey;  // keys scored at once by the block
    constexpr int kDimGroups = HeadDim / kVector;                  // value phase: groups of 8 dimensions...
    constexpr int kKeyLanes = kThreadsPerBlock / kDimGroups;       // ...times lanes over keys
    static_assert(kTile % kKeysPerPass == 0 && kTile % kKeyLanes == 0 && kTile == 2 * kWarpSize);
    __shared__ float scores[Group][kTile];                         // becomes e^(s - max)
    __shared__ float slice_max[Group];
    __shared__ float slice_sum[Group];
    __shared__ float warp_sums[kWarps][Group * HeadDim];

    const int context = context_lengths[blockIdx.z];
    const int* pages = kv.page_tables + static_cast<std::size_t>(blockIdx.z) * kv.page_table_stride;
    const int kv_width = num_kv_heads * HeadDim;
    const int kv_offset = blockIdx.x * HeadDim;
    const std::size_t head_base = static_cast<std::size_t>(blockIdx.z) * num_heads + blockIdx.x * Group;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    // This thread's 8 query dimensions, for every head of the group.
    const int score_dims = (threadIdx.x % kLanesPerKey) * kVector;
    const int score_key = threadIdx.x / kLanesPerKey;
    float queries[Group][kVector];
#pragma unroll
    for (int h = 0; h < Group; ++h) {
#pragma unroll
        for (int i = 0; i < kVector; ++i) {
            queries[h][i] = q[(head_base + h) * HeadDim + score_dims + i];
        }
    }
    const int value_dims = (threadIdx.x % kDimGroups) * kVector;
    const int value_lane = threadIdx.x / kDimGroups;

    for (int slice = blockIdx.y; slice < SplitsFor(context); slice += gridDim.y) {
    const int begin = slice * kTile;
    const int n = min(kTile, context - begin);   // keys in this slice
    float acc[Group][kVector] = {};

    // 0. Every load of the slice at once: the values don't depend on the
    //    scores, so their memory waits overlap with the keys' instead of
    //    following them (FP32 caches, exact mode only, load the values later
    //    to spare registers).
    constexpr int kScorePasses = kTile / kKeysPerPass;
    constexpr int kValuePasses = kTile / kKeyLanes;
    constexpr bool kEarlyValues = F != KvFormat::kFp32;
    KvChunk<F> keys[kScorePasses];
    KvChunk<F> values[kEarlyValues ? kValuePasses : 1];
#pragma unroll
    for (int pass = 0; pass < kScorePasses; ++pass) {
        if (pass * kKeysPerPass < n) {   // the same for the whole block: a short slice loads less
            const int j = min(pass * kKeysPerPass + score_key, n - 1);
            keys[pass] = LoadKvChunk<F>(cache.keys, cache.key_scales, PagedRow(pages, kv.page_shift, begin + j),
                                        kv_width, HeadDim, kv_offset + score_dims);
        }
    }
    if constexpr (kEarlyValues) {
#pragma unroll
        for (int pass = 0; pass < kValuePasses; ++pass) {
            if (pass * kKeyLanes < n) {
                const int j = min(pass * kKeyLanes + value_lane, n - 1);
                values[pass] = LoadKvChunk<F>(cache.values, cache.value_scales,
                                              PagedRow(pages, kv.page_shift, begin + j), kv_width, HeadDim,
                                              kv_offset + value_dims);
            }
        }
    }

    // 1. Scores.
#pragma unroll
    for (int pass = 0; pass < kScorePasses; ++pass) {
        if (pass * kKeysPerPass >= n) {
            break;
        }
        const int j = pass * kKeysPerPass + score_key;
        float k[kVector];
        Unpack<F>(keys[pass], k);
#pragma unroll
        for (int h = 0; h < Group; ++h) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < kVector; ++i) {
                dot += queries[h][i] * k[i];
            }
#pragma unroll
            for (int offset = kLanesPerKey / 2; offset > 0; offset /= 2) {
                dot += __shfl_xor_sync(kFullWarpMask, dot, offset);
            }
            if (j < n && threadIdx.x % kLanesPerKey == 0) {
                scores[h][j] = dot * scale;
            }
        }
    }
    __syncthreads();

    // 2. Softmax, a warp per head: the slice's max and e^(s - max).
    for (int h = warp; h < Group; h += kWarps) {
        const float s0 = lane < n ? scores[h][lane] : -FLT_MAX;
        const float s1 = lane + kWarpSize < n ? scores[h][lane + kWarpSize] : -FLT_MAX;
        const float max_score = WarpMax(fmaxf(s0, s1));
        const float p0 = lane < n ? expf(s0 - max_score) : 0.0f;
        const float p1 = lane + kWarpSize < n ? expf(s1 - max_score) : 0.0f;
        if (lane < n) {
            scores[h][lane] = p0;
        }
        if (lane + kWarpSize < n) {
            scores[h][lane + kWarpSize] = p1;
        }
        const float sum = WarpSum(p0 + p1);
        if (lane == 0) {
            slice_max[h] = max_score;
            slice_sum[h] = sum;
        }
    }
    __syncthreads();

    // 3. Values: this thread's 8 dimensions over every kKeyLanes-th key.
#pragma unroll
    for (int pass = 0; pass < kValuePasses; ++pass) {
        if (pass * kKeyLanes >= n) {
            break;
        }
        const int j = pass * kKeyLanes + value_lane;
        float v[kVector];
        if constexpr (kEarlyValues) {
            Unpack<F>(values[pass], v);
        } else {
            LoadKv8<F>(cache.values, cache.value_scales, PagedRow(pages, kv.page_shift, begin + min(j, n - 1)),
                       kv_width, HeadDim, kv_offset + value_dims, v);
        }
#pragma unroll
        for (int h = 0; h < Group; ++h) {
            const float p = j < n ? scores[h][j] : 0.0f;
#pragma unroll
            for (int i = 0; i < kVector; ++i) {
                acc[h][i] += p * v[i];
            }
        }
    }

    // Sum the key lanes: the ones in one warp by shuffles, then the warps.
#pragma unroll
    for (int h = 0; h < Group; ++h) {
#pragma unroll
        for (int i = 0; i < kVector; ++i) {
#pragma unroll
            for (int offset = kDimGroups; offset < kWarpSize; offset *= 2) {
                acc[h][i] += __shfl_xor_sync(kFullWarpMask, acc[h][i], offset);
            }
            if (lane < kDimGroups) {
                warp_sums[warp][h * HeadDim + value_dims + i] = acc[h][i];
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < Group * HeadDim; e += kThreadsPerBlock) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            total += warp_sums[w][e];
        }
        partial_outputs[((head_base + e / HeadDim) * kMaxSplits + slice) * HeadDim + e % HeadDim] = total;
    }
    if (threadIdx.x < Group) {
        partial_maxes[(head_base + threadIdx.x) * kMaxSplits + slice] = slice_max[threadIdx.x];
        partial_sums[(head_base + threadIdx.x) * kMaxSplits + slice] = slice_sum[threadIdx.x];
    }
    __syncthreads();   // the next slice reuses the shared scores and sums
    }
}

// Merges the splits of one query head of one sequence (block (head,
// sequence)). Each split's partial was scaled by its own max m_s; rescaling
// by e^(m_s - M), with M the overall max, puts them on a common footing:
//   out = sum_s w_s * acc_s   with   w_s = e^(m_s - M) / sum_t e^(m_t - M) * l_t
// The weights w_s are computed once per split (not once per dimension).
__global__ void CombineSplitsKernel(const int* context_lengths, int num_heads, int head_dim,
                                    const float* partial_outputs, const float* partial_maxes,
                                    const float* partial_sums, float* out) {
    __shared__ float weights[kMaxSplits];
    const std::size_t head = static_cast<std::size_t>(blockIdx.y) * num_heads + blockIdx.x;
    const int splits = SplitsFor(context_lengths[blockIdx.y]);
    const float* maxes = partial_maxes + head * kMaxSplits;
    const float* sums = partial_sums + head * kMaxSplits;

    // splits <= kMaxSplits = blockDim.x: one split per thread.
    const int s = threadIdx.x;
    const float max_score = s < splits ? maxes[s] : -FLT_MAX;
    const float overall_max = BlockMax(max_score);
    const float scale = s < splits ? expf(max_score - overall_max) : 0.0f;
    const float total = BlockSum(s < splits ? scale * sums[s] : 0.0f);
    weights[s] = scale / total;
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < splits; ++t) {
            acc += weights[t] * partial_outputs[(head * kMaxSplits + t) * head_dim + d];
        }
        out[head * head_dim + d] = acc;
    }
}

DecodeAttentionShape CheckedShape(const DecodeAttentionShape& shape) {
    RequireArgument(shape.num_heads > 0 && shape.num_kv_heads > 0 && shape.max_context > 0 && shape.max_batch > 0 &&
                        shape.num_heads % shape.num_kv_heads == 0,
                    "DecodeAttention: bad shape");
    RequireArgument(shape.head_dim == 64 || shape.head_dim == 128, "DecodeAttention: head_dim must be 64 or 128");
    RequireArgument(shape.num_heads / shape.num_kv_heads <= kMaxGroup, "DecodeAttention: too many query heads per KV head");
    RequireArgument(SplitsFor(shape.max_context) <= kMaxSplits, "DecodeAttention: context too long");
    return shape;
}

std::size_t PartialCount(const DecodeAttentionShape& shape) {
    return static_cast<std::size_t>(shape.max_batch) * shape.num_heads * kMaxSplits;
}

template <KvFormat F, int HeadDim, int Group>
void LaunchGroup(dim3 grid, cudaStream_t stream, const float* q, const KvLayer& cache, const PagedKv& kv,
                 const int* context_lengths, int num_heads, int num_kv_heads, float scale,
                 DecodeAttentionWorkspace& workspace) {
    SplitAttentionKernel<F, HeadDim, Group><<<grid, kThreadsPerBlock, 0, stream>>>(
        q, cache, kv, context_lengths, num_heads, num_kv_heads, scale, workspace.PartialOutputs(),
        workspace.PartialMaxes(), workspace.PartialSums());
}

// The kernel for this cache format, head size and group size (each a
// template parameter, so registers hold exactly what is used).
template <KvFormat F, int HeadDim>
void LaunchSplits(dim3 grid, cudaStream_t stream, const float* q, const KvLayer& cache, const PagedKv& kv,
                  const int* context_lengths, int num_heads, int num_kv_heads, float scale,
                  DecodeAttentionWorkspace& workspace) {
    const auto launch = [&]<int Group>() {
        LaunchGroup<F, HeadDim, Group>(grid, stream, q, cache, kv, context_lengths, num_heads, num_kv_heads, scale,
                                       workspace);
    };
    switch (num_heads / num_kv_heads) {
    case 1: launch.template operator()<1>(); break;
    case 2: launch.template operator()<2>(); break;
    case 3: launch.template operator()<3>(); break;
    case 4: launch.template operator()<4>(); break;
    case 5: launch.template operator()<5>(); break;
    case 6: launch.template operator()<6>(); break;
    case 7: launch.template operator()<7>(); break;
    default: launch.template operator()<8>(); break;
    }
}

template <KvFormat F>
void LaunchSplits(int head_dim, dim3 grid, cudaStream_t stream, const float* q, const KvLayer& cache, const PagedKv& kv,
                  const int* context_lengths, int num_heads, int num_kv_heads, float scale,
                  DecodeAttentionWorkspace& workspace) {
    if (head_dim == 128) {
        LaunchSplits<F, 128>(grid, stream, q, cache, kv, context_lengths, num_heads, num_kv_heads, scale, workspace);
    } else {
        LaunchSplits<F, 64>(grid, stream, q, cache, kv, context_lengths, num_heads, num_kv_heads, scale, workspace);
    }
}

}  // namespace

// Allocates the per-split partial results once, for up to max_batch
// sequences of any context up to max_context.
DecodeAttentionWorkspace::DecodeAttentionWorkspace(const DecodeAttentionShape& shape)
    : shape_(CheckedShape(shape)),
      partial_outputs_(PartialCount(shape) * shape.head_dim),
      partial_maxes_(PartialCount(shape)),
      partial_sums_(PartialCount(shape)) {}

// Causal attention for one new token in each of `batch` sequences (decode).
//   q:    batch x num_heads x head_dim, already rotated; row b is sequence b
//   kv:   sequence b's page table is row b of kv.page_tables
//   context_lengths[b] (GPU memory): positions sequence b attends to,
//         including the new token's own, between 1 and max_context
//   out:  batch x num_heads x head_dim
// Two launches: the split kernel, then the combine kernel; both have the
// same shape for a given batch size.
void DecodeAttention(const float* q, const KvLayer& cache, const PagedKv& kv, const int* context_lengths, int batch,
                     float* out, DecodeAttentionWorkspace& workspace, cudaStream_t stream) {
    const DecodeAttentionShape& shape = workspace.Shape();
    RequireArgument(batch >= 0 && batch <= shape.max_batch, "DecodeAttention: batch out of range");
    if (batch == 0) {
        return;
    }
    const dim3 split_grid(shape.num_kv_heads, BlocksPerSequence(batch, shape.num_kv_heads, shape.max_context), batch);
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    switch (cache.format) {
    case KvFormat::kFp32:
        LaunchSplits<KvFormat::kFp32>(shape.head_dim, split_grid, stream, q, cache, kv, context_lengths,
                                      shape.num_heads, shape.num_kv_heads, scale, workspace);
        break;
    case KvFormat::kBf16:
        LaunchSplits<KvFormat::kBf16>(shape.head_dim, split_grid, stream, q, cache, kv, context_lengths,
                                      shape.num_heads, shape.num_kv_heads, scale, workspace);
        break;
    case KvFormat::kFp8:
        LaunchSplits<KvFormat::kFp8>(shape.head_dim, split_grid, stream, q, cache, kv, context_lengths,
                                     shape.num_heads, shape.num_kv_heads, scale, workspace);
        break;
    }
    CheckKernelLaunch();
    const dim3 combine_grid(shape.num_heads, batch);
    CombineSplitsKernel<<<combine_grid, kThreadsPerBlock, 0, stream>>>(
        context_lengths, shape.num_heads, shape.head_dim, workspace.PartialOutputs(),
        workspace.PartialMaxes(), workspace.PartialSums(), out);
    CheckKernelLaunch();
}

}  // namespace murmur
