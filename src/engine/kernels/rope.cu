#include "engine/kernels/rope.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/block_reduce.cuh"
#include "engine/kernels/launch.cuh"

#include <cmath>
#include <cstddef>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 64;
constexpr int kMaxHeadDim = 256;
constexpr float kFp8Max = 448.0f;

// Rotates one head vector by its position (the "rotate half" layout): each
// thread turns the pair (i, i + half) by position * inverse_frequencies[i].
//   a' = a cos - b sin,   b' = b cos + a sin
__device__ void RotateHead(const float* in, float* out, const float* inverse_frequencies, int head_dim,
                           float position) {
    const int half = head_dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float sin_angle;
        float cos_angle;
        sincosf(position * inverse_frequencies[i], &sin_angle, &cos_angle);
        const float a = in[i];
        const float b = in[i + half];
        out[i] = a * cos_angle - b * sin_angle;
        out[i + half] = b * cos_angle + a * sin_angle;
    }
}

// Writes one key or value head (head_dim values staged in shared memory)
// into its cache row in format F. FP8 scales the head so its largest
// magnitude becomes E4M3's largest value, and stores the scale.
template <KvFormat F>
__device__ void StoreHead(const float* staged, int head_dim, void* data, float* scales, std::size_t cache_row,
                          int num_kv_heads, int head) {
    const std::size_t base = (cache_row * num_kv_heads + head) * head_dim;
    if constexpr (F == KvFormat::kFp32) {
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            static_cast<float*>(data)[base + i] = staged[i];
        }
    } else if constexpr (F == KvFormat::kBf16) {
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            static_cast<__nv_bfloat16*>(data)[base + i] = __float2bfloat16_rn(staged[i]);
        }
    } else {
        float largest = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            largest = fmaxf(largest, fabsf(staged[i]));
        }
        largest = BlockMax(largest);
        const float scale = largest > 0.0f ? largest / kFp8Max : 1.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            static_cast<std::uint8_t*>(data)[base + i] = __nv_cvt_float_to_fp8(staged[i] / scale, __NV_SATFINITE, __NV_E4M3);
        }
        if (threadIdx.x == 0) {
            scales[cache_row * num_kv_heads + head] = scale;
        }
    }
}

// One block per (row, head slot). A row of `qkv` holds num_heads query
// heads, then num_kv_heads key heads, then num_kv_heads value heads. Query
// heads are rotated into `q`; key heads are rotated and value heads copied
// into shared memory, then stored in the cache's format at the row's cache row.
template <KvFormat F>
__global__ void RopeAndCacheQkvKernel(const float* qkv, const int* positions, const int* cache_rows,
                                      const float* inverse_frequencies, int num_heads, int num_kv_heads,
                                      int head_dim, float* q, KvLayer cache) {
    __shared__ float staged[kMaxHeadDim];
    const int slots = num_heads + 2 * num_kv_heads;
    const int row = blockIdx.x / slots;
    const int slot = blockIdx.x % slots;
    const int position = positions[row];
    const std::size_t cache_row = static_cast<std::size_t>(cache_rows[row]);
    const float* in = qkv + static_cast<std::size_t>(row) * slots * head_dim + static_cast<std::size_t>(slot) * head_dim;

    if (slot < num_heads) {
        float* out = q + (static_cast<std::size_t>(row) * num_heads + slot) * head_dim;
        RotateHead(in, out, inverse_frequencies, head_dim, static_cast<float>(position));
        return;
    }
    const bool key = slot < num_heads + num_kv_heads;
    if (key) {
        RotateHead(in, staged, inverse_frequencies, head_dim, static_cast<float>(position));
    } else {
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            staged[i] = in[i];
        }
    }
    __syncthreads();
    const int head = key ? slot - num_heads : slot - num_heads - num_kv_heads;
    StoreHead<F>(staged, head_dim, key ? cache.keys : cache.values, key ? cache.key_scales : cache.value_scales,
                 cache_row, num_kv_heads, head);
}

}  // namespace

// The rotation speed of each dimension pair: theta^(-2i / head_dim) for
// i = 0 .. head_dim/2 - 1. Pair 0 turns fastest (1 radian per position); the
// last pair turns very slowly, so together they encode position at every scale.
std::vector<float> RopeInverseFrequencies(int head_dim, float theta) {
    RequireArgument(head_dim > 0 && head_dim % 2 == 0 && theta > 0.0f, "RopeInverseFrequencies: bad arguments");
    std::vector<float> frequencies(head_dim / 2);
    for (int i = 0; i < head_dim / 2; ++i) {
        frequencies[i] = static_cast<float>(1.0 / std::pow(static_cast<double>(theta), 2.0 * i / head_dim));
    }
    return frequencies;
}

// Rotary position embedding fused with the KV-cache write, for `rows`
// tokens: reads the stacked projections `qkv` (rows x (num_heads + 2 *
// num_kv_heads) * head_dim), rotates row r by positions[r], writes the
// rotated queries to `q` (rows x num_heads * head_dim), and the rotated keys
// and the values to row cache_rows[r] of this layer's cache (one row of
// num_kv_heads * head_dim per cached token), in the cache's format. One
// launch instead of two RoPE kernels and two copies. `positions` and
// `cache_rows` live in GPU memory, so rows may belong to different sequences
// and a recorded CUDA graph can replay the launch for new tokens.
void RopeAndCacheQkv(const float* qkv, int rows, const int* positions, const int* cache_rows,
                     const float* inverse_frequencies, const QkvLayout& layout, float* q, const KvLayer& cache,
                     cudaStream_t stream) {
    RequireArgument(rows >= 0 && layout.num_heads > 0 && layout.num_kv_heads > 0 && layout.head_dim > 0 &&
                        layout.head_dim % 2 == 0 && layout.head_dim <= kMaxHeadDim,
                    "RopeAndCacheQkv: bad shape");
    if (rows == 0) {
        return;
    }
    const int blocks = rows * (layout.num_heads + 2 * layout.num_kv_heads);
    const auto launch = [&](auto kernel) {
        kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(qkv, positions, cache_rows, inverse_frequencies,
                                                        layout.num_heads, layout.num_kv_heads, layout.head_dim, q,
                                                        cache);
    };
    switch (cache.format) {
    case KvFormat::kFp32:
        launch(RopeAndCacheQkvKernel<KvFormat::kFp32>);
        break;
    case KvFormat::kBf16:
        launch(RopeAndCacheQkvKernel<KvFormat::kBf16>);
        break;
    case KvFormat::kFp8:
        launch(RopeAndCacheQkvKernel<KvFormat::kFp8>);
        break;
    }
    CheckKernelLaunch();
}

}  // namespace murmur
