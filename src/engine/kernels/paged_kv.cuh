#pragma once

// Addressing in the paged KV cache, shared by the kernels that read or write it.

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

namespace murmur {

// How cached keys and values are stored.
enum class KvFormat : std::uint8_t {
    kFp32,   // 4 bytes per value: exactly what the projections produced
    kBf16,   // 2 bytes, rounded to BF16
    kFp8,    // 1 byte (E4M3), plus one FP32 scale per token and KV head: v = fp8 * scale
};

// One layer's cached keys and values: one row per cached token (row page *
// page size + slot), each kv_heads x head_dim values in `format`. FP8 rows
// carry kv_heads scales each.
struct KvLayer {
    KvFormat format;
    void* keys;
    void* values;
    float* key_scales;     // kFp8 only: rows x kv_heads
    float* value_scales;   // kFp8 only
};

// Where one sequence's keys and values live: a page table of `page_tables`
// rows (one per sequence, `page_table_stride` entries apart) mapping each
// page of positions to a physical page of the cache. Pages hold
// 1 << page_shift positions.
struct PagedKv {
    const int* page_tables;
    int page_table_stride;
    int page_shift;
};

// The cache row of `position` in a sequence whose page table is `pages`:
// physical page * page size + offset within the page.
__device__ __forceinline__ std::size_t PagedRow(const int* pages, int page_shift, int position) {
    const int page = pages[position >> page_shift];
    return (static_cast<std::size_t>(page) << page_shift) | static_cast<std::size_t>(position & ((1 << page_shift) - 1));
}

#ifdef __CUDACC__

// Value `index` (0 .. kv_heads * head_dim) of cache row `row`, as FP32.
template <KvFormat F>
__device__ __forceinline__ float LoadKv(const void* data, const float* scales, std::size_t row, int kv_width,
                                        int head_dim, int index) {
    const std::size_t at = row * kv_width + index;
    if constexpr (F == KvFormat::kFp32) {
        return static_cast<const float*>(data)[at];
    } else if constexpr (F == KvFormat::kBf16) {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[at]);
    } else {
        const __half value(__nv_cvt_fp8_to_halfraw(static_cast<const std::uint8_t*>(data)[at], __NV_E4M3));
        return __half2float(value) * scales[row * (kv_width / head_dim) + index / head_dim];
    }
}

// Values `index` .. `index` + 7 of one cache row as they are stored (32,
// 16 or 8 bytes, and FP8's scale): loaded now, unpacked to FP32 later, so a
// kernel can issue many loads before it needs any of them.
template <KvFormat F>
struct KvChunk;
template <>
struct KvChunk<KvFormat::kFp32> {
    float4 low, high;
};
template <>
struct KvChunk<KvFormat::kBf16> {
    uint4 bits;
};
template <>
struct KvChunk<KvFormat::kFp8> {
    uint2 bits;
    float scale;
};

// `index` must be a multiple of 8 within one head.
template <KvFormat F>
__device__ __forceinline__ KvChunk<F> LoadKvChunk(const void* data, const float* scales, std::size_t row,
                                                  int kv_width, int head_dim, int index) {
    const std::size_t at = row * kv_width + index;
    if constexpr (F == KvFormat::kFp32) {
        const float4* p = reinterpret_cast<const float4*>(static_cast<const float*>(data) + at);
        return {p[0], p[1]};
    } else if constexpr (F == KvFormat::kBf16) {
        return {*reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(data) + at)};
    } else {
        return {*reinterpret_cast<const uint2*>(static_cast<const std::uint8_t*>(data) + at),
                scales[row * (kv_width / head_dim) + index / head_dim]};
    }
}

template <KvFormat F>
__device__ __forceinline__ void Unpack(const KvChunk<F>& chunk, float (&out)[8]) {
    if constexpr (F == KvFormat::kFp32) {
        out[0] = chunk.low.x, out[1] = chunk.low.y, out[2] = chunk.low.z, out[3] = chunk.low.w;
        out[4] = chunk.high.x, out[5] = chunk.high.y, out[6] = chunk.high.z, out[7] = chunk.high.w;
    } else if constexpr (F == KvFormat::kBf16) {
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&chunk.bits);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(pairs[i]);
            out[2 * i] = f.x;
            out[2 * i + 1] = f.y;
        }
    } else {
        const __nv_fp8x2_storage_t* pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&chunk.bits);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pairs[i], __NV_E4M3)));
            out[2 * i] = f.x * chunk.scale;
            out[2 * i + 1] = f.y * chunk.scale;
        }
    }
}

// Values `index` .. `index` + 7 of cache row `row` as FP32, in one load.
template <KvFormat F>
__device__ __forceinline__ void LoadKv8(const void* data, const float* scales, std::size_t row, int kv_width,
                                        int head_dim, int index, float (&out)[8]) {
    Unpack<F>(LoadKvChunk<F>(data, scales, row, kv_width, head_dim, index), out);
}

#endif

}  // namespace murmur
