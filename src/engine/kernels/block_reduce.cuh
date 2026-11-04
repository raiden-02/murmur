#pragma once

// Device-side reductions shared by the engine's kernels.

#include <cfloat>

namespace murmur {

inline constexpr int kWarpSize = 32;
inline constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

// Sum across the 32 lanes of a warp; every lane receives the result.
__device__ __forceinline__ float WarpSum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float WarpMax(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(kFullWarpMask, value, offset));
    }
    return value;
}

// Combines one value per thread across the whole block; every thread
// receives the result. blockDim.x must be a multiple of 32 (at most 1024),
// and every thread of the block must call it (it contains barriers).
template <typename WarpOp>
__device__ float BlockReduce(float value, float identity, WarpOp warp_op) {
    __shared__ float warp_results[kWarpSize];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int warp_count = blockDim.x / kWarpSize;

    value = warp_op(value);
    __syncthreads();   // warp_results may still be read from a previous call
    if (lane == 0) {
        warp_results[warp] = value;
    }
    __syncthreads();
    value = lane < warp_count ? warp_results[lane] : identity;
    return warp_op(value);
}

__device__ __forceinline__ float BlockSum(float value) {
    return BlockReduce(value, 0.0f, [](float v) { return WarpSum(v); });
}

__device__ __forceinline__ float BlockMax(float value) {
    return BlockReduce(value, -FLT_MAX, [](float v) { return WarpMax(v); });
}

}  // namespace murmur
