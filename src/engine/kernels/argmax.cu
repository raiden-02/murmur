#include "engine/kernels/argmax.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/block_reduce.cuh"
#include "engine/kernels/launch.cuh"

#include <cmath>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 1024;

struct Best {
    float value;
    std::int32_t index;
};

// The better of two candidates: the larger value, or the smaller index on a
// tie (so the first maximum wins, as in torch.argmax).
__device__ __forceinline__ Best Better(Best a, Best b) {
    return (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
}

__device__ __forceinline__ Best WarpBest(Best best) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const Best other{__shfl_xor_sync(kFullWarpMask, best.value, offset),
                         __shfl_xor_sync(kFullWarpMask, best.index, offset)};
        best = Better(best, other);
    }
    return best;
}

// A single block scans all values (about 150 per thread for a vocabulary),
// then combines the per-thread winners warp by warp.
__global__ void ArgmaxKernel(const float* values, int count, std::int32_t* out_index) {
    __shared__ Best warp_best[kWarpSize];
    Best best{-INFINITY, 0};
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        best = Better(best, Best{values[i], i});
    }
    best = WarpBest(best);
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_best[warp] = best;
    }
    __syncthreads();
    if (warp == 0) {
        best = lane < blockDim.x / kWarpSize ? warp_best[lane] : Best{-INFINITY, 0};
        best = WarpBest(best);
        if (lane == 0) {
            *out_index = best.index;
        }
    }
}

}  // namespace

// Writes the index of the largest of `count` values to *out_index (GPU
// memory). Ties go to the lowest index. NaN values are never chosen; if
// nothing beats -infinity the result is 0.
void Argmax(const float* values, int count, std::int32_t* out_index, cudaStream_t stream) {
    RequireArgument(count > 0, "Argmax: count must be positive");
    ArgmaxKernel<<<1, kThreadsPerBlock, 0, stream>>>(values, count, out_index);
    CheckKernelLaunch();
}

}  // namespace murmur
