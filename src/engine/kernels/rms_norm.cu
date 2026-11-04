#include "engine/kernels/rms_norm.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/block_reduce.cuh"
#include "engine/kernels/launch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kF16Max = 65504.0f;

__device__ __forceinline__ void Store(float* out, float value) {
    *out = value;
}

// Rounds to the nearest BF16 (ties to even).
__device__ __forceinline__ void Store(__nv_bfloat16* out, float value) {
    *out = __float2bfloat16_rn(value);
}

// Rounds to the nearest FP16, saturating at its largest finite value.
__device__ __forceinline__ void Store(__half* out, float value) {
    *out = __float2half_rn(fminf(fmaxf(value, -kF16Max), kF16Max));
}

// Widths up to this many values per thread keep the row in registers.
constexpr int kMaxItems = 16;

// One block per row: the block sums the squares together (BlockSum), then
// every thread scales its share of the row and stores it as Out. Each
// thread's Items values are loaded at once and kept for the second pass, so
// the block waits on memory once instead of once per value.
template <typename Out, int Items>
__global__ void RmsNormKernel(const float* in, const float* weight, Out* out, int width, float epsilon) {
    const float* x = in + static_cast<std::size_t>(blockIdx.x) * width;
    Out* y = out + static_cast<std::size_t>(blockIdx.x) * width;

    float values[Items];
#pragma unroll
    for (int k = 0; k < Items; ++k) {
        const int i = threadIdx.x + k * kThreadsPerBlock;
        values[k] = i < width ? x[i] : 0.0f;
    }
    float sum_of_squares = 0.0f;
#pragma unroll
    for (int k = 0; k < Items; ++k) {
        sum_of_squares += values[k] * values[k];
    }
    sum_of_squares = BlockSum(sum_of_squares);
    const float inverse_rms = 1.0f / sqrtf(sum_of_squares / width + epsilon);

#pragma unroll
    for (int k = 0; k < Items; ++k) {
        const int i = threadIdx.x + k * kThreadsPerBlock;
        if (i < width) {
            Store(&y[i], weight[i] * (values[k] * inverse_rms));
        }
    }
}

// The kernel with the fewest values per thread that covers `width`.
template <typename Out>
void LaunchRmsNorm(const float* in, const float* weight, Out* out, int rows, int width, float epsilon,
                   cudaStream_t stream) {
    RequireArgument(width <= kMaxItems * kThreadsPerBlock, "RmsNorm: row too wide");
    const auto launch = [&](auto kernel) {
        kernel<<<rows, kThreadsPerBlock, 0, stream>>>(in, weight, out, width, epsilon);
    };
    if (width <= 4 * kThreadsPerBlock) {
        launch(RmsNormKernel<Out, 4>);
    } else if (width <= 8 * kThreadsPerBlock) {
        launch(RmsNormKernel<Out, 8>);
    } else {
        launch(RmsNormKernel<Out, kMaxItems>);
    }
    CheckKernelLaunch();
}

}  // namespace

// Root-mean-square normalization of each row of a row-major rows x width
// matrix: y = weight * x / sqrt(mean(x^2) + epsilon). `out` may alias `in`.
void RmsNorm(const float* in, const float* weight, float* out, int rows, int width, float epsilon,
             cudaStream_t stream) {
    RequireArgument(rows >= 0 && width > 0 && epsilon > 0.0f, "RmsNorm: bad shape or epsilon");
    if (rows == 0) {
        return;
    }
    LaunchRmsNorm(in, weight, out, rows, width, epsilon, stream);
}

// RmsNorm with the result rounded to BF16, ready for a BF16 matrix product.
void RmsNormToBf16(const float* in, const float* weight, std::uint16_t* out, int rows, int width, float epsilon,
                   cudaStream_t stream) {
    RequireArgument(rows >= 0 && width > 0 && epsilon > 0.0f, "RmsNormToBf16: bad shape or epsilon");
    if (rows == 0) {
        return;
    }
    LaunchRmsNorm(in, weight, reinterpret_cast<__nv_bfloat16*>(out), rows, width, epsilon, stream);
}

// RmsNorm with the result rounded to FP16 (saturating), ready for GemmF16.
void RmsNormToF16(const float* in, const float* weight, std::uint16_t* out, int rows, int width, float epsilon,
                  cudaStream_t stream) {
    RequireArgument(rows >= 0 && width > 0 && epsilon > 0.0f, "RmsNormToF16: bad shape or epsilon");
    if (rows == 0) {
        return;
    }
    LaunchRmsNorm(in, weight, reinterpret_cast<__half*>(out), rows, width, epsilon, stream);
}

}  // namespace murmur
