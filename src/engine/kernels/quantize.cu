#include "engine/kernels/quantize.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/block_reduce.cuh"
#include "engine/kernels/launch.cuh"
#include "engine/kernels/weight_decode.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstddef>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr float kFp8Max = 448.0f;   // the largest finite E4M3 value
constexpr int kInt4Levels = 15;     // q in 0..15
constexpr int kWeightsPerLane = kInt4GroupSize / kWarpSize;
static_assert(kWeightsPerLane == 4, "a lane packs its weights into two bytes");

// One block per row: the row's largest magnitude maps to E4M3's largest
// value, so the row uses the format's whole range; everything is divided by
// that scale and rounded to the nearest E4M3.
__global__ void QuantizeToFp8Kernel(const Bf16Bits* in, int cols, std::uint8_t* out, float* row_scales) {
    const Bf16Bits* row = in + static_cast<std::size_t>(blockIdx.x) * cols;
    std::uint8_t* row_out = out + static_cast<std::size_t>(blockIdx.x) * cols;
    float largest = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        largest = fmaxf(largest, fabsf(Bf16ToFloat(row[i])));
    }
    largest = BlockMax(largest);
    const float scale = largest > 0.0f ? largest / kFp8Max : 1.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] = __nv_cvt_float_to_fp8(Bf16ToFloat(row[i]) / scale, __NV_SATFINITE, __NV_E4M3);
    }
    if (threadIdx.x == 0) {
        row_scales[blockIdx.x] = scale;
    }
}

// The 4-bit code of w on the grid base + q * step, q in 0..15.
__device__ __forceinline__ float Int4Code(float w, float step, float base) {
    return step > 0.0f ? fminf(fmaxf(rintf((w - base) / step), 0.0f), kInt4Levels) : 0.0f;
}

// The group's rounding error on the grid spanning [low, high] (scale and
// min rounded to FP16, as stored): the sum over the warp of each weight's
// squared error times its column's importance.
__device__ float GroupError(const float (&w)[kWeightsPerLane], const float (&importance)[kWeightsPerLane], float low,
                            float high) {
    const float step = __half2float(__float2half_rn((high - low) / kInt4Levels));
    const float base = __half2float(__float2half_rn(low));
    float error = 0.0f;
#pragma unroll
    for (int i = 0; i < kWeightsPerLane; ++i) {
        const float d = Int4Code(w[i], step, base) * step + base - w[i];
        error += importance[i] * d * d;
    }
    return WarpSum(error);
}

// One warp per group of 128 consecutive weights of a row; each lane holds 4.
// The group's range is cut into 15 equal steps and every weight rounded to
// the nearest step (asymmetric round-to-nearest). Which range: spanning
// [min, max] wastes steps on a few outliers, so the kernel also tries
// clipping up to kMaxClip of the span off either end, and keeps the range
// with the smallest error (outliers then saturate, everything else gets
// finer steps). The error weighs each column by its importance (see
// ChannelImportance) when one is given, so columns that meet large
// activations are rounded most carefully. Scale and min are rounded to FP16
// before use, so codes are chosen against the values the kernels will
// actually reconstruct.
__global__ void QuantizeToInt4Kernel(const Bf16Bits* in, int groups, int groups_per_row,
                                     const float* __restrict__ column_importance, std::uint8_t* out,
                                     std::uint32_t* group_ranges) {
    constexpr int kClipSteps = 11;
    constexpr float kMaxClip = 0.2f;
    const int group = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpSize;
    if (group >= groups) {
        return;
    }
    const int lane = threadIdx.x % kWarpSize;
    const Bf16Bits* source = in + static_cast<std::size_t>(group) * kInt4GroupSize + lane * kWeightsPerLane;
    float w[kWeightsPerLane];
    float low = INFINITY;
    float high = -INFINITY;
#pragma unroll
    for (int i = 0; i < kWeightsPerLane; ++i) {
        w[i] = Bf16ToFloat(source[i]);
        low = fminf(low, w[i]);
        high = fmaxf(high, w[i]);
    }
    low = -WarpMax(-low);
    high = WarpMax(high);
    float importance[kWeightsPerLane];
    const int first_column = (group % groups_per_row) * kInt4GroupSize + lane * kWeightsPerLane;
#pragma unroll
    for (int i = 0; i < kWeightsPerLane; ++i) {
        importance[i] = column_importance != nullptr ? column_importance[first_column + i] : 1.0f;
    }

    const float span = high - low;
    float best_low = low;
    float best_high = high;
    float best_error = GroupError(w, importance, low, high);
    for (int a = 0; a < kClipSteps; ++a) {
        for (int b = 0; b < kClipSteps; ++b) {
            const float try_low = low + span * kMaxClip * a / (kClipSteps - 1);
            const float try_high = high - span * kMaxClip * b / (kClipSteps - 1);
            const float error = GroupError(w, importance, try_low, try_high);
            if (error < best_error) {   // every lane sees the same sums, so they agree
                best_error = error;
                best_low = try_low;
                best_high = try_high;
            }
        }
    }

    const __half scale = __float2half_rn((best_high - best_low) / kInt4Levels);
    const __half min = __float2half_rn(best_low);
    const float step = __half2float(scale);
    const float base = __half2float(min);
    unsigned packed = 0;
#pragma unroll
    for (int i = 0; i < kWeightsPerLane; ++i) {
        packed |= static_cast<unsigned>(Int4Code(w[i], step, base)) << (4 * i);
    }
    reinterpret_cast<std::uint16_t*>(out)[static_cast<std::size_t>(group) * (kInt4GroupSize / 4) + lane] =
        static_cast<std::uint16_t>(packed);
    if (lane == 0) {
        const __half2 range = __halves2half2(scale, min);
        group_ranges[group] = *reinterpret_cast<const std::uint32_t*>(&range);
    }
}

// One block per row: sums[c] += x[r][c]^2, after RMSNorm when norm_weight is given.
__global__ void AccumulateInputSquaresKernel(const float* x, int width, const float* norm_weight, float epsilon,
                                             float* sums) {
    const float* row = x + static_cast<std::size_t>(blockIdx.x) * width;
    float inverse_rms = 1.0f;
    if (norm_weight != nullptr) {
        float sum_of_squares = 0.0f;
        for (int i = threadIdx.x; i < width; i += blockDim.x) {
            sum_of_squares += row[i] * row[i];
        }
        inverse_rms = 1.0f / sqrtf(BlockSum(sum_of_squares) / width + epsilon);
    }
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = norm_weight != nullptr ? norm_weight[i] * (row[i] * inverse_rms) : row[i];
        atomicAdd(&sums[i], v * v);
    }
}

// One block per row: RMSNorm (optional), then the row's largest magnitude
// maps to E4M3's largest value and every value is rounded to E4M3.
__global__ void QuantizeRowsToFp8Kernel(const float* x, int width, const float* norm_weight, float epsilon,
                                        std::uint8_t* out, float* row_scales) {
    const float* row = x + static_cast<std::size_t>(blockIdx.x) * width;
    std::uint8_t* row_out = out + static_cast<std::size_t>(blockIdx.x) * width;
    float inverse_rms = 1.0f;
    if (norm_weight != nullptr) {
        float sum_of_squares = 0.0f;
        for (int i = threadIdx.x; i < width; i += blockDim.x) {
            sum_of_squares += row[i] * row[i];
        }
        inverse_rms = 1.0f / sqrtf(BlockSum(sum_of_squares) / width + epsilon);
    }
    const auto value = [&](int i) { return norm_weight != nullptr ? norm_weight[i] * (row[i] * inverse_rms) : row[i]; };
    float largest = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        largest = fmaxf(largest, fabsf(value(i)));
    }
    largest = BlockMax(largest);
    const float scale = largest > 0.0f ? largest / kFp8Max : 1.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        row_out[i] = __nv_cvt_float_to_fp8(value(i) / scale, __NV_SATFINITE, __NV_E4M3);
    }
    if (threadIdx.x == 0) {
        row_scales[blockIdx.x] = scale;
    }
}

// One block per output row: out row i = weight row row_ids[i] (or row i),
// decoded to FP32.
__global__ void DequantizeRowsKernel(WeightView w, const std::int32_t* row_ids, float* out) {
    const int row = row_ids != nullptr ? row_ids[blockIdx.x] : static_cast<int>(blockIdx.x);
    float* dst = out + static_cast<std::size_t>(blockIdx.x) * w.in_features;
    for (int i = threadIdx.x; i < w.in_features; i += blockDim.x) {
        dst[i] = DecodeWeight(w, row, i);
    }
}

}  // namespace

// Converts a rows x cols BF16 matrix (GPU memory) to E4M3 with one scale per
// row: out[r][c] * row_scales[r] approximates in[r][c]. A per-row scale lets
// rows with small weights use the format's whole range too.
void QuantizeToFp8(const Bf16Bits* in, int rows, int cols, std::uint8_t* out, float* row_scales, cudaStream_t stream) {
    RequireArgument(rows > 0 && cols > 0, "QuantizeToFp8: bad shape");
    QuantizeToFp8Kernel<<<rows, kThreadsPerBlock, 0, stream>>>(in, cols, out, row_scales);
    CheckKernelLaunch();
}

// Converts a rows x cols BF16 matrix (GPU memory) to 4-bit codes, two per
// byte (the even column in the low nibble), with a {scale, min} pair per
// group of kInt4GroupSize consecutive weights of a row: q * scale + min
// approximates the weight. cols must be a multiple of the group size.
// column_importance (GPU memory, cols values, or null for equal weights)
// steers each group's range towards the columns that matter most.
void QuantizeToInt4(const Bf16Bits* in, int rows, int cols, const float* column_importance, std::uint8_t* out,
                    std::uint32_t* group_ranges, cudaStream_t stream) {
    RequireArgument(rows > 0 && cols > 0 && cols % kInt4GroupSize == 0, "QuantizeToInt4: bad shape");
    const int groups_per_row = cols / kInt4GroupSize;
    const int groups = rows * groups_per_row;
    QuantizeToInt4Kernel<<<CeilDiv(groups, kWarpsPerBlock), kThreadsPerBlock, 0, stream>>>(
        in, groups, groups_per_row, column_importance, out, group_ranges);
    CheckKernelLaunch();
}

// Adds x_c^2 over `rows` rows of x (rows x width, FP32) into sums (width
// values, GPU memory), normalizing each row first when norm_weight is given:
// the calibration statistics of ChannelImportance.
void AccumulateInputSquares(const float* x, int rows, int width, const float* norm_weight, float epsilon, float* sums,
                            cudaStream_t stream) {
    RequireArgument(rows >= 0 && width > 0, "AccumulateInputSquares: bad shape");
    if (rows == 0) {
        return;
    }
    AccumulateInputSquaresKernel<<<rows, kThreadsPerBlock, 0, stream>>>(x, width, norm_weight, epsilon, sums);
    CheckKernelLaunch();
}

// Rounds each row of x (rows x width, FP32; RMS-normalized first when
// norm_weight is given) to E4M3 with one scale per row: out[r][c] *
// row_scales[r] approximates the value. Activations for FP8 tensor cores.
void QuantizeRowsToFp8(const float* x, int rows, int width, const float* norm_weight, float epsilon, std::uint8_t* out,
                       float* row_scales, cudaStream_t stream) {
    RequireArgument(rows >= 0 && width > 0, "QuantizeRowsToFp8: bad shape");
    if (rows == 0) {
        return;
    }
    QuantizeRowsToFp8Kernel<<<rows, kThreadsPerBlock, 0, stream>>>(x, width, norm_weight, epsilon, out, row_scales);
    CheckKernelLaunch();
}

// Decodes `count` rows of `weight` to FP32: out row i is weight row
// row_ids[i], or weight row i when row_ids is null. With row ids this is the
// embedding lookup; without, it widens a whole matrix. Row ids live in GPU
// memory and must be valid rows: the kernel trusts them.
void DequantizeRows(const WeightView& weight, const std::int32_t* row_ids, int count, float* out, cudaStream_t stream) {
    RequireArgument(count >= 0 && weight.in_features > 0 && (row_ids != nullptr || count <= weight.out_features),
                    "DequantizeRows: bad shape");
    if (count == 0) {
        return;
    }
    DequantizeRowsKernel<<<count, kThreadsPerBlock, 0, stream>>>(weight, row_ids, out);
    CheckKernelLaunch();
}

}  // namespace murmur
