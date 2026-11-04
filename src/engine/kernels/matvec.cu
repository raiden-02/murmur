#include "engine/kernels/matvec.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/block_reduce.cuh"
#include "engine/kernels/launch.cuh"
#include "engine/kernels/weight_decode.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kWeightsPerLoad = 8;          // one load: 8 consecutive weights of a row
constexpr int kBytesInFlightPerLane = 128;  // loads issued before any is used
constexpr int kLoadsPerGroup = kInt4GroupSize / kWeightsPerLoad;

// What one load brings in: 8 weights in each format.
template <WeightFormat F>
struct Packed8;
template <>
struct Packed8<WeightFormat::kBf16> {
    using Bits = uint4;      // 8 x 16 bits
};
template <>
struct Packed8<WeightFormat::kFp8> {
    using Bits = uint2;      // 8 x 8 bits
};
template <>
struct Packed8<WeightFormat::kInt4> {
    using Bits = unsigned;   // 8 x 4 bits
};

__device__ __forceinline__ float Bf16Low(unsigned pair) {
    return __uint_as_float(pair << 16);
}

__device__ __forceinline__ float Bf16High(unsigned pair) {
    return __uint_as_float(pair & 0xFFFF0000u);
}

// Dot products of 8 weights with 8 FP32 activations (a = x[0..3], b =
// x[4..7]), one per format. FP8 leaves out the row's scale (applied once at
// the end); INT4 applies its group's: sum (q * s + m) * x = s * sum q x + m * sum x.
__device__ __forceinline__ float Dot8(uint4 w, float4 a, float4 b) {
    float sum = Bf16Low(w.x) * a.x;
    sum += Bf16High(w.x) * a.y;
    sum += Bf16Low(w.y) * a.z;
    sum += Bf16High(w.y) * a.w;
    sum += Bf16Low(w.z) * b.x;
    sum += Bf16High(w.z) * b.y;
    sum += Bf16Low(w.w) * b.z;
    sum += Bf16High(w.w) * b.w;
    return sum;
}

__device__ __forceinline__ float Dot8(uint2 w, float4 a, float4 b) {
    const float2 w01 = Fp8PairToFloat2(static_cast<std::uint16_t>(w.x));
    const float2 w23 = Fp8PairToFloat2(static_cast<std::uint16_t>(w.x >> 16));
    const float2 w45 = Fp8PairToFloat2(static_cast<std::uint16_t>(w.y));
    const float2 w67 = Fp8PairToFloat2(static_cast<std::uint16_t>(w.y >> 16));
    float sum = w01.x * a.x;
    sum += w01.y * a.y;
    sum += w23.x * a.z;
    sum += w23.y * a.w;
    sum += w45.x * b.x;
    sum += w45.y * b.y;
    sum += w67.x * b.z;
    sum += w67.y * b.w;
    return sum;
}

__device__ __forceinline__ float Dot8(unsigned w, float4 a, float4 b, std::uint32_t packed_range) {
    float dot = NibbleToFloat(w, 0) * a.x;
    dot += NibbleToFloat(w, 1) * a.y;
    dot += NibbleToFloat(w, 2) * a.z;
    dot += NibbleToFloat(w, 3) * a.w;
    dot += NibbleToFloat(w, 4) * b.x;
    dot += NibbleToFloat(w, 5) * b.y;
    dot += NibbleToFloat(w, 6) * b.z;
    dot += NibbleToFloat(w, 7) * b.w;
    const float x_sum = (a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w);
    const float2 range = GroupRange(packed_range);
    return range.x * dot + range.y * x_sum;
}

// weight * (x * inverse_rms), element-wise: RMSNorm in the same order as the
// reference (scale first, then the learned weight).
__device__ __forceinline__ float4 Normalize(float4 x, float4 weight, float inverse_rms) {
    return {weight.x * (x.x * inverse_rms), weight.y * (x.y * inverse_rms), weight.z * (x.z * inverse_rms),
            weight.w * (x.w * inverse_rms)};
}

__device__ __forceinline__ float Silu(float g) {
    return g / (1.0f + expf(-g));
}

// y = W.x for a weight matrix W in format F and an FP32 vector x, when
// in_features is a multiple of 8 (every model shape).
//
// Decode reads every weight exactly once, so this kernel is about keeping
// enough bytes in flight to saturate DRAM. Each warp owns kRowsPerWarp rows.
// A load brings 8 consecutive weights of a row (16 bytes in BF16, 8 in FP8,
// 4 in INT4), coalesced across the warp, so lanes always read 8 consecutive
// activations each and x is read the same way in every format. Each lane
// issues 128 bytes of loads, spread over its rows, before using any, so
// memory latency overlaps instead of adding up; the narrower the format, the
// more loads that takes. x is small and shared by every warp, so it's read
// through the cache (__ldg); weights are read once, so __ldcs marks them as
// streaming. Lane partial sums meet in a warp shuffle.
//
// Each activation a lane loads serves all of its warp's rows, so more rows
// per warp means fewer activation bytes per weight byte: narrow formats need
// that (an INT4 load of 4 bytes needs 32 bytes of activations).
//
// Fused work (see MatVecOptions):
//   kNorm:   the block first computes x's RMS (every block computes the same
//            value, deterministically), then normalizes x as it reads it.
//   kSwiGlu: the warp's rows come in (gate, up) pairs, rows 2o and 2o + 1,
//            and it stores SiLU(gate) * up as output o.
template <WeightFormat F, int kRowsPerWarp, bool kNorm, bool kSwiGlu>
__global__ void __launch_bounds__(kThreadsPerBlock)
    MatVecVectorKernel(WeightView w, const float* __restrict__ x, float* __restrict__ y,
                       const float* __restrict__ norm_weight, float norm_epsilon, const float* __restrict__ bias,
                       bool accumulate) {
    static_assert(!kSwiGlu || kRowsPerWarp % 2 == 0, "SwiGLU pairs each gate row with its up row");
    using Bits = typename Packed8<F>::Bits;
    constexpr bool kInt4 = F == WeightFormat::kInt4;
    constexpr int kLoadsInFlight = kBytesInFlightPerLane / static_cast<int>(sizeof(Bits)) / kRowsPerWarp;
    const int vectors = w.in_features / kWeightsPerLoad;
    const int groups = w.in_features / kInt4GroupSize;
    const auto* x_vectors = reinterpret_cast<const float4*>(x);
    const auto* norm_vectors = reinterpret_cast<const float4*>(norm_weight);

    float inverse_rms = 1.0f;
    if constexpr (kNorm) {
        float sum_of_squares = 0.0f;
        for (int i = threadIdx.x; i < w.in_features; i += blockDim.x) {
            const float v = x[i];
            sum_of_squares += v * v;
        }
        inverse_rms = 1.0f / sqrtf(BlockSum(sum_of_squares) / w.in_features + norm_epsilon);
    }

    const int lane = threadIdx.x % kWarpSize;
    const int warp_index = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpSize;
    const int first_row = warp_index * kRowsPerWarp;
    if (first_row >= w.out_features) {
        return;   // past the end; no block-level sync follows
    }

    // Rows past the end read row 0 instead (always valid) and are never stored.
    const Bits* rows[kRowsPerWarp];
    int row_index[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        row_index[r] = first_row + r < w.out_features ? first_row + r : 0;
        rows[r] = reinterpret_cast<const Bits*>(w.data) + static_cast<std::size_t>(row_index[r]) * vectors;
    }

    float sums[kRowsPerWarp] = {};
    for (int base = lane; base < vectors; base += kWarpSize * kLoadsInFlight) {
        Bits bits[kRowsPerWarp][kLoadsInFlight];
        std::uint32_t ranges[kRowsPerWarp][kInt4 ? kLoadsInFlight : 1];
#pragma unroll
        for (int k = 0; k < kLoadsInFlight; ++k) {
            const int v = base + k * kWarpSize;
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                bits[r][k] = v < vectors ? __ldcs(rows[r] + v) : Bits{};
                if constexpr (kInt4) {
                    ranges[r][k] = v < vectors ? __ldg(w.group_ranges + static_cast<std::size_t>(row_index[r]) * groups +
                                                       v / kLoadsPerGroup)
                                               : 0u;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < kLoadsInFlight; ++k) {
            const int v = base + k * kWarpSize;
            if (v < vectors) {
                float4 a = __ldg(x_vectors + 2 * v);
                float4 b = __ldg(x_vectors + 2 * v + 1);
                if constexpr (kNorm) {
                    a = Normalize(a, __ldg(norm_vectors + 2 * v), inverse_rms);
                    b = Normalize(b, __ldg(norm_vectors + 2 * v + 1), inverse_rms);
                }
#pragma unroll
                for (int r = 0; r < kRowsPerWarp; ++r) {
                    if constexpr (kInt4) {
                        sums[r] += Dot8(bits[r][k], a, b, ranges[r][k]);
                    } else {
                        sums[r] += Dot8(bits[r][k], a, b);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        sums[r] = WarpSum(sums[r]);
        if constexpr (F == WeightFormat::kFp8) {
            sums[r] *= w.row_scales[row_index[r]];
        }
    }
    if (lane != 0) {
        return;
    }
    if constexpr (kSwiGlu) {
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; r += 2) {
            if (first_row + r < w.out_features) {
                y[(first_row + r) / 2] = Silu(sums[r]) * sums[r + 1];
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row = first_row + r;
            if (row < w.out_features) {
                const float value = bias != nullptr ? sums[r] + bias[row] : sums[r];
                y[row] = accumulate ? y[row] + value : value;
            }
        }
    }
}

// Any width: one warp per row, one weight at a time, bias and accumulate only.
// Not used by the model (its widths are multiples of 8); kept so every shape
// gives correct results.
__global__ void MatVecScalarKernel(WeightView w, const float* __restrict__ x, float* __restrict__ y,
                                   const float* __restrict__ bias, bool accumulate) {
    const int lane = threadIdx.x % kWarpSize;
    const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpSize;
    if (row >= w.out_features) {
        return;
    }
    float sum = 0.0f;
    for (int i = lane; i < w.in_features; i += kWarpSize) {
        sum += DecodeWeight(w, row, i) * x[i];
    }
    sum = WarpSum(sum);
    if (lane == 0) {
        const float value = bias != nullptr ? sum + bias[row] : sum;
        y[row] = accumulate ? y[row] + value : value;
    }
}

template <WeightFormat F, int kRowsPerWarp, bool kNorm, bool kSwiGlu>
void LaunchVector(const WeightView& w, const float* x, float* y, const MatVecOptions& o, cudaStream_t stream) {
    const int warps = CeilDiv(w.out_features, kRowsPerWarp);
    MatVecVectorKernel<F, kRowsPerWarp, kNorm, kSwiGlu><<<CeilDiv(warps, kWarpsPerBlock), kThreadsPerBlock, 0,
                                                          stream>>>(w, x, y, o.norm_weight, o.norm_epsilon, o.bias,
                                                                    o.accumulate);
}

template <WeightFormat F, bool kNorm, bool kSwiGlu>
void LaunchVectorWithRows(int rows_per_warp, const WeightView& w, const float* x, float* y, const MatVecOptions& o,
                          cudaStream_t stream) {
    switch (rows_per_warp) {
    case 1:
        if constexpr (!kSwiGlu) {
            LaunchVector<F, 1, kNorm, kSwiGlu>(w, x, y, o, stream);
        }
        break;
    case 2:
        LaunchVector<F, 2, kNorm, kSwiGlu>(w, x, y, o, stream);
        break;
    case 4:
        LaunchVector<F, 4, kNorm, kSwiGlu>(w, x, y, o, stream);
        break;
    default:
        LaunchVector<F, 8, kNorm, kSwiGlu>(w, x, y, o, stream);
        break;
    }
}

// Rows per warp: tall matrices amortize more loads (and activation reads)
// per warp, while short ones need few rows per warp to give all 80 SMs work.
// Narrower formats want more rows, since each weight byte needs more
// activation bytes (measured with `murmur-lab matvec --weights ...`).
int RowsPerWarp(WeightFormat format, int rows, bool swiglu) {
    const int narrowness = format == WeightFormat::kBf16 ? 1 : format == WeightFormat::kFp8 ? 2 : 4;
    const int base = swiglu ? 2 : rows >= 16384 ? 4 : rows >= 4096 ? 2 : 1;
    return std::min(8, base * narrowness);
}

template <WeightFormat F, bool kNorm>
void LaunchVectorForShape(const WeightView& w, const float* x, float* y, const MatVecOptions& o,
                          cudaStream_t stream) {
    const int rows_per_warp = RowsPerWarp(F, w.out_features, o.swiglu);
    if (o.swiglu) {
        LaunchVectorWithRows<F, kNorm, true>(rows_per_warp, w, x, y, o, stream);
    } else {
        LaunchVectorWithRows<F, kNorm, false>(rows_per_warp, w, x, y, o, stream);
    }
}

template <WeightFormat F>
void LaunchVectorForFormat(const WeightView& w, const float* x, float* y, const MatVecOptions& o,
                           cudaStream_t stream) {
    if (o.norm_weight != nullptr) {
        LaunchVectorForShape<F, true>(w, x, y, o, stream);
    } else {
        LaunchVectorForShape<F, false>(w, x, y, o, stream);
    }
}

}  // namespace

// One token through a linear layer: y = W.x (+ bias) with W in any weight
// format (out_features x in_features) and x, y FP32, plus the fused work in
// `options`. With swiglu, W interleaves gate and up rows and y receives
// out_features / 2 values. `x`, `y` and `norm_weight` must not overlap; all
// pointers must be 16-byte aligned (cudaMalloc's are).
void MatVec(const WeightView& weight, const float* x, float* y, const MatVecOptions& options, cudaStream_t stream) {
    const int rows = weight.out_features;
    const int in = weight.in_features;
    RequireArgument(rows > 0 && in > 0 && weight.data != nullptr, "MatVec: bad shape");
    RequireArgument(!options.swiglu || (rows % 2 == 0 && options.bias == nullptr && !options.accumulate),
                    "MatVec: swiglu needs an even row count and no bias or accumulate");
    RequireArgument(options.norm_weight == nullptr || options.norm_epsilon > 0.0f,
                    "MatVec: RMSNorm needs a positive epsilon");
    if (in % kWeightsPerLoad != 0) {
        RequireArgument(options.norm_weight == nullptr && !options.swiglu,
                        "MatVec: norm and swiglu need in_features to be a multiple of 8");
        MatVecScalarKernel<<<CeilDiv(rows, kWarpsPerBlock), kThreadsPerBlock, 0, stream>>>(weight, x, y, options.bias,
                                                                                           options.accumulate);
        CheckKernelLaunch();
        return;
    }
    switch (weight.format) {
    case WeightFormat::kBf16:
        LaunchVectorForFormat<WeightFormat::kBf16>(weight, x, y, options, stream);
        break;
    case WeightFormat::kFp8:
        LaunchVectorForFormat<WeightFormat::kFp8>(weight, x, y, options, stream);
        break;
    case WeightFormat::kInt4:
        LaunchVectorForFormat<WeightFormat::kInt4>(weight, x, y, options, stream);
        break;
    }
    CheckKernelLaunch();
}

}  // namespace murmur
