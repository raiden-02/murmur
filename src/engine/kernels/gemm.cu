#include "engine/kernels/gemm.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/launch.cuh"
#include "engine/kernels/tensor_core.cuh"
#include "engine/kernels/weight_decode.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>

namespace murmur {

namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kBlockN = 64;                  // weight rows (outputs) per block
constexpr int kWarpN = kBlockN / kWarps;     // per warp: two 8-wide mma tiles
constexpr int kRowPad = 16;                  // bytes after each shared row, so fragment reads hit distinct banks
constexpr int kBlocksPerSm = 2;              // split K until this many blocks per SM exist
constexpr int kMaxSharedBytes = 99 * 1024;   // Ada's per-block opt-in limit
constexpr int kMaxBlockM = 128;              // activation rows per block, at most
static_assert(kWarpN == 16, "each warp computes two n8 tiles");

// The K extent of a tile, and the bytes it takes of one weight row, per format.
template <WeightFormat F>
struct WeightTile;
template <>
struct WeightTile<WeightFormat::kBf16> {
    static constexpr int kBlockK = 64;
    static constexpr int kRowBytes = 128;
};
template <>
struct WeightTile<WeightFormat::kFp8> {
    static constexpr int kBlockK = 128;
    static constexpr int kRowBytes = 128;
};
template <>
struct WeightTile<WeightFormat::kInt4> {
    static constexpr int kBlockK = kInt4GroupSize;   // one group per tile: one {scale, min} per row
    static constexpr int kRowBytes = 64;
};

// One pipeline stage in shared memory: an activation tile (kBlockM x kBlockK
// FP16), a weight tile (kBlockN rows, packed as in global memory) and, for
// INT4, each weight row's group range. Three stages are in flight (one
// computing, two loading) when they fit in shared memory, else two.
template <WeightFormat F, int kBlockM>
struct Stage {
    static constexpr int kBlockK = WeightTile<F>::kBlockK;
    static constexpr int kAPitch = kBlockK * 2 + kRowPad;
    static constexpr int kBPitch = WeightTile<F>::kRowBytes + kRowPad;
    static constexpr int kABytes = kBlockM * kAPitch;
    static constexpr int kBBytes = kBlockN * kBPitch;
    static constexpr int kRangeBytes = F == WeightFormat::kInt4 ? kBlockN * 4 : 0;
    static constexpr int kBytes = kABytes + kBBytes + kRangeBytes;
    static constexpr int kStages = 3 * kBytes <= kMaxSharedBytes ? 3 : 2;
    static constexpr int kSharedBytes = kStages * kBytes;
    static_assert(kABytes % 16 == 0 && kBBytes % 16 == 0, "16-byte copies need 16-byte aligned rows");
    static_assert(kSharedBytes <= kMaxSharedBytes, "stage too large");
};

// A B-fragment register: the weights at columns k, k + 1 of one row of the
// shared weight tile, as an FP16 pair. `k` is even and relative to the tile.
template <WeightFormat F>
__device__ __forceinline__ unsigned WeightPair(const unsigned char* row, int k, __half2 group_scale,
                                               __half2 group_min) {
    if constexpr (F == WeightFormat::kBf16) {
        const unsigned pair = *reinterpret_cast<const unsigned*>(row + 2 * k);
        return Bits(__floats2half2_rn(__uint_as_float(pair << 16), __uint_as_float(pair & 0xFFFF0000u)));
    } else if constexpr (F == WeightFormat::kFp8) {
        return Bits(Fp8PairToHalf2(*reinterpret_cast<const std::uint16_t*>(row + k)));
    } else {
        return Bits(__hfma2(NibblePairToHalf2(row[k / 2]), group_scale, group_min));
    }
}

// What happens to each pair of outputs (columns c, c + 1 of row r): the FP8
// row scales, then SwiGLU (columns c and c + 1 are a gate and its up row)
// or bias and residual add.
struct Epilogue {
    float* y;
    int columns;                 // weight rows
    const float* row_scales;     // FP8 weights: column c is scaled by row_scales[c]
    const float* bias;
    bool accumulate;
    bool swiglu;

    __device__ __forceinline__ void Store(int r, int c, float v0, float v1) const {
        if (row_scales != nullptr) {
            v0 *= row_scales[c];
            v1 *= row_scales[c + 1];
        }
        if (swiglu) {
            y[static_cast<std::size_t>(r) * (columns / 2) + c / 2] = v0 / (1.0f + expf(-v0)) * v1;
            return;
        }
        if (bias != nullptr) {
            v0 += bias[c];
            v1 += bias[c + 1];
        }
        auto* out = reinterpret_cast<float2*>(y + static_cast<std::size_t>(r) * columns + c);
        if (accumulate) {
            const float2 old = *out;
            v0 += old.x;
            v1 += old.y;
        }
        *out = {v0, v1};
    }
};

// Issues the copies of one K tile into `stage`.
template <WeightFormat F, int kBlockM>
__device__ __forceinline__ void LoadTile(unsigned char* stage, const __half* x, int rows, const WeightView& w,
                                         int row0, int n0, int k_tile) {
    using S = Stage<F, kBlockM>;
    constexpr int kAChunks = S::kBlockK * 2 / 16;
    for (int c = threadIdx.x; c < kBlockM * kAChunks; c += kThreads) {
        const int r = c / kAChunks;
        const int part = c % kAChunks;
        const bool valid = row0 + r < rows;
        const __half* source = x + static_cast<std::size_t>(valid ? row0 + r : 0) * w.in_features +
                               k_tile * S::kBlockK + part * 8;
        CopyAsync16(stage + r * S::kAPitch + part * 16, source, valid);
    }
    constexpr int kBChunks = WeightTile<F>::kRowBytes / 16;
    const std::size_t row_bytes = static_cast<std::size_t>(w.in_features / S::kBlockK) * WeightTile<F>::kRowBytes;
    unsigned char* b = stage + S::kABytes;
    for (int c = threadIdx.x; c < kBlockN * kBChunks; c += kThreads) {
        const int r = c / kBChunks;
        const int part = c % kBChunks;
        const bool valid = n0 + r < w.out_features;
        const unsigned char* source = w.data + static_cast<std::size_t>(valid ? n0 + r : 0) * row_bytes +
                                      static_cast<std::size_t>(k_tile) * WeightTile<F>::kRowBytes + part * 16;
        CopyAsync16(b + r * S::kBPitch + part * 16, source, valid);
    }
    if constexpr (F == WeightFormat::kInt4) {
        const int groups = w.in_features / kInt4GroupSize;
        unsigned char* ranges = b + S::kBBytes;
        for (int r = threadIdx.x; r < kBlockN; r += kThreads) {
            const int n = min(n0 + r, w.out_features - 1);
            CopyAsync4(ranges + 4 * r, w.group_ranges + static_cast<std::size_t>(n) * groups + k_tile);
        }
    }
}

// The tensor-core work on one K tile in shared memory: every warp multiplies
// all kBlockM activation rows by its 16 weight rows, 16 columns of K at a time.
template <WeightFormat F, int kBlockM>
__device__ __forceinline__ void ComputeTile(const unsigned char* stage, float (&acc)[kBlockM / 16][2][4]) {
    using S = Stage<F, kBlockM>;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g = lane / 4;   // the fragment's row (A) or column (B) within an 8-wide group
    const int t = lane % 4;   // its pair of K columns

    const unsigned char* rows[2];
    __half2 group_scale[2] = {};
    __half2 group_min[2] = {};
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int r = warp * kWarpN + nt * 8 + g;
        rows[nt] = stage + S::kABytes + r * S::kBPitch;
        if constexpr (F == WeightFormat::kInt4) {
            const __half2 range = *reinterpret_cast<const __half2*>(stage + S::kABytes + S::kBBytes + 4 * r);
            group_scale[nt] = __low2half2(range);
            group_min[nt] = __high2half2(range);
        }
    }

#pragma unroll
    for (int kk = 0; kk < S::kBlockK / 16; ++kk) {
        unsigned a[kBlockM / 16][4];
#pragma unroll
        for (int mt = 0; mt < kBlockM / 16; ++mt) {
            LoadMatrixX4(a[mt], stage + (mt * 16 + lane % 16) * S::kAPitch + (kk * 16 + (lane / 16) * 8) * 2);
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const unsigned b0 = WeightPair<F>(rows[nt], kk * 16 + 2 * t, group_scale[nt], group_min[nt]);
            const unsigned b1 = WeightPair<F>(rows[nt], kk * 16 + 2 * t + 8, group_scale[nt], group_min[nt]);
#pragma unroll
            for (int mt = 0; mt < kBlockM / 16; ++mt) {
                Mma(acc[mt][nt], a[mt], b0, b1);
            }
        }
    }
}

// y = x . W^T for FP16 activations x (rows x K) and weights W in format F
// (N x K), on tensor cores. Block (m, n, split) computes rows [m * kBlockM,
// ...) x outputs [n * 64, ...) over its share of K, streaming K tiles
// through a cp.async pipeline of S::kStages tiles. Weights are unpacked to FP16 as
// they are read from shared memory, so global memory only ever moves the
// packed bytes. With one split the block applies the epilogue itself;
// otherwise it writes FP32 partial sums for ReduceSplitsKernel.
template <WeightFormat F, int kBlockM>
__global__ void __launch_bounds__(kThreads)
    GemmKernel(const __half* __restrict__ x, int rows, WeightView w, int tiles_per_split, Epilogue epilogue,
               float* __restrict__ partials) {
    using S = Stage<F, kBlockM>;
    extern __shared__ __align__(16) unsigned char shared[];
    const int row0 = blockIdx.x * kBlockM;
    const int n0 = blockIdx.y * kBlockN;
    const int split = blockIdx.z;
    const int k_tiles = w.in_features / S::kBlockK;
    const int first_tile = split * tiles_per_split;
    const int tiles = min(tiles_per_split, k_tiles - first_tile);

    float acc[kBlockM / 16][2][4] = {};
#pragma unroll
    for (int s = 0; s < S::kStages - 1; ++s) {
        if (s < tiles) {
            LoadTile<F, kBlockM>(shared + s * S::kBytes, x, rows, w, row0, n0, first_tile + s);
        }
        CommitCopies();
    }
    for (int i = 0; i < tiles; ++i) {
        WaitCopies<S::kStages - 2>();   // tile i has landed
        __syncthreads();                // ... for every thread, and tile i - 1's stage is free again
        const int next = i + S::kStages - 1;
        if (next < tiles) {
            LoadTile<F, kBlockM>(shared + (next % S::kStages) * S::kBytes, x, rows, w, row0, n0, first_tile + next);
        }
        CommitCopies();
        ComputeTile<F, kBlockM>(shared + (i % S::kStages) * S::kBytes, acc);
    }

    // Thread (g, t) holds rows g and g + 8 of each 16-row tile, columns 2t, 2t + 1 of each 8-wide tile.
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int mt = 0; mt < kBlockM / 16; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int c = n0 + warp * kWarpN + nt * 8 + 2 * (lane % 4);
            if (c >= w.out_features) {
                continue;
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int r = row0 + mt * 16 + lane / 4 + 8 * half;
                if (r >= rows) {
                    continue;
                }
                const float v0 = acc[mt][nt][2 * half];
                const float v1 = acc[mt][nt][2 * half + 1];
                if (gridDim.z == 1) {
                    epilogue.Store(r, c, v0, v1);
                } else {
                    *reinterpret_cast<float2*>(partials + (static_cast<std::size_t>(split) * rows + r) *
                                                              w.out_features + c) = {v0, v1};
                }
            }
        }
    }
}

// Scales FP8 tensor-core products back to real values (by the activation
// row's scale; the epilogue applies the weight row's) and applies the
// epilogue: one thread per pair of outputs.
__global__ void FinishScaledProductsKernel(const float* __restrict__ products, const float* __restrict__ x_scales,
                                           int rows, Epilogue epilogue) {
    const int pairs = epilogue.columns / 2;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * pairs) {
        return;
    }
    const int r = index / pairs;
    const int c = 2 * (index % pairs);
    const float2 product = *reinterpret_cast<const float2*>(products + static_cast<std::size_t>(r) * epilogue.columns + c);
    epilogue.Store(r, c, product.x * x_scales[r], product.y * x_scales[r]);
}

// Adds up the splits' partial sums, in split order (deterministic), and
// applies the epilogue: one thread per pair of outputs.
__global__ void ReduceSplitsKernel(const float* __restrict__ partials, int splits, int rows, Epilogue epilogue) {
    const int pairs = epilogue.columns / 2;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * pairs) {
        return;
    }
    const int r = index / pairs;
    const int c = 2 * (index % pairs);
    float2 sum = {0.0f, 0.0f};
    for (int s = 0; s < splits; ++s) {
        const float2 part = *reinterpret_cast<const float2*>(
            partials + (static_cast<std::size_t>(s) * rows + r) * epilogue.columns + c);
        sum.x += part.x;
        sum.y += part.y;
    }
    epilogue.Store(r, c, sum.x, sum.y);
}

template <WeightFormat F>
constexpr int BlockK() {
    return WeightTile<F>::kBlockK;
}

int BlockKFor(WeightFormat format) {
    switch (format) {
    case WeightFormat::kBf16:
        return BlockK<WeightFormat::kBf16>();
    case WeightFormat::kFp8:
        return BlockK<WeightFormat::kFp8>();
    case WeightFormat::kInt4:
        return BlockK<WeightFormat::kInt4>();
    }
    return 0;
}

// Lets the kernels use more than the default 48 KB of shared memory.
template <WeightFormat F, int kBlockM>
void AllowSharedMemory() {
    CudaCheck(cudaFuncSetAttribute(GemmKernel<F, kBlockM>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Stage<F, kBlockM>::kSharedBytes));
}

template <WeightFormat F>
void AllowSharedMemoryForFormat() {
    AllowSharedMemory<F, 16>();
    AllowSharedMemory<F, 32>();
    AllowSharedMemory<F, 64>();
    AllowSharedMemory<F, kMaxBlockM>();
}

template <WeightFormat F, int kBlockM>
void Launch(const std::uint16_t* x, int rows, const WeightView& w, const Epilogue& epilogue, GemmWorkspace& workspace,
            cudaStream_t stream) {
    const int m_tiles = CeilDiv(rows, kBlockM);
    const int n_tiles = CeilDiv(w.out_features, kBlockN);
    const int k_tiles = w.in_features / BlockK<F>();
    // Few output tiles (short matrices, few rows) can't fill the GPU: split K
    // until about kBlocksPerSm blocks per SM exist, within the workspace.
    int splits = 1;
    const int blocks = m_tiles * n_tiles;
    const int wanted = kBlocksPerSm * workspace.SmCount();
    if (blocks < wanted) {
        splits = std::min(CeilDiv(wanted, blocks), k_tiles);
        const std::size_t per_split = static_cast<std::size_t>(rows) * w.out_features;
        splits = std::max(1, std::min<int>(splits, static_cast<int>(workspace.Capacity() / per_split)));
    }
    const int tiles_per_split = CeilDiv(k_tiles, splits);
    splits = CeilDiv(k_tiles, tiles_per_split);   // no empty splits

    const dim3 grid(m_tiles, n_tiles, splits);
    GemmKernel<F, kBlockM><<<grid, kThreads, Stage<F, kBlockM>::kSharedBytes, stream>>>(
        reinterpret_cast<const __half*>(x), rows, w, tiles_per_split, epilogue, workspace.Partials());
    CheckKernelLaunch();
    if (splits > 1) {
        const int pairs = rows * (w.out_features / 2);
        ReduceSplitsKernel<<<CeilDiv(pairs, 256), 256, 0, stream>>>(workspace.Partials(), splits, rows, epilogue);
        CheckKernelLaunch();
    }
}

template <WeightFormat F>
void LaunchForRows(const std::uint16_t* x, int rows, const WeightView& w, const Epilogue& epilogue,
                   GemmWorkspace& workspace, cudaStream_t stream) {
    // Taller blocks unpack each weight once for more rows (prompt chunks);
    // shorter ones waste less on padding rows (decode batches).
    if (rows <= 16) {
        Launch<F, 16>(x, rows, w, epilogue, workspace, stream);
    } else if (rows <= 32) {
        Launch<F, 32>(x, rows, w, epilogue, workspace, stream);
    } else if (rows <= 64) {
        Launch<F, 64>(x, rows, w, epilogue, workspace, stream);
    } else {
        Launch<F, kMaxBlockM>(x, rows, w, epilogue, workspace, stream);
    }
}

}  // namespace

// Room for the partial sums of any split launch: splitting only happens
// while there are fewer than kBlocksPerSm * SMs output tiles, so the splits
// together hold at most about twice that many tiles of kMaxBlockM x 64 sums. Also
// raises the kernels' shared-memory limits, once, before any launch (and so
// before any graph recording).
GemmWorkspace::GemmWorkspace() {
    int device = 0;
    CudaCheck(cudaGetDevice(&device));
    CudaCheck(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device));
    partials_ = DeviceBuffer<float>(static_cast<std::size_t>(2 * kBlocksPerSm * sm_count_) * kMaxBlockM * kBlockN);
    AllowSharedMemoryForFormat<WeightFormat::kBf16>();
    AllowSharedMemoryForFormat<WeightFormat::kFp8>();
    AllowSharedMemoryForFormat<WeightFormat::kInt4>();
}

// Whether GemmF16 handles this matrix: K a multiple of the format's tile
// depth and an even number of outputs (every model matrix).
bool GemmF16Supports(const WeightView& weight) {
    return weight.in_features > 0 && weight.in_features % BlockKFor(weight.format) == 0 &&
           weight.out_features > 0 && weight.out_features % 2 == 0;
}

// A linear layer for several rows on tensor cores, with weights in any
// format: y = x . W^T, x FP16 (rows x in_features, raw bits, row-major), y
// FP32. Bias, residual add (accumulate) and SwiGLU (W interleaving gate and
// up rows; y is then rows x out_features / 2) as in MatVecOptions; RMSNorm
// must already be applied to x. Products are summed in FP32.
void GemmF16(const std::uint16_t* x, int rows, const WeightView& weight, float* y, const MatVecOptions& options,
             GemmWorkspace& workspace, cudaStream_t stream) {
    RequireArgument(rows >= 0 && GemmF16Supports(weight), "GemmF16: unsupported shape");
    RequireArgument(options.norm_weight == nullptr, "GemmF16: normalize x before the product");
    RequireArgument(!options.swiglu || (options.bias == nullptr && !options.accumulate),
                    "GemmF16: swiglu takes no bias or accumulate");
    if (rows == 0) {
        return;
    }
    const Epilogue epilogue{.y = y,
                            .columns = weight.out_features,
                            .row_scales = weight.format == WeightFormat::kFp8 ? weight.row_scales : nullptr,
                            .bias = options.bias,
                            .accumulate = options.accumulate,
                            .swiglu = options.swiglu};
    switch (weight.format) {
    case WeightFormat::kBf16:
        LaunchForRows<WeightFormat::kBf16>(x, rows, weight, epilogue, workspace, stream);
        break;
    case WeightFormat::kFp8:
        LaunchForRows<WeightFormat::kFp8>(x, rows, weight, epilogue, workspace, stream);
        break;
    case WeightFormat::kInt4:
        LaunchForRows<WeightFormat::kInt4>(x, rows, weight, epilogue, workspace, stream);
        break;
    }
}

// Turns the raw products of FP8 codes (rows x out_features, from
// Blas::LinearFp8) into the layer's output: each product times its
// activation row's scale and its weight row's scale, then bias, residual add
// or SwiGLU as in MatVecOptions. `products` may be `y` itself unless the
// options accumulate or apply SwiGLU.
void FinishFp8Products(const float* products, int rows, const float* x_scales, const WeightView& weight, float* y,
                       const MatVecOptions& options, cudaStream_t stream) {
    RequireArgument(rows >= 0 && weight.format == WeightFormat::kFp8 && weight.out_features % 2 == 0,
                    "FinishFp8Products: FP8 weights with an even number of outputs");
    RequireArgument(products != y || (!options.accumulate && !options.swiglu),
                    "FinishFp8Products: accumulate and swiglu need separate products");
    if (rows == 0) {
        return;
    }
    const Epilogue epilogue{.y = y,
                            .columns = weight.out_features,
                            .row_scales = weight.row_scales,
                            .bias = options.bias,
                            .accumulate = options.accumulate,
                            .swiglu = options.swiglu};
    const int pairs = rows * (weight.out_features / 2);
    FinishScaledProductsKernel<<<CeilDiv(pairs, 256), 256, 0, stream>>>(products, x_scales, rows, epilogue);
    CheckKernelLaunch();
}

}  // namespace murmur
