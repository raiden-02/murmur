#include "engine/kernels/elementwise.cuh"

#include "common/cuda_check.hpp"
#include "common/device_info.hpp"
#include "engine/kernels/launch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kF16Max = 65504.0f;

// Enough blocks to fill the GPU several times over (16 per SM); each thread
// then walks the array with a grid-sized stride, so any count works with
// one launch.
constexpr int kBlocksPerSm = 16;

int GridFor(std::size_t count) {
    const int needed = CeilDiv(count, kThreadsPerBlock);
    const int most = kBlocksPerSm * CurrentDeviceSmCount();
    return needed < most ? needed : most;
}

__global__ void ConvertBf16ToF32Kernel(const std::uint16_t* in, float* out, std::size_t count) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        // BF16 is the top half of an FP32: shifting it back up is exact.
        out[i] = __uint_as_float(static_cast<unsigned>(in[i]) << 16);
    }
}

__global__ void ConvertF32ToBf16Kernel(const float* in, std::uint16_t* out, std::size_t count) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(in[i]));
    }
}

__global__ void ConvertF32ToF16Kernel(const float* in, std::uint16_t* out, std::size_t count) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        out[i] = __half_as_ushort(__float2half_rn(fminf(fmaxf(in[i], -kF16Max), kF16Max)));
    }
}

// The grid strides over every row at once: a few rows still fill the GPU.
__global__ void AddBiasKernel(float* x, const float* bias, int width, std::size_t count) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        x[i] += bias[i % width];
    }
}

// Output i of every row at once: SiLU(gate) * up of pair i, from one 8-byte
// load. With `rounded`, also stores it as BF16 there.
__global__ void SwiGluPairsKernel(const float2* pairs, float* out, __nv_bfloat16* rounded, std::size_t count) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float2 gate_up = pairs[i];
        const float y = gate_up.x / (1.0f + expf(-gate_up.x)) * gate_up.y;
        out[i] = y;
        if (rounded != nullptr) {
            rounded[i] = __float2bfloat16_rn(y);
        }
    }
}

// One block per gathered row: out row r = in row row_indices[r].
__global__ void GatherRowsKernel(const float* in, const int* row_indices, int width, float* out) {
    const float* src = in + static_cast<std::size_t>(row_indices[blockIdx.x]) * width;
    float* dst = out + static_cast<std::size_t>(blockIdx.x) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace

// Widens BF16 values (stored as raw 16-bit patterns) to FP32. Lossless.
void ConvertBf16ToF32(const std::uint16_t* in, float* out, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    ConvertBf16ToF32Kernel<<<GridFor(count), kThreadsPerBlock, 0, stream>>>(in, out, count);
    CheckKernelLaunch();
}

// x[r][c] += bias[c] for a row-major rows x width matrix.
void AddBias(float* x, const float* bias, int rows, int width, cudaStream_t stream) {
    RequireArgument(rows >= 0 && width > 0, "AddBias: bad shape");
    if (rows == 0) {
        return;
    }
    const std::size_t count = static_cast<std::size_t>(rows) * width;
    AddBiasKernel<<<GridFor(count), kThreadsPerBlock, 0, stream>>>(x, bias, width, count);
    CheckKernelLaunch();
}

// The SwiGLU activation for rows of interleaved gate and up projections
// (each row 2 * width wide: gate 0, up 0, gate 1, ...): out = SiLU(gate) *
// up with SiLU(g) = g / (1 + e^-g). out is rows x width. If `rounded` isn't
// null it receives the same results rounded to BF16 (raw 16-bit patterns),
// ready for the next BF16 product without a separate conversion pass.
void SwiGluPairs(const float* gate_up, float* out, int rows, int width, cudaStream_t stream,
                 std::uint16_t* rounded) {
    RequireArgument(rows >= 0 && width > 0, "SwiGluPairs: bad shape");
    if (rows == 0) {
        return;
    }
    const std::size_t count = static_cast<std::size_t>(rows) * width;
    SwiGluPairsKernel<<<GridFor(count), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float2*>(gate_up), out, reinterpret_cast<__nv_bfloat16*>(rounded), count);
    CheckKernelLaunch();
}

// Copies rows row_indices[0..count) of the row-major matrix `in` (rows
// `width` floats wide) into consecutive rows of `out`. The indices live in
// GPU memory and must be valid rows of `in`: the kernel trusts them.
void GatherRows(const float* in, const int* row_indices, int count, int width, float* out, cudaStream_t stream) {
    RequireArgument(count >= 0 && width > 0, "GatherRows: bad shape");
    if (count == 0) {
        return;
    }
    GatherRowsKernel<<<count, kThreadsPerBlock, 0, stream>>>(in, row_indices, width, out);
    CheckKernelLaunch();
}

// Rounds FP32 values to the nearest BF16 (ties to even), stored as raw
// 16-bit patterns: keeps FP32's range, drops 16 bits of precision.
void ConvertF32ToBf16(const float* in, std::uint16_t* out, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    ConvertF32ToBf16Kernel<<<GridFor(count), kThreadsPerBlock, 0, stream>>>(in, out, count);
    CheckKernelLaunch();
}

// Rounds FP32 values to the nearest FP16, saturating at its largest finite
// value (65,504) instead of overflowing to infinity: 3 more bits of
// precision than BF16, a much smaller range.
void ConvertF32ToF16(const float* in, std::uint16_t* out, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    ConvertF32ToF16Kernel<<<GridFor(count), kThreadsPerBlock, 0, stream>>>(in, out, count);
    CheckKernelLaunch();
}

}  // namespace murmur
